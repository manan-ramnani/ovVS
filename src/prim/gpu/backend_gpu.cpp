#include "internal.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <new>

#if defined(OVVS_WITH_SYCL)
#include <sycl/sycl.hpp>
#if defined(OVVS_WITH_MKL)
#include <oneapi/mkl.hpp>
#endif
#endif

#if defined(OVVS_WITH_MKL)
#include <mkl_lapacke.h>
#endif

namespace ovvs {
namespace impl {

#if defined(OVVS_WITH_SYCL)
static sycl::queue& gpu_queue() {
  static sycl::queue q{sycl::gpu_selector_v};
  return q;
}

template <typename T>
struct ThreadUsmScratch {
  explicit ThreadUsmScratch(const sycl::queue& queue_value)
      : queue(queue_value), context(queue_value.get_context()) {}

  ThreadUsmScratch(const ThreadUsmScratch&) = delete;
  ThreadUsmScratch& operator=(const ThreadUsmScratch&) = delete;

  ~ThreadUsmScratch() noexcept {
    if (!ptr) return;
    try {
      queue.wait();
    } catch (...) {
    }
    try {
      sycl::free(ptr, context);
    } catch (...) {
    }
  }

  T* ptr = nullptr;
  size_t capacity = 0;
  sycl::queue queue;
  sycl::context context;
};

template <typename T>
static T* usm_scratch(size_t n, GpuWorkStats* stats = nullptr) {
  /* A search can run on multiple resources concurrently. Keep each worker's
     reusable scratch independent; every caller waits before returning it. The
     queue/context holder drains and frees the thread's high-water allocation. */
  auto& q = gpu_queue();
  static thread_local ThreadUsmScratch<T> scratch(q);
  if (n > scratch.capacity) {
    if (scratch.ptr) {
      sycl::free(scratch.ptr, scratch.context);
      scratch.ptr = nullptr;
      scratch.capacity = 0;
    }
    if (stats) stats->allocation(n, sizeof(T));
    scratch.ptr = sycl::malloc_shared<T>(n, q);
    scratch.capacity = scratch.ptr ? n : 0;
  }
  return scratch.ptr;
}

static float* usm_f(size_t n, GpuWorkStats* stats = nullptr) {
  return usm_scratch<float>(n, stats);
}
static int64_t* usm_i64(size_t n, GpuWorkStats* stats = nullptr) {
  return usm_scratch<int64_t>(n, stats);
}
static sycl::half* usm_h(size_t n, GpuWorkStats* stats = nullptr) {
  return usm_scratch<sycl::half>(n, stats);
}
static std::int8_t* usm_i8(size_t n, GpuWorkStats* stats = nullptr) {
  return usm_scratch<std::int8_t>(n, stats);
}

template <typename T>
class ScopedDeviceUsm {
 public:
  ScopedDeviceUsm(sycl::queue& q, size_t count, GpuWorkStats* stats = nullptr)
      : q_(&q), stats_(stats) {
    if (count != 0) {
      if (stats_) stats_->allocation(count, sizeof(T));
      ptr_ = sycl::malloc_device<T>(count, q);
      if (!ptr_) throw std::bad_alloc();
    }
  }

  ScopedDeviceUsm(const ScopedDeviceUsm&) = delete;
  ScopedDeviceUsm& operator=(const ScopedDeviceUsm&) = delete;

  ~ScopedDeviceUsm() noexcept {
    if (!ptr_) return;
    try {
      if (stats_) stats_->wait();
      q_->wait_and_throw();
    } catch (...) {
      /* Waiting still drains submitted work before storage is released. */
    }
    try {
      sycl::free(ptr_, *q_);
    } catch (...) {
    }
  }

  T* get() const { return ptr_; }

 private:
  sycl::queue* q_ = nullptr;
  T* ptr_ = nullptr;
  GpuWorkStats* stats_ = nullptr;
};

static bool gpu_pointer_accessible(sycl::queue& q, const void* ptr) {
  if (!ptr) return false;
  try {
    return sycl::get_pointer_type(const_cast<void*>(ptr), q.get_context()) != sycl::usm::alloc::unknown;
  } catch (...) {
    return false;
  }
}

static bool checked_product(size_t a, size_t b, size_t& product) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
  product = a * b;
  return true;
}

static size_t next_power_of_two(size_t value) {
  if (value <= 1) return 1;
  --value;
  for (size_t shift = 1; shift < sizeof(size_t) * 8; shift <<= 1) value |= value >> shift;
  return value + 1;
}

static void pack_f16(sycl::queue& q, const float* src, sycl::half* dst, size_t n) {
  if (ovvs_usm_is_shared(src)) {
    q.parallel_for(n, [=](sycl::id<1> i) { dst[i] = static_cast<sycl::half>(src[i]); }).wait_and_throw();
    return;
  }
  sycl::buffer<float, 1> b(const_cast<float*>(src), sycl::range<1>(n));
  q.submit([&](sycl::handler& h) {
     auto a = b.get_access<sycl::access::mode::read>(h);
     h.parallel_for(n, [=](sycl::id<1> i) { dst[i] = static_cast<sycl::half>(a[i]); });
   }).wait_and_throw();
}

static void pack_i8(sycl::queue& q, const float* src, std::int8_t* dst, size_t n, float scale) {
  const float inv = 1.f / scale;
  if (ovvs_usm_is_shared(src)) {
    q.parallel_for(n, [=](sycl::id<1> i) {
       float v = src[i] * inv;
       if (v > 127.f) v = 127.f;
       if (v < -127.f) v = -127.f;
       dst[i] = static_cast<std::int8_t>(v);
     }).wait_and_throw();
    return;
  }
  sycl::buffer<float, 1> b(const_cast<float*>(src), sycl::range<1>(n));
  q.submit([&](sycl::handler& h) {
     auto a = b.get_access<sycl::access::mode::read>(h);
     h.parallel_for(n, [=](sycl::id<1> i) {
       float v = a[i] * inv;
       if (v > 127.f) v = 127.f;
       if (v < -127.f) v = -127.f;
       dst[i] = static_cast<std::int8_t>(v);
     });
   }).wait_and_throw();
}
#endif

void* ovvs_usm_malloc(size_t bytes) {
  if (bytes == 0) bytes = 1;
#if defined(OVVS_WITH_SYCL)
  try {
    void* p = sycl::malloc_shared(bytes, gpu_queue());
    if (p) return p;
  } catch (...) {
  }
#endif
  return std::malloc(bytes);
}

/* Bumped on every shared-USM release. A cached device mirror keyed on a source pointer is
   only safe while that pointer cannot have been recycled, and a pointer cannot be recycled
   without a free happening first, so this is a sound (conservative) invalidation signal. */
static std::atomic<uint64_t> g_usm_free_generation{0};

uint64_t ovvs_usm_free_generation() { return g_usm_free_generation.load(std::memory_order_relaxed); }

void ovvs_usm_free(void* p) {
  if (!p) return;
  g_usm_free_generation.fetch_add(1, std::memory_order_relaxed);
#if defined(OVVS_WITH_SYCL)
  try {
    if (sycl::get_pointer_type(p, gpu_queue().get_context()) != sycl::usm::alloc::unknown) {
      sycl::free(p, gpu_queue());
      return;
    }
  } catch (...) {
  }
#endif
  std::free(p);
}

void* ovvs_gpu_workspace_alloc(size_t bytes) {
  if (bytes == 0) return nullptr;
#if defined(OVVS_WITH_SYCL)
  try {
    if (!gpu_available()) return nullptr;
    return sycl::malloc_device(bytes, gpu_queue());
  } catch (...) {
  }
#else
  (void)bytes;
#endif
  return nullptr;
}

void ovvs_gpu_workspace_free(void* p) {
  if (!p) return;
#if defined(OVVS_WITH_SYCL)
  try {
    sycl::free(p, gpu_queue());
  } catch (...) {
  }
#else
  (void)p;
#endif
}

bool ovvs_usm_is_shared(const void* p) {
  if (!p) return false;
#if defined(OVVS_WITH_SYCL)
  try {
    auto k = sycl::get_pointer_type(const_cast<void*>(p), gpu_queue().get_context());
    return k == sycl::usm::alloc::shared || k == sycl::usm::alloc::host;
  } catch (...) {
  }
#endif
  return false;
}



bool gpu_available() {
#if defined(OVVS_WITH_SYCL)
  try {
    auto d = gpu_queue().get_device();
    if (d.is_gpu()) return true;
  } catch (...) {
  }
#endif
  return ov_device_available("GPU");
}

bool gpu_vector_add(const float* a, const float* b, float* c, int64_t n) {
#if defined(OVVS_WITH_SYCL)
  try {
    auto& q = gpu_queue();
    sycl::buffer<float, 1> ba(const_cast<float*>(a), sycl::range<1>(static_cast<size_t>(n)));
    sycl::buffer<float, 1> bb(const_cast<float*>(b), sycl::range<1>(static_cast<size_t>(n)));
    sycl::buffer<float, 1> bc(c, sycl::range<1>(static_cast<size_t>(n)));
    q.submit([&](sycl::handler& h) {
      auto aa = ba.get_access<sycl::access::mode::read>(h);
      auto bbv = bb.get_access<sycl::access::mode::read>(h);
      auto cc = bc.get_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> i) { cc[i] = aa[i] + bbv[i]; });
    });
    q.wait();
    return true;
  } catch (...) {
    return false;
  }
#else
  (void)a;
  (void)b;
  (void)c;
  (void)n;
  return false;
#endif
}

#if defined(OVVS_WITH_SYCL)
static bool gemm_sycl_usm(const float* a, const float* b, float* c, int64_t m,
                          int64_t n, int64_t k, bool trans_b,
                          GpuWorkStats* stats = nullptr) {
  auto& q = gpu_queue();
  const size_t M = static_cast<size_t>(m), N = static_cast<size_t>(n), K = static_cast<size_t>(k);
  const size_t bsz = trans_b ? N * K : K * N;
  const bool a_usm = ovvs_usm_is_shared(a);
  const bool b_usm = ovvs_usm_is_shared(b);
  const bool c_usm = ovvs_usm_is_shared(c);
  size_t scratch_n = 0;
  if (!a_usm) scratch_n += M * K;
  if (!b_usm) scratch_n += bsz;
  if (!c_usm) scratch_n += M * N;
  float* scratch = scratch_n ? usm_f(scratch_n, stats) : nullptr;
  if (scratch_n && !scratch) return false;
  float* sp = scratch;
  float* A = a_usm ? const_cast<float*>(a) : (std::memcpy(sp, a, M * K * sizeof(float)), sp);
  if (!a_usm) sp += M * K;
  float* B = b_usm ? const_cast<float*>(b) : (std::memcpy(sp, b, bsz * sizeof(float)), sp);
  if (!b_usm) sp += bsz;
  float* C = c_usm ? c : sp;
  const bool tb = trans_b;
  if (stats) stats->kernel();
  q.parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> id) {
    const size_t i = id[0];
    const size_t j = id[1];
    float s = 0.f;
    if (!tb) {
      for (size_t t = 0; t < K; ++t) s += A[i * K + t] * B[t * N + j];
    } else {
      for (size_t t = 0; t < K; ++t) s += A[i * K + t] * B[j * K + t];
    }
    C[i * N + j] = s;
  });
  if (stats) stats->wait();
  q.wait_and_throw();
  if (!c_usm) std::memcpy(c, C, M * N * sizeof(float));
  return true;
}

#if defined(OVVS_WITH_MKL)
static bool gemm_mkl_usm(const float* a, const float* b, float* c, int64_t m,
                         int64_t n, int64_t k, bool trans_b,
                         GpuWorkStats* stats = nullptr) {
  auto& q = gpu_queue();
  const size_t M = static_cast<size_t>(m), N = static_cast<size_t>(n), K = static_cast<size_t>(k);
  const size_t bsz = trans_b ? N * K : K * N;
  const bool a_usm = ovvs_usm_is_shared(a);
  const bool b_usm = ovvs_usm_is_shared(b);
  const bool c_usm = ovvs_usm_is_shared(c);
  size_t scratch_n = 0;
  if (!a_usm) scratch_n += M * K;
  if (!b_usm) scratch_n += bsz;
  if (!c_usm) scratch_n += M * N;
  float* scratch = scratch_n ? usm_f(scratch_n, stats) : nullptr;
  if (scratch_n && !scratch) return false;
  float* sp = scratch;
  float* A = a_usm ? const_cast<float*>(a) : (std::memcpy(sp, a, M * K * sizeof(float)), sp);
  if (!a_usm) sp += M * K;
  float* B = b_usm ? const_cast<float*>(b) : (std::memcpy(sp, b, bsz * sizeof(float)), sp);
  if (!b_usm) sp += bsz;
  float* C = c_usm ? c : sp;
  const auto transA = oneapi::mkl::transpose::nontrans;
  const auto transB = trans_b ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;
  const std::int64_t lda = static_cast<std::int64_t>(K);
  const std::int64_t ldb = trans_b ? static_cast<std::int64_t>(K) : static_cast<std::int64_t>(N);
  const std::int64_t ldc = static_cast<std::int64_t>(N);
  if (stats) stats->kernel();
  oneapi::mkl::blas::row_major::gemm(q, transA, transB, m, n, k, 1.f, A, lda, B, ldb, 0.f, C, ldc);
  if (stats) stats->wait();
  q.wait_and_throw();
  if (!c_usm) std::memcpy(c, C, M * N * sizeof(float));
  return true;
}

static bool gemm_mkl_f16(const float* a, const float* b, float* c, int64_t m, int64_t n, int64_t k,
                         bool trans_b) {
  auto& q = gpu_queue();
  const size_t M = static_cast<size_t>(m), N = static_cast<size_t>(n), K = static_cast<size_t>(k);
  const size_t bsz = trans_b ? N * K : K * N;
  sycl::half* packed = usm_h(M * K + bsz);
  float* C = ovvs_usm_is_shared(c) ? c : usm_f(M * N);
  if (!packed || !C) return false;
  sycl::half* A = packed;
  sycl::half* B = packed + M * K;
  pack_f16(q, a, A, M * K);
  pack_f16(q, b, B, bsz);
  const auto transA = oneapi::mkl::transpose::nontrans;
  const auto transB = trans_b ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;
  const std::int64_t lda = static_cast<std::int64_t>(K);
  const std::int64_t ldb = trans_b ? static_cast<std::int64_t>(K) : static_cast<std::int64_t>(N);
  const std::int64_t ldc = static_cast<std::int64_t>(N);
  oneapi::mkl::blas::row_major::gemm(q, transA, transB, m, n, k, 1.f, A, lda, B, ldb, 0.f, C, ldc);
  q.wait_and_throw();
  if (!ovvs_usm_is_shared(c)) std::memcpy(c, C, M * N * sizeof(float));
  return true;
}

static bool gemm_mkl_i8(const float* a, const float* b, float* c, int64_t m, int64_t n, int64_t k,
                        bool trans_b) {
  auto& q = gpu_queue();
  const size_t M = static_cast<size_t>(m), N = static_cast<size_t>(n), K = static_cast<size_t>(k);
  const size_t asz = M * K;
  const size_t bsz = trans_b ? N * K : K * N;
  std::int8_t* packed = usm_i8(asz + bsz);
  float* C = ovvs_usm_is_shared(c) ? c : usm_f(M * N);
  if (!packed || !C) return false;
  std::int8_t* A = packed;
  std::int8_t* B = packed + asz;
  auto maxabs = [](const float* p, size_t n) {
    float m = 0.f;
    for (size_t i = 0; i < n; ++i) {
      const float v = std::fabs(p[i]);
      if (v > m) m = v;
    }
    return m;
  };
  const float sa = std::max(maxabs(a, asz) / 127.f, 1e-8f);
  const float sb = std::max(maxabs(b, bsz) / 127.f, 1e-8f);
  pack_i8(q, a, A, asz, sa);
  pack_i8(q, b, B, bsz, sb);
  const float alpha = sa * sb;
  const auto transA = oneapi::mkl::transpose::nontrans;
  const auto transB = trans_b ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;
  const std::int64_t lda = static_cast<std::int64_t>(K);
  const std::int64_t ldb = trans_b ? static_cast<std::int64_t>(K) : static_cast<std::int64_t>(N);
  const std::int64_t ldc = static_cast<std::int64_t>(N);
  oneapi::mkl::blas::row_major::gemm(q, transA, transB, m, n, k, alpha, A, lda, B, ldb, 0.f, C, ldc);
  q.wait_and_throw();
  if (!ovvs_usm_is_shared(c)) std::memcpy(c, C, M * N * sizeof(float));
  return true;
}
#endif

enum class GpuGemmKind { Unset, Mkl, Sycl };
static GpuGemmKind g_gpu_gemm = GpuGemmKind::Unset;
static std::once_flag g_gpu_gemm_once;

static void pick_gpu_gemm(GpuWorkStats* stats = nullptr) {
  std::call_once(g_gpu_gemm_once, [stats] {
    g_gpu_gemm = GpuGemmKind::Sycl;
#if defined(OVVS_WITH_MKL)
    try {
      const int64_t m = 64, n = 128, k = 32;
      std::vector<float> A(static_cast<size_t>(m * k), 0.1f);
      std::vector<float> B(static_cast<size_t>(n * k), 0.2f);
      std::vector<float> Cm(static_cast<size_t>(m * n), 0.f);
      std::vector<float> Cs(static_cast<size_t>(m * n), 0.f);
      const auto t0 = std::chrono::steady_clock::now();
      const bool mok =
          gemm_mkl_usm(A.data(), B.data(), Cm.data(), m, n, k, true, stats);
      const auto t1 = std::chrono::steady_clock::now();
      const bool sok =
          gemm_sycl_usm(A.data(), B.data(), Cs.data(), m, n, k, true, stats);
      const auto t2 = std::chrono::steady_clock::now();
      if (mok && sok) {
        bool close = true;
        for (size_t i = 0; i < Cm.size(); ++i) {
          if (std::fabs(Cm[i] - Cs[i]) > 2e-2f) {
            close = false;
            break;
          }
        }
        const double mkl_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double sycl_ms =
            std::chrono::duration<double, std::milli>(t2 - t1).count();
        g_gpu_gemm =
            (close && mkl_ms <= sycl_ms) ? GpuGemmKind::Mkl : GpuGemmKind::Sycl;
        if (!close) g_gpu_gemm = GpuGemmKind::Sycl;
      } else if (mok) {
        g_gpu_gemm = GpuGemmKind::Mkl;
      }
      (void)sok;
    } catch (...) {
      g_gpu_gemm = GpuGemmKind::Sycl;
    }
#endif
  });
}
#endif

#if defined(OVVS_WITH_SYCL)
static bool gemm_sycl_f16(const float* a, const float* b, float* c, int64_t m, int64_t n, int64_t k,
                          bool trans_b) {
  auto& q = gpu_queue();
  const size_t M = static_cast<size_t>(m), N = static_cast<size_t>(n), K = static_cast<size_t>(k);
  const size_t bsz = trans_b ? N * K : K * N;
  sycl::half* Ah = sycl::malloc_shared<sycl::half>(M * K, q);
  sycl::half* Bh = sycl::malloc_shared<sycl::half>(bsz, q);
  float* C = ovvs_usm_is_shared(c) ? c : static_cast<float*>(ovvs_usm_malloc(M * N * sizeof(float)));
  if (!Ah || !Bh || !C) {
    if (Ah) sycl::free(Ah, q);
    if (Bh) sycl::free(Bh, q);
    if (C && !ovvs_usm_is_shared(c)) ovvs_usm_free(C);
    return false;
  }
  for (size_t i = 0; i < M * K; ++i) Ah[i] = static_cast<sycl::half>(a[i]);
  for (size_t i = 0; i < bsz; ++i) Bh[i] = static_cast<sycl::half>(b[i]);
  const bool tb = trans_b;
  q.parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> id) {
    const size_t i = id[0];
    const size_t j = id[1];
    sycl::half s(0.f);
    if (!tb) {
      for (size_t t = 0; t < K; ++t) s += Ah[i * K + t] * Bh[t * N + j];
    } else {
      for (size_t t = 0; t < K; ++t) s += Ah[i * K + t] * Bh[j * K + t];
    }
    C[i * N + j] = static_cast<float>(s);
  });
  q.wait_and_throw();
  if (!ovvs_usm_is_shared(c)) {
    std::memcpy(c, C, M * N * sizeof(float));
    ovvs_usm_free(C);
  }
  sycl::free(Ah, q);
  sycl::free(Bh, q);
  return true;
}
#endif

bool gpu_gemm_compute(ResourcesData& r, ovvsDType compute, const float* a, const float* b, float* c,
                      int64_t m, int64_t n, int64_t k, bool trans_b) {
  if (compute == OVVS_DTYPE_F32) return gpu_gemm(r, a, b, c, m, n, k, trans_b);
#if defined(OVVS_WITH_SYCL)
  if (gpu_available()) {
    try {
#if defined(OVVS_WITH_MKL)
      if (compute == OVVS_DTYPE_F16 && gemm_mkl_f16(a, b, c, m, n, k, trans_b)) {
        r.last_compute_dtype = OVVS_DTYPE_F16;
        return true;
      }
      if (compute == OVVS_DTYPE_I8 && gemm_mkl_i8(a, b, c, m, n, k, trans_b)) {
        r.last_compute_dtype = OVVS_DTYPE_I8;
        return true;
      }
#endif
      if (compute == OVVS_DTYPE_F16 && gemm_sycl_f16(a, b, c, m, n, k, trans_b)) {
        r.last_compute_dtype = OVVS_DTYPE_F16;
        return true;
      }
    } catch (...) {
    }
  }
#endif
  if (ov_matmul_compute(r, "GPU", compute, a, b, c, m, n, k, trans_b)) return true;
  return false;
}

bool gpu_gemm(ResourcesData& r, const float* a, const float* b, float* c, int64_t m, int64_t n,
              int64_t k, bool trans_b, GpuWorkStats* stats) {
#if defined(OVVS_WITH_SYCL)
  try {
    pick_gpu_gemm(stats);
#if defined(OVVS_WITH_MKL)
    if (g_gpu_gemm == GpuGemmKind::Mkl) {
      if (gemm_mkl_usm(a, b, c, m, n, k, trans_b, stats)) {
        r.last_compute_dtype = OVVS_DTYPE_F32;
        return true;
      }
    }
#endif
    if (gemm_sycl_usm(a, b, c, m, n, k, trans_b, stats)) {
      r.last_compute_dtype = OVVS_DTYPE_F32;
      return true;
    }
  } catch (...) {
  }
#endif
  const bool ok = ov_matmul(r, "GPU", a, b, c, m, n, k, trans_b);
  if (ok) r.last_compute_dtype = OVVS_DTYPE_F32;
  return ok;
}

bool gpu_topk(ResourcesData& r, const float* scores, int64_t rows, int64_t cols, int64_t k,
              int64_t* indices, float* values, bool largest, GpuWorkStats* stats) {
#if defined(OVVS_WITH_SYCL)
  /* Subgroup-friendly per-row partial select on USM shared; correctness first. */
  try {
    auto& q = gpu_queue();
    const size_t R = static_cast<size_t>(rows);
    const size_t C = static_cast<size_t>(cols);
    const size_t KK = static_cast<size_t>(std::min(k, cols));
    const bool s_usm = ovvs_usm_is_shared(scores);
    float* S = s_usm ? const_cast<float*>(scores) : usm_f(R * C + R * KK, stats);
    int64_t* I = usm_i64(R * KK, stats);
    if (!S || !I) throw std::bad_alloc();
    float* V = s_usm ? usm_f(R * KK, stats) : (S + R * C);
    if (!s_usm) std::memcpy(S, scores, R * C * sizeof(float));
    if (!V) throw std::bad_alloc();
    const bool lg = largest;
    if (stats) stats->kernel();
    q.parallel_for(sycl::range<1>(R), [=](sycl::id<1> rid) {
      const size_t row = rid[0];
      for (size_t t = 0; t < KK; ++t) {
        int64_t best_i = -1;
        float best_v = lg ? -std::numeric_limits<float>::infinity()
                         : std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < C; ++j) {
          bool used = false;
          for (size_t u = 0; u < t; ++u) {
            if (I[row * KK + u] == static_cast<int64_t>(j)) {
              used = true;
              break;
            }
          }
          if (used) continue;
          const float v = S[row * C + j];
          if (lg ? v > best_v : v < best_v) {
            best_v = v;
            best_i = static_cast<int64_t>(j);
          }
        }
        I[row * KK + t] = best_i;
        V[row * KK + t] = best_v;
      }
    });
    if (stats) stats->wait();
    q.wait_and_throw();
    std::memcpy(indices, I, R * KK * sizeof(int64_t));
    std::memcpy(values, V, R * KK * sizeof(float));
    return true;
  } catch (...) {
  }
#endif
  return ov_topk(r, "GPU", scores, rows, cols, k, indices, values, largest);
}

ovvsStatus gpu_ivfpq_scan_select(ResourcesData& r, const IvfPqScanTask* tasks,
                                 int64_t task_count, const float* luts,
                                 int64_t lut_elements, const int64_t* packed_ids,
                                 const uint8_t* packed_codes, int64_t packed_rows,
                                 const uint8_t* allow_bitset, int64_t allow_bitset_bytes,
                                 int64_t nq, int32_t pq_m, int32_t ks, int32_t krefine,
                                 int32_t* packed_positions, int32_t* counts,
                                 GpuWorkStats* stats) {
  (void)r;
#if defined(OVVS_WITH_SYCL)
  constexpr size_t kWorkGroupSize = 128;
  constexpr size_t kSortCapacity = 2 * kWorkGroupSize;

  if (!tasks || !luts || !packed_codes || !packed_positions || !counts ||
      task_count <= 0 || lut_elements <= 0 || packed_rows <= 0 || nq <= 0 ||
      pq_m <= 0 || ks <= 0 || ks > 256 || krefine <= 0) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  if (task_count > std::numeric_limits<int32_t>::max() ||
      packed_rows > std::numeric_limits<int32_t>::max() ||
      nq > std::numeric_limits<int32_t>::max()) {
    return OVVS_STATUS_SHAPE_MISMATCH;
  }
  if (krefine > static_cast<int32_t>(kWorkGroupSize)) {
    return OVVS_STATUS_DEVICE_UNAVAILABLE;
  }
  if ((allow_bitset == nullptr) != (allow_bitset_bytes == 0) ||
      allow_bitset_bytes < 0 ||
      (allow_bitset && (!packed_ids ||
                        allow_bitset_bytes > std::numeric_limits<int64_t>::max() / 8 ||
                        allow_bitset_bytes < (packed_rows + 7) / 8))) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }

  size_t lut_per_task = 0;
  size_t output_slots = 0;
  size_t packed_code_elements = 0;
  if (!checked_product(static_cast<size_t>(pq_m), static_cast<size_t>(ks), lut_per_task) ||
      !checked_product(static_cast<size_t>(nq), static_cast<size_t>(krefine), output_slots) ||
      !checked_product(static_cast<size_t>(packed_rows), static_cast<size_t>(pq_m),
                       packed_code_elements) ||
      lut_per_task > static_cast<size_t>(lut_elements) ||
      static_cast<uint64_t>(lut_elements) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(float)) ||
      static_cast<uint64_t>(task_count) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(IvfPqScanTask)) ||
      static_cast<uint64_t>(allow_bitset_bytes) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return OVVS_STATUS_SHAPE_MISMATCH;
  }
  (void)packed_code_elements;

  try {
    auto& q = gpu_queue();
    const auto device = q.get_device();
    if (!device.is_gpu() ||
        device.get_info<sycl::info::device::max_work_group_size>() < kWorkGroupSize ||
        !gpu_pointer_accessible(q, packed_codes) ||
        (allow_bitset && !gpu_pointer_accessible(q, packed_ids))) {
      return OVVS_STATUS_DEVICE_UNAVAILABLE;
    }

    const size_t sort_local_bytes =
        kSortCapacity * (sizeof(float) + 2 * sizeof(int32_t));
    const size_t local_mem_bytes = device.get_info<sycl::info::device::local_mem_size>();
    if (lut_per_task > (std::numeric_limits<size_t>::max() - sort_local_bytes) /
                           sizeof(float) ||
        lut_per_task * sizeof(float) + sort_local_bytes > local_mem_bytes) {
      return OVVS_STATUS_DEVICE_UNAVAILABLE;
    }

    std::vector<int32_t> query_task_offsets(static_cast<size_t>(nq) + 1, 0);
    int64_t cursor = 0;
    for (int64_t query = 0; query < nq; ++query) {
      query_task_offsets[static_cast<size_t>(query)] = static_cast<int32_t>(cursor);
      int64_t expected_dense_offset = 0;
      const int64_t query_task_begin = cursor;
      while (cursor < task_count && tasks[cursor].query_index == query) {
        const IvfPqScanTask& task = tasks[cursor];
        if (task.list_begin < 0 || task.list_count <= 0 ||
            task.list_begin > packed_rows - task.list_count || task.lut_offset < 0 ||
            task.lut_offset > lut_elements - static_cast<int64_t>(lut_per_task) ||
            task.query_dense_offset != expected_dense_offset ||
            task.list_count > std::numeric_limits<int32_t>::max() - expected_dense_offset) {
          return OVVS_STATUS_INVALID_ARGUMENT;
        }
        const int64_t task_end = task.list_begin + task.list_count;
        for (int64_t prior = query_task_begin; prior < cursor; ++prior) {
          const int64_t prior_end = tasks[prior].list_begin + tasks[prior].list_count;
          if (task.list_begin < prior_end && tasks[prior].list_begin < task_end) {
            return OVVS_STATUS_INVALID_ARGUMENT;
          }
        }
        const float* task_lut = luts + task.lut_offset;
        for (size_t element = 0; element < lut_per_task; ++element) {
          if (!std::isfinite(task_lut[element])) return OVVS_STATUS_INVALID_ARGUMENT;
        }
        expected_dense_offset += task.list_count;
        ++cursor;
      }
      query_task_offsets[static_cast<size_t>(query + 1)] = static_cast<int32_t>(cursor);
      if (cursor < task_count && tasks[cursor].query_index < query + 1) {
        return OVVS_STATUS_INVALID_ARGUMENT;
      }
    }
    if (cursor != task_count) return OVVS_STATUS_INVALID_ARGUMENT;

    ScopedDeviceUsm<IvfPqScanTask> device_tasks(q, static_cast<size_t>(task_count), stats);
    ScopedDeviceUsm<float> device_luts(q, static_cast<size_t>(lut_elements), stats);
    ScopedDeviceUsm<uint8_t> device_allow(q, static_cast<size_t>(allow_bitset_bytes), stats);
    ScopedDeviceUsm<int32_t> device_query_offsets(q, static_cast<size_t>(nq) + 1, stats);
    ScopedDeviceUsm<float> list_scores(q,
                                       static_cast<size_t>(task_count) *
                                           static_cast<size_t>(krefine),
                                       stats);
    ScopedDeviceUsm<int32_t> list_ordinals(q,
                                           static_cast<size_t>(task_count) *
                                               static_cast<size_t>(krefine),
                                           stats);
    ScopedDeviceUsm<int32_t> list_positions(q,
                                            static_cast<size_t>(task_count) *
                                                static_cast<size_t>(krefine),
                                            stats);
    ScopedDeviceUsm<int32_t> device_positions(q, output_slots, stats);
    ScopedDeviceUsm<int32_t> device_counts(q, static_cast<size_t>(nq), stats);
    ScopedDeviceUsm<int32_t> invalid_input(q, 1, stats);

    std::vector<sycl::event> input_events;
    input_events.reserve(5);
    if (stats) stats->h2d(static_cast<size_t>(task_count), sizeof(IvfPqScanTask));
    input_events.push_back(q.memcpy(device_tasks.get(), tasks,
                                    static_cast<size_t>(task_count) *
                                        sizeof(IvfPqScanTask)));
    if (stats) stats->h2d(static_cast<size_t>(lut_elements), sizeof(float));
    input_events.push_back(q.memcpy(device_luts.get(), luts,
                                    static_cast<size_t>(lut_elements) * sizeof(float)));
    if (stats) stats->h2d(static_cast<size_t>(nq) + 1, sizeof(int32_t));
    input_events.push_back(q.memcpy(device_query_offsets.get(), query_task_offsets.data(),
                                    (static_cast<size_t>(nq) + 1) * sizeof(int32_t)));
    input_events.push_back(q.memset(invalid_input.get(), 0, sizeof(int32_t)));
    if (allow_bitset) {
      if (stats) stats->h2d(static_cast<size_t>(allow_bitset_bytes));
      input_events.push_back(q.memcpy(device_allow.get(), allow_bitset,
                                      static_cast<size_t>(allow_bitset_bytes)));
    }

    const IvfPqScanTask* const task_data = device_tasks.get();
    const float* const lut_data = device_luts.get();
    const int64_t* const id_data = packed_ids;
    const uint8_t* const code_data = packed_codes;
    const uint8_t* const allow_data = allow_bitset ? device_allow.get() : nullptr;
    float* const task_scores = list_scores.get();
    int32_t* const task_ordinals = list_ordinals.get();
    int32_t* const task_positions = list_positions.get();
    int32_t* const invalid = invalid_input.get();
    const int32_t M = pq_m;
    const int32_t KS = ks;
    const int32_t K = krefine;
    const int64_t valid_id_rows = packed_rows;

    if (stats) stats->kernel();
    const sycl::event scan_event = q.submit([&](sycl::handler& h) {
      h.depends_on(input_events);
      sycl::local_accessor<float, 1> local_lut(sycl::range<1>(lut_per_task), h);
      sycl::local_accessor<float, 1> local_scores(sycl::range<1>(kSortCapacity), h);
      sycl::local_accessor<int32_t, 1> local_ordinals(sycl::range<1>(kSortCapacity), h);
      sycl::local_accessor<int32_t, 1> local_positions(sycl::range<1>(kSortCapacity), h);
      h.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(task_count) * kWorkGroupSize),
                            sycl::range<1>(kWorkGroupSize)),
          [=](sycl::nd_item<1> item) {
            const size_t task_index = item.get_group_linear_id();
            const size_t lane = item.get_local_linear_id();
            const IvfPqScanTask task = task_data[task_index];
            const float infinity = std::numeric_limits<float>::infinity();
            const int32_t no_ordinal = std::numeric_limits<int32_t>::max();

            for (size_t element = lane; element < lut_per_task;
                 element += kWorkGroupSize) {
              local_lut[element] = lut_data[static_cast<size_t>(task.lut_offset) + element];
            }
            if (lane < static_cast<size_t>(K)) {
              local_scores[lane] = infinity;
              local_ordinals[lane] = no_ordinal;
              local_positions[lane] = -1;
            }
            item.barrier(sycl::access::fence_space::local_space);

            for (int64_t base = 0; base < task.list_count;
                 base += static_cast<int64_t>(kWorkGroupSize)) {
              const size_t incoming = static_cast<size_t>(K) + lane;
              const int64_t row = base + static_cast<int64_t>(lane);
              float score = infinity;
              int32_t ordinal = no_ordinal;
              int32_t position = -1;
              if (row < task.list_count) {
                const int64_t packed_position = task.list_begin + row;
                bool candidate_allowed = true;
                if (allow_data) {
                  const int64_t id = id_data[packed_position];
                  if (id < 0 || id >= valid_id_rows) {
                    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>(*invalid)
                        .fetch_or(1);
                    candidate_allowed = false;
                  } else {
                    candidate_allowed =
                        ((allow_data[static_cast<size_t>(id >> 3)] >> (id & 7)) & 1u) != 0;
                  }
                }
                if (candidate_allowed) {
                  float candidate_score = 0.f;
                  bool code_valid = true;
                  const size_t code_offset =
                      static_cast<size_t>(packed_position) * static_cast<size_t>(M);
                  for (int32_t subspace = 0; subspace < M; ++subspace) {
                    const uint8_t code = code_data[code_offset + static_cast<size_t>(subspace)];
                    if (static_cast<int32_t>(code) >= KS) {
                      sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                       sycl::memory_scope::device,
                                       sycl::access::address_space::global_space>(*invalid)
                          .fetch_or(1);
                      code_valid = false;
                      break;
                    }
                    candidate_score +=
                        local_lut[static_cast<size_t>(subspace) * static_cast<size_t>(KS) +
                                  static_cast<size_t>(code)];
                  }
                  if (code_valid && sycl::isfinite(candidate_score)) {
                    score = candidate_score;
                    ordinal = static_cast<int32_t>(task.query_dense_offset + row);
                    position = static_cast<int32_t>(packed_position);
                  } else if (code_valid) {
                    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>(*invalid)
                        .fetch_or(1);
                  }
                }
              }
              local_scores[incoming] = score;
              local_ordinals[incoming] = ordinal;
              local_positions[incoming] = position;
              if (lane < kWorkGroupSize - static_cast<size_t>(K)) {
                const size_t unused = static_cast<size_t>(K) + kWorkGroupSize + lane;
                local_scores[unused] = infinity;
                local_ordinals[unused] = no_ordinal;
                local_positions[unused] = -1;
              }
              item.barrier(sycl::access::fence_space::local_space);

              for (size_t width = 2; width <= kSortCapacity; width <<= 1) {
                for (size_t stride = width >> 1; stride > 0; stride >>= 1) {
                  const size_t left =
                      (lane / stride) * (stride << 1) + (lane % stride);
                  const size_t right = left + stride;
                  const float left_score = local_scores[left];
                  const float right_score = local_scores[right];
                  const int32_t left_ordinal = local_ordinals[left];
                  const int32_t right_ordinal = local_ordinals[right];
                  const bool left_better =
                      left_score < right_score ||
                      (left_score == right_score && left_ordinal < right_ordinal);
                  const bool right_better =
                      right_score < left_score ||
                      (right_score == left_score && right_ordinal < left_ordinal);
                  const bool ascending = (left & width) == 0;
                  if ((ascending && right_better) || (!ascending && left_better)) {
                    local_scores[left] = right_score;
                    local_scores[right] = left_score;
                    local_ordinals[left] = right_ordinal;
                    local_ordinals[right] = left_ordinal;
                    const int32_t left_position = local_positions[left];
                    local_positions[left] = local_positions[right];
                    local_positions[right] = left_position;
                  }
                  item.barrier(sycl::access::fence_space::local_space);
                }
              }
            }

            if (lane < static_cast<size_t>(K)) {
              const size_t output = task_index * static_cast<size_t>(K) + lane;
              task_scores[output] = local_scores[lane];
              task_ordinals[output] = local_ordinals[lane];
              task_positions[output] = local_positions[lane];
            }
          });
    });

    const int32_t* const query_offsets = device_query_offsets.get();
    int32_t* const output_positions = device_positions.get();
    int32_t* const output_counts = device_counts.get();
    if (stats) stats->kernel();
    const sycl::event merge_event = q.submit([&](sycl::handler& h) {
      h.depends_on(scan_event);
      sycl::local_accessor<float, 1> local_scores(sycl::range<1>(kSortCapacity), h);
      sycl::local_accessor<int32_t, 1> local_ordinals(sycl::range<1>(kSortCapacity), h);
      sycl::local_accessor<int32_t, 1> local_positions(sycl::range<1>(kSortCapacity), h);
      h.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(nq) * kWorkGroupSize),
                            sycl::range<1>(kWorkGroupSize)),
          [=](sycl::nd_item<1> item) {
            const size_t query = item.get_group_linear_id();
            const size_t lane = item.get_local_linear_id();
            const float infinity = std::numeric_limits<float>::infinity();
            const int32_t no_ordinal = std::numeric_limits<int32_t>::max();
            if (lane < static_cast<size_t>(K)) {
              local_scores[lane] = infinity;
              local_ordinals[lane] = no_ordinal;
              local_positions[lane] = -1;
            }
            item.barrier(sycl::access::fence_space::local_space);

            const int32_t task_begin = query_offsets[query];
            const int32_t task_end = query_offsets[query + 1];
            for (int32_t task = task_begin; task < task_end; ++task) {
              const size_t incoming = static_cast<size_t>(K) + lane;
              if (lane < static_cast<size_t>(K)) {
                const size_t source = static_cast<size_t>(task) * static_cast<size_t>(K) + lane;
                local_scores[incoming] = task_scores[source];
                local_ordinals[incoming] = task_ordinals[source];
                local_positions[incoming] = task_positions[source];
              } else {
                local_scores[incoming] = infinity;
                local_ordinals[incoming] = no_ordinal;
                local_positions[incoming] = -1;
              }
              if (lane < kWorkGroupSize - static_cast<size_t>(K)) {
                const size_t unused = static_cast<size_t>(K) + kWorkGroupSize + lane;
                local_scores[unused] = infinity;
                local_ordinals[unused] = no_ordinal;
                local_positions[unused] = -1;
              }
              item.barrier(sycl::access::fence_space::local_space);

              for (size_t width = 2; width <= kSortCapacity; width <<= 1) {
                for (size_t stride = width >> 1; stride > 0; stride >>= 1) {
                  const size_t left =
                      (lane / stride) * (stride << 1) + (lane % stride);
                  const size_t right = left + stride;
                  const float left_score = local_scores[left];
                  const float right_score = local_scores[right];
                  const int32_t left_ordinal = local_ordinals[left];
                  const int32_t right_ordinal = local_ordinals[right];
                  const bool left_better =
                      left_score < right_score ||
                      (left_score == right_score && left_ordinal < right_ordinal);
                  const bool right_better =
                      right_score < left_score ||
                      (right_score == left_score && right_ordinal < left_ordinal);
                  const bool ascending = (left & width) == 0;
                  if ((ascending && right_better) || (!ascending && left_better)) {
                    local_scores[left] = right_score;
                    local_scores[right] = left_score;
                    local_ordinals[left] = right_ordinal;
                    local_ordinals[right] = left_ordinal;
                    const int32_t left_position = local_positions[left];
                    local_positions[left] = local_positions[right];
                    local_positions[right] = left_position;
                  }
                  item.barrier(sycl::access::fence_space::local_space);
                }
              }
            }

            if (lane < static_cast<size_t>(K)) {
              output_positions[query * static_cast<size_t>(K) + lane] =
                  local_positions[lane];
            }
            if (lane == 0) {
              int32_t selected = 0;
              for (int32_t rank = 0; rank < K; ++rank) {
                if (local_positions[static_cast<size_t>(rank)] >= 0) ++selected;
              }
              output_counts[query] = selected;
            }
          });
    });

    std::vector<int32_t> staged_positions(output_slots, -1);
    std::vector<int32_t> staged_counts(static_cast<size_t>(nq), 0);
    int32_t invalid_value = 0;
    if (stats) stats->d2h(output_slots, sizeof(int32_t));
    sycl::event positions_copy = q.submit([&](sycl::handler& h) {
      h.depends_on(merge_event);
      h.memcpy(staged_positions.data(), output_positions, output_slots * sizeof(int32_t));
    });
    if (stats) stats->d2h(static_cast<size_t>(nq), sizeof(int32_t));
    sycl::event counts_copy = q.submit([&](sycl::handler& h) {
      h.depends_on(merge_event);
      h.memcpy(staged_counts.data(), output_counts,
               static_cast<size_t>(nq) * sizeof(int32_t));
    });
    if (stats) stats->d2h(1, sizeof(int32_t));
    sycl::event invalid_copy = q.submit([&](sycl::handler& h) {
      h.depends_on(merge_event);
      h.memcpy(&invalid_value, invalid, sizeof(int32_t));
    });
    if (stats) stats->wait();
    positions_copy.wait_and_throw();
    if (stats) stats->wait();
    counts_copy.wait_and_throw();
    if (stats) stats->wait();
    invalid_copy.wait_and_throw();
    if (invalid_value != 0) return OVVS_STATUS_INVALID_ARGUMENT;

    for (int64_t query = 0; query < nq; ++query) {
      const int32_t selected = staged_counts[static_cast<size_t>(query)];
      if (selected < 0 || selected > K) return OVVS_STATUS_ERROR;
      for (int32_t rank = 0; rank < K; ++rank) {
        const int32_t position =
            staged_positions[static_cast<size_t>(query) * static_cast<size_t>(K) +
                             static_cast<size_t>(rank)];
        if ((rank < selected && (position < 0 || position >= packed_rows)) ||
            (rank >= selected && position != -1)) {
          return OVVS_STATUS_ERROR;
        }
      }
    }
    std::memcpy(packed_positions, staged_positions.data(), output_slots * sizeof(int32_t));
    std::memcpy(counts, staged_counts.data(), static_cast<size_t>(nq) * sizeof(int32_t));
    r.last_compute_dtype = OVVS_DTYPE_F32;
    return OVVS_STATUS_SUCCESS;
  } catch (const std::length_error&) {
    return OVVS_STATUS_SHAPE_MISMATCH;
  } catch (const std::bad_alloc&) {
    return OVVS_STATUS_OOM;
  } catch (...) {
    return OVVS_STATUS_ERROR;
  }
#else
  (void)tasks;
  (void)task_count;
  (void)luts;
  (void)lut_elements;
  (void)packed_ids;
  (void)packed_codes;
  (void)packed_rows;
  (void)allow_bitset;
  (void)allow_bitset_bytes;
  (void)nq;
  (void)pq_m;
  (void)ks;
  (void)krefine;
  (void)packed_positions;
  (void)counts;
  (void)stats;
  return OVVS_STATUS_DEVICE_UNAVAILABLE;
#endif
}

bool gpu_gather_rows(ResourcesData& r, const float* src, int64_t src_rows, int64_t dim,
                     const int64_t* idx, int64_t nidx, float* out,
                     GpuWorkStats* stats) {
#if defined(OVVS_WITH_SYCL)
  try {
    auto& q = gpu_queue();
    const size_t D = static_cast<size_t>(dim);
    const size_t N = static_cast<size_t>(nidx);
    const size_t SR = static_cast<size_t>(src_rows);
    const bool src_usm = ovvs_usm_is_shared(src);
    float* S = src_usm ? const_cast<float*>(src) : usm_f(SR * D + N * D, stats);
    int64_t* I = usm_i64(N, stats);
    if (!S || !I) throw std::bad_alloc();
    float* O = src_usm ? usm_f(N * D, stats) : (S + SR * D);
    if (!O) throw std::bad_alloc();
    if (!src_usm) std::memcpy(S, src, SR * D * sizeof(float));
    if (ovvs_usm_is_shared(idx)) {
      I = const_cast<int64_t*>(idx);
    } else {
      std::memcpy(I, idx, N * sizeof(int64_t));
    }
    if (stats) stats->kernel();
    q.parallel_for(sycl::range<2>(N, D), [=](sycl::id<2> id) {
      const size_t i = id[0];
      const size_t j = id[1];
      const int64_t row = I[i];
      O[i * D + j] = (row >= 0 && static_cast<size_t>(row) < SR) ? S[static_cast<size_t>(row) * D + j] : 0.f;
    });
    if (stats) stats->wait();
    q.wait_and_throw();
    std::memcpy(out, O, N * D * sizeof(float));
    return true;
  } catch (...) {
  }
#endif
#if defined(OVVS_WITH_SYCL)
  /* A persistent shared dataset must never fall through to an OpenVINO Gather
     that re-uploads the full index after a direct SYCL failure. */
  if (ovvs_usm_is_shared(src)) return false;
#endif
  return ov_gather_rows(r, "GPU", src, src_rows, dim, idx, nidx, out);
}

int32_t sycl_enabled() {
#if defined(OVVS_WITH_SYCL)
  return sycl_gpu_available() ? 1 : 0;
#else
  return 0;
#endif
}

bool sycl_gpu_available() {
#if defined(OVVS_WITH_SYCL)
  try {
    return gpu_queue().get_device().is_gpu();
  } catch (...) {
    return false;
  }
#else
  return false;
#endif
}

/* XMX (systolic matrix engine) capability probe. fp16 COMPUTE on the GPU only pays
   where matrix hardware exists: on Xe-LPG the packed-half walk measured ALU-bound at
   0.62x of the fp32 walk (fp16 there is a memory/coverage feature; int8 is the speed
   path). This is the routing hook for the portability pass -- XMX-class SKUs
   (Arc/Xe-HPG, Lunar Lake/Battlemage Xe2) are where a joint-matrix fp16 distance
   kernel becomes worth building. Returns 1 (XMX), 0 (none), -1 (no usable GPU). */
int32_t gpu_has_xmx() {
#if defined(OVVS_WITH_SYCL)
  try {
    if (!sycl_gpu_available()) return -1;
    const auto device = gpu_queue().get_device();
#if defined(SYCL_EXT_ONEAPI_MATRIX_VERSION) || defined(SYCL_EXT_ONEAPI_MATRIX)
    const auto combos = device.get_info<
        sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
    return combos.empty() ? 0 : 1;
#else
    (void)device;
    return 0; /* toolchain cannot ask; report no-XMX rather than guess */
#endif
  } catch (...) {
    return 0; /* the query itself failing means no usable matrix path */
  }
#else
  return -1;
#endif
}

ovvsStatus gpu_nndescent_build(ResourcesData& r, const float* dataset, int64_t n,
                               int64_t dim, ovvsMetric metric, int32_t degree,
                               int32_t iters, int32_t* graph,
                               NnDescentBuildStats* stats) {
  if (stats) *stats = {};
  {
    std::lock_guard<std::mutex> lock(r.nndescent_stats_mutex);
    r.nndescent_iterations_run = 0;
    r.nndescent_changed_edges = 0;
    r.nndescent_pending_new_edges = 0;
    r.nndescent_change_ratio = 0.0;
    r.nndescent_peak_device_bytes = 0;
    r.nndescent_converged = false;
  }
#if defined(OVVS_WITH_SYCL)
  if (!sycl_gpu_available()) return OVVS_STATUS_DEVICE_UNAVAILABLE;
  if (!dataset || !graph || n <= 1 || dim <= 0 || degree <= 0 || iters <= 0) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  if (n > std::numeric_limits<int32_t>::max() || static_cast<int64_t>(degree) >= n) {
    return OVVS_STATUS_SHAPE_MISMATCH;
  }
  if (metric != OVVS_METRIC_L2_EXPANDED && metric != OVVS_METRIC_L2_SQRT_EXPANDED &&
      metric != OVVS_METRIC_INNER_PRODUCT && metric != OVVS_METRIC_COSINE_EXPANDED) {
    return OVVS_STATUS_UNSUPPORTED;
  }

  try {
    auto& q = gpu_queue();
    const auto note_submission = [&]() noexcept {
      if (stats) stats->submission();
    };
    const auto note_kernel = [&]() noexcept {
      if (stats) {
        stats->submission();
        stats->gpu.kernel();
      }
    };
    const auto note_wait = [&]() noexcept {
      if (stats) stats->gpu.wait();
    };
    const auto note_h2d = [&](size_t bytes) noexcept {
      if (stats) {
        stats->submission();
        stats->gpu.h2d(bytes);
      }
    };
    const auto note_d2h = [&](size_t bytes) noexcept {
      if (stats) {
        stats->submission();
        stats->gpu.d2h(bytes);
      }
    };
    const auto device = q.get_device();
    const size_t N = static_cast<size_t>(n);
    const size_t D = static_cast<size_t>(dim);
    const size_t DEG = static_cast<size_t>(degree);
    constexpr size_t kSampleCap = 16u;
    constexpr size_t kReverseSourceChunk = 196608u;
    constexpr size_t kProposalVertexChunk = 16384u;
    constexpr size_t kVertexGroupsPerLaunch = 65536u;
    constexpr uint32_t kNewMask = 0x80000000u;
    constexpr uint32_t kIdMask = 0x7fffffffu;
    constexpr uint32_t kInvalidId = 0x7fffffffu;
    constexpr double kTerminationThreshold = 0.0001;
    const size_t sample_count = std::min(DEG, kSampleCap);
    const size_t max_bi_samples = sample_count * 2u;
    const size_t proposals_per_vertex = sample_count * 12u;
    const int met = static_cast<int>(metric);
    const int64_t iteration_count = static_cast<int64_t>(iters);

    size_t dataset_count = 0;
    size_t graph_count = 0;
    size_t reverse_count = 0;
    if (!checked_product(N, D, dataset_count) || !checked_product(N, DEG, graph_count) ||
        !checked_product(N, sample_count, reverse_count)) {
      return OVVS_STATUS_SHAPE_MISMATCH;
    }
    const size_t reverse_chunk = std::min(N, kReverseSourceChunk);
    const size_t proposal_chunk = std::min(N, kProposalVertexChunk);
    size_t reverse_inbox_count = 0;
    size_t proposal_inbox_count = 0;
    if (!checked_product(reverse_chunk, sample_count, reverse_inbox_count) ||
        !checked_product(proposal_chunk, proposals_per_vertex, proposal_inbox_count)) {
      return OVVS_STATUS_SHAPE_MISMATCH;
    }
    const size_t link_count = std::max(reverse_inbox_count, proposal_inbox_count);
    if (link_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      return OVVS_STATUS_SHAPE_MISMATCH;
    }

    size_t dataset_bytes = 0;
    size_t graph_plane_bytes = 0;
    size_t reverse_plane_bytes = 0;
    size_t link_bytes = 0;
    size_t proposal_id_bytes = 0;
    size_t proposal_distance_bytes = 0;
    size_t row_bytes = 0;
    if (!checked_product(dataset_count, sizeof(float), dataset_bytes) ||
        !checked_product(graph_count, sizeof(uint32_t), graph_plane_bytes) ||
        !checked_product(reverse_count, sizeof(uint32_t), reverse_plane_bytes) ||
        !checked_product(link_count, sizeof(int32_t), link_bytes) ||
        !checked_product(proposal_inbox_count, sizeof(uint32_t), proposal_id_bytes) ||
        !checked_product(proposal_inbox_count, sizeof(float), proposal_distance_bytes) ||
        !checked_product(N, sizeof(uint32_t), row_bytes)) {
      return OVVS_STATUS_SHAPE_MISMATCH;
    }
    const bool dataset_direct = gpu_pointer_accessible(q, dataset);
    const size_t max_alloc = device.get_info<sycl::info::device::max_mem_alloc_size>();
    if ((!dataset_direct && dataset_bytes > max_alloc) || graph_plane_bytes > max_alloc ||
        reverse_plane_bytes > max_alloc || link_bytes > max_alloc ||
        proposal_id_bytes > max_alloc || proposal_distance_bytes > max_alloc ||
        row_bytes > max_alloc) {
      return OVVS_STATUS_OOM;
    }

    size_t device_bytes = 0;
    const auto add_device_bytes = [&](size_t bytes) {
      if (bytes > std::numeric_limits<size_t>::max() - device_bytes) return false;
      device_bytes += bytes;
      return true;
    };
    if ((!dataset_direct && !add_device_bytes(dataset_bytes)) ||
        !add_device_bytes(graph_plane_bytes) || !add_device_bytes(graph_plane_bytes) ||
        !add_device_bytes(graph_plane_bytes) || !add_device_bytes(graph_plane_bytes) ||
        !add_device_bytes(reverse_plane_bytes) || !add_device_bytes(reverse_plane_bytes) ||
        !add_device_bytes(link_bytes) || !add_device_bytes(proposal_id_bytes) ||
        !add_device_bytes(proposal_distance_bytes) || !add_device_bytes(row_bytes) ||
        !add_device_bytes(row_bytes) || !add_device_bytes(2u * sizeof(int32_t))) {
      return OVVS_STATUS_SHAPE_MISMATCH;
    }
    const uint64_t global_mem = device.get_info<sycl::info::device::global_mem_size>();
    if (static_cast<uint64_t>(device_bytes) > global_mem) return OVVS_STATUS_OOM;
    if (device_bytes > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
      return OVVS_STATUS_SHAPE_MISMATCH;
    }
    if (stats) {
      stats->gpu_instrumented = true;
      stats->peak_owned_bytes = static_cast<int64_t>(device_bytes);
    }

    size_t twice_degree = 0;
    if (!checked_product(DEG, 2u, twice_degree)) {
      return OVVS_STATUS_SHAPE_MISMATCH;
    }
    const size_t convergence_hash_capacity =
        next_power_of_two(std::max<size_t>(2u, twice_degree));
    if (convergence_hash_capacity == 0) return OVVS_STATUS_SHAPE_MISMATCH;
    size_t convergence_local_bytes = 0;
    size_t matrix_cells = 0;
    size_t join_local_bytes = 0;
    size_t minimum_cells = 0;
    size_t minimum_local_bytes = 0;
    if (!checked_product(convergence_hash_capacity, sizeof(int32_t), convergence_local_bytes) ||
        !checked_product(max_bi_samples, max_bi_samples, matrix_cells) ||
        !checked_product(matrix_cells, 2u * sizeof(float), join_local_bytes) ||
        !checked_product(max_bi_samples, 3u, minimum_cells) ||
        !checked_product(minimum_cells, sizeof(uint32_t) + sizeof(float),
                         minimum_local_bytes) ||
        max_bi_samples >
            (std::numeric_limits<size_t>::max() - join_local_bytes - 2u * sizeof(int32_t)) /
                (2u * sizeof(uint32_t))) {
      return OVVS_STATUS_SHAPE_MISMATCH;
    }
    join_local_bytes += max_bi_samples * 2u * sizeof(uint32_t) + 2u * sizeof(int32_t);
    if (minimum_local_bytes > std::numeric_limits<size_t>::max() - join_local_bytes) {
      return OVVS_STATUS_SHAPE_MISMATCH;
    }
    join_local_bytes += minimum_local_bytes;
    const size_t local_mem = device.get_info<sycl::info::device::local_mem_size>();
    if (std::max(join_local_bytes, convergence_local_bytes) > local_mem) {
      return OVVS_STATUS_OOM;
    }

    const size_t max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
    if (max_work_group_size == 0) return OVVS_STATUS_DEVICE_UNAVAILABLE;
    size_t work_group_size = 1;
    while (work_group_size < 128u && work_group_size <= max_work_group_size / 2u) {
      work_group_size <<= 1u;
    }

    GpuWorkStats* const gpu_stats = stats ? &stats->gpu : nullptr;
    ScopedDeviceUsm<float> dataset_copy(q, dataset_direct ? 0u : dataset_count, gpu_stats);
    ScopedDeviceUsm<uint32_t> ids_a(q, graph_count, gpu_stats);
    ScopedDeviceUsm<uint32_t> ids_b(q, graph_count, gpu_stats);
    ScopedDeviceUsm<float> distances_a(q, graph_count, gpu_stats);
    ScopedDeviceUsm<float> distances_b(q, graph_count, gpu_stats);
    ScopedDeviceUsm<uint32_t> reverse_new(q, reverse_count, gpu_stats);
    ScopedDeviceUsm<uint32_t> reverse_old(q, reverse_count, gpu_stats);
    ScopedDeviceUsm<int32_t> inbox_links(q, link_count, gpu_stats);
    ScopedDeviceUsm<uint32_t> proposal_ids(q, proposal_inbox_count, gpu_stats);
    ScopedDeviceUsm<float> proposal_distances(q, proposal_inbox_count, gpu_stats);
    ScopedDeviceUsm<int32_t> heads(q, N, gpu_stats);
    ScopedDeviceUsm<uint32_t> active_targets(q, N, gpu_stats);
    ScopedDeviceUsm<int32_t> scalars(q, 2u, gpu_stats);
    uint32_t* const reverse_new_ptr = reverse_new.get();
    uint32_t* const reverse_old_ptr = reverse_old.get();
    int32_t* const inbox_links_ptr = inbox_links.get();
    uint32_t* const proposal_ids_ptr = proposal_ids.get();
    float* const proposal_distances_ptr = proposal_distances.get();
    int32_t* const heads_ptr = heads.get();
    uint32_t* const active_targets_ptr = active_targets.get();
    int32_t* const scalars_ptr = scalars.get();
    const float* DS = dataset_direct ? dataset : dataset_copy.get();
    if (!dataset_direct) {
      note_h2d(dataset_bytes);
      q.memcpy(dataset_copy.get(), dataset, dataset_bytes);
    }
    note_wait();
    q.wait_and_throw();

    note_submission();
    q.fill(scalars_ptr + 1, 0, 1u);
    note_wait();
    q.wait_and_throw();
    note_kernel();
    q.parallel_for(sycl::range<1>(dataset_count), [=](sycl::id<1> index) {
      if (!sycl::isfinite(DS[index])) {
        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            invalid(scalars_ptr[1]);
        invalid.store(1);
      }
    });
    note_wait();
    q.wait_and_throw();
    int32_t invalid_numeric = 0;
    note_d2h(sizeof(invalid_numeric));
    q.memcpy(&invalid_numeric, scalars_ptr + 1, sizeof(invalid_numeric));
    note_wait();
    q.wait_and_throw();
    if (invalid_numeric != 0) return OVVS_STATUS_INVALID_ARGUMENT;

    uint32_t* current_ids = ids_a.get();
    uint32_t* next_ids = ids_b.get();
    float* current_distances = distances_a.get();
    float* next_distances = distances_b.get();

    for (size_t vertex_offset = 0; vertex_offset < N; vertex_offset += kVertexGroupsPerLaunch) {
      const size_t launch_vertices = std::min(kVertexGroupsPerLaunch, N - vertex_offset);
      size_t global_size = 0;
      if (!checked_product(launch_vertices, work_group_size, global_size)) {
        return OVVS_STATUS_SHAPE_MISMATCH;
      }
      note_kernel();
      q.submit([&](sycl::handler& h) {
         h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size),
                                         sycl::range<1>(work_group_size)),
                        [=](sycl::nd_item<1> item) {
           const size_t lid = item.get_local_linear_id();
           const size_t vertex = vertex_offset + item.get_group_linear_id();
           const uint64_t modulus = static_cast<uint64_t>(N - 1u);

           auto mix32 = [](uint32_t value) {
             value ^= value >> 16u;
             value *= 0x7feb352du;
             value ^= value >> 15u;
             value *= 0x846ca68bu;
             value ^= value >> 16u;
             return value;
           };
           auto gcd64 = [](uint64_t a, uint64_t b) {
             while (b != 0u) {
               const uint64_t remainder = a % b;
               a = b;
               b = remainder;
             }
             return a;
           };

           const uint32_t seed = mix32(static_cast<uint32_t>(vertex) ^ 0x9e3779b9u);
           const uint64_t start = static_cast<uint64_t>(seed) % modulus;
           uint64_t step = 1u + static_cast<uint64_t>(mix32(seed ^ 0xa511e9b3u)) % modulus;
           while (gcd64(step, modulus) != 1u) step = step == modulus ? 1u : step + 1u;

           for (size_t edge = lid; edge < DEG; edge += work_group_size) {
             const uint64_t permuted =
                 (start + static_cast<uint64_t>(edge) * step) % modulus;
             const size_t id = (vertex + 1u + static_cast<size_t>(permuted)) % N;
             float l2 = 0.f;
             float ip = 0.f;
             float nx = 0.f;
             float ny = 0.f;
             const bool needs_ip = met == static_cast<int>(OVVS_METRIC_INNER_PRODUCT) ||
                                   met == static_cast<int>(OVVS_METRIC_COSINE_EXPANDED);
             for (size_t d = 0; d < D; ++d) {
               const float a = DS[vertex * D + d];
               const float b = DS[id * D + d];
               if (!needs_ip) {
                 const float delta = a - b;
                 l2 += delta * delta;
               } else {
                 ip += a * b;
                 if (met == static_cast<int>(OVVS_METRIC_COSINE_EXPANDED)) {
                   nx += a * a;
                   ny += b * b;
                 }
               }
             }
             float distance = l2;
             if (met == static_cast<int>(OVVS_METRIC_L2_SQRT_EXPANDED)) {
               distance = sycl::sqrt(l2);
             } else if (met == static_cast<int>(OVVS_METRIC_INNER_PRODUCT)) {
               distance = -ip;
             } else if (met == static_cast<int>(OVVS_METRIC_COSINE_EXPANDED)) {
               const float safe_nx = nx > 1e-12f ? nx : 1e-12f;
               const float safe_ny = ny > 1e-12f ? ny : 1e-12f;
               distance = 1.f - ip / (sycl::sqrt(safe_nx) * sycl::sqrt(safe_ny));
             }
             if (!sycl::isfinite(distance)) {
               sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                   invalid(scalars_ptr[1]);
               invalid.store(1);
               distance = std::numeric_limits<float>::max();
             }
             const size_t position = vertex * DEG + edge;
             current_ids[position] = kNewMask | static_cast<uint32_t>(id);
             current_distances[position] = distance;
           }
           item.barrier(sycl::access::fence_space::global_and_local);

           if (lid == 0) {
             const size_t row = vertex * DEG;
             for (size_t edge = 1; edge < DEG; ++edge) {
               const uint32_t tagged = current_ids[row + edge];
               const uint32_t raw = tagged & kIdMask;
               const float distance = current_distances[row + edge];
               size_t insert = edge;
               while (insert > 0) {
                 const float previous_distance = current_distances[row + insert - 1u];
                 const uint32_t previous_raw = current_ids[row + insert - 1u] & kIdMask;
                 if (previous_distance < distance ||
                     (previous_distance == distance && previous_raw <= raw)) {
                   break;
                 }
                 current_ids[row + insert] = current_ids[row + insert - 1u];
                 current_distances[row + insert] = previous_distance;
                 --insert;
               }
               current_ids[row + insert] = tagged;
               current_distances[row + insert] = distance;
             }
           }
         });
       });
    }
    note_wait();
    q.wait_and_throw();
    note_d2h(sizeof(invalid_numeric));
    q.memcpy(&invalid_numeric, scalars_ptr + 1, sizeof(invalid_numeric));
    note_wait();
    q.wait_and_throw();
    if (invalid_numeric != 0) return OVVS_STATUS_INVALID_ARGUMENT;

    std::vector<int32_t> changed_per_row(N);
    std::vector<uint32_t> pending_new_per_row(N);
    int32_t iterations_run = 0;
    int64_t last_changed_edges = 0;
    int64_t last_pending_new_edges = 0;
    double last_change_ratio = 0.0;
    bool converged = false;

    for (int64_t iteration = 0; iteration < iteration_count; ++iteration) {
      for (size_t vertex_offset = 0; vertex_offset < N;
           vertex_offset += kVertexGroupsPerLaunch) {
        const size_t launch_vertices = std::min(kVertexGroupsPerLaunch, N - vertex_offset);
        size_t global_size = 0;
        if (!checked_product(launch_vertices, work_group_size, global_size)) {
          return OVVS_STATUS_SHAPE_MISMATCH;
        }
        note_kernel();
        q.submit([&](sycl::handler& h) {
          h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size),
                                           sycl::range<1>(work_group_size)),
                         [=](sycl::nd_item<1> item) {
            const size_t lid = item.get_local_linear_id();
            const size_t vertex = vertex_offset + item.get_group_linear_id();
            const size_t row = vertex * DEG;
            for (size_t edge = lid; edge < DEG; edge += work_group_size) {
              next_ids[row + edge] = current_ids[row + edge];
              next_distances[row + edge] = current_distances[row + edge];
            }
            item.barrier(sycl::access::fence_space::global_and_local);
            if (lid == 0) {
              size_t sampled = 0;
              for (size_t edge = 0; edge < DEG && sampled < sample_count; ++edge) {
                const uint32_t tagged = current_ids[row + edge];
                if ((tagged & kNewMask) != 0u) {
                  next_ids[row + edge] = tagged & kIdMask;
                  ++sampled;
                }
              }
            }
          });
        });
      }
      note_wait();
      q.wait_and_throw();

      note_submission();
      q.fill(reverse_new_ptr, kInvalidId, reverse_count);
      note_submission();
      q.fill(reverse_old_ptr, kInvalidId, reverse_count);
      note_wait();
      q.wait_and_throw();

      const auto build_reverse = [&](bool select_new, uint32_t* reverse_graph) -> ovvsStatus {
        for (size_t source_offset = 0; source_offset < N; source_offset += reverse_chunk) {
          const size_t source_count = std::min(reverse_chunk, N - source_offset);
          note_submission();
          q.fill(heads_ptr, -1, N);
          note_submission();
          q.fill(scalars_ptr, 0, 1u);
          note_wait();
          q.wait_and_throw();

          note_kernel();
          q.parallel_for(sycl::range<1>(source_count), [=](sycl::id<1> source_index) {
            const size_t local_source = source_index[0];
            const size_t source = source_offset + local_source;
            size_t sample_rank = 0;
            for (size_t edge = 0; edge < DEG && sample_rank < sample_count; ++edge) {
              const uint32_t tagged = current_ids[source * DEG + edge];
              const bool edge_is_new = (tagged & kNewMask) != 0u;
              if (edge_is_new != select_new) continue;
              const uint32_t target = tagged & kIdMask;
              const size_t slot_size = local_source * sample_count + sample_rank;
              const int32_t slot = static_cast<int32_t>(slot_size);
              sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::global_space>
                  target_head(heads_ptr[target]);
              const int32_t prior = target_head.exchange(slot);
              inbox_links_ptr[slot_size] = prior;
              if (prior == -1) {
                sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    active_count(scalars_ptr[0]);
                const int32_t active_slot = active_count.fetch_add(1);
                active_targets_ptr[static_cast<size_t>(active_slot)] = target;
              }
              ++sample_rank;
            }
          });
          note_wait();
          q.wait_and_throw();

          int32_t active_count = 0;
          note_d2h(sizeof(active_count));
          q.memcpy(&active_count, scalars_ptr, sizeof(active_count));
          note_wait();
          q.wait_and_throw();
          if (active_count < 0 || static_cast<size_t>(active_count) > N) {
            return OVVS_STATUS_ERROR;
          }

          note_kernel();
          q.parallel_for(sycl::range<1>(static_cast<size_t>(active_count)),
                         [=](sycl::id<1> active_index) {
            const uint32_t target = active_targets_ptr[active_index];
            uint32_t best_ids[kSampleCap];
            uint32_t best_priorities[kSampleCap];
            for (size_t i = 0; i < kSampleCap; ++i) {
              best_ids[i] = kInvalidId;
              best_priorities[i] = std::numeric_limits<uint32_t>::max();
            }
            size_t count = 0;
            auto mix32 = [](uint32_t value) {
              value ^= value >> 16u;
              value *= 0x7feb352du;
              value ^= value >> 15u;
              value *= 0x846ca68bu;
              value ^= value >> 16u;
              return value;
            };
            const uint32_t class_seed = select_new ? 0x9e3779b9u : 0xa511e9b3u;
            const uint32_t priority_seed =
                mix32(target ^ class_seed ^
                      (static_cast<uint32_t>(iteration) + 1u) * 0x85ebca6bu);
            auto consider = [&](uint32_t source) {
              if (source == kInvalidId || source == target ||
                  static_cast<size_t>(source) >= N) {
                return;
              }
              for (size_t i = 0; i < count; ++i) {
                if (best_ids[i] == source) return;
              }
              const uint32_t priority = mix32(source ^ priority_seed);
              size_t position = 0;
              while (position < count &&
                     (best_priorities[position] < priority ||
                      (best_priorities[position] == priority && best_ids[position] < source))) {
                ++position;
              }
              if (position >= sample_count) return;
              const size_t new_count = count < sample_count ? count + 1u : count;
              for (size_t i = new_count; i > position + 1u; --i) {
                best_ids[i - 1u] = best_ids[i - 2u];
                best_priorities[i - 1u] = best_priorities[i - 2u];
              }
              best_ids[position] = source;
              best_priorities[position] = priority;
              count = new_count;
            };

            for (size_t i = 0; i < sample_count; ++i) {
              consider(reverse_graph[static_cast<size_t>(target) * sample_count + i]);
            }
            int32_t link = heads_ptr[target];
            size_t traversed = 0;
            while (link >= 0 && traversed < reverse_inbox_count) {
              const size_t slot = static_cast<size_t>(link);
              const size_t local_source = slot / sample_count;
              consider(static_cast<uint32_t>(source_offset + local_source));
              link = inbox_links_ptr[slot];
              ++traversed;
            }
            for (size_t i = 0; i < sample_count; ++i) {
              reverse_graph[static_cast<size_t>(target) * sample_count + i] =
                  i < count ? best_ids[i] : kInvalidId;
            }
          });
          note_wait();
          q.wait_and_throw();
        }
        return OVVS_STATUS_SUCCESS;
      };

      ovvsStatus reverse_status = build_reverse(true, reverse_new_ptr);
      if (reverse_status != OVVS_STATUS_SUCCESS) return reverse_status;
      reverse_status = build_reverse(false, reverse_old_ptr);
      if (reverse_status != OVVS_STATUS_SUCCESS) return reverse_status;

      for (size_t vertex_offset = 0; vertex_offset < N; vertex_offset += proposal_chunk) {
        const size_t launch_vertices = std::min(proposal_chunk, N - vertex_offset);
        size_t global_size = 0;
        if (!checked_product(launch_vertices, work_group_size, global_size)) {
          return OVVS_STATUS_SHAPE_MISMATCH;
        }
        note_submission();
        q.fill(heads_ptr, -1, N);
        note_submission();
        q.fill(scalars_ptr, 0, 1u);
        note_wait();
        q.wait_and_throw();

        note_kernel();
        q.submit([&](sycl::handler& h) {
           sycl::local_accessor<uint32_t, 1> new_ids(sycl::range<1>(max_bi_samples), h);
           sycl::local_accessor<uint32_t, 1> old_ids(sycl::range<1>(max_bi_samples), h);
           sycl::local_accessor<float, 1> new_new_distances(sycl::range<1>(matrix_cells), h);
           sycl::local_accessor<float, 1> new_old_distances(sycl::range<1>(matrix_cells), h);
           sycl::local_accessor<uint32_t, 1> minimum_ids(sycl::range<1>(minimum_cells), h);
           sycl::local_accessor<float, 1> minimum_distances(sycl::range<1>(minimum_cells), h);
           sycl::local_accessor<int32_t, 1> state(sycl::range<1>(2), h);
           h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size),
                                           sycl::range<1>(work_group_size)),
                          [=](sycl::nd_item<1> item) {
             const size_t lid = item.get_local_linear_id();
             const size_t vertex = vertex_offset + item.get_group_linear_id();
             if (lid == 0) {
               for (size_t i = 0; i < max_bi_samples; ++i) {
                 new_ids[i] = kInvalidId;
                 old_ids[i] = kInvalidId;
               }
               size_t new_count = 0;
               size_t old_count = 0;
               auto add_unique = [](auto& ids, size_t& count, size_t capacity, uint32_t id) {
                 if (id == kInvalidId) return;
                 for (size_t i = 0; i < count; ++i) {
                   if (ids[i] == id) return;
                 }
                 if (count < capacity) ids[count++] = id;
               };
               size_t forward_new = 0;
               size_t forward_old = 0;
               for (size_t edge = 0; edge < DEG; ++edge) {
                 const uint32_t tagged = current_ids[vertex * DEG + edge];
                 const uint32_t raw = tagged & kIdMask;
                 if ((tagged & kNewMask) != 0u) {
                   if (forward_new < sample_count) {
                     add_unique(new_ids, new_count, max_bi_samples, raw);
                     ++forward_new;
                   }
                 } else if (forward_old < sample_count) {
                   add_unique(old_ids, old_count, max_bi_samples, raw);
                   ++forward_old;
                 }
               }
               for (size_t i = 0; i < sample_count; ++i) {
                 add_unique(new_ids, new_count, max_bi_samples,
                            reverse_new_ptr[vertex * sample_count + i]);
               }
               for (size_t i = 0; i < sample_count; ++i) {
                 const uint32_t candidate = reverse_old_ptr[vertex * sample_count + i];
                 bool promoted_new = false;
                 for (size_t j = 0; j < new_count; ++j) {
                   if (new_ids[j] == candidate) {
                     promoted_new = true;
                     break;
                   }
                 }
                 if (!promoted_new) {
                   add_unique(old_ids, old_count, max_bi_samples, candidate);
                 }
               }
               for (size_t i = 0; i < old_count;) {
                 bool promoted_new = false;
                 for (size_t j = 0; j < new_count; ++j) {
                   if (old_ids[i] == new_ids[j]) {
                     promoted_new = true;
                     break;
                   }
                 }
                 if (promoted_new) {
                   for (size_t j = i + 1u; j < old_count; ++j) old_ids[j - 1u] = old_ids[j];
                   --old_count;
                 } else {
                   ++i;
                 }
               }
               state[0] = static_cast<int32_t>(new_count);
               state[1] = static_cast<int32_t>(old_count);
             }
             item.barrier(sycl::access::fence_space::local_space);

             const size_t new_count = static_cast<size_t>(state[0]);
             const size_t old_count = static_cast<size_t>(state[1]);
              auto point_distance = [&](uint32_t lhs, uint32_t rhs) {
               float l2 = 0.f;
               float ip = 0.f;
               float nx = 0.f;
               float ny = 0.f;
               const bool needs_ip = met == static_cast<int>(OVVS_METRIC_INNER_PRODUCT) ||
                                     met == static_cast<int>(OVVS_METRIC_COSINE_EXPANDED);
               for (size_t d = 0; d < D; ++d) {
                 const float a = DS[static_cast<size_t>(lhs) * D + d];
                 const float b = DS[static_cast<size_t>(rhs) * D + d];
                 if (!needs_ip) {
                   const float delta = a - b;
                   l2 += delta * delta;
                 } else {
                   ip += a * b;
                   if (met == static_cast<int>(OVVS_METRIC_COSINE_EXPANDED)) {
                     nx += a * a;
                     ny += b * b;
                   }
                 }
               }
               if (!needs_ip) {
                 return met == static_cast<int>(OVVS_METRIC_L2_SQRT_EXPANDED) ? sycl::sqrt(l2)
                                                                             : l2;
               }
               if (met == static_cast<int>(OVVS_METRIC_INNER_PRODUCT)) return -ip;
               const float safe_nx = nx > 1e-12f ? nx : 1e-12f;
               const float safe_ny = ny > 1e-12f ? ny : 1e-12f;
                return 1.f - ip / (sycl::sqrt(safe_nx) * sycl::sqrt(safe_ny));
              };
              auto checked_distance = [&](float distance) {
                if (sycl::isfinite(distance)) return distance;
                sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    invalid(scalars_ptr[1]);
                invalid.store(1);
                return std::numeric_limits<float>::max();
              };

              for (size_t cell = lid; cell < new_count * new_count;
                  cell += work_group_size) {
               const size_t i = cell / new_count;
               const size_t j = cell % new_count;
               if (i == j) {
                 new_new_distances[cell] = std::numeric_limits<float>::max();
               } else if (i < j) {
                 const float distance =
                     checked_distance(point_distance(new_ids[i], new_ids[j]));
                 new_new_distances[i * new_count + j] = distance;
                 new_new_distances[j * new_count + i] = distance;
               }
             }
             for (size_t cell = lid; cell < new_count * old_count;
                  cell += work_group_size) {
               const size_t i = cell / old_count;
               const size_t j = cell % old_count;
               const float distance = checked_distance(point_distance(new_ids[i], old_ids[j]));
               new_old_distances[i * old_count + j] = distance;
             }
             item.barrier(sycl::access::fence_space::local_space);

             auto nearer = [](float lhs_distance, uint32_t lhs_id,
                              float rhs_distance, uint32_t rhs_id) {
               return lhs_distance < rhs_distance ||
                      (lhs_distance == rhs_distance && lhs_id < rhs_id);
             };
             const size_t new_new_minimum_base = 0;
             const size_t new_old_row_minimum_base = new_count;
             const size_t new_old_column_minimum_base = 2u * new_count;
             const size_t minimum_count = 2u * new_count + old_count;
             for (size_t minimum_index = lid; minimum_index < minimum_count;
                  minimum_index += work_group_size) {
               uint32_t best_id = kInvalidId;
               float best_distance = std::numeric_limits<float>::max();
               if (minimum_index < new_old_row_minimum_base) {
                 const size_t i = minimum_index;
                 for (size_t j = 0; j < new_count; ++j) {
                   if (i == j) continue;
                   const float distance = new_new_distances[i * new_count + j];
                   if (best_id == kInvalidId ||
                       nearer(distance, new_ids[j], best_distance, best_id)) {
                     best_id = new_ids[j];
                     best_distance = distance;
                   }
                 }
               } else if (minimum_index < new_old_column_minimum_base) {
                 const size_t i = minimum_index - new_old_row_minimum_base;
                 for (size_t j = 0; j < old_count; ++j) {
                   const float distance = new_old_distances[i * old_count + j];
                   if (best_id == kInvalidId ||
                       nearer(distance, old_ids[j], best_distance, best_id)) {
                     best_id = old_ids[j];
                     best_distance = distance;
                   }
                 }
               } else {
                 const size_t j = minimum_index - new_old_column_minimum_base;
                 for (size_t i = 0; i < new_count; ++i) {
                   const float distance = new_old_distances[i * old_count + j];
                   if (best_id == kInvalidId ||
                       nearer(distance, new_ids[i], best_distance, best_id)) {
                     best_id = new_ids[i];
                     best_distance = distance;
                   }
                 }
               }
               minimum_ids[minimum_index] = best_id;
               minimum_distances[minimum_index] = best_distance;
             }
             item.barrier(sycl::access::fence_space::local_space);

             if (lid == 0) {
               const size_t proposal_base =
                   item.get_group_linear_id() * proposals_per_vertex;
               size_t proposal_ordinal = 0;
               auto emit_directed = [&](uint32_t target, uint32_t candidate, float distance) {
                 if (target == candidate || target == kInvalidId || candidate == kInvalidId ||
                     !sycl::isfinite(distance) || proposal_ordinal >= proposals_per_vertex) {
                   return;
                 }
                 const size_t slot_size = proposal_base + proposal_ordinal++;
                 const int32_t slot = static_cast<int32_t>(slot_size);
                 proposal_ids_ptr[slot_size] = candidate;
                 proposal_distances_ptr[slot_size] = distance;
                 sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>
                     target_head(heads_ptr[target]);
                 const int32_t prior = target_head.exchange(slot);
                 inbox_links_ptr[slot_size] = prior;
                 if (prior == -1) {
                   sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                       active_count(scalars_ptr[0]);
                   const int32_t active_slot = active_count.fetch_add(1);
                   active_targets_ptr[static_cast<size_t>(active_slot)] = target;
                 }
               };
               auto emit_reciprocal = [&](uint32_t a, uint32_t b, float distance) {
                 emit_directed(a, b, distance);
                 emit_directed(b, a, distance);
               };

               for (size_t i = 0; i < new_count; ++i) {
                 uint32_t best_id = minimum_ids[new_new_minimum_base + i];
                 float best_distance =
                     minimum_distances[new_new_minimum_base + i];
                 if (best_id != kInvalidId) {
                   emit_reciprocal(new_ids[i], best_id, best_distance);
                 }

                 best_id = minimum_ids[new_old_row_minimum_base + i];
                 best_distance =
                     minimum_distances[new_old_row_minimum_base + i];
                 if (best_id != kInvalidId) {
                   emit_reciprocal(new_ids[i], best_id, best_distance);
                 }
               }

               for (size_t j = 0; j < old_count; ++j) {
                 const uint32_t best_id =
                     minimum_ids[new_old_column_minimum_base + j];
                 const float best_distance =
                     minimum_distances[new_old_column_minimum_base + j];
                 if (best_id != kInvalidId) {
                   emit_reciprocal(old_ids[j], best_id, best_distance);
                 }
               }
             }
           });
         });
        note_wait();
        q.wait_and_throw();

        int32_t scalar_values[2] = {0, 0};
        note_d2h(sizeof(scalar_values));
        q.memcpy(scalar_values, scalars_ptr, sizeof(scalar_values));
        note_wait();
        q.wait_and_throw();
        if (scalar_values[1] != 0) return OVVS_STATUS_INVALID_ARGUMENT;
        const int32_t active_count = scalar_values[0];
        if (active_count < 0 || static_cast<size_t>(active_count) > N) {
          return OVVS_STATUS_ERROR;
        }

        note_kernel();
        q.parallel_for(sycl::range<1>(static_cast<size_t>(active_count)),
                       [=](sycl::id<1> active_index) {
          const uint32_t target = active_targets_ptr[active_index];
          const size_t row = static_cast<size_t>(target) * DEG;
          int32_t link = heads_ptr[target];
          size_t traversed = 0;
          while (link >= 0 && traversed < proposal_inbox_count) {
            const size_t slot = static_cast<size_t>(link);
            const uint32_t candidate = proposal_ids_ptr[slot];
            const float candidate_distance = proposal_distances_ptr[slot];
            link = inbox_links_ptr[slot];
            ++traversed;
            if (candidate == target || candidate == kInvalidId ||
                static_cast<size_t>(candidate) >= N || !sycl::isfinite(candidate_distance)) {
              continue;
            }

            bool duplicate = false;
            for (size_t edge = 0; edge < DEG; ++edge) {
              if ((next_ids[row + edge] & kIdMask) == candidate) {
                duplicate = true;
                break;
              }
            }
            if (duplicate) continue;

            const float worst_distance = next_distances[row + DEG - 1u];
            const uint32_t worst_id = next_ids[row + DEG - 1u] & kIdMask;
            if (candidate_distance > worst_distance ||
                (candidate_distance == worst_distance && candidate >= worst_id)) {
              continue;
            }

            uint32_t tagged_candidate = kNewMask | candidate;
            size_t new_rank = 0;
            for (size_t edge = 0; edge < DEG; ++edge) {
              const uint32_t tagged = current_ids[row + edge];
              const bool edge_is_new = (tagged & kNewMask) != 0u;
              if ((tagged & kIdMask) == candidate) {
                if (!edge_is_new || new_rank < sample_count) {
                  tagged_candidate = candidate;
                }
                break;
              }
              if (edge_is_new) ++new_rank;
            }

            size_t insert = 0;
            while (insert < DEG) {
              const float distance = next_distances[row + insert];
              const uint32_t id = next_ids[row + insert] & kIdMask;
              if (candidate_distance < distance ||
                  (candidate_distance == distance && candidate < id)) {
                break;
              }
              ++insert;
            }
            if (insert >= DEG) continue;
            for (size_t edge = DEG - 1u; edge > insert; --edge) {
              next_ids[row + edge] = next_ids[row + edge - 1u];
              next_distances[row + edge] = next_distances[row + edge - 1u];
            }
            next_ids[row + insert] = tagged_candidate;
            next_distances[row + insert] = candidate_distance;
          }
        });
        note_wait();
        q.wait_and_throw();
      }

      for (size_t vertex_offset = 0; vertex_offset < N;
           vertex_offset += kVertexGroupsPerLaunch) {
        const size_t launch_vertices = std::min(kVertexGroupsPerLaunch, N - vertex_offset);
        size_t global_size = 0;
        if (!checked_product(launch_vertices, work_group_size, global_size)) {
          return OVVS_STATUS_SHAPE_MISMATCH;
        }
        note_kernel();
        q.submit([&](sycl::handler& h) {
          sycl::local_accessor<int32_t, 1> row_hash(
              sycl::range<1>(convergence_hash_capacity), h);
          h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size),
                                           sycl::range<1>(work_group_size)),
                         [=](sycl::nd_item<1> item) {
            const size_t lid = item.get_local_linear_id();
            const size_t vertex = vertex_offset + item.get_group_linear_id();
            auto group = item.get_group();
            for (size_t slot = lid; slot < convergence_hash_capacity;
                 slot += work_group_size) {
              row_hash[slot] = -1;
            }
            item.barrier(sycl::access::fence_space::local_space);
            if (lid == 0) {
              for (size_t edge = 0; edge < DEG; ++edge) {
                const int32_t id = static_cast<int32_t>(
                    current_ids[vertex * DEG + edge] & kIdMask);
                size_t slot = static_cast<size_t>(static_cast<uint32_t>(id) * 0x9e3779b9u) &
                              (convergence_hash_capacity - 1u);
                while (row_hash[slot] != -1 && row_hash[slot] != id) {
                  slot = (slot + 1u) & (convergence_hash_capacity - 1u);
                }
                row_hash[slot] = id;
              }
            }
            item.barrier(sycl::access::fence_space::local_space);
            uint32_t local_changed = 0;
            uint32_t local_pending_new = 0;
            for (size_t edge = lid; edge < DEG; edge += work_group_size) {
              const uint32_t tagged = next_ids[vertex * DEG + edge];
              const int32_t id = static_cast<int32_t>(tagged & kIdMask);
              size_t slot = static_cast<size_t>(static_cast<uint32_t>(id) * 0x9e3779b9u) &
                            (convergence_hash_capacity - 1u);
              while (row_hash[slot] != -1 && row_hash[slot] != id) {
                slot = (slot + 1u) & (convergence_hash_capacity - 1u);
              }
              if (row_hash[slot] != id) ++local_changed;
              if ((tagged & kNewMask) != 0u) ++local_pending_new;
            }
            const uint32_t row_changed =
                sycl::reduce_over_group(group, local_changed, sycl::plus<uint32_t>());
            const uint32_t row_pending_new =
                sycl::reduce_over_group(group, local_pending_new, sycl::plus<uint32_t>());
            if (lid == 0) {
              heads_ptr[vertex] = static_cast<int32_t>(row_changed);
              active_targets_ptr[vertex] = row_pending_new;
            }
          });
        });
      }
      note_wait();
      q.wait_and_throw();
      note_d2h(N * sizeof(int32_t));
      q.memcpy(changed_per_row.data(), heads_ptr, N * sizeof(int32_t));
      note_d2h(N * sizeof(uint32_t));
      q.memcpy(pending_new_per_row.data(), active_targets_ptr, N * sizeof(uint32_t));
      note_wait();
      q.wait_and_throw();

      int64_t changed_edges = 0;
      int64_t pending_new_edges = 0;
      for (const int32_t row_changed : changed_per_row) {
        if (row_changed < 0 || row_changed > degree) return OVVS_STATUS_ERROR;
        changed_edges += static_cast<int64_t>(row_changed);
      }
      for (const uint32_t row_pending_new : pending_new_per_row) {
        if (row_pending_new > static_cast<uint32_t>(degree)) return OVVS_STATUS_ERROR;
        pending_new_edges += static_cast<int64_t>(row_pending_new);
      }
      const double change_ratio =
          static_cast<double>(changed_edges) / static_cast<double>(graph_count);
      if (!std::isfinite(change_ratio) || change_ratio < 0.0 || change_ratio > 1.0) {
        return OVVS_STATUS_ERROR;
      }

      std::swap(current_ids, next_ids);
      std::swap(current_distances, next_distances);
      ++iterations_run;
      last_changed_edges = changed_edges;
      last_pending_new_edges = pending_new_edges;
      last_change_ratio = change_ratio;
      if (iterations_run >= 2 && change_ratio < kTerminationThreshold &&
          pending_new_edges == 0) {
        converged = true;
        break;
      }
    }

    note_kernel();
    q.parallel_for(sycl::range<1>(graph_count), [=](sycl::id<1> index) {
      next_ids[index] = current_ids[index] & kIdMask;
    });
    note_wait();
    q.wait_and_throw();
    note_d2h(graph_plane_bytes);
    q.memcpy(graph, next_ids, graph_plane_bytes);
    note_wait();
    q.wait_and_throw();

    if (stats) {
      stats->iterations = iterations_run;
      stats->final_changed_edges = last_changed_edges;
      stats->final_pending_new_edges = last_pending_new_edges;
      stats->converged = converged;
    }

    {
      std::lock_guard<std::mutex> lock(r.nndescent_stats_mutex);
      r.nndescent_iterations_run = iterations_run;
      r.nndescent_changed_edges = last_changed_edges;
      r.nndescent_pending_new_edges = last_pending_new_edges;
      r.nndescent_change_ratio = last_change_ratio;
      r.nndescent_peak_device_bytes = static_cast<int64_t>(device_bytes);
      r.nndescent_converged = converged;
    }
    return OVVS_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return OVVS_STATUS_OOM;
  } catch (...) {
    return OVVS_STATUS_ERROR;
  }
#else
  (void)r;
  (void)dataset;
  (void)n;
  (void)dim;
  (void)metric;
  (void)degree;
  (void)iters;
  (void)graph;
  (void)stats;
  return OVVS_STATUS_DEVICE_UNAVAILABLE;
#endif
}

ovvsStatus gpu_cagra_optimize_ranked(ResourcesData& r, const int32_t* initial,
                                      int64_t n, int32_t initial_degree,
                                      int32_t final_degree,
                                      std::vector<int32_t>& output) {
  output.clear();
#if defined(OVVS_WITH_SYCL)
  (void)r;
  constexpr size_t kMaxInitialDegree = 64u;
  constexpr size_t kMaxReverseTake = kMaxInitialDegree / 2u;
  if (!initial || n <= 1 || initial_degree <= 0 || final_degree <= 0 ||
      final_degree > initial_degree || initial_degree >= n || final_degree >= n) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  if (n > std::numeric_limits<int32_t>::max()) return OVVS_STATUS_SHAPE_MISMATCH;
  if (static_cast<size_t>(initial_degree) > kMaxInitialDegree) {
    return OVVS_STATUS_UNSUPPORTED;
  }
  if (!sycl_gpu_available()) return OVVS_STATUS_DEVICE_UNAVAILABLE;

  try {
    auto& q = gpu_queue();
    const auto device = q.get_device();
    if (!device.has(sycl::aspect::usm_device_allocations)) {
      return OVVS_STATUS_UNSUPPORTED;
    }
    const size_t N = static_cast<size_t>(n);
    const size_t I = static_cast<size_t>(initial_degree);
    const size_t F = static_cast<size_t>(final_degree);
    const size_t reverse_take_limit = F / 2u;
    const size_t padded = next_power_of_two(I);
    size_t initial_count = 0;
    size_t final_count = 0;
    size_t global_items = 0;
    if (padded == 0 || !checked_product(N, I, initial_count) ||
        !checked_product(N, F, final_count) ||
        !checked_product(N, padded, global_items) || N == std::numeric_limits<size_t>::max()) {
      return OVVS_STATUS_SHAPE_MISMATCH;
    }

    const size_t max_work_group_size =
        device.get_info<sycl::info::device::max_work_group_size>();
    const size_t local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
    if (padded > max_work_group_size || padded * sizeof(uint64_t) > local_mem_size) {
      return OVVS_STATUS_UNSUPPORTED;
    }

    const auto bytes_for = [](size_t count, size_t width, size_t& bytes) {
      return checked_product(count, width, bytes);
    };
    size_t initial_bytes = 0;
    size_t lookup_bytes = 0;
    size_t final_bytes = 0;
    size_t counts_bytes = 0;
    size_t offsets_bytes = 0;
    size_t reverse_bytes = 0;
    if (!bytes_for(initial_count, sizeof(int32_t), initial_bytes) ||
        !bytes_for(initial_count, sizeof(uint64_t), lookup_bytes) ||
        !bytes_for(final_count, sizeof(int32_t), final_bytes) ||
        !bytes_for(N, sizeof(uint32_t), counts_bytes) ||
        !bytes_for(N + 1u, sizeof(uint64_t), offsets_bytes) ||
        !bytes_for(final_count, sizeof(uint64_t), reverse_bytes)) {
      return OVVS_STATUS_SHAPE_MISMATCH;
    }
    const size_t max_alloc = device.get_info<sycl::info::device::max_mem_alloc_size>();
    if (std::max({initial_bytes, lookup_bytes, final_bytes, counts_bytes,
                  offsets_bytes, reverse_bytes}) > max_alloc) {
      return OVVS_STATUS_OOM;
    }
    const auto checked_sum = [](std::initializer_list<size_t> values, size_t& total) {
      total = 0;
      for (size_t value : values) {
        if (value > std::numeric_limits<size_t>::max() - total) return false;
        total += value;
      }
      return true;
    };
    size_t phase_a_bytes = 0;
    size_t phase_b_bytes = 0;
    if (!checked_sum({initial_bytes, lookup_bytes, final_bytes, sizeof(int32_t)},
                     phase_a_bytes) ||
        !checked_sum({final_bytes, counts_bytes, offsets_bytes, counts_bytes,
                      reverse_bytes, final_bytes, sizeof(int32_t)},
                     phase_b_bytes)) {
      return OVVS_STATUS_SHAPE_MISMATCH;
    }
    const uint64_t global_mem = device.get_info<sycl::info::device::global_mem_size>();
    if (static_cast<uint64_t>(std::max(phase_a_bytes, phase_b_bytes)) > global_mem) {
      return OVVS_STATUS_OOM;
    }

    ScopedDeviceUsm<int32_t> forward(q, final_count);
    ScopedDeviceUsm<int32_t> status(q, 1u);
    int32_t* const forward_ptr = forward.get();
    int32_t* const status_ptr = status.get();
    int32_t host_status = 0;

    {
      ScopedDeviceUsm<int32_t> initial_device(q, initial_count);
      ScopedDeviceUsm<uint64_t> rank_lookup(q, initial_count);
      int32_t* const initial_ptr = initial_device.get();
      uint64_t* const rank_ptr = rank_lookup.get();

      const sycl::event input_copy = q.memcpy(initial_ptr, initial, initial_bytes);
      const sycl::event status_fill = q.fill(status_ptr, 0, 1u);
      const sycl::event rank_event = q.submit([&](sycl::handler& h) {
        h.depends_on(input_copy);
        h.depends_on(status_fill);
        sycl::local_accessor<uint64_t, 1> row_keys(sycl::range<1>(padded), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global_items), sycl::range<1>(padded)),
            [=](sycl::nd_item<1> item) {
              const size_t row = item.get_group_linear_id();
              const size_t lane = item.get_local_linear_id();
              uint64_t key = std::numeric_limits<uint64_t>::max();
              if (lane < I) {
                const int32_t id = initial_ptr[row * I + lane];
                if (id < 0 || static_cast<size_t>(id) >= N ||
                    static_cast<size_t>(id) == row) {
                  sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device,
                                   sycl::access::address_space::global_space>(*status_ptr)
                      .fetch_or(1);
                } else {
                  key = (static_cast<uint64_t>(static_cast<uint32_t>(id)) << 32u) |
                        static_cast<uint32_t>(lane);
                }
              }
              row_keys[lane] = key;
              item.barrier(sycl::access::fence_space::local_space);
              for (size_t width = 2u; width <= padded; width <<= 1u) {
                for (size_t stride = width >> 1u; stride != 0; stride >>= 1u) {
                  const size_t other = lane ^ stride;
                  if (other > lane) {
                    const bool ascending = (lane & width) == 0;
                    const uint64_t left = row_keys[lane];
                    const uint64_t right = row_keys[other];
                    if ((left > right) == ascending) {
                      row_keys[lane] = right;
                      row_keys[other] = left;
                    }
                  }
                  item.barrier(sycl::access::fence_space::local_space);
                }
              }
              if (lane < I) {
                const uint64_t sorted = row_keys[lane];
                rank_ptr[row * I + lane] = sorted;
                if (lane != 0 && (sorted >> 32u) == (row_keys[lane - 1u] >> 32u)) {
                  sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device,
                                   sycl::access::address_space::global_space>(*status_ptr)
                      .fetch_or(2);
                }
              }
            });
      });
      sycl::event validation_copy =
          q.memcpy(&host_status, status_ptr, sizeof(host_status), rank_event);
      validation_copy.wait_and_throw();
      if (host_status != 0) return OVVS_STATUS_ERROR;

      sycl::event forward_event = q.submit([&](sycl::handler& h) {
        h.depends_on(rank_event);
        sycl::local_accessor<uint64_t, 1> candidates(sycl::range<1>(padded), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global_items), sycl::range<1>(padded)),
            [=](sycl::nd_item<1> item) {
              const size_t row = item.get_group_linear_id();
              const size_t lane = item.get_local_linear_id();
              uint64_t key = std::numeric_limits<uint64_t>::max();
              if (lane < I) {
                const int32_t y = initial_ptr[row * I + lane];
                uint32_t detours = 0;
                const uint64_t needle =
                    static_cast<uint64_t>(static_cast<uint32_t>(y)) << 32u;
                for (size_t z_rank = 0; z_rank < lane; ++z_rank) {
                  const int32_t z = initial_ptr[row * I + z_rank];
                  const size_t z_base = static_cast<size_t>(z) * I;
                  size_t lo = 0;
                  size_t hi = I;
                  while (lo < hi) {
                    const size_t mid = lo + (hi - lo) / 2u;
                    if (rank_ptr[z_base + mid] < needle) {
                      lo = mid + 1u;
                    } else {
                      hi = mid;
                    }
                  }
                  uint32_t z_y_rank = static_cast<uint32_t>(I);
                  if (lo < I && (rank_ptr[z_base + lo] >> 32u) ==
                                    static_cast<uint32_t>(y)) {
                    z_y_rank = static_cast<uint32_t>(rank_ptr[z_base + lo]);
                  }
                  if (z_y_rank < lane) ++detours;
                }
                key = (static_cast<uint64_t>(detours) << 38u) |
                      (static_cast<uint64_t>(lane) << 32u) |
                      static_cast<uint32_t>(y);
              }
              candidates[lane] = key;
              item.barrier(sycl::access::fence_space::local_space);
              for (size_t width = 2u; width <= padded; width <<= 1u) {
                for (size_t stride = width >> 1u; stride != 0; stride >>= 1u) {
                  const size_t other = lane ^ stride;
                  if (other > lane) {
                    const bool ascending = (lane & width) == 0;
                    const uint64_t left = candidates[lane];
                    const uint64_t right = candidates[other];
                    if ((left > right) == ascending) {
                      candidates[lane] = right;
                      candidates[other] = left;
                    }
                  }
                  item.barrier(sycl::access::fence_space::local_space);
                }
              }
              if (lane < F) {
                forward_ptr[row * F + lane] =
                    static_cast<int32_t>(static_cast<uint32_t>(candidates[lane]));
              }
            });
      });
      forward_event.wait_and_throw();
    }

    ScopedDeviceUsm<uint32_t> reverse_counts(q, N);
    ScopedDeviceUsm<uint64_t> reverse_offsets(q, N + 1u);
    ScopedDeviceUsm<uint32_t> reverse_cursors(q, N);
    ScopedDeviceUsm<uint64_t> reverse_keys(q, final_count);
    ScopedDeviceUsm<int32_t> result(q, final_count);
    uint32_t* const counts_ptr = reverse_counts.get();
    uint64_t* const offsets_ptr = reverse_offsets.get();
    uint32_t* const cursors_ptr = reverse_cursors.get();
    uint64_t* const reverse_ptr = reverse_keys.get();
    int32_t* const result_ptr = result.get();

    const sycl::event counts_fill = q.fill(counts_ptr, uint32_t{0}, N);
    const sycl::event count_event = q.submit([&](sycl::handler& h) {
      h.depends_on(counts_fill);
      h.parallel_for(sycl::range<1>(final_count), [=](sycl::id<1> index) {
        const int32_t destination = forward_ptr[index];
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            count(counts_ptr[static_cast<size_t>(destination)]);
        count.fetch_add(1u);
      });
    });
    const sycl::event scan_event = q.submit([&](sycl::handler& h) {
      h.depends_on(count_event);
      h.single_task([=]() {
        uint64_t offset = 0;
        offsets_ptr[0] = 0;
        for (size_t row = 0; row < N; ++row) {
          offset += static_cast<uint64_t>(counts_ptr[row]);
          offsets_ptr[row + 1u] = offset;
        }
        if (offset != static_cast<uint64_t>(final_count)) {
          sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>(*status_ptr)
              .fetch_or(4);
        }
      });
    });
    const sycl::event cursors_fill = q.fill(cursors_ptr, uint32_t{0}, N);
    const sycl::event scatter_event = q.submit([&](sycl::handler& h) {
      h.depends_on(scan_event);
      h.depends_on(cursors_fill);
      h.parallel_for(sycl::range<2>(N, F), [=](sycl::id<2> index) {
        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            status_value(*status_ptr);
        if (status_value.load() != 0) return;
        const uint32_t source = static_cast<uint32_t>(index[0]);
        const uint32_t rank = static_cast<uint32_t>(index[1]);
        const size_t edge = static_cast<size_t>(source) * F + rank;
        const size_t destination = static_cast<size_t>(forward_ptr[edge]);
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            cursor(cursors_ptr[destination]);
        const uint32_t local_slot = cursor.fetch_add(1u);
        const uint64_t slot = offsets_ptr[destination] + local_slot;
        if (slot >= offsets_ptr[destination + 1u]) {
          sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>(*status_ptr)
              .fetch_or(8);
          return;
        }
        reverse_ptr[static_cast<size_t>(slot)] =
            (static_cast<uint64_t>(rank) << 32u) | source;
      });
    });
    const sycl::event result_fill = q.fill(result_ptr, int32_t{-1}, final_count);
    const sycl::event select_event = q.submit([&](sycl::handler& h) {
      h.depends_on(scatter_event);
      h.depends_on(result_fill);
      h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> row_id) {
        const size_t row = row_id[0];
        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            status_value(*status_ptr);
        if (status_value.load() != 0) return;
        if (cursors_ptr[row] != counts_ptr[row]) {
          sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>(*status_ptr)
              .fetch_or(16);
          return;
        }
        uint64_t best[kMaxReverseTake];
        size_t best_count = 0;
        for (size_t i = 0; i < kMaxReverseTake; ++i) {
          best[i] = std::numeric_limits<uint64_t>::max();
        }
        const size_t forward_base = row * F;
        const uint64_t begin = offsets_ptr[row];
        const uint64_t end = offsets_ptr[row + 1u];
        for (uint64_t cursor = begin; cursor < end; ++cursor) {
          const uint64_t key = reverse_ptr[static_cast<size_t>(cursor)];
          const int32_t source =
              static_cast<int32_t>(static_cast<uint32_t>(key));
          bool already_forward = false;
          for (size_t rank = 0; rank < F; ++rank) {
            if (forward_ptr[forward_base + rank] == source) {
              already_forward = true;
              break;
            }
          }
          if (already_forward || reverse_take_limit == 0) continue;
          size_t position = 0;
          while (position < best_count && best[position] < key) ++position;
          if (position >= reverse_take_limit) continue;
          const size_t upper =
              best_count < reverse_take_limit ? best_count : reverse_take_limit - 1u;
          for (size_t move = upper; move > position; --move) {
            best[move] = best[move - 1u];
          }
          best[position] = key;
          if (best_count < reverse_take_limit) ++best_count;
        }

        const size_t forward_take = F - best_count;
        size_t forward_position = 0;
        size_t reverse_position = 0;
        size_t output_position = 0;
        while (forward_position < forward_take || reverse_position < best_count) {
          if (forward_position < forward_take) {
            result_ptr[forward_base + output_position++] =
                forward_ptr[forward_base + forward_position++];
          }
          if (reverse_position < best_count) {
            result_ptr[forward_base + output_position++] =
                static_cast<int32_t>(static_cast<uint32_t>(best[reverse_position++]));
          }
        }
        if (output_position != F) {
          sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>(*status_ptr)
              .fetch_or(32);
          return;
        }
        for (size_t left = 0; left < F; ++left) {
          const int32_t id = result_ptr[forward_base + left];
          if (id < 0 || static_cast<size_t>(id) >= N ||
              static_cast<size_t>(id) == row) {
            sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(*status_ptr)
                .fetch_or(64);
          }
          for (size_t right = 0; right < left; ++right) {
            if (result_ptr[forward_base + right] == id) {
              sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::global_space>(*status_ptr)
                  .fetch_or(128);
            }
          }
        }
      });
    });

    std::vector<int32_t> staged(final_count);
    host_status = 0;
    sycl::event result_copy = q.memcpy(staged.data(), result_ptr, final_bytes, select_event);
    sycl::event final_status_copy =
        q.memcpy(&host_status, status_ptr, sizeof(host_status), select_event);
    result_copy.wait_and_throw();
    final_status_copy.wait_and_throw();
    if (host_status != 0) return OVVS_STATUS_ERROR;
    output.swap(staged);
    return OVVS_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return OVVS_STATUS_OOM;
  } catch (...) {
    return OVVS_STATUS_ERROR;
  }
#else
  (void)r;
  (void)initial;
  (void)n;
  (void)initial_degree;
  (void)final_degree;
  return OVVS_STATUS_DEVICE_UNAVAILABLE;
#endif
}

#if defined(OVVS_WITH_SYCL)
/* ---------------------------------------------------------------------------------------
   uint8 mirror of a graph-walk dataset.

   The walk is transaction bound, not bandwidth or compute bound: a 128-dim fp32 row costs
   eight cache-line transactions, a uint8 row costs two. Measured in isolation, shrinking the
   rows alone buys ~1.11x because the old geometry can only ever have one candidate in flight;
   combined with one sub-group per candidate it is ~4.5-5.9x, because only then do the eight
   in-flight rows actually overlap.

   The mirror is EXACT, not an approximation, when every element is an integer in [0,255] and
   dim <= kInt8MaxDim: each partial sum is then at most dim*255^2 < 2^24, so the fp32 result
   holds the exact integer whatever order the reduction runs in. That makes the int8 walk
   bitwise identical to the fp32 walk on such data (SIFT, and any uint8-native corpus), which
   is why no recall tolerance is needed for it. Arbitrary float embeddings do not qualify and
   silently keep the fp32 path; lossy scalar quantization with a rerank pass is a separate,
   separately measured decision.
   --------------------------------------------------------------------------------------- */
constexpr size_t kInt8MaxDim = 256u;

static bool gpu_int8_mirror_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("OVVS_GPU_INT8");
    return !(env && env[0] == '0');
  }();
  return enabled;
}

/* Cache guard for the (pointer, rows, dim) key. The free generation makes pointer reuse
   impossible to miss; the strided content sample additionally catches a dataset rewritten in
   place. Both are cheap enough to run on every search call. */
static uint64_t sample_fingerprint(const float* p, size_t count, size_t rows, size_t dim) {
  uint64_t h = 1469598103934665603ull;
  const auto mix = [&h](uint64_t v) {
    h ^= v;
    h *= 1099511628211ull;
  };
  mix(rows);
  mix(dim);
  mix(count);
  mix(ovvs_usm_free_generation());
  const size_t samples = std::min<size_t>(count, 64u);
  const size_t stride = samples == 0 ? 1u : std::max<size_t>(1u, count / samples);
  for (size_t i = 0; i < count; i += stride) {
    uint32_t bits = 0;
    std::memcpy(&bits, &p[i], sizeof(bits));
    mix(bits);
  }
  return h;
}

/* Returns the cached mirror, building it on first use. nullptr means "this dataset does not
   qualify" and is itself cached, so a float corpus pays the scan once, not once per search. */
static const uint8_t* gpu_int8_mirror(ResourcesData& r, sycl::queue& q, const float* dataset,
                                      size_t rows, size_t dim, size_t count) {
  if (!gpu_int8_mirror_enabled() || dim == 0 || dim > kInt8MaxDim || count == 0) return nullptr;
  const uint64_t fingerprint = sample_fingerprint(dataset, count, rows, dim);

  std::lock_guard<std::mutex> lock(r.gpu_int8_mutex);
  if (r.gpu_int8_source == dataset && r.gpu_int8_rows == rows && r.gpu_int8_dim == dim) {
    if (r.gpu_int8_fingerprint == fingerprint) {
      return r.gpu_int8_usable ? static_cast<const uint8_t*>(r.gpu_int8_data) : nullptr;
    }
    if (!r.gpu_int8_usable && !r.gpu_int8_data) {
      /* Same corpus shape, new fingerprint (mutation bumped the generation), but the
         verdict was REJECT: content edits cannot make a rejected corpus eligible in any
         workload we care about, and re-scanning 4n bytes per mutation chunk was measured
         at -23%% on 1M updates. Re-cache the reject under the new fingerprint; fp32 is
         always correct. A rows/dim/pointer change still rescans from scratch. */
      r.gpu_int8_fingerprint = fingerprint;
      return nullptr;
    }
  }

  if (r.gpu_int8_data) {
    q.wait_and_throw();
    ovvs_usm_free(r.gpu_int8_data);
    r.gpu_int8_data = nullptr;
  }
  r.gpu_int8_source = dataset;
  r.gpu_int8_rows = rows;
  r.gpu_int8_dim = dim;
  r.gpu_int8_fingerprint = fingerprint;
  r.gpu_int8_usable = false;

  /* Shared USM, not device workspace: the same physical DDR on this iGPU, but the CPU
     walk can read the mirror too -- one copy serves both engines (G3 stays intact). */
  void* raw = ovvs_usm_malloc(count + sizeof(int32_t));
  if (!raw) return nullptr;
  if (sycl::get_pointer_type(raw, q.get_context()) == sycl::usm::alloc::unknown) {
    /* SYCL alloc failed and ovvs_usm_malloc fell back to std::malloc: the GPU kernel
       cannot read that. Bail out to the fp32 path rather than fault. */
    ovvs_usm_free(raw);
    return nullptr;
  }
  auto* mirror = static_cast<uint8_t*>(raw);
  auto* reject = reinterpret_cast<int32_t*>(mirror + count);
  try {
    q.memset(reject, 0, sizeof(int32_t)).wait_and_throw();
    q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
       const float v = dataset[i];
       const bool ok = v >= 0.f && v <= 255.f && v == sycl::floor(v);
       if (!ok) {
         sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                          sycl::access::address_space::global_space>(*reject)
             .store(1);
       }
       mirror[i] = static_cast<uint8_t>(ok ? static_cast<int32_t>(v) : 0);
     }).wait_and_throw();
    int32_t rejected = 0;
    q.memcpy(&rejected, reject, sizeof(rejected)).wait_and_throw();
    if (rejected != 0) {
      ovvs_usm_free(raw);
      return nullptr;
    }
  } catch (...) {
    ovvs_usm_free(raw);
    return nullptr;
  }
  r.gpu_int8_data = raw;
  r.gpu_int8_usable = true;
  return mirror;
}
#endif

/* Host-readable view of the int8 mirror for the CPU walk leg. Builds (or returns the
   cached) shared-USM mirror; nullptr when SYCL is off, the GPU is absent, or the
   dataset does not qualify -- callers fall back to fp32. */
const uint8_t* gpu_cagra_int8_mirror_host(ResourcesData& r, const float* dataset, int64_t rows,
                                          int64_t dim) {
#if defined(OVVS_WITH_SYCL)
  if (!gpu_available() || rows <= 0 || dim <= 0) return nullptr;
  try {
    const size_t count = static_cast<size_t>(rows) * static_cast<size_t>(dim);
    return gpu_int8_mirror(r, gpu_queue(), dataset, static_cast<size_t>(rows),
                           static_cast<size_t>(dim), count);
  } catch (...) {
    return nullptr;
  }
#else
  (void)r;
  (void)dataset;
  (void)rows;
  (void)dim;
  return nullptr;
#endif
}

/* Explicit mirror invalidation for in-place dataset mutation. The fingerprint samples
   only 64 strided values, so an update that misses every sample would leave a STALE
   mirror serving searches -- a correctness hole that predates the CPU leg (GPU search
   after update had the same exposure). Mutation bumps the same generation the
   fingerprint already mixes in, forcing a rebuild on the next walk. */
void ovvs_gpu_mirror_invalidate() { g_usm_free_generation.fetch_add(1, std::memory_order_relaxed); }

bool gpu_cagra_walk(ResourcesData& r, const float* dataset, int64_t n, int64_t dim, ovvsMetric metric,
                    const int32_t* graph, int32_t degree, const float* queries, int64_t nq, int64_t k,
                    int32_t itopk, int32_t search_width, const uint8_t* bitset, int64_t* neighbors,
                    float* distances, const uint16_t* dataset_f16, const uint8_t* dataset_u8) {
#if defined(OVVS_WITH_SYCL)
  if (!gpu_available()) return false;
  /* fp16 primary storage: no fp32 dataset. The walk can still run entirely on the
     caller-provided int8 mirror; prim_graph_walk only offers such batches after
     verifying every query qualifies for the int8 path, so the fp32 fallback branches
     are provably dead and DS stays null. */
  if (!dataset && !dataset_u8 && !dataset_f16) return false;
  if (!graph || !queries || !neighbors || !distances) return false;
  /* fp32-storage walks may install or rebuild the Resources-cached int8 mirror, and a
     rebuild frees the buffer a concurrent walk could still be reading -- serialize
     them. fp16 walks have no cached GPU state (DS16/mirror8 are caller-owned, staging
     is per-call scoped) and may overlap freely. */
  std::unique_lock<std::mutex> mirror_walk_lock;
  if (dataset != nullptr) {
    mirror_walk_lock = std::unique_lock<std::mutex>(r.gpu_walk_mutex);
  }
  if (n <= 0 || nq <= 0 || k <= 0 || dim <= 0 || degree <= 0) return false;
  if (n > std::numeric_limits<int32_t>::max() || k > std::numeric_limits<int32_t>::max()) return false;
  if (metric != OVVS_METRIC_L2_EXPANDED && metric != OVVS_METRIC_L2_SQRT_EXPANDED &&
      metric != OVVS_METRIC_INNER_PRODUCT && metric != OVVS_METRIC_COSINE_EXPANDED) {
    return false;
  }
  itopk = std::max(itopk, static_cast<int32_t>(k));
  search_width = std::max(1, search_width);
  if (itopk > std::numeric_limits<int>::max() / 6) return false;
  constexpr size_t kMaxVisitedBytesPerQuery = 8u * 1024u * 1024u;
  constexpr size_t kVisitedAllocationTarget = 64u * 1024u * 1024u;
  /* Candidates staged per gather/score/admit round. Temporarily env-tunable: `commit_frontier`
     has two O(tile) per-lane passes (duplicate scan, prefix compaction) so its cost grows with
     the SQUARE of this, while every other phase grows linearly -- which makes sweeping it a
     clean way to size those passes against the rest of the walk. */
  const size_t kFrontierTileSize = []() -> size_t {
    const char* env = std::getenv("OVVS_CAGRA_TILE");
    if (env && *env) {
      const long parsed = std::strtol(env, nullptr, 10);
      if (parsed >= 8 && parsed <= 256) return static_cast<size_t>(parsed);
    }
    return 64u;
  }();
  try {
    auto& q = gpu_queue();
    const size_t N = static_cast<size_t>(n);
    const size_t D = static_cast<size_t>(dim);
    const size_t NQ = static_cast<size_t>(nq);
    const size_t DEG = static_cast<size_t>(degree);
    const size_t KK = static_cast<size_t>(k);
    const size_t IT = static_cast<size_t>(itopk);
    const size_t BEAM = IT * 2u;
    const size_t SW = static_cast<size_t>(search_width);
    const int met = static_cast<int>(metric);
    /* The walk marks SW beam slots expanded per iteration and only re-opens a slot when an
       eviction lands on an already-expanded one, so it terminates near BEAM/SW. Measured maxima
       (32-seed era) are 71-76 where BEAM/SW is 64 and 38-40 where it is 32 - the excess over
       BEAM/SW does not grow with the ratio, so an additive slack of 24 clears every observed
       maximum by >= 12 while keeping visited_capacity a true upper bound on visits. Against the
       previous +32 this is what moves the degree-32 configs (and d16 itopk=64/width=4) back
       across a power-of-two boundary: their visited tables halve, 64 KiB -> 32 KiB per query.
       The cap never binding means traversal, and so the output, is unchanged. If it ever did
       bind, the walk stops early: MAX_ITERATIONS in the walk counters is the canary. */
    const size_t iteration_bound = (SW > 0 ? BEAM / SW : BEAM) + 24u;
    const int max_iters =
        static_cast<int>(std::max<size_t>(24u, std::min<size_t>(iteration_bound, 1u << 20)));
    if (BEAM < IT || SW > std::numeric_limits<int32_t>::max()) return false;

    size_t ds_count = 0, graph_count = 0, query_count = 0, output_count = 0;
    if (!checked_product(N, D, ds_count) || !checked_product(N, DEG, graph_count) ||
        !checked_product(NQ, D, query_count) || !checked_product(NQ, KK, output_count)) {
      return false;
    }
    if (ds_count > std::numeric_limits<size_t>::max() / sizeof(float) ||
        graph_count > std::numeric_limits<size_t>::max() / sizeof(int32_t) ||
        query_count > std::numeric_limits<size_t>::max() / sizeof(float) ||
        output_count > std::numeric_limits<size_t>::max() / sizeof(int64_t)) {
      return false;
    }

    const auto device = q.get_device();
    const size_t max_wg = device.get_info<sycl::info::device::max_work_group_size>();
    size_t work_group_size = 1;
    while (work_group_size <= max_wg / 2 && work_group_size < 128) work_group_size <<= 1;
    /* The sub-group-per-candidate geometry is the recorded winner (the 2.4-5.7x walk
       rewrite); its A/B isolation knob is retired. Devices without sub-group 16 still
       fall back through force_sub_group_16 below. */
    constexpr bool subgroup_geometry = true;
    bool force_sub_group_16 = false;
    {
      const auto widths = device.get_info<sycl::info::device::sub_group_sizes>();
      force_sub_group_16 = std::find(widths.begin(), widths.end(), 16u) != widths.end() &&
                           work_group_size % 16u == 0;
    }

    size_t local_bytes = 0;
    if (!checked_product(BEAM, sizeof(int32_t) + sizeof(float) + sizeof(uint8_t), local_bytes)) return false;
    size_t pick_bytes = 0;
    size_t frontier_bytes = 0;
    if (!checked_product(SW, sizeof(int32_t), pick_bytes) ||
        !checked_product(kFrontierTileSize, 3u * sizeof(int32_t) + sizeof(float), frontier_bytes)) {
      return false;
    }
    constexpr size_t kLocalMetadataBytes = 64u;
    const size_t max_size = std::numeric_limits<size_t>::max();
    if (pick_bytes > max_size - frontier_bytes - kLocalMetadataBytes) return false;
    const size_t auxiliary_bytes = pick_bytes + frontier_bytes + kLocalMetadataBytes;
    if (local_bytes > max_size - auxiliary_bytes) return false;
    local_bytes += auxiliary_bytes;
    const size_t local_mem = device.get_info<sycl::info::device::local_mem_size>();
    if (local_bytes > local_mem) return false;

    if (IT > std::numeric_limits<size_t>::max() / 16u || SW > std::numeric_limits<size_t>::max() / 32u) {
      return false;
    }
    /* Stop expanding once `patience` consecutive iterations admit nothing to the beam.
       The walk otherwise runs until every beam slot has been expanded, which is a fixed
       ~BEAM/search_width iterations whether or not the frontier is still improving.
       0 disables it, which is the default until the recall/throughput frontier says
       otherwise -- it is the one change here that is NOT output-preserving. */
    const int stall_patience = []() -> int {
      const char* env = std::getenv("OVVS_CAGRA_PATIENCE");
      if (!env || !*env) return 0;
      const long parsed = std::strtol(env, nullptr, 10);
      if (parsed <= 0) return 0;
      return static_cast<int>(std::min<long>(parsed, 1 << 20));
    }();
    /* Shared with the CPU walk (mixer.cpp) and the build-path walk (graphs.cpp) so the two
       can never drift; CPU/GPU ID parity depends on an identical seed stream. */
    const size_t nseeds =
        static_cast<size_t>(cagra_seed_count(n, itopk, search_width));
    if (nseeds == 0) return false;

    /* Beam-membership dedup: replaces the global visited hash whenever the seed phase can
       never overfill the beam (nseeds <= itopk). Under that condition every previously-scored
       id is either still in the beam -- caught by an SLM membership scan at commit -- or was
       evicted or rejected while the beam was full, at which point its distance was >= the
       beam's max; the max is non-increasing once the beam is full and distances are
       deterministic, so the existing >=-worst pre-filter rejects it at every later encounter.
       Before the beam first fills nothing is ever rejected, so every scored id IS in the beam.
       Output is therefore bit-identical; only evals/query rises (revisited candidates are
       re-scored, then rejected), which the work counters price. In exchange the walk sheds its
       only per-query global state: the CAS probe chain (~2,300 probes/query), the table init,
       and 16-64 KiB/query of footprint that thrashes L2 as the batch grows. The nseeds > itopk
       case keeps the hash: a seed evicted at capacity itopk could otherwise be readmitted
       while the beam grows toward 2*itopk.

       MEASURED at one-work-group-per-query: a 14% REGRESSION (the extra evals cost more than
       the probes saved while only 64 queries are in flight), so the hash stays the default
       there. The dedup exists for the sub-group-per-query mapping, whose per-query state must
       fit SLM; it is enabled alongside it below, or forced for A/B via OVVS_CAGRA_BEAM_DEDUP=1
       (=0 forces the hash and thereby one-work-group-per-query). Experiment-lane knob. */
    const int qpw_requested = [&]() -> int {
      const char* env = std::getenv("OVVS_CAGRA_QPW");
      /* VERDICT (Xe-LPG): the sub-group-per-query mapping loses at practical batches
         (0.82x at b1024 vs the classic path) -- 1 stays the default. The lane is kept
         for SKUs with more thread slots / SLM per Xe-core; see docs/env.md. */
      if (!env || !*env) return 1;
      const long parsed = std::strtol(env, nullptr, 10);
      return parsed == 8 ? 8 : 1;
    }();
    const bool beam_dedup = [&]() {
      const char* env = std::getenv("OVVS_CAGRA_BEAM_DEDUP");
      if (env && env[0] == '0') return false;
      if (env && env[0] == '1') return nseeds <= IT;
      return qpw_requested > 1 && nseeds <= IT;
    }();

    /* Sub-group-per-query mapping: one 128-item work-group hosts eight independent queries,
       one per sub-group, and that walk path uses no work-group barriers at all -- every
       collective is sub-group scoped, so queries advance and finish independently and the
       device holds 8x more queries in flight. That concurrency is what the latency-bound walk
       is starved of: the b1-vs-b1024 accounting shows per-query latency is the same alone or
       co-resident, i.e. QPS = resident queries / latency, and residency is capped by thread
       slots, 1/8 of which each query occupies today. Requires the sub-group geometry at
       sub-group 16 (so sub-groups == queries == 8), the beam dedup (per-query state must be
       SLM-resident; the visited hash is neither needed nor wired in that path), and the
       8-slot SLM footprint. Falls back to one query per work-group otherwise.
       Experiment lane: OVVS_CAGRA_QPW=8 opts in; on Xe-LPG the classic path won. */
    size_t QPW = 1u;
    size_t bh_capacity = 1u; /* beam-membership hash entries per query (sub-group path) */
    /* Companion to the beam dedup in the sub-group path: a small per-query direct-mapped
       cache of rejected or evicted ids, probed (one SLM read) alongside the beam scan at
       commit. Filtering on ANY subset of previously-scored ids is exact -- the same
       eviction-monotonicity proof -- and a collision merely forgets history, so entries are
       overwritten freely. A recency RING was tried first and parked: revisits are NOT
       temporally local (64 entries caught 16% of revisit evals, 256 caught 47% while paying
       an O(ring) scan); the hashed cache reaches arbitrary history at O(1). Without either,
       the dedup's extra evals (+43% at d32) ate the mapping's entire concurrency win.
       Experiment-lane knob: OVVS_CAGRA_SEEN_RING (entries, rounded down to a power of
       two; 0 disables; measured default 0 -- any cache loses residency on Xe-LPG). */
    size_t seen_ring = 0u;
    /* beam_dedup is no longer required for the sub-group mapping: req_gate showed the
       fabric has ~8x request headroom at the walk's rates and achieved residency at
       practical batches is ~150-170 queries (~5 MiB of visited tables, not the feared
       16 MiB), so the classic global hash -- which avoids the dedup's +43% revisit evals
       entirely -- is a legitimate sgq companion, selected via OVVS_CAGRA_BEAM_DEDUP=0. */
    if (qpw_requested == 8 && subgroup_geometry && force_sub_group_16 &&
        work_group_size == 128u) {
      seen_ring = [&]() -> size_t {
        const char* env = std::getenv("OVVS_CAGRA_SEEN_RING");
        long parsed = 0; /* MEASURED default: any cache loses -- SLM per query caps resident
                            queries, and residency is worth more than the evals recovered
                            (cache 256 = 0.46x, 512 = 0.50x vs no cache 0.82x at b1024). */
        if (env && *env) parsed = std::strtol(env, nullptr, 10);
        if (parsed <= 0) return 0u;
        size_t s = static_cast<size_t>(std::min<long>(parsed, 4096));
        while ((s & (s - 1u)) != 0u) s &= s - 1u; /* round down to a power of two */
        return s;
      }();
      /* The sub-group path's own SLM footprint: beam + picks + one shared tile (ids
         compacted in place) + a keep byte per tile entry + the beam-membership hash + the
         optional seen cache. Everything else lives in registers. Residency per Xe-core is
         128KiB / (8 slots), and every byte here costs resident queries -- but req_gate
         showed achieved residency at b1024 is fill-limited (~170 sub-groups) well below
         the SLM cap, so the hash's half-KiB is free at practical batch sizes. */
      bh_capacity = next_power_of_two(2u * BEAM);
      const size_t sg_slot_bytes =
          BEAM * (sizeof(int32_t) + sizeof(float) + sizeof(uint8_t)) + SW * sizeof(int32_t) +
          kFrontierTileSize * (sizeof(int32_t) + sizeof(uint8_t)) +
          bh_capacity * sizeof(int32_t) + seen_ring * sizeof(int32_t) + 64u;
      size_t slots_bytes = 0;
      if (kFrontierTileSize <= 64u &&
          checked_product(sg_slot_bytes, static_cast<size_t>(8u), slots_bytes) &&
          slots_bytes <= local_mem) {
        QPW = 8u;
        local_bytes = slots_bytes;
      } else {
        seen_ring = 0u;
      }
    }

    if (SW > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(max_iters)) return false;
    const uint64_t expansion_budget = static_cast<uint64_t>(max_iters) * static_cast<uint64_t>(SW);
    if (DEG != 0 && expansion_budget > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(DEG)) {
      return false;
    }
    const uint64_t edge_budget = expansion_budget * static_cast<uint64_t>(DEG);
    const uint64_t visit_budget = std::min<uint64_t>(N, static_cast<uint64_t>(nseeds) + edge_budget);
    if (visit_budget > std::numeric_limits<size_t>::max() / 2u) return false;
    /* Temporary sizing knob. The table's membership semantics are unaffected by its capacity
       (only the load factor, hence the average probe count, moves), so scaling it up is a clean
       way to ask how much the walk pays for probe locality: a larger table has the same probe
       count but a working set that is that much worse behaved in cache. */
    const size_t visited_headroom = []() -> size_t {
      const char* env = std::getenv("OVVS_CAGRA_VISITED_MULT");
      if (env && *env) {
        const long parsed = std::strtol(env, nullptr, 10);
        if (parsed >= 1 && parsed <= 64) return static_cast<size_t>(parsed);
      }
      return 2u;
    }();
    /* Beam dedup needs no table at all; 2 entries keep the pointer arithmetic and the
       power-of-two mask valid while costing 8 bytes per query. */
    const size_t visited_capacity =
        beam_dedup ? 2u
                   : next_power_of_two(
                         std::max<size_t>(2u, static_cast<size_t>(visit_budget) * visited_headroom));
    if (visited_capacity == 0 || visited_capacity > kMaxVisitedBytesPerQuery / sizeof(int32_t)) return false;

    const bool ds_direct = dataset ? gpu_pointer_accessible(q, dataset) : true;
    const bool ds16_direct = dataset_f16 ? gpu_pointer_accessible(q, dataset_f16) : true;
    const bool graph_direct = gpu_pointer_accessible(q, graph);
    const bool query_direct = gpu_pointer_accessible(q, queries);
    const bool out_i_direct = gpu_pointer_accessible(q, neighbors);
    const bool out_d_direct = gpu_pointer_accessible(q, distances);
    const bool bitset_direct = bitset && gpu_pointer_accessible(q, bitset);
    const size_t bitset_bytes = (N + 7u) / 8u;
    const size_t dataset_bytes = ds_count * sizeof(float);
    const size_t graph_bytes = graph_count * sizeof(int32_t);
    size_t index_upload_bytes = 0;
    if (!ds_direct) {
      index_upload_bytes = dataset_bytes;
    }
    if (!graph_direct) {
      if (index_upload_bytes > std::numeric_limits<size_t>::max() - graph_bytes) return false;
      index_upload_bytes += graph_bytes;
    }
    if (index_upload_bytes > static_cast<size_t>(std::numeric_limits<int64_t>::max())) return false;

    /* The index copies stay scoped: they only exist when the caller's dataset/graph are not
       device-accessible, which is the exceptional path and is already reported as an upload. */
    ScopedDeviceUsm<float> ds_copy(q, ds_direct ? 0u : ds_count);
    ScopedDeviceUsm<uint16_t> ds16_copy(q, (dataset_f16 && !ds16_direct) ? ds_count : 0u);
    ScopedDeviceUsm<int32_t> graph_copy(q, graph_direct ? 0u : graph_count);

    /* Built and cached once per dataset; nullptr whenever the corpus is not exactly
       representable as uint8, in which case every path below stays fp32. */
    const uint8_t* DS8 = dataset_u8 ? dataset_u8
                         : (dataset && ds_direct ? gpu_int8_mirror(r, q, dataset, N, D, ds_count)
                                                 : nullptr);
    if (!dataset && !DS8 && !dataset_f16) return false;

    const size_t visited_bytes_per_query = visited_capacity * sizeof(int32_t);
    size_t queries_per_launch = std::max<size_t>(1u, kVisitedAllocationTarget / visited_bytes_per_query);
    queries_per_launch = std::min(queries_per_launch, NQ);
    size_t visited_count = 0;
    if (!checked_product(queries_per_launch, visited_capacity, visited_count)) return false;
    const size_t max_alloc = device.get_info<sycl::info::device::max_mem_alloc_size>();
    if (visited_count > max_alloc / sizeof(int32_t)) return false;

    bool count_work = false;
    {
      std::lock_guard<std::mutex> lock(r.cagra_walk_counter_mutex);
      count_work = r.cagra_walk_counters_enabled;
    }

    /* One persistent per-Resources device buffer holds the visited table plus any query,
       output, bitset and counter staging, instead of four allocate/free/drain cycles per
       call. Sections are 256-byte aligned so each stays independently addressable. */
    constexpr size_t kSection = 256u;
    const auto align_up = [](size_t v) { return (v + kSection - 1u) / kSection * kSection; };
    size_t ws_bytes = 0;
    const size_t off_visited = ws_bytes;
    ws_bytes = align_up(ws_bytes + visited_count * sizeof(int32_t));
    const size_t off_query = ws_bytes;
    if (!query_direct) ws_bytes = align_up(ws_bytes + query_count * sizeof(float));
    const size_t off_out_i = ws_bytes;
    if (!out_i_direct) ws_bytes = align_up(ws_bytes + output_count * sizeof(int64_t));
    const size_t off_out_d = ws_bytes;
    if (!out_d_direct) ws_bytes = align_up(ws_bytes + output_count * sizeof(float));
    const size_t off_bitset = ws_bytes;
    if (bitset && !bitset_direct) ws_bytes = align_up(ws_bytes + bitset_bytes);
    const size_t off_counters = ws_bytes;
    if (count_work) ws_bytes = align_up(ws_bytes + OVVS_CAGRA_WALK_COUNTER_COUNT * sizeof(int64_t));
    /* uint8 query mirror plus a one-word reject flag. Only reserved when the dataset already
       qualified, so a float corpus pays nothing for it. */
    const size_t off_query8 = ws_bytes;
    if (DS8) ws_bytes = align_up(ws_bytes + query_count + sizeof(int32_t));
    /* Query queue for the sub-group-per-query mapping: persistent sub-groups claim query
       indices from this counter, so a finished query's thread immediately starts the next
       one instead of idling until its whole work-group drains. VTune measured the
       fixed-assignment version at 29% occupancy against the classic mapping's 84% -- the
       coarse work-groups, each held open by its slowest of eight queries, were starving the
       device of the very concurrency the mapping exists to provide. */
    const size_t off_queue = ws_bytes;
    if (QPW > 1) ws_bytes = align_up(ws_bytes + sizeof(int32_t));

    if (r.gpu_walk_workspace_bytes < ws_bytes) {
      if (r.gpu_walk_workspace) {
        q.wait_and_throw();
        ovvs_gpu_workspace_free(r.gpu_walk_workspace);
        r.gpu_walk_workspace = nullptr;
        r.gpu_walk_workspace_bytes = 0;
      }
      void* fresh = ovvs_gpu_workspace_alloc(ws_bytes);
      if (!fresh) return false;
      r.gpu_walk_workspace = fresh;
      r.gpu_walk_workspace_bytes = ws_bytes;
    }
    auto* ws = static_cast<uint8_t*>(r.gpu_walk_workspace);

    int32_t* VISITED = reinterpret_cast<int32_t*>(ws + off_visited);
    float* query_stage = query_direct ? nullptr : reinterpret_cast<float*>(ws + off_query);
    int64_t* out_i_stage = out_i_direct ? nullptr : reinterpret_cast<int64_t*>(ws + off_out_i);
    float* out_d_stage = out_d_direct ? nullptr : reinterpret_cast<float*>(ws + off_out_d);
    uint8_t* bitset_stage = (bitset && !bitset_direct) ? ws + off_bitset : nullptr;
    int64_t* COUNTERS = count_work ? reinterpret_cast<int64_t*>(ws + off_counters) : nullptr;

    const float* DS = ds_direct ? dataset : ds_copy.get();
    /* fp16 primary storage: native half loads, converted to fp32 in-register (a single
       hardware instruction on Xe). Priority inside every distance function is
       int8 mirror (4x fewer bytes, exact) -> fp16 (2x) -> fp32, per query. */
    const sycl::half* DS16 = reinterpret_cast<const sycl::half*>(
        dataset_f16 ? (ds16_direct ? dataset_f16 : ds16_copy.get()) : nullptr);
    const int32_t* G = graph_direct ? graph : graph_copy.get();
    const float* Q = query_direct ? queries : query_stage;
    int64_t* OUTI = out_i_direct ? neighbors : out_i_stage;
    float* OUTD = out_d_direct ? distances : out_d_stage;
    const uint8_t* BS = !bitset ? nullptr : (bitset_direct ? bitset : bitset_stage);

    int32_t* QUEUE = QPW > 1 ? reinterpret_cast<int32_t*>(ws + off_queue) : nullptr;

    if (!ds_direct && dataset) q.memcpy(ds_copy.get(), dataset, dataset_bytes);
    if (dataset_f16 && !ds16_direct) {
      q.memcpy(ds16_copy.get(), dataset_f16, ds_count * sizeof(uint16_t));
    }
    if (!graph_direct) q.memcpy(graph_copy.get(), graph, graph_bytes);
    if (QUEUE) q.memset(QUEUE, 0, sizeof(int32_t));
    sycl::event query_upload;
    if (!query_direct) query_upload = q.memcpy(query_stage, queries, query_count * sizeof(float));
    if (bitset && !bitset_direct) q.memcpy(bitset_stage, bitset, bitset_bytes);
    if (COUNTERS) q.memset(COUNTERS, 0, sizeof(int64_t) * OVVS_CAGRA_WALK_COUNTER_COUNT);

    /* The queries are quantized per call -- they are tiny next to the dataset -- and the whole
       int8 path is abandoned for this call if any query element falls outside the exact range,
       so a uint8 corpus queried with float vectors still returns fp32-correct answers. The
       verdict is left in device memory and read by the walk itself: reading it back here would
       cost three extra queue drains on every search, which at batch one is a measurable share
       of the whole call. */
    const uint8_t* Q8 = nullptr;
    const int32_t* Q8_OK = nullptr;
    if (DS8) {
      auto* query8 = ws + off_query8;
      auto* reject = reinterpret_cast<int32_t*>(query8 + query_count);
      const float* src = Q;
      const sycl::event cleared = q.memset(reject, 0, sizeof(int32_t));
      q.submit([&](sycl::handler& h) {
        h.depends_on({cleared, query_upload});
        h.parallel_for(sycl::range<1>(query_count), [=](sycl::id<1> i) {
          const float v = src[i];
          const bool ok = v >= 0.f && v <= 255.f && v == sycl::floor(v);
          if (!ok) {
            sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(*reject)
                .store(1);
          }
          query8[i] = static_cast<uint8_t>(ok ? static_cast<int32_t>(v) : 0);
        });
      });
      Q8 = query8;
      Q8_OK = reject;
    }
    q.wait_and_throw();

    for (size_t query_offset = 0; query_offset < NQ; query_offset += queries_per_launch) {
      const size_t launch_queries = std::min(queries_per_launch, NQ - query_offset);
      /* qpw > 1 implies beam_dedup implies a single launch in practice; reset the claim
         counter anyway if a second launch ever happens (each launch ends in a wait). */
      if (QUEUE && query_offset != 0) {
        q.memset(QUEUE, 0, sizeof(int32_t));
        q.wait_and_throw();
      }
      q.submit([&](sycl::handler& h) {
         /* One slot per query hosted by the work-group: slot 0 for the classic mapping,
            eight independent slots for the sub-group-per-query mapping. SLM per query caps
            resident queries per Xe-core, so the sub-group path uses only the beam, the
            picks, one shared tile (raw ids compacted in place) and a keep byte per tile
            entry; the classic-mapping-only arrays shrink to a token element there. */
         const size_t wg_tile = QPW > 1 ? 1u : kFrontierTileSize;
         sycl::local_accessor<int32_t, 1> candidate_ids(sycl::range<1>(BEAM * QPW), h);
         sycl::local_accessor<float, 1> candidate_distances(sycl::range<1>(BEAM * QPW), h);
         sycl::local_accessor<uint8_t, 1> candidate_expanded(sycl::range<1>(BEAM * QPW), h);
         sycl::local_accessor<int32_t, 1> picks(sycl::range<1>(SW * QPW), h);
         sycl::local_accessor<int32_t, 1> frontier_ids(sycl::range<1>(kFrontierTileSize * QPW), h);
         sycl::local_accessor<float, 1> frontier_scores(sycl::range<1>(wg_tile), h);
         sycl::local_accessor<int32_t, 1> raw_ids(sycl::range<1>(wg_tile), h);
         sycl::local_accessor<int32_t, 1> keep_flags(sycl::range<1>(wg_tile), h);
         sycl::local_accessor<int32_t, 1> state(sycl::range<1>(8), h);
         sycl::local_accessor<uint32_t, 1> query_hash_state(sycl::range<1>(1), h);
         sycl::local_accessor<uint8_t, 1> tile_keep(
             sycl::range<1>(QPW > 1 ? kFrontierTileSize * QPW : 1u), h);
         sycl::local_accessor<int32_t, 1> beam_hash(
             sycl::range<1>(QPW > 1 ? bh_capacity * QPW : 1u), h);
         sycl::local_accessor<int32_t, 1> seen_ids(
             sycl::range<1>(std::max<size_t>(1u, seen_ring * QPW)), h);
         auto walk_body = [=](sycl::nd_item<1> item) {
           const size_t lid = item.get_local_linear_id();
           const size_t qi = query_offset + item.get_group_linear_id();
           const size_t visited_base = item.get_group_linear_id() * visited_capacity;
           auto subgroup = item.get_sub_group();
           const size_t subgroup_id = subgroup.get_group_linear_id();
           const size_t subgroup_lane = subgroup.get_local_linear_id();
           const size_t subgroup_size = subgroup.get_local_linear_range();
           const size_t subgroup_count = subgroup.get_group_linear_range();
           /* Uniform across the work-group: one broadcast read of the verdict the query
              quantizer left behind, instead of a host round trip before the launch. */
           const bool use_int8 = DS8 != nullptr && Q8_OK[0] == 0;

           if (QPW > 1) {
             /* ---- Sub-group-per-query walk: one sub-group owns one query end to end. ----
                No work-group barriers exist on this path (the tail guard below returns early,
                which only a barrier-free path permits): every collective is sub-group scoped,
                so queries advance and finish independently, a serial phase idles 16 lanes
                instead of 128, and the device holds QPW times more queries in flight. The
                traversal is the SAME algorithm as the one-work-group path under beam_dedup --
                same tile order, same frozen per-tile threshold, same admission sequence --
                so recall and every work counter must be bit-identical to that path.
                Requires beam_dedup (guaranteed by the host gate): there is no visited hash
                here, and the walk's whole per-query state lives in its SLM slot. */
             const size_t lane = subgroup_lane;
             const size_t sgsz = subgroup_size;
             const size_t qslot = subgroup_id;
             const size_t cb = qslot * BEAM;                 /* beam slice */
             const size_t fb = qslot * kFrontierTileSize;    /* frontier slice */
             const size_t pb = qslot * SW;                   /* picks slice */
             const size_t rb = qslot * seen_ring;            /* seen-cache slice */

             /* Per-query control state lives in uniform registers, not SLM: a single
                sub-group owns it and every update below happens in uniform control flow.
                Mutable, because a persistent sub-group serves many queries in turn; the
                lambdas below capture by reference and always see the current query. */
             size_t sq = 0;        /* the query this sub-group is currently serving */
             size_t vbase = 0;     /* this query's visited-table region (hash mode) */
             uint32_t qhash = 0;
             int32_t s_count = 0;  /* beam occupancy */
             int32_t s_worst = 0;  /* worst beam slot */
             int32_t s_admit = 0;  /* admitted-anything-this-iteration flag */
             int32_t s_stall = 0;  /* patience counter */
             int64_t q_iters = 0;  /* iterations of the current query */
             int64_t e_evals = 0, e_seed_evals = 0, e_admits = 0, e_survivors = 0;
             int64_t e_queries = 0, e_iters_total = 0, e_iters_max = 0;
             auto seen_slot = [&](int32_t id) {
               return (static_cast<uint32_t>(id) * 2654435761u) & (seen_ring - 1u);
             };

             /* Exact SLM hash of CURRENT beam membership: open addressing, linear probe,
                -1 empty, -2 tombstone. Inserted on admission, tombstoned on eviction, and
                rebuilt from the beam when tombstones crowd it, so membership equals
                candidate_ids[0..s_count) at every commit -- the commit-time probe then
                replaces an O(beam) SLM scan per candidate with ~1 read. req_gate showed the
                walk is bookkeeping-bound with the beam scan as the largest term. */
             const size_t hb = qslot * bh_capacity;
             const uint32_t bh_mask = static_cast<uint32_t>(bh_capacity - 1u);
             int32_t bh_tombs = 0;
             auto bh_probe0 = [&](int32_t id) {
               return (static_cast<uint32_t>(id) * 2654435761u) & bh_mask;
             };
             auto bh_contains = [&](int32_t id) {
               uint32_t s = bh_probe0(id);
               for (;;) {
                 const int32_t v = beam_hash[hb + s];
                 if (v == id) return true;
                 if (v == -1) return false;
                 s = (s + 1u) & bh_mask;
               }
             };
             auto bh_insert = [&](int32_t id) {
               uint32_t s = bh_probe0(id);
               for (;;) {
                 const int32_t v = beam_hash[hb + s];
                 if (v < 0) { /* empty or tombstone */
                   if (lane == 0) beam_hash[hb + s] = id;
                   if (v == -2) --bh_tombs;
                   sycl::group_barrier(subgroup);
                   return;
                 }
                 s = (s + 1u) & bh_mask;
               }
             };
             auto bh_remove = [&](int32_t id) {
               uint32_t s = bh_probe0(id);
               for (;;) {
                 if (beam_hash[hb + s] == id) {
                   if (lane == 0) beam_hash[hb + s] = -2;
                   ++bh_tombs;
                   sycl::group_barrier(subgroup);
                   return;
                 }
                 s = (s + 1u) & bh_mask;
               }
             };
             /* Classic visited-hash membership for the sub-group path (beam_dedup off):
                the same CAS probe-and-insert as the work-group path, against this query's
                claimed region. Only this sub-group ever touches the region; lanes of one
                tile insert concurrently, and every id reaching it is tile-distinct. */
             auto sg_visit_once = [&](int32_t id) {
               using Slot = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                             sycl::memory_scope::work_group,
                                             sycl::access::address_space::global_space>;
               const size_t vmask = visited_capacity - 1u;
               size_t slot = (static_cast<uint32_t>(id) * 2654435761u) & vmask;
               for (size_t probe = 0; probe < visited_capacity; ++probe) {
                 Slot entry(VISITED[vbase + slot]);
                 int32_t current = entry.load();
                 if (current == id) return false;
                 if (current == -1) {
                   int32_t expected = -1;
                   if (entry.compare_exchange_strong(expected, id)) return true;
                   if (expected == id) return false;
                 }
                 slot = (slot + 1u) & vmask;
               }
               return false;
             };

             auto bh_rebuild_if_crowded = [&]() {
               if (static_cast<size_t>(s_count + bh_tombs) * 4u < bh_capacity * 3u) return;
               for (size_t i = lane; i < bh_capacity; i += sgsz) beam_hash[hb + i] = -1;
               bh_tombs = 0;
               sycl::group_barrier(subgroup);
               for (int32_t s = 0; s < s_count; ++s) {
                 const int32_t id = candidate_ids[cb + static_cast<size_t>(s)];
                 uint32_t p = bh_probe0(id);
                 for (;;) {
                   if (beam_hash[hb + p] == -1) {
                     if (lane == 0) beam_hash[hb + p] = id;
                     break;
                   }
                   p = (p + 1u) & bh_mask;
                 }
                 sycl::group_barrier(subgroup);
               }
             };

             auto sg_allowed = [&](int32_t id) {
               return BS == nullptr || ((BS[static_cast<size_t>(id) >> 3] >> (id & 7)) & 1u) != 0;
             };

             const bool needs_ip_q = met == static_cast<int>(OVVS_METRIC_INNER_PRODUCT) ||
                                     met == static_cast<int>(OVVS_METRIC_COSINE_EXPANDED);
             const bool needs_norms_q = met == static_cast<int>(OVVS_METRIC_COSINE_EXPANDED);
             auto finish_q = [&](float l2, float ip, float nx, float ny) {
               if (!needs_ip_q) {
                 if (met == static_cast<int>(OVVS_METRIC_L2_SQRT_EXPANDED)) return sycl::sqrt(l2);
                 return l2;
               }
               if (met == static_cast<int>(OVVS_METRIC_INNER_PRODUCT)) return -ip;
               if (needs_norms_q) {
                 const float safe_nx = nx > 1e-12f ? nx : 1e-12f;
                 const float safe_ny = ny > 1e-12f ? ny : 1e-12f;
                 return 1.f - ip / (sycl::sqrt(safe_nx) * sycl::sqrt(safe_ny));
               }
               return 0.f;
             };

             /* Query-row hoist, exactly as in the one-work-group path but bound to sq and
                reloaded whenever this sub-group claims a new query. */
             const bool packed_q8 = use_int8 && D == 8u * sgsz;
             const bool reg_qf = !use_int8 && D == 8u * sgsz;
             uint32_t q_w0 = 0u, q_w1 = 0u;
             int32_t q_sq = 0;
             float q_f[8];
             auto load_query_regs = [&]() {
               q_sq = 0;
               if (packed_q8) {
                 const uint32_t* qp =
                     reinterpret_cast<const uint32_t*>(Q8 + sq * D + lane * 8u);
                 q_w0 = qp[0];
                 q_w1 = qp[1];
                 for (int j = 0; j < 4; ++j) {
                   const int32_t a0 = static_cast<int32_t>((q_w0 >> (8 * j)) & 0xFFu);
                   const int32_t a1 = static_cast<int32_t>((q_w1 >> (8 * j)) & 0xFFu);
                   q_sq += a0 * a0 + a1 * a1;
                 }
               }
               if (reg_qf) {
                 for (int k = 0; k < 8; ++k) {
                   q_f[k] = Q[sq * D + lane + static_cast<size_t>(k) * sgsz];
                 }
               }
             };

             auto sg_distance = [&](int32_t id) -> float {
               if (use_int8) {
                 if (packed_q8) {
                   const uint32_t* dp = reinterpret_cast<const uint32_t*>(
                       DS8 + static_cast<size_t>(id) * D + lane * 8u);
                   const uint32_t b_w0 = dp[0];
                   const uint32_t b_w1 = dp[1];
                   int32_t qd = 0, dd = 0;
                   for (int j = 0; j < 4; ++j) {
                     const int32_t b0 = static_cast<int32_t>((b_w0 >> (8 * j)) & 0xFFu);
                     const int32_t b1 = static_cast<int32_t>((b_w1 >> (8 * j)) & 0xFFu);
                     const int32_t a0 = static_cast<int32_t>((q_w0 >> (8 * j)) & 0xFFu);
                     const int32_t a1 = static_cast<int32_t>((q_w1 >> (8 * j)) & 0xFFu);
                     qd += a0 * b0 + a1 * b1;
                     dd += b0 * b0 + b1 * b1;
                   }
                   if (!needs_ip_q) {
                     const int32_t l2 = sycl::reduce_over_group(subgroup, q_sq + dd - 2 * qd,
                                                                sycl::plus<int32_t>());
                     return finish_q(static_cast<float>(l2), 0.f, 0.f, 0.f);
                   }
                   const int32_t ip = sycl::reduce_over_group(subgroup, qd, sycl::plus<int32_t>());
                   if (!needs_norms_q) return finish_q(0.f, static_cast<float>(ip), 0.f, 0.f);
                   const int32_t nx = sycl::reduce_over_group(subgroup, q_sq, sycl::plus<int32_t>());
                   const int32_t ny = sycl::reduce_over_group(subgroup, dd, sycl::plus<int32_t>());
                   return finish_q(0.f, static_cast<float>(ip), static_cast<float>(nx),
                                   static_cast<float>(ny));
                 }
                 const uint8_t* qrow = Q8 + sq * D;
                 const uint8_t* drow = DS8 + static_cast<size_t>(id) * D;
                 int32_t l2 = 0, ip = 0, nx = 0, ny = 0;
                 const size_t chunks = (D + 7u) / 8u;
                 for (size_t c = lane; c < chunks; c += sgsz) {
                   const size_t off = c * 8u;
                   const size_t lim = (D - off) < 8u ? (D - off) : 8u;
                   for (size_t j = 0; j < lim; ++j) {
                     const int32_t a = static_cast<int32_t>(qrow[off + j]);
                     const int32_t b = static_cast<int32_t>(drow[off + j]);
                     if (!needs_ip_q) {
                       const int32_t delta = a - b;
                       l2 += delta * delta;
                     } else {
                       ip += a * b;
                       if (needs_norms_q) {
                         nx += a * a;
                         ny += b * b;
                       }
                     }
                   }
                 }
                 if (!needs_ip_q) {
                   l2 = sycl::reduce_over_group(subgroup, l2, sycl::plus<int32_t>());
                   return finish_q(static_cast<float>(l2), 0.f, 0.f, 0.f);
                 }
                 ip = sycl::reduce_over_group(subgroup, ip, sycl::plus<int32_t>());
                 if (needs_norms_q) {
                   nx = sycl::reduce_over_group(subgroup, nx, sycl::plus<int32_t>());
                   ny = sycl::reduce_over_group(subgroup, ny, sycl::plus<int32_t>());
                 }
                 return finish_q(0.f, static_cast<float>(ip), static_cast<float>(nx),
                                 static_cast<float>(ny));
               }
               if (DS16) {
                 /* Two halfs per dword load: naive 2-byte loads de-coalesce and ran 40%
                    BEHIND the fp32 walk; packed dwords restore the fp32 access shape at
                    half the bytes (same fix the int8 path uses, one level up). */
                 float l2 = 0.f, ip = 0.f, nx = 0.f, ny = 0.f;
                 if ((D & 1u) == 0u) {
                   const uint32_t* row32 =
                       reinterpret_cast<const uint32_t*>(DS16 + static_cast<size_t>(id) * D);
                   for (size_t pr = lane; pr < D / 2u; pr += sgsz) {
                     const uint32_t packed = row32[pr];
                     const float b0 = static_cast<float>(
                         sycl::bit_cast<sycl::half>(static_cast<uint16_t>(packed & 0xFFFFu)));
                     const float b1 = static_cast<float>(
                         sycl::bit_cast<sycl::half>(static_cast<uint16_t>(packed >> 16)));
                     const float a0 = Q[sq * D + 2u * pr];
                     const float a1 = Q[sq * D + 2u * pr + 1u];
                     if (!needs_ip_q) {
                       const float d0 = a0 - b0;
                       const float d1 = a1 - b1;
                       l2 += d0 * d0 + d1 * d1;
                     }
                     if (needs_ip_q) ip += a0 * b0 + a1 * b1;
                     if (needs_norms_q) {
                       nx += a0 * a0 + a1 * a1;
                       ny += b0 * b0 + b1 * b1;
                     }
                   }
                 } else {
                   for (size_t d = lane; d < D; d += sgsz) {
                     const float a = Q[sq * D + d];
                     const float b = static_cast<float>(DS16[static_cast<size_t>(id) * D + d]);
                     const float delta = a - b;
                     if (!needs_ip_q) l2 += delta * delta;
                     if (needs_ip_q) ip += a * b;
                     if (needs_norms_q) {
                       nx += a * a;
                       ny += b * b;
                     }
                   }
                 }
                 if (!needs_ip_q) {
                   l2 = sycl::reduce_over_group(subgroup, l2, sycl::plus<float>());
                   return finish_q(l2, 0.f, 0.f, 0.f);
                 }
                 ip = sycl::reduce_over_group(subgroup, ip, sycl::plus<float>());
                 if (needs_norms_q) {
                   nx = sycl::reduce_over_group(subgroup, nx, sycl::plus<float>());
                   ny = sycl::reduce_over_group(subgroup, ny, sycl::plus<float>());
                 }
                 return finish_q(0.f, ip, nx, ny);
               }
               float l2 = 0.f, ip = 0.f, nx = 0.f, ny = 0.f;
               if (reg_qf) {
                 for (int k = 0; k < 8; ++k) {
                   const size_t d = lane + static_cast<size_t>(k) * sgsz;
                   const float a = q_f[k];
                   const float b = DS[static_cast<size_t>(id) * D + d];
                   const float delta = a - b;
                   if (!needs_ip_q) l2 += delta * delta;
                   if (needs_ip_q) ip += a * b;
                   if (needs_norms_q) {
                     nx += a * a;
                     ny += b * b;
                   }
                 }
               } else {
                 for (size_t d = lane; d < D; d += sgsz) {
                   const float a = Q[sq * D + d];
                   const float b = DS[static_cast<size_t>(id) * D + d];
                   const float delta = a - b;
                   if (!needs_ip_q) l2 += delta * delta;
                   if (needs_ip_q) ip += a * b;
                   if (needs_norms_q) {
                     nx += a * a;
                     ny += b * b;
                   }
                 }
               }
               if (!needs_ip_q) {
                 l2 = sycl::reduce_over_group(subgroup, l2, sycl::plus<float>());
                 return finish_q(l2, 0.f, 0.f, 0.f);
               }
               ip = sycl::reduce_over_group(subgroup, ip, sycl::plus<float>());
               if (needs_norms_q) {
                 nx = sycl::reduce_over_group(subgroup, nx, sycl::plus<float>());
                 ny = sycl::reduce_over_group(subgroup, ny, sycl::plus<float>());
               }
               return finish_q(0.f, ip, nx, ny);
             };

             /* Commit the raw tile IN PLACE: frontier_ids[fb..] holds the raw ids (or -1)
                on entry and the kept ids, compacted and order-preserved, on exit -- the
                verdict and position of every entry come from tile_keep, and each lane pulls
                its own kept ids into registers before the compaction barrier, so the
                left-moving writes cannot clobber anything unread. Same keep rule and same
                order as the one-work-group beam_dedup path, so the walk is bit-identical.
                Requires kFrontierTileSize/sgsz <= 4 (host gates the tile at 64). */
             auto sg_commit = [&](size_t raw_count) -> int32_t {
               for (size_t r = lane; r < raw_count; r += sgsz) {
                 const int32_t id = frontier_ids[fb + r];
                 bool seen = id < 0;
                 for (size_t prev = 0; prev < r && !seen; ++prev) {
                   seen = frontier_ids[fb + prev] == id;
                 }
                 if (!seen) {
                   /* dedup mode: beam membership (revisits re-scored, then pre-filtered);
                      hash mode: full visited membership, no revisit evals at all. */
                   seen = beam_dedup ? bh_contains(id) : !sg_visit_once(id);
                 }
                 if (!seen && seen_ring != 0u) {
                   seen = seen_ids[rb + seen_slot(id)] == id;
                 }
                 tile_keep[fb + r] = seen ? uint8_t{0} : uint8_t{1};
               }
               sycl::group_barrier(subgroup);
               int32_t held_ids[4];
               int32_t held_pos[4];
               size_t held = 0;
               for (size_t r = lane; r < raw_count; r += sgsz) {
                 if (tile_keep[fb + r] == 0) continue;
                 int32_t position = 0;
                 for (size_t prev = 0; prev < r; ++prev) position += tile_keep[fb + prev];
                 held_ids[held] = frontier_ids[fb + r];
                 held_pos[held] = position;
                 ++held;
               }
               sycl::group_barrier(subgroup); /* every read is done; in-place writes begin */
               for (size_t hslot = 0; hslot < held; ++hslot) {
                 frontier_ids[fb + static_cast<size_t>(held_pos[hslot])] = held_ids[hslot];
               }
               int32_t total = 0;
               for (size_t r = 0; r < raw_count; ++r) total += tile_keep[fb + r];
               sycl::group_barrier(subgroup);
               return total;
             };

             /* Score and admit FUSED, per candidate, against a threshold frozen at tile
                entry. The one-work-group path stages scores and admits in a second pass
                because eight sub-groups score in parallel there; here the same sub-group
                does both, and with the threshold frozen the admission predicate, order and
                every counter are identical to the staged form -- while the score staging,
                the keep verdicts and one barrier disappear. The score is a reduction, so
                every lane already holds it in a register. */
             auto sg_score_admit = [&](int32_t fcount, size_t capacity, bool seeds) {
               const bool beam_full = s_count >= static_cast<int32_t>(capacity);
               const float admit_threshold =
                   beam_full ? candidate_distances[cb + static_cast<size_t>(s_worst)]
                             : std::numeric_limits<float>::max();
               for (int32_t f = 0; f < fcount; ++f) {
                 const int32_t id = frontier_ids[fb + static_cast<size_t>(f)];
                 const float distance = sg_distance(id);
                 ++e_evals;
                 if (seeds) ++e_seed_evals;
                 ++e_admits;
                 if (!(distance < admit_threshold)) {
                   /* Scored and rejected: remember it so a revisit is dropped pre-score. */
                   if (seen_ring != 0u && lane == 0) {
                     seen_ids[rb + seen_slot(id)] = id;
                   }
                   continue;
                 }
                 ++e_survivors;
                 if (s_count < static_cast<int32_t>(capacity)) {
                   if (lane == 0) {
                     candidate_ids[cb + static_cast<size_t>(s_count)] = id;
                     candidate_distances[cb + static_cast<size_t>(s_count)] = distance;
                     candidate_expanded[cb + static_cast<size_t>(s_count)] = 0;
                   }
                   if (s_count == 0 ||
                       distance > candidate_distances[cb + static_cast<size_t>(s_worst)]) {
                     s_worst = s_count;
                   }
                   ++s_count;
                   s_admit = 1;
                   if (beam_dedup) bh_insert(id);
                 } else {
                   const float worst_distance =
                       candidate_distances[cb + static_cast<size_t>(s_worst)];
                   if (distance < worst_distance) {
                     /* The evicted id leaves the beam: retire it from the membership hash
                        (and optionally remember it in the seen cache) before the slot is
                        overwritten. */
                     const int32_t evicted_id =
                         candidate_ids[cb + static_cast<size_t>(s_worst)];
                     if (seen_ring != 0u && lane == 0) {
                       seen_ids[rb + seen_slot(evicted_id)] = evicted_id;
                     }
                     if (beam_dedup) {
                       bh_remove(evicted_id);
                       bh_insert(id);
                     }
                     if (lane == 0) {
                       candidate_ids[cb + static_cast<size_t>(s_worst)] = id;
                       candidate_distances[cb + static_cast<size_t>(s_worst)] = distance;
                       candidate_expanded[cb + static_cast<size_t>(s_worst)] = 0;
                     }
                     s_admit = 1;
                     sycl::group_barrier(subgroup);
                     const uint32_t bcount = static_cast<uint32_t>(s_count);
                     float lane_distance = -std::numeric_limits<float>::max();
                     uint32_t lane_slot = std::numeric_limits<uint32_t>::max();
                     for (uint32_t i = static_cast<uint32_t>(lane); i < bcount;
                          i += static_cast<uint32_t>(sgsz)) {
                       const float d2 = candidate_distances[cb + i];
                       if (d2 > lane_distance) {
                         lane_distance = d2;
                         lane_slot = i;
                       }
                     }
                     const float worst_far =
                         sycl::reduce_over_group(subgroup, lane_distance, sycl::maximum<float>());
                     const uint32_t proposal =
                         lane_slot != std::numeric_limits<uint32_t>::max() &&
                                 lane_distance == worst_far
                             ? lane_slot
                             : std::numeric_limits<uint32_t>::max();
                     const uint32_t worst =
                         sycl::reduce_over_group(subgroup, proposal, sycl::minimum<uint32_t>());
                     s_worst = static_cast<int32_t>(worst);
                     if (beam_dedup) bh_rebuild_if_crowded();
                   }
                 }
               }
               sycl::group_barrier(subgroup); /* beam writes visible before the next phase */
             };

             /* The complete walk for the query in `sq` -- unchanged from the fixed
                assignment; only who runs it changed. */
             auto serve_query = [&]() {
             for (size_t seed_base = 0; seed_base < nseeds; seed_base += kFrontierTileSize) {
               const size_t seed_raw_count = std::min(kFrontierTileSize, nseeds - seed_base);
               for (size_t raw = lane; raw < seed_raw_count; raw += sgsz) {
                 const uint64_t mixed = static_cast<uint64_t>(seed_base + raw) * 9973u +
                                        static_cast<uint64_t>(qhash) * 13u;
                 const int32_t id = static_cast<int32_t>(mixed % N);
                 frontier_ids[fb + raw] = sg_allowed(id) ? id : -1;
               }
               sycl::group_barrier(subgroup);
               const int32_t kept = sg_commit(seed_raw_count);
               sg_score_admit(kept, IT, true);
             }

             for (int iter = 0; iter < max_iters; ++iter) {
               ++q_iters;
               int32_t npicks = 0;
               for (size_t s = 0; s < SW; ++s) {
                 float lane_distance = std::numeric_limits<float>::max();
                 uint32_t lane_slot = std::numeric_limits<uint32_t>::max();
                 const uint32_t candidate_count = static_cast<uint32_t>(s_count);
                 for (uint32_t i = static_cast<uint32_t>(lane); i < candidate_count;
                      i += static_cast<uint32_t>(sgsz)) {
                   const float distance = candidate_distances[cb + i];
                   if (!candidate_expanded[cb + i] &&
                       distance < std::numeric_limits<float>::max() &&
                       (distance < lane_distance ||
                        (distance == lane_distance && i < lane_slot))) {
                     lane_distance = distance;
                     lane_slot = i;
                   }
                 }
                 const float best_distance =
                     sycl::reduce_over_group(subgroup, lane_distance, sycl::minimum<float>());
                 const uint32_t proposal =
                     lane_slot != std::numeric_limits<uint32_t>::max() &&
                             lane_distance == best_distance
                         ? lane_slot
                         : std::numeric_limits<uint32_t>::max();
                 const uint32_t best_slot =
                     sycl::reduce_over_group(subgroup, proposal, sycl::minimum<uint32_t>());
                 if (best_slot == std::numeric_limits<uint32_t>::max()) break;
                 if (lane == static_cast<size_t>(best_slot) % sgsz) {
                   candidate_expanded[cb + best_slot] = 1;
                 }
                 if (lane == 0) {
                   picks[pb + static_cast<size_t>(npicks)] = candidate_ids[cb + best_slot];
                 }
                 ++npicks;
               }
               sycl::group_barrier(subgroup); /* picks visible to every lane */
               if (npicks == 0) break;

               s_admit = 0;
               const size_t frontier_total = static_cast<size_t>(npicks) * DEG;
               for (size_t frontier_base = 0; frontier_base < frontier_total;
                    frontier_base += kFrontierTileSize) {
                 const size_t frontier_raw_count =
                     std::min(kFrontierTileSize, frontier_total - frontier_base);
                 for (size_t raw = lane; raw < frontier_raw_count; raw += sgsz) {
                   const size_t edge = frontier_base + raw;
                   const int32_t picked = picks[pb + edge / DEG];
                   const int32_t neighbor = G[static_cast<size_t>(picked) * DEG + edge % DEG];
                   const bool ok = neighbor >= 0 && static_cast<size_t>(neighbor) < N &&
                                   sg_allowed(neighbor);
                   frontier_ids[fb + raw] = ok ? neighbor : -1;
                 }
                 sycl::group_barrier(subgroup);
                 const int32_t kept = sg_commit(frontier_raw_count);
                 sg_score_admit(kept, BEAM, false);
               }
               if (stall_patience > 0) {
                 s_stall = s_admit != 0 ? 0 : s_stall + 1;
                 if (s_stall >= stall_patience) break;
               }
             }

             sycl::group_barrier(subgroup);
             if (lane == 0) {
               const int32_t count = s_count;
               const int32_t selected_count = std::min(count, static_cast<int32_t>(KK));
               for (int32_t a = 0; a < selected_count; ++a) {
                 int32_t best = a;
                 for (int32_t b = a + 1; b < count; ++b) {
                   if (candidate_distances[cb + static_cast<size_t>(b)] <
                       candidate_distances[cb + static_cast<size_t>(best)]) {
                     best = b;
                   }
                 }
                 const float saved_distance = candidate_distances[cb + static_cast<size_t>(a)];
                 const int32_t saved_id = candidate_ids[cb + static_cast<size_t>(a)];
                 candidate_distances[cb + static_cast<size_t>(a)] =
                     candidate_distances[cb + static_cast<size_t>(best)];
                 candidate_ids[cb + static_cast<size_t>(a)] =
                     candidate_ids[cb + static_cast<size_t>(best)];
                 candidate_distances[cb + static_cast<size_t>(best)] = saved_distance;
                 candidate_ids[cb + static_cast<size_t>(best)] = saved_id;
               }
               for (size_t t = 0; t < KK; ++t) {
                 if (t < static_cast<size_t>(count)) {
                   OUTI[sq * KK + t] = candidate_ids[cb + t];
                   float distance = candidate_distances[cb + t];
                   if (met == static_cast<int>(OVVS_METRIC_INNER_PRODUCT)) distance = -distance;
                   OUTD[sq * KK + t] = distance;
                 } else {
                   OUTI[sq * KK + t] = -1;
                   OUTD[sq * KK + t] = std::numeric_limits<float>::max();
                 }
               }
             }
             e_iters_total += q_iters;
             if (q_iters > e_iters_max) e_iters_max = q_iters;
             ++e_queries;
             };

             /* Claim loop: each sub-group independently pulls the next unserved query.
                Assignment does not affect any query's result, so output and every counter
                are identical to a fixed assignment -- but a finished sub-group starts new
                work immediately instead of idling until its work-group's slowest resident
                query finishes, which is what held measured occupancy to 29% of peak. */
             for (;;) {
               int32_t claimed = 0;
               if (lane == 0) {
                 claimed = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>(*QUEUE)
                               .fetch_add(1);
               }
               claimed = sycl::group_broadcast(subgroup, claimed, 0);
               if (static_cast<size_t>(claimed) >= launch_queries) break;
               sq = query_offset + static_cast<size_t>(claimed);
               s_count = 0;
               s_worst = 0;
               s_admit = 0;
               s_stall = 0;
               q_iters = 0;
               if (lane == 0) qhash = cagra_query_hash(Q + sq * D, static_cast<int64_t>(D));
               qhash = sycl::group_broadcast(subgroup, qhash, 0);
               load_query_regs();
               if (beam_dedup) {
                 for (size_t i = lane; i < bh_capacity; i += sgsz) beam_hash[hb + i] = -1;
                 bh_tombs = 0;
               } else {
                 vbase = static_cast<size_t>(claimed) * visited_capacity;
                 for (size_t i = lane; i < visited_capacity; i += sgsz) {
                   VISITED[vbase + i] = -1;
                 }
               }
               for (size_t i = lane; i < seen_ring; i += sgsz) seen_ids[rb + i] = -1;
               sycl::group_barrier(subgroup);
               serve_query();
             }

             if (COUNTERS != nullptr && lane == 0 && e_queries != 0) {
               using Counter = sycl::atomic_ref<int64_t, sycl::memory_order::relaxed,
                                                sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>;
               Counter(COUNTERS[OVVS_CAGRA_WALK_COUNTER_QUERIES]).fetch_add(e_queries);
               Counter(COUNTERS[OVVS_CAGRA_WALK_COUNTER_EVALS]).fetch_add(e_evals);
               Counter(COUNTERS[OVVS_CAGRA_WALK_COUNTER_SEED_EVALS]).fetch_add(e_seed_evals);
               Counter(COUNTERS[OVVS_CAGRA_WALK_COUNTER_ITERATIONS]).fetch_add(e_iters_total);
               Counter(COUNTERS[OVVS_CAGRA_WALK_COUNTER_MAX_ITERATIONS]).fetch_max(e_iters_max);
               Counter(COUNTERS[OVVS_CAGRA_WALK_COUNTER_ADMITS]).fetch_add(e_admits);
               Counter(COUNTERS[OVVS_CAGRA_WALK_COUNTER_SURVIVORS]).fetch_add(e_survivors);
             }
             return;
           }

           if (!beam_dedup) {
             for (size_t i = lid; i < visited_capacity; i += work_group_size) {
               VISITED[visited_base + i] = -1;
             }
           }
           if (lid == 0) {
             state[0] = 0;
             state[3] = 0;
             state[4] = 0;
             state[5] = 0;
             state[6] = 0;
             query_hash_state[0] = cagra_query_hash(Q + qi * D, static_cast<int64_t>(D));
           }
           item.barrier(sycl::access::fence_space::global_and_local);

           int64_t c_evals = 0, c_seed_evals = 0, c_iters = 0, c_full = 0, c_admits = 0,
                   c_survivors = 0;

           auto allowed_id = [&](int32_t id) {
             return BS == nullptr || ((BS[static_cast<size_t>(id) >> 3] >> (id & 7)) & 1u) != 0;
           };
           /* Probe-and-insert against the per-query visited table. Every id handed to this
              is distinct within its tile (duplicates are removed first), so concurrent
              inserters never contend for the same key and the resulting membership set is
              exactly what inserting them one at a time would have produced -- only the
              physical slot layout can differ, which nothing observes. Work-group scope is
              enough: the table region belongs to a single work-group. */
           auto visit_once = [&](int32_t id) {
             using Slot = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                           sycl::memory_scope::work_group,
                                           sycl::access::address_space::global_space>;
             const size_t mask = visited_capacity - 1u;
             size_t slot = (static_cast<uint32_t>(id) * 2654435761u) & mask;
             for (size_t probe = 0; probe < visited_capacity; ++probe) {
               Slot entry(VISITED[visited_base + slot]);
               int32_t current = entry.load();
               if (current == id) return false;
               if (current == -1) {
                 int32_t expected = -1;
                 if (entry.compare_exchange_strong(expected, id)) return true;
                 if (expected == id) return false;
               }
               slot = (slot + 1u) & mask;
             }
             sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                              sycl::memory_scope::work_group,
                              sycl::access::address_space::local_space>(state[4])
                 .fetch_add(1);
             return false;
           };

           /* Membership scan for the beam-dedup path. The beam was last written before the
              barrier that precedes every commit, so candidate_ids[0..count) is stable here;
              the scan is per-work-item over SLM and pipelines at ~1 probe/cycle, against the
              global CAS round trip it replaces. */
           auto in_beam = [&](int32_t id) {
             const int32_t count = state[0];
             for (int32_t s = 0; s < count; ++s) {
               if (candidate_ids[static_cast<size_t>(s)] == id) return true;
             }
             return false;
           };

           /* Turns raw_ids[0..raw_count) -- ids or -1 -- into frontier_ids, keeping exactly
              the entries the serial gather kept and in the same relative order, and marking
              them visited. Replaces a chain of up to 64 dependent global probes per tile,
              run by lane 0 with everyone else parked, with three parallel local passes. */
           auto commit_frontier = [&](size_t raw_count) {
             /* Duplicate scan and insert are fused: the scan only reads raw_ids, which has
                been stable since the gather's barrier, so no synchronization is needed between
                deciding an entry is unique and claiming its slot. */
             for (size_t r = lid; r < raw_count; r += work_group_size) {
               const int32_t id = raw_ids[r];
               bool duplicate = id < 0;
               for (size_t prev = 0; prev < r && !duplicate; ++prev) {
                 duplicate = raw_ids[prev] == id;
               }
               keep_flags[r] =
                   (!duplicate && (beam_dedup ? !in_beam(id) : visit_once(id))) ? 1 : 0;
             }
             item.barrier(sycl::access::fence_space::local_space);
             for (size_t r = lid; r < raw_count; r += work_group_size) {
               if (keep_flags[r] == 0) continue;
               int32_t position = 0;
               for (size_t prev = 0; prev < r; ++prev) position += keep_flags[prev];
               frontier_ids[static_cast<size_t>(position)] = raw_ids[r];
             }
             if (lid == 0) {
               int32_t total = 0;
               for (size_t r = 0; r < raw_count; ++r) total += keep_flags[r];
               state[1] = total;
             }
             item.barrier(sycl::access::fence_space::local_space);
           };

           const bool needs_ip = met == static_cast<int>(OVVS_METRIC_INNER_PRODUCT) ||
                                 met == static_cast<int>(OVVS_METRIC_COSINE_EXPANDED);
           const bool needs_norms = met == static_cast<int>(OVVS_METRIC_COSINE_EXPANDED);

           /* Shared tail: turns the four reduced accumulators into the metric's distance.
              Written once so the fp32 and int8 lanes cannot drift, and so the int8 lane is
              provably bit-identical -- it feeds the same arithmetic the same exact values. */
           auto finish_metric = [&](float l2, float ip, float nx, float ny) {
             if (!needs_ip) {
               if (met == static_cast<int>(OVVS_METRIC_L2_SQRT_EXPANDED)) return sycl::sqrt(l2);
               return l2;
             }
             if (met == static_cast<int>(OVVS_METRIC_INNER_PRODUCT)) return -ip;
             if (needs_norms) {
               const float safe_nx = nx > 1e-12f ? nx : 1e-12f;
               const float safe_ny = ny > 1e-12f ? ny : 1e-12f;
               return 1.f - ip / (sycl::sqrt(safe_nx) * sycl::sqrt(safe_ny));
             }
             return 0.f;
           };

           /* The query row is immutable for the whole walk, yet it was re-read from global
              memory for every candidate -- ~1,300 redundant row fetches per query that the
              compiler cannot hoist on its own, because it cannot prove the row does not
              alias the visited-table writes. Each lane owns a fixed slice of the dimensions
              (the chunk/element maps in subgroup_distance are deterministic in the lane id),
              so the slice is loaded into registers once here. Specialised to the exact-fit
              shape D == 8*subgroup_size (e.g. 128 dims at sub-group 16) so the buffers are
              statically indexed and cannot be demoted to scratch; any other shape keeps the
              original loads untouched. */
           const bool packed_q8 = use_int8 && D == 8u * subgroup_size;
           uint32_t q_w0 = 0u, q_w1 = 0u; /* this lane's 8 query bytes, packed */
           int32_t q_sq = 0;              /* sum of their squares (norm contribution) */
           if (packed_q8) {
             const uint32_t* qp =
                 reinterpret_cast<const uint32_t*>(Q8 + qi * D + subgroup_lane * 8u);
             q_w0 = qp[0];
             q_w1 = qp[1];
             for (int j = 0; j < 4; ++j) {
               const int32_t a0 = static_cast<int32_t>((q_w0 >> (8 * j)) & 0xFFu);
               const int32_t a1 = static_cast<int32_t>((q_w1 >> (8 * j)) & 0xFFu);
               q_sq += a0 * a0 + a1 * a1;
             }
           }
           const bool reg_qf = !use_int8 && D == 8u * subgroup_size;
           float q_f[8];
           if (reg_qf) {
             for (int k = 0; k < 8; ++k) {
               q_f[k] = Q[qi * D + subgroup_lane + static_cast<size_t>(k) * subgroup_size];
             }
           }

           /* One sub-group owns a whole candidate, so the work-group has as many candidates
              in flight as it has sub-groups and the per-candidate reduction costs a sub-group
              butterfly instead of a work-group barrier. The old shape gave each of the 128
              work-items a single dimension and could never have more than one row outstanding,
              which is why it ran at a fraction of a percent of peak on a latency-bound gather. */
           /* TEMPORARY A/B: the pre-rewrite geometry, where the whole work-group cooperates on
              ONE candidate at a time and each work-item owns a single dimension. Kept behind a
              flag so the sub-group rewrite can finally be measured on its own -- it was only ever
              measured bundled with the int8 mirror and the parallel gather. The int8 fast path is
              mirrored here deliberately, so the toggle changes geometry and nothing else. */
           auto cooperative_distance = [&](int32_t id) {
             auto group = item.get_group();
             if (use_int8) {
               const uint8_t* qrow = Q8 + qi * D;
               const uint8_t* drow = DS8 + static_cast<size_t>(id) * D;
               int32_t l2 = 0, ip = 0, nx = 0, ny = 0;
               for (size_t d = lid; d < D; d += work_group_size) {
                 const int32_t a = static_cast<int32_t>(qrow[d]);
                 const int32_t b = static_cast<int32_t>(drow[d]);
                 if (!needs_ip) {
                   const int32_t delta = a - b;
                   l2 += delta * delta;
                 } else {
                   ip += a * b;
                   if (needs_norms) {
                     nx += a * a;
                     ny += b * b;
                   }
                 }
               }
               if (!needs_ip) {
                 l2 = sycl::reduce_over_group(group, l2, sycl::plus<int32_t>());
                 return finish_metric(static_cast<float>(l2), 0.f, 0.f, 0.f);
               }
               ip = sycl::reduce_over_group(group, ip, sycl::plus<int32_t>());
               if (needs_norms) {
                 nx = sycl::reduce_over_group(group, nx, sycl::plus<int32_t>());
                 ny = sycl::reduce_over_group(group, ny, sycl::plus<int32_t>());
               }
               return finish_metric(0.f, static_cast<float>(ip), static_cast<float>(nx),
                                    static_cast<float>(ny));
             }
             if (DS16) {
               float l2 = 0.f, ip = 0.f, nx = 0.f, ny = 0.f;
               if ((D & 1u) == 0u) {
                 const uint32_t* row32 =
                     reinterpret_cast<const uint32_t*>(DS16 + static_cast<size_t>(id) * D);
                 for (size_t pr = lid; pr < D / 2u; pr += work_group_size) {
                   const uint32_t packed = row32[pr];
                   const float b0 = static_cast<float>(
                       sycl::bit_cast<sycl::half>(static_cast<uint16_t>(packed & 0xFFFFu)));
                   const float b1 = static_cast<float>(
                       sycl::bit_cast<sycl::half>(static_cast<uint16_t>(packed >> 16)));
                   const float a0 = Q[qi * D + 2u * pr];
                   const float a1 = Q[qi * D + 2u * pr + 1u];
                   if (!needs_ip) {
                     const float d0 = a0 - b0;
                     const float d1 = a1 - b1;
                     l2 += d0 * d0 + d1 * d1;
                   }
                   if (needs_ip) ip += a0 * b0 + a1 * b1;
                   if (needs_norms) {
                     nx += a0 * a0 + a1 * a1;
                     ny += b0 * b0 + b1 * b1;
                   }
                 }
               } else {
                 for (size_t d = lid; d < D; d += work_group_size) {
                   const float a = Q[qi * D + d];
                   const float b = static_cast<float>(DS16[static_cast<size_t>(id) * D + d]);
                   const float delta = a - b;
                   if (!needs_ip) l2 += delta * delta;
                   if (needs_ip) ip += a * b;
                   if (needs_norms) {
                     nx += a * a;
                     ny += b * b;
                   }
                 }
               }
               if (!needs_ip) {
                 l2 = sycl::reduce_over_group(group, l2, sycl::plus<float>());
                 return finish_metric(l2, 0.f, 0.f, 0.f);
               }
               ip = sycl::reduce_over_group(group, ip, sycl::plus<float>());
               if (needs_norms) {
                 nx = sycl::reduce_over_group(group, nx, sycl::plus<float>());
                 ny = sycl::reduce_over_group(group, ny, sycl::plus<float>());
               }
               return finish_metric(0.f, ip, nx, ny);
             }
             float l2 = 0.f, ip = 0.f, nx = 0.f, ny = 0.f;
             for (size_t d = lid; d < D; d += work_group_size) {
               const float a = Q[qi * D + d];
               const float b = DS[static_cast<size_t>(id) * D + d];
               const float delta = a - b;
               if (!needs_ip) l2 += delta * delta;
               if (needs_ip) ip += a * b;
               if (needs_norms) {
                 nx += a * a;
                 ny += b * b;
               }
             }
             if (!needs_ip) {
               l2 = sycl::reduce_over_group(group, l2, sycl::plus<float>());
               return finish_metric(l2, 0.f, 0.f, 0.f);
             }
             ip = sycl::reduce_over_group(group, ip, sycl::plus<float>());
             if (needs_norms) {
               nx = sycl::reduce_over_group(group, nx, sycl::plus<float>());
               ny = sycl::reduce_over_group(group, ny, sycl::plus<float>());
             }
             return finish_metric(0.f, ip, nx, ny);
           };

           auto subgroup_distance = [&](int32_t id) {
             if (use_int8) {
               /* 8 contiguous bytes per lane: a 128-dim row is two cache lines for the whole
                  sub-group, against eight for the fp32 row. Exact -- see kInt8MaxDim. */
               if (packed_q8) {
                 /* Exact-fit fast path: the data row as two dword loads per lane instead of
                    eight byte loads, the query slice from the registers hoisted above, and
                    L2 via the integer identity sum(a-b)^2 = sum a^2 + sum b^2 - 2*sum ab.
                    Every term is bounded by dim*255^2 < 2^24 (see kInt8MaxDim), so the int32
                    arithmetic cannot wrap and each lane's value is bit-identical to the
                    subtractive form over the same owned bytes -- the reduction sees exactly
                    the inputs it saw before. */
                 const uint32_t* dp = reinterpret_cast<const uint32_t*>(
                     DS8 + static_cast<size_t>(id) * D + subgroup_lane * 8u);
                 const uint32_t b_w0 = dp[0];
                 const uint32_t b_w1 = dp[1];
                 int32_t qd = 0, dd = 0;
                 for (int j = 0; j < 4; ++j) {
                   const int32_t b0 = static_cast<int32_t>((b_w0 >> (8 * j)) & 0xFFu);
                   const int32_t b1 = static_cast<int32_t>((b_w1 >> (8 * j)) & 0xFFu);
                   const int32_t a0 = static_cast<int32_t>((q_w0 >> (8 * j)) & 0xFFu);
                   const int32_t a1 = static_cast<int32_t>((q_w1 >> (8 * j)) & 0xFFu);
                   qd += a0 * b0 + a1 * b1;
                   dd += b0 * b0 + b1 * b1;
                 }
                 if (!needs_ip) {
                   const int32_t l2 = sycl::reduce_over_group(subgroup, q_sq + dd - 2 * qd,
                                                              sycl::plus<int32_t>());
                   return finish_metric(static_cast<float>(l2), 0.f, 0.f, 0.f);
                 }
                 const int32_t ip = sycl::reduce_over_group(subgroup, qd, sycl::plus<int32_t>());
                 if (!needs_norms) return finish_metric(0.f, static_cast<float>(ip), 0.f, 0.f);
                 const int32_t nx = sycl::reduce_over_group(subgroup, q_sq, sycl::plus<int32_t>());
                 const int32_t ny = sycl::reduce_over_group(subgroup, dd, sycl::plus<int32_t>());
                 return finish_metric(0.f, static_cast<float>(ip), static_cast<float>(nx),
                                      static_cast<float>(ny));
               }
               const uint8_t* qrow = Q8 + qi * D;
               const uint8_t* drow = DS8 + static_cast<size_t>(id) * D;
               int32_t l2 = 0, ip = 0, nx = 0, ny = 0;
               const size_t chunks = (D + 7u) / 8u;
               for (size_t c = subgroup_lane; c < chunks; c += subgroup_size) {
                 const size_t off = c * 8u;
                 const size_t lim = (D - off) < 8u ? (D - off) : 8u;
                 for (size_t j = 0; j < lim; ++j) {
                   const int32_t a = static_cast<int32_t>(qrow[off + j]);
                   const int32_t b = static_cast<int32_t>(drow[off + j]);
                   if (!needs_ip) {
                     const int32_t delta = a - b;
                     l2 += delta * delta;
                   } else {
                     ip += a * b;
                     if (needs_norms) {
                       nx += a * a;
                       ny += b * b;
                     }
                   }
                 }
               }
               if (!needs_ip) {
                 l2 = sycl::reduce_over_group(subgroup, l2, sycl::plus<int32_t>());
                 return finish_metric(static_cast<float>(l2), 0.f, 0.f, 0.f);
               }
               ip = sycl::reduce_over_group(subgroup, ip, sycl::plus<int32_t>());
               if (needs_norms) {
                 nx = sycl::reduce_over_group(subgroup, nx, sycl::plus<int32_t>());
                 ny = sycl::reduce_over_group(subgroup, ny, sycl::plus<int32_t>());
               }
               return finish_metric(0.f, static_cast<float>(ip), static_cast<float>(nx),
                                    static_cast<float>(ny));
             }
             if (DS16) {
               float l2 = 0.f, ip = 0.f, nx = 0.f, ny = 0.f;
               if ((D & 1u) == 0u) {
                 const uint32_t* row32 =
                     reinterpret_cast<const uint32_t*>(DS16 + static_cast<size_t>(id) * D);
                 for (size_t pr = subgroup_lane; pr < D / 2u; pr += subgroup_size) {
                   const uint32_t packed = row32[pr];
                   const float b0 = static_cast<float>(
                       sycl::bit_cast<sycl::half>(static_cast<uint16_t>(packed & 0xFFFFu)));
                   const float b1 = static_cast<float>(
                       sycl::bit_cast<sycl::half>(static_cast<uint16_t>(packed >> 16)));
                   const float a0 = Q[qi * D + 2u * pr];
                   const float a1 = Q[qi * D + 2u * pr + 1u];
                   if (!needs_ip) {
                     const float d0 = a0 - b0;
                     const float d1 = a1 - b1;
                     l2 += d0 * d0 + d1 * d1;
                   }
                   if (needs_ip) ip += a0 * b0 + a1 * b1;
                   if (needs_norms) {
                     nx += a0 * a0 + a1 * a1;
                     ny += b0 * b0 + b1 * b1;
                   }
                 }
               } else {
                 for (size_t d = subgroup_lane; d < D; d += subgroup_size) {
                   const float a = Q[qi * D + d];
                   const float b = static_cast<float>(DS16[static_cast<size_t>(id) * D + d]);
                   const float delta = a - b;
                   if (!needs_ip) l2 += delta * delta;
                   if (needs_ip) ip += a * b;
                   if (needs_norms) {
                     nx += a * a;
                     ny += b * b;
                   }
                 }
               }
               if (!needs_ip) {
                 l2 = sycl::reduce_over_group(subgroup, l2, sycl::plus<float>());
                 return finish_metric(l2, 0.f, 0.f, 0.f);
               }
               ip = sycl::reduce_over_group(subgroup, ip, sycl::plus<float>());
               if (needs_norms) {
                 nx = sycl::reduce_over_group(subgroup, nx, sycl::plus<float>());
                 ny = sycl::reduce_over_group(subgroup, ny, sycl::plus<float>());
               }
               return finish_metric(0.f, ip, nx, ny);
             }
             float l2 = 0.f, ip = 0.f, nx = 0.f, ny = 0.f;
             if (reg_qf) {
               /* Same elements in the same order as the loop below (k-th owned element is
                  subgroup_lane + k*subgroup_size), so the float accumulation order -- and
                  therefore the result -- is bit-identical; only the query loads move from
                  global memory to the registers hoisted above. */
               for (int k = 0; k < 8; ++k) {
                 const size_t d = subgroup_lane + static_cast<size_t>(k) * subgroup_size;
                 const float a = q_f[k];
                 const float b = DS[static_cast<size_t>(id) * D + d];
                 const float delta = a - b;
                 if (!needs_ip) l2 += delta * delta;
                 if (needs_ip) ip += a * b;
                 if (needs_norms) {
                   nx += a * a;
                   ny += b * b;
                 }
               }
             } else {
               for (size_t d = subgroup_lane; d < D; d += subgroup_size) {
                 const float a = Q[qi * D + d];
                 const float b = DS[static_cast<size_t>(id) * D + d];
                 const float delta = a - b;
                 if (!needs_ip) l2 += delta * delta;
                 if (needs_ip) ip += a * b;
                 if (needs_norms) {
                   nx += a * a;
                   ny += b * b;
                 }
               }
             }
             if (!needs_ip) {
               l2 = sycl::reduce_over_group(subgroup, l2, sycl::plus<float>());
               return finish_metric(l2, 0.f, 0.f, 0.f);
             }
             ip = sycl::reduce_over_group(subgroup, ip, sycl::plus<float>());
             if (needs_norms) {
               nx = sycl::reduce_over_group(subgroup, nx, sycl::plus<float>());
               ny = sycl::reduce_over_group(subgroup, ny, sycl::plus<float>());
             }
             return finish_metric(0.f, ip, nx, ny);
           };

           /* Scores a frontier tile with every sub-group busy, then admits in the ORIGINAL
              tile order. The beam therefore ends up holding exactly what the serial version
              produced -- parallelism is added without reordering a single admission.

              The admissions run on sub-group 0 rather than lane 0 alone because the expensive
              part is not the insert, it is finding the new worst slot after an eviction: that
              was an O(beam) serial rescan with 127 work-items parked, and at itopk=128 it ran
              256 dependent local reads for every one of the ~2,200 candidates a query scores.
              A sub-group argmax gets the identical slot -- strict > keeps the lowest index
              within a lane, and ties across lanes resolve to the lowest index -- in
              beam/sub-group steps plus a butterfly. */
           auto score_and_admit = [&](int32_t frontier_count, size_t capacity, bool seeds) {
             /* Once the beam is full, `consider` is a strict no-op for any candidate at or
                beyond the current worst -- and the worst only ever falls as a tile is admitted,
                so a candidate that fails this test at tile entry would fail it at every later
                point too. Marking those is therefore exactly equivalent to calling `consider`
                on them, and it keeps the expensive part of the admit loop (a sub-group barrier,
                a broadcast, and on eviction an argmax) off the large majority of candidates,
                which late in a walk are all rejects.

                The threshold is read before the scoring loop, from state that has been stable
                since the previous barrier, so the verdict can be recorded by the sub-group that
                computed the score and the pass needs no barrier of its own. */
             const bool beam_full = state[0] >= static_cast<int32_t>(capacity);
             const float admit_threshold =
                 beam_full ? candidate_distances[static_cast<size_t>(state[3])]
                           : std::numeric_limits<float>::max();
             if (subgroup_geometry) {
               for (int32_t f = static_cast<int32_t>(subgroup_id); f < frontier_count;
                    f += static_cast<int32_t>(subgroup_count)) {
                 const float score = subgroup_distance(frontier_ids[static_cast<size_t>(f)]);
                 if (subgroup_lane == 0) {
                   frontier_scores[static_cast<size_t>(f)] = score;
                   keep_flags[static_cast<size_t>(f)] = score < admit_threshold ? 1 : 0;
                 }
               }
             } else {
               /* Every work-item walks the same candidate list, because the reduction below is a
                  work-group collective and must be reached by all of them. */
               for (int32_t f = 0; f < frontier_count; ++f) {
                 const float score = cooperative_distance(frontier_ids[static_cast<size_t>(f)]);
                 if (lid == 0) {
                   frontier_scores[static_cast<size_t>(f)] = score;
                   keep_flags[static_cast<size_t>(f)] = score < admit_threshold ? 1 : 0;
                 }
               }
             }
             item.barrier(sycl::access::fence_space::local_space);
             if (subgroup_id == 0) {
               for (int32_t f = 0; f < frontier_count; ++f) {
                 int32_t rescan = 0;
                 if (keep_flags[static_cast<size_t>(f)] == 0) {
                   if (subgroup_lane == 0) {
                     ++c_evals;
                     if (seeds) ++c_seed_evals;
                     ++c_admits;
                   }
                   continue;
                 }
                 if (subgroup_lane == 0) {
                   ++c_evals;
                   if (seeds) ++c_seed_evals;
                   ++c_admits;
                   ++c_survivors;
                   const int32_t id = frontier_ids[static_cast<size_t>(f)];
                   const float distance = frontier_scores[static_cast<size_t>(f)];
                   const int32_t count = state[0];
                   if (count < static_cast<int32_t>(capacity)) {
                     candidate_ids[static_cast<size_t>(count)] = id;
                     candidate_distances[static_cast<size_t>(count)] = distance;
                     candidate_expanded[static_cast<size_t>(count)] = 0;
                     if (count == 0 ||
                         distance > candidate_distances[static_cast<size_t>(state[3])]) {
                       state[3] = count;
                     }
                     state[0] = count + 1;
                     state[5] = 1;
                   } else {
                     const int32_t worst = state[3];
                     if (distance < candidate_distances[static_cast<size_t>(worst)]) {
                       candidate_ids[static_cast<size_t>(worst)] = id;
                       candidate_distances[static_cast<size_t>(worst)] = distance;
                       candidate_expanded[static_cast<size_t>(worst)] = 0;
                       state[5] = 1;
                       rescan = 1;
                     }
                   }
                 }
                 sycl::group_barrier(subgroup);
                 rescan = sycl::group_broadcast(subgroup, rescan, 0);
                 if (rescan != 0) {
                   const uint32_t count = static_cast<uint32_t>(state[0]);
                   float lane_distance = -std::numeric_limits<float>::max();
                   uint32_t lane_slot = std::numeric_limits<uint32_t>::max();
                   for (uint32_t i = static_cast<uint32_t>(subgroup_lane); i < count;
                        i += static_cast<uint32_t>(subgroup_size)) {
                     const float distance = candidate_distances[static_cast<size_t>(i)];
                     if (distance > lane_distance) {
                       lane_distance = distance;
                       lane_slot = i;
                     }
                   }
                   const float worst_distance =
                       sycl::reduce_over_group(subgroup, lane_distance, sycl::maximum<float>());
                   const uint32_t proposal =
                       lane_slot != std::numeric_limits<uint32_t>::max() &&
                               lane_distance == worst_distance
                           ? lane_slot
                           : std::numeric_limits<uint32_t>::max();
                   const uint32_t worst =
                       sycl::reduce_over_group(subgroup, proposal, sycl::minimum<uint32_t>());
                   if (subgroup_lane == 0) state[3] = static_cast<int32_t>(worst);
                   sycl::group_barrier(subgroup);
                 }
               }
             }
           };

           for (size_t seed_base = 0; seed_base < nseeds; seed_base += kFrontierTileSize) {
             const size_t seed_raw_count = std::min(kFrontierTileSize, nseeds - seed_base);
             for (size_t raw = lid; raw < seed_raw_count; raw += work_group_size) {
               const uint64_t mixed = static_cast<uint64_t>(seed_base + raw) * 9973u +
                                      static_cast<uint64_t>(query_hash_state[0]) * 13u;
               const int32_t id = static_cast<int32_t>(mixed % N);
               raw_ids[raw] = allowed_id(id) ? id : -1;
             }
             item.barrier(sycl::access::fence_space::local_space);
             commit_frontier(seed_raw_count);
             score_and_admit(state[1], IT, true);
             item.barrier(sycl::access::fence_space::local_space);
           }

           for (int iter = 0; iter < max_iters; ++iter) {
             if (lid == 0) ++c_iters;
             if (subgroup_id == 0) {
               /* Preserve the serial (distance, candidate-slot) order. A selected
                  slot is marked by the same modulo owner that rescans it, so
                  successive picks need no additional work-group barrier.

                  search_width 1 used to take a separate lane-0 branch that scanned every beam
                  slot serially with 127 work-items idle -- O(2*itopk) per iteration, and at
                  width 1 there are twice as many iterations to pay it on. This path returns the
                  identical slot for any width (strict `<` keeps the lowest index within a lane,
                  ties across lanes resolve to the lowest index), so the special case is gone. */
               int32_t npicks = 0;
               for (size_t s = 0; s < SW; ++s) {
                 float lane_distance = std::numeric_limits<float>::max();
                 uint32_t lane_slot = std::numeric_limits<uint32_t>::max();
                 const uint32_t candidate_count = static_cast<uint32_t>(state[0]);
                 for (uint32_t i = static_cast<uint32_t>(subgroup_lane); i < candidate_count;
                      i += static_cast<uint32_t>(subgroup_size)) {
                   const float distance = candidate_distances[static_cast<size_t>(i)];
                   if (!candidate_expanded[static_cast<size_t>(i)] &&
                       distance < std::numeric_limits<float>::max() &&
                       (distance < lane_distance || (distance == lane_distance && i < lane_slot))) {
                     lane_distance = distance;
                     lane_slot = i;
                   }
                 }

                 const float best_distance =
                     sycl::reduce_over_group(subgroup, lane_distance, sycl::minimum<float>());
                 const uint32_t proposal =
                     lane_slot != std::numeric_limits<uint32_t>::max() && lane_distance == best_distance
                         ? lane_slot
                         : std::numeric_limits<uint32_t>::max();
                 const uint32_t best_slot =
                     sycl::reduce_over_group(subgroup, proposal, sycl::minimum<uint32_t>());
                 if (best_slot == std::numeric_limits<uint32_t>::max()) break;
                 if (subgroup_lane == static_cast<size_t>(best_slot) % subgroup_size) {
                   candidate_expanded[static_cast<size_t>(best_slot)] = 1;
                 }
                 if (subgroup_lane == 0) {
                   picks[static_cast<size_t>(npicks)] = candidate_ids[static_cast<size_t>(best_slot)];
                 }
                 ++npicks;
               }
               if (subgroup_lane == 0) state[2] = npicks;
             }
             item.barrier(sycl::access::fence_space::local_space);
             const int32_t npicks = state[2];
             if (npicks == 0) break;

             if (lid == 0) state[5] = 0;
             const size_t frontier_total = static_cast<size_t>(npicks) * DEG;
             for (size_t frontier_base = 0; frontier_base < frontier_total;
                  frontier_base += kFrontierTileSize) {
               const size_t frontier_raw_count =
                   std::min(kFrontierTileSize, frontier_total - frontier_base);
               /* Adjacency and bitset reads are independent per edge, so the whole work-group
                  issues them at once. Done on lane 0 they formed a chain of up to 64 dependent
                  global loads per tile with 127 work-items parked -- the serial half of the
                  walk, and far more expensive than the distances it was feeding. */
               for (size_t raw = lid; raw < frontier_raw_count; raw += work_group_size) {
                 const size_t edge = frontier_base + raw;
                 const int32_t picked = picks[edge / DEG];
                 const int32_t neighbor = G[static_cast<size_t>(picked) * DEG + edge % DEG];
                 const bool ok =
                     neighbor >= 0 && static_cast<size_t>(neighbor) < N && allowed_id(neighbor);
                 raw_ids[raw] = ok ? neighbor : -1;
               }
               item.barrier(sycl::access::fence_space::local_space);
               commit_frontier(frontier_raw_count);
               score_and_admit(state[1], BEAM, false);
               item.barrier(sycl::access::fence_space::local_space);
             }
             if (stall_patience > 0) {
               if (lid == 0) state[6] = state[5] != 0 ? 0 : state[6] + 1;
               item.barrier(sycl::access::fence_space::local_space);
               if (state[6] >= stall_patience) break;
             }
           }

           if (lid == 0) {
             const int32_t count = state[0];
             const int32_t selected_count = std::min(count, static_cast<int32_t>(KK));
             for (int32_t a = 0; a < selected_count; ++a) {
               int32_t best = a;
               for (int32_t b = a + 1; b < count; ++b) {
                 if (candidate_distances[static_cast<size_t>(b)] <
                     candidate_distances[static_cast<size_t>(best)]) {
                   best = b;
                 }
               }
               const float saved_distance = candidate_distances[static_cast<size_t>(a)];
               const int32_t saved_id = candidate_ids[static_cast<size_t>(a)];
               candidate_distances[static_cast<size_t>(a)] = candidate_distances[static_cast<size_t>(best)];
               candidate_ids[static_cast<size_t>(a)] = candidate_ids[static_cast<size_t>(best)];
               candidate_distances[static_cast<size_t>(best)] = saved_distance;
               candidate_ids[static_cast<size_t>(best)] = saved_id;
             }
             for (size_t t = 0; t < KK; ++t) {
               if (t < static_cast<size_t>(count)) {
                 OUTI[qi * KK + t] = candidate_ids[t];
                 float distance = candidate_distances[t];
                 if (met == static_cast<int>(OVVS_METRIC_INNER_PRODUCT)) distance = -distance;
                 OUTD[qi * KK + t] = distance;
               } else {
                 OUTI[qi * KK + t] = -1;
                 OUTD[qi * KK + t] = std::numeric_limits<float>::max();
               }
             }
           }

           if (COUNTERS != nullptr && lid == 0) {
             /* Every counter above is incremented only on lane 0, so one atomic per
                work-group (per query) is sufficient and the hot loop stays clean. */
             using Counter =
                 sycl::atomic_ref<int64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>;
             Counter(COUNTERS[OVVS_CAGRA_WALK_COUNTER_QUERIES]).fetch_add(int64_t{1});
             Counter(COUNTERS[OVVS_CAGRA_WALK_COUNTER_EVALS]).fetch_add(c_evals);
             Counter(COUNTERS[OVVS_CAGRA_WALK_COUNTER_SEED_EVALS]).fetch_add(c_seed_evals);
             Counter(COUNTERS[OVVS_CAGRA_WALK_COUNTER_ITERATIONS]).fetch_add(c_iters);
             Counter(COUNTERS[OVVS_CAGRA_WALK_COUNTER_MAX_ITERATIONS]).fetch_max(c_iters);
             Counter(COUNTERS[OVVS_CAGRA_WALK_COUNTER_TABLE_FULL])
                 .fetch_add(c_full + static_cast<int64_t>(state[4]));
             Counter(COUNTERS[OVVS_CAGRA_WALK_COUNTER_ADMITS]).fetch_add(c_admits);
             Counter(COUNTERS[OVVS_CAGRA_WALK_COUNTER_SURVIVORS]).fetch_add(c_survivors);
           }
         };
         /* Persistent mapping: launch only as many work-groups as the device can hold
            resident (16 per Xe-core x 4 Xe-cores, probed) -- the queue keeps every
            sub-group fed until the queries run out, so surplus work-groups would only pay
            dispatch cost to find an empty queue. */
         const size_t launch_groups =
             QPW > 1 ? std::min<size_t>((launch_queries + QPW - 1u) / QPW, 64u)
                     : launch_queries;
         const sycl::nd_range<1> range(sycl::range<1>(launch_groups * work_group_size),
                                       sycl::range<1>(work_group_size));
         /* 16 is the natural Xe SIMD width and the width the geometry was tuned at: it gives
            eight candidates in flight per 128-item work-group and exactly two cache lines per
            int8 row. The body is width-agnostic, so a device without 16 simply runs whatever
            the compiler picks rather than losing the GPU path altogether. */
         if (force_sub_group_16) {
           h.parallel_for(range, [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
             walk_body(item);
           });
         } else {
           h.parallel_for(range, [=](sycl::nd_item<1> item) { walk_body(item); });
         }
       }).wait_and_throw();
    }

    if (!out_i_direct) q.memcpy(neighbors, out_i_stage, output_count * sizeof(int64_t));
    if (!out_d_direct) q.memcpy(distances, out_d_stage, output_count * sizeof(float));
    q.wait_and_throw();
    if (COUNTERS) {
      int64_t host_counters[OVVS_CAGRA_WALK_COUNTER_COUNT] = {};
      q.memcpy(host_counters, COUNTERS, sizeof(host_counters));
      q.wait_and_throw();
      std::lock_guard<std::mutex> lock(r.cagra_walk_counter_mutex);
      for (int i = 0; i < OVVS_CAGRA_WALK_COUNTER_COUNT; ++i) {
        if (i == OVVS_CAGRA_WALK_COUNTER_MAX_ITERATIONS) {
          r.cagra_walk_counters[i] = std::max(r.cagra_walk_counters[i], host_counters[i]);
        } else {
          r.cagra_walk_counters[i] += host_counters[i];
        }
      }
    }
    {
      const auto saturating_add = [](int64_t current, int64_t delta) {
        return delta > std::numeric_limits<int64_t>::max() - current
                   ? std::numeric_limits<int64_t>::max()
                   : current + delta;
      };
      std::lock_guard<std::mutex> lock(r.cagra_transfer_mutex);
      if (ds_direct && graph_direct) {
        r.cagra_direct_index_calls = saturating_add(r.cagra_direct_index_calls, 1);
      }
      if (index_upload_bytes != 0) {
        r.cagra_index_upload_calls = saturating_add(r.cagra_index_upload_calls, 1);
        r.cagra_index_upload_bytes = saturating_add(
            r.cagra_index_upload_bytes, static_cast<int64_t>(index_upload_bytes));
      }
      r.cagra_walk_calls = saturating_add(r.cagra_walk_calls, 1);
    }
    return true;
  } catch (...) {
    return false;
  }
#else
  (void)r;
  (void)dataset;
  (void)n;
  (void)dim;
  (void)metric;
  (void)graph;
  (void)degree;
  (void)queries;
  (void)nq;
  (void)k;
  (void)itopk;
  (void)search_width;
  (void)bitset;
  (void)neighbors;
  (void)distances;
  (void)dataset_f16;
  (void)dataset_u8;
  return false;
#endif
}

bool gpu_pairwise(ResourcesData& r, ovvsMetric metric, const float* x, int64_t nx, const float* y,
                  int64_t ny, int64_t dim, float* out, float metric_arg,
                  GpuWorkStats* stats) {
#if defined(OVVS_WITH_SYCL)
  if (metric == OVVS_METRIC_L2_EXPANDED || metric == OVVS_METRIC_INNER_PRODUCT ||
      metric == OVVS_METRIC_COSINE_EXPANDED) {
    return false;
  }
  try {
    auto& q = gpu_queue();
    const size_t NX = static_cast<size_t>(nx), NY = static_cast<size_t>(ny), D = static_cast<size_t>(dim);
    const bool x_usm = ovvs_usm_is_shared(x);
    const bool y_usm = ovvs_usm_is_shared(y);
    const bool o_usm = ovvs_usm_is_shared(out);
    size_t need = 0;
    if (!x_usm) need += NX * D;
    if (!y_usm) need += NY * D;
    if (!o_usm) need += NX * NY;
    float* sc = need ? usm_f(need, stats) : nullptr;
    if (need && !sc) throw std::bad_alloc();
    float* sp = sc;
    float* X = x_usm ? const_cast<float*>(x) : (std::memcpy(sp, x, NX * D * sizeof(float)), sp);
    if (!x_usm) sp += NX * D;
    float* Y = y_usm ? const_cast<float*>(y) : (std::memcpy(sp, y, NY * D * sizeof(float)), sp);
    if (!y_usm) sp += NY * D;
    float* O = o_usm ? out : sp;
    const int met = static_cast<int>(metric);
    const float p = metric_arg > 0.f ? metric_arg : 2.f;
    if (stats) stats->kernel();
    q.parallel_for(sycl::range<2>(NX, NY), [=](sycl::id<2> id) {
      const size_t i = id[0];
      const size_t j = id[1];
      if (met == 4) {
        float s = 0.f;
        for (size_t d = 0; d < D; ++d) {
          const uint32_t ax = X[i * D + d] >= 0.f;
          const uint32_t ay = Y[j * D + d] >= 0.f;
          s += static_cast<float>(ax ^ ay);
        }
        O[i * NY + j] = s;
      } else {
        float s = 0.f;
        for (size_t d = 0; d < D; ++d) s += sycl::pow(sycl::fabs(X[i * D + d] - Y[j * D + d]), p);
        O[i * NY + j] = sycl::pow(s, 1.f / p);
      }
    });
    if (stats) stats->wait();
    q.wait_and_throw();
    if (!ovvs_usm_is_shared(out)) std::memcpy(out, O, NX * NY * sizeof(float));
    (void)r;
    return true;
  } catch (...) {
    return false;
  }
#else
  (void)r;
  (void)metric;
  (void)x;
  (void)nx;
  (void)y;
  (void)ny;
  (void)dim;
  (void)out;
  (void)metric_arg;
  (void)stats;
  return false;
#endif
}

bool mkl_gesvd_components(const float* centered, int64_t n, int64_t dim, int32_t ncomp, float* components) {
#if defined(OVVS_WITH_MKL)
  if (!centered || !components || n <= 0 || dim <= 0 || ncomp <= 0) return false;
  ncomp = std::min(ncomp, static_cast<int32_t>(std::min(n, dim)));
  std::vector<float> a(centered, centered + n * dim);
  std::vector<float> s(static_cast<size_t>(std::min(n, dim)));
  std::vector<float> vt(static_cast<size_t>(std::min(n, dim) * dim));
  std::vector<float> superb(static_cast<size_t>(std::max<int64_t>(std::min(n, dim) - 1, 1)));
  const lapack_int rc = LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'N', 'S', static_cast<lapack_int>(n),
                                       static_cast<lapack_int>(dim), a.data(), static_cast<lapack_int>(dim),
                                       s.data(), nullptr, static_cast<lapack_int>(n), vt.data(),
                                       static_cast<lapack_int>(dim), superb.data());
  if (rc != 0) return false;
  for (int32_t c = 0; c < ncomp; ++c) {
    std::memcpy(components + static_cast<size_t>(c) * dim, vt.data() + static_cast<size_t>(c) * dim,
                static_cast<size_t>(dim) * sizeof(float));
  }
  return true;
#else
  (void)centered;
  (void)n;
  (void)dim;
  (void)ncomp;
  (void)components;
  return false;
#endif
}

bool mkl_syev_smallest(float* a, int64_t n, int32_t ncomp, float* embed) {
#if defined(OVVS_WITH_MKL)
  if (!a || !embed || n <= 0 || ncomp <= 0) return false;
  ncomp = std::min(ncomp, static_cast<int32_t>(n));
  std::vector<float> w(static_cast<size_t>(n));
  const lapack_int rc =
      LAPACKE_ssyev(LAPACK_ROW_MAJOR, 'V', 'U', static_cast<lapack_int>(n), a, static_cast<lapack_int>(n),
                    w.data());
  if (rc != 0) return false;
  /* Eigenvectors are rows after row-major syev? LAPACKE_ssyev overwrites A with eigenvectors as columns
     in row-major: A[i, k] is component i of eigenvector k. Smallest are first. */
  for (int64_t i = 0; i < n; ++i) {
    for (int32_t c = 0; c < ncomp; ++c) {
      embed[i * ncomp + c] = a[i * n + c];
    }
  }
  return true;
#else
  (void)a;
  (void)n;
  (void)ncomp;
  (void)embed;
  return false;
#endif
}

}  // namespace impl
}  // namespace ovvs
