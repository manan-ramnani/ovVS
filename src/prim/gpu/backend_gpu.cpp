#include "internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
static T* usm_scratch(size_t n) {
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
    scratch.ptr = sycl::malloc_shared<T>(n, q);
    scratch.capacity = scratch.ptr ? n : 0;
  }
  return scratch.ptr;
}

static float* usm_f(size_t n) { return usm_scratch<float>(n); }
static int64_t* usm_i64(size_t n) { return usm_scratch<int64_t>(n); }
static sycl::half* usm_h(size_t n) { return usm_scratch<sycl::half>(n); }
static std::int8_t* usm_i8(size_t n) { return usm_scratch<std::int8_t>(n); }

template <typename T>
class ScopedDeviceUsm {
 public:
  ScopedDeviceUsm(sycl::queue& q, size_t count) : q_(&q) {
    if (count != 0) {
      ptr_ = sycl::malloc_device<T>(count, q);
      if (!ptr_) throw std::bad_alloc();
    }
  }

  ScopedDeviceUsm(const ScopedDeviceUsm&) = delete;
  ScopedDeviceUsm& operator=(const ScopedDeviceUsm&) = delete;

  ~ScopedDeviceUsm() noexcept {
    if (!ptr_) return;
    try {
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
    q.parallel_for(n, [=](sycl::id<1> i) { dst[i] = static_cast<sycl::half>(src[i]); }).wait();
    return;
  }
  sycl::buffer<float, 1> b(const_cast<float*>(src), sycl::range<1>(n));
  q.submit([&](sycl::handler& h) {
     auto a = b.get_access<sycl::access::mode::read>(h);
     h.parallel_for(n, [=](sycl::id<1> i) { dst[i] = static_cast<sycl::half>(a[i]); });
   }).wait();
}

static void pack_i8(sycl::queue& q, const float* src, std::int8_t* dst, size_t n, float scale) {
  const float inv = 1.f / scale;
  if (ovvs_usm_is_shared(src)) {
    q.parallel_for(n, [=](sycl::id<1> i) {
       float v = src[i] * inv;
       if (v > 127.f) v = 127.f;
       if (v < -127.f) v = -127.f;
       dst[i] = static_cast<std::int8_t>(v);
     }).wait();
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
   }).wait();
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

void ovvs_usm_free(void* p) {
  if (!p) return;
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
static bool gemm_sycl_usm(const float* a, const float* b, float* c, int64_t m, int64_t n, int64_t k,
                          bool trans_b) {
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
  float* scratch = scratch_n ? usm_f(scratch_n) : nullptr;
  if (scratch_n && !scratch) return false;
  float* sp = scratch;
  float* A = a_usm ? const_cast<float*>(a) : (std::memcpy(sp, a, M * K * sizeof(float)), sp);
  if (!a_usm) sp += M * K;
  float* B = b_usm ? const_cast<float*>(b) : (std::memcpy(sp, b, bsz * sizeof(float)), sp);
  if (!b_usm) sp += bsz;
  float* C = c_usm ? c : sp;
  const bool tb = trans_b;
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
  q.wait();
  if (!c_usm) std::memcpy(c, C, M * N * sizeof(float));
  return true;
}

#if defined(OVVS_WITH_MKL)
static bool gemm_mkl_usm(const float* a, const float* b, float* c, int64_t m, int64_t n, int64_t k,
                         bool trans_b) {
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
  float* scratch = scratch_n ? usm_f(scratch_n) : nullptr;
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
  oneapi::mkl::blas::row_major::gemm(q, transA, transB, m, n, k, 1.f, A, lda, B, ldb, 0.f, C, ldc);
  q.wait();
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
  q.wait();
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
  q.wait();
  if (!ovvs_usm_is_shared(c)) std::memcpy(c, C, M * N * sizeof(float));
  return true;
}
#endif

enum class GpuGemmKind { Unset, Mkl, Sycl };
static GpuGemmKind g_gpu_gemm = GpuGemmKind::Unset;

static void pick_gpu_gemm() {
  if (g_gpu_gemm != GpuGemmKind::Unset) return;
  g_gpu_gemm = GpuGemmKind::Sycl;
#if defined(OVVS_WITH_MKL)
  try {
    const int64_t m = 64, n = 128, k = 32;
    std::vector<float> A(static_cast<size_t>(m * k), 0.1f), B(static_cast<size_t>(n * k), 0.2f);
    std::vector<float> Cm(static_cast<size_t>(m * n), 0.f), Cs(static_cast<size_t>(m * n), 0.f);
    const auto t0 = std::chrono::steady_clock::now();
    const bool mok = gemm_mkl_usm(A.data(), B.data(), Cm.data(), m, n, k, true);
    const auto t1 = std::chrono::steady_clock::now();
    const bool sok = gemm_sycl_usm(A.data(), B.data(), Cs.data(), m, n, k, true);
    const auto t2 = std::chrono::steady_clock::now();
    if (mok && sok) {
      bool close = true;
      for (size_t i = 0; i < Cm.size(); ++i) {
        if (std::fabs(Cm[i] - Cs[i]) > 2e-2f) {
          close = false;
          break;
        }
      }
      const double mkl_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      const double sycl_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
      g_gpu_gemm = (close && mkl_ms <= sycl_ms) ? GpuGemmKind::Mkl : GpuGemmKind::Sycl;
      if (!close) g_gpu_gemm = GpuGemmKind::Sycl;
    } else if (mok) {
      g_gpu_gemm = GpuGemmKind::Mkl;
    }
    (void)sok;
  } catch (...) {
    g_gpu_gemm = GpuGemmKind::Sycl;
  }
#endif
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
  q.wait();
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
              int64_t k, bool trans_b) {
#if defined(OVVS_WITH_SYCL)
  try {
    pick_gpu_gemm();
#if defined(OVVS_WITH_MKL)
    if (g_gpu_gemm == GpuGemmKind::Mkl) {
      if (gemm_mkl_usm(a, b, c, m, n, k, trans_b)) {
        r.last_compute_dtype = OVVS_DTYPE_F32;
        return true;
      }
    }
#endif
    if (gemm_sycl_usm(a, b, c, m, n, k, trans_b)) {
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
              int64_t* indices, float* values, bool largest) {
#if defined(OVVS_WITH_SYCL)
  /* Subgroup-friendly per-row partial select on USM shared; correctness first. */
  try {
    auto& q = gpu_queue();
    const size_t R = static_cast<size_t>(rows);
    const size_t C = static_cast<size_t>(cols);
    const size_t KK = static_cast<size_t>(std::min(k, cols));
    const bool s_usm = ovvs_usm_is_shared(scores);
    float* S = s_usm ? const_cast<float*>(scores) : usm_f(R * C + R * KK);
    int64_t* I = usm_i64(R * KK);
    if (!S || !I) throw std::bad_alloc();
    float* V = s_usm ? usm_f(R * KK) : (S + R * C);
    if (!s_usm) std::memcpy(S, scores, R * C * sizeof(float));
    if (!V) throw std::bad_alloc();
    const bool lg = largest;
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
    q.wait();
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
                                 int32_t* packed_positions, int32_t* counts) {
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

    ScopedDeviceUsm<IvfPqScanTask> device_tasks(q, static_cast<size_t>(task_count));
    ScopedDeviceUsm<float> device_luts(q, static_cast<size_t>(lut_elements));
    ScopedDeviceUsm<uint8_t> device_allow(q, static_cast<size_t>(allow_bitset_bytes));
    ScopedDeviceUsm<int32_t> device_query_offsets(q, static_cast<size_t>(nq) + 1);
    ScopedDeviceUsm<float> list_scores(q,
                                       static_cast<size_t>(task_count) *
                                           static_cast<size_t>(krefine));
    ScopedDeviceUsm<int32_t> list_ordinals(q,
                                           static_cast<size_t>(task_count) *
                                               static_cast<size_t>(krefine));
    ScopedDeviceUsm<int32_t> list_positions(q,
                                            static_cast<size_t>(task_count) *
                                                static_cast<size_t>(krefine));
    ScopedDeviceUsm<int32_t> device_positions(q, output_slots);
    ScopedDeviceUsm<int32_t> device_counts(q, static_cast<size_t>(nq));
    ScopedDeviceUsm<int32_t> invalid_input(q, 1);

    std::vector<sycl::event> input_events;
    input_events.reserve(5);
    input_events.push_back(q.memcpy(device_tasks.get(), tasks,
                                    static_cast<size_t>(task_count) *
                                        sizeof(IvfPqScanTask)));
    input_events.push_back(q.memcpy(device_luts.get(), luts,
                                    static_cast<size_t>(lut_elements) * sizeof(float)));
    input_events.push_back(q.memcpy(device_query_offsets.get(), query_task_offsets.data(),
                                    (static_cast<size_t>(nq) + 1) * sizeof(int32_t)));
    input_events.push_back(q.memset(invalid_input.get(), 0, sizeof(int32_t)));
    if (allow_bitset) {
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
    sycl::event positions_copy = q.submit([&](sycl::handler& h) {
      h.depends_on(merge_event);
      h.memcpy(staged_positions.data(), output_positions, output_slots * sizeof(int32_t));
    });
    sycl::event counts_copy = q.submit([&](sycl::handler& h) {
      h.depends_on(merge_event);
      h.memcpy(staged_counts.data(), output_counts,
               static_cast<size_t>(nq) * sizeof(int32_t));
    });
    sycl::event invalid_copy = q.submit([&](sycl::handler& h) {
      h.depends_on(merge_event);
      h.memcpy(&invalid_value, invalid, sizeof(int32_t));
    });
    positions_copy.wait_and_throw();
    counts_copy.wait_and_throw();
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
  return OVVS_STATUS_DEVICE_UNAVAILABLE;
#endif
}

bool gpu_gather_rows(ResourcesData& r, const float* src, int64_t src_rows, int64_t dim,
                     const int64_t* idx, int64_t nidx, float* out) {
#if defined(OVVS_WITH_SYCL)
  try {
    auto& q = gpu_queue();
    const size_t D = static_cast<size_t>(dim);
    const size_t N = static_cast<size_t>(nidx);
    const size_t SR = static_cast<size_t>(src_rows);
    const bool src_usm = ovvs_usm_is_shared(src);
    float* S = src_usm ? const_cast<float*>(src) : usm_f(SR * D + N * D);
    int64_t* I = usm_i64(N);
    if (!S || !I) throw std::bad_alloc();
    float* O = src_usm ? usm_f(N * D) : (S + SR * D);
    if (!O) throw std::bad_alloc();
    if (!src_usm) std::memcpy(S, src, SR * D * sizeof(float));
    if (ovvs_usm_is_shared(idx)) {
      I = const_cast<int64_t*>(idx);
    } else {
      std::memcpy(I, idx, N * sizeof(int64_t));
    }
    q.parallel_for(sycl::range<2>(N, D), [=](sycl::id<2> id) {
      const size_t i = id[0];
      const size_t j = id[1];
      const int64_t row = I[i];
      O[i * D + j] = (row >= 0 && static_cast<size_t>(row) < SR) ? S[static_cast<size_t>(row) * D + j] : 0.f;
    });
    q.wait();
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

ovvsStatus gpu_nndescent_build(ResourcesData& r, const float* dataset, int64_t n,
                               int64_t dim, ovvsMetric metric, int32_t degree,
                               int32_t iters, int32_t* graph) {
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
    if (!checked_product(convergence_hash_capacity, sizeof(int32_t), convergence_local_bytes) ||
        !checked_product(max_bi_samples, max_bi_samples, matrix_cells) ||
        !checked_product(matrix_cells, 2u * sizeof(float), join_local_bytes) ||
        max_bi_samples >
            (std::numeric_limits<size_t>::max() - join_local_bytes - 2u * sizeof(int32_t)) /
                (2u * sizeof(uint32_t))) {
      return OVVS_STATUS_SHAPE_MISMATCH;
    }
    join_local_bytes += max_bi_samples * 2u * sizeof(uint32_t) + 2u * sizeof(int32_t);
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

    ScopedDeviceUsm<float> dataset_copy(q, dataset_direct ? 0u : dataset_count);
    ScopedDeviceUsm<uint32_t> ids_a(q, graph_count);
    ScopedDeviceUsm<uint32_t> ids_b(q, graph_count);
    ScopedDeviceUsm<float> distances_a(q, graph_count);
    ScopedDeviceUsm<float> distances_b(q, graph_count);
    ScopedDeviceUsm<uint32_t> reverse_new(q, reverse_count);
    ScopedDeviceUsm<uint32_t> reverse_old(q, reverse_count);
    ScopedDeviceUsm<int32_t> inbox_links(q, link_count);
    ScopedDeviceUsm<uint32_t> proposal_ids(q, proposal_inbox_count);
    ScopedDeviceUsm<float> proposal_distances(q, proposal_inbox_count);
    ScopedDeviceUsm<int32_t> heads(q, N);
    ScopedDeviceUsm<uint32_t> active_targets(q, N);
    ScopedDeviceUsm<int32_t> scalars(q, 2u);
    uint32_t* const reverse_new_ptr = reverse_new.get();
    uint32_t* const reverse_old_ptr = reverse_old.get();
    int32_t* const inbox_links_ptr = inbox_links.get();
    uint32_t* const proposal_ids_ptr = proposal_ids.get();
    float* const proposal_distances_ptr = proposal_distances.get();
    int32_t* const heads_ptr = heads.get();
    uint32_t* const active_targets_ptr = active_targets.get();
    int32_t* const scalars_ptr = scalars.get();
    const float* DS = dataset_direct ? dataset : dataset_copy.get();
    if (!dataset_direct) q.memcpy(dataset_copy.get(), dataset, dataset_bytes);
    q.wait_and_throw();

    q.fill(scalars_ptr + 1, 0, 1u);
    q.wait_and_throw();
    q.parallel_for(sycl::range<1>(dataset_count), [=](sycl::id<1> index) {
      if (!sycl::isfinite(DS[index])) {
        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            invalid(scalars_ptr[1]);
        invalid.store(1);
      }
    });
    q.wait_and_throw();
    int32_t invalid_numeric = 0;
    q.memcpy(&invalid_numeric, scalars_ptr + 1, sizeof(invalid_numeric));
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
    q.wait_and_throw();
    q.memcpy(&invalid_numeric, scalars_ptr + 1, sizeof(invalid_numeric));
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
      q.wait_and_throw();

      q.fill(reverse_new_ptr, kInvalidId, reverse_count);
      q.fill(reverse_old_ptr, kInvalidId, reverse_count);
      q.wait_and_throw();

      const auto build_reverse = [&](bool select_new, uint32_t* reverse_graph) -> ovvsStatus {
        for (size_t source_offset = 0; source_offset < N; source_offset += reverse_chunk) {
          const size_t source_count = std::min(reverse_chunk, N - source_offset);
          q.fill(heads_ptr, -1, N);
          q.fill(scalars_ptr, 0, 1u);
          q.wait_and_throw();

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
          q.wait_and_throw();

          int32_t active_count = 0;
          q.memcpy(&active_count, scalars_ptr, sizeof(active_count));
          q.wait_and_throw();
          if (active_count < 0 || static_cast<size_t>(active_count) > N) {
            return OVVS_STATUS_ERROR;
          }

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
        q.fill(heads_ptr, -1, N);
        q.fill(scalars_ptr, 0, 1u);
        q.wait_and_throw();

        q.submit([&](sycl::handler& h) {
           sycl::local_accessor<uint32_t, 1> new_ids(sycl::range<1>(max_bi_samples), h);
           sycl::local_accessor<uint32_t, 1> old_ids(sycl::range<1>(max_bi_samples), h);
           sycl::local_accessor<float, 1> new_new_distances(sycl::range<1>(matrix_cells), h);
           sycl::local_accessor<float, 1> new_old_distances(sycl::range<1>(matrix_cells), h);
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
               auto nearer = [](float lhs_distance, uint32_t lhs_id, float rhs_distance,
                                uint32_t rhs_id) {
                 return lhs_distance < rhs_distance ||
                        (lhs_distance == rhs_distance && lhs_id < rhs_id);
               };

               for (size_t i = 0; i < new_count; ++i) {
                 uint32_t best_id = kInvalidId;
                 float best_distance = std::numeric_limits<float>::max();
                 for (size_t j = 0; j < new_count; ++j) {
                   if (i == j) continue;
                   const float distance = new_new_distances[i * new_count + j];
                   if (best_id == kInvalidId ||
                       nearer(distance, new_ids[j], best_distance, best_id)) {
                     best_id = new_ids[j];
                     best_distance = distance;
                   }
                 }
                 if (best_id != kInvalidId) {
                   emit_reciprocal(new_ids[i], best_id, best_distance);
                 }

                 best_id = kInvalidId;
                 best_distance = std::numeric_limits<float>::max();
                 for (size_t j = 0; j < old_count; ++j) {
                   const float distance = new_old_distances[i * old_count + j];
                   if (best_id == kInvalidId ||
                       nearer(distance, old_ids[j], best_distance, best_id)) {
                     best_id = old_ids[j];
                     best_distance = distance;
                   }
                 }
                 if (best_id != kInvalidId) {
                   emit_reciprocal(new_ids[i], best_id, best_distance);
                 }
               }

               for (size_t j = 0; j < old_count; ++j) {
                 uint32_t best_id = kInvalidId;
                 float best_distance = std::numeric_limits<float>::max();
                 for (size_t i = 0; i < new_count; ++i) {
                   const float distance = new_old_distances[i * old_count + j];
                   if (best_id == kInvalidId ||
                       nearer(distance, new_ids[i], best_distance, best_id)) {
                     best_id = new_ids[i];
                     best_distance = distance;
                   }
                 }
                 if (best_id != kInvalidId) {
                   emit_reciprocal(old_ids[j], best_id, best_distance);
                 }
               }
             }
           });
         });
        q.wait_and_throw();

        int32_t scalar_values[2] = {0, 0};
        q.memcpy(scalar_values, scalars_ptr, sizeof(scalar_values));
        q.wait_and_throw();
        if (scalar_values[1] != 0) return OVVS_STATUS_INVALID_ARGUMENT;
        const int32_t active_count = scalar_values[0];
        if (active_count < 0 || static_cast<size_t>(active_count) > N) {
          return OVVS_STATUS_ERROR;
        }

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
        q.wait_and_throw();
      }

      for (size_t vertex_offset = 0; vertex_offset < N;
           vertex_offset += kVertexGroupsPerLaunch) {
        const size_t launch_vertices = std::min(kVertexGroupsPerLaunch, N - vertex_offset);
        size_t global_size = 0;
        if (!checked_product(launch_vertices, work_group_size, global_size)) {
          return OVVS_STATUS_SHAPE_MISMATCH;
        }
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
      q.wait_and_throw();
      q.memcpy(changed_per_row.data(), heads_ptr, N * sizeof(int32_t));
      q.memcpy(pending_new_per_row.data(), active_targets_ptr, N * sizeof(uint32_t));
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

    q.parallel_for(sycl::range<1>(graph_count), [=](sycl::id<1> index) {
      next_ids[index] = current_ids[index] & kIdMask;
    });
    q.wait_and_throw();
    q.memcpy(graph, next_ids, graph_plane_bytes);
    q.wait_and_throw();

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
  return OVVS_STATUS_DEVICE_UNAVAILABLE;
#endif
}

bool gpu_cagra_walk(ResourcesData& r, const float* dataset, int64_t n, int64_t dim, ovvsMetric metric,
                    const int32_t* graph, int32_t degree, const float* queries, int64_t nq, int64_t k,
                    int32_t itopk, int32_t search_width, const uint8_t* bitset, int64_t* neighbors,
                    float* distances) {
#if defined(OVVS_WITH_SYCL)
  if (!gpu_available()) return false;
  if (!dataset || !graph || !queries || !neighbors || !distances) return false;
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
    const int max_iters = std::max(24, itopk * 6);
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

    size_t local_bytes = 0;
    if (!checked_product(BEAM, sizeof(int32_t) + sizeof(float) + sizeof(uint8_t), local_bytes)) return false;
    size_t pick_bytes = 0;
    if (!checked_product(SW, sizeof(int32_t), pick_bytes) ||
        local_bytes > std::numeric_limits<size_t>::max() - pick_bytes - 64u) {
      return false;
    }
    local_bytes += pick_bytes + 64u;
    const size_t local_mem = device.get_info<sycl::info::device::local_mem_size>();
    if (local_bytes > local_mem) return false;

    if (IT > std::numeric_limits<size_t>::max() / 16u || SW > std::numeric_limits<size_t>::max() / 32u) {
      return false;
    }
    size_t nseeds = std::max(IT * 16u, SW * 32u);
    nseeds = std::max<size_t>(nseeds, 512u);
    nseeds = std::min(nseeds, N);

    if (SW > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(max_iters)) return false;
    const uint64_t expansion_budget = static_cast<uint64_t>(max_iters) * static_cast<uint64_t>(SW);
    if (DEG != 0 && expansion_budget > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(DEG)) {
      return false;
    }
    const uint64_t edge_budget = expansion_budget * static_cast<uint64_t>(DEG);
    const uint64_t visit_budget = std::min<uint64_t>(N, static_cast<uint64_t>(nseeds) + edge_budget);
    if (visit_budget > std::numeric_limits<size_t>::max() / 2u) return false;
    const size_t visited_capacity =
        next_power_of_two(std::max<size_t>(2u, static_cast<size_t>(visit_budget) * 2u));
    if (visited_capacity == 0 || visited_capacity > kMaxVisitedBytesPerQuery / sizeof(int32_t)) return false;

    const bool ds_direct = gpu_pointer_accessible(q, dataset);
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

    ScopedDeviceUsm<float> ds_copy(q, ds_direct ? 0u : ds_count);
    ScopedDeviceUsm<int32_t> graph_copy(q, graph_direct ? 0u : graph_count);
    ScopedDeviceUsm<float> query_copy(q, query_direct ? 0u : query_count);
    ScopedDeviceUsm<int64_t> out_i_copy(q, out_i_direct ? 0u : output_count);
    ScopedDeviceUsm<float> out_d_copy(q, out_d_direct ? 0u : output_count);
    ScopedDeviceUsm<uint8_t> bitset_copy(q, !bitset || bitset_direct ? 0u : bitset_bytes);

    const float* DS = ds_direct ? dataset : ds_copy.get();
    const int32_t* G = graph_direct ? graph : graph_copy.get();
    const float* Q = query_direct ? queries : query_copy.get();
    int64_t* OUTI = out_i_direct ? neighbors : out_i_copy.get();
    float* OUTD = out_d_direct ? distances : out_d_copy.get();
    const uint8_t* BS = !bitset ? nullptr : (bitset_direct ? bitset : bitset_copy.get());

    if (!ds_direct) q.memcpy(ds_copy.get(), dataset, dataset_bytes);
    if (!graph_direct) q.memcpy(graph_copy.get(), graph, graph_bytes);
    if (!query_direct) q.memcpy(query_copy.get(), queries, query_count * sizeof(float));
    if (bitset && !bitset_direct) q.memcpy(bitset_copy.get(), bitset, bitset_bytes);
    q.wait_and_throw();

    const size_t visited_bytes_per_query = visited_capacity * sizeof(int32_t);
    size_t queries_per_launch = std::max<size_t>(1u, kVisitedAllocationTarget / visited_bytes_per_query);
    queries_per_launch = std::min(queries_per_launch, NQ);
    size_t visited_count = 0;
    if (!checked_product(queries_per_launch, visited_capacity, visited_count)) return false;
    const size_t max_alloc = device.get_info<sycl::info::device::max_mem_alloc_size>();
    if (visited_count > max_alloc / sizeof(int32_t)) return false;
    ScopedDeviceUsm<int32_t> visited(q, visited_count);
    int32_t* VISITED = visited.get();

    for (size_t query_offset = 0; query_offset < NQ; query_offset += queries_per_launch) {
      const size_t launch_queries = std::min(queries_per_launch, NQ - query_offset);
      q.submit([&](sycl::handler& h) {
         sycl::local_accessor<int32_t, 1> candidate_ids(sycl::range<1>(BEAM), h);
         sycl::local_accessor<float, 1> candidate_distances(sycl::range<1>(BEAM), h);
         sycl::local_accessor<uint8_t, 1> candidate_expanded(sycl::range<1>(BEAM), h);
         sycl::local_accessor<int32_t, 1> picks(sycl::range<1>(SW), h);
         sycl::local_accessor<int32_t, 1> state(sycl::range<1>(4), h);
         sycl::local_accessor<uint32_t, 1> query_hash_state(sycl::range<1>(1), h);
         h.parallel_for(sycl::nd_range<1>(sycl::range<1>(launch_queries * work_group_size),
                                         sycl::range<1>(work_group_size)),
                        [=](sycl::nd_item<1> item) {
           const size_t lid = item.get_local_linear_id();
           const size_t qi = query_offset + item.get_group_linear_id();
           const size_t visited_base = item.get_group_linear_id() * visited_capacity;
           auto group = item.get_group();

           for (size_t i = lid; i < visited_capacity; i += work_group_size) {
             VISITED[visited_base + i] = -1;
           }
           if (lid == 0) {
             state[0] = 0;
             query_hash_state[0] = cagra_query_hash(Q + qi * D, static_cast<int64_t>(D));
           }
           item.barrier(sycl::access::fence_space::global_and_local);

           auto allowed_id = [&](int32_t id) {
             return BS == nullptr || ((BS[static_cast<size_t>(id) >> 3] >> (id & 7)) & 1u) != 0;
           };
           auto visit_once = [&](int32_t id) {
             const size_t mask = visited_capacity - 1u;
             size_t slot = (static_cast<uint32_t>(id) * 2654435761u) & mask;
             for (size_t probe = 0; probe < visited_capacity; ++probe) {
               int32_t& entry = VISITED[visited_base + slot];
               if (entry == id) return false;
               if (entry == -1) {
                 entry = id;
                 return true;
               }
               slot = (slot + 1u) & mask;
             }
             return false;
           };
           auto cooperative_distance = [&](int32_t id) {
             float l2 = 0.f, ip = 0.f, nx = 0.f, ny = 0.f;
             const bool needs_ip = met == static_cast<int>(OVVS_METRIC_INNER_PRODUCT) ||
                                   met == static_cast<int>(OVVS_METRIC_COSINE_EXPANDED);
             for (size_t d = lid; d < D; d += work_group_size) {
               const float a = Q[qi * D + d];
               const float b = DS[static_cast<size_t>(id) * D + d];
               const float delta = a - b;
               if (!needs_ip) l2 += delta * delta;
               if (needs_ip) ip += a * b;
               if (met == static_cast<int>(OVVS_METRIC_COSINE_EXPANDED)) {
                 nx += a * a;
                 ny += b * b;
               }
             }
             if (!needs_ip) {
               l2 = sycl::reduce_over_group(group, l2, sycl::plus<float>());
               if (met == static_cast<int>(OVVS_METRIC_L2_SQRT_EXPANDED)) return sycl::sqrt(l2);
               return l2;
             }
             ip = sycl::reduce_over_group(group, ip, sycl::plus<float>());
             if (met == static_cast<int>(OVVS_METRIC_INNER_PRODUCT)) return -ip;
             if (met == static_cast<int>(OVVS_METRIC_COSINE_EXPANDED)) {
               nx = sycl::reduce_over_group(group, nx, sycl::plus<float>());
               ny = sycl::reduce_over_group(group, ny, sycl::plus<float>());
               const float safe_nx = nx > 1e-12f ? nx : 1e-12f;
               const float safe_ny = ny > 1e-12f ? ny : 1e-12f;
               return 1.f - ip / (sycl::sqrt(safe_nx) * sycl::sqrt(safe_ny));
             }
             return 0.f;
           };
           auto consider = [&](int32_t id, float distance, size_t capacity) {
             int32_t count = state[0];
             if (count < static_cast<int32_t>(capacity)) {
               candidate_ids[static_cast<size_t>(count)] = id;
               candidate_distances[static_cast<size_t>(count)] = distance;
               candidate_expanded[static_cast<size_t>(count)] = 0;
               state[0] = count + 1;
               return;
             }
             int32_t worst = 0;
             for (int32_t i = 1; i < count; ++i) {
               if (candidate_distances[static_cast<size_t>(i)] >
                   candidate_distances[static_cast<size_t>(worst)]) {
                 worst = i;
               }
             }
             if (distance < candidate_distances[static_cast<size_t>(worst)]) {
               candidate_ids[static_cast<size_t>(worst)] = id;
               candidate_distances[static_cast<size_t>(worst)] = distance;
               candidate_expanded[static_cast<size_t>(worst)] = 0;
             }
           };

           for (size_t seed = 0; seed < nseeds; ++seed) {
             if (lid == 0) {
               const uint64_t mixed =
                   static_cast<uint64_t>(seed) * 9973u + static_cast<uint64_t>(query_hash_state[0]) * 13u;
               const int32_t id = static_cast<int32_t>(mixed % N);
               state[1] = allowed_id(id) && visit_once(id) ? id : -1;
             }
             item.barrier(sycl::access::fence_space::local_space);
             const int32_t id = state[1];
             const float score = id >= 0 ? cooperative_distance(id) : 0.f;
             if (lid == 0 && id >= 0) consider(id, score, IT);
             item.barrier(sycl::access::fence_space::local_space);
           }

           for (int iter = 0; iter < max_iters; ++iter) {
             if (lid == 0) {
               int32_t npicks = 0;
               for (size_t s = 0; s < SW; ++s) {
                 int32_t best = -1;
                 float best_distance = std::numeric_limits<float>::max();
                 for (int32_t i = 0; i < state[0]; ++i) {
                   if (!candidate_expanded[static_cast<size_t>(i)] &&
                       candidate_distances[static_cast<size_t>(i)] < best_distance) {
                     best = i;
                     best_distance = candidate_distances[static_cast<size_t>(i)];
                   }
                 }
                 if (best < 0) break;
                 candidate_expanded[static_cast<size_t>(best)] = 1;
                 picks[static_cast<size_t>(npicks++)] = candidate_ids[static_cast<size_t>(best)];
               }
               state[2] = npicks;
             }
             item.barrier(sycl::access::fence_space::local_space);
             const int32_t npicks = state[2];
             if (npicks == 0) break;

             for (int32_t pi = 0; pi < npicks; ++pi) {
               const int32_t picked = picks[static_cast<size_t>(pi)];
               for (size_t edge = 0; edge < DEG; ++edge) {
                 if (lid == 0) {
                   const int32_t neighbor = G[static_cast<size_t>(picked) * DEG + edge];
                   state[1] = neighbor >= 0 && static_cast<size_t>(neighbor) < N && allowed_id(neighbor) &&
                                      visit_once(neighbor)
                                  ? neighbor
                                  : -1;
                 }
                 item.barrier(sycl::access::fence_space::local_space);
                 const int32_t neighbor = state[1];
                 const float score = neighbor >= 0 ? cooperative_distance(neighbor) : 0.f;
                 if (lid == 0 && neighbor >= 0) consider(neighbor, score, BEAM);
                 item.barrier(sycl::access::fence_space::local_space);
               }
             }
           }

           if (lid == 0) {
             const int32_t count = state[0];
             for (int32_t a = 0; a < count; ++a) {
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
         });
       }).wait_and_throw();
    }

    if (!out_i_direct) q.memcpy(neighbors, out_i_copy.get(), output_count * sizeof(int64_t));
    if (!out_d_direct) q.memcpy(distances, out_d_copy.get(), output_count * sizeof(float));
    q.wait_and_throw();
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
  return false;
#endif
}

bool gpu_pairwise(ResourcesData& r, ovvsMetric metric, const float* x, int64_t nx, const float* y,
                  int64_t ny, int64_t dim, float* out, float metric_arg) {
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
    float* sc = need ? usm_f(need) : nullptr;
    if (need && !sc) throw std::bad_alloc();
    float* sp = sc;
    float* X = x_usm ? const_cast<float*>(x) : (std::memcpy(sp, x, NX * D * sizeof(float)), sp);
    if (!x_usm) sp += NX * D;
    float* Y = y_usm ? const_cast<float*>(y) : (std::memcpy(sp, y, NY * D * sizeof(float)), sp);
    if (!y_usm) sp += NY * D;
    float* O = o_usm ? out : sp;
    const int met = static_cast<int>(metric);
    const float p = metric_arg > 0.f ? metric_arg : 2.f;
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
    q.wait();
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
