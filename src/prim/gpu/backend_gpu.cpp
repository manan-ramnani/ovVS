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
static T* usm_scratch(size_t n) {
  static T* p = nullptr;
  static size_t cap = 0;
  auto& q = gpu_queue();
  if (n > cap) {
    if (p) sycl::free(p, q);
    p = sycl::malloc_shared<T>(n, q);
    cap = p ? n : 0;
  }
  return p;
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
  return ov_gather_rows(r, "GPU", src, src_rows, dim, idx, nidx, out);
}

int32_t sycl_enabled() {
#if defined(OVVS_WITH_SYCL)
  return 1;
#else
  return 0;
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

    if (!ds_direct) q.memcpy(ds_copy.get(), dataset, ds_count * sizeof(float));
    if (!graph_direct) q.memcpy(graph_copy.get(), graph, graph_count * sizeof(int32_t));
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
           if (lid == 0) state[0] = 0;
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
               const uint64_t mixed = static_cast<uint64_t>(seed) * 9973u + static_cast<uint64_t>(qi) * 13u;
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
    (void)r;
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
