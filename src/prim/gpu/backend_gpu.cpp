#include "internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <new>

#if defined(IOVS_WITH_SYCL)
#include <sycl/sycl.hpp>
#if defined(IOVS_WITH_MKL)
#include <oneapi/mkl.hpp>
#endif
#endif

#if defined(IOVS_WITH_MKL)
#include <mkl_lapacke.h>
#endif

namespace iovs {
namespace impl {

#if defined(IOVS_WITH_SYCL)
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
#endif

void* iovs_usm_malloc(size_t bytes) {
  if (bytes == 0) bytes = 1;
#if defined(IOVS_WITH_SYCL)
  try {
    void* p = sycl::malloc_shared(bytes, gpu_queue());
    if (p) return p;
  } catch (...) {
  }
#endif
  return std::malloc(bytes);
}

void iovs_usm_free(void* p) {
  if (!p) return;
#if defined(IOVS_WITH_SYCL)
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

bool iovs_usm_is_shared(const void* p) {
  if (!p) return false;
#if defined(IOVS_WITH_SYCL)
  try {
    auto k = sycl::get_pointer_type(const_cast<void*>(p), gpu_queue().get_context());
    return k == sycl::usm::alloc::shared || k == sycl::usm::alloc::host;
  } catch (...) {
  }
#endif
  return false;
}



bool gpu_available() {
#if defined(IOVS_WITH_SYCL)
  try {
    auto d = gpu_queue().get_device();
    if (d.is_gpu()) return true;
  } catch (...) {
  }
#endif
  return ov_device_available("GPU");
}

bool gpu_vector_add(const float* a, const float* b, float* c, int64_t n) {
#if defined(IOVS_WITH_SYCL)
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

#if defined(IOVS_WITH_SYCL)
static bool gemm_sycl_usm(const float* a, const float* b, float* c, int64_t m, int64_t n, int64_t k,
                          bool trans_b) {
  auto& q = gpu_queue();
  const size_t M = static_cast<size_t>(m), N = static_cast<size_t>(n), K = static_cast<size_t>(k);
  const size_t bsz = trans_b ? N * K : K * N;
  const bool a_usm = iovs_usm_is_shared(a);
  const bool b_usm = iovs_usm_is_shared(b);
  const bool c_usm = iovs_usm_is_shared(c);
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

#if defined(IOVS_WITH_MKL)
static bool gemm_mkl_usm(const float* a, const float* b, float* c, int64_t m, int64_t n, int64_t k,
                         bool trans_b) {
  auto& q = gpu_queue();
  const size_t M = static_cast<size_t>(m), N = static_cast<size_t>(n), K = static_cast<size_t>(k);
  const size_t bsz = trans_b ? N * K : K * N;
  const bool a_usm = iovs_usm_is_shared(a);
  const bool b_usm = iovs_usm_is_shared(b);
  const bool c_usm = iovs_usm_is_shared(c);
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
#endif

enum class GpuGemmKind { Unset, Mkl, Sycl };
static GpuGemmKind g_gpu_gemm = GpuGemmKind::Unset;

static void pick_gpu_gemm() {
  if (g_gpu_gemm != GpuGemmKind::Unset) return;
  g_gpu_gemm = GpuGemmKind::Sycl;
#if defined(IOVS_WITH_MKL)
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

bool gpu_gemm(ResourcesData& r, const float* a, const float* b, float* c, int64_t m, int64_t n,
              int64_t k, bool trans_b) {
#if defined(IOVS_WITH_SYCL)
  try {
    pick_gpu_gemm();
#if defined(IOVS_WITH_MKL)
    if (g_gpu_gemm == GpuGemmKind::Mkl) {
      if (gemm_mkl_usm(a, b, c, m, n, k, trans_b)) return true;
    }
#endif
    if (gemm_sycl_usm(a, b, c, m, n, k, trans_b)) return true;
  } catch (...) {
  }
#endif
  return ov_matmul(r, "GPU", a, b, c, m, n, k, trans_b);
}

bool gpu_topk(ResourcesData& r, const float* scores, int64_t rows, int64_t cols, int64_t k,
              int64_t* indices, float* values, bool largest) {
#if defined(IOVS_WITH_SYCL)
  /* Subgroup-friendly per-row partial select on USM shared; correctness first. */
  try {
    auto& q = gpu_queue();
    const size_t R = static_cast<size_t>(rows);
    const size_t C = static_cast<size_t>(cols);
    const size_t KK = static_cast<size_t>(std::min(k, cols));
    const bool s_usm = iovs_usm_is_shared(scores);
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
#if defined(IOVS_WITH_SYCL)
  try {
    auto& q = gpu_queue();
    const size_t D = static_cast<size_t>(dim);
    const size_t N = static_cast<size_t>(nidx);
    const size_t SR = static_cast<size_t>(src_rows);
    const bool src_usm = iovs_usm_is_shared(src);
    float* S = src_usm ? const_cast<float*>(src) : usm_f(SR * D + N * D);
    int64_t* I = usm_i64(N);
    if (!S || !I) throw std::bad_alloc();
    float* O = src_usm ? usm_f(N * D) : (S + SR * D);
    if (!O) throw std::bad_alloc();
    if (!src_usm) std::memcpy(S, src, SR * D * sizeof(float));
    if (iovs_usm_is_shared(idx)) {
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
#if defined(IOVS_WITH_SYCL)
  return 1;
#else
  return 0;
#endif
}

bool gpu_cagra_walk(ResourcesData& r, const float* dataset, int64_t n, int64_t dim, iovsMetric metric,
                    const int32_t* graph, int32_t degree, const float* queries, int64_t nq, int64_t k,
                    int32_t itopk, int32_t search_width, const uint8_t* bitset, int64_t* neighbors,
                    float* distances) {
#if defined(IOVS_WITH_SYCL)
  /* Fused iGPU walk: one work-item per query, global itopk heap + per-vertex seen[n], in-kernel L2.
     Heaps and seen are sized to the real itopk and n (not 64 / 4096). Host prim walk if they cannot fit. */
  if (!gpu_available()) return false;
  if (n <= 0 || nq <= 0 || k <= 0 || dim <= 0 || degree <= 0) return false;
  itopk = std::max(itopk, static_cast<int32_t>(k));
  search_width = std::max(1, search_width);
  constexpr int64_t kMaxItopk = 4096;
  constexpr int64_t kMaxSeenBytes = 64 * 1024 * 1024;
  if (itopk > kMaxItopk) return false;
  if (nq > 0 && n > kMaxSeenBytes / nq) return false;
  try {
    auto& q = gpu_queue();
    const size_t N = static_cast<size_t>(n);
    const size_t D = static_cast<size_t>(dim);
    const size_t NQ = static_cast<size_t>(nq);
    const size_t DEG = static_cast<size_t>(degree);
    const size_t KK = static_cast<size_t>(k);
    const size_t IT = static_cast<size_t>(itopk);
    const size_t SW = static_cast<size_t>(search_width);
    const int met = static_cast<int>(metric);
    const int max_iters = std::max(24, itopk * 6);
    float* DS = iovs_usm_is_shared(dataset) ? const_cast<float*>(dataset)
                                           : static_cast<float*>(iovs_usm_malloc(N * D * sizeof(float)));
    int32_t* G = iovs_usm_is_shared(graph) ? const_cast<int32_t*>(graph)
                                          : static_cast<int32_t*>(iovs_usm_malloc(N * DEG * sizeof(int32_t)));
    float* Q = iovs_usm_is_shared(queries) ? const_cast<float*>(queries)
                                          : static_cast<float*>(iovs_usm_malloc(NQ * D * sizeof(float)));
    int64_t* OUTI = iovs_usm_is_shared(neighbors) ? neighbors
                                                 : static_cast<int64_t*>(iovs_usm_malloc(NQ * KK * sizeof(int64_t)));
    float* OUTD = iovs_usm_is_shared(distances) ? distances
                                               : static_cast<float*>(iovs_usm_malloc(NQ * KK * sizeof(float)));
    if (!DS || !G || !Q || !OUTI || !OUTD) throw std::bad_alloc();
    if (!iovs_usm_is_shared(dataset)) std::memcpy(DS, dataset, N * D * sizeof(float));
    if (!iovs_usm_is_shared(graph)) std::memcpy(G, graph, N * DEG * sizeof(int32_t));
    if (!iovs_usm_is_shared(queries)) std::memcpy(Q, queries, NQ * D * sizeof(float));
    const uint8_t* bits = bitset;
    std::vector<uint8_t> all_one;
    if (!bits) {
      all_one.assign((N + 7) / 8, 0xff);
      bits = all_one.data();
    }
    uint8_t* BS = iovs_usm_is_shared(bits) ? const_cast<uint8_t*>(bits)
                                          : static_cast<uint8_t*>(iovs_usm_malloc(((N + 7) / 8)));
    if (!BS) throw std::bad_alloc();
    if (!iovs_usm_is_shared(bits)) std::memcpy(BS, bits, (N + 7) / 8);
    uint8_t* Seen = static_cast<uint8_t*>(iovs_usm_malloc(NQ * N));
    int64_t* HID = static_cast<int64_t*>(iovs_usm_malloc(NQ * IT * sizeof(int64_t)));
    float* HD = static_cast<float*>(iovs_usm_malloc(NQ * IT * sizeof(float)));
    if (!Seen || !HID || !HD) throw std::bad_alloc();
    q.parallel_for(sycl::range<1>(NQ), [=](sycl::id<1> qiid) {
        const size_t qi = qiid[0];
        const size_t sbase = qi * N;
        const size_t hbase = qi * IT;
        for (size_t i = 0; i < N; ++i) Seen[sbase + i] = 0;
        auto dist_one = [&](size_t id) {
          float s = 0.f;
          float ip = 0.f, nx = 0.f, ny = 0.f;
          for (size_t d = 0; d < D; ++d) {
            const float a = Q[qi * D + d];
            const float b = DS[id * D + d];
            const float t = a - b;
            s += t * t;
            ip += a * b;
            nx += a * a;
            ny += b * b;
          }
          if (met == 2) return -ip;
          if (met == 3) {
            const float den = sycl::sqrt(nx) * sycl::sqrt(ny);
            return 1.f - ip / (den > 1e-12f ? den : 1e-12f);
          }
          return s;
        };
        auto allowed_id = [&](size_t id) { return (BS[id >> 3] >> (id & 7)) & 1; };
        int nkeep = 0;
        auto consider = [&](int64_t id, float dv) {
          if (id < 0 || static_cast<size_t>(id) >= N) return;
          if (Seen[sbase + static_cast<size_t>(id)] != 0) return;
          Seen[sbase + static_cast<size_t>(id)] = 1;
          if (nkeep < static_cast<int>(IT)) {
            HID[hbase + static_cast<size_t>(nkeep)] = id;
            HD[hbase + static_cast<size_t>(nkeep)] = dv;
            ++nkeep;
            return;
          }
          int wi = 0;
          float wv = HD[hbase];
          for (int i = 1; i < nkeep; ++i) {
            if (HD[hbase + static_cast<size_t>(i)] > wv) {
              wv = HD[hbase + static_cast<size_t>(i)];
              wi = i;
            }
          }
          if (dv < wv) {
            HID[hbase + static_cast<size_t>(wi)] = id;
            HD[hbase + static_cast<size_t>(wi)] = dv;
          }
        };
        for (size_t s = 0; s < SW && s < N; ++s) {
          const int64_t id = static_cast<int64_t>((s * 9973 + qi * 13) % N);
          if (!allowed_id(static_cast<size_t>(id))) continue;
          consider(id, dist_one(static_cast<size_t>(id)));
        }
        for (int iter = 0; iter < max_iters; ++iter) {
          int pick = -1;
          float best = 1e30f;
          for (int i = 0; i < nkeep; ++i) {
            const int64_t id = HID[hbase + static_cast<size_t>(i)];
            if (id < 0 || static_cast<size_t>(id) >= N) continue;
            if (Seen[sbase + static_cast<size_t>(id)] != 2 && HD[hbase + static_cast<size_t>(i)] < best) {
              best = HD[hbase + static_cast<size_t>(i)];
              pick = i;
            }
          }
          if (pick < 0) break;
          const int64_t pid = HID[hbase + static_cast<size_t>(pick)];
          Seen[sbase + static_cast<size_t>(pid)] = 2;
          for (size_t e = 0; e < DEG; ++e) {
            const int32_t nb = G[static_cast<size_t>(pid) * DEG + e];
            if (nb < 0 || static_cast<size_t>(nb) >= N) continue;
            if (!allowed_id(static_cast<size_t>(nb))) continue;
            consider(nb, dist_one(static_cast<size_t>(nb)));
          }
        }
        for (int a = 0; a < nkeep; ++a) {
          int best = a;
          for (int b = a + 1; b < nkeep; ++b)
            if (HD[hbase + static_cast<size_t>(b)] < HD[hbase + static_cast<size_t>(best)]) best = b;
          const float td = HD[hbase + static_cast<size_t>(a)];
          const int64_t ti = HID[hbase + static_cast<size_t>(a)];
          HD[hbase + static_cast<size_t>(a)] = HD[hbase + static_cast<size_t>(best)];
          HID[hbase + static_cast<size_t>(a)] = HID[hbase + static_cast<size_t>(best)];
          HD[hbase + static_cast<size_t>(best)] = td;
          HID[hbase + static_cast<size_t>(best)] = ti;
        }
        for (size_t t = 0; t < KK; ++t) {
          if (t < static_cast<size_t>(nkeep)) {
            OUTI[qi * KK + t] = HID[hbase + t];
            float d = HD[hbase + t];
            if (met == 2) d = -d;
            OUTD[qi * KK + t] = d;
          } else {
            OUTI[qi * KK + t] = -1;
            OUTD[qi * KK + t] = 3.4e38f;
          }
        }
      });
    q.wait();
    if (!iovs_usm_is_shared(neighbors)) std::memcpy(neighbors, OUTI, NQ * KK * sizeof(int64_t));
    if (!iovs_usm_is_shared(distances)) std::memcpy(distances, OUTD, NQ * KK * sizeof(float));
    if (!iovs_usm_is_shared(dataset)) iovs_usm_free(DS);
    if (!iovs_usm_is_shared(graph)) iovs_usm_free(G);
    if (!iovs_usm_is_shared(queries)) iovs_usm_free(Q);
    if (!iovs_usm_is_shared(neighbors)) iovs_usm_free(OUTI);
    if (!iovs_usm_is_shared(distances)) iovs_usm_free(OUTD);
    if (!iovs_usm_is_shared(bits)) iovs_usm_free(BS);
    iovs_usm_free(Seen);
    iovs_usm_free(HID);
    iovs_usm_free(HD);
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

bool gpu_pairwise(ResourcesData& r, iovsMetric metric, const float* x, int64_t nx, const float* y,
                  int64_t ny, int64_t dim, float* out, float metric_arg) {
#if defined(IOVS_WITH_SYCL)
  if (metric == IOVS_METRIC_L2_EXPANDED || metric == IOVS_METRIC_INNER_PRODUCT ||
      metric == IOVS_METRIC_COSINE_EXPANDED) {
    return false;
  }
  try {
    auto& q = gpu_queue();
    const size_t NX = static_cast<size_t>(nx), NY = static_cast<size_t>(ny), D = static_cast<size_t>(dim);
    const bool x_usm = iovs_usm_is_shared(x);
    const bool y_usm = iovs_usm_is_shared(y);
    const bool o_usm = iovs_usm_is_shared(out);
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
    if (!iovs_usm_is_shared(out)) std::memcpy(out, O, NX * NY * sizeof(float));
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
#if defined(IOVS_WITH_MKL)
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
#if defined(IOVS_WITH_MKL)
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
}  // namespace iovs
