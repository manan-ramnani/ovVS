#include "internal.hpp"

#include <new>

#if defined(IOVS_WITH_SYCL)
#include <sycl/sycl.hpp>
#endif

namespace iovs {
namespace impl {

#if defined(IOVS_WITH_SYCL)
static sycl::queue& gpu_queue() {
  static sycl::queue q{sycl::gpu_selector_v};
  return q;
}

template <typename T>
static T* usm_t(size_t n) {
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

static float* usm_f(size_t n) { return usm_t<float>(n); }
static int64_t* usm_i64(size_t n) { return usm_t<int64_t>(n); }
#endif

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

bool gpu_gemm(ResourcesData& r, const float* a, const float* b, float* c, int64_t m, int64_t n,
              int64_t k, bool trans_b) {
#if defined(IOVS_WITH_SYCL)
  try {
    auto& q = gpu_queue();
    const size_t M = static_cast<size_t>(m), N = static_cast<size_t>(n), K = static_cast<size_t>(k);
    const size_t bsz = trans_b ? N * K : K * N;
    float* A = usm_f(M * K + bsz + M * N);
    if (!A) throw std::bad_alloc();
    float* B = A + M * K;
    float* C = B + bsz;
    std::memcpy(A, a, M * K * sizeof(float));
    std::memcpy(B, b, bsz * sizeof(float));
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
    std::memcpy(c, C, M * N * sizeof(float));
    return true;
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
    float* S = usm_f(R * C + R * KK);
    int64_t* I = usm_i64(R * KK);
    if (!S || !I) throw std::bad_alloc();
    float* V = S + R * C;
    std::memcpy(S, scores, R * C * sizeof(float));
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
    float* S = usm_f(SR * D + N * D);
    int64_t* I = usm_i64(N);
    if (!S || !I) throw std::bad_alloc();
    float* O = S + SR * D;
    std::memcpy(S, src, SR * D * sizeof(float));
    std::memcpy(I, idx, N * sizeof(int64_t));
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
    sycl::buffer<float, 1> bds(const_cast<float*>(dataset), sycl::range<1>(N * D));
    sycl::buffer<int32_t, 1> bg(const_cast<int32_t*>(graph), sycl::range<1>(N * DEG));
    sycl::buffer<float, 1> bq(const_cast<float*>(queries), sycl::range<1>(NQ * D));
    sycl::buffer<int64_t, 1> bn(neighbors, sycl::range<1>(NQ * KK));
    sycl::buffer<float, 1> bd(distances, sycl::range<1>(NQ * KK));
    const uint8_t* bits = bitset;
    std::vector<uint8_t> all_one;
    if (!bits) {
      all_one.assign((N + 7) / 8, 0xff);
      bits = all_one.data();
    }
    sycl::buffer<uint8_t, 1> bb(const_cast<uint8_t*>(bits), sycl::range<1>((N + 7) / 8));
    /* 0 = unseen, 1 = in heap / seen, 2 = expanded. Length NQ*N so id>=4096 is visitable. */
    sycl::buffer<uint8_t, 1> bseen(sycl::range<1>(NQ * N));
    sycl::buffer<int64_t, 1> bhid(sycl::range<1>(NQ * IT));
    sycl::buffer<float, 1> bhd(sycl::range<1>(NQ * IT));
    q.submit([&](sycl::handler& h) {
      auto DS = bds.get_access<sycl::access::mode::read>(h);
      auto G = bg.get_access<sycl::access::mode::read>(h);
      auto Q = bq.get_access<sycl::access::mode::read>(h);
      auto OUTI = bn.get_access<sycl::access::mode::write>(h);
      auto OUTD = bd.get_access<sycl::access::mode::write>(h);
      auto BS = bb.get_access<sycl::access::mode::read>(h);
      auto Seen = bseen.get_access<sycl::access::mode::read_write>(h);
      auto HID = bhid.get_access<sycl::access::mode::read_write>(h);
      auto HD = bhd.get_access<sycl::access::mode::read_write>(h);
      h.parallel_for(sycl::range<1>(NQ), [=](sycl::id<1> qiid) {
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
    });
    q.wait();
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

}  // namespace impl
}  // namespace iovs
