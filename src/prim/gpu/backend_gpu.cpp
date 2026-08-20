#include "internal.hpp"

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
  if (ov_matmul(r, "GPU", a, b, c, m, n, k, trans_b)) return true;
#if defined(IOVS_WITH_SYCL)
  try {
    auto& q = gpu_queue();
    const size_t M = static_cast<size_t>(m), N = static_cast<size_t>(n), K = static_cast<size_t>(k);
    sycl::buffer<float, 1> ba(const_cast<float*>(a), sycl::range<1>(M * K));
    sycl::buffer<float, 1> bb(const_cast<float*>(b), sycl::range<1>(trans_b ? N * K : K * N));
    sycl::buffer<float, 1> bc(c, sycl::range<1>(M * N));
    q.submit([&](sycl::handler& h) {
      auto A = ba.get_access<sycl::access::mode::read>(h);
      auto B = bb.get_access<sycl::access::mode::read>(h);
      auto C = bc.get_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> id) {
        const size_t i = id[0];
        const size_t j = id[1];
        float s = 0.f;
        if (!trans_b) {
          for (size_t t = 0; t < K; ++t) s += A[i * K + t] * B[t * N + j];
        } else {
          for (size_t t = 0; t < K; ++t) s += A[i * K + t] * B[j * K + t];
        }
        C[i * N + j] = s;
      });
    });
    q.wait();
    return true;
  } catch (...) {
    return false;
  }
#else
  (void)r;
  (void)a;
  (void)b;
  (void)c;
  (void)m;
  (void)n;
  (void)k;
  (void)trans_b;
  return false;
#endif
}

bool gpu_topk(ResourcesData& r, const float* scores, int64_t rows, int64_t cols, int64_t k,
              int64_t* indices, float* values, bool largest) {
  if (ov_topk(r, "GPU", scores, rows, cols, k, indices, values, largest)) return true;
#if defined(IOVS_WITH_SYCL)
  /* Subgroup-friendly per-row partial select; correctness first. */
  try {
    auto& q = gpu_queue();
    const size_t R = static_cast<size_t>(rows);
    const size_t C = static_cast<size_t>(cols);
    const size_t KK = static_cast<size_t>(std::min(k, cols));
    sycl::buffer<float, 1> bs(const_cast<float*>(scores), sycl::range<1>(R * C));
    sycl::buffer<int64_t, 1> bi(indices, sycl::range<1>(R * KK));
    sycl::buffer<float, 1> bv(values, sycl::range<1>(R * KK));
    q.submit([&](sycl::handler& h) {
      auto S = bs.get_access<sycl::access::mode::read>(h);
      auto I = bi.get_access<sycl::access::mode::write>(h);
      auto V = bv.get_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<1>(R), [=](sycl::id<1> rid) {
        const size_t r = rid[0];
        for (size_t t = 0; t < KK; ++t) {
          int64_t best_i = -1;
          float best_v = largest ? -std::numeric_limits<float>::infinity()
                                 : std::numeric_limits<float>::infinity();
          for (size_t j = 0; j < C; ++j) {
            bool used = false;
            for (size_t u = 0; u < t; ++u) {
              if (I[r * KK + u] == static_cast<int64_t>(j)) {
                used = true;
                break;
              }
            }
            if (used) continue;
            const float v = S[r * C + j];
            if (largest ? v > best_v : v < best_v) {
              best_v = v;
              best_i = static_cast<int64_t>(j);
            }
          }
          I[r * KK + t] = best_i;
          V[r * KK + t] = best_v;
        }
      });
    });
    q.wait();
    return true;
  } catch (...) {
    return false;
  }
#else
  (void)r;
  (void)scores;
  (void)rows;
  (void)cols;
  (void)k;
  (void)indices;
  (void)values;
  (void)largest;
  return false;
#endif
}

bool gpu_gather_rows(ResourcesData& r, const float* src, int64_t src_rows, int64_t dim,
                     const int64_t* idx, int64_t nidx, float* out) {
  if (ov_gather_rows(r, "GPU", src, src_rows, dim, idx, nidx, out)) return true;
#if defined(IOVS_WITH_SYCL)
  try {
    auto& q = gpu_queue();
    const size_t D = static_cast<size_t>(dim);
    const size_t N = static_cast<size_t>(nidx);
    sycl::buffer<float, 1> bsrc(const_cast<float*>(src),
                                sycl::range<1>(static_cast<size_t>(src_rows) * D));
    sycl::buffer<int64_t, 1> bidx(const_cast<int64_t*>(idx), sycl::range<1>(N));
    sycl::buffer<float, 1> bout(out, sycl::range<1>(N * D));
    q.submit([&](sycl::handler& h) {
      auto S = bsrc.get_access<sycl::access::mode::read>(h);
      auto I = bidx.get_access<sycl::access::mode::read>(h);
      auto O = bout.get_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<2>(N, D), [=](sycl::id<2> id) {
        const size_t i = id[0];
        const size_t j = id[1];
        const int64_t r = I[i];
        O[i * D + j] = (r >= 0 && r < src_rows) ? S[static_cast<size_t>(r) * D + j] : 0.f;
      });
    });
    q.wait();
    return true;
  } catch (...) {
    return false;
  }
#else
  (void)r;
  (void)src;
  (void)src_rows;
  (void)dim;
  (void)idx;
  (void)nidx;
  (void)out;
  return false;
#endif
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
  /* Fused iGPU walk: one work-item per query, SLM itopk + visited hashmap, in-kernel L2. */
  if (!gpu_available()) return false;
  try {
    auto& q = gpu_queue();
    itopk = std::max(itopk, static_cast<int32_t>(k));
    itopk = std::min(itopk, 64);
    search_width = std::max(1, search_width);
    const size_t N = static_cast<size_t>(n);
    const size_t D = static_cast<size_t>(dim);
    const size_t NQ = static_cast<size_t>(nq);
    const size_t DEG = static_cast<size_t>(degree);
    const size_t KK = static_cast<size_t>(k);
    const size_t IT = static_cast<size_t>(itopk);
    const size_t SW = static_cast<size_t>(search_width);
    const int met = static_cast<int>(metric);
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
    q.submit([&](sycl::handler& h) {
      auto DS = bds.get_access<sycl::access::mode::read>(h);
      auto G = bg.get_access<sycl::access::mode::read>(h);
      auto Q = bq.get_access<sycl::access::mode::read>(h);
      auto OUTI = bn.get_access<sycl::access::mode::write>(h);
      auto OUTD = bd.get_access<sycl::access::mode::write>(h);
      auto BS = bb.get_access<sycl::access::mode::read>(h);
      sycl::local_accessor<float, 1> slm_d(sycl::range<1>(64), h);
      sycl::local_accessor<int64_t, 1> slm_id(sycl::range<1>(64), h);
      h.parallel_for(sycl::nd_range<1>(sycl::range<1>(NQ), sycl::range<1>(1)), [=](sycl::nd_item<1> itm) {
        const size_t qi = itm.get_global_id(0);
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
        auto allowed_id = [&](size_t id) {
          return (BS[id >> 3] >> (id & 7)) & 1;
        };
        /* tiny hashmap: 256-slot open address of seen ids */
        int32_t seen[256];
        for (int i = 0; i < 256; ++i) seen[i] = -1;
        auto mark = [&](int32_t id) {
          uint32_t h = static_cast<uint32_t>(id) * 2654435761u;
          for (int p = 0; p < 256; ++p) {
            uint32_t s = (h + static_cast<uint32_t>(p)) & 255u;
            if (seen[s] == -1 || seen[s] == id) {
              seen[s] = id;
              return;
            }
          }
        };
        auto was = [&](int32_t id) {
          uint32_t h = static_cast<uint32_t>(id) * 2654435761u;
          for (int p = 0; p < 256; ++p) {
            uint32_t s = (h + static_cast<uint32_t>(p)) & 255u;
            if (seen[s] == -1) return false;
            if (seen[s] == id) return true;
          }
          return true;
        };
        int nkeep = 0;
        for (size_t s = 0; s < SW && s < N; ++s) {
          int64_t id = static_cast<int64_t>((s * 9973 + qi * 13) % N);
          if (!allowed_id(static_cast<size_t>(id)) || was(static_cast<int32_t>(id))) continue;
          mark(static_cast<int32_t>(id));
          slm_id[nkeep] = id;
          slm_d[nkeep] = dist_one(static_cast<size_t>(id));
          ++nkeep;
        }
        char expd[4096];
        const size_t expn = N < 4096 ? N : 4096;
        for (size_t i = 0; i < expn; ++i) expd[i] = 0;
        const int max_iters = itopk * 6 > 24 ? itopk * 6 : 24;
        for (int iter = 0; iter < max_iters; ++iter) {
          int pick = -1;
          float best = 1e30f;
          for (int i = 0; i < nkeep; ++i) {
            const int64_t id = slm_id[i];
            if (id < 0 || static_cast<size_t>(id) >= expn) continue;
            if (!expd[id] && slm_d[i] < best) {
              best = slm_d[i];
              pick = i;
            }
          }
          if (pick < 0) break;
          const int64_t pid = slm_id[pick];
          expd[pid] = 1;
          for (size_t e = 0; e < DEG; ++e) {
            const int32_t nb = G[static_cast<size_t>(pid) * DEG + e];
            if (nb < 0 || static_cast<size_t>(nb) >= N) continue;
            if (!allowed_id(static_cast<size_t>(nb)) || was(nb)) continue;
            mark(nb);
            const float dv = dist_one(static_cast<size_t>(nb));
            if (nkeep < static_cast<int>(IT)) {
              slm_id[nkeep] = nb;
              slm_d[nkeep] = dv;
              ++nkeep;
            } else {
              int wi = 0;
              float wv = slm_d[0];
              for (int i = 1; i < nkeep; ++i) {
                if (slm_d[i] > wv) {
                  wv = slm_d[i];
                  wi = i;
                }
              }
              if (dv < wv) {
                slm_id[wi] = nb;
                slm_d[wi] = dv;
              }
            }
          }
        }
        for (int a = 0; a < nkeep; ++a) {
          int best = a;
          for (int b = a + 1; b < nkeep; ++b)
            if (slm_d[b] < slm_d[best]) best = b;
          const float td = slm_d[a];
          const int64_t ti = slm_id[a];
          slm_d[a] = slm_d[best];
          slm_id[a] = slm_id[best];
          slm_d[best] = td;
          slm_id[best] = ti;
        }
        for (size_t t = 0; t < KK; ++t) {
          if (t < static_cast<size_t>(nkeep)) {
            OUTI[qi * KK + t] = slm_id[t];
            float d = slm_d[t];
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
