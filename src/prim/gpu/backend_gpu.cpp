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
          float best_v = largest ? -sycl::numeric_limits<float>::infinity()
                                 : sycl::numeric_limits<float>::infinity();
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

}  // namespace impl
}  // namespace iovs
