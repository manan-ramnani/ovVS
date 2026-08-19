#include "internal.hpp"

#include <cstring>

namespace iovs {
namespace impl {

static iovsDevice policy_force(iovsPolicy p) {
  switch (p) {
    case IOVS_POLICY_FORCE_NPU:
      return IOVS_DEVICE_NPU;
    case IOVS_POLICY_FORCE_GPU:
      return IOVS_DEVICE_GPU;
    case IOVS_POLICY_FORCE_CPU:
      return IOVS_DEVICE_CPU;
    default:
      return IOVS_DEVICE_AUTO;
  }
}

iovsDevice choose_device(ResourcesData& r, const char* op, int64_t flops_or_elems) {
  const iovsDevice forced = policy_force(r.policy);
  if (forced != IOVS_DEVICE_AUTO) return forced;

  const bool npu_shaped =
      std::strcmp(op, "gemm") == 0 || std::strcmp(op, "pairwise") == 0 ||
      std::strcmp(op, "topk") == 0 || std::strcmp(op, "gather") == 0;
  if (r.npu_available && npu_shaped && flops_or_elems >= 256 * 256 * 32) {
    return IOVS_DEVICE_NPU;
  }
  if (r.gpu_available && flops_or_elems >= 64 * 64) {
    if (npu_shaped) return IOVS_DEVICE_GPU;
  }
  return IOVS_DEVICE_CPU;
}

static iovsStatus finish_forced_fail(ResourcesData& r) {
  ++r.npu_fallbacks;
  return IOVS_STATUS_DEVICE_UNAVAILABLE;
}

iovsStatus prim_gemm(ResourcesData& r, const float* a, const float* b, float* c, int64_t m, int64_t n,
                     int64_t k, bool trans_b) {
  const iovsDevice d = choose_device(r, "gemm", m * n * k);
  if (d == IOVS_DEVICE_NPU) {
    if (npu_gemm(r, a, b, c, m, n, k, trans_b)) {
      r.last_device = IOVS_DEVICE_NPU;
      return IOVS_STATUS_SUCCESS;
    }
    if (r.policy == IOVS_POLICY_FORCE_NPU) return finish_forced_fail(r);
  }
  if (d == IOVS_DEVICE_GPU || (d == IOVS_DEVICE_NPU && r.policy != IOVS_POLICY_FORCE_CPU)) {
    if (gpu_gemm(r, a, b, c, m, n, k, trans_b)) {
      r.last_device = IOVS_DEVICE_GPU;
      return IOVS_STATUS_SUCCESS;
    }
    if (r.policy == IOVS_POLICY_FORCE_GPU) return finish_forced_fail(r);
  }
  if (r.policy == IOVS_POLICY_FORCE_NPU || r.policy == IOVS_POLICY_FORCE_GPU) {
    return finish_forced_fail(r);
  }
  cpu_gemm(a, b, c, m, n, k, trans_b);
  r.last_device = IOVS_DEVICE_CPU;
  return IOVS_STATUS_SUCCESS;
}

iovsStatus prim_topk(ResourcesData& r, const float* scores, int64_t rows, int64_t cols, int64_t k,
                     int64_t* indices, float* values, bool largest) {
  const iovsDevice d = choose_device(r, "topk", rows * cols);
  if (d == IOVS_DEVICE_NPU) {
    if (npu_topk(r, scores, rows, cols, k, indices, values, largest)) {
      r.last_device = IOVS_DEVICE_NPU;
      return IOVS_STATUS_SUCCESS;
    }
    if (r.policy == IOVS_POLICY_FORCE_NPU) return finish_forced_fail(r);
  }
  if (d == IOVS_DEVICE_GPU || (d == IOVS_DEVICE_NPU && r.policy != IOVS_POLICY_FORCE_CPU)) {
    if (gpu_topk(r, scores, rows, cols, k, indices, values, largest)) {
      r.last_device = IOVS_DEVICE_GPU;
      return IOVS_STATUS_SUCCESS;
    }
    if (r.policy == IOVS_POLICY_FORCE_GPU) return finish_forced_fail(r);
  }
  if (r.policy == IOVS_POLICY_FORCE_NPU || r.policy == IOVS_POLICY_FORCE_GPU) {
    return finish_forced_fail(r);
  }
  cpu_topk(scores, rows, cols, k, indices, values, largest);
  r.last_device = IOVS_DEVICE_CPU;
  return IOVS_STATUS_SUCCESS;
}

iovsStatus prim_gather_rows(ResourcesData& r, const float* src, int64_t src_rows, int64_t dim,
                            const int64_t* idx, int64_t nidx, float* out) {
  const iovsDevice d = choose_device(r, "gather", nidx * dim);
  if (d == IOVS_DEVICE_NPU) {
    if (npu_gather_rows(r, src, src_rows, dim, idx, nidx, out)) {
      r.last_device = IOVS_DEVICE_NPU;
      return IOVS_STATUS_SUCCESS;
    }
    if (r.policy == IOVS_POLICY_FORCE_NPU) return finish_forced_fail(r);
  }
  if (d == IOVS_DEVICE_GPU || (d == IOVS_DEVICE_NPU && r.policy != IOVS_POLICY_FORCE_CPU)) {
    if (gpu_gather_rows(r, src, src_rows, dim, idx, nidx, out)) {
      r.last_device = IOVS_DEVICE_GPU;
      return IOVS_STATUS_SUCCESS;
    }
    if (r.policy == IOVS_POLICY_FORCE_GPU) return finish_forced_fail(r);
  }
  if (r.policy == IOVS_POLICY_FORCE_NPU || r.policy == IOVS_POLICY_FORCE_GPU) {
    return finish_forced_fail(r);
  }
  cpu_gather_rows(src, src_rows, dim, idx, nidx, out);
  r.last_device = IOVS_DEVICE_CPU;
  return IOVS_STATUS_SUCCESS;
}

iovsStatus prim_pairwise(ResourcesData& r, iovsMetric metric, const float* x, int64_t nx,
                         const float* y, int64_t ny, int64_t dim, float* out, float metric_arg) {
  if (metric == IOVS_METRIC_L2_EXPANDED || metric == IOVS_METRIC_INNER_PRODUCT ||
      metric == IOVS_METRIC_COSINE_EXPANDED) {
    std::vector<float> xnorm(static_cast<size_t>(nx)), ynorm(static_cast<size_t>(ny));
    for (int64_t i = 0; i < nx; ++i) xnorm[static_cast<size_t>(i)] = nrm2sq(x + i * dim, dim);
    for (int64_t j = 0; j < ny; ++j) ynorm[static_cast<size_t>(j)] = nrm2sq(y + j * dim, dim);
    const iovsStatus gs = prim_gemm(r, x, y, out, nx, ny, dim, true);
    if (gs != IOVS_STATUS_SUCCESS) return gs;
    for (int64_t i = 0; i < nx; ++i) {
      for (int64_t j = 0; j < ny; ++j) {
        const float ip = out[i * ny + j];
        if (metric == IOVS_METRIC_INNER_PRODUCT) {
          out[i * ny + j] = -ip;
        } else if (metric == IOVS_METRIC_COSINE_EXPANDED) {
          const float nxv = std::sqrt(std::max(xnorm[static_cast<size_t>(i)], 1e-12f));
          const float nyv = std::sqrt(std::max(ynorm[static_cast<size_t>(j)], 1e-12f));
          out[i * ny + j] = 1.f - ip / (nxv * nyv);
        } else {
          out[i * ny + j] = xnorm[static_cast<size_t>(i)] + ynorm[static_cast<size_t>(j)] - 2.f * ip;
        }
      }
    }
    return IOVS_STATUS_SUCCESS;
  }
  if (r.policy == IOVS_POLICY_FORCE_NPU || r.policy == IOVS_POLICY_FORCE_GPU) {
    return finish_forced_fail(r);
  }
  cpu_pairwise(metric, x, nx, y, ny, dim, out, metric_arg);
  r.last_device = IOVS_DEVICE_CPU;
  return IOVS_STATUS_SUCCESS;
}

void brute_search_impl(ResourcesData& r, const float* dataset, int64_t n, int64_t dim,
                       const float* queries, int64_t nq, iovsMetric metric, int64_t k,
                       const uint8_t* bitset, int64_t* neighbors, float* distances) {
  std::vector<float> scores(static_cast<size_t>(nq * n));
  prim_pairwise(r, metric, queries, nq, dataset, n, dim, scores.data(), 2.f);
  if (bitset) {
    for (int64_t i = 0; i < nq; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        if (!allowed(bitset, j)) {
          scores[static_cast<size_t>(i * n + j)] = metric_largest(metric) ? -kInf : kInf;
        }
      }
    }
  }
  prim_topk(r, scores.data(), nq, n, k, neighbors, distances, metric_largest(metric));
  if (metric == IOVS_METRIC_INNER_PRODUCT) {
    for (int64_t i = 0; i < nq * k; ++i) distances[i] = -distances[i];
  }
}

void kmeans_fit_impl(ResourcesData& r, const float* x, int64_t n, int64_t dim, int32_t k,
                     int32_t iters, std::vector<float>& centroids) {
  k = std::max(1, std::min(k, static_cast<int32_t>(n)));
  centroids.assign(static_cast<size_t>(k) * static_cast<size_t>(dim), 0.f);
  auto rng = rng_from(42);
  std::uniform_int_distribution<int64_t> pick(0, n - 1);
  std::vector<int64_t> used;
  for (int32_t c = 0; c < k; ++c) {
    int64_t i = pick(rng);
    for (int t = 0; t < 8 && std::find(used.begin(), used.end(), i) != used.end(); ++t) i = pick(rng);
    used.push_back(i);
    std::memcpy(centroids.data() + static_cast<size_t>(c) * dim, x + i * dim,
                static_cast<size_t>(dim) * sizeof(float));
  }
  std::vector<int64_t> labels(static_cast<size_t>(n));
  std::vector<float> dist(static_cast<size_t>(n));
  std::vector<float> scores(static_cast<size_t>(n) * static_cast<size_t>(k));
  for (int it = 0; it < iters; ++it) {
    prim_pairwise(r, IOVS_METRIC_L2_EXPANDED, x, n, centroids.data(), k, dim, scores.data(), 2.f);
    prim_topk(r, scores.data(), n, k, 1, labels.data(), dist.data(), false);
    std::vector<float> sum(static_cast<size_t>(k) * static_cast<size_t>(dim), 0.f);
    std::vector<int32_t> cnt(static_cast<size_t>(k), 0);
    for (int64_t i = 0; i < n; ++i) {
      const int64_t c = labels[static_cast<size_t>(i)];
      if (c < 0 || c >= k) continue;
      ++cnt[static_cast<size_t>(c)];
      float* dst = sum.data() + c * dim;
      const float* src = x + i * dim;
      for (int64_t d = 0; d < dim; ++d) dst[d] += src[d];
    }
    for (int32_t c = 0; c < k; ++c) {
      if (cnt[static_cast<size_t>(c)] == 0) continue;
      float* dst = centroids.data() + static_cast<size_t>(c) * dim;
      const float* src = sum.data() + static_cast<size_t>(c) * dim;
      const float inv = 1.f / static_cast<float>(cnt[static_cast<size_t>(c)]);
      for (int64_t d = 0; d < dim; ++d) dst[d] = src[d] * inv;
    }
  }
}

}  // namespace impl
}  // namespace iovs
