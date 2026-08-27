#include "internal.hpp"

namespace ovvs {
namespace impl {

void cpu_gemm(const float* a, const float* b, float* c, int64_t m, int64_t n, int64_t k,
              bool trans_b) {
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      float s = 0.f;
      if (!trans_b) {
        for (int64_t t = 0; t < k; ++t) s += a[i * k + t] * b[t * n + j];
      } else {
        for (int64_t t = 0; t < k; ++t) s += a[i * k + t] * b[j * k + t];
      }
      c[i * n + j] = s;
    }
  }
}

void cpu_topk(const float* scores, int64_t rows, int64_t cols, int64_t k, int64_t* indices,
              float* values, bool largest) {
  k = std::min(k, cols);
  std::vector<int64_t> idx(static_cast<size_t>(cols));
  for (int64_t r = 0; r < rows; ++r) {
    const float* row = scores + r * cols;
    std::iota(idx.begin(), idx.end(), 0);
    if (largest) {
      std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                        [&](int64_t i, int64_t j) { return row[i] > row[j]; });
    } else {
      std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                        [&](int64_t i, int64_t j) { return row[i] < row[j]; });
    }
    for (int64_t t = 0; t < k; ++t) {
      indices[r * k + t] = idx[static_cast<size_t>(t)];
      values[r * k + t] = row[idx[static_cast<size_t>(t)]];
    }
  }
}

void cpu_gather_rows(const float* src, int64_t src_rows, int64_t dim, const int64_t* idx,
                     int64_t nidx, float* out) {
  for (int64_t i = 0; i < nidx; ++i) {
    const int64_t r = idx[i];
    if (r < 0 || r >= src_rows) {
      std::fill(out + i * dim, out + (i + 1) * dim, 0.f);
      continue;
    }
    std::memcpy(out + i * dim, src + r * dim, static_cast<size_t>(dim) * sizeof(float));
  }
}

void cpu_pairwise(ovvsMetric metric, const float* x, int64_t nx, const float* y, int64_t ny,
                  int64_t dim, float* out, float metric_arg) {
  if (metric == OVVS_METRIC_L2_EXPANDED || metric == OVVS_METRIC_INNER_PRODUCT ||
      metric == OVVS_METRIC_COSINE_EXPANDED) {
    std::vector<float> xnorm(static_cast<size_t>(nx)), ynorm(static_cast<size_t>(ny));
    for (int64_t i = 0; i < nx; ++i) xnorm[static_cast<size_t>(i)] = nrm2sq(x + i * dim, dim);
    for (int64_t j = 0; j < ny; ++j) ynorm[static_cast<size_t>(j)] = nrm2sq(y + j * dim, dim);
    cpu_gemm(x, y, out, nx, ny, dim, true);
    for (int64_t i = 0; i < nx; ++i) {
      for (int64_t j = 0; j < ny; ++j) {
        const float ip = out[i * ny + j];
        if (metric == OVVS_METRIC_INNER_PRODUCT) {
          out[i * ny + j] = -ip;
        } else if (metric == OVVS_METRIC_COSINE_EXPANDED) {
          const float nxv = std::sqrt(std::max(xnorm[static_cast<size_t>(i)], 1e-12f));
          const float nyv = std::sqrt(std::max(ynorm[static_cast<size_t>(j)], 1e-12f));
          out[i * ny + j] = 1.f - ip / (nxv * nyv);
        } else {
          out[i * ny + j] = xnorm[static_cast<size_t>(i)] + ynorm[static_cast<size_t>(j)] - 2.f * ip;
        }
      }
    }
    return;
  }
  for (int64_t i = 0; i < nx; ++i) {
    for (int64_t j = 0; j < ny; ++j) {
      out[i * ny + j] = distance_one(metric, x + i * dim, y + j * dim, dim, metric_arg);
    }
  }
}

}  // namespace impl
}  // namespace ovvs
