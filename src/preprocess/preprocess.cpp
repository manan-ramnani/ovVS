#include "internal.hpp"

using namespace iovs::impl;

namespace {

struct Sq {
  int64_t dim = 0;
  std::vector<float> lo;
  std::vector<float> scale; /* (hi-lo)/255 */
};

struct Pq {
  int64_t dim = 0;
  int32_t pq_m = 0;
  int32_t ks = 0;
  int32_t dsub = 0;
  std::vector<float> codebooks;
};

struct BinQ {
  int64_t dim = 0;
  int64_t nbytes = 0;
};

struct Pca {
  int64_t dim = 0;
  int32_t ncomp = 0;
  std::vector<float> mean;
  std::vector<float> components; /* [ncomp, dim] */
};

}  // namespace

iovsStatus iovsSqFit(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                     iovsSqModel_t* model) {
  if (!res || !dataset || !model || n <= 0 || dim <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* m = new Sq();
  m->dim = dim;
  m->lo.assign(static_cast<size_t>(dim), kInf);
  std::vector<float> hi(static_cast<size_t>(dim), -kInf);
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t d = 0; d < dim; ++d) {
      const float v = dataset[i * dim + d];
      m->lo[static_cast<size_t>(d)] = std::min(m->lo[static_cast<size_t>(d)], v);
      hi[static_cast<size_t>(d)] = std::max(hi[static_cast<size_t>(d)], v);
    }
  }
  m->scale.resize(static_cast<size_t>(dim));
  for (int64_t d = 0; d < dim; ++d) {
    m->scale[static_cast<size_t>(d)] =
        (hi[static_cast<size_t>(d)] - m->lo[static_cast<size_t>(d)]) / 255.f;
    if (m->scale[static_cast<size_t>(d)] < 1e-12f) m->scale[static_cast<size_t>(d)] = 1.f;
  }
  *model = reinterpret_cast<iovsSqModel_t>(m);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsSqEncode(iovsSqModel_t model, const float* x, int64_t n, uint8_t* codes) {
  if (!model || !x || !codes || n <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* m = reinterpret_cast<Sq*>(model);
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t d = 0; d < m->dim; ++d) {
      float t = (x[i * m->dim + d] - m->lo[static_cast<size_t>(d)]) / m->scale[static_cast<size_t>(d)];
      t = std::min(255.f, std::max(0.f, t));
      codes[i * m->dim + d] = static_cast<uint8_t>(t + 0.5f);
    }
  }
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsSqDecode(iovsSqModel_t model, const uint8_t* codes, int64_t n, float* x) {
  if (!model || !codes || !x || n <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* m = reinterpret_cast<Sq*>(model);
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t d = 0; d < m->dim; ++d) {
      x[i * m->dim + d] =
          m->lo[static_cast<size_t>(d)] + m->scale[static_cast<size_t>(d)] * codes[i * m->dim + d];
    }
  }
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsSqDestroy(iovsSqModel_t model) {
  delete reinterpret_cast<Sq*>(model);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsPqFit(iovsResources_t res, const float* dataset, int64_t n, int64_t dim, int32_t pq_m,
                     int32_t pq_nbits, iovsPqModel_t* model) {
  if (!res || !dataset || !model || n <= 0 || dim <= 0 || pq_m <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  if (dim % pq_m != 0) return IOVS_STATUS_SHAPE_MISMATCH;
  auto* m = new Pq();
  m->dim = dim;
  m->pq_m = pq_m;
  m->ks = 1 << std::min(std::max(pq_nbits, 4), 8);
  m->dsub = static_cast<int32_t>(dim / pq_m);
  m->codebooks.assign(static_cast<size_t>(pq_m) * m->ks * m->dsub, 0.f);
  std::vector<float> sub(static_cast<size_t>(n) * m->dsub);
  for (int32_t s = 0; s < pq_m; ++s) {
    for (int64_t i = 0; i < n; ++i) {
      std::memcpy(sub.data() + i * m->dsub, dataset + i * dim + static_cast<int64_t>(s) * m->dsub,
                  static_cast<size_t>(m->dsub) * sizeof(float));
    }
    std::vector<float> cents;
    kmeans_fit_impl(*rd(res), sub.data(), n, m->dsub, m->ks, 8, cents);
    std::memcpy(m->codebooks.data() + static_cast<size_t>(s) * m->ks * m->dsub, cents.data(),
                cents.size() * sizeof(float));
  }
  *model = reinterpret_cast<iovsPqModel_t>(m);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsPqEncode(iovsPqModel_t model, const float* x, int64_t n, uint8_t* codes) {
  if (!model || !x || !codes) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* m = reinterpret_cast<Pq*>(model);
  for (int64_t i = 0; i < n; ++i) {
    for (int32_t s = 0; s < m->pq_m; ++s) {
      const float* sub = x + i * m->dim + static_cast<int64_t>(s) * m->dsub;
      const float* cb = m->codebooks.data() + static_cast<size_t>(s) * m->ks * m->dsub;
      int best = 0;
      float bd = kInf;
      for (int32_t c = 0; c < m->ks; ++c) {
        const float d = l2sq(sub, cb + c * m->dsub, m->dsub);
        if (d < bd) {
          bd = d;
          best = c;
        }
      }
      codes[i * m->pq_m + s] = static_cast<uint8_t>(best);
    }
  }
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsPqDecode(iovsPqModel_t model, const uint8_t* codes, int64_t n, float* x) {
  if (!model || !codes || !x) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* m = reinterpret_cast<Pq*>(model);
  for (int64_t i = 0; i < n; ++i) {
    for (int32_t s = 0; s < m->pq_m; ++s) {
      const uint8_t c = codes[i * m->pq_m + s];
      const float* cb = m->codebooks.data() + (static_cast<size_t>(s) * m->ks + c) * m->dsub;
      std::memcpy(x + i * m->dim + static_cast<int64_t>(s) * m->dsub, cb,
                  static_cast<size_t>(m->dsub) * sizeof(float));
    }
  }
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsPqDestroy(iovsPqModel_t model) {
  delete reinterpret_cast<Pq*>(model);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsBinaryFit(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                         iovsBinaryQuantizer_t* model) {
  if (!res || !dataset || !model || n <= 0 || dim <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* m = new BinQ();
  m->dim = dim;
  m->nbytes = (dim + 7) / 8;
  (void)dataset;
  *model = reinterpret_cast<iovsBinaryQuantizer_t>(m);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsBinaryEncode(iovsBinaryQuantizer_t model, const float* x, int64_t n, uint8_t* codes) {
  if (!model || !x || !codes) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* m = reinterpret_cast<BinQ*>(model);
  std::fill(codes, codes + n * m->nbytes, 0);
  for (int64_t i = 0; i < n; ++i) {
    uint8_t* row = codes + i * m->nbytes;
    for (int64_t d = 0; d < m->dim; ++d) {
      if (x[i * m->dim + d] >= 0.f) row[d >> 3] |= static_cast<uint8_t>(1u << (d & 7));
    }
  }
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsBinaryDestroy(iovsBinaryQuantizer_t model) {
  delete reinterpret_cast<BinQ*>(model);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsPcaFit(iovsResources_t res, const float* dataset, int64_t n, int64_t dim, int32_t ncomp,
                      iovsPcaModel_t* model) {
  if (!res || !dataset || !model || n <= 0 || dim <= 0 || ncomp <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  ncomp = std::min(ncomp, static_cast<int32_t>(dim));
  auto* m = new Pca();
  m->dim = dim;
  m->ncomp = ncomp;
  m->mean.assign(static_cast<size_t>(dim), 0.f);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t d = 0; d < dim; ++d) m->mean[static_cast<size_t>(d)] += dataset[i * dim + d];
  for (int64_t d = 0; d < dim; ++d) m->mean[static_cast<size_t>(d)] /= static_cast<float>(n);
  std::vector<float> xc(static_cast<size_t>(n) * static_cast<size_t>(dim));
  for (int64_t i = 0; i < n; ++i)
    for (int64_t d = 0; d < dim; ++d)
      xc[static_cast<size_t>(i * dim + d)] = dataset[i * dim + d] - m->mean[static_cast<size_t>(d)];
  /* C = X^T X  (dim x dim) via GEMM: C = X^T @ X, trans_b=false with A=X^T is awkward;
     compute C[i,j] = sum_n xc[n,i]*xc[n,j] using gemm on (dim,n) x (n,dim). */
  std::vector<float> xt(static_cast<size_t>(dim) * static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    for (int64_t d = 0; d < dim; ++d) xt[static_cast<size_t>(d * n + i)] = xc[static_cast<size_t>(i * dim + d)];
  std::vector<float> cov(static_cast<size_t>(dim * dim));
  prim_gemm(*rd(res), xt.data(), xc.data(), cov.data(), dim, dim, n, false);
  const float invn = 1.f / static_cast<float>(std::max<int64_t>(n - 1, 1));
  for (float& v : cov) v *= invn;
  m->components.assign(static_cast<size_t>(ncomp) * static_cast<size_t>(dim), 0.f);
  if (mkl_gesvd_components(xc.data(), n, dim, ncomp, m->components.data())) {
    *model = reinterpret_cast<iovsPcaModel_t>(m);
    return IOVS_STATUS_SUCCESS;
  }
  auto rng = rng_from(11);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  for (int32_t c = 0; c < ncomp; ++c) {
    std::vector<float> v(static_cast<size_t>(dim));
    for (int64_t d = 0; d < dim; ++d) v[static_cast<size_t>(d)] = u(rng);
    for (int it = 0; it < 40; ++it) {
      std::vector<float> nv(static_cast<size_t>(dim), 0.f);
      for (int64_t i = 0; i < dim; ++i)
        for (int64_t j = 0; j < dim; ++j)
          nv[static_cast<size_t>(i)] += cov[static_cast<size_t>(i * dim + j)] * v[static_cast<size_t>(j)];
      for (int32_t p = 0; p < c; ++p) {
        float ip = 0.f;
        const float* uvec = m->components.data() + static_cast<size_t>(p) * dim;
        for (int64_t d = 0; d < dim; ++d) ip += nv[static_cast<size_t>(d)] * uvec[d];
        for (int64_t d = 0; d < dim; ++d) nv[static_cast<size_t>(d)] -= ip * uvec[d];
      }
      float nrm = 0.f;
      for (float x : nv) nrm += x * x;
      nrm = std::sqrt(std::max(nrm, 1e-12f));
      for (int64_t d = 0; d < dim; ++d) v[static_cast<size_t>(d)] = nv[static_cast<size_t>(d)] / nrm;
    }
    std::memcpy(m->components.data() + static_cast<size_t>(c) * dim, v.data(),
                static_cast<size_t>(dim) * sizeof(float));
  }
  *model = reinterpret_cast<iovsPcaModel_t>(m);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsPcaTransform(iovsPcaModel_t model, const float* x, int64_t n, float* out) {
  if (!model || !x || !out) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* m = reinterpret_cast<Pca*>(model);
  for (int64_t i = 0; i < n; ++i) {
    for (int32_t c = 0; c < m->ncomp; ++c) {
      float s = 0.f;
      const float* w = m->components.data() + static_cast<size_t>(c) * m->dim;
      for (int64_t d = 0; d < m->dim; ++d) s += (x[i * m->dim + d] - m->mean[static_cast<size_t>(d)]) * w[d];
      out[i * m->ncomp + c] = s;
    }
  }
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsPcaDestroy(iovsPcaModel_t model) {
  delete reinterpret_cast<Pca*>(model);
  return IOVS_STATUS_SUCCESS;
}
