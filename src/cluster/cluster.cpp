#include "internal.hpp"

using namespace iovs::impl;

namespace {

struct KMeans {
  std::vector<float> centroids;
  int32_t k = 0;
  int64_t dim = 0;
};

struct Slink {
  std::vector<int64_t> labels;
};

struct Spectral {
  std::vector<int64_t> labels;
};

struct SpectralEmbed {
  std::vector<float> z;
  int64_t n = 0;
  int32_t ncomp = 0;
};

int64_t ufind(std::vector<int64_t>& p, int64_t x) {
  while (p[static_cast<size_t>(x)] != x) {
    p[static_cast<size_t>(x)] = p[static_cast<size_t>(p[static_cast<size_t>(x)])];
    x = p[static_cast<size_t>(x)];
  }
  return x;
}

void unite(std::vector<int64_t>& p, std::vector<int64_t>& r, int64_t a, int64_t b) {
  a = ufind(p, a);
  b = ufind(p, b);
  if (a == b) return;
  if (r[static_cast<size_t>(a)] < r[static_cast<size_t>(b)]) std::swap(a, b);
  p[static_cast<size_t>(b)] = a;
  if (r[static_cast<size_t>(a)] == r[static_cast<size_t>(b)]) ++r[static_cast<size_t>(a)];
}

}  // namespace

iovsStatus iovsKMeansFit(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                         int32_t nclusters, int32_t iters, iovsKMeansModel_t* model) {
  if (!res || !dataset || !model || n <= 0 || dim <= 0 || nclusters <= 0) {
    return IOVS_STATUS_INVALID_ARGUMENT;
  }
  auto* m = new KMeans();
  m->k = nclusters;
  m->dim = dim;
  kmeans_fit_impl(*rd(res), dataset, n, dim, nclusters, std::max(4, iters), m->centroids);
  *model = reinterpret_cast<iovsKMeansModel_t>(m);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsKMeansPredict(iovsResources_t res, iovsKMeansModel_t model, const float* x, int64_t n,
                             int64_t* labels, float* distances) {
  if (!res || !model || !x || !labels || n <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* m = reinterpret_cast<KMeans*>(model);
  std::vector<float> scores(static_cast<size_t>(n) * static_cast<size_t>(m->k));
  prim_pairwise(*rd(res), IOVS_METRIC_L2_EXPANDED, x, n, m->centroids.data(), m->k, m->dim,
                scores.data(), 2.f);
  std::vector<float> tmp(static_cast<size_t>(n));
  prim_topk(*rd(res), scores.data(), n, m->k, 1, labels, distances ? distances : tmp.data(), false);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsKMeansCentroids(iovsKMeansModel_t model, const float** data, int32_t* nclusters,
                               int64_t* dim) {
  if (!model || !data || !nclusters || !dim) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* m = reinterpret_cast<KMeans*>(model);
  *data = m->centroids.data();
  *nclusters = m->k;
  *dim = m->dim;
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsKMeansDestroy(iovsKMeansModel_t model) {
  delete reinterpret_cast<KMeans*>(model);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsSlinkFit(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                        int32_t nclusters, int32_t knn, iovsSlinkModel_t* model) {
  if (!res || !dataset || !model || n <= 0 || nclusters <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  knn = std::max(1, std::min(knn, static_cast<int32_t>(n - 1)));
  std::vector<int64_t> nbr(static_cast<size_t>(n) * static_cast<size_t>(knn));
  std::vector<float> dist(static_cast<size_t>(n) * static_cast<size_t>(knn));
  const iovsStatus st =
      iovsAllNeighbors(res, dataset, n, dim, IOVS_METRIC_L2_EXPANDED, knn, nbr.data(), dist.data());
  if (st != IOVS_STATUS_SUCCESS) return st;
  struct Edge {
    float d;
    int64_t a, b;
    bool operator<(const Edge& o) const { return d < o.d; }
  };
  std::vector<Edge> edges;
  edges.reserve(static_cast<size_t>(n) * knn);
  for (int64_t i = 0; i < n; ++i) {
    for (int32_t t = 0; t < knn; ++t) {
      const int64_t j = nbr[static_cast<size_t>(i * knn + t)];
      if (j < 0 || j <= i) continue;
      edges.push_back({dist[static_cast<size_t>(i * knn + t)], i, j});
    }
  }
  std::sort(edges.begin(), edges.end());
  std::vector<int64_t> parent(static_cast<size_t>(n)), rank(static_cast<size_t>(n), 0);
  std::iota(parent.begin(), parent.end(), 0);
  int64_t comps = n;
  for (const auto& e : edges) {
    if (ufind(parent, e.a) == ufind(parent, e.b)) continue;
    unite(parent, rank, e.a, e.b);
    --comps;
    if (comps <= nclusters) break;
  }
  auto* m = new Slink();
  m->labels.resize(static_cast<size_t>(n));
  std::vector<int64_t> map(static_cast<size_t>(n), -1);
  int64_t next = 0;
  for (int64_t i = 0; i < n; ++i) {
    const int64_t r = ufind(parent, i);
    if (map[static_cast<size_t>(r)] < 0) map[static_cast<size_t>(r)] = next++;
    m->labels[static_cast<size_t>(i)] = map[static_cast<size_t>(r)];
  }
  *model = reinterpret_cast<iovsSlinkModel_t>(m);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsSlinkLabels(iovsSlinkModel_t model, const int64_t** labels, int64_t* n) {
  if (!model || !labels || !n) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* m = reinterpret_cast<Slink*>(model);
  *labels = m->labels.data();
  *n = static_cast<int64_t>(m->labels.size());
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsSlinkDestroy(iovsSlinkModel_t model) {
  delete reinterpret_cast<Slink*>(model);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsSpectralFit(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                           int32_t nclusters, int32_t knn, iovsSpectralModel_t* model) {
  if (!res || !dataset || !model || n <= 0 || nclusters <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  knn = std::max(1, std::min(knn, static_cast<int32_t>(n - 1)));
  std::vector<int64_t> nbr(static_cast<size_t>(n) * static_cast<size_t>(knn));
  std::vector<float> dist(static_cast<size_t>(n) * static_cast<size_t>(knn));
  iovsAllNeighbors(res, dataset, n, dim, IOVS_METRIC_L2_EXPANDED, knn, nbr.data(), dist.data());
  float mean = 0.f;
  for (float v : dist) mean += v;
  mean /= static_cast<float>(std::max<size_t>(dist.size(), 1));
  const float sigma = std::max(mean, 1e-6f);
  std::vector<float> W(static_cast<size_t>(n * n), 0.f);
  std::vector<float> deg(static_cast<size_t>(n), 0.f);
  for (int64_t i = 0; i < n; ++i) {
    for (int32_t t = 0; t < knn; ++t) {
      const int64_t j = nbr[static_cast<size_t>(i * knn + t)];
      if (j < 0) continue;
      const float a = std::exp(-dist[static_cast<size_t>(i * knn + t)] / (2.f * sigma));
      W[static_cast<size_t>(i * n + j)] = std::max(W[static_cast<size_t>(i * n + j)], a);
      W[static_cast<size_t>(j * n + i)] = std::max(W[static_cast<size_t>(j * n + i)], a);
    }
  }
  for (int64_t i = 0; i < n; ++i) {
    float s = 0.f;
    for (int64_t j = 0; j < n; ++j) s += W[static_cast<size_t>(i * n + j)];
    deg[static_cast<size_t>(i)] = s;
  }
  /* Normalized Laplacian L = I - D^{-1/2} W D^{-1/2}; oneMKL syev when available. */
  const int32_t k = nclusters;
  std::vector<float> embed(static_cast<size_t>(n) * static_cast<size_t>(k), 0.f);
  std::vector<float> L(static_cast<size_t>(n * n), 0.f);
  for (int64_t i = 0; i < n; ++i) {
    const float di = deg[static_cast<size_t>(i)];
    const float isi = di > 1e-12f ? 1.f / std::sqrt(di) : 0.f;
    for (int64_t j = 0; j < n; ++j) {
      const float dj = deg[static_cast<size_t>(j)];
      const float isj = dj > 1e-12f ? 1.f / std::sqrt(dj) : 0.f;
      float v = (i == j ? 1.f : 0.f) - isi * W[static_cast<size_t>(i * n + j)] * isj;
      L[static_cast<size_t>(i * n + j)] = v;
    }
  }
  const bool syev_ok = mkl_syev_smallest(L.data(), n, k, embed.data());
  auto rng = rng_from(5);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  for (int32_t c = 0; !syev_ok && c < k; ++c) {
    std::vector<float> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = u(rng);
    for (int it = 0; it < 40; ++it) {
      std::vector<float> nv(static_cast<size_t>(n), 0.f);
      for (int64_t i = 0; i < n; ++i) {
        if (deg[static_cast<size_t>(i)] <= 1e-12f) continue;
        float s = 0.f;
        for (int64_t j = 0; j < n; ++j) s += W[static_cast<size_t>(i * n + j)] * v[static_cast<size_t>(j)];
        nv[static_cast<size_t>(i)] = s / deg[static_cast<size_t>(i)];
      }
      /* orthogonalize against previous embeddings */
      for (int32_t p = 0; p < c; ++p) {
        float ip = 0.f;
        for (int64_t i = 0; i < n; ++i) ip += nv[static_cast<size_t>(i)] * embed[static_cast<size_t>(i * k + p)];
        for (int64_t i = 0; i < n; ++i) nv[static_cast<size_t>(i)] -= ip * embed[static_cast<size_t>(i * k + p)];
      }
      float nrm = 0.f;
      for (float x : nv) nrm += x * x;
      nrm = std::sqrt(std::max(nrm, 1e-12f));
      for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = nv[static_cast<size_t>(i)] / nrm;
    }
    for (int64_t i = 0; i < n; ++i) embed[static_cast<size_t>(i * k + c)] = v[static_cast<size_t>(i)];
  }
  std::vector<float> cents;
  kmeans_fit_impl(*rd(res), embed.data(), n, k, nclusters, 12, cents);
  std::vector<int64_t> labels(static_cast<size_t>(n));
  std::vector<float> dd(static_cast<size_t>(n));
  std::vector<float> sc(static_cast<size_t>(n) * static_cast<size_t>(nclusters));
  prim_pairwise(*rd(res), IOVS_METRIC_L2_EXPANDED, embed.data(), n, cents.data(), nclusters, k, sc.data(),
                2.f);
  prim_topk(*rd(res), sc.data(), n, nclusters, 1, labels.data(), dd.data(), false);
  auto* m = new Spectral();
  m->labels = std::move(labels);
  *model = reinterpret_cast<iovsSpectralModel_t>(m);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsSpectralLabels(iovsSpectralModel_t model, const int64_t** labels, int64_t* n) {
  if (!model || !labels || !n) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* m = reinterpret_cast<Spectral*>(model);
  *labels = m->labels.data();
  *n = static_cast<int64_t>(m->labels.size());
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsSpectralDestroy(iovsSpectralModel_t model) {
  delete reinterpret_cast<Spectral*>(model);
  return IOVS_STATUS_SUCCESS;
}

static iovsStatus spectral_embed_fill(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                      int32_t ncomp, int32_t knn, std::vector<float>& embed) {
  knn = std::max(1, std::min(knn, static_cast<int32_t>(n - 1)));
  ncomp = std::max(1, ncomp);
  std::vector<int64_t> nbr(static_cast<size_t>(n) * static_cast<size_t>(knn));
  std::vector<float> dist(static_cast<size_t>(n) * static_cast<size_t>(knn));
  const iovsStatus st =
      iovsAllNeighbors(res, dataset, n, dim, IOVS_METRIC_L2_EXPANDED, knn, nbr.data(), dist.data());
  if (st != IOVS_STATUS_SUCCESS) return st;
  float mean = 0.f;
  for (float v : dist) mean += v;
  mean /= static_cast<float>(std::max<size_t>(dist.size(), 1));
  const float sigma = std::max(mean, 1e-6f);
  std::vector<float> W(static_cast<size_t>(n * n), 0.f);
  std::vector<float> deg(static_cast<size_t>(n), 0.f);
  for (int64_t i = 0; i < n; ++i) {
    for (int32_t t = 0; t < knn; ++t) {
      const int64_t j = nbr[static_cast<size_t>(i * knn + t)];
      if (j < 0) continue;
      const float a = std::exp(-dist[static_cast<size_t>(i * knn + t)] / (2.f * sigma));
      W[static_cast<size_t>(i * n + j)] = std::max(W[static_cast<size_t>(i * n + j)], a);
      W[static_cast<size_t>(j * n + i)] = std::max(W[static_cast<size_t>(j * n + i)], a);
    }
  }
  for (int64_t i = 0; i < n; ++i) {
    float s = 0.f;
    for (int64_t j = 0; j < n; ++j) s += W[static_cast<size_t>(i * n + j)];
    deg[static_cast<size_t>(i)] = s;
  }
  embed.assign(static_cast<size_t>(n) * static_cast<size_t>(ncomp), 0.f);
  std::vector<float> L(static_cast<size_t>(n * n), 0.f);
  for (int64_t i = 0; i < n; ++i) {
    const float di = deg[static_cast<size_t>(i)];
    const float isi = di > 1e-12f ? 1.f / std::sqrt(di) : 0.f;
    for (int64_t j = 0; j < n; ++j) {
      const float dj = deg[static_cast<size_t>(j)];
      const float isj = dj > 1e-12f ? 1.f / std::sqrt(dj) : 0.f;
      L[static_cast<size_t>(i * n + j)] =
          (i == j ? 1.f : 0.f) - isi * W[static_cast<size_t>(i * n + j)] * isj;
    }
  }
  const bool syev_ok = mkl_syev_smallest(L.data(), n, ncomp, embed.data());
  auto rng = rng_from(5);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  for (int32_t c = 0; !syev_ok && c < ncomp; ++c) {
    std::vector<float> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = u(rng);
    for (int it = 0; it < 40; ++it) {
      std::vector<float> nv(static_cast<size_t>(n), 0.f);
      for (int64_t i = 0; i < n; ++i) {
        if (deg[static_cast<size_t>(i)] <= 1e-12f) continue;
        float s = 0.f;
        for (int64_t j = 0; j < n; ++j) s += W[static_cast<size_t>(i * n + j)] * v[static_cast<size_t>(j)];
        nv[static_cast<size_t>(i)] = s / deg[static_cast<size_t>(i)];
      }
      for (int32_t p = 0; p < c; ++p) {
        float ip = 0.f;
        for (int64_t i = 0; i < n; ++i) ip += nv[static_cast<size_t>(i)] * embed[static_cast<size_t>(i * ncomp + p)];
        for (int64_t i = 0; i < n; ++i) nv[static_cast<size_t>(i)] -= ip * embed[static_cast<size_t>(i * ncomp + p)];
      }
      float nrm = 0.f;
      for (float x : nv) nrm += x * x;
      nrm = std::sqrt(std::max(nrm, 1e-12f));
      for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = nv[static_cast<size_t>(i)] / nrm;
    }
    for (int64_t i = 0; i < n; ++i) embed[static_cast<size_t>(i * ncomp + c)] = v[static_cast<size_t>(i)];
  }
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsSpectralEmbedFit(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                                int32_t ncomp, int32_t knn, iovsSpectralEmbed_t* model) {
  if (!res || !dataset || !model || n <= 0 || dim <= 0 || ncomp <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* m = new SpectralEmbed();
  m->n = n;
  m->ncomp = ncomp;
  const iovsStatus st = spectral_embed_fill(res, dataset, n, dim, ncomp, knn, m->z);
  if (st != IOVS_STATUS_SUCCESS) {
    delete m;
    return st;
  }
  *model = reinterpret_cast<iovsSpectralEmbed_t>(m);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsSpectralEmbedData(iovsSpectralEmbed_t model, const float** data, int64_t* n, int32_t* ncomp) {
  if (!model || !data || !n || !ncomp) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* m = reinterpret_cast<SpectralEmbed*>(model);
  *data = m->z.data();
  *n = m->n;
  *ncomp = m->ncomp;
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsSpectralEmbedDestroy(iovsSpectralEmbed_t model) {
  delete reinterpret_cast<SpectralEmbed*>(model);
  return IOVS_STATUS_SUCCESS;
}
