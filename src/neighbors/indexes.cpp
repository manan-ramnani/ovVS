#include "internal.hpp"

using namespace iovs::impl;

namespace {

struct Dataset {
  UsmFloatVec x;
  int64_t n = 0;
  int64_t dim = 0;
  iovsMetric metric = IOVS_METRIC_L2_EXPANDED;
};

void copy_ds(Dataset& d, const float* x, int64_t n, int64_t dim, iovsMetric m) {
  d.n = n;
  d.dim = dim;
  d.metric = m;
  d.x.assign(x, x + n * dim);
}

struct BruteIndex : Dataset {
  iovsDType dtype = IOVS_DTYPE_F32;
};

struct IvfFlat {
  Dataset ds;
  int32_t nlist = 0;
  std::vector<float> centroids;
  std::vector<std::vector<int64_t>> lists;
};

struct IvfPq {
  Dataset ds;
  int32_t nlist = 0;
  int32_t pq_m = 0;
  int32_t pq_ks = 256;
  int32_t dsub = 0;
  std::vector<float> centroids;
  std::vector<int32_t> assign;
  std::vector<float> codebooks; /* [pq_m, pq_ks, dsub] */
  std::vector<uint8_t> codes;   /* [n, pq_m] */
  std::vector<std::vector<int64_t>> lists;
};

void build_ivf_lists(const int32_t* assign, int64_t n, int32_t nlist,
                     std::vector<std::vector<int64_t>>& lists) {
  lists.assign(static_cast<size_t>(nlist), {});
  for (int64_t i = 0; i < n; ++i) {
    const int32_t a = assign[i];
    if (a >= 0 && a < nlist) lists[static_cast<size_t>(a)].push_back(i);
  }
}

void pq_train(ResourcesData& r, const float* x, int64_t n, int64_t dim, int32_t pq_m, int32_t ks,
              std::vector<float>& codebooks) {
  const int32_t dsub = static_cast<int32_t>(dim / pq_m);
  codebooks.assign(static_cast<size_t>(pq_m) * ks * dsub, 0.f);
  std::vector<float> sub(static_cast<size_t>(n) * dsub);
  for (int32_t m = 0; m < pq_m; ++m) {
    for (int64_t i = 0; i < n; ++i) {
      std::memcpy(sub.data() + i * dsub, x + i * dim + static_cast<int64_t>(m) * dsub,
                  static_cast<size_t>(dsub) * sizeof(float));
    }
    std::vector<float> cents;
    kmeans_fit_impl(r, sub.data(), n, dsub, ks, 8, cents);
    std::memcpy(codebooks.data() + static_cast<size_t>(m) * ks * dsub, cents.data(),
                cents.size() * sizeof(float));
  }
}

void pq_encode(const float* x, int64_t n, int64_t dim, int32_t pq_m, int32_t ks, int32_t dsub,
               const float* codebooks, uint8_t* codes) {
  for (int64_t i = 0; i < n; ++i) {
    for (int32_t m = 0; m < pq_m; ++m) {
      const float* sub = x + i * dim + static_cast<int64_t>(m) * dsub;
      const float* cb = codebooks + static_cast<size_t>(m) * ks * dsub;
      int best = 0;
      float best_d = kInf;
      for (int32_t c = 0; c < ks; ++c) {
        const float d = l2sq(sub, cb + c * dsub, dsub);
        if (d < best_d) {
          best_d = d;
          best = c;
        }
      }
      codes[i * pq_m + m] = static_cast<uint8_t>(best);
    }
  }
}


}  // namespace

/* ---------------- brute ---------------- */

iovsStatus iovsBruteForceBuild(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                               iovsMetric metric, iovsBruteForceIndex_t* index) {
  return iovsBruteForceBuildTyped(res, dataset, n, dim, metric, IOVS_DTYPE_F32, index);
}

iovsStatus iovsBruteForceBuildTyped(iovsResources_t res, const void* dataset, int64_t n, int64_t dim,
                                    iovsMetric metric, iovsDType dtype, iovsBruteForceIndex_t* index) {
  if (!res || !dataset || !index || n <= 0 || dim <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* ix = new BruteIndex();
  ix->n = n;
  ix->dim = dim;
  ix->metric = metric;
  ix->dtype = dtype;
  convert_to_f32(dtype, dataset, n, dim, ix->x);
  *index = reinterpret_cast<iovsBruteForceIndex_t>(ix);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsBruteForceSearch(iovsResources_t res, iovsBruteForceIndex_t index, const float* queries,
                                int64_t nq, int64_t k, const uint8_t* bitset, int64_t* neighbors,
                                float* distances) {
  if (!res || !index || !queries || !neighbors || !distances || nq <= 0 || k <= 0) {
    return IOVS_STATUS_INVALID_ARGUMENT;
  }
  auto* ix = reinterpret_cast<BruteIndex*>(index);
  return brute_search_impl(*rd(res), ix->x.data(), ix->n, ix->dim, queries, nq, ix->metric, k, bitset,
                           neighbors, distances);
}

iovsStatus iovsBruteForceDestroy(iovsBruteForceIndex_t index) {
  delete reinterpret_cast<BruteIndex*>(index);
  return IOVS_STATUS_SUCCESS;
}

namespace iovs {
namespace impl {
int64_t brute_force_dim(iovsBruteForceIndex_t index) {
  if (!index) return 0;
  return reinterpret_cast<BruteIndex*>(index)->dim;
}
}  // namespace impl
}  // namespace iovs

/* ---------------- IVF-Flat ---------------- */

iovsStatus iovsIvfFlatBuild(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                            iovsMetric metric, int32_t nlist, iovsIvfFlatIndex_t* index) {
  if (!res || !dataset || !index || n <= 0 || dim <= 0 || nlist <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  nlist = std::min(nlist, static_cast<int32_t>(n));
  auto* ix = new IvfFlat();
  copy_ds(ix->ds, dataset, n, dim, metric);
  ix->nlist = nlist;
  kmeans_fit_impl(*rd(res), dataset, n, dim, nlist, 12, ix->centroids);
  std::vector<int64_t> labels(static_cast<size_t>(n));
  std::vector<float> d(static_cast<size_t>(n));
  std::vector<float> scores(static_cast<size_t>(n) * static_cast<size_t>(nlist));
  prim_pairwise(*rd(res), IOVS_METRIC_L2_EXPANDED, dataset, n, ix->centroids.data(), nlist, dim,
                scores.data(), 2.f);
  prim_topk(*rd(res), scores.data(), n, nlist, 1, labels.data(), d.data(), false);
  std::vector<int32_t> assign(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) assign[static_cast<size_t>(i)] = static_cast<int32_t>(labels[static_cast<size_t>(i)]);
  build_ivf_lists(assign.data(), n, nlist, ix->lists);
  *index = reinterpret_cast<iovsIvfFlatIndex_t>(ix);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsIvfFlatSearch(iovsResources_t res, iovsIvfFlatIndex_t index, const float* queries,
                             int64_t nq, int64_t k, int32_t nprobe, const uint8_t* bitset,
                             int64_t* neighbors, float* distances) {
  if (!res || !index || !queries || !neighbors || !distances) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<IvfFlat*>(index);
  nprobe = std::max(1, std::min(nprobe, ix->nlist));
  std::vector<int64_t> cl(static_cast<size_t>(nq) * static_cast<size_t>(nprobe));
  std::vector<float> cd(static_cast<size_t>(nq) * static_cast<size_t>(nprobe));
  std::vector<float> cscores(static_cast<size_t>(nq) * static_cast<size_t>(ix->nlist));
  prim_pairwise(*rd(res), IOVS_METRIC_L2_EXPANDED, queries, nq, ix->centroids.data(), ix->nlist,
                ix->ds.dim, cscores.data(), 2.f);
  prim_topk(*rd(res), cscores.data(), nq, ix->nlist, nprobe, cl.data(), cd.data(), false);
  const int64_t dim = ix->ds.dim;
  for (int64_t q = 0; q < nq; ++q) {
    std::vector<int64_t> ids;
    for (int32_t p = 0; p < nprobe; ++p) {
      const int64_t c = cl[q * nprobe + p];
      if (c < 0 || c >= ix->nlist) continue;
      ids.insert(ids.end(), ix->lists[static_cast<size_t>(c)].begin(),
                 ix->lists[static_cast<size_t>(c)].end());
    }
    if (ids.empty()) {
      for (int64_t t = 0; t < k; ++t) {
        neighbors[q * k + t] = -1;
        distances[q * k + t] = kInf;
      }
      continue;
    }
    std::vector<float> gathered(ids.size() * static_cast<size_t>(dim));
    prim_gather_rows(*rd(res), ix->ds.x.data(), ix->ds.n, dim, ids.data(),
                     static_cast<int64_t>(ids.size()), gathered.data());
    std::vector<float> sc(ids.size());
    prim_pairwise(*rd(res), ix->ds.metric, queries + q * dim, 1, gathered.data(),
                  static_cast<int64_t>(ids.size()), dim, sc.data(), 2.f);
    if (bitset) {
      for (size_t i = 0; i < ids.size(); ++i) {
        if (!allowed(bitset, ids[i])) sc[i] = kInf;
      }
    }
    const int64_t kk = std::min(k, static_cast<int64_t>(ids.size()));
    std::vector<int64_t> ti(static_cast<size_t>(kk));
    std::vector<float> tv(static_cast<size_t>(kk));
    prim_topk(*rd(res), sc.data(), 1, static_cast<int64_t>(ids.size()), kk, ti.data(), tv.data(),
              metric_largest(ix->ds.metric));
    for (int64_t t = 0; t < k; ++t) {
      if (t < kk) {
        neighbors[q * k + t] = ids[static_cast<size_t>(ti[static_cast<size_t>(t)])];
        float d = tv[static_cast<size_t>(t)];
        if (ix->ds.metric == IOVS_METRIC_INNER_PRODUCT) d = -d;
        distances[q * k + t] = d;
      } else {
        neighbors[q * k + t] = -1;
        distances[q * k + t] = kInf;
      }
    }
  }
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsIvfFlatDestroy(iovsIvfFlatIndex_t index) {
  delete reinterpret_cast<IvfFlat*>(index);
  return IOVS_STATUS_SUCCESS;
}

constexpr uint32_t kIvfFlatMagic = 0x31465649u; /* 'IVF1' */

iovsStatus iovsIvfFlatSerialize(iovsIvfFlatIndex_t index, const char* path) {
  if (!index || !path) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<IvfFlat*>(index);
  std::ofstream f(path, std::ios::binary);
  if (!f) return IOVS_STATUS_IO;
  f.write(reinterpret_cast<const char*>(&kIvfFlatMagic), 4);
  int32_t ver = 1;
  f.write(reinterpret_cast<const char*>(&ver), 4);
  f.write(reinterpret_cast<const char*>(&ix->ds.n), 8);
  f.write(reinterpret_cast<const char*>(&ix->ds.dim), 8);
  f.write(reinterpret_cast<const char*>(&ix->nlist), 4);
  int32_t metric = static_cast<int32_t>(ix->ds.metric);
  f.write(reinterpret_cast<const char*>(&metric), 4);
  f.write(reinterpret_cast<const char*>(ix->centroids.data()),
          static_cast<std::streamsize>(ix->centroids.size() * sizeof(float)));
  f.write(reinterpret_cast<const char*>(ix->ds.x.data()),
          static_cast<std::streamsize>(ix->ds.x.size() * sizeof(float)));
  for (int32_t c = 0; c < ix->nlist; ++c) {
    int32_t sz = static_cast<int32_t>(ix->lists[static_cast<size_t>(c)].size());
    f.write(reinterpret_cast<const char*>(&sz), 4);
    if (sz > 0) {
      f.write(reinterpret_cast<const char*>(ix->lists[static_cast<size_t>(c)].data()),
              static_cast<std::streamsize>(static_cast<size_t>(sz) * sizeof(int64_t)));
    }
  }
  return f.good() ? IOVS_STATUS_SUCCESS : IOVS_STATUS_IO;
}

iovsStatus iovsIvfFlatDeserialize(iovsResources_t res, const char* path, iovsIvfFlatIndex_t* index) {
  if (!res || !path || !index) return IOVS_STATUS_INVALID_ARGUMENT;
  std::ifstream f(path, std::ios::binary);
  if (!f) return IOVS_STATUS_IO;
  uint32_t magic = 0;
  f.read(reinterpret_cast<char*>(&magic), 4);
  if (magic != kIvfFlatMagic) return IOVS_STATUS_IO;
  int32_t ver = 0;
  f.read(reinterpret_cast<char*>(&ver), 4);
  auto* ix = new IvfFlat();
  f.read(reinterpret_cast<char*>(&ix->ds.n), 8);
  f.read(reinterpret_cast<char*>(&ix->ds.dim), 8);
  f.read(reinterpret_cast<char*>(&ix->nlist), 4);
  int32_t metric = 0;
  f.read(reinterpret_cast<char*>(&metric), 4);
  ix->ds.metric = static_cast<iovsMetric>(metric);
  ix->centroids.resize(static_cast<size_t>(ix->nlist) * static_cast<size_t>(ix->ds.dim));
  ix->ds.x.resize(static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->ds.dim));
  f.read(reinterpret_cast<char*>(ix->centroids.data()),
         static_cast<std::streamsize>(ix->centroids.size() * sizeof(float)));
  f.read(reinterpret_cast<char*>(ix->ds.x.data()),
         static_cast<std::streamsize>(ix->ds.x.size() * sizeof(float)));
  ix->lists.resize(static_cast<size_t>(ix->nlist));
  for (int32_t c = 0; c < ix->nlist; ++c) {
    int32_t sz = 0;
    f.read(reinterpret_cast<char*>(&sz), 4);
    ix->lists[static_cast<size_t>(c)].resize(static_cast<size_t>(std::max(sz, 0)));
    if (sz > 0) {
      f.read(reinterpret_cast<char*>(ix->lists[static_cast<size_t>(c)].data()),
             static_cast<std::streamsize>(static_cast<size_t>(sz) * sizeof(int64_t)));
    }
  }
  if (!f) {
    delete ix;
    return IOVS_STATUS_IO;
  }
  *index = reinterpret_cast<iovsIvfFlatIndex_t>(ix);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsIvfFlatExtend(iovsResources_t res, iovsIvfFlatIndex_t index, const float* extra,
                             int64_t nextra) {
  if (!res || !index || !extra || nextra <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<IvfFlat*>(index);
  const int64_t dim = ix->ds.dim;
  const int64_t old_n = ix->ds.n;
  ix->ds.x.insert(ix->ds.x.end(), extra, extra + nextra * dim);
  ix->ds.n += nextra;
  std::vector<int64_t> labels(static_cast<size_t>(nextra));
  std::vector<float> d(static_cast<size_t>(nextra));
  std::vector<float> scores(static_cast<size_t>(nextra) * static_cast<size_t>(ix->nlist));
  prim_pairwise(*rd(res), IOVS_METRIC_L2_EXPANDED, extra, nextra, ix->centroids.data(), ix->nlist, dim,
                scores.data(), 2.f);
  prim_topk(*rd(res), scores.data(), nextra, ix->nlist, 1, labels.data(), d.data(), false);
  for (int64_t i = 0; i < nextra; ++i) {
    const int32_t a = static_cast<int32_t>(labels[static_cast<size_t>(i)]);
    if (a >= 0 && a < ix->nlist) ix->lists[static_cast<size_t>(a)].push_back(old_n + i);
  }
  return IOVS_STATUS_SUCCESS;
}

/* ---------------- IVF-PQ ---------------- */

iovsStatus iovsIvfPqBuild(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                          iovsMetric metric, int32_t nlist, int32_t pq_m, int32_t pq_nbits,
                          iovsIvfPqIndex_t* index) {
  if (!res || !dataset || !index || n <= 0 || dim <= 0 || nlist <= 0 || pq_m <= 0) {
    return IOVS_STATUS_INVALID_ARGUMENT;
  }
  if (dim % pq_m != 0) return IOVS_STATUS_SHAPE_MISMATCH;
  nlist = std::min(nlist, static_cast<int32_t>(n));
  const int32_t ks = 1 << std::min(std::max(pq_nbits, 4), 8);
  auto* ix = new IvfPq();
  copy_ds(ix->ds, dataset, n, dim, metric);
  ix->nlist = nlist;
  ix->pq_m = pq_m;
  ix->pq_ks = ks;
  ix->dsub = static_cast<int32_t>(dim / pq_m);
  kmeans_fit_impl(*rd(res), dataset, n, dim, nlist, 12, ix->centroids);
  std::vector<int64_t> labels(static_cast<size_t>(n));
  std::vector<float> d(static_cast<size_t>(n));
  std::vector<float> scores(static_cast<size_t>(n) * static_cast<size_t>(nlist));
  prim_pairwise(*rd(res), IOVS_METRIC_L2_EXPANDED, dataset, n, ix->centroids.data(), nlist, dim,
                scores.data(), 2.f);
  prim_topk(*rd(res), scores.data(), n, nlist, 1, labels.data(), d.data(), false);
  ix->assign.resize(static_cast<size_t>(n));
  std::vector<float> resid(static_cast<size_t>(n) * static_cast<size_t>(dim));
  for (int64_t i = 0; i < n; ++i) {
    const int32_t a = static_cast<int32_t>(labels[static_cast<size_t>(i)]);
    ix->assign[static_cast<size_t>(i)] = a;
    const float* c = ix->centroids.data() + static_cast<size_t>(a) * dim;
    for (int64_t t = 0; t < dim; ++t) resid[static_cast<size_t>(i * dim + t)] = dataset[i * dim + t] - c[t];
  }
  build_ivf_lists(ix->assign.data(), n, nlist, ix->lists);
  pq_train(*rd(res), resid.data(), n, dim, pq_m, ks, ix->codebooks);
  ix->codes.resize(static_cast<size_t>(n) * static_cast<size_t>(pq_m));
  pq_encode(resid.data(), n, dim, pq_m, ks, ix->dsub, ix->codebooks.data(), ix->codes.data());
  *index = reinterpret_cast<iovsIvfPqIndex_t>(ix);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsIvfPqSearch(iovsResources_t res, iovsIvfPqIndex_t index, const float* queries,
                           int64_t nq, int64_t k, int32_t nprobe, int32_t krefine,
                           const uint8_t* bitset, int64_t* neighbors, float* distances) {
  if (!res || !index || !queries || !neighbors || !distances) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<IvfPq*>(index);
  nprobe = std::max(1, std::min(nprobe, ix->nlist));
  if (krefine < static_cast<int32_t>(k)) krefine = static_cast<int32_t>(k);
  std::vector<int64_t> cl(static_cast<size_t>(nq) * static_cast<size_t>(nprobe));
  std::vector<float> cd(static_cast<size_t>(nq) * static_cast<size_t>(nprobe));
  std::vector<float> cscores(static_cast<size_t>(nq) * static_cast<size_t>(ix->nlist));
  prim_pairwise(*rd(res), IOVS_METRIC_L2_EXPANDED, queries, nq, ix->centroids.data(), ix->nlist,
                ix->ds.dim, cscores.data(), 2.f);
  prim_topk(*rd(res), cscores.data(), nq, ix->nlist, nprobe, cl.data(), cd.data(), false);
  const int64_t dim = ix->ds.dim;
  std::vector<float> tables(static_cast<size_t>(ix->pq_m) * static_cast<size_t>(ix->pq_ks));
  for (int64_t q = 0; q < nq; ++q) {
    std::vector<int64_t> ids;
    std::vector<float> adc;
    const float* query = queries + q * dim;
    for (int32_t p = 0; p < nprobe; ++p) {
      const int64_t c = cl[q * nprobe + p];
      if (c < 0 || c >= ix->nlist) continue;
      const float* cent = ix->centroids.data() + c * dim;
      std::vector<float> qres(static_cast<size_t>(dim));
      for (int64_t t = 0; t < dim; ++t) qres[static_cast<size_t>(t)] = query[t] - cent[t];
      for (int32_t m = 0; m < ix->pq_m; ++m) {
        const float* sub = qres.data() + static_cast<size_t>(m) * ix->dsub;
        const float* cb = ix->codebooks.data() + static_cast<size_t>(m) * ix->pq_ks * ix->dsub;
        for (int32_t cs = 0; cs < ix->pq_ks; ++cs) {
          tables[static_cast<size_t>(m * ix->pq_ks + cs)] = l2sq(sub, cb + cs * ix->dsub, ix->dsub);
        }
      }
      std::vector<int64_t> list_ids;
      std::vector<uint8_t> list_codes;
      for (int64_t id : ix->lists[static_cast<size_t>(c)]) {
        if (bitset && !allowed(bitset, id)) continue;
        list_ids.push_back(id);
        const uint8_t* code = ix->codes.data() + id * ix->pq_m;
        list_codes.insert(list_codes.end(), code, code + ix->pq_m);
      }
      if (!list_ids.empty()) {
        std::vector<float> scores(list_ids.size());
        const iovsStatus as =
            prim_pq_adc(*rd(res), tables.data(), ix->pq_m, ix->pq_ks, list_codes.data(),
                        static_cast<int64_t>(list_ids.size()), scores.data());
        if (as != IOVS_STATUS_SUCCESS) return as;
        ids.insert(ids.end(), list_ids.begin(), list_ids.end());
        adc.insert(adc.end(), scores.begin(), scores.end());
      }
    }
    if (ids.empty()) {
      for (int64_t t = 0; t < k; ++t) {
        neighbors[q * k + t] = -1;
        distances[q * k + t] = kInf;
      }
      continue;
    }
    const int64_t kr = std::min(static_cast<int64_t>(krefine), static_cast<int64_t>(ids.size()));
    std::vector<int64_t> ti(static_cast<size_t>(kr));
    std::vector<float> tv(static_cast<size_t>(kr));
    prim_topk(*rd(res), adc.data(), 1, static_cast<int64_t>(ids.size()), kr, ti.data(), tv.data(), false);
    std::vector<int64_t> cand(static_cast<size_t>(kr));
    for (int64_t t = 0; t < kr; ++t) cand[static_cast<size_t>(t)] = ids[static_cast<size_t>(ti[static_cast<size_t>(t)])];
    std::vector<float> gathered(static_cast<size_t>(kr) * static_cast<size_t>(dim));
    prim_gather_rows(*rd(res), ix->ds.x.data(), ix->ds.n, dim, cand.data(), kr, gathered.data());
    std::vector<float> sc(static_cast<size_t>(kr));
    prim_pairwise(*rd(res), ix->ds.metric, query, 1, gathered.data(), kr, dim, sc.data(), 2.f);
    const int64_t kk = std::min(k, kr);
    std::vector<int64_t> fi(static_cast<size_t>(kk));
    std::vector<float> fv(static_cast<size_t>(kk));
    prim_topk(*rd(res), sc.data(), 1, kr, kk, fi.data(), fv.data(), metric_largest(ix->ds.metric));
    for (int64_t t = 0; t < k; ++t) {
      if (t < kk) {
        neighbors[q * k + t] = cand[static_cast<size_t>(fi[static_cast<size_t>(t)])];
        float d = fv[static_cast<size_t>(t)];
        if (ix->ds.metric == IOVS_METRIC_INNER_PRODUCT) d = -d;
        distances[q * k + t] = d;
      } else {
        neighbors[q * k + t] = -1;
        distances[q * k + t] = kInf;
      }
    }
  }
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsIvfPqDestroy(iovsIvfPqIndex_t index) {
  delete reinterpret_cast<IvfPq*>(index);
  return IOVS_STATUS_SUCCESS;
}

constexpr uint32_t kIvfPqMagic = 0x31515049u; /* 'IPQ1' */

iovsStatus iovsIvfPqSerialize(iovsIvfPqIndex_t index, const char* path) {
  if (!index || !path) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<IvfPq*>(index);
  std::ofstream f(path, std::ios::binary);
  if (!f) return IOVS_STATUS_IO;
  f.write(reinterpret_cast<const char*>(&kIvfPqMagic), 4);
  int32_t ver = 1;
  f.write(reinterpret_cast<const char*>(&ver), 4);
  f.write(reinterpret_cast<const char*>(&ix->ds.n), 8);
  f.write(reinterpret_cast<const char*>(&ix->ds.dim), 8);
  f.write(reinterpret_cast<const char*>(&ix->nlist), 4);
  f.write(reinterpret_cast<const char*>(&ix->pq_m), 4);
  f.write(reinterpret_cast<const char*>(&ix->pq_ks), 4);
  f.write(reinterpret_cast<const char*>(&ix->dsub), 4);
  int32_t metric = static_cast<int32_t>(ix->ds.metric);
  f.write(reinterpret_cast<const char*>(&metric), 4);
  f.write(reinterpret_cast<const char*>(ix->centroids.data()),
          static_cast<std::streamsize>(ix->centroids.size() * sizeof(float)));
  f.write(reinterpret_cast<const char*>(ix->codebooks.data()),
          static_cast<std::streamsize>(ix->codebooks.size() * sizeof(float)));
  f.write(reinterpret_cast<const char*>(ix->codes.data()), static_cast<std::streamsize>(ix->codes.size()));
  f.write(reinterpret_cast<const char*>(ix->ds.x.data()),
          static_cast<std::streamsize>(ix->ds.x.size() * sizeof(float)));
  f.write(reinterpret_cast<const char*>(ix->assign.data()),
          static_cast<std::streamsize>(ix->assign.size() * sizeof(int32_t)));
  for (int32_t c = 0; c < ix->nlist; ++c) {
    int32_t sz = static_cast<int32_t>(ix->lists[static_cast<size_t>(c)].size());
    f.write(reinterpret_cast<const char*>(&sz), 4);
    if (sz > 0)
      f.write(reinterpret_cast<const char*>(ix->lists[static_cast<size_t>(c)].data()),
              static_cast<std::streamsize>(static_cast<size_t>(sz) * sizeof(int64_t)));
  }
  return f.good() ? IOVS_STATUS_SUCCESS : IOVS_STATUS_IO;
}

iovsStatus iovsIvfPqDeserialize(iovsResources_t res, const char* path, iovsIvfPqIndex_t* index) {
  if (!res || !path || !index) return IOVS_STATUS_INVALID_ARGUMENT;
  std::ifstream f(path, std::ios::binary);
  if (!f) return IOVS_STATUS_IO;
  uint32_t magic = 0;
  f.read(reinterpret_cast<char*>(&magic), 4);
  if (magic != kIvfPqMagic) return IOVS_STATUS_IO;
  int32_t ver = 0;
  f.read(reinterpret_cast<char*>(&ver), 4);
  auto* ix = new IvfPq();
  f.read(reinterpret_cast<char*>(&ix->ds.n), 8);
  f.read(reinterpret_cast<char*>(&ix->ds.dim), 8);
  f.read(reinterpret_cast<char*>(&ix->nlist), 4);
  f.read(reinterpret_cast<char*>(&ix->pq_m), 4);
  f.read(reinterpret_cast<char*>(&ix->pq_ks), 4);
  f.read(reinterpret_cast<char*>(&ix->dsub), 4);
  int32_t metric = 0;
  f.read(reinterpret_cast<char*>(&metric), 4);
  ix->ds.metric = static_cast<iovsMetric>(metric);
  ix->centroids.resize(static_cast<size_t>(ix->nlist) * static_cast<size_t>(ix->ds.dim));
  ix->codebooks.resize(static_cast<size_t>(ix->pq_m) * ix->pq_ks * ix->dsub);
  ix->codes.resize(static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->pq_m));
  ix->ds.x.resize(static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->ds.dim));
  ix->assign.resize(static_cast<size_t>(ix->ds.n));
  f.read(reinterpret_cast<char*>(ix->centroids.data()),
         static_cast<std::streamsize>(ix->centroids.size() * sizeof(float)));
  f.read(reinterpret_cast<char*>(ix->codebooks.data()),
         static_cast<std::streamsize>(ix->codebooks.size() * sizeof(float)));
  f.read(reinterpret_cast<char*>(ix->codes.data()), static_cast<std::streamsize>(ix->codes.size()));
  f.read(reinterpret_cast<char*>(ix->ds.x.data()),
         static_cast<std::streamsize>(ix->ds.x.size() * sizeof(float)));
  f.read(reinterpret_cast<char*>(ix->assign.data()),
         static_cast<std::streamsize>(ix->assign.size() * sizeof(int32_t)));
  ix->lists.resize(static_cast<size_t>(ix->nlist));
  for (int32_t c = 0; c < ix->nlist; ++c) {
    int32_t sz = 0;
    f.read(reinterpret_cast<char*>(&sz), 4);
    ix->lists[static_cast<size_t>(c)].resize(static_cast<size_t>(std::max(sz, 0)));
    if (sz > 0)
      f.read(reinterpret_cast<char*>(ix->lists[static_cast<size_t>(c)].data()),
             static_cast<std::streamsize>(static_cast<size_t>(sz) * sizeof(int64_t)));
  }
  if (!f) {
    delete ix;
    return IOVS_STATUS_IO;
  }
  *index = reinterpret_cast<iovsIvfPqIndex_t>(ix);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsIvfPqExtend(iovsResources_t res, iovsIvfPqIndex_t index, const float* extra, int64_t nextra) {
  if (!res || !index || !extra || nextra <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<IvfPq*>(index);
  const int64_t dim = ix->ds.dim;
  const int64_t old_n = ix->ds.n;
  ix->ds.x.insert(ix->ds.x.end(), extra, extra + nextra * dim);
  ix->ds.n += nextra;
  std::vector<int64_t> labels(static_cast<size_t>(nextra));
  std::vector<float> d(static_cast<size_t>(nextra));
  std::vector<float> scores(static_cast<size_t>(nextra) * static_cast<size_t>(ix->nlist));
  prim_pairwise(*rd(res), IOVS_METRIC_L2_EXPANDED, extra, nextra, ix->centroids.data(), ix->nlist, dim,
                scores.data(), 2.f);
  prim_topk(*rd(res), scores.data(), nextra, ix->nlist, 1, labels.data(), d.data(), false);
  std::vector<float> resid(static_cast<size_t>(nextra) * static_cast<size_t>(dim));
  ix->assign.resize(static_cast<size_t>(ix->ds.n));
  for (int64_t i = 0; i < nextra; ++i) {
    const int32_t a = static_cast<int32_t>(labels[static_cast<size_t>(i)]);
    ix->assign[static_cast<size_t>(old_n + i)] = a;
    if (a >= 0 && a < ix->nlist) ix->lists[static_cast<size_t>(a)].push_back(old_n + i);
    const float* c = ix->centroids.data() + static_cast<size_t>(std::max(a, 0)) * dim;
    for (int64_t t = 0; t < dim; ++t) resid[static_cast<size_t>(i * dim + t)] = extra[i * dim + t] - c[t];
  }
  ix->codes.resize(static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->pq_m));
  pq_encode(resid.data(), nextra, dim, ix->pq_m, ix->pq_ks, ix->dsub, ix->codebooks.data(),
            ix->codes.data() + old_n * ix->pq_m);
  return IOVS_STATUS_SUCCESS;
}
