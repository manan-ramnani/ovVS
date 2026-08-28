#include "internal.hpp"

using namespace ovvs::impl;

namespace {

struct Dataset {
  UsmFloatVec x;
  int64_t n = 0;
  int64_t dim = 0;
  ovvsMetric metric = OVVS_METRIC_L2_EXPANDED;
};

void copy_ds(Dataset& d, const float* x, int64_t n, int64_t dim, ovvsMetric m) {
  d.n = n;
  d.dim = dim;
  d.metric = m;
  d.x.assign(x, x + n * dim);
}

struct BruteIndex : Dataset {
  ovvsDType dtype = OVVS_DTYPE_F32;
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
  std::vector<int64_t> list_offsets; /* [nlist + 1] */
  std::vector<int64_t> packed_ids;   /* list-major [n] */
  std::vector<uint8_t> packed_codes; /* list-major [n, pq_m] */
};

void build_ivf_lists(const int32_t* assign, int64_t n, int32_t nlist,
                     std::vector<std::vector<int64_t>>& lists) {
  lists.assign(static_cast<size_t>(nlist), {});
  for (int64_t i = 0; i < n; ++i) {
    const int32_t a = assign[i];
    if (a >= 0 && a < nlist) lists[static_cast<size_t>(a)].push_back(i);
  }
}

ovvsStatus make_ivf_pq_packed(int64_t n, int32_t nlist, int32_t pq_m, int32_t pq_ks,
                              const std::vector<int32_t>& assign,
                              const std::vector<uint8_t>& codes,
                              const std::vector<std::vector<int64_t>>& lists,
                              std::vector<int64_t>& list_offsets,
                              std::vector<int64_t>& packed_ids,
                              std::vector<uint8_t>& packed_codes) {
  if (n <= 0 || nlist <= 0 || pq_m <= 0 || pq_ks <= 0 || pq_ks > 256 ||
      lists.size() != static_cast<size_t>(nlist) ||
      assign.size() != static_cast<size_t>(n) ||
      static_cast<uint64_t>(n) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
              static_cast<uint64_t>(pq_m) ||
      codes.size() != static_cast<size_t>(n) * static_cast<size_t>(pq_m)) {
    return OVVS_STATUS_ERROR;
  }

  std::vector<int64_t> next_offsets(static_cast<size_t>(nlist) + 1, 0);
  uint64_t total = 0;
  for (int32_t c = 0; c < nlist; ++c) {
    const size_t list_size = lists[static_cast<size_t>(c)].size();
    if (static_cast<uint64_t>(list_size) > static_cast<uint64_t>(n) - total) {
      return OVVS_STATUS_ERROR;
    }
    total += static_cast<uint64_t>(list_size);
    next_offsets[static_cast<size_t>(c + 1)] = static_cast<int64_t>(total);
  }
  if (total != static_cast<uint64_t>(n)) return OVVS_STATUS_ERROR;

  std::vector<int64_t> next_ids(static_cast<size_t>(n));
  std::vector<uint8_t> next_codes(static_cast<size_t>(n) * static_cast<size_t>(pq_m));
  std::vector<uint8_t> seen(static_cast<size_t>(n), 0);
  size_t position = 0;
  for (int32_t c = 0; c < nlist; ++c) {
    for (int64_t id : lists[static_cast<size_t>(c)]) {
      if (id < 0 || id >= n || seen[static_cast<size_t>(id)] != 0 ||
          assign[static_cast<size_t>(id)] != c) {
        return OVVS_STATUS_ERROR;
      }
      const uint8_t* source = codes.data() + static_cast<size_t>(id) * static_cast<size_t>(pq_m);
      for (int32_t m = 0; m < pq_m; ++m) {
        if (source[static_cast<size_t>(m)] >= pq_ks) return OVVS_STATUS_ERROR;
      }
      seen[static_cast<size_t>(id)] = 1;
      next_ids[position] = id;
      std::memcpy(next_codes.data() + position * static_cast<size_t>(pq_m), source,
                  static_cast<size_t>(pq_m));
      ++position;
    }
  }

  list_offsets.swap(next_offsets);
  packed_ids.swap(next_ids);
  packed_codes.swap(next_codes);
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus rebuild_ivf_pq_packed(IvfPq& ix) {
  return make_ivf_pq_packed(ix.ds.n, ix.nlist, ix.pq_m, ix.pq_ks, ix.assign, ix.codes,
                            ix.lists, ix.list_offsets, ix.packed_ids, ix.packed_codes);
}

void saturating_add(int64_t& value, int64_t delta) noexcept {
  if (delta <= 0) return;
  if (value > std::numeric_limits<int64_t>::max() - delta) {
    value = std::numeric_limits<int64_t>::max();
  } else {
    value += delta;
  }
}

void record_ivf_pq_rebuild(ResourcesData& resources, int64_t rows) noexcept {
  try {
    std::lock_guard<std::mutex> lock(resources.ivfpq_stats_mutex);
    saturating_add(resources.ivfpq_packed_rebuilds, 1);
    saturating_add(resources.ivfpq_packed_rebuild_rows, rows);
  } catch (...) {
  }
}

void record_ivf_pq_search_layout(ResourcesData& resources, int64_t direct_rows,
                                 int64_t filtered_copy_bytes) noexcept {
  try {
    std::lock_guard<std::mutex> lock(resources.ivfpq_stats_mutex);
    saturating_add(resources.ivfpq_unfiltered_direct_rows, direct_rows);
    saturating_add(resources.ivfpq_filtered_code_copy_bytes, filtered_copy_bytes);
  } catch (...) {
  }
}

ovvsStatus pq_train(ResourcesData& r, const float* x, int64_t n, int64_t dim, int32_t pq_m,
                    int32_t ks, std::vector<float>& codebooks) {
  const int32_t dsub = static_cast<int32_t>(dim / pq_m);
  codebooks.assign(static_cast<size_t>(pq_m) * ks * dsub, 0.f);
  std::vector<float> sub(static_cast<size_t>(n) * dsub);
  for (int32_t m = 0; m < pq_m; ++m) {
    for (int64_t i = 0; i < n; ++i) {
      std::memcpy(sub.data() + i * dsub, x + i * dim + static_cast<int64_t>(m) * dsub,
                  static_cast<size_t>(dsub) * sizeof(float));
    }
    std::vector<float> cents;
    const ovvsStatus status = kmeans_fit_impl(r, sub.data(), n, dsub, ks, 8, cents);
    if (status != OVVS_STATUS_SUCCESS) return status;
    std::memcpy(codebooks.data() + static_cast<size_t>(m) * ks * dsub, cents.data(),
                cents.size() * sizeof(float));
  }
  return OVVS_STATUS_SUCCESS;
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

ovvsStatus ovvsBruteForceBuild(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                               ovvsMetric metric, ovvsBruteForceIndex_t* index) {
  return ovvsBruteForceBuildTyped(res, dataset, n, dim, metric, OVVS_DTYPE_F32, index);
}

ovvsStatus ovvsBruteForceBuildTyped(ovvsResources_t res, const void* dataset, int64_t n, int64_t dim,
                                    ovvsMetric metric, ovvsDType dtype, ovvsBruteForceIndex_t* index) {
  if (!res || !dataset || !index || n <= 0 || dim <= 0) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* ix = new BruteIndex();
  ix->n = n;
  ix->dim = dim;
  ix->metric = metric;
  ix->dtype = dtype;
  convert_to_f32(dtype, dataset, n, dim, ix->x);
  *index = reinterpret_cast<ovvsBruteForceIndex_t>(ix);
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsBruteForceSearch(ovvsResources_t res, ovvsBruteForceIndex_t index, const float* queries,
                                int64_t nq, int64_t k, const uint8_t* bitset, int64_t* neighbors,
                                float* distances) {
  if (!res || !index || !queries || !neighbors || !distances || nq <= 0 || k <= 0) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  auto* ix = reinterpret_cast<BruteIndex*>(index);
  return brute_search_impl(*rd(res), ix->x.data(), ix->n, ix->dim, queries, nq, ix->metric, k, bitset,
                           neighbors, distances);
}

ovvsStatus ovvsBruteForceDestroy(ovvsBruteForceIndex_t index) {
  delete reinterpret_cast<BruteIndex*>(index);
  return OVVS_STATUS_SUCCESS;
}

namespace ovvs {
namespace impl {
int64_t brute_force_dim(ovvsBruteForceIndex_t index) {
  if (!index) return 0;
  return reinterpret_cast<BruteIndex*>(index)->dim;
}
}  // namespace impl
}  // namespace ovvs

/* ---------------- IVF-Flat ---------------- */

ovvsStatus ovvsIvfFlatBuild(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                            ovvsMetric metric, int32_t nlist, ovvsIvfFlatIndex_t* index) {
  if (!res || !dataset || !index || n <= 0 || dim <= 0 || nlist <= 0) return OVVS_STATUS_INVALID_ARGUMENT;
  nlist = std::min(nlist, static_cast<int32_t>(n));
  auto* ix = new IvfFlat();
  copy_ds(ix->ds, dataset, n, dim, metric);
  ix->nlist = nlist;
  kmeans_fit_impl(*rd(res), dataset, n, dim, nlist, 12, ix->centroids);
  std::vector<int64_t> labels(static_cast<size_t>(n));
  std::vector<float> d(static_cast<size_t>(n));
  std::vector<float> scores(static_cast<size_t>(n) * static_cast<size_t>(nlist));
  prim_pairwise(*rd(res), OVVS_METRIC_L2_EXPANDED, dataset, n, ix->centroids.data(), nlist, dim,
                scores.data(), 2.f);
  prim_topk(*rd(res), scores.data(), n, nlist, 1, labels.data(), d.data(), false);
  std::vector<int32_t> assign(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) assign[static_cast<size_t>(i)] = static_cast<int32_t>(labels[static_cast<size_t>(i)]);
  build_ivf_lists(assign.data(), n, nlist, ix->lists);
  *index = reinterpret_cast<ovvsIvfFlatIndex_t>(ix);
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsIvfFlatSearch(ovvsResources_t res, ovvsIvfFlatIndex_t index, const float* queries,
                             int64_t nq, int64_t k, int32_t nprobe, const uint8_t* bitset,
                             int64_t* neighbors, float* distances) {
  if (!res || !index || !queries || !neighbors || !distances) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<IvfFlat*>(index);
  nprobe = std::max(1, std::min(nprobe, ix->nlist));
  std::vector<int64_t> cl(static_cast<size_t>(nq) * static_cast<size_t>(nprobe));
  std::vector<float> cd(static_cast<size_t>(nq) * static_cast<size_t>(nprobe));
  std::vector<float> cscores(static_cast<size_t>(nq) * static_cast<size_t>(ix->nlist));
  prim_pairwise(*rd(res), OVVS_METRIC_L2_EXPANDED, queries, nq, ix->centroids.data(), ix->nlist,
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
        if (ix->ds.metric == OVVS_METRIC_INNER_PRODUCT) d = -d;
        distances[q * k + t] = d;
      } else {
        neighbors[q * k + t] = -1;
        distances[q * k + t] = kInf;
      }
    }
  }
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsIvfFlatDestroy(ovvsIvfFlatIndex_t index) {
  delete reinterpret_cast<IvfFlat*>(index);
  return OVVS_STATUS_SUCCESS;
}

constexpr uint32_t kIvfFlatMagic = 0x31465649u; /* 'IVF1' */

ovvsStatus ovvsIvfFlatSerialize(ovvsIvfFlatIndex_t index, const char* path) {
  if (!index || !path) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<IvfFlat*>(index);
  std::ofstream f(path, std::ios::binary);
  if (!f) return OVVS_STATUS_IO;
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
  return f.good() ? OVVS_STATUS_SUCCESS : OVVS_STATUS_IO;
}

ovvsStatus ovvsIvfFlatDeserialize(ovvsResources_t res, const char* path, ovvsIvfFlatIndex_t* index) {
  if (!res || !path || !index) return OVVS_STATUS_INVALID_ARGUMENT;
  std::ifstream f(path, std::ios::binary);
  if (!f) return OVVS_STATUS_IO;
  uint32_t magic = 0;
  f.read(reinterpret_cast<char*>(&magic), 4);
  if (magic != kIvfFlatMagic) return OVVS_STATUS_IO;
  int32_t ver = 0;
  f.read(reinterpret_cast<char*>(&ver), 4);
  auto* ix = new IvfFlat();
  f.read(reinterpret_cast<char*>(&ix->ds.n), 8);
  f.read(reinterpret_cast<char*>(&ix->ds.dim), 8);
  f.read(reinterpret_cast<char*>(&ix->nlist), 4);
  int32_t metric = 0;
  f.read(reinterpret_cast<char*>(&metric), 4);
  ix->ds.metric = static_cast<ovvsMetric>(metric);
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
    return OVVS_STATUS_IO;
  }
  *index = reinterpret_cast<ovvsIvfFlatIndex_t>(ix);
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsIvfFlatExtend(ovvsResources_t res, ovvsIvfFlatIndex_t index, const float* extra,
                             int64_t nextra) {
  if (!res || !index || !extra || nextra <= 0) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<IvfFlat*>(index);
  const int64_t dim = ix->ds.dim;
  const int64_t old_n = ix->ds.n;
  ix->ds.x.insert(ix->ds.x.end(), extra, extra + nextra * dim);
  ix->ds.n += nextra;
  std::vector<int64_t> labels(static_cast<size_t>(nextra));
  std::vector<float> d(static_cast<size_t>(nextra));
  std::vector<float> scores(static_cast<size_t>(nextra) * static_cast<size_t>(ix->nlist));
  prim_pairwise(*rd(res), OVVS_METRIC_L2_EXPANDED, extra, nextra, ix->centroids.data(), ix->nlist, dim,
                scores.data(), 2.f);
  prim_topk(*rd(res), scores.data(), nextra, ix->nlist, 1, labels.data(), d.data(), false);
  for (int64_t i = 0; i < nextra; ++i) {
    const int32_t a = static_cast<int32_t>(labels[static_cast<size_t>(i)]);
    if (a >= 0 && a < ix->nlist) ix->lists[static_cast<size_t>(a)].push_back(old_n + i);
  }
  return OVVS_STATUS_SUCCESS;
}

/* ---------------- IVF-PQ ---------------- */

ovvsStatus ovvsIvfPqBuild(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                          ovvsMetric metric, int32_t nlist, int32_t pq_m, int32_t pq_nbits,
                          ovvsIvfPqIndex_t* index) {
  if (index) *index = nullptr;
  if (!res || !dataset || !index || n <= 0 || dim <= 0 || nlist <= 0 || pq_m <= 0) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  if (metric < OVVS_METRIC_L2_EXPANDED || metric > OVVS_METRIC_LP_UNEXPANDED) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  if (dim % pq_m != 0) return OVVS_STATUS_SHAPE_MISMATCH;
  if (n > std::numeric_limits<int32_t>::max() ||
      dim > std::numeric_limits<int64_t>::max() / n) {
    return OVVS_STATUS_SHAPE_MISMATCH;
  }
  nlist = std::min(nlist, static_cast<int32_t>(n));
  const int32_t ks = 1 << std::min(std::max(pq_nbits, 4), 8);
  const size_t max_float_elements = std::numeric_limits<size_t>::max() / sizeof(float);
  if (static_cast<uint64_t>(n) >
          static_cast<uint64_t>(max_float_elements) / static_cast<uint64_t>(dim) ||
      static_cast<uint64_t>(n) >
          static_cast<uint64_t>(max_float_elements) / static_cast<uint64_t>(nlist) ||
      static_cast<uint64_t>(n) >
          static_cast<uint64_t>(max_float_elements) / static_cast<uint64_t>(ks) ||
      static_cast<uint64_t>(dim) >
          static_cast<uint64_t>(max_float_elements) / static_cast<uint64_t>(ks) ||
      static_cast<uint64_t>(n) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
              static_cast<uint64_t>(pq_m)) {
    return OVVS_STATUS_SHAPE_MISMATCH;
  }
  try {
    auto ix = std::make_unique<IvfPq>();
    copy_ds(ix->ds, dataset, n, dim, metric);
    ix->nlist = nlist;
    ix->pq_m = pq_m;
    ix->pq_ks = ks;
    ix->dsub = static_cast<int32_t>(dim / pq_m);
    ovvsStatus status =
        kmeans_fit_impl(*rd(res), dataset, n, dim, nlist, 12, ix->centroids);
    if (status != OVVS_STATUS_SUCCESS) return status;
    std::vector<int64_t> labels(static_cast<size_t>(n));
    std::vector<float> d(static_cast<size_t>(n));
    std::vector<float> scores(static_cast<size_t>(n) * static_cast<size_t>(nlist));
    status = prim_pairwise(*rd(res), OVVS_METRIC_L2_EXPANDED, dataset, n,
                           ix->centroids.data(), nlist, dim, scores.data(), 2.f);
    if (status != OVVS_STATUS_SUCCESS) return status;
    status = prim_topk(*rd(res), scores.data(), n, nlist, 1, labels.data(), d.data(), false);
    if (status != OVVS_STATUS_SUCCESS) return status;
    ix->assign.resize(static_cast<size_t>(n));
    std::vector<float> resid(static_cast<size_t>(n) * static_cast<size_t>(dim));
    for (int64_t i = 0; i < n; ++i) {
      const int64_t label = labels[static_cast<size_t>(i)];
      if (label < 0 || label >= nlist) return OVVS_STATUS_ERROR;
      const int32_t a = static_cast<int32_t>(label);
      ix->assign[static_cast<size_t>(i)] = a;
      const float* c = ix->centroids.data() + static_cast<size_t>(a) * dim;
      for (int64_t t = 0; t < dim; ++t) {
        resid[static_cast<size_t>(i * dim + t)] = dataset[i * dim + t] - c[t];
      }
    }
    build_ivf_lists(ix->assign.data(), n, nlist, ix->lists);
    status = pq_train(*rd(res), resid.data(), n, dim, pq_m, ks, ix->codebooks);
    if (status != OVVS_STATUS_SUCCESS) return status;
    ix->codes.resize(static_cast<size_t>(n) * static_cast<size_t>(pq_m));
    pq_encode(resid.data(), n, dim, pq_m, ks, ix->dsub, ix->codebooks.data(),
              ix->codes.data());
    status = rebuild_ivf_pq_packed(*ix);
    if (status != OVVS_STATUS_SUCCESS) return status;
    *index = reinterpret_cast<ovvsIvfPqIndex_t>(ix.release());
    record_ivf_pq_rebuild(*rd(res), n);
    return OVVS_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return OVVS_STATUS_OOM;
  } catch (...) {
    return OVVS_STATUS_ERROR;
  }
}

ovvsStatus ovvsIvfPqSearch(ovvsResources_t res, ovvsIvfPqIndex_t index, const float* queries,
                           int64_t nq, int64_t k, int32_t nprobe, int32_t krefine,
                           const uint8_t* bitset, int64_t* neighbors, float* distances) {
  if (!res || !index || !queries || !neighbors || !distances || nq <= 0 || k <= 0 ||
      k > std::numeric_limits<int32_t>::max()) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  if (nq > std::numeric_limits<int64_t>::max() / k ||
      static_cast<uint64_t>(nq) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(int64_t)) /
              static_cast<uint64_t>(k)) {
    return OVVS_STATUS_SHAPE_MISMATCH;
  }
  auto* ix = reinterpret_cast<IvfPq*>(index);
  if (ix->ds.n <= 0 || ix->ds.dim <= 0 || ix->nlist <= 0 || ix->pq_m <= 0 ||
      ix->pq_ks <= 0 ||
      ix->list_offsets.size() != static_cast<size_t>(ix->nlist) + 1 ||
      ix->list_offsets.front() != 0 || ix->list_offsets.back() != ix->ds.n ||
      ix->packed_ids.size() != static_cast<size_t>(ix->ds.n) ||
      static_cast<uint64_t>(ix->ds.n) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
              static_cast<uint64_t>(ix->pq_m) ||
      ix->packed_codes.size() !=
          static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->pq_m) ||
      static_cast<uint64_t>(nq) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(float)) /
              static_cast<uint64_t>(ix->ds.dim) ||
      static_cast<uint64_t>(nq) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(float)) /
              static_cast<uint64_t>(ix->nlist) ||
      static_cast<uint64_t>(nq) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(int64_t)) /
              static_cast<uint64_t>(ix->nlist)) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  nprobe = std::max(1, std::min(nprobe, ix->nlist));
  if (krefine < static_cast<int32_t>(k)) krefine = static_cast<int32_t>(k);
  try {
    int64_t direct_rows = 0;
    int64_t filtered_copy_bytes = 0;
    std::vector<int64_t> cl(static_cast<size_t>(nq) * static_cast<size_t>(nprobe));
    std::vector<float> cd(static_cast<size_t>(nq) * static_cast<size_t>(nprobe));
    std::vector<float> cscores(static_cast<size_t>(nq) * static_cast<size_t>(ix->nlist));
    ovvsStatus status =
        prim_pairwise(*rd(res), OVVS_METRIC_L2_EXPANDED, queries, nq,
                      ix->centroids.data(), ix->nlist, ix->ds.dim, cscores.data(), 2.f);
    if (status != OVVS_STATUS_SUCCESS) return status;
    status = prim_topk(*rd(res), cscores.data(), nq, ix->nlist, nprobe, cl.data(),
                       cd.data(), false);
    if (status != OVVS_STATUS_SUCCESS) return status;

    const int64_t dim = ix->ds.dim;
    const size_t table_elements =
        static_cast<size_t>(ix->pq_m) * static_cast<size_t>(ix->pq_ks);
    std::vector<int64_t> query_raw_rows(static_cast<size_t>(nq), 0);
    for (int64_t q = 0; q < nq; ++q) {
      int64_t rows = 0;
      for (int32_t p = 0; p < nprobe; ++p) {
        const int64_t c = cl[q * nprobe + p];
        if (c < 0 || c >= ix->nlist) return OVVS_STATUS_ERROR;
        const int64_t list_begin = ix->list_offsets[static_cast<size_t>(c)];
        const int64_t list_end = ix->list_offsets[static_cast<size_t>(c + 1)];
        if (list_begin < 0 || list_end < list_begin ||
            list_end > static_cast<int64_t>(ix->packed_ids.size())) {
          return OVVS_STATUS_ERROR;
        }
        const int64_t list_size = list_end - list_begin;
        if (list_size > ix->ds.n - rows) return OVVS_STATUS_ERROR;
        rows += list_size;
      }
      query_raw_rows[static_cast<size_t>(q)] = rows;
    }

    const size_t output_elements = static_cast<size_t>(nq) * static_cast<size_t>(k);
    std::vector<int64_t> staged_neighbors(output_elements, -1);
    std::vector<float> staged_distances(output_elements, kInf);

    struct AdcPlan {
      int64_t query = 0;
      int32_t cluster = 0;
      int64_t list_begin = 0;
      int64_t list_end = 0;
      int64_t rows = 0;
      int64_t output_offset = 0;
    };

    std::vector<AdcPlan> plans;
    std::vector<PqAdcTask> tasks;
    std::vector<int64_t> query_offsets;
    std::vector<int64_t> candidate_ids;
    std::vector<float> adc;
    std::vector<float> lut_storage;
    std::vector<uint8_t> filtered_codes;
    std::vector<float> qres(static_cast<size_t>(dim));

    std::vector<int64_t> topk_indices;
    std::vector<float> topk_values;
    std::vector<int64_t> candidates;
    std::vector<float> gathered;
    std::vector<float> refined_scores;
    std::vector<int64_t> final_indices;
    std::vector<float> final_values;

    constexpr int64_t kMaxQueriesPerBlock = 32;
    for (int64_t block_begin = 0; block_begin < nq;) {
      int64_t block_end = block_begin;
      int64_t block_raw_rows = 0;
      while (block_end < nq && block_end - block_begin < kMaxQueriesPerBlock) {
        const int64_t query_rows = query_raw_rows[static_cast<size_t>(block_end)];
        if (block_end != block_begin && query_rows > ix->ds.n - block_raw_rows) break;
        block_raw_rows += query_rows;
        ++block_end;
        if (block_raw_rows >= ix->ds.n) break;
      }
      if (block_end == block_begin) return OVVS_STATUS_ERROR;

      const int64_t block_queries = block_end - block_begin;
      plans.clear();
      query_offsets.assign(static_cast<size_t>(block_queries) + 1, 0);
      int64_t candidate_rows = 0;
      for (int64_t q = block_begin; q < block_end; ++q) {
        for (int32_t p = 0; p < nprobe; ++p) {
          const int64_t c = cl[q * nprobe + p];
          if (c < 0 || c >= ix->nlist) return OVVS_STATUS_ERROR;
          const int64_t list_begin = ix->list_offsets[static_cast<size_t>(c)];
          const int64_t list_end = ix->list_offsets[static_cast<size_t>(c + 1)];
          if (list_begin < 0 || list_end < list_begin ||
              list_end > static_cast<int64_t>(ix->packed_ids.size())) {
            return OVVS_STATUS_ERROR;
          }

          int64_t accepted_rows = list_end - list_begin;
          if (bitset && accepted_rows > 0) {
            accepted_rows = 0;
            for (int64_t position = list_begin; position < list_end; ++position) {
              const int64_t id = ix->packed_ids[static_cast<size_t>(position)];
              if (id < 0 || id >= ix->ds.n) return OVVS_STATUS_ERROR;
              if (allowed(bitset, id)) ++accepted_rows;
            }
          }
          if (accepted_rows <= 0) continue;
          if (accepted_rows > ix->ds.n - candidate_rows) return OVVS_STATUS_ERROR;
          plans.push_back(
              {q, static_cast<int32_t>(c), list_begin, list_end, accepted_rows, candidate_rows});
          candidate_rows += accepted_rows;
        }
        query_offsets[static_cast<size_t>(q - block_begin + 1)] = candidate_rows;
      }

      if (plans.size() > std::numeric_limits<size_t>::max() / table_elements) {
        return OVVS_STATUS_SHAPE_MISMATCH;
      }
      const size_t candidate_count = static_cast<size_t>(candidate_rows);
      if (candidate_count > std::numeric_limits<size_t>::max() /
                                static_cast<size_t>(ix->pq_m)) {
        return OVVS_STATUS_SHAPE_MISMATCH;
      }

      candidate_ids.resize(candidate_count);
      adc.resize(candidate_count);
      tasks.resize(plans.size());
      lut_storage.resize(plans.size() * table_elements);
      if (bitset) {
        filtered_codes.resize(candidate_count * static_cast<size_t>(ix->pq_m));
      } else {
        filtered_codes.clear();
      }

      for (size_t task_index = 0; task_index < plans.size(); ++task_index) {
        const AdcPlan& plan = plans[task_index];
        const float* query = queries + plan.query * dim;
        const float* cent = ix->centroids.data() + static_cast<int64_t>(plan.cluster) * dim;
        for (int64_t t = 0; t < dim; ++t) {
          qres[static_cast<size_t>(t)] = query[t] - cent[t];
        }

        float* table = lut_storage.data() + task_index * table_elements;
        for (int32_t m = 0; m < ix->pq_m; ++m) {
          const float* sub = qres.data() + static_cast<size_t>(m) * ix->dsub;
          const float* cb =
              ix->codebooks.data() + static_cast<size_t>(m) * ix->pq_ks * ix->dsub;
          for (int32_t cs = 0; cs < ix->pq_ks; ++cs) {
            table[static_cast<size_t>(m * ix->pq_ks + cs)] =
                l2sq(sub, cb + cs * ix->dsub, ix->dsub);
          }
        }

        const size_t output_offset = static_cast<size_t>(plan.output_offset);
        const uint8_t* task_codes = nullptr;
        if (!bitset) {
          const size_t list_begin = static_cast<size_t>(plan.list_begin);
          const size_t rows = static_cast<size_t>(plan.rows);
          std::memcpy(candidate_ids.data() + output_offset,
                      ix->packed_ids.data() + list_begin, rows * sizeof(int64_t));
          task_codes =
              ix->packed_codes.data() + list_begin * static_cast<size_t>(ix->pq_m);
          saturating_add(direct_rows, plan.rows);
        } else {
          size_t destination = output_offset;
          for (int64_t position = plan.list_begin; position < plan.list_end; ++position) {
            const size_t packed_position = static_cast<size_t>(position);
            const int64_t id = ix->packed_ids[packed_position];
            if (id < 0 || id >= ix->ds.n) return OVVS_STATUS_ERROR;
            if (!allowed(bitset, id)) continue;
            candidate_ids[destination] = id;
            std::memcpy(filtered_codes.data() + destination * static_cast<size_t>(ix->pq_m),
                        ix->packed_codes.data() +
                            packed_position * static_cast<size_t>(ix->pq_m),
                        static_cast<size_t>(ix->pq_m));
            ++destination;
          }
          if (destination != output_offset + static_cast<size_t>(plan.rows)) {
            return OVVS_STATUS_ERROR;
          }
          task_codes =
              filtered_codes.data() + output_offset * static_cast<size_t>(ix->pq_m);
          const size_t copied_bytes =
              static_cast<size_t>(plan.rows) * static_cast<size_t>(ix->pq_m);
          saturating_add(
              filtered_copy_bytes,
              copied_bytes > static_cast<size_t>(std::numeric_limits<int64_t>::max())
                  ? std::numeric_limits<int64_t>::max()
                  : static_cast<int64_t>(copied_bytes));
        }
        tasks[task_index] = {table, task_codes, plan.rows, plan.output_offset};
      }

      if (candidate_rows > 0) {
        if (tasks.empty()) return OVVS_STATUS_ERROR;
        status = prim_pq_adc_batch(*rd(res), tasks.data(), static_cast<int64_t>(tasks.size()),
                                   ix->pq_m, ix->pq_ks, adc.data(), candidate_rows);
        if (status != OVVS_STATUS_SUCCESS) return status;
      } else if (!tasks.empty()) {
        return OVVS_STATUS_ERROR;
      }

      int64_t max_refine = 0;
      for (int64_t local_query = 0; local_query < block_queries; ++local_query) {
        const int64_t count = query_offsets[static_cast<size_t>(local_query + 1)] -
                              query_offsets[static_cast<size_t>(local_query)];
        max_refine = std::max(max_refine,
                              std::min(static_cast<int64_t>(krefine), count));
      }
      const size_t refine_count = static_cast<size_t>(max_refine);
      if (refine_count > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
        return OVVS_STATUS_SHAPE_MISMATCH;
      }
      topk_indices.resize(refine_count);
      topk_values.resize(refine_count);
      candidates.resize(refine_count);
      gathered.resize(refine_count * static_cast<size_t>(dim));
      refined_scores.resize(refine_count);
      const size_t final_count = static_cast<size_t>(std::min(k, max_refine));
      final_indices.resize(final_count);
      final_values.resize(final_count);

      for (int64_t local_query = 0; local_query < block_queries; ++local_query) {
        const int64_t q = block_begin + local_query;
        const int64_t query_begin = query_offsets[static_cast<size_t>(local_query)];
        const int64_t query_end = query_offsets[static_cast<size_t>(local_query + 1)];
        const int64_t query_rows = query_end - query_begin;
        if (query_rows <= 0) continue;

        const int64_t kr = std::min(static_cast<int64_t>(krefine), query_rows);
        status = prim_topk(*rd(res), adc.data() + query_begin, 1, query_rows, kr,
                           topk_indices.data(), topk_values.data(), false);
        if (status != OVVS_STATUS_SUCCESS) return status;
        for (int64_t t = 0; t < kr; ++t) {
          const int64_t selected = topk_indices[static_cast<size_t>(t)];
          if (selected < 0 || selected >= query_rows) return OVVS_STATUS_ERROR;
          candidates[static_cast<size_t>(t)] =
              candidate_ids[static_cast<size_t>(query_begin + selected)];
        }
        status = prim_gather_rows(*rd(res), ix->ds.x.data(), ix->ds.n, dim,
                                  candidates.data(), kr,
                                 gathered.data());
        if (status != OVVS_STATUS_SUCCESS) return status;
        const float* query = queries + q * dim;
        status = prim_pairwise(*rd(res), ix->ds.metric, query, 1, gathered.data(), kr, dim,
                               refined_scores.data(), 2.f);
        if (status != OVVS_STATUS_SUCCESS) return status;
        const int64_t kk = std::min(k, kr);
        status = prim_topk(*rd(res), refined_scores.data(), 1, kr, kk,
                           final_indices.data(), final_values.data(),
                           metric_largest(ix->ds.metric));
        if (status != OVVS_STATUS_SUCCESS) return status;
        for (int64_t t = 0; t < kk; ++t) {
          const int64_t selected = final_indices[static_cast<size_t>(t)];
          if (selected < 0 || selected >= kr) return OVVS_STATUS_ERROR;
          staged_neighbors[static_cast<size_t>(q * k + t)] =
              candidates[static_cast<size_t>(selected)];
          float d = final_values[static_cast<size_t>(t)];
          if (ix->ds.metric == OVVS_METRIC_INNER_PRODUCT) d = -d;
          staged_distances[static_cast<size_t>(q * k + t)] = d;
        }
      }
      block_begin = block_end;
    }

    std::memcpy(neighbors, staged_neighbors.data(), staged_neighbors.size() * sizeof(int64_t));
    std::memcpy(distances, staged_distances.data(), staged_distances.size() * sizeof(float));
    record_ivf_pq_search_layout(*rd(res), direct_rows, filtered_copy_bytes);
    return OVVS_STATUS_SUCCESS;
  } catch (const std::length_error&) {
    return OVVS_STATUS_SHAPE_MISMATCH;
  } catch (const std::bad_alloc&) {
    return OVVS_STATUS_OOM;
  } catch (...) {
    return OVVS_STATUS_ERROR;
  }
}

ovvsStatus ovvsIvfPqDestroy(ovvsIvfPqIndex_t index) {
  delete reinterpret_cast<IvfPq*>(index);
  return OVVS_STATUS_SUCCESS;
}

constexpr uint32_t kIvfPqMagic = 0x31515049u; /* 'IPQ1' */
constexpr uint64_t kIvfPqV1HeaderBytes = sizeof(uint32_t) + sizeof(int32_t) +
                                         2 * sizeof(int64_t) + 5 * sizeof(int32_t);

ovvsStatus ovvsIvfPqSerialize(ovvsIvfPqIndex_t index, const char* path) {
  if (!index || !path) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<IvfPq*>(index);
  std::ofstream f(path, std::ios::binary);
  if (!f) return OVVS_STATUS_IO;
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
  return f.good() ? OVVS_STATUS_SUCCESS : OVVS_STATUS_IO;
}

ovvsStatus ovvsIvfPqDeserialize(ovvsResources_t res, const char* path, ovvsIvfPqIndex_t* index) {
  if (index) *index = nullptr;
  if (!res || !path || !index) return OVVS_STATUS_INVALID_ARGUMENT;
  try {
    std::ifstream f(path, std::ios::binary);
    if (!f) return OVVS_STATUS_IO;
    f.seekg(0, std::ios::end);
    const std::streamoff file_end = f.tellg();
    if (file_end < 0) return OVVS_STATUS_IO;
    const uint64_t file_bytes = static_cast<uint64_t>(file_end);
    f.seekg(0, std::ios::beg);
    if (!f) return OVVS_STATUS_IO;
    const auto read_exact = [&](void* destination, size_t bytes) {
      if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) return false;
      f.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(bytes));
      return static_cast<bool>(f);
    };
    const auto checked_product = [](uint64_t a, uint64_t b, size_t& product) {
      if (b != 0 && a > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / b) {
        return false;
      }
      product = static_cast<size_t>(a * b);
      return true;
    };
    const auto checked_add_bytes = [](uint64_t count, uint64_t element_bytes,
                                      uint64_t& total) {
      if (element_bytes != 0 &&
          count > (std::numeric_limits<uint64_t>::max() - total) / element_bytes) {
        return false;
      }
      total += count * element_bytes;
      return true;
    };

    uint32_t magic = 0;
    int32_t version = 0;
    if (!read_exact(&magic, sizeof(magic)) || !read_exact(&version, sizeof(version)) ||
        magic != kIvfPqMagic) {
      return OVVS_STATUS_IO;
    }
    if (version != 1) return OVVS_STATUS_UNSUPPORTED;

    auto ix = std::make_unique<IvfPq>();
    int32_t metric = 0;
    if (!read_exact(&ix->ds.n, sizeof(ix->ds.n)) ||
        !read_exact(&ix->ds.dim, sizeof(ix->ds.dim)) ||
        !read_exact(&ix->nlist, sizeof(ix->nlist)) ||
        !read_exact(&ix->pq_m, sizeof(ix->pq_m)) ||
        !read_exact(&ix->pq_ks, sizeof(ix->pq_ks)) ||
        !read_exact(&ix->dsub, sizeof(ix->dsub)) || !read_exact(&metric, sizeof(metric))) {
      return OVVS_STATUS_IO;
    }
    if (ix->ds.n <= 0 || ix->ds.n > std::numeric_limits<int32_t>::max() || ix->ds.dim <= 0 ||
        ix->nlist <= 0 || ix->nlist > ix->ds.n || ix->pq_m <= 0 || ix->pq_ks < 16 ||
        ix->pq_ks > 256 || (ix->pq_ks & (ix->pq_ks - 1)) != 0 || ix->dsub <= 0 ||
        ix->ds.dim % ix->pq_m != 0 || ix->dsub != ix->ds.dim / ix->pq_m ||
        metric < OVVS_METRIC_L2_EXPANDED || metric > OVVS_METRIC_LP_UNEXPANDED) {
      return OVVS_STATUS_IO;
    }
    ix->ds.metric = static_cast<ovvsMetric>(metric);

    size_t centroid_count = 0;
    size_t codebook_prefix = 0;
    size_t codebook_count = 0;
    size_t code_count = 0;
    size_t dataset_count = 0;
    if (!checked_product(static_cast<uint64_t>(ix->nlist), static_cast<uint64_t>(ix->ds.dim),
                         centroid_count) ||
        !checked_product(static_cast<uint64_t>(ix->pq_m), static_cast<uint64_t>(ix->pq_ks),
                         codebook_prefix) ||
        !checked_product(static_cast<uint64_t>(codebook_prefix), static_cast<uint64_t>(ix->dsub),
                         codebook_count) ||
        !checked_product(static_cast<uint64_t>(ix->ds.n), static_cast<uint64_t>(ix->pq_m),
                         code_count) ||
        !checked_product(static_cast<uint64_t>(ix->ds.n), static_cast<uint64_t>(ix->ds.dim),
                         dataset_count) ||
        centroid_count > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) /
                             sizeof(float) ||
        codebook_count > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) /
                             sizeof(float) ||
        dataset_count > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) /
                            sizeof(float)) {
      return OVVS_STATUS_IO;
    }
    uint64_t expected_bytes = kIvfPqV1HeaderBytes;
    if (!checked_add_bytes(centroid_count, sizeof(float), expected_bytes) ||
        !checked_add_bytes(codebook_count, sizeof(float), expected_bytes) ||
        !checked_add_bytes(code_count, sizeof(uint8_t), expected_bytes) ||
        !checked_add_bytes(dataset_count, sizeof(float), expected_bytes) ||
        !checked_add_bytes(static_cast<uint64_t>(ix->ds.n), sizeof(int32_t), expected_bytes) ||
        !checked_add_bytes(static_cast<uint64_t>(ix->nlist), sizeof(int32_t), expected_bytes) ||
        !checked_add_bytes(static_cast<uint64_t>(ix->ds.n), sizeof(int64_t), expected_bytes) ||
        file_bytes != expected_bytes) {
      return OVVS_STATUS_IO;
    }

    ix->centroids.resize(centroid_count);
    ix->codebooks.resize(codebook_count);
    ix->codes.resize(code_count);
    ix->ds.x.resize(dataset_count);
    ix->assign.resize(static_cast<size_t>(ix->ds.n));
    if (!read_exact(ix->centroids.data(), ix->centroids.size() * sizeof(float)) ||
        !read_exact(ix->codebooks.data(), ix->codebooks.size() * sizeof(float)) ||
        !read_exact(ix->codes.data(), ix->codes.size()) ||
        !read_exact(ix->ds.x.data(), ix->ds.x.size() * sizeof(float)) ||
        !read_exact(ix->assign.data(), ix->assign.size() * sizeof(int32_t))) {
      return OVVS_STATUS_IO;
    }

    ix->lists.resize(static_cast<size_t>(ix->nlist));
    int64_t total_ids = 0;
    for (int32_t c = 0; c < ix->nlist; ++c) {
      int32_t list_size = 0;
      if (!read_exact(&list_size, sizeof(list_size)) || list_size < 0 ||
          list_size > ix->ds.n - total_ids) {
        return OVVS_STATUS_IO;
      }
      total_ids += list_size;
      auto& list = ix->lists[static_cast<size_t>(c)];
      list.resize(static_cast<size_t>(list_size));
      if (list_size > 0 && !read_exact(list.data(), list.size() * sizeof(int64_t))) {
        return OVVS_STATUS_IO;
      }
    }
    if (total_ids != ix->ds.n || rebuild_ivf_pq_packed(*ix) != OVVS_STATUS_SUCCESS) {
      return OVVS_STATUS_IO;
    }
    const int64_t rebuilt_rows = ix->ds.n;
    *index = reinterpret_cast<ovvsIvfPqIndex_t>(ix.release());
    record_ivf_pq_rebuild(*rd(res), rebuilt_rows);
    return OVVS_STATUS_SUCCESS;
  } catch (const std::length_error&) {
    return OVVS_STATUS_IO;
  } catch (const std::bad_alloc&) {
    return OVVS_STATUS_OOM;
  } catch (...) {
    return OVVS_STATUS_ERROR;
  }
}

ovvsStatus ovvsIvfPqExtend(ovvsResources_t res, ovvsIvfPqIndex_t index, const float* extra, int64_t nextra) {
  if (!res || !index || !extra || nextra <= 0) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<IvfPq*>(index);
  const int64_t dim = ix->ds.dim;
  const int64_t old_n = ix->ds.n;
  if (old_n <= 0 || dim <= 0 || ix->nlist <= 0 || ix->pq_m <= 0 || ix->pq_ks <= 0 ||
      ix->dsub <= 0 || nextra > std::numeric_limits<int64_t>::max() - old_n ||
      nextra > std::numeric_limits<int64_t>::max() / dim) {
    return OVVS_STATUS_SHAPE_MISMATCH;
  }
  const int64_t new_n = old_n + nextra;
  if (new_n > std::numeric_limits<int32_t>::max() ||
      static_cast<uint64_t>(nextra) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
              static_cast<uint64_t>(ix->nlist) ||
      static_cast<uint64_t>(nextra) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
              static_cast<uint64_t>(dim) ||
      static_cast<uint64_t>(new_n) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
              static_cast<uint64_t>(dim) ||
      static_cast<uint64_t>(new_n) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
              static_cast<uint64_t>(ix->pq_m)) {
    return OVVS_STATUS_SHAPE_MISMATCH;
  }
  const size_t old_dataset_elements =
      static_cast<size_t>(old_n) * static_cast<size_t>(dim);
  const size_t old_code_elements =
      static_cast<size_t>(old_n) * static_cast<size_t>(ix->pq_m);
  if (ix->ds.x.size() != old_dataset_elements ||
      ix->assign.size() != static_cast<size_t>(old_n) ||
      ix->codes.size() != old_code_elements ||
      ix->lists.size() != static_cast<size_t>(ix->nlist) ||
      ix->list_offsets.size() != static_cast<size_t>(ix->nlist) + 1 ||
      ix->packed_ids.size() != static_cast<size_t>(old_n) ||
      ix->packed_codes.size() != old_code_elements) {
    return OVVS_STATUS_ERROR;
  }

  try {
    const size_t extra_elements =
        static_cast<size_t>(nextra) * static_cast<size_t>(dim);
    std::vector<float> extra_values(extra, extra + extra_elements);
    std::vector<int64_t> labels(static_cast<size_t>(nextra));
    std::vector<float> label_distances(static_cast<size_t>(nextra));
    std::vector<float> scores(static_cast<size_t>(nextra) * static_cast<size_t>(ix->nlist));
    ovvsStatus status =
        prim_pairwise(*rd(res), OVVS_METRIC_L2_EXPANDED, extra_values.data(), nextra,
                      ix->centroids.data(), ix->nlist, dim, scores.data(), 2.f);
    if (status != OVVS_STATUS_SUCCESS) return status;
    status = prim_topk(*rd(res), scores.data(), nextra, ix->nlist, 1, labels.data(),
                       label_distances.data(), false);
    if (status != OVVS_STATUS_SUCCESS) return status;

    std::vector<int32_t> next_assign = ix->assign;
    std::vector<std::vector<int64_t>> next_lists = ix->lists;
    next_assign.resize(static_cast<size_t>(new_n));
    std::vector<float> residuals(static_cast<size_t>(nextra) * static_cast<size_t>(dim));
    for (int64_t i = 0; i < nextra; ++i) {
      const int64_t label = labels[static_cast<size_t>(i)];
      if (label < 0 || label >= ix->nlist) return OVVS_STATUS_ERROR;
      const int32_t assignment = static_cast<int32_t>(label);
      next_assign[static_cast<size_t>(old_n + i)] = assignment;
      next_lists[static_cast<size_t>(assignment)].push_back(old_n + i);
      const float* centroid =
          ix->centroids.data() + static_cast<size_t>(assignment) * static_cast<size_t>(dim);
      for (int64_t t = 0; t < dim; ++t) {
        residuals[static_cast<size_t>(i * dim + t)] =
            extra_values[static_cast<size_t>(i * dim + t)] - centroid[t];
      }
    }

    std::vector<uint8_t> next_codes = ix->codes;
    next_codes.resize(static_cast<size_t>(new_n) * static_cast<size_t>(ix->pq_m));
    pq_encode(residuals.data(), nextra, dim, ix->pq_m, ix->pq_ks, ix->dsub,
              ix->codebooks.data(),
              next_codes.data() + static_cast<size_t>(old_n) * static_cast<size_t>(ix->pq_m));

    std::vector<int64_t> next_offsets;
    std::vector<int64_t> next_packed_ids;
    std::vector<uint8_t> next_packed_codes;
    status = make_ivf_pq_packed(new_n, ix->nlist, ix->pq_m, ix->pq_ks, next_assign,
                                next_codes, next_lists, next_offsets, next_packed_ids,
                                next_packed_codes);
    if (status != OVVS_STATUS_SUCCESS) return status;

    const size_t new_dataset_elements =
        static_cast<size_t>(new_n) * static_cast<size_t>(dim);
    ix->ds.x.reserve(new_dataset_elements);
    ix->ds.x.insert(ix->ds.x.end(), extra_values.begin(), extra_values.end());
    ix->assign.swap(next_assign);
    ix->codes.swap(next_codes);
    ix->lists.swap(next_lists);
    ix->list_offsets.swap(next_offsets);
    ix->packed_ids.swap(next_packed_ids);
    ix->packed_codes.swap(next_packed_codes);
    ix->ds.n = new_n;
    record_ivf_pq_rebuild(*rd(res), new_n);
    return OVVS_STATUS_SUCCESS;
  } catch (const std::length_error&) {
    return OVVS_STATUS_SHAPE_MISMATCH;
  } catch (const std::bad_alloc&) {
    return OVVS_STATUS_OOM;
  } catch (...) {
    return OVVS_STATUS_ERROR;
  }
}
