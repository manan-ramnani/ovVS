#include "internal.hpp"

#include <atomic>
#include <chrono>
#include <fstream>
#include <thread>

using namespace ovvs::impl;

namespace {

constexpr uint32_t kCagraMagic = 0x31475243u; /* 'CRG1' */

using BuildClock = std::chrono::steady_clock;

int64_t elapsed_ns(BuildClock::time_point begin) noexcept {
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(BuildClock::now() - begin).count();
  return elapsed < 0 ? 0 : elapsed;
}

int64_t logical_bytes(int64_t rows, int64_t width, size_t element_size) noexcept {
  if (rows <= 0 || width <= 0 || element_size == 0) return 0;
  const uint64_t a = static_cast<uint64_t>(rows);
  const uint64_t b = static_cast<uint64_t>(width);
  const uint64_t limit = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  if (a > limit / b || a * b > limit / element_size) {
    return std::numeric_limits<int64_t>::max();
  }
  return static_cast<int64_t>(a * b * element_size);
}

void saturating_add(int64_t& value, int64_t delta) noexcept {
  if (delta <= 0) return;
  if (value > std::numeric_limits<int64_t>::max() - delta) {
    value = std::numeric_limits<int64_t>::max();
  } else {
    value += delta;
  }
}

void publish_cagra_build_stats(ResourcesData& resources,
                               const ovvsCagraBuildStatsV1& delta) {
  std::lock_guard<std::mutex> lock(resources.cagra_build_stats_mutex);
  auto& total = resources.cagra_build_stats;
#define OVVS_CAGRA_ADD(field) saturating_add(total.field, delta.field)
  OVVS_CAGRA_ADD(successful_calls);
  OVVS_CAGRA_ADD(rows);
  OVVS_CAGRA_ADD(dataset_copy_bytes);
  OVVS_CAGRA_ADD(initializer_graph_payload_bytes);
  OVVS_CAGRA_ADD(published_graph_copy_bytes);
  OVVS_CAGRA_ADD(nndescent_initializer_calls);
  OVVS_CAGRA_ADD(ivfpq_initializer_calls);
  OVVS_CAGRA_ADD(iterative_initializer_calls);
  OVVS_CAGRA_ADD(total_wall_ns);
  OVVS_CAGRA_ADD(dataset_copy_ns);
  OVVS_CAGRA_ADD(initializer_ns);
  OVVS_CAGRA_ADD(optimizer_prune_merge_ns);
  OVVS_CAGRA_ADD(index_materialize_ns);
  OVVS_CAGRA_ADD(initializer_final_cpu_calls);
  OVVS_CAGRA_ADD(initializer_final_gpu_calls);
  OVVS_CAGRA_ADD(initializer_final_npu_calls);
  OVVS_CAGRA_ADD(nndescent_gpu_iterations);
  OVVS_CAGRA_ADD(nndescent_gpu_converged_calls);
  OVVS_CAGRA_ADD(nndescent_gpu_final_changed_edges);
  OVVS_CAGRA_ADD(nndescent_gpu_final_pending_new_edges);
  OVVS_CAGRA_ADD(nndescent_gpu_instrumented_calls);
  OVVS_CAGRA_ADD(nndescent_gpu_allocation_calls);
  OVVS_CAGRA_ADD(nndescent_gpu_allocation_bytes);
  OVVS_CAGRA_ADD(nndescent_gpu_h2d_calls);
  OVVS_CAGRA_ADD(nndescent_gpu_h2d_bytes);
  OVVS_CAGRA_ADD(nndescent_gpu_d2h_calls);
  OVVS_CAGRA_ADD(nndescent_gpu_d2h_bytes);
  OVVS_CAGRA_ADD(nndescent_gpu_kernel_launches);
  OVVS_CAGRA_ADD(nndescent_gpu_submission_calls);
  OVVS_CAGRA_ADD(nndescent_gpu_wait_calls);
#undef OVVS_CAGRA_ADD
  total.nndescent_gpu_peak_owned_bytes_max =
      std::max(total.nndescent_gpu_peak_owned_bytes_max,
               delta.nndescent_gpu_peak_owned_bytes_max);
}

struct Dataset {
  UsmFloatVec x;
  /* fp16 primary storage (OVVS_CAGRA_F16=1): when populated, `x` is EMPTY and every
     vector lives here as IEEE binary16 -- half the resident bytes of fp32 for every
     corpus. Integer-valued data (SIFT-class) converts exactly, so results stay
     bitwise-identical there; for general float corpora the storage rounds once at
     insert and the effect is judged by measured recall, never assumed. */
  UsmU16Vec x16;
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

struct IvfRabitq {
  Dataset ds;
  int32_t nlist = 0;
  std::vector<float> centroids;
  std::vector<int32_t> assign;
  std::vector<float> scales;
  std::vector<uint8_t> signs;
  std::vector<uint8_t> rand_sign;
  std::vector<std::vector<int64_t>> lists;
  int64_t nbytes = 0;
};

struct NnGraph {
  int64_t n = 0;
  int32_t degree = 0;
  std::vector<int32_t> ids;
};

struct CagraIndex {
  Dataset ds;
  int32_t degree = 0;
  UsmI32Vec graph;
  bool has_dataset = true;
  /* Index-owned int8 mirror, fp16 mode only (shared USM: the GPU walk reads it too).
     Unlike the Resources-cached fp32-mode mirror it is updated in place by mutation, so
     it needs no fingerprinting and can never serve stale rows. Dropped permanently the
     first time a mutation stores a vector that is not integer-valued in [0,255]. */
  UsmU8Vec mirror8;
  bool mirror8_ok = false;
  int32_t pq_m = 0;
  int32_t pq_ks = 0;
  int32_t dsub = 0;
  std::vector<float> codebooks;
  std::vector<uint8_t> codes;
  /* Tombstones, one bit per row. A deleted row keeps its edges and still routes traffic --
     dropping it from traversal would fragment the graph, since every node is a routing hop for
     its neighbours -- it is merely never returned by a search. Left empty while nothing has been
     deleted, so an unmutated index allocates nothing for this. */
  std::vector<uint8_t> deleted;
  int64_t deleted_count = 0;
  /* Slot reuse. A tombstoned row can be handed to a later insert, which is what stops churn from
     growing the footprint forever -- but reusing a row recycles its id, and a caller still holding
     the old id would silently read someone else's vector. So each row carries a generation, and a
     public id is `slot | (generation << 32)`. Generation 0 packs to the bare slot, so an index
     that has never reused anything returns exactly the ids it always did, and both vectors stay
     empty until the first reuse. A stale id is then detectably stale rather than silently wrong. */
  std::vector<uint32_t> generation;
  std::vector<int32_t> free_slots;
};

inline uint32_t cagra_generation_of(const CagraIndex& ix, int64_t slot) {
  if (ix.generation.empty() || slot < 0 ||
      static_cast<size_t>(slot) >= ix.generation.size()) {
    return 0u;
  }
  return ix.generation[static_cast<size_t>(slot)];
}

static bool cagra_f16_mode() {
  const char* env = std::getenv("OVVS_CAGRA_F16");
  return env && *env && *env != '0';
}

static bool cagra_is_f16(const CagraIndex& ix) { return !ix.ds.x16.empty(); }

static bool cagra_dataset_present(const CagraIndex& ix) {
  return ix.has_dataset && (!ix.ds.x.empty() || !ix.ds.x16.empty());
}

/* dst must hold dim floats. */
static void cagra_row_f32(const CagraIndex& ix, int64_t row, float* dst) {
  const int64_t dim = ix.ds.dim;
  if (!ix.ds.x.empty()) {
    std::copy(ix.ds.x.begin() + row * dim, ix.ds.x.begin() + (row + 1) * dim, dst);
  } else {
    f16_row_to_f32(ix.ds.x16.data() + row * dim, dst, dim);
  }
}

static bool cagra_f16_row_int8_ok(const uint16_t* row, int64_t dim) {
  for (int64_t i = 0; i < dim; ++i) {
    const float v = f16_bits_to_f32(row[i]);
    if (!(v >= 0.f && v <= 255.f && v == std::floor(v))) return false;
  }
  return true;
}

/* Rebuilds the index-owned int8 mirror from fp16 storage. Eligibility is judged on the
   STORED (fp16) values -- the mirror only has to agree with what the walk would read. */
static void cagra_build_mirror8(CagraIndex* ix) {
  ix->mirror8_ok = false;
  UsmU8Vec().swap(ix->mirror8);
  if (ix->ds.x16.empty() || ix->ds.dim > 256) return;
  const size_t count = ix->ds.x16.size();
  for (size_t i = 0; i < count; ++i) {
    const float v = f16_bits_to_f32(ix->ds.x16[i]);
    if (!(v >= 0.f && v <= 255.f && v == std::floor(v))) return;
  }
  ix->mirror8.resize(count);
  for (size_t i = 0; i < count; ++i) {
    ix->mirror8[i] = static_cast<uint8_t>(
        static_cast<int32_t>(f16_bits_to_f32(ix->ds.x16[i])));
  }
  ix->mirror8_ok = true;
}

/* Converts fp32 primary storage to fp16 and releases the fp32 copy. Called at the end
   of build (allow_lossy: the caller set the mode for THIS build, rounding is the
   documented deal) and after loading an fp32 file (lossless conversions only -- the file
   predates the env, and silently rounding stored float data on load is the one footgun;
   OVVS_CAGRA_F16=force overrides). */
static void cagra_finalize_storage(CagraIndex* ix, bool allow_lossy) {
  if (!cagra_f16_mode() || ix->ds.x.empty()) return;
  if (!allow_lossy) {
    const char* env = std::getenv("OVVS_CAGRA_F16");
    const bool force = env && std::strcmp(env, "force") == 0;
    if (!force) {
      for (const float v : ix->ds.x) {
        if (f16_bits_to_f32(f32_to_f16_bits(v)) != v) return; /* stay fp32 */
      }
    }
  }
  const size_t count = ix->ds.x.size();
  ix->ds.x16.resize(count);
  for (size_t i = 0; i < count; ++i) ix->ds.x16[i] = f32_to_f16_bits(ix->ds.x[i]);
  UsmFloatVec().swap(ix->ds.x);
  cagra_build_mirror8(ix);
}

/* The walk-storage trio graph_search forwards to the engines. */
struct CagraStorage {
  const float* f32 = nullptr;
  const uint16_t* f16 = nullptr;
  const uint8_t* u8 = nullptr;
};
static CagraStorage cagra_storage(const CagraIndex& ix) {
  CagraStorage s;
  if (!ix.ds.x.empty()) {
    s.f32 = ix.ds.x.data();
  } else if (!ix.ds.x16.empty()) {
    s.f16 = ix.ds.x16.data();
    if (ix.mirror8_ok) s.u8 = ix.mirror8.data();
  }
  return s;
}

/* Writes one vector into an existing slot, converting for the active storage and
   keeping the int8 mirror coherent: updated in place, or dropped permanently the first
   time an ineligible vector lands (the walk then reads fp16 directly). */
static void cagra_write_row(CagraIndex* ix, int64_t slot, const float* vec) {
  const int64_t dim = ix->ds.dim;
  if (!ix->ds.x.empty()) {
    std::copy(vec, vec + dim, ix->ds.x.begin() + slot * dim);
    return;
  }
  uint16_t* dst = ix->ds.x16.data() + slot * dim;
  f32_row_to_f16(vec, dst, dim);
  if (ix->mirror8_ok) {
    if (cagra_f16_row_int8_ok(dst, dim)) {
      uint8_t* m = ix->mirror8.data() + slot * dim;
      for (int64_t i = 0; i < dim; ++i) {
        m[i] = static_cast<uint8_t>(static_cast<int32_t>(f16_bits_to_f32(dst[i])));
      }
    } else {
      ix->mirror8_ok = false;
      UsmU8Vec().swap(ix->mirror8);
    }
  }
}

static void cagra_append_row(CagraIndex* ix, const float* vec) {
  const int64_t dim = ix->ds.dim;
  const int64_t slot = ix->ds.n;
  if (!ix->ds.x16.empty()) {
    ix->ds.x16.resize(ix->ds.x16.size() + static_cast<size_t>(dim));
    if (ix->mirror8_ok) ix->mirror8.resize(ix->mirror8.size() + static_cast<size_t>(dim));
    ix->ds.n = slot + 1;
    cagra_write_row(ix, slot, vec);
  } else {
    ix->ds.x.insert(ix->ds.x.end(), vec, vec + dim);
    ix->ds.n = slot + 1;
  }
}

inline int64_t cagra_pack_id(const CagraIndex& ix, int64_t slot) {
  if (ix.generation.empty() || slot < 0) return slot;
  return slot | (static_cast<int64_t>(cagra_generation_of(ix, slot)) << 32);
}

/* Splits a public id and rejects it if the row has been reused since the id was handed out. */
inline bool cagra_resolve_id(const CagraIndex& ix, int64_t id, int64_t* slot) {
  if (id < 0) return false;
  const int64_t row = id & 0xFFFFFFFFLL;
  const uint32_t gen = static_cast<uint32_t>(static_cast<uint64_t>(id) >> 32);
  if (row >= ix.ds.n) return false;
  if (gen != cagra_generation_of(ix, row)) return false;
  *slot = row;
  return true;
}

inline bool cagra_row_deleted(const CagraIndex& ix, int64_t row) {
  if (ix.deleted.empty() || row < 0) return false;
  const size_t byte = static_cast<size_t>(row) >> 3;
  if (byte >= ix.deleted.size()) return false;
  return ((ix.deleted[byte] >> (row & 7)) & 1u) != 0;
}

struct HnswIndex {
  Dataset ds;
  int32_t degree = 0;
  int32_t hnsw_m = 0;
  std::vector<int32_t> graph;
  std::vector<int32_t> level;
  int32_t max_level = 0;
  int64_t enter = 0;
};

struct VamanaIndex {
  Dataset ds;
  int32_t degree = 0;
  std::vector<int32_t> graph;
  const int32_t* graph_view = nullptr;
  const float* x_view = nullptr;
  bool mmapped = false;
#ifdef _WIN32
  HANDLE file_handle = INVALID_HANDLE_VALUE;
  HANDLE map_handle = nullptr;
  void* view = nullptr;
#else
  int fd = -1;
  void* view = nullptr;
  size_t map_size = 0;
#endif
};

struct ScannIndex {
  Dataset ds;
  ovvsIvfPqIndex_t pq = nullptr;
  std::vector<float> aniso;
};

struct IvfPqGuard {
  ovvsIvfPqIndex_t value = nullptr;
  ~IvfPqGuard() {
    if (value) (void)ovvsIvfPqDestroy(value);
  }
};

int64_t nbytes_bits(int64_t dim) { return (dim + 7) / 8; }

void pack_signs(const float* v, int64_t dim, const uint8_t* rand_sign, uint8_t* out) {
  const int64_t nb = nbytes_bits(dim);
  std::fill(out, out + nb, 0);
  for (int64_t i = 0; i < dim; ++i) {
    float x = v[i];
    if (rand_sign && rand_sign[static_cast<size_t>(i)]) x = -x;
    if (x >= 0.f) out[i >> 3] |= static_cast<uint8_t>(1u << (i & 7));
  }
}

float rabitq_ip(const float* qres, const uint8_t* code, const uint8_t* rand_sign, int64_t dim,
                float scale) {
  float s = 0.f;
  const float inv = 1.f / std::sqrt(static_cast<float>(std::max<int64_t>(dim, 1)));
  for (int64_t i = 0; i < dim; ++i) {
    float bit = ((code[i >> 3] >> (i & 7)) & 1u) ? 1.f : -1.f;
    if (rand_sign && rand_sign[static_cast<size_t>(i)]) bit = -bit;
    s += qres[i] * bit;
  }
  return scale * s * inv;
}

void build_ivf_lists(const int32_t* assign, int64_t n, int32_t nlist,
                     std::vector<std::vector<int64_t>>& lists) {
  lists.assign(static_cast<size_t>(nlist), {});
  for (int64_t i = 0; i < n; ++i) {
    const int32_t a = assign[i];
    if (a >= 0 && a < nlist) lists[static_cast<size_t>(a)].push_back(i);
  }
}

ovvsStatus knn_graph_brute(ResourcesData& r, const float* x, int64_t n, int64_t dim,
                           ovvsMetric metric, int32_t degree, std::vector<int32_t>& graph) {
  degree = std::min(degree, static_cast<int32_t>(std::max<int64_t>(1, n - 1)));
  graph.assign(static_cast<size_t>(n) * static_cast<size_t>(degree), -1);
  std::vector<float> scores(static_cast<size_t>(n) * static_cast<size_t>(n));
  const ovvsStatus pairwise_status = prim_pairwise(r, metric, x, n, x, n, dim, scores.data(), 2.f);
  if (pairwise_status != OVVS_STATUS_SUCCESS) return pairwise_status;
  for (int64_t i = 0; i < n; ++i) scores[static_cast<size_t>(i * n + i)] = kInf;
  std::vector<int64_t> idx(static_cast<size_t>(n) * static_cast<size_t>(degree));
  std::vector<float> val(static_cast<size_t>(n) * static_cast<size_t>(degree));
  const ovvsStatus topk_status = prim_topk(r, scores.data(), n, n, degree, idx.data(), val.data(), false);
  if (topk_status != OVVS_STATUS_SUCCESS) return topk_status;
  for (int64_t i = 0; i < n; ++i) {
    for (int32_t t = 0; t < degree; ++t) {
      graph[static_cast<size_t>(i * degree + t)] =
          static_cast<int32_t>(idx[static_cast<size_t>(i * degree + t)]);
    }
  }
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus nndescent_build(ResourcesData& r, const float* x, int64_t n, int64_t dim,
                            ovvsMetric metric, int32_t degree, int32_t iters,
                            std::vector<int32_t>& graph,
                            NnDescentBuildStats* build_stats = nullptr) {
  if (build_stats) *build_stats = {};
  degree = std::min(degree, static_cast<int32_t>(std::max<int64_t>(1, n - 1)));
  /* Exact kNN is cheap through prim_pairwise while n^2 scores fit in ~64MiB. */
  if (n <= 4096) {
    return knn_graph_brute(r, x, n, dim, metric, degree, graph);
  }
  graph.assign(static_cast<size_t>(n) * static_cast<size_t>(degree), -1);
  const ovvsStatus accelerated =
      prim_nndescent_build(r, x, n, dim, metric, degree, iters, graph.data(), build_stats);
  if (accelerated == OVVS_STATUS_SUCCESS) return accelerated;
  if (accelerated != OVVS_STATUS_UNSUPPORTED) return accelerated;
  if (build_stats) *build_stats = {};

  auto rng = rng_from(7);
  std::uniform_int_distribution<int64_t> pick(0, n - 1);
  for (int64_t i = 0; i < n; ++i) {
    std::vector<int32_t> row;
    while (static_cast<int32_t>(row.size()) < degree) {
      int64_t j = pick(rng);
      if (j == i) continue;
      if (std::find(row.begin(), row.end(), static_cast<int32_t>(j)) != row.end()) continue;
      row.push_back(static_cast<int32_t>(j));
    }
    std::memcpy(graph.data() + i * degree, row.data(), static_cast<size_t>(degree) * sizeof(int32_t));
  }
  for (int it = 0; it < iters; ++it) {
    for (int64_t i = 0; i < n; ++i) {
      std::vector<int64_t> cands;
      cands.reserve(static_cast<size_t>(degree) * (degree + 1));
      for (int32_t a = 0; a < degree; ++a) {
        const int32_t u = graph[static_cast<size_t>(i * degree + a)];
        if (u < 0 || u == static_cast<int32_t>(i)) continue;
        cands.push_back(u);
        for (int32_t b = 0; b < degree; ++b) {
          const int32_t v = graph[static_cast<size_t>(static_cast<int64_t>(u) * degree + b)];
          if (v < 0 || v == static_cast<int32_t>(i)) continue;
          cands.push_back(v);
        }
      }
      std::sort(cands.begin(), cands.end());
      cands.erase(std::unique(cands.begin(), cands.end()), cands.end());
      if (cands.empty()) continue;
      std::vector<float> gathered(cands.size() * static_cast<size_t>(dim));
      std::vector<float> sc(cands.size());
      const ovvsStatus gather_status =
          prim_gather_rows(r, x, n, dim, cands.data(), static_cast<int64_t>(cands.size()),
                           gathered.data());
      if (gather_status != OVVS_STATUS_SUCCESS) return gather_status;
      const ovvsStatus pairwise_status =
          prim_pairwise(r, metric, x + i * dim, 1, gathered.data(),
                        static_cast<int64_t>(cands.size()), dim, sc.data(), 2.f);
      if (pairwise_status != OVVS_STATUS_SUCCESS) return pairwise_status;
      const int64_t kk = std::min(static_cast<int64_t>(degree), static_cast<int64_t>(cands.size()));
      std::vector<int64_t> ti(static_cast<size_t>(kk));
      std::vector<float> tv(static_cast<size_t>(kk));
      const ovvsStatus topk_status =
          prim_topk(r, sc.data(), 1, static_cast<int64_t>(cands.size()), kk, ti.data(), tv.data(), false);
      if (topk_status != OVVS_STATUS_SUCCESS) return topk_status;
      for (int64_t t = 0; t < degree; ++t) {
        if (t < kk) {
          graph[static_cast<size_t>(i * degree + t)] =
              static_cast<int32_t>(cands[static_cast<size_t>(ti[static_cast<size_t>(t)])]);
        } else {
          graph[static_cast<size_t>(i * degree + t)] = -1;
        }
      }
    }
  }
  r.last_device = OVVS_DEVICE_CPU;
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus graph_search(ResourcesData& r, const float* dataset, int64_t n, int64_t dim,
                        ovvsMetric metric, const int32_t* graph, int32_t degree,
                        const float* queries, int64_t nq, int64_t k, int32_t itopk,
                        int32_t search_width, const uint8_t* bitset, int64_t* neighbors,
                        float* distances, const uint16_t* dataset_f16 = nullptr,
                        const uint8_t* dataset_u8 = nullptr) {
  return prim_graph_walk(r, dataset, n, dim, metric, graph, degree, queries, nq, k, itopk,
                         search_width, bitset, neighbors, distances, dataset_f16, dataset_u8);
}

void pq_encode_rows(const float* x, int64_t n, int64_t dim, int32_t pq_m, int32_t ks, int32_t dsub,
                    const float* codebooks, uint8_t* codes) {
  for (int64_t i = 0; i < n; ++i) {
    for (int32_t m = 0; m < pq_m; ++m) {
      const float* sub = x + i * dim + static_cast<int64_t>(m) * dsub;
      const float* cb = codebooks + static_cast<size_t>(m) * ks * dsub;
      int best = 0;
      float bd = kInf;
      for (int32_t c = 0; c < ks; ++c) {
        const float d = l2sq(sub, cb + c * dsub, dsub);
        if (d < bd) {
          bd = d;
          best = c;
        }
      }
      codes[i * pq_m + m] = static_cast<uint8_t>(best);
    }
  }
}

float pq_adc(const float* query, int64_t dim, const uint8_t* code, int32_t pq_m, int32_t ks, int32_t dsub,
             const float* codebooks) {
  float s = 0.f;
  for (int32_t m = 0; m < pq_m; ++m) {
    const float* sub = query + static_cast<int64_t>(m) * dsub;
    const float* cb = codebooks + (static_cast<size_t>(m) * ks + code[m]) * dsub;
    s += l2sq(sub, cb, dsub);
  }
  (void)dim;
  return s;
}

void graph_search_pq(const float* query_ds, int64_t n, int64_t dim, const int32_t* graph, int32_t degree,
                     const uint8_t* codes, int32_t pq_m, int32_t ks, int32_t dsub, const float* codebooks,
                     const float* queries, int64_t nq, int64_t k, int32_t itopk, int32_t search_width,
                     const uint8_t* bitset, int64_t* neighbors, float* distances) {
  itopk = std::max(itopk, static_cast<int32_t>(k));
  search_width = std::max(1, search_width);
  struct Node {
    float d;
    int64_t id;
  };
  for (int64_t q = 0; q < nq; ++q) {
    const float* query = queries + q * dim;
    std::vector<uint8_t> seen(static_cast<size_t>(n), 0);
    std::vector<char> expanded(static_cast<size_t>(n), 0);
    std::vector<Node> cand;
    const int64_t nseeds = cagra_seed_count(n, itopk, search_width);
    for (int64_t s = 0; s < nseeds; ++s) {
      int64_t id = (s * 9973 + q * 13) % n;
      if (!allowed(bitset, id) || seen[static_cast<size_t>(id)]) continue;
      seen[static_cast<size_t>(id)] = 1;
      cand.push_back({pq_adc(query, dim, codes + id * pq_m, pq_m, ks, dsub, codebooks), id});
    }
    if (static_cast<int32_t>(cand.size()) > itopk) {
      std::nth_element(cand.begin(), cand.begin() + itopk, cand.end(),
                       [](const Node& a, const Node& b) { return a.d < b.d; });
      cand.resize(static_cast<size_t>(itopk));
    }
    int iters = 0;
    const int max_iters = std::max(24, itopk * 6);
    while (iters++ < max_iters) {
      std::vector<int64_t> picks;
      picks.reserve(static_cast<size_t>(search_width));
      for (int s = 0; s < search_width; ++s) {
        int64_t pick = -1;
        float pick_d = kInf;
        for (const auto& c : cand) {
          if (expanded[static_cast<size_t>(c.id)]) continue;
          bool already = false;
          for (int64_t p : picks) {
            if (p == c.id) {
              already = true;
              break;
            }
          }
          if (already) continue;
          if (c.d < pick_d) {
            pick_d = c.d;
            pick = c.id;
          }
        }
        if (pick < 0) break;
        picks.push_back(pick);
      }
      if (picks.empty()) break;
      for (int64_t pick : picks) {
        expanded[static_cast<size_t>(pick)] = 1;
        const int32_t* nbrs = graph + pick * degree;
        for (int32_t e = 0; e < degree; ++e) {
          const int32_t nb = nbrs[e];
          if (nb < 0 || static_cast<int64_t>(nb) >= n) continue;
          if (seen[static_cast<size_t>(nb)]) continue;
          if (!allowed(bitset, nb)) continue;
          seen[static_cast<size_t>(nb)] = 1;
          cand.push_back({pq_adc(query, dim, codes + static_cast<int64_t>(nb) * pq_m, pq_m, ks, dsub, codebooks),
                          nb});
        }
      }
      const int32_t keep = itopk * 2;
      if (static_cast<int32_t>(cand.size()) > keep) {
        std::nth_element(cand.begin(), cand.begin() + keep, cand.end(),
                         [](const Node& a, const Node& b) { return a.d < b.d; });
        cand.resize(static_cast<size_t>(keep));
      }
    }
    std::sort(cand.begin(), cand.end(), [](const Node& a, const Node& b) { return a.d < b.d; });
    for (int64_t t = 0; t < k; ++t) {
      if (t < static_cast<int64_t>(cand.size())) {
        neighbors[q * k + t] = cand[static_cast<size_t>(t)].id;
        distances[q * k + t] = cand[static_cast<size_t>(t)].d;
      } else {
        neighbors[q * k + t] = -1;
        distances[q * k + t] = kInf;
      }
    }
  }
  (void)query_ds;
}

ovvsStatus cagra_init_ivfpq(ResourcesData& r, const float* x, int64_t n, int64_t dim,
                            ovvsMetric metric, int32_t degree, std::vector<int32_t>& graph) {
  IvfPqGuard pq;
  const int32_t nlist = std::max(2, std::min(8, static_cast<int32_t>(n / 4)));
  int32_t pq_m = 1;
  for (int32_t c = std::min(4, static_cast<int32_t>(dim)); c >= 1; --c) {
    if (dim % c == 0) {
      pq_m = c;
      break;
    }
  }
  auto* res = reinterpret_cast<ovvsResources_t>(&r);
  const ovvsStatus build_status =
      ovvsIvfPqBuild(res, x, n, dim, metric, nlist, pq_m, 8, &pq.value);
  if (build_status != OVVS_STATUS_SUCCESS) return build_status;
  graph.assign(static_cast<size_t>(n) * static_cast<size_t>(degree), -1);
  std::vector<int64_t> nb(static_cast<size_t>(degree) + 1);
  std::vector<float> ds(static_cast<size_t>(degree) + 1);
  for (int64_t i = 0; i < n; ++i) {
    const ovvsStatus search_status =
        ovvsIvfPqSearch(res, pq.value, x + i * dim, 1, degree + 1, nlist,
                        static_cast<int32_t>(degree) + 1, nullptr, nb.data(), ds.data());
    if (search_status != OVVS_STATUS_SUCCESS) return search_status;
    int filled = 0;
    for (int32_t t = 0; t < degree + 1 && filled < degree; ++t) {
      if (nb[static_cast<size_t>(t)] == i || nb[static_cast<size_t>(t)] < 0) continue;
      graph[static_cast<size_t>(i * degree + filled)] = static_cast<int32_t>(nb[static_cast<size_t>(t)]);
      ++filled;
    }
  }
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus cagra_init_iterative(ResourcesData& r, const float* x, int64_t n, int64_t dim,
                                ovvsMetric metric, int32_t degree,
                                std::vector<int32_t>& graph,
                                NnDescentBuildStats* build_stats) {
  const ovvsStatus init_status =
      nndescent_build(r, x, n, dim, metric, degree, 4, graph, build_stats);
  if (init_status != OVVS_STATUS_SUCCESS) return init_status;
  std::vector<int32_t> next = graph;
  std::vector<int64_t> nb(static_cast<size_t>(degree) + 1);
  std::vector<float> ds(static_cast<size_t>(degree) + 1);
  for (int it = 0; it < 2; ++it) {
    for (int64_t i = 0; i < n; ++i) {
      const ovvsStatus search_status =
          graph_search(r, x, n, dim, metric, graph.data(), degree, x + i * dim, 1, degree + 1,
                       degree * 2, 2, nullptr, nb.data(), ds.data());
      if (search_status != OVVS_STATUS_SUCCESS) return search_status;
      int filled = 0;
      for (int32_t t = 0; t < degree + 1 && filled < degree; ++t) {
        if (nb[static_cast<size_t>(t)] == i || nb[static_cast<size_t>(t)] < 0) continue;
        next[static_cast<size_t>(i * degree + filled)] = static_cast<int32_t>(nb[static_cast<size_t>(t)]);
        ++filled;
      }
      while (filled < degree) {
        next[static_cast<size_t>(i * degree + filled)] = -1;
        ++filled;
      }
    }
    graph.swap(next);
  }
  return OVVS_STATUS_SUCCESS;
}

void robust_prune(const float* x, int64_t dim, ovvsMetric metric, int64_t p, std::vector<int32_t>& cand,
                  int32_t degree, float alpha, int32_t* out_row);

/* Records a graph row exactly as it was before `robust_prune` overwrote it, so an insert that
   fails partway can be undone without having copied the whole index first. */
struct CagraGraphRowBackup {
  int64_t row;
  std::vector<int32_t> edges;
};

ovvsStatus cagra_insert_one(CagraIndex* ix, ResourcesData& r, const float* vec,
                            std::vector<CagraGraphRowBackup>* journal) {
  const int64_t old_n = ix->ds.n;
  const int64_t dim = ix->ds.dim;
  std::vector<int64_t> nb(static_cast<size_t>(ix->degree));
  std::vector<float> ds(static_cast<size_t>(ix->degree));
  const ovvsStatus search_status =
      graph_search(r, ix->ds.x.data(), old_n, dim, ix->ds.metric, ix->graph.data(),
                   ix->degree, vec, 1, ix->degree, ix->degree * 2, 2, nullptr, nb.data(),
                   ds.data());
  if (search_status != OVVS_STATUS_SUCCESS) return search_status;
  ix->ds.x.insert(ix->ds.x.end(), vec, vec + dim);
  ix->ds.n = old_n + 1;
  ix->graph.resize(static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->degree), -1);
  for (int32_t t = 0; t < ix->degree; ++t) {
    ix->graph[static_cast<size_t>(old_n * ix->degree + t)] =
        t < static_cast<int32_t>(nb.size()) ? static_cast<int32_t>(nb[static_cast<size_t>(t)]) : -1;
  }
  for (int32_t t = 0; t < ix->degree; ++t) {
    const int32_t u = ix->graph[static_cast<size_t>(old_n * ix->degree + t)];
    if (u < 0) continue;
    if (journal) {
      const auto row_begin = ix->graph.begin() + static_cast<int64_t>(u) * ix->degree;
      journal->push_back({static_cast<int64_t>(u),
                          std::vector<int32_t>(row_begin, row_begin + ix->degree)});
    }
    std::vector<int32_t> cand;
    for (int32_t e = 0; e < ix->degree; ++e) {
      const int32_t v = ix->graph[static_cast<size_t>(static_cast<int64_t>(u) * ix->degree + e)];
      if (v >= 0) cand.push_back(v);
    }
    cand.push_back(static_cast<int32_t>(old_n));
    robust_prune(ix->ds.x.data(), dim, ix->ds.metric, u, cand, ix->degree, 1.2f,
                 ix->graph.data() + static_cast<int64_t>(u) * ix->degree);
  }
  return OVVS_STATUS_SUCCESS;
}

void robust_prune(const float* x, int64_t dim, ovvsMetric metric, int64_t p, std::vector<int32_t>& cand,
                  int32_t degree, float alpha, int32_t* out_row) {
  std::sort(cand.begin(), cand.end());
  cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
  std::sort(cand.begin(), cand.end(), [&](int32_t a, int32_t b) {
    return distance_one(metric, x + p * dim, x + static_cast<int64_t>(a) * dim, dim, 2.f) <
           distance_one(metric, x + p * dim, x + static_cast<int64_t>(b) * dim, dim, 2.f);
  });
  std::vector<int32_t> kept;
  for (int32_t v : cand) {
    if (v == static_cast<int32_t>(p) || v < 0) continue;
    bool ok = true;
    const float dpv = distance_one(metric, x + p * dim, x + static_cast<int64_t>(v) * dim, dim, 2.f);
    for (int32_t u : kept) {
      const float duv = distance_one(metric, x + static_cast<int64_t>(u) * dim,
                                     x + static_cast<int64_t>(v) * dim, dim, 2.f);
      if (alpha * duv <= dpv) {
        ok = false;
        break;
      }
    }
    if (ok) kept.push_back(v);
    if (static_cast<int32_t>(kept.size()) >= degree) break;
  }
  std::fill(out_row, out_row + degree, -1);
  for (size_t t = 0; t < kept.size(); ++t) out_row[t] = kept[t];
}

}  // namespace

ovvsStatus ovvs::impl::cagra_optimize_ranked(const int32_t* initial, int64_t n,
                                              int32_t initial_degree, int32_t final_degree,
                                              std::vector<int32_t>& output) {
  output.clear();
  if (!initial || n <= 1 || initial_degree <= 0 || final_degree <= 0 ||
      final_degree > initial_degree || initial_degree >= n || final_degree >= n) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  if (n > std::numeric_limits<int32_t>::max()) return OVVS_STATUS_SHAPE_MISMATCH;

  const size_t node_count = static_cast<size_t>(n);
  const size_t initial_width = static_cast<size_t>(initial_degree);
  const size_t final_width = static_cast<size_t>(final_degree);
  if (node_count > std::numeric_limits<size_t>::max() / initial_width ||
      node_count > std::numeric_limits<size_t>::max() / final_width) {
    return OVVS_STATUS_SHAPE_MISMATCH;
  }
  const size_t initial_count = node_count * initial_width;
  const size_t final_count = node_count * final_width;

  try {
    /* Sorted packed (neighbor ID, original rank) rows provide bounded rank lookup without
       allocating a dense N-by-N table. */
    std::vector<uint64_t> rank_lookup(initial_count);
    for (size_t row = 0; row < node_count; ++row) {
      const size_t row_base = row * initial_width;
      for (size_t rank = 0; rank < initial_width; ++rank) {
        const int32_t id = initial[row_base + rank];
        if (id < 0 || static_cast<int64_t>(id) >= n || static_cast<size_t>(id) == row) {
          return OVVS_STATUS_ERROR;
        }
        rank_lookup[row_base + rank] =
            (static_cast<uint64_t>(static_cast<uint32_t>(id)) << 32) |
            static_cast<uint32_t>(rank);
      }
      auto begin = rank_lookup.begin() + static_cast<std::ptrdiff_t>(row_base);
      auto end = begin + initial_degree;
      std::sort(begin, end);
      for (auto it = begin + 1; it != end; ++it) {
        if ((*it >> 32) == (*(it - 1) >> 32)) return OVVS_STATUS_ERROR;
      }
    }

    const auto rank_of = [&](size_t row, int32_t id) {
      const size_t row_base = row * initial_width;
      const auto begin = rank_lookup.begin() + static_cast<std::ptrdiff_t>(row_base);
      const auto end = begin + initial_degree;
      const uint64_t needle = static_cast<uint64_t>(static_cast<uint32_t>(id)) << 32;
      const auto it = std::lower_bound(begin, end, needle);
      if (it == end || (*it >> 32) != (needle >> 32)) {
        return static_cast<uint32_t>(initial_degree);
      }
      return static_cast<uint32_t>(*it);
    };

    struct RankedCandidate {
      uint32_t detours;
      uint32_t initial_rank;
      int32_t id;
    };
    const auto candidate_less = [](const RankedCandidate& a, const RankedCandidate& b) {
      if (a.detours != b.detours) return a.detours < b.detours;
      if (a.initial_rank != b.initial_rank) return a.initial_rank < b.initial_rank;
      return a.id < b.id;
    };

    std::vector<int32_t> forward(final_count);
    std::vector<RankedCandidate> candidates(initial_width);
    for (size_t row = 0; row < node_count; ++row) {
      const size_t initial_base = row * initial_width;
      for (uint32_t y_rank = 0; y_rank < static_cast<uint32_t>(initial_degree); ++y_rank) {
        const int32_t y = initial[initial_base + y_rank];
        uint32_t detours = 0;
        for (uint32_t z_rank = 0; z_rank < y_rank; ++z_rank) {
          const int32_t z = initial[initial_base + z_rank];
          /* z_rank < y_rank is already strict, so the paper's
             max(rank_X(z), rank_z(y)) < rank_X(y) reduces to this check. */
          if (rank_of(static_cast<size_t>(z), y) < y_rank) ++detours;
        }
        candidates[y_rank] = {detours, y_rank, y};
      }
      std::partial_sort(candidates.begin(), candidates.begin() + final_degree, candidates.end(),
                        candidate_less);
      const size_t forward_base = row * final_width;
      for (size_t rank = 0; rank < final_width; ++rank) {
        forward[forward_base + rank] = candidates[rank].id;
      }
    }

    /* Store reverse candidates as a flat CSR graph. Keys sort by the source edge's
       post-prune forward rank and then source ID, giving a total deterministic order. */
    std::vector<size_t> reverse_counts(node_count, 0);
    for (size_t source = 0; source < node_count; ++source) {
      const size_t row_base = source * final_width;
      for (size_t rank = 0; rank < final_width; ++rank) {
        ++reverse_counts[static_cast<size_t>(forward[row_base + rank])];
      }
    }
    std::vector<size_t> reverse_offsets(node_count + 1, 0);
    for (size_t row = 0; row < node_count; ++row) {
      reverse_offsets[row + 1] = reverse_offsets[row] + reverse_counts[row];
    }
    if (reverse_offsets.back() != final_count) return OVVS_STATUS_ERROR;

    std::vector<uint64_t> reverse_keys(final_count);
    std::fill(reverse_counts.begin(), reverse_counts.end(), 0);
    for (size_t source = 0; source < node_count; ++source) {
      const size_t row_base = source * final_width;
      for (uint32_t rank = 0; rank < static_cast<uint32_t>(final_degree); ++rank) {
        const size_t destination = static_cast<size_t>(forward[row_base + rank]);
        const size_t slot = reverse_offsets[destination] + reverse_counts[destination]++;
        reverse_keys[slot] = (static_cast<uint64_t>(rank) << 32) |
                             static_cast<uint32_t>(source);
      }
    }

    for (size_t row = 0; row < node_count; ++row) {
      const size_t begin_offset = reverse_offsets[row];
      const size_t end_offset = reverse_offsets[row + 1];
      auto begin = reverse_keys.begin() + static_cast<std::ptrdiff_t>(begin_offset);
      auto end = reverse_keys.begin() + static_cast<std::ptrdiff_t>(end_offset);
      std::sort(begin, end);
      size_t accepted = 0;
      const auto forward_begin = forward.begin() + static_cast<std::ptrdiff_t>(row * final_width);
      const auto forward_end = forward_begin + final_degree;
      for (auto it = begin; it != end && accepted < final_width; ++it) {
        const int32_t source = static_cast<int32_t>(static_cast<uint32_t>(*it));
        if (std::find(forward_begin, forward_end, source) != forward_end) continue;
        reverse_keys[begin_offset + accepted++] = *it;
      }
      reverse_counts[row] = accepted;
    }

    std::vector<int32_t> forward_row(final_width);
    for (size_t row = 0; row < node_count; ++row) {
      const size_t row_base = row * final_width;
      std::copy_n(forward.begin() + static_cast<std::ptrdiff_t>(row_base), final_degree,
                  forward_row.begin());
      const size_t reverse_take = std::min(final_width / 2, reverse_counts[row]);
      const size_t forward_take = final_width - reverse_take;
      size_t forward_pos = 0;
      size_t reverse_pos = 0;
      size_t output_pos = 0;
      while (forward_pos < forward_take || reverse_pos < reverse_take) {
        if (forward_pos < forward_take) {
          forward[row_base + output_pos++] = forward_row[forward_pos++];
        }
        if (reverse_pos < reverse_take) {
          const uint64_t key = reverse_keys[reverse_offsets[row] + reverse_pos++];
          forward[row_base + output_pos++] =
              static_cast<int32_t>(static_cast<uint32_t>(key));
        }
      }
      if (output_pos != final_width) return OVVS_STATUS_ERROR;
    }

    output.swap(forward);
    return OVVS_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return OVVS_STATUS_OOM;
  } catch (...) {
    return OVVS_STATUS_ERROR;
  }
}

ovvsStatus ovvsIvfRabitqBuild(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                              ovvsMetric metric, int32_t nlist, ovvsIvfRabitqIndex_t* index) {
  if (!res || !dataset || !index || n <= 0 || dim <= 0 || nlist <= 0) return OVVS_STATUS_INVALID_ARGUMENT;
  nlist = std::min(nlist, static_cast<int32_t>(n));
  auto* ix = new IvfRabitq();
  copy_ds(ix->ds, dataset, n, dim, metric);
  ix->nlist = nlist;
  ix->nbytes = nbytes_bits(dim);
  ix->rand_sign.resize(static_cast<size_t>(dim));
  auto rng = rng_from(99);
  std::bernoulli_distribution coin(0.5);
  for (int64_t i = 0; i < dim; ++i) ix->rand_sign[static_cast<size_t>(i)] = coin(rng) ? 1 : 0;
  kmeans_fit_impl(*rd(res), dataset, n, dim, nlist, 12, ix->centroids);
  std::vector<int64_t> labels(static_cast<size_t>(n));
  std::vector<float> d(static_cast<size_t>(n));
  std::vector<float> scores(static_cast<size_t>(n) * static_cast<size_t>(nlist));
  prim_pairwise(*rd(res), OVVS_METRIC_L2_EXPANDED, dataset, n, ix->centroids.data(), nlist, dim,
                scores.data(), 2.f);
  prim_topk(*rd(res), scores.data(), n, nlist, 1, labels.data(), d.data(), false);
  ix->assign.resize(static_cast<size_t>(n));
  ix->scales.resize(static_cast<size_t>(n));
  ix->signs.assign(static_cast<size_t>(n * ix->nbytes), 0);
  std::vector<float> resid(static_cast<size_t>(dim));
  for (int64_t i = 0; i < n; ++i) {
    const int32_t a = static_cast<int32_t>(labels[static_cast<size_t>(i)]);
    ix->assign[static_cast<size_t>(i)] = a;
    const float* c = ix->centroids.data() + static_cast<size_t>(a) * dim;
    for (int64_t t = 0; t < dim; ++t) resid[static_cast<size_t>(t)] = dataset[i * dim + t] - c[t];
    ix->scales[static_cast<size_t>(i)] = std::sqrt(std::max(nrm2sq(resid.data(), dim), 1e-12f));
    pack_signs(resid.data(), dim, ix->rand_sign.data(), ix->signs.data() + i * ix->nbytes);
  }
  build_ivf_lists(ix->assign.data(), n, nlist, ix->lists);
  *index = reinterpret_cast<ovvsIvfRabitqIndex_t>(ix);
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsIvfRabitqSearch(ovvsResources_t res, ovvsIvfRabitqIndex_t index, const float* queries,
                               int64_t nq, int64_t k, int32_t nprobe, int32_t krefine,
                               const uint8_t* bitset, int64_t* neighbors, float* distances) {
  if (!res || !index || !queries || !neighbors || !distances) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<IvfRabitq*>(index);
  nprobe = std::max(1, std::min(nprobe, ix->nlist));
  if (krefine < static_cast<int32_t>(k)) krefine = static_cast<int32_t>(k);
  std::vector<int64_t> cl(static_cast<size_t>(nq) * static_cast<size_t>(nprobe));
  std::vector<float> cd(static_cast<size_t>(nq) * static_cast<size_t>(nprobe));
  std::vector<float> cscores(static_cast<size_t>(nq) * static_cast<size_t>(ix->nlist));
  prim_pairwise(*rd(res), OVVS_METRIC_L2_EXPANDED, queries, nq, ix->centroids.data(), ix->nlist,
                ix->ds.dim, cscores.data(), 2.f);
  prim_topk(*rd(res), cscores.data(), nq, ix->nlist, nprobe, cl.data(), cd.data(), false);
  const int64_t dim = ix->ds.dim;
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
      const float qn = nrm2sq(qres.data(), dim);
      for (int64_t id : ix->lists[static_cast<size_t>(c)]) {
        if (bitset && !allowed(bitset, id)) continue;
        const float ip =
            rabitq_ip(qres.data(), ix->signs.data() + id * ix->nbytes, ix->rand_sign.data(), dim,
                      ix->scales[static_cast<size_t>(id)]);
        const float sc = qn + ix->scales[static_cast<size_t>(id)] * ix->scales[static_cast<size_t>(id)] -
                         2.f * ip;
        ids.push_back(id);
        adc.push_back(sc);
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
        float dv = fv[static_cast<size_t>(t)];
        if (ix->ds.metric == OVVS_METRIC_INNER_PRODUCT) dv = -dv;
        distances[q * k + t] = dv;
      } else {
        neighbors[q * k + t] = -1;
        distances[q * k + t] = kInf;
      }
    }
  }
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsIvfRabitqDestroy(ovvsIvfRabitqIndex_t index) {
  delete reinterpret_cast<IvfRabitq*>(index);
  return OVVS_STATUS_SUCCESS;
}

constexpr uint32_t kRabitqMagic = 0x31425152u; /* 'RQB1' */

ovvsStatus ovvsIvfRabitqSerialize(ovvsIvfRabitqIndex_t index, const char* path) {
  if (!index || !path) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<IvfRabitq*>(index);
  std::ofstream f(path, std::ios::binary);
  if (!f) return OVVS_STATUS_IO;
  f.write(reinterpret_cast<const char*>(&kRabitqMagic), 4);
  int32_t ver = 1;
  f.write(reinterpret_cast<const char*>(&ver), 4);
  f.write(reinterpret_cast<const char*>(&ix->ds.n), 8);
  f.write(reinterpret_cast<const char*>(&ix->ds.dim), 8);
  f.write(reinterpret_cast<const char*>(&ix->nlist), 4);
  f.write(reinterpret_cast<const char*>(&ix->nbytes), 8);
  int32_t metric = static_cast<int32_t>(ix->ds.metric);
  f.write(reinterpret_cast<const char*>(&metric), 4);
  f.write(reinterpret_cast<const char*>(ix->centroids.data()),
          static_cast<std::streamsize>(ix->centroids.size() * sizeof(float)));
  f.write(reinterpret_cast<const char*>(ix->rand_sign.data()), static_cast<std::streamsize>(ix->rand_sign.size()));
  f.write(reinterpret_cast<const char*>(ix->scales.data()),
          static_cast<std::streamsize>(ix->scales.size() * sizeof(float)));
  f.write(reinterpret_cast<const char*>(ix->signs.data()), static_cast<std::streamsize>(ix->signs.size()));
  f.write(reinterpret_cast<const char*>(ix->assign.data()),
          static_cast<std::streamsize>(ix->assign.size() * sizeof(int32_t)));
  f.write(reinterpret_cast<const char*>(ix->ds.x.data()),
          static_cast<std::streamsize>(ix->ds.x.size() * sizeof(float)));
  for (int32_t c = 0; c < ix->nlist; ++c) {
    int32_t sz = static_cast<int32_t>(ix->lists[static_cast<size_t>(c)].size());
    f.write(reinterpret_cast<const char*>(&sz), 4);
    if (sz > 0)
      f.write(reinterpret_cast<const char*>(ix->lists[static_cast<size_t>(c)].data()),
              static_cast<std::streamsize>(static_cast<size_t>(sz) * sizeof(int64_t)));
  }
  return f.good() ? OVVS_STATUS_SUCCESS : OVVS_STATUS_IO;
}

ovvsStatus ovvsIvfRabitqDeserialize(ovvsResources_t res, const char* path, ovvsIvfRabitqIndex_t* index) {
  if (!res || !path || !index) return OVVS_STATUS_INVALID_ARGUMENT;
  std::ifstream f(path, std::ios::binary);
  if (!f) return OVVS_STATUS_IO;
  uint32_t magic = 0;
  f.read(reinterpret_cast<char*>(&magic), 4);
  if (magic != kRabitqMagic) return OVVS_STATUS_IO;
  int32_t ver = 0;
  f.read(reinterpret_cast<char*>(&ver), 4);
  auto* ix = new IvfRabitq();
  f.read(reinterpret_cast<char*>(&ix->ds.n), 8);
  f.read(reinterpret_cast<char*>(&ix->ds.dim), 8);
  f.read(reinterpret_cast<char*>(&ix->nlist), 4);
  f.read(reinterpret_cast<char*>(&ix->nbytes), 8);
  int32_t metric = 0;
  f.read(reinterpret_cast<char*>(&metric), 4);
  ix->ds.metric = static_cast<ovvsMetric>(metric);
  ix->centroids.resize(static_cast<size_t>(ix->nlist) * static_cast<size_t>(ix->ds.dim));
  ix->rand_sign.resize(static_cast<size_t>(ix->ds.dim));
  ix->scales.resize(static_cast<size_t>(ix->ds.n));
  ix->signs.resize(static_cast<size_t>(ix->ds.n * ix->nbytes));
  ix->assign.resize(static_cast<size_t>(ix->ds.n));
  ix->ds.x.resize(static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->ds.dim));
  f.read(reinterpret_cast<char*>(ix->centroids.data()),
         static_cast<std::streamsize>(ix->centroids.size() * sizeof(float)));
  f.read(reinterpret_cast<char*>(ix->rand_sign.data()), static_cast<std::streamsize>(ix->rand_sign.size()));
  f.read(reinterpret_cast<char*>(ix->scales.data()),
         static_cast<std::streamsize>(ix->scales.size() * sizeof(float)));
  f.read(reinterpret_cast<char*>(ix->signs.data()), static_cast<std::streamsize>(ix->signs.size()));
  f.read(reinterpret_cast<char*>(ix->assign.data()),
         static_cast<std::streamsize>(ix->assign.size() * sizeof(int32_t)));
  f.read(reinterpret_cast<char*>(ix->ds.x.data()),
         static_cast<std::streamsize>(ix->ds.x.size() * sizeof(float)));
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
    return OVVS_STATUS_IO;
  }
  *index = reinterpret_cast<ovvsIvfRabitqIndex_t>(ix);
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsIvfRabitqExtend(ovvsResources_t res, ovvsIvfRabitqIndex_t index, const float* extra,
                               int64_t nextra) {
  if (!res || !index || !extra || nextra <= 0) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<IvfRabitq*>(index);
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
  ix->assign.resize(static_cast<size_t>(ix->ds.n));
  ix->scales.resize(static_cast<size_t>(ix->ds.n));
  ix->signs.resize(static_cast<size_t>(ix->ds.n * ix->nbytes));
  std::vector<float> resid(static_cast<size_t>(dim));
  for (int64_t i = 0; i < nextra; ++i) {
    const int32_t a = static_cast<int32_t>(labels[static_cast<size_t>(i)]);
    ix->assign[static_cast<size_t>(old_n + i)] = a;
    if (a >= 0 && a < ix->nlist) ix->lists[static_cast<size_t>(a)].push_back(old_n + i);
    const float* c = ix->centroids.data() + static_cast<size_t>(std::max(a, 0)) * dim;
    for (int64_t t = 0; t < dim; ++t) resid[static_cast<size_t>(t)] = extra[i * dim + t] - c[t];
    ix->scales[static_cast<size_t>(old_n + i)] = std::sqrt(std::max(nrm2sq(resid.data(), dim), 1e-12f));
    pack_signs(resid.data(), dim, ix->rand_sign.data(), ix->signs.data() + (old_n + i) * ix->nbytes);
  }
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsNnDescentBuild(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                              ovvsMetric metric, int32_t graph_degree, int32_t iterations,
                              ovvsNnDescentGraph_t* graph) {
  if (!graph) return OVVS_STATUS_INVALID_ARGUMENT;
  *graph = nullptr;
  if (!res || !dataset || n <= 1 || dim <= 0 || graph_degree <= 0) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  if (n > std::numeric_limits<int32_t>::max() || dim > std::numeric_limits<int64_t>::max() / n) {
    return OVVS_STATUS_SHAPE_MISMATCH;
  }
  if (metric < OVVS_METRIC_L2_EXPANDED || metric > OVVS_METRIC_LP_UNEXPANDED) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  try {
    auto g = std::make_unique<NnGraph>();
    g->n = n;
    g->degree = std::min(graph_degree, static_cast<int32_t>(n - 1));
    const ovvsStatus status =
        nndescent_build(*rd(res), dataset, n, dim, metric, g->degree,
                        std::max(1, iterations), g->ids);
    if (status != OVVS_STATUS_SUCCESS) return status;
    *graph = reinterpret_cast<ovvsNnDescentGraph_t>(g.release());
    return OVVS_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return OVVS_STATUS_OOM;
  } catch (...) {
    return OVVS_STATUS_ERROR;
  }
}

ovvsStatus ovvsNnDescentNeighbors(ovvsNnDescentGraph_t graph, const int32_t** ids, int64_t* n,
                                  int32_t* degree) {
  if (!graph || !ids || !n || !degree) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* g = reinterpret_cast<NnGraph*>(graph);
  *ids = g->ids.data();
  *n = g->n;
  *degree = g->degree;
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsNnDescentDestroy(ovvsNnDescentGraph_t graph) {
  delete reinterpret_cast<NnGraph*>(graph);
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsCagraBuildEx(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                            ovvsMetric metric, int32_t graph_degree, int32_t intermediate_degree,
                            ovvsCagraBuildAlgo algo, ovvsCagraIndex_t* index) {
  if (!index) return OVVS_STATUS_INVALID_ARGUMENT;
  *index = nullptr;
  if (!res || !dataset || n <= 1 || dim <= 0) return OVVS_STATUS_INVALID_ARGUMENT;
  if (n > std::numeric_limits<int32_t>::max() || dim > std::numeric_limits<int64_t>::max() / n) {
    return OVVS_STATUS_SHAPE_MISMATCH;
  }
  if (metric < OVVS_METRIC_L2_EXPANDED || metric > OVVS_METRIC_LP_UNEXPANDED ||
      algo < OVVS_CAGRA_BUILD_NN_DESCENT || algo > OVVS_CAGRA_BUILD_ITERATIVE) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  auto* resources = rd(res);
  if (resources->policy == OVVS_POLICY_FORCE_NPU) {
    ++resources->npu_fallbacks;
    return OVVS_STATUS_DEVICE_UNAVAILABLE;
  }
  if (resources->policy == OVVS_POLICY_FORCE_GPU &&
      algo != OVVS_CAGRA_BUILD_NN_DESCENT) {
    return OVVS_STATUS_DEVICE_UNAVAILABLE;
  }
  graph_degree = std::max(1, std::min(graph_degree, static_cast<int32_t>(n - 1)));
  intermediate_degree = std::max(graph_degree, std::min(intermediate_degree, static_cast<int32_t>(n - 1)));
  const auto total_begin = BuildClock::now();
  try {
    auto ix = std::make_unique<CagraIndex>();
    ovvsCagraBuildStatsV1 call_stats{};
    call_stats.successful_calls = 1;
    call_stats.rows = n;
    call_stats.dataset_copy_bytes = logical_bytes(n, dim, sizeof(float));
    call_stats.initializer_graph_payload_bytes =
        logical_bytes(n, intermediate_degree, sizeof(int32_t));
    call_stats.published_graph_copy_bytes =
        logical_bytes(n, graph_degree, sizeof(int32_t));
    if (algo == OVVS_CAGRA_BUILD_NN_DESCENT) {
      call_stats.nndescent_initializer_calls = 1;
    } else if (algo == OVVS_CAGRA_BUILD_IVF_PQ) {
      call_stats.ivfpq_initializer_calls = 1;
    } else {
      call_stats.iterative_initializer_calls = 1;
    }

    const auto dataset_begin = BuildClock::now();
    copy_ds(ix->ds, dataset, n, dim, metric);
    if (!std::all_of(ix->ds.x.begin(), ix->ds.x.end(),
                     [](float value) { return std::isfinite(value); })) {
      return OVVS_STATUS_INVALID_ARGUMENT;
    }
    call_stats.dataset_copy_ns = elapsed_ns(dataset_begin);
    ix->degree = graph_degree;
    ix->has_dataset = true;
    std::vector<int32_t> init;
    NnDescentBuildStats nndescent_stats{};
    ovvsStatus status = OVVS_STATUS_SUCCESS;
    const auto initializer_begin = BuildClock::now();
    if (algo == OVVS_CAGRA_BUILD_IVF_PQ) {
      status = cagra_init_ivfpq(*rd(res), ix->ds.x.data(), n, dim, metric,
                                intermediate_degree, init);
    } else if (algo == OVVS_CAGRA_BUILD_ITERATIVE) {
      status = cagra_init_iterative(*rd(res), ix->ds.x.data(), n, dim, metric,
                                    intermediate_degree, init, &nndescent_stats);
    } else {
      status = nndescent_build(*rd(res), ix->ds.x.data(), n, dim, metric,
                               intermediate_degree, 6, init, &nndescent_stats);
    }
    if (status != OVVS_STATUS_SUCCESS) return status;
    call_stats.initializer_ns = elapsed_ns(initializer_begin);
    const ovvsDevice initializer_device = resources->last_device;
    if (initializer_device == OVVS_DEVICE_CPU) {
      call_stats.initializer_final_cpu_calls = 1;
    } else if (initializer_device == OVVS_DEVICE_GPU) {
      call_stats.initializer_final_gpu_calls = 1;
    } else if (initializer_device == OVVS_DEVICE_NPU) {
      call_stats.initializer_final_npu_calls = 1;
    }
    if (nndescent_stats.gpu_instrumented) {
      call_stats.nndescent_gpu_iterations = nndescent_stats.iterations;
      call_stats.nndescent_gpu_converged_calls = nndescent_stats.converged ? 1 : 0;
      call_stats.nndescent_gpu_final_changed_edges = nndescent_stats.final_changed_edges;
      call_stats.nndescent_gpu_final_pending_new_edges =
          nndescent_stats.final_pending_new_edges;
      call_stats.nndescent_gpu_instrumented_calls = 1;
      call_stats.nndescent_gpu_allocation_calls = nndescent_stats.gpu.allocation_calls;
      call_stats.nndescent_gpu_allocation_bytes = nndescent_stats.gpu.allocation_bytes;
      call_stats.nndescent_gpu_h2d_calls = nndescent_stats.gpu.h2d_calls;
      call_stats.nndescent_gpu_h2d_bytes = nndescent_stats.gpu.h2d_bytes;
      call_stats.nndescent_gpu_d2h_calls = nndescent_stats.gpu.d2h_calls;
      call_stats.nndescent_gpu_d2h_bytes = nndescent_stats.gpu.d2h_bytes;
      call_stats.nndescent_gpu_kernel_launches = nndescent_stats.gpu.kernel_launches;
      call_stats.nndescent_gpu_submission_calls = nndescent_stats.submission_calls;
      call_stats.nndescent_gpu_wait_calls = nndescent_stats.gpu.wait_calls;
      call_stats.nndescent_gpu_peak_owned_bytes_max = nndescent_stats.peak_owned_bytes;
    }

    std::vector<int32_t> pruned;
    const auto optimizer_begin = BuildClock::now();
    status = prim_cagra_optimize_ranked(*resources, init.data(), n,
                                        intermediate_degree, graph_degree, pruned);
    if (status == OVVS_STATUS_UNSUPPORTED) {
      status = cagra_optimize_ranked(init.data(), n, intermediate_degree,
                                     graph_degree, pruned);
      if (status == OVVS_STATUS_SUCCESS) resources->last_device = OVVS_DEVICE_CPU;
    }
    if (status != OVVS_STATUS_SUCCESS) return status;
    call_stats.optimizer_prune_merge_ns = elapsed_ns(optimizer_begin);

    const auto materialize_begin = BuildClock::now();
    ix->graph.assign(pruned.begin(), pruned.end());
    call_stats.index_materialize_ns = elapsed_ns(materialize_begin);
    /* The public total is the inner build wall through persistent materialization.
       Telemetry merge, handle publication, and serialization are outside it. */
    call_stats.total_wall_ns = elapsed_ns(total_begin);
    publish_cagra_build_stats(*resources, call_stats);
    cagra_finalize_storage(ix.get(), /*allow_lossy=*/true);
    *index = reinterpret_cast<ovvsCagraIndex_t>(ix.release());
    return OVVS_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return OVVS_STATUS_OOM;
  } catch (...) {
    return OVVS_STATUS_ERROR;
  }
}

ovvsStatus ovvsCagraBuild(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                          ovvsMetric metric, int32_t graph_degree, int32_t intermediate_degree,
                          ovvsCagraIndex_t* index) {
  return ovvsCagraBuildEx(res, dataset, n, dim, metric, graph_degree, intermediate_degree,
                          OVVS_CAGRA_BUILD_NN_DESCENT, index);
}

ovvsStatus ovvsCagraSearch(ovvsResources_t res, ovvsCagraIndex_t index, const float* queries,
                           int64_t nq, int64_t k, int32_t itopk_size, int32_t search_width,
                           const uint8_t* bitset, int64_t* neighbors, float* distances) {
  if (!res || !index || !queries || !neighbors || !distances || nq <= 0 || k <= 0 ||
      k > std::numeric_limits<int32_t>::max()) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  if (nq > std::numeric_limits<int64_t>::max() / k) return OVVS_STATUS_SHAPE_MISMATCH;
  try {
    auto* ix = reinterpret_cast<CagraIndex*>(index);
    if (ix->pq_m > 0 && !ix->codes.empty()) {
    auto* resources = rd(res);
    if (resources->policy == OVVS_POLICY_FORCE_GPU || resources->policy == OVVS_POLICY_FORCE_NPU) {
      if (resources->policy == OVVS_POLICY_FORCE_NPU) ++resources->npu_fallbacks;
      return OVVS_STATUS_DEVICE_UNAVAILABLE;
    }
    graph_search_pq(ix->ds.x.empty() ? nullptr : ix->ds.x.data(), ix->ds.n, ix->ds.dim, ix->graph.data(),
                    ix->degree, ix->codes.data(), ix->pq_m, ix->pq_ks, ix->dsub, ix->codebooks.data(),
                    queries, nq, k, itopk_size, search_width, bitset, neighbors, distances);
    resources->last_device = OVVS_DEVICE_CPU;
    if (ix->has_dataset && !ix->ds.x.empty()) {
      const int64_t dim = ix->ds.dim;
      for (int64_t q = 0; q < nq; ++q) {
        std::vector<int64_t> cand(static_cast<size_t>(k));
        int64_t nc = 0;
        for (int64_t t = 0; t < k; ++t) {
          const int64_t candidate = neighbors[q * k + t];
          if (candidate >= 0 && !cagra_row_deleted(*ix, candidate)) {
            cand[static_cast<size_t>(nc++)] = candidate;
          }
        }
        if (nc == 0) continue;
        std::vector<float> gathered(static_cast<size_t>(nc) * static_cast<size_t>(dim));
        ovvsStatus status = prim_gather_rows(*rd(res), ix->ds.x.data(), ix->ds.n, dim,
                                             cand.data(), nc, gathered.data());
        if (status != OVVS_STATUS_SUCCESS) return status;
        std::vector<float> sc(static_cast<size_t>(nc));
        status = prim_pairwise(*rd(res), ix->ds.metric, queries + q * dim, 1,
                               gathered.data(), nc, dim, sc.data(), 2.f);
        if (status != OVVS_STATUS_SUCCESS) return status;
        std::vector<int64_t> fi(static_cast<size_t>(nc));
        std::vector<float> fv(static_cast<size_t>(nc));
        status = prim_topk(*rd(res), sc.data(), 1, nc, nc, fi.data(), fv.data(),
                           metric_largest(ix->ds.metric));
        if (status != OVVS_STATUS_SUCCESS) return status;
        for (int64_t t = 0; t < k; ++t) {
          if (t < nc) {
            const int64_t selected = fi[static_cast<size_t>(t)];
            if (selected < 0 || selected >= nc) return OVVS_STATUS_ERROR;
            neighbors[q * k + t] = cagra_pack_id(*ix, cand[static_cast<size_t>(selected)]);
            float d = fv[static_cast<size_t>(t)];
            if (ix->ds.metric == OVVS_METRIC_INNER_PRODUCT) d = -d;
            distances[q * k + t] = d;
          } else {
            neighbors[q * k + t] = -1;
            distances[q * k + t] = kInf;
          }
        }
      }
    }
      return OVVS_STATUS_SUCCESS;
    }
    if (!cagra_dataset_present(*ix)) return OVVS_STATUS_INVALID_ARGUMENT;
    const CagraStorage st = cagra_storage(*ix);
    if (ix->deleted_count > 0) {
      /* Tombstoned rows are traversed like any other -- they are routing hops for their
         neighbours, and dropping them from the walk would fragment the graph -- so they are
         removed here instead, after the search. Over-fetch first so a result set thinned by
         deletions still yields k live rows; the extra depth is capped at itopk_size so the walk
         does not silently do more work than the caller asked for. */
      const int64_t widened = std::max<int64_t>(k * 2, k + 16);
      const int64_t k_internal = std::min<int64_t>(widened, std::max<int64_t>(itopk_size, k));
      if (k_internal > std::numeric_limits<int64_t>::max() / nq) return OVVS_STATUS_SHAPE_MISMATCH;
      std::vector<int64_t> wide_ids(static_cast<size_t>(nq * k_internal));
      std::vector<float> wide_distances(static_cast<size_t>(nq * k_internal));
      const ovvsStatus status = prim_graph_walk(
          *rd(res), st.f32, ix->ds.n, ix->ds.dim, ix->ds.metric, ix->graph.data(),
          ix->degree, queries, nq, k_internal, itopk_size, search_width, bitset, wide_ids.data(),
          wide_distances.data(), st.f16, st.u8);
      if (status != OVVS_STATUS_SUCCESS) return status;
      for (int64_t q = 0; q < nq; ++q) {
        int64_t kept = 0;
        for (int64_t t = 0; t < k_internal && kept < k; ++t) {
          const int64_t candidate = wide_ids[static_cast<size_t>(q * k_internal + t)];
          if (candidate < 0 || cagra_row_deleted(*ix, candidate)) continue;
          neighbors[q * k + kept] = cagra_pack_id(*ix, candidate);
          distances[q * k + kept] = wide_distances[static_cast<size_t>(q * k_internal + t)];
          ++kept;
        }
        for (int64_t t = kept; t < k; ++t) {
          neighbors[q * k + t] = -1;
          distances[q * k + t] = kInf;
        }
      }
      return OVVS_STATUS_SUCCESS;
    }
    const ovvsStatus status = prim_graph_walk(
        *rd(res), st.f32, ix->ds.n, ix->ds.dim, ix->ds.metric, ix->graph.data(),
        ix->degree, queries, nq, k, itopk_size, search_width, bitset, neighbors, distances,
        st.f16, st.u8);
    if (status == OVVS_STATUS_SUCCESS && !ix->generation.empty()) {
      for (int64_t t = 0; t < nq * k; ++t) {
        if (neighbors[t] >= 0) neighbors[t] = cagra_pack_id(*ix, neighbors[t]);
      }
    }
    return status;
  } catch (const std::bad_alloc&) {
    return OVVS_STATUS_OOM;
  } catch (...) {
    return OVVS_STATUS_ERROR;
  }
}

ovvsStatus ovvsCagraSerializeEx(ovvsCagraIndex_t index, const char* path, int32_t include_dataset) {
  if (!index || !path) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<CagraIndex*>(index);
  std::ofstream f(path, std::ios::binary);
  if (!f) return OVVS_STATUS_IO;
  f.write(reinterpret_cast<const char*>(&kCagraMagic), 4);
  /* The version only moves when tombstones are actually present. An index that has never been
     deleted from still writes v2 and stays readable by existing builds; one that HAS been deleted
     from writes v3, so an older build rejects the file outright instead of silently ignoring the
     tombstone section and resurrecting deleted rows. Failing closed is the point. */
  const bool has_tombstones =
      (ix->deleted_count > 0 && !ix->deleted.empty()) || !ix->generation.empty();
  const bool f16_store = ix->ds.x.empty() && !ix->ds.x16.empty();
  int32_t flags = 0;
  if (include_dataset && ix->has_dataset && (!ix->ds.x.empty() || f16_store)) flags |= 1;
  if (ix->pq_m > 0 && !ix->codes.empty()) flags |= 2;
  if (has_tombstones) flags |= 4;
  if (f16_store && (flags & 1)) flags |= 8;
  /* v4 = fp16 dataset payload; older builds fail closed on it, exactly as v3 made them
     fail closed on tombstones rather than resurrect deleted rows. */
  int32_t ver = (flags & 8) ? 4 : (has_tombstones ? 3 : 2);
  f.write(reinterpret_cast<const char*>(&ver), 4);
  f.write(reinterpret_cast<const char*>(&flags), 4);
  f.write(reinterpret_cast<const char*>(&ix->ds.n), 8);
  f.write(reinterpret_cast<const char*>(&ix->ds.dim), 8);
  f.write(reinterpret_cast<const char*>(&ix->degree), 4);
  int32_t metric = static_cast<int32_t>(ix->ds.metric);
  f.write(reinterpret_cast<const char*>(&metric), 4);
  f.write(reinterpret_cast<const char*>(ix->graph.data()),
          static_cast<std::streamsize>(ix->graph.size() * sizeof(int32_t)));
  if (flags & 1) {
    if (flags & 8) {
      f.write(reinterpret_cast<const char*>(ix->ds.x16.data()),
              static_cast<std::streamsize>(ix->ds.x16.size() * sizeof(uint16_t)));
    } else {
      f.write(reinterpret_cast<const char*>(ix->ds.x.data()),
              static_cast<std::streamsize>(ix->ds.x.size() * sizeof(float)));
    }
  }
  if (flags & 2) {
    f.write(reinterpret_cast<const char*>(&ix->pq_m), 4);
    f.write(reinterpret_cast<const char*>(&ix->pq_ks), 4);
    f.write(reinterpret_cast<const char*>(&ix->dsub), 4);
    f.write(reinterpret_cast<const char*>(ix->codebooks.data()),
            static_cast<std::streamsize>(ix->codebooks.size() * sizeof(float)));
    f.write(reinterpret_cast<const char*>(ix->codes.data()),
            static_cast<std::streamsize>(ix->codes.size()));
  }
  if (flags & 4) {
    f.write(reinterpret_cast<const char*>(&ix->deleted_count), 8);
    int64_t deleted_bytes = static_cast<int64_t>(ix->deleted.size());
    f.write(reinterpret_cast<const char*>(&deleted_bytes), 8);
    f.write(reinterpret_cast<const char*>(ix->deleted.data()),
            static_cast<std::streamsize>(ix->deleted.size()));
    /* Generations must round-trip or reuse would stop detecting stale ids across a reload. */
    int64_t generation_rows = static_cast<int64_t>(ix->generation.size());
    f.write(reinterpret_cast<const char*>(&generation_rows), 8);
    f.write(reinterpret_cast<const char*>(ix->generation.data()),
            static_cast<std::streamsize>(ix->generation.size() * sizeof(uint32_t)));
  }
  return f.good() ? OVVS_STATUS_SUCCESS : OVVS_STATUS_IO;
}

ovvsStatus ovvsCagraSerialize(ovvsCagraIndex_t index, const char* path) {
  return ovvsCagraSerializeEx(index, path, 1);
}

ovvsStatus ovvsCagraDeserialize(ovvsResources_t res, const char* path, ovvsCagraIndex_t* index) {
  if (!res || !path || !index) return OVVS_STATUS_INVALID_ARGUMENT;
  std::ifstream f(path, std::ios::binary);
  if (!f) return OVVS_STATUS_IO;
  uint32_t magic = 0;
  f.read(reinterpret_cast<char*>(&magic), 4);
  if (magic != kCagraMagic) return OVVS_STATUS_IO;
  int32_t ver = 0;
  f.read(reinterpret_cast<char*>(&ver), 4);
  if (ver > 4) return OVVS_STATUS_IO;
  auto* ix = new CagraIndex();
  int32_t flags = 1;
  if (ver >= 2) f.read(reinterpret_cast<char*>(&flags), 4);
  f.read(reinterpret_cast<char*>(&ix->ds.n), 8);
  f.read(reinterpret_cast<char*>(&ix->ds.dim), 8);
  f.read(reinterpret_cast<char*>(&ix->degree), 4);
  int32_t metric = 0;
  f.read(reinterpret_cast<char*>(&metric), 4);
  ix->ds.metric = static_cast<ovvsMetric>(metric);
  ix->graph.resize(static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->degree));
  f.read(reinterpret_cast<char*>(ix->graph.data()),
         static_cast<std::streamsize>(ix->graph.size() * sizeof(int32_t)));
  if (flags & 1) {
    const size_t count = static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->ds.dim);
    if (flags & 8) {
      ix->ds.x16.resize(count);
      f.read(reinterpret_cast<char*>(ix->ds.x16.data()),
             static_cast<std::streamsize>(count * sizeof(uint16_t)));
      cagra_build_mirror8(ix);
    } else {
      ix->ds.x.resize(count);
      f.read(reinterpret_cast<char*>(ix->ds.x.data()),
             static_cast<std::streamsize>(count * sizeof(float)));
    }
    ix->has_dataset = true;
  } else {
    ix->has_dataset = false;
  }
  if (flags & 2) {
    f.read(reinterpret_cast<char*>(&ix->pq_m), 4);
    f.read(reinterpret_cast<char*>(&ix->pq_ks), 4);
    f.read(reinterpret_cast<char*>(&ix->dsub), 4);
    ix->codebooks.resize(static_cast<size_t>(ix->pq_m) * ix->pq_ks * ix->dsub);
    ix->codes.resize(static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->pq_m));
    f.read(reinterpret_cast<char*>(ix->codebooks.data()),
           static_cast<std::streamsize>(ix->codebooks.size() * sizeof(float)));
    f.read(reinterpret_cast<char*>(ix->codes.data()), static_cast<std::streamsize>(ix->codes.size()));
  }
  if (flags & 4) {
    f.read(reinterpret_cast<char*>(&ix->deleted_count), 8);
    int64_t deleted_bytes = 0;
    f.read(reinterpret_cast<char*>(&deleted_bytes), 8);
    int64_t generation_rows = 0;
    const int64_t expected_bytes = (ix->ds.n + 7) / 8;
    if (deleted_bytes < 0 || deleted_bytes > expected_bytes) {
      delete ix;
      return OVVS_STATUS_IO;
    }
    ix->deleted.resize(static_cast<size_t>(deleted_bytes));
    f.read(reinterpret_cast<char*>(ix->deleted.data()),
           static_cast<std::streamsize>(ix->deleted.size()));
    f.read(reinterpret_cast<char*>(&generation_rows), 8);
    if (generation_rows < 0 || generation_rows > ix->ds.n) {
      delete ix;
      return OVVS_STATUS_IO;
    }
    ix->generation.resize(static_cast<size_t>(generation_rows));
    f.read(reinterpret_cast<char*>(ix->generation.data()),
           static_cast<std::streamsize>(ix->generation.size() * sizeof(uint32_t)));
    if (ix->deleted_count < 0 || ix->deleted_count > ix->ds.n) {
      delete ix;
      return OVVS_STATUS_IO;
    }
    /* Tombstoned rows are reusable again after a reload; the free list is derived, not stored. */
    for (int64_t row = 0; row < ix->ds.n; ++row) {
      if (cagra_row_deleted(*ix, row)) ix->free_slots.push_back(static_cast<int32_t>(row));
    }
  }
  if (!f) {
    delete ix;
    return OVVS_STATUS_IO;
  }
  cagra_finalize_storage(ix, /*allow_lossy=*/false); /* fp32 file: lossless-only */
  *index = reinterpret_cast<ovvsCagraIndex_t>(ix);
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsCagraDelete(ovvsResources_t res, ovvsCagraIndex_t index, const int64_t* ids,
                           int64_t nids) {
  if (!res || !index || !ids || nids < 0) return OVVS_STATUS_INVALID_ARGUMENT;
  if (nids == 0) return OVVS_STATUS_SUCCESS;
  auto* ix = reinterpret_cast<CagraIndex*>(index);
  std::vector<int64_t> slots(static_cast<size_t>(nids));
  for (int64_t i = 0; i < nids; ++i) {
    if (!cagra_resolve_id(*ix, ids[i], &slots[static_cast<size_t>(i)])) {
      return OVVS_STATUS_INVALID_ARGUMENT;
    }
  }
  try {
    if (ix->deleted.empty()) {
      ix->deleted.assign(static_cast<size_t>((ix->ds.n + 7) / 8), 0u);
    }
    for (int64_t i = 0; i < nids; ++i) {
      const int64_t slot = slots[static_cast<size_t>(i)];
      const size_t byte = static_cast<size_t>(slot) >> 3;
      const uint8_t mask = static_cast<uint8_t>(1u << (slot & 7));
      if ((ix->deleted[byte] & mask) != 0) continue;  /* idempotent */
      ix->deleted[byte] |= mask;
      ++ix->deleted_count;
      ix->free_slots.push_back(static_cast<int32_t>(slot));
    }
    rd(res)->last_device = OVVS_DEVICE_CPU;
    return OVVS_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return OVVS_STATUS_OOM;
  } catch (...) {
    return OVVS_STATUS_ERROR;
  }
}

ovvsStatus ovvsCagraCounts(ovvsCagraIndex_t index, int64_t* live, int64_t* deleted) {
  if (!index) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<CagraIndex*>(index);
  if (live) *live = ix->ds.n - ix->deleted_count;
  if (deleted) *deleted = ix->deleted_count;
  return OVVS_STATUS_SUCCESS;
}

/* Rebuilds one row's out-edges from a fresh search and re-prunes the in-neighbours it names, which
   is the same repair `cagra_insert_one` performs for a newly appended row. Used by update, where
   the vector at `row` has changed and its old neighbourhood no longer describes it. */
/* Search effort for mutation repair. Swept at degree 32 (relink_sweep.py /
   relink_quality.py): itopk = degree, width = 1 mutates 1.53x faster than the old
   2*degree/2, and the graph it leaves behind is not worse -- paired same-window
   queries on identically-churned indexes read 1.06x/+0.0000 at 100K (10% cumulative
   churn) and 0.98x/-0.0006 at 1M, both inside noise. graph_search clamps itopk to at
   least k, so this is the minimum beam that can still return `degree` out-edges.
   OVVS_CAGRA_RELINK_ITOPK / OVVS_CAGRA_RELINK_WIDTH override for sweeps. */
static void cagra_relink_effort(int32_t degree, int32_t* itopk, int32_t* width) {
  *itopk = degree;
  *width = 1;
  const char* env = std::getenv("OVVS_CAGRA_RELINK_ITOPK");
  if (env && *env) {
    const long parsed = std::strtol(env, nullptr, 10);
    if (parsed > 0) *itopk = static_cast<int32_t>(parsed);
  }
  env = std::getenv("OVVS_CAGRA_RELINK_WIDTH");
  if (env && *env) {
    const long parsed = std::strtol(env, nullptr, 10);
    if (parsed > 0) *width = static_cast<int32_t>(parsed);
  }
}

static ovvsStatus cagra_relink_one(CagraIndex* ix, ResourcesData& r, int64_t row) {
  /* The vector at `row` was just rewritten in place; the int8 mirror fingerprint only
     samples 64 strided values and can miss it. Force a rebuild on the next walk. */
  ovvs_gpu_mirror_invalidate();
  const int64_t dim = ix->ds.dim;
  const int32_t degree = ix->degree;
  int32_t relink_itopk = 0, relink_width = 0;
  cagra_relink_effort(degree, &relink_itopk, &relink_width);
  std::vector<int64_t> nb(static_cast<size_t>(degree));
  std::vector<float> nd(static_cast<size_t>(degree));
  const ovvsStatus status =
      graph_search(r, ix->ds.x.data(), ix->ds.n, dim, ix->ds.metric, ix->graph.data(), degree,
                   ix->ds.x.data() + row * dim, 1, degree, relink_itopk, relink_width, nullptr,
                   nb.data(), nd.data());
  if (status != OVVS_STATUS_SUCCESS) return status;
  for (int32_t t = 0; t < degree; ++t) {
    int32_t found = t < static_cast<int32_t>(nb.size()) ? static_cast<int32_t>(nb[static_cast<size_t>(t)]) : -1;
    if (found == static_cast<int32_t>(row)) found = -1;  /* never self-loop */
    ix->graph[static_cast<size_t>(row * degree + t)] = found;
  }
  for (int32_t t = 0; t < degree; ++t) {
    const int32_t u = ix->graph[static_cast<size_t>(row * degree + t)];
    if (u < 0) continue;
    std::vector<int32_t> cand;
    for (int32_t e = 0; e < degree; ++e) {
      const int32_t v = ix->graph[static_cast<size_t>(static_cast<int64_t>(u) * degree + e)];
      if (v >= 0) cand.push_back(v);
    }
    cand.push_back(static_cast<int32_t>(row));
    robust_prune(ix->ds.x.data(), dim, ix->ds.metric, u, cand, degree, 1.2f,
                 ix->graph.data() + static_cast<int64_t>(u) * degree);
  }
  return OVVS_STATUS_SUCCESS;
}

/* Mutation batches are processed in chunks so a search never runs against a graph more
   than ~1.5% of the corpus stale. OVVS_CAGRA_MUTATE_CHUNK overrides for sweeps. */
static int64_t cagra_mutate_chunk(int64_t n) {
  const char* env = std::getenv("OVVS_CAGRA_MUTATE_CHUNK");
  if (env && *env) {
    const long parsed = std::strtol(env, nullptr, 10);
    if (parsed > 0) return static_cast<int64_t>(parsed);
  }
  return std::clamp<int64_t>(n / 64, 256, 4096);
}

/* robust_prune over either storage form. fp32 goes straight through; fp16 materializes
   the target and candidate rows into an fp32 scratch, runs the identical prune logic on
   remapped local ids, and maps the kept ids back. The remap is stable because both the
   wrapper and robust_prune sort ids ascending before deduping. */
static void cagra_prune_rows(CagraIndex* ix, int64_t p, std::vector<int32_t>& cand, float alpha,
                             int32_t* out_row) {
  const int64_t dim = ix->ds.dim;
  if (!ix->ds.x.empty()) {
    robust_prune(ix->ds.x.data(), dim, ix->ds.metric, p, cand, ix->degree, alpha, out_row);
    return;
  }
  std::sort(cand.begin(), cand.end());
  cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
  std::vector<float> scratch((cand.size() + 1) * static_cast<size_t>(dim));
  cagra_row_f32(*ix, p, scratch.data());
  std::vector<int32_t> local(cand.size());
  for (size_t i = 0; i < cand.size(); ++i) {
    cagra_row_f32(*ix, cand[i], scratch.data() + (i + 1) * static_cast<size_t>(dim));
    local[i] = static_cast<int32_t>(i + 1);
  }
  robust_prune(scratch.data(), dim, ix->ds.metric, 0, local, ix->degree, alpha, out_row);
  for (int32_t tt = 0; tt < ix->degree; ++tt) {
    const int32_t v = out_row[tt];
    out_row[tt] = v > 0 ? cand[static_cast<size_t>(v - 1)] : -1;
  }
}

/* Batched form of the insert/update repair `cagra_relink_one` performs: `count` rows
   whose vectors are already in place get their out-edges rebuilt from ONE batched
   graph_search -- the threaded CPU walk, so mutation finally uses every core -- and the
   in-neighbours the new rows name are re-pruned once per distinct target row with every
   new candidate at hand, instead of once per (insert, target) pair.

   Output is deterministic for a given input at any thread count: back-link groups are
   disjoint rows, and all candidates are collected before any prune runs. Relative to the
   serial path the semantics differ in two measured-not-assumed ways (churn.py judges
   both): searches inside one chunk see the graph as it stood at chunk entry, and a
   target named by several inserts is pruned once with all of them together.

   Engine choice for the search: in fp32 mode it is pinned to the CPU -- the GPU
   backend caches a fingerprinted int8 mirror of the fp32 dataset, and every chunk's
   in-place row rewrites would force a full-mirror rebuild (128MB rescan at 1M) to stay
   correct. In fp16 mode there is NO cached GPU state at all: x16, mirror8 and the graph
   are shared USM read live by the kernel, and mirror8 is updated in place by the very
   row writes that precede this call -- so the adaptive router may hand chunks to the
   GPU where it wins (>=320K-row corpora; chunks are 256-4096 queries, singles still
   land on the CPU). A caller policy of FORCE_CPU is an absolute veto in both modes;
   OVVS_CAGRA_MUTATE_GPU=0 disables the GPU assist for sweeps. */
static ovvsStatus cagra_relink_batch(CagraIndex* ix, ResourcesData& r, const int64_t* slots,
                                     int64_t count, std::vector<CagraGraphRowBackup>* journal) {
  if (count <= 0) return OVVS_STATUS_SUCCESS;
  /* Callers rewrote these rows' vectors in place; the mirror fingerprint samples only
     64 strided values and can miss that. Invalidate so this chunk's own batched search
     (and any later query) sees fresh data. */
  ovvs_gpu_mirror_invalidate();
  const int64_t dim = ix->ds.dim;
  const int32_t degree = ix->degree;
  std::vector<float> qbuf(static_cast<size_t>(count) * static_cast<size_t>(dim));
  for (int64_t i = 0; i < count; ++i) {
    cagra_row_f32(*ix, slots[i], qbuf.data() + i * dim);
  }
  std::vector<int64_t> nb(static_cast<size_t>(count) * static_cast<size_t>(degree));
  std::vector<float> nd(static_cast<size_t>(count) * static_cast<size_t>(degree));
  const CagraStorage st = cagra_storage(*ix);
  int32_t relink_itopk = 0, relink_width = 0;
  cagra_relink_effort(degree, &relink_itopk, &relink_width);
  const ovvsPolicy caller_policy = r.policy;
  ovvsPolicy relink_policy = OVVS_POLICY_FORCE_CPU;
  if (cagra_is_f16(*ix) && caller_policy != OVVS_POLICY_FORCE_CPU) {
    const char* env = std::getenv("OVVS_CAGRA_MUTATE_GPU");
    if (!env || *env != '0') relink_policy = OVVS_POLICY_HETERO;
  }
  r.policy = relink_policy;
  const ovvsStatus status =
      graph_search(r, st.f32, ix->ds.n, dim, ix->ds.metric, ix->graph.data(), degree,
                   qbuf.data(), count, degree, relink_itopk, relink_width, nullptr, nb.data(),
                   nd.data(), st.f16, st.u8);
  r.policy = caller_policy;
  if (status != OVVS_STATUS_SUCCESS) return status;

  for (int64_t i = 0; i < count; ++i) {
    const int64_t row = slots[i];
    for (int32_t t = 0; t < degree; ++t) {
      int32_t found = static_cast<int32_t>(nb[static_cast<size_t>(i * degree + t)]);
      if (found == static_cast<int32_t>(row)) found = -1; /* never self-loop */
      ix->graph[static_cast<size_t>(row * degree + t)] = found;
    }
  }

  /* (target row, new in-neighbour) pairs, grouped by target row. */
  std::vector<std::pair<int32_t, int32_t>> links;
  links.reserve(static_cast<size_t>(count) * static_cast<size_t>(degree));
  for (int64_t i = 0; i < count; ++i) {
    for (int32_t t = 0; t < degree; ++t) {
      const int32_t u = ix->graph[static_cast<size_t>(slots[i] * degree + t)];
      if (u >= 0) links.emplace_back(u, static_cast<int32_t>(slots[i]));
    }
  }
  std::sort(links.begin(), links.end());
  std::vector<std::pair<size_t, size_t>> groups;
  for (size_t b = 0; b < links.size();) {
    size_t e = b + 1;
    while (e < links.size() && links[e].first == links[b].first) ++e;
    groups.emplace_back(b, e);
    b = e;
  }
  if (journal) {
    for (const auto& g : groups) {
      const int64_t u = links[g.first].first;
      const auto row_begin = ix->graph.begin() + u * degree;
      journal->push_back({u, std::vector<int32_t>(row_begin, row_begin + degree)});
    }
  }

  const auto prune_group = [&](size_t gi) {
    const int64_t u = links[groups[gi].first].first;
    std::vector<int32_t> cand;
    cand.reserve(static_cast<size_t>(degree) + (groups[gi].second - groups[gi].first));
    for (int32_t e = 0; e < degree; ++e) {
      const int32_t v = ix->graph[static_cast<size_t>(u * degree + e)];
      if (v >= 0) cand.push_back(v);
    }
    for (size_t l = groups[gi].first; l < groups[gi].second; ++l) cand.push_back(links[l].second);
    cagra_prune_rows(ix, u, cand, 1.2f, ix->graph.data() + u * degree);
  };

  int64_t nthreads = [&]() -> int64_t {
    const char* env = std::getenv("OVVS_CAGRA_MUTATE_THREADS");
    if (env && *env) {
      const long parsed = std::strtol(env, nullptr, 10);
      if (parsed > 0) return static_cast<int64_t>(parsed);
    }
    const unsigned hw = std::thread::hardware_concurrency();
    return hw > 0 ? static_cast<int64_t>(hw) : 1;
  }();
  /* One worker per ~64 groups: a single-op call has only `degree` tiny prunes, and
     spawning a pool for that costs more than the prunes (measured +1.3 ms per single). */
  nthreads = std::min<int64_t>(nthreads, static_cast<int64_t>(groups.size()) / 64);
  if (nthreads <= 1) {
    for (size_t gi = 0; gi < groups.size(); ++gi) prune_group(gi);
    return OVVS_STATUS_SUCCESS;
  }
  std::atomic<bool> threw{false};
  const std::function<void(int64_t)> body = [&](int64_t gi) {
    if (threw.load(std::memory_order_relaxed)) return;
    try {
      prune_group(static_cast<size_t>(gi));
    } catch (...) {
      threw.store(true);
    }
  };
  r.pool(static_cast<int>(nthreads)).run(0, static_cast<int64_t>(groups.size()), body);
  /* A throwing prune leaves some targets un-pruned; their rows still hold valid (older)
     edges, so the graph stays consistent -- edges are hints. Report it all the same. */
  if (threw.load()) return OVVS_STATUS_OOM;
  return OVVS_STATUS_SUCCESS;
}

static bool cagra_serial_mutate() {
  const char* env = std::getenv("OVVS_CAGRA_SERIAL_MUTATE");
  return env && *env && *env != '0';
}

ovvsStatus ovvsCagraUpdate(ovvsResources_t res, ovvsCagraIndex_t index, const int64_t* ids,
                           const float* vectors, int64_t nids) {
  if (!res || !index || !ids || !vectors || nids < 0) return OVVS_STATUS_INVALID_ARGUMENT;
  if (nids == 0) return OVVS_STATUS_SUCCESS;
  auto* ix = reinterpret_cast<CagraIndex*>(index);
  if (!cagra_dataset_present(*ix)) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* resources = rd(res);
  if (resources->policy == OVVS_POLICY_FORCE_NPU) {
    ++resources->npu_fallbacks;
    return OVVS_STATUS_DEVICE_UNAVAILABLE;
  }
  if (resources->policy == OVVS_POLICY_FORCE_GPU) return OVVS_STATUS_DEVICE_UNAVAILABLE;
  std::vector<int64_t> slots(static_cast<size_t>(nids));
  for (int64_t i = 0; i < nids; ++i) {
    if (!cagra_resolve_id(*ix, ids[i], &slots[static_cast<size_t>(i)])) {
      return OVVS_STATUS_INVALID_ARGUMENT;
    }
  }
  const int64_t dim = ix->ds.dim;
  try {
    if (cagra_serial_mutate() && !cagra_is_f16(*ix)) {
      for (int64_t i = 0; i < nids; ++i) {
        const int64_t row = slots[static_cast<size_t>(i)];
        /* Keep the old vector so a failed relink can be undone; the graph rows the relink rewrites
           are repaired by the relink itself on the next successful call, and leaving a stale edge
           is safe because edges are only hints. Losing the caller's vector would not be. */
        std::vector<float> previous(ix->ds.x.begin() + row * dim,
                                    ix->ds.x.begin() + (row + 1) * dim);
        std::copy(vectors + i * dim, vectors + (i + 1) * dim, ix->ds.x.begin() + row * dim);
        const ovvsStatus status = cagra_relink_one(ix, *resources, row);
        if (status != OVVS_STATUS_SUCCESS) {
          std::copy(previous.begin(), previous.end(), ix->ds.x.begin() + row * dim);
          return status;
        }
        /* Updating a tombstoned row revives it: the caller supplied a live vector for that id. */
        if (cagra_row_deleted(*ix, row)) {
          ix->deleted[static_cast<size_t>(row) >> 3] &=
              static_cast<uint8_t>(~(1u << (row & 7)));
          --ix->deleted_count;
        }
      }
    } else {
      /* Batched: write a chunk of vectors, repair every touched row from one threaded
         search. Failure restores this chunk's vectors (earlier chunks stand, exactly as
         the serial path leaves earlier updates standing); stale edges are safe. */
      const int64_t chunk = cagra_mutate_chunk(ix->ds.n);
      for (int64_t base = 0; base < nids; base += chunk) {
        const int64_t cnt = std::min<int64_t>(chunk, nids - base);
        std::vector<float> previous(static_cast<size_t>(cnt) * static_cast<size_t>(dim));
        for (int64_t i = 0; i < cnt; ++i) {
          const int64_t row = slots[static_cast<size_t>(base + i)];
          cagra_row_f32(*ix, row, previous.data() + i * dim);
          cagra_write_row(ix, row, vectors + (base + i) * dim);
        }
        const ovvsStatus status =
            cagra_relink_batch(ix, *resources, slots.data() + base, cnt, nullptr);
        if (status != OVVS_STATUS_SUCCESS) {
          for (int64_t i = 0; i < cnt; ++i) {
            const int64_t row = slots[static_cast<size_t>(base + i)];
            cagra_write_row(ix, row, previous.data() + i * dim);
          }
          return status;
        }
        for (int64_t i = 0; i < cnt; ++i) {
          const int64_t row = slots[static_cast<size_t>(base + i)];
          if (cagra_row_deleted(*ix, row)) {
            ix->deleted[static_cast<size_t>(row) >> 3] &=
                static_cast<uint8_t>(~(1u << (row & 7)));
            --ix->deleted_count;
          }
        }
      }
    }
    if (ix->pq_m > 0 && !ix->codes.empty()) {
      pq_encode_rows(ix->ds.x.data(), ix->ds.n, dim, ix->pq_m, ix->pq_ks, ix->dsub,
                     ix->codebooks.data(), ix->codes.data());
    }
    resources->last_device = OVVS_DEVICE_CPU;
    return OVVS_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return OVVS_STATUS_OOM;
  } catch (...) {
    return OVVS_STATUS_ERROR;
  }
}

/* Claims a tombstoned row for a new vector: bumps its generation so every id handed out for the
   previous occupant is now detectably stale, revives it, overwrites the vector, and repairs the
   neighbourhood. Returns false only if there is no reusable row. */
static bool cagra_claim_free_slot(CagraIndex* ix, int64_t* slot) {
  while (!ix->free_slots.empty()) {
    const int64_t candidate = ix->free_slots.back();
    ix->free_slots.pop_back();
    /* A row can be revived by ovvsCagraUpdate while it still sits on this list, so entries are
       validated on the way out rather than removed eagerly on revival. */
    if (candidate >= 0 && candidate < ix->ds.n && cagra_row_deleted(*ix, candidate)) {
      *slot = candidate;
      return true;
    }
  }
  return false;
}

ovvsStatus ovvsCagraExtendEx(ovvsResources_t res, ovvsCagraIndex_t index, const float* extra,
                             int64_t nextra, int64_t* out_ids) {
  if (!res || !index || !extra || nextra <= 0) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<CagraIndex*>(index);
  if (!cagra_dataset_present(*ix)) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* resources = rd(res);
  if (resources->policy == OVVS_POLICY_FORCE_NPU) {
    ++resources->npu_fallbacks;
    return OVVS_STATUS_DEVICE_UNAVAILABLE;
  }
  if (resources->policy == OVVS_POLICY_FORCE_GPU) return OVVS_STATUS_DEVICE_UNAVAILABLE;
  const int64_t dim = ix->ds.dim;
  if (nextra > std::numeric_limits<int32_t>::max() - ix->ds.n ||
      dim > std::numeric_limits<int64_t>::max() / nextra) {
    return OVVS_STATUS_SHAPE_MISMATCH;
  }
  try {
    /* This used to be `CagraIndex staged = *ix;` -- a deep copy of the dataset and the graph, so
       inserting ONE vector into a SIFT1M index peaked around 1.8 GB. The copy bought all-or-nothing
       semantics, which are worth keeping; the cost is not. Insert in place instead and journal the
       only thing that is destructively overwritten: the <= degree neighbour rows `robust_prune`
       rewrites per insert. That is degree*degree*4 bytes, about 1 KB at degree 16.
       Everything else is append-only and unwound by truncating. */
    const int64_t original_n = ix->ds.n;
    /* Grow both buffers at most once for the whole batch. Two things are being balanced here.
       Reserving the exact final size would make every single-vector Extend reallocate and copy
       the whole dataset, i.e. O(n) per insert -- and inserting one vector at a time is precisely
       the on-device-memory workload this library exists for. Letting the vectors grow themselves
       instead reallocates repeatedly *within* a batch. So: grow only when short, and when growing,
       take at least half again, which keeps incremental insert amortised O(1) while a batch pays
       a single reallocation. */
    const auto grow = [](auto& buffer, size_t needed) {
      if (buffer.capacity() >= needed) return;
      buffer.reserve(std::max(needed, buffer.capacity() + buffer.capacity() / 2));
    };
    if (!ix->ds.x16.empty()) {
      grow(ix->ds.x16, static_cast<size_t>(original_n + nextra) * static_cast<size_t>(dim));
      if (ix->mirror8_ok) {
        grow(ix->mirror8, static_cast<size_t>(original_n + nextra) * static_cast<size_t>(dim));
      }
    } else {
      grow(ix->ds.x, static_cast<size_t>(original_n + nextra) * static_cast<size_t>(dim));
    }
    grow(ix->graph, static_cast<size_t>(original_n + nextra) * static_cast<size_t>(ix->degree));
    std::vector<CagraGraphRowBackup> journal;
    bool failed = false;
    ovvsStatus failure = OVVS_STATUS_SUCCESS;

    if (cagra_serial_mutate() && !cagra_is_f16(*ix)) {
      for (int64_t i = 0; i < nextra && !failed; ++i) {
        /* Reuse is only offered when the caller can be told which rows it got. Without out_ids a
           reused row would be invisible to them, so plain Extend keeps appending. */
        int64_t reused = -1;
        if (out_ids != nullptr && cagra_claim_free_slot(ix, &reused)) {
          std::vector<float> previous(ix->ds.x.begin() + reused * dim,
                                      ix->ds.x.begin() + (reused + 1) * dim);
          std::copy(extra + i * dim, extra + (i + 1) * dim, ix->ds.x.begin() + reused * dim);
          if (ix->generation.empty()) ix->generation.assign(static_cast<size_t>(ix->ds.n), 0u);
          ++ix->generation[static_cast<size_t>(reused)];
          ix->deleted[static_cast<size_t>(reused) >> 3] &=
              static_cast<uint8_t>(~(1u << (reused & 7)));
          --ix->deleted_count;
          const ovvsStatus status = cagra_relink_one(ix, *resources, reused);
          if (status != OVVS_STATUS_SUCCESS) {
            std::copy(previous.begin(), previous.end(), ix->ds.x.begin() + reused * dim);
            ix->deleted[static_cast<size_t>(reused) >> 3] |=
                static_cast<uint8_t>(1u << (reused & 7));
            ++ix->deleted_count;
            ix->free_slots.push_back(static_cast<int32_t>(reused));
            failed = true;
            failure = status;
            break;
          }
          out_ids[i] = cagra_pack_id(*ix, reused);
          continue;
        }
        const int64_t appended = ix->ds.n;
        const ovvsStatus status = cagra_insert_one(ix, *resources, extra + i * dim, &journal);
        if (status != OVVS_STATUS_SUCCESS) {
          failed = true;
          failure = status;
          break;
        }
        if (out_ids != nullptr) out_ids[i] = appended;
      }
    } else {
      /* Batched: claim slots and write vectors for a whole chunk, then repair every new
         row from one threaded search. Slot claiming stays serial (it is bookkeeping);
         the searches and the back-link prunes are where the time was. */
      const int64_t chunk = cagra_mutate_chunk(original_n);
      for (int64_t base = 0; base < nextra && !failed; base += chunk) {
        const int64_t cnt = std::min<int64_t>(chunk, nextra - base);
        std::vector<int64_t> slots(static_cast<size_t>(cnt));
        std::vector<std::pair<int64_t, std::vector<float>>> reused_prev;
        for (int64_t i = 0; i < cnt; ++i) {
          const float* vec = extra + (base + i) * dim;
          int64_t reused = -1;
          if (out_ids != nullptr && cagra_claim_free_slot(ix, &reused)) {
            std::vector<float> previous(static_cast<size_t>(dim));
            cagra_row_f32(*ix, reused, previous.data());
            reused_prev.emplace_back(reused, std::move(previous));
            cagra_write_row(ix, reused, vec);
            if (ix->generation.empty()) ix->generation.assign(static_cast<size_t>(ix->ds.n), 0u);
            ++ix->generation[static_cast<size_t>(reused)];
            ix->deleted[static_cast<size_t>(reused) >> 3] &=
                static_cast<uint8_t>(~(1u << (reused & 7)));
            --ix->deleted_count;
            slots[static_cast<size_t>(i)] = reused;
            out_ids[base + i] = cagra_pack_id(*ix, reused);
          } else {
            const int64_t appended = ix->ds.n;
            cagra_append_row(ix, vec);
            slots[static_cast<size_t>(i)] = appended;
            if (out_ids != nullptr) out_ids[base + i] = appended;
          }
        }
        ix->graph.resize(static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->degree), -1);
        const ovvsStatus status = cagra_relink_batch(ix, *resources, slots.data(), cnt, &journal);
        if (status != OVVS_STATUS_SUCCESS) {
          /* Give back this chunk's reused slots; appended rows are truncated below with
             the rest of the call, and the journal restore covers rewritten rows. */
          for (const auto& rp : reused_prev) {
            cagra_write_row(ix, rp.first, rp.second.data());
            ix->deleted[static_cast<size_t>(rp.first) >> 3] |=
                static_cast<uint8_t>(1u << (rp.first & 7));
            ++ix->deleted_count;
            ix->free_slots.push_back(static_cast<int32_t>(rp.first));
          }
          failed = true;
          failure = status;
        }
      }
    }

    if (failed) {
      /* Unwound newest first, so a row rewritten by several inserts lands back on its oldest
         snapshot -- the state it had before this call started. */
      for (size_t j = journal.size(); j-- > 0;) {
        const CagraGraphRowBackup& backup = journal[j];
        if (backup.row < original_n) {
          std::copy(backup.edges.begin(), backup.edges.end(),
                    ix->graph.begin() + backup.row * ix->degree);
        }
      }
      if (!ix->ds.x16.empty()) {
        ix->ds.x16.resize(static_cast<size_t>(original_n) * static_cast<size_t>(dim));
        if (ix->mirror8_ok) {
          ix->mirror8.resize(static_cast<size_t>(original_n) * static_cast<size_t>(dim));
        }
      } else {
        ix->ds.x.resize(static_cast<size_t>(original_n) * static_cast<size_t>(dim));
      }
      ix->ds.n = original_n;
      ix->graph.resize(static_cast<size_t>(original_n) * static_cast<size_t>(ix->degree));
      return failure;
    }

    if (!ix->deleted.empty()) {
      ix->deleted.resize(static_cast<size_t>((ix->ds.n + 7) / 8), 0u);
    }
    if (!ix->generation.empty()) {
      ix->generation.resize(static_cast<size_t>(ix->ds.n), 0u);
    }
    if (ix->pq_m > 0) {
      ix->codes.resize(static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->pq_m));
      pq_encode_rows(ix->ds.x.data(), ix->ds.n, dim, ix->pq_m, ix->pq_ks, ix->dsub,
                     ix->codebooks.data(), ix->codes.data());
    }
    resources->last_device = OVVS_DEVICE_CPU;
    return OVVS_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return OVVS_STATUS_OOM;
  } catch (...) {
    return OVVS_STATUS_ERROR;
  }
}

ovvsStatus ovvsCagraExtend(ovvsResources_t res, ovvsCagraIndex_t index, const float* extra,
                           int64_t nextra) {
  return ovvsCagraExtendEx(res, index, extra, nextra, nullptr);
}

ovvsStatus ovvsCagraQuantize(ovvsResources_t res, ovvsCagraIndex_t index, int32_t pq_m, int32_t pq_nbits) {
  if (!res || !index || pq_m <= 0) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<CagraIndex*>(index);
  if (!ix->has_dataset || ix->ds.x.empty()) return OVVS_STATUS_INVALID_ARGUMENT;
  if (ix->ds.dim % pq_m != 0) return OVVS_STATUS_SHAPE_MISMATCH;
  auto* resources = rd(res);
  if (resources->policy == OVVS_POLICY_FORCE_NPU) {
    ++resources->npu_fallbacks;
    return OVVS_STATUS_DEVICE_UNAVAILABLE;
  }
  if (resources->policy == OVVS_POLICY_FORCE_GPU) return OVVS_STATUS_DEVICE_UNAVAILABLE;
  const int32_t pq_ks = 1 << std::min(std::max(pq_nbits, 4), 8);
  if (static_cast<uint64_t>(ix->ds.dim) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
              static_cast<uint64_t>(pq_ks) ||
      static_cast<uint64_t>(ix->ds.n) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
              static_cast<uint64_t>(pq_m)) {
    return OVVS_STATUS_SHAPE_MISMATCH;
  }
  try {
    CagraIndex staged = *ix;
    staged.pq_m = pq_m;
    staged.pq_ks = pq_ks;
    staged.dsub = static_cast<int32_t>(staged.ds.dim / pq_m);
    staged.codebooks.assign(static_cast<size_t>(staged.pq_m) * staged.pq_ks *
                                static_cast<size_t>(staged.dsub),
                            0.f);
    std::vector<float> sub(static_cast<size_t>(staged.ds.n) *
                           static_cast<size_t>(staged.dsub));
    for (int32_t m = 0; m < staged.pq_m; ++m) {
      for (int64_t i = 0; i < staged.ds.n; ++i) {
        std::memcpy(sub.data() + i * staged.dsub,
                    staged.ds.x.data() + i * staged.ds.dim +
                        static_cast<int64_t>(m) * staged.dsub,
                    static_cast<size_t>(staged.dsub) * sizeof(float));
      }
      std::vector<float> cents;
      const ovvsStatus status =
          kmeans_fit_impl(*resources, sub.data(), staged.ds.n, staged.dsub,
                          staged.pq_ks, 8, cents);
      if (status != OVVS_STATUS_SUCCESS) return status;
      std::memcpy(staged.codebooks.data() +
                      static_cast<size_t>(m) * staged.pq_ks * staged.dsub,
                  cents.data(), cents.size() * sizeof(float));
    }
    staged.codes.resize(static_cast<size_t>(staged.ds.n) *
                        static_cast<size_t>(staged.pq_m));
    pq_encode_rows(staged.ds.x.data(), staged.ds.n, staged.ds.dim, staged.pq_m,
                   staged.pq_ks, staged.dsub, staged.codebooks.data(),
                   staged.codes.data());
    *ix = std::move(staged);
    resources->last_device = OVVS_DEVICE_CPU;
    return OVVS_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return OVVS_STATUS_OOM;
  } catch (...) {
    return OVVS_STATUS_ERROR;
  }
}

ovvsStatus ovvsCagraDetachDataset(ovvsCagraIndex_t index) {
  if (!index) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<CagraIndex*>(index);
  ix->ds.x.clear();
  ix->ds.x.shrink_to_fit();
  ix->has_dataset = false;
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsCagraAttachDataset(ovvsCagraIndex_t index, const float* dataset, int64_t n, int64_t dim) {
  if (!index || !dataset || n <= 0 || dim <= 0) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<CagraIndex*>(index);
  if (n != ix->ds.n || dim != ix->ds.dim) return OVVS_STATUS_SHAPE_MISMATCH;
  ix->ds.x.assign(dataset, dataset + n * dim);
  ix->has_dataset = true;
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsCagraDestroy(ovvsCagraIndex_t index) {
  delete reinterpret_cast<CagraIndex*>(index);
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsHnswFromCagra(ovvsResources_t res, ovvsCagraIndex_t cagra, ovvsHnswIndex_t* index) {
  if (!index) return OVVS_STATUS_INVALID_ARGUMENT;
  *index = nullptr;
  if (!res || !cagra) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* cg = reinterpret_cast<CagraIndex*>(cagra);
  if (!cg->has_dataset || cg->ds.x.empty()) return OVVS_STATUS_INVALID_ARGUMENT;
  if (cg->ds.metric != OVVS_METRIC_L2_EXPANDED || cg->degree > 10000) {
    return OVVS_STATUS_UNSUPPORTED;
  }
  try {
    auto hx = std::make_unique<HnswIndex>();
    hx->ds = cg->ds;
    hx->degree = cg->degree;
    hx->hnsw_m = cg->degree;
    hx->graph.assign(cg->graph.begin(), cg->graph.end());
    hx->level.assign(static_cast<size_t>(cg->ds.n), 0);
    hx->max_level = 0;
    hx->enter = 0;
    *index = reinterpret_cast<ovvsHnswIndex_t>(hx.release());
    return OVVS_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return OVVS_STATUS_OOM;
  } catch (...) {
    return OVVS_STATUS_ERROR;
  }
}

ovvsStatus ovvsHnswSearch(ovvsResources_t res, ovvsHnswIndex_t index, const float* queries, int64_t nq,
                          int64_t k, int32_t ef, int64_t* neighbors, float* distances) {
  if (!res || !index || !queries || !neighbors || !distances || nq <= 0 || k <= 0 ||
      k > std::numeric_limits<int32_t>::max()) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  if (nq > std::numeric_limits<int64_t>::max() / k) return OVVS_STATUS_SHAPE_MISMATCH;
  try {
    auto* hx = reinterpret_cast<HnswIndex*>(index);
    return graph_search(*rd(res), hx->ds.x.data(), hx->ds.n, hx->ds.dim,
                        hx->ds.metric, hx->graph.data(), hx->degree, queries, nq, k,
                        std::max(ef, static_cast<int32_t>(k)), 1, nullptr, neighbors,
                        distances);
  } catch (const std::bad_alloc&) {
    return OVVS_STATUS_OOM;
  } catch (...) {
    return OVVS_STATUS_ERROR;
  }
}

namespace {
template <typename T>
void write_pod(std::ostream& f, const T& v) {
  f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T>
void read_pod(std::istream& f, T& v) {
  f.read(reinterpret_cast<char*>(&v), sizeof(T));
}
}  // namespace

ovvsStatus ovvsHnswSerialize(ovvsHnswIndex_t index, const char* path) {
  /* hnswlib Index::saveIndex layout (little-endian host). Documented in docs/devices.md. */
  if (!index || !path) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* hx = reinterpret_cast<HnswIndex*>(index);
  std::ofstream f(path, std::ios::binary);
  if (!f) return OVVS_STATUS_IO;
  const size_t M = static_cast<size_t>(std::max(hx->hnsw_m, 1));
  if (M > std::numeric_limits<size_t>::max() / 2) return OVVS_STATUS_SHAPE_MISMATCH;
  const size_t maxM0 = std::max(static_cast<size_t>(hx->degree), M * 2);
  const size_t maxM = M;
  const size_t n = static_cast<size_t>(hx->ds.n);
  const size_t dim = static_cast<size_t>(hx->ds.dim);
  const size_t size_links_level0 = sizeof(unsigned int) + maxM0 * sizeof(unsigned int);
  const size_t data_size = dim * sizeof(float);
  const size_t offsetLevel0 = 0;
  const size_t offsetData = size_links_level0;
  const size_t label_offset = size_links_level0 + data_size;
  const size_t size_data_per_element = label_offset + sizeof(size_t);
  /* CAGRA supplies only a base graph. Advertising synthetic upper levels while
     writing empty upper-link blocks makes hnswlib dereference null link lists. */
  const int maxlevel = 0;
  const unsigned int enter =
      static_cast<unsigned int>(hx->enter >= 0 && hx->enter < hx->ds.n ? hx->enter : 0);
  const double mult = 1.0 / std::log(static_cast<double>(std::max<size_t>(M, 2)));
  const size_t ef_construction = 200;
  write_pod(f, offsetLevel0);
  write_pod(f, n); /* max_elements */
  write_pod(f, n); /* cur_element_count */
  write_pod(f, size_data_per_element);
  write_pod(f, label_offset);
  write_pod(f, offsetData);
  write_pod(f, maxlevel);
  write_pod(f, enter);
  write_pod(f, maxM);
  write_pod(f, maxM0);
  write_pod(f, M);
  write_pod(f, mult);
  write_pod(f, ef_construction);
  std::vector<char> row(size_data_per_element, 0);
  for (size_t i = 0; i < n; ++i) {
    std::fill(row.begin(), row.end(), 0);
    unsigned int links = 0;
    for (int32_t t = 0; t < hx->degree; ++t) {
      const int32_t nb = hx->graph[i * static_cast<size_t>(hx->degree) + static_cast<size_t>(t)];
      if (nb < 0) continue;
      const unsigned int encoded = static_cast<unsigned int>(nb);
      std::memcpy(row.data() + sizeof(unsigned int) + links * sizeof(unsigned int),
                  &encoded, sizeof(encoded));
      ++links;
    }
    std::memcpy(row.data(), &links, sizeof(unsigned int));
    std::memcpy(row.data() + offsetData, hx->ds.x.data() + i * dim, data_size);
    const size_t label = i;
    std::memcpy(row.data() + label_offset, &label, sizeof(size_t));
    f.write(row.data(), static_cast<std::streamsize>(size_data_per_element));
  }
  for (size_t i = 0; i < n; ++i) {
    const unsigned int linkListSize = 0; /* single-layer CAGRA export */
    write_pod(f, linkListSize);
  }
  return f.good() ? OVVS_STATUS_SUCCESS : OVVS_STATUS_IO;
}

ovvsStatus ovvsHnswDeserialize(ovvsResources_t res, const char* path, ovvsHnswIndex_t* index) {
  if (!res || !path || !index) return OVVS_STATUS_INVALID_ARGUMENT;
  std::ifstream f(path, std::ios::binary);
  if (!f) return OVVS_STATUS_IO;
  size_t offsetLevel0 = 0, max_elements = 0, cur = 0, size_data = 0, label_off = 0, offsetData = 0;
  int maxlevel = 0;
  unsigned int enter = 0;
  size_t maxM = 0, maxM0 = 0, M = 0, efc = 0;
  double mult = 0;
  read_pod(f, offsetLevel0);
  read_pod(f, max_elements);
  read_pod(f, cur);
  read_pod(f, size_data);
  read_pod(f, label_off);
  read_pod(f, offsetData);
  read_pod(f, maxlevel);
  read_pod(f, enter);
  read_pod(f, maxM);
  read_pod(f, maxM0);
  read_pod(f, M);
  read_pod(f, mult);
  read_pod(f, efc);
  if (!f || cur == 0 || cur > max_elements || M == 0 || M > 10000 || maxM != M ||
      maxM0 < M || maxM0 > 20000 || size_data < offsetData) {
    return OVVS_STATUS_IO;
  }
  const size_t dim = (label_off > offsetData) ? (label_off - offsetData) / sizeof(float) : 0;
  if (dim == 0 || dim > 4096) return OVVS_STATUS_IO;
  auto* hx = new HnswIndex();
  hx->ds.n = static_cast<int64_t>(cur);
  hx->ds.dim = static_cast<int64_t>(dim);
  hx->ds.metric = OVVS_METRIC_L2_EXPANDED;
  hx->degree = static_cast<int32_t>(maxM0);
  hx->hnsw_m = static_cast<int32_t>(M);
  hx->max_level = 0;
  hx->enter = enter < cur ? enter : 0;
  hx->graph.assign(cur * maxM0, -1);
  hx->level.assign(cur, 0);
  hx->ds.x.resize(cur * dim);
  std::vector<char> row(size_data);
  for (size_t i = 0; i < cur; ++i) {
    f.read(row.data(), static_cast<std::streamsize>(size_data));
    if (!f) {
      delete hx;
      return OVVS_STATUS_IO;
    }
    unsigned int links = 0;
    std::memcpy(&links, row.data(), sizeof(unsigned int));
    if (links > maxM0) {
      delete hx;
      return OVVS_STATUS_IO;
    }
    for (unsigned int t = 0; t < links; ++t) {
      unsigned int encoded = 0;
      std::memcpy(&encoded, row.data() + sizeof(unsigned int) + t * sizeof(unsigned int),
                  sizeof(encoded));
      if (encoded >= cur) {
        delete hx;
        return OVVS_STATUS_IO;
      }
      hx->graph[i * maxM0 + t] = static_cast<int32_t>(encoded);
    }
    std::memcpy(hx->ds.x.data() + i * dim, row.data() + offsetData, dim * sizeof(float));
  }
  *index = reinterpret_cast<ovvsHnswIndex_t>(hx);
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsHnswDestroy(ovvsHnswIndex_t index) {
  delete reinterpret_cast<HnswIndex*>(index);
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsVamanaBuild(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                           ovvsMetric metric, int32_t graph_degree, float alpha,
                           ovvsVamanaIndex_t* index) {
  if (!index) return OVVS_STATUS_INVALID_ARGUMENT;
  *index = nullptr;
  if (!res || !dataset || n <= 1 || dim <= 0 || graph_degree <= 0) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  if (n > std::numeric_limits<int32_t>::max() || dim > std::numeric_limits<int64_t>::max() / n) {
    return OVVS_STATUS_SHAPE_MISMATCH;
  }
  if (metric < OVVS_METRIC_L2_EXPANDED || metric > OVVS_METRIC_LP_UNEXPANDED) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  auto* resources = rd(res);
  if (resources->policy == OVVS_POLICY_FORCE_NPU) {
    ++resources->npu_fallbacks;
    return OVVS_STATUS_DEVICE_UNAVAILABLE;
  }
  if (resources->policy == OVVS_POLICY_FORCE_GPU) {
    /* Robust prune remains a host stage until the Vamana GPU phase. */
    return OVVS_STATUS_DEVICE_UNAVAILABLE;
  }
  graph_degree = std::max(1, std::min(graph_degree, static_cast<int32_t>(n - 1)));
  if (alpha <= 0.f) alpha = 1.2f;
  try {
    auto ix = std::make_unique<VamanaIndex>();
    copy_ds(ix->ds, dataset, n, dim, metric);
    ix->degree = graph_degree;
    std::vector<int32_t> init;
    const int32_t init_degree = static_cast<int32_t>(
        std::min<int64_t>(n - 1, static_cast<int64_t>(graph_degree) * 2));
    const ovvsStatus status =
        nndescent_build(*rd(res), ix->ds.x.data(), n, dim, metric, init_degree, 5, init);
    if (status != OVVS_STATUS_SUCCESS) return status;
    ix->graph.assign(static_cast<size_t>(n) * static_cast<size_t>(graph_degree), -1);
    for (int64_t i = 0; i < n; ++i) {
      std::vector<int32_t> cand;
      for (int32_t t = 0; t < init_degree; ++t) {
        const int32_t nb = init[static_cast<size_t>(i * init_degree + t)];
        if (nb >= 0) cand.push_back(nb);
      }
      robust_prune(ix->ds.x.data(), dim, metric, i, cand, graph_degree, alpha,
                   ix->graph.data() + i * graph_degree);
    }
    resources->last_device = OVVS_DEVICE_CPU;
    *index = reinterpret_cast<ovvsVamanaIndex_t>(ix.release());
    return OVVS_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return OVVS_STATUS_OOM;
  } catch (...) {
    return OVVS_STATUS_ERROR;
  }
}

ovvsStatus ovvsVamanaSearch(ovvsResources_t res, ovvsVamanaIndex_t index, const float* queries,
                            int64_t nq, int64_t k, int32_t beam, const uint8_t* bitset,
                            int64_t* neighbors, float* distances) {
  if (!res || !index || !queries || !neighbors || !distances || nq <= 0 || k <= 0 ||
      k > std::numeric_limits<int32_t>::max()) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  if (nq > std::numeric_limits<int64_t>::max() / k) return OVVS_STATUS_SHAPE_MISMATCH;
  try {
    auto* ix = reinterpret_cast<VamanaIndex*>(index);
    const float* x = ix->x_view ? ix->x_view : ix->ds.x.data();
    const int32_t* g = ix->graph_view ? ix->graph_view : ix->graph.data();
    return prim_graph_walk(*rd(res), x, ix->ds.n, ix->ds.dim, ix->ds.metric, g,
                           ix->degree, queries, nq, k,
                           std::max(beam, static_cast<int32_t>(k)), 1, bitset,
                           neighbors, distances);
  } catch (const std::bad_alloc&) {
    return OVVS_STATUS_OOM;
  } catch (...) {
    return OVVS_STATUS_ERROR;
  }
}

constexpr uint32_t kVamanaMagic = 0x314D4156u; /* 'VAM1' */

ovvsStatus ovvsVamanaSerialize(ovvsVamanaIndex_t index, const char* path) {
  if (!index || !path) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<VamanaIndex*>(index);
  std::ofstream f(path, std::ios::binary);
  if (!f) return OVVS_STATUS_IO;
  f.write(reinterpret_cast<const char*>(&kVamanaMagic), 4);
  int32_t ver = 1;
  f.write(reinterpret_cast<const char*>(&ver), 4);
  f.write(reinterpret_cast<const char*>(&ix->ds.n), 8);
  f.write(reinterpret_cast<const char*>(&ix->ds.dim), 8);
  f.write(reinterpret_cast<const char*>(&ix->degree), 4);
  int32_t metric = static_cast<int32_t>(ix->ds.metric);
  f.write(reinterpret_cast<const char*>(&metric), 4);
  const int32_t* g = ix->graph_view ? ix->graph_view : ix->graph.data();
  const float* x = ix->x_view ? ix->x_view : ix->ds.x.data();
  f.write(reinterpret_cast<const char*>(g),
          static_cast<std::streamsize>(static_cast<size_t>(ix->ds.n * ix->degree) * sizeof(int32_t)));
  f.write(reinterpret_cast<const char*>(x),
          static_cast<std::streamsize>(static_cast<size_t>(ix->ds.n * ix->ds.dim) * sizeof(float)));
  return f.good() ? OVVS_STATUS_SUCCESS : OVVS_STATUS_IO;
}

ovvsStatus ovvsVamanaDeserialize(ovvsResources_t res, const char* path, ovvsVamanaIndex_t* index) {
  if (!res || !path || !index) return OVVS_STATUS_INVALID_ARGUMENT;
  std::ifstream f(path, std::ios::binary);
  if (!f) return OVVS_STATUS_IO;
  uint32_t magic = 0;
  f.read(reinterpret_cast<char*>(&magic), 4);
  if (magic != kVamanaMagic) return OVVS_STATUS_IO;
  int32_t ver = 0;
  f.read(reinterpret_cast<char*>(&ver), 4);
  auto* ix = new VamanaIndex();
  f.read(reinterpret_cast<char*>(&ix->ds.n), 8);
  f.read(reinterpret_cast<char*>(&ix->ds.dim), 8);
  f.read(reinterpret_cast<char*>(&ix->degree), 4);
  int32_t metric = 0;
  f.read(reinterpret_cast<char*>(&metric), 4);
  ix->ds.metric = static_cast<ovvsMetric>(metric);
  ix->graph.resize(static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->degree));
  ix->ds.x.resize(static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->ds.dim));
  f.read(reinterpret_cast<char*>(ix->graph.data()),
         static_cast<std::streamsize>(ix->graph.size() * sizeof(int32_t)));
  f.read(reinterpret_cast<char*>(ix->ds.x.data()),
         static_cast<std::streamsize>(ix->ds.x.size() * sizeof(float)));
  if (!f) {
    delete ix;
    return OVVS_STATUS_IO;
  }
  *index = reinterpret_cast<ovvsVamanaIndex_t>(ix);
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsVamanaMmap(ovvsResources_t res, const char* path, ovvsVamanaIndex_t* index) {
  if (!res || !path || !index) return OVVS_STATUS_INVALID_ARGUMENT;
#ifdef _WIN32
  HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                          FILE_ATTRIBUTE_NORMAL, nullptr);
  if (fh == INVALID_HANDLE_VALUE) return OVVS_STATUS_IO;
  LARGE_INTEGER sz;
  if (!GetFileSizeEx(fh, &sz) || sz.QuadPart < 32) {
    CloseHandle(fh);
    return OVVS_STATUS_IO;
  }
  HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!mh) {
    CloseHandle(fh);
    return OVVS_STATUS_IO;
  }
  void* view = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
  if (!view) {
    CloseHandle(mh);
    CloseHandle(fh);
    return OVVS_STATUS_IO;
  }
  const auto* p = static_cast<const uint8_t*>(view);
  uint32_t magic = 0;
  std::memcpy(&magic, p, 4);
  if (magic != kVamanaMagic) {
    UnmapViewOfFile(view);
    CloseHandle(mh);
    CloseHandle(fh);
    return OVVS_STATUS_IO;
  }
  auto* ix = new VamanaIndex();
  std::memcpy(&ix->ds.n, p + 8, 8);
  std::memcpy(&ix->ds.dim, p + 16, 8);
  std::memcpy(&ix->degree, p + 24, 4);
  int32_t metric = 0;
  std::memcpy(&metric, p + 28, 4);
  ix->ds.metric = static_cast<ovvsMetric>(metric);
  const size_t gsz = static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->degree) * sizeof(int32_t);
  ix->graph_view = reinterpret_cast<const int32_t*>(p + 32);
  ix->x_view = reinterpret_cast<const float*>(p + 32 + gsz);
  ix->mmapped = true;
  ix->file_handle = fh;
  ix->map_handle = mh;
  ix->view = view;
  *index = reinterpret_cast<ovvsVamanaIndex_t>(ix);
  return OVVS_STATUS_SUCCESS;
#else
  int fd = open(path, O_RDONLY);
  if (fd < 0) return OVVS_STATUS_IO;
  struct stat st {};
  if (fstat(fd, &st) != 0 || st.st_size < 32) {
    close(fd);
    return OVVS_STATUS_IO;
  }
  void* view = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
  if (view == MAP_FAILED) {
    close(fd);
    return OVVS_STATUS_IO;
  }
  const auto* p = static_cast<const uint8_t*>(view);
  uint32_t magic = 0;
  std::memcpy(&magic, p, 4);
  if (magic != kVamanaMagic) {
    munmap(view, static_cast<size_t>(st.st_size));
    close(fd);
    return OVVS_STATUS_IO;
  }
  auto* ix = new VamanaIndex();
  std::memcpy(&ix->ds.n, p + 8, 8);
  std::memcpy(&ix->ds.dim, p + 16, 8);
  std::memcpy(&ix->degree, p + 24, 4);
  int32_t metric = 0;
  std::memcpy(&metric, p + 28, 4);
  ix->ds.metric = static_cast<ovvsMetric>(metric);
  const size_t gsz = static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->degree) * sizeof(int32_t);
  ix->graph_view = reinterpret_cast<const int32_t*>(p + 32);
  ix->x_view = reinterpret_cast<const float*>(p + 32 + gsz);
  ix->mmapped = true;
  ix->fd = fd;
  ix->view = view;
  ix->map_size = static_cast<size_t>(st.st_size);
  *index = reinterpret_cast<ovvsVamanaIndex_t>(ix);
  return OVVS_STATUS_SUCCESS;
#endif
}

ovvsStatus ovvsVamanaDestroy(ovvsVamanaIndex_t index) {
  auto* ix = reinterpret_cast<VamanaIndex*>(index);
  if (!ix) return OVVS_STATUS_SUCCESS;
#ifdef _WIN32
  if (ix->view) UnmapViewOfFile(ix->view);
  if (ix->map_handle) CloseHandle(ix->map_handle);
  if (ix->file_handle != INVALID_HANDLE_VALUE) CloseHandle(ix->file_handle);
#else
  if (ix->view && ix->view != MAP_FAILED) munmap(ix->view, ix->map_size);
  if (ix->fd >= 0) close(ix->fd);
#endif
  delete ix;
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsScannBuild(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                          ovvsMetric metric, int32_t nlist, int32_t pq_m, ovvsScannIndex_t* index) {
  if (!res || !dataset || !index) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* ix = new ScannIndex();
  copy_ds(ix->ds, dataset, n, dim, metric);
  ix->aniso.resize(static_cast<size_t>(dim), 1.f);
  std::vector<float> mean(static_cast<size_t>(dim), 0.f);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t d = 0; d < dim; ++d) mean[static_cast<size_t>(d)] += dataset[i * dim + d];
  for (int64_t d = 0; d < dim; ++d) mean[static_cast<size_t>(d)] /= static_cast<float>(n);
  for (int64_t d = 0; d < dim; ++d) {
    float v = 0.f;
    for (int64_t i = 0; i < n; ++i) {
      const float t = dataset[i * dim + d] - mean[static_cast<size_t>(d)];
      v += t * t;
    }
    ix->aniso[static_cast<size_t>(d)] = std::sqrt(v / static_cast<float>(std::max<int64_t>(n, 1)) + 1e-6f);
  }
  /* Score-aware / anisotropic: train IVF-PQ in weighted space (ScaNN-style AVQ). */
  std::vector<float> scaled(static_cast<size_t>(n) * static_cast<size_t>(dim));
  for (int64_t i = 0; i < n; ++i)
    for (int64_t d = 0; d < dim; ++d)
      scaled[static_cast<size_t>(i * dim + d)] =
          dataset[i * dim + d] * ix->aniso[static_cast<size_t>(d)];
  const ovvsStatus st = ovvsIvfPqBuild(res, scaled.data(), n, dim, metric, nlist, pq_m, 8, &ix->pq);
  if (st != OVVS_STATUS_SUCCESS) {
    delete ix;
    return st;
  }
  *index = reinterpret_cast<ovvsScannIndex_t>(ix);
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsScannSearch(ovvsResources_t res, ovvsScannIndex_t index, const float* queries,
                           int64_t nq, int64_t k, int32_t nprobe, int32_t krefine,
                           int64_t* neighbors, float* distances) {
  if (!res || !index || !queries || !neighbors || !distances) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<ScannIndex*>(index);
  const int64_t dim = ix->ds.dim;
  if (krefine < static_cast<int32_t>(k)) krefine = static_cast<int32_t>(k);
  std::vector<float> q(static_cast<size_t>(nq) * static_cast<size_t>(dim));
  for (int64_t i = 0; i < nq; ++i)
    for (int64_t d = 0; d < dim; ++d)
      q[static_cast<size_t>(i * dim + d)] = queries[i * dim + d] * ix->aniso[static_cast<size_t>(d)];
  std::vector<int64_t> cand(static_cast<size_t>(nq) * static_cast<size_t>(krefine));
  std::vector<float> cd(static_cast<size_t>(nq) * static_cast<size_t>(krefine));
  const ovvsStatus st =
      ovvsIvfPqSearch(res, ix->pq, q.data(), nq, krefine, nprobe, krefine, nullptr, cand.data(), cd.data());
  if (st != OVVS_STATUS_SUCCESS) return st;
  /* Re-rank on the original (unscaled) vectors — ScaNN-style residual refine. */
  for (int64_t qi = 0; qi < nq; ++qi) {
    std::vector<int64_t> ids;
    ids.reserve(static_cast<size_t>(krefine));
    for (int32_t t = 0; t < krefine; ++t) {
      const int64_t id = cand[static_cast<size_t>(qi * krefine + t)];
      if (id >= 0 && id < ix->ds.n) ids.push_back(id);
    }
    if (ids.empty()) {
      for (int64_t t = 0; t < k; ++t) {
        neighbors[qi * k + t] = -1;
        distances[qi * k + t] = kInf;
      }
      continue;
    }
    std::vector<float> gathered(ids.size() * static_cast<size_t>(dim));
    prim_gather_rows(*rd(res), ix->ds.x.data(), ix->ds.n, dim, ids.data(), static_cast<int64_t>(ids.size()),
                     gathered.data());
    std::vector<float> sc(ids.size());
    prim_pairwise(*rd(res), ix->ds.metric, queries + qi * dim, 1, gathered.data(),
                  static_cast<int64_t>(ids.size()), dim, sc.data(), 2.f);
    const int64_t kk = std::min(k, static_cast<int64_t>(ids.size()));
    std::vector<int64_t> fi(static_cast<size_t>(kk));
    std::vector<float> fv(static_cast<size_t>(kk));
    prim_topk(*rd(res), sc.data(), 1, static_cast<int64_t>(ids.size()), kk, fi.data(), fv.data(),
              metric_largest(ix->ds.metric));
    for (int64_t t = 0; t < k; ++t) {
      if (t < kk) {
        neighbors[qi * k + t] = ids[static_cast<size_t>(fi[static_cast<size_t>(t)])];
        float d = fv[static_cast<size_t>(t)];
        if (ix->ds.metric == OVVS_METRIC_INNER_PRODUCT) d = -d;
        distances[qi * k + t] = d;
      } else {
        neighbors[qi * k + t] = -1;
        distances[qi * k + t] = kInf;
      }
    }
  }
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsScannDestroy(ovvsScannIndex_t index) {
  auto* ix = reinterpret_cast<ScannIndex*>(index);
  if (ix) {
    ovvsIvfPqDestroy(ix->pq);
    delete ix;
  }
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsAllNeighbors(ovvsResources_t res, const float* dataset, int64_t n, int64_t dim,
                            ovvsMetric metric, int32_t k, int64_t* neighbors, float* distances) {
  if (!res || !dataset || !neighbors || !distances || n <= 1 || k <= 0) return OVVS_STATUS_INVALID_ARGUMENT;
  k = std::min(k, static_cast<int32_t>(n - 1));
  std::vector<float> scores(static_cast<size_t>(n) * static_cast<size_t>(n));
  prim_pairwise(*rd(res), metric, dataset, n, dataset, n, dim, scores.data(), 2.f);
  for (int64_t i = 0; i < n; ++i) scores[static_cast<size_t>(i * n + i)] = kInf;
  prim_topk(*rd(res), scores.data(), n, n, k, neighbors, distances, metric_largest(metric));
  if (metric == OVVS_METRIC_INNER_PRODUCT) {
    for (int64_t i = 0; i < n * k; ++i) distances[i] = -distances[i];
  }
  return OVVS_STATUS_SUCCESS;
}
