#pragma once

#include "ovvs/ovvs.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ovvs {
namespace impl {

constexpr float kInf = std::numeric_limits<float>::infinity();

/* Pairwise scores are always distances (smaller is better): L2, -IP, 1-cos, Hamming.
   Search must take the minimum. INNER_PRODUCT is converted back to +IP after topk. */
inline bool metric_largest(ovvsMetric) { return false; }

inline float l2sq(const float* a, const float* b, int64_t d) {
  float s = 0.f;
  for (int64_t i = 0; i < d; ++i) {
    const float t = a[i] - b[i];
    s += t * t;
  }
  return s;
}

inline float dot(const float* a, const float* b, int64_t d) {
  float s = 0.f;
  for (int64_t i = 0; i < d; ++i) s += a[i] * b[i];
  return s;
}

inline float nrm2sq(const float* a, int64_t d) {
  float s = 0.f;
  for (int64_t i = 0; i < d; ++i) s += a[i] * a[i];
  return s;
}

inline int hamming_u8(const uint8_t* a, const uint8_t* b, int64_t nbytes) {
  int s = 0;
  for (int64_t i = 0; i < nbytes; ++i) {
    unsigned v = static_cast<unsigned>(a[i] ^ b[i]);
#if defined(_MSC_VER)
    s += static_cast<int>(__popcnt(v));
#else
    s += __builtin_popcount(v);
#endif
  }
  return s;
}

inline float distance_one(ovvsMetric metric, const float* x, const float* y, int64_t d,
                          float metric_arg) {
  switch (metric) {
    case OVVS_METRIC_L2_EXPANDED:
      return l2sq(x, y, d);
    case OVVS_METRIC_L2_SQRT_EXPANDED:
      return std::sqrt(l2sq(x, y, d));
    case OVVS_METRIC_INNER_PRODUCT:
      return -dot(x, y, d);
    case OVVS_METRIC_COSINE_EXPANDED: {
      const float nx = std::sqrt(std::max(nrm2sq(x, d), 1e-12f));
      const float ny = std::sqrt(std::max(nrm2sq(y, d), 1e-12f));
      return 1.f - dot(x, y, d) / (nx * ny);
    }
    case OVVS_METRIC_BITWISE_HAMMING: {
      int s = 0;
      for (int64_t i = 0; i < d; ++i) {
        const uint32_t ax = static_cast<uint32_t>(x[i] >= 0.f);
        const uint32_t ay = static_cast<uint32_t>(y[i] >= 0.f);
        s += static_cast<int>(ax ^ ay);
      }
      return static_cast<float>(s);
    }
    case OVVS_METRIC_LP_UNEXPANDED: {
      const float p = metric_arg > 0.f ? metric_arg : 2.f;
      float s = 0.f;
      for (int64_t i = 0; i < d; ++i) s += std::pow(std::fabs(x[i] - y[i]), p);
      return std::pow(s, 1.f / p);
    }
    default:
      return l2sq(x, y, d);
  }
}

inline bool allowed(const uint8_t* bitset, int64_t i) {
  if (!bitset) return true;
  return (bitset[i >> 3] >> (i & 7)) & 1u;
}

struct MixerDecision {
  ovvsDevice device = OVVS_DEVICE_CPU;
  bool npu_attempted = false;
  bool npu_failed = false;
};

struct PqAdcTask {
  const float* tables = nullptr;
  const uint8_t* codes = nullptr;
  int64_t rows = 0;
  int64_t output_offset = 0;
};

struct PqAdcChunk {
  const float* tables = nullptr;
  const uint8_t* codes = nullptr;
  int32_t valid_rows = 0;
  int32_t bucket_rows = 0;
  int64_t output_offset = 0;
};

/* Offset-only descriptor for the fused IVF-PQ GPU scan/select path. Offsets are
   expressed in elements/rows of the flattened inputs, never as transient host
   pointers. query_dense_offset preserves query -> probe -> list-row tie order. */
struct IvfPqScanTask {
  int32_t query_index = 0;
  int64_t list_begin = 0;
  int64_t list_count = 0;
  int64_t lut_offset = 0;
  int64_t query_dense_offset = 0;
};

/* Call-local accounting for GPU work explicitly issued by ovVS. OpenVINO owns
   its internal transfers and submissions, so those deliberately remain opaque. */
struct GpuWorkStats {
  int64_t allocation_calls = 0;
  int64_t allocation_bytes = 0;
  int64_t h2d_calls = 0;
  int64_t h2d_bytes = 0;
  int64_t d2h_calls = 0;
  int64_t d2h_bytes = 0;
  int64_t kernel_launches = 0;
  int64_t wait_calls = 0;

  static void add(int64_t& value, uint64_t delta) noexcept {
    const uint64_t room = static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - value);
    value = delta > room ? std::numeric_limits<int64_t>::max()
                         : value + static_cast<int64_t>(delta);
  }

  static uint64_t bytes(size_t count, size_t element_size) noexcept {
    if (element_size != 0 && count > std::numeric_limits<size_t>::max() / element_size) {
      return std::numeric_limits<uint64_t>::max();
    }
    return static_cast<uint64_t>(count * element_size);
  }

  void allocation(size_t count, size_t element_size) noexcept {
    add(allocation_calls, 1);
    add(allocation_bytes, bytes(count, element_size));
  }
  void h2d(size_t count, size_t element_size = 1) noexcept {
    add(h2d_calls, 1);
    add(h2d_bytes, bytes(count, element_size));
  }
  void d2h(size_t count, size_t element_size = 1) noexcept {
    add(d2h_calls, 1);
    add(d2h_bytes, bytes(count, element_size));
  }
  void kernel() noexcept { add(kernel_launches, 1); }
  void wait() noexcept { add(wait_calls, 1); }
};

/* Per-call GPU NN-Descent accounting. This object is threaded through the
   CAGRA initializer so build telemetry never snapshots resource-global
   last-call diagnostics. */
struct NnDescentBuildStats {
  GpuWorkStats gpu;
  int64_t iterations = 0;
  int64_t final_changed_edges = 0;
  int64_t final_pending_new_edges = 0;
  int64_t submission_calls = 0;
  int64_t peak_owned_bytes = 0;
  bool converged = false;
  bool gpu_instrumented = false;

  void submission() noexcept { GpuWorkStats::add(submission_calls, 1); }
};

struct ResourcesData {
  ovvsPolicy policy = OVVS_POLICY_AUTO;
  bool npu_available = false;
  bool gpu_available = false;
  ovvsDevice last_device = OVVS_DEVICE_CPU;
  ovvsDType last_compute_dtype = OVVS_DTYPE_F32;
  std::string sku = "generic-cpu";
  std::string npu_name;
  std::string gpu_name;
  std::string cache_dir;
  int32_t npu_compile_fails = 0;
  int32_t npu_runtime_fails = 0;
  int32_t npu_fallbacks = 0;
  std::mutex cagra_transfer_mutex;
  int64_t cagra_walk_calls = 0;
  int64_t cagra_direct_index_calls = 0;
  int64_t cagra_index_upload_calls = 0;
  int64_t cagra_index_upload_bytes = 0;
  // Opt-in graph-walk work counters. Off by default so the production kernel is
  // byte-identical and pays nothing; the walk branches on a uniform null pointer.
  // Index order must match OVVS_CAGRA_WALK_COUNTER_* in ovvs.h.
  std::mutex cagra_walk_counter_mutex;
  bool cagra_walk_counters_enabled = false;
  int64_t cagra_walk_counters[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  std::mutex cagra_build_stats_mutex;
  ovvsCagraBuildStatsV1 cagra_build_stats{};
  std::mutex nndescent_stats_mutex;
  int32_t nndescent_iterations_run = 0;
  int64_t nndescent_changed_edges = 0;
  int64_t nndescent_pending_new_edges = 0;
  double nndescent_change_ratio = 0.0;
  int64_t nndescent_peak_device_bytes = 0;
  bool nndescent_converged = false;
  std::mutex ivfpq_stats_mutex;
  int64_t ivfpq_packed_rebuilds = 0;
  int64_t ivfpq_packed_rebuild_rows = 0;
  int64_t ivfpq_unfiltered_direct_rows = 0;
  int64_t ivfpq_unfiltered_id_copy_bytes_avoided = 0;
  int64_t ivfpq_selected_id_resolutions = 0;
  int64_t ivfpq_filtered_code_copy_bytes = 0;
  ovvsIvfPqSearchStatsV1 ivfpq_search_stats{};
  std::mutex pq_adc_stats_mutex;
  int64_t pq_adc_calls = 0;
  int64_t pq_adc_logical_tasks = 0;
  int64_t pq_adc_chunks = 0;
  int64_t pq_adc_valid_rows = 0;
  int64_t pq_adc_padded_rows = 0;
  int64_t pq_adc_npu_requests = 0;
  int64_t pq_adc_npu_rows = 0;
  int64_t pq_adc_npu_capacity_rows = 0;
  int64_t pq_adc_npu_transformed_chunks = 0;
  int64_t pq_adc_npu_transformed_rows = 0;
  int64_t pq_adc_cpu_rows = 0;
  bool npu_busy = false;
  ovvsDevice large_gemm_winner = OVVS_DEVICE_AUTO;
  int64_t large_gemm_flops = 100000LL * 32LL * 768LL;
  std::vector<float> scratch;

  float* scratch_f(size_t n) {
    if (scratch.size() < n) scratch.resize(n);
    return scratch.data();
  }

  /* Device-local scratch reused across graph-walk calls. Allocating and freeing the visited
     table and the query/output staging on every search cost four device allocations, four
     frees and four queue drains per call -- a fixed tax that dominates batch-one latency.
     Owned per Resources because the documented concurrency model is one Resources per worker. */
  void* gpu_walk_workspace = nullptr;
  size_t gpu_walk_workspace_bytes = 0;

  /* Device-side uint8 mirror of a graph-walk dataset. Only built when every element is an
     integer in [0,255] and the dimension is small enough that the exact integer distance
     still fits a float mantissa, in which case the int8 walk is BITWISE identical to the
     fp32 walk while touching a quarter of the cache lines per candidate. Keyed on the
     source pointer plus a sampled fingerprint so a freed-and-reallocated dataset landing
     on the same address cannot be mistaken for a cache hit. */
  std::mutex gpu_int8_mutex;
  const void* gpu_int8_source = nullptr;
  size_t gpu_int8_rows = 0;
  size_t gpu_int8_dim = 0;
  uint64_t gpu_int8_fingerprint = 0;
  void* gpu_int8_data = nullptr;
  bool gpu_int8_usable = false;

  ~ResourcesData();
};

inline ResourcesData* rd(ovvsResources_t r) { return reinterpret_cast<ResourcesData*>(r); }

void probe_fill(ResourcesData& r);
std::string probe_json();
void append_energy_probe_json(std::ostringstream& o);
void append_shave_elf_probe_json(std::ostringstream& o);
bool npu_shave_profile_adc(int* shave_tasks, int* dpu_tasks, std::vector<uint8_t>* blob,
                           std::string* exec_types);

/* Shared USM (SYCL) or heap. Dataset/graph vectors use UsmAllocator so iGPU binds these pointers. */
void* ovvs_usm_malloc(size_t bytes);
void ovvs_usm_free(void* p);
bool ovvs_usm_is_shared(const void* p);

/* Device-local (not shared) allocations for reusable GPU scratch. No-ops without SYCL. */
void* ovvs_gpu_workspace_alloc(size_t bytes);
void ovvs_gpu_workspace_free(void* p);

template <typename T>
struct UsmAllocator {
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using propagate_on_container_move_assignment = std::true_type;
  UsmAllocator() noexcept = default;
  template <typename U>
  UsmAllocator(const UsmAllocator<U>&) noexcept {}
  T* allocate(std::size_t n) {
    void* p = ovvs_usm_malloc(n == 0 ? sizeof(T) : n * sizeof(T));
    if (!p) throw std::bad_alloc();
    return static_cast<T*>(p);
  }
  void deallocate(T* p, std::size_t) noexcept { ovvs_usm_free(p); }
  template <typename U>
  bool operator==(const UsmAllocator<U>&) const noexcept {
    return true;
  }
  template <typename U>
  bool operator!=(const UsmAllocator<U>&) const noexcept {
    return false;
  }
};

using UsmFloatVec = std::vector<float, UsmAllocator<float>>;
using UsmI32Vec = std::vector<int32_t, UsmAllocator<int32_t>>;
using UsmI64Vec = std::vector<int64_t, UsmAllocator<int64_t>>;
using UsmU8Vec = std::vector<uint8_t, UsmAllocator<uint8_t>>;

/* Device backends. Boolean paths return false when unavailable/failed; status
   paths preserve validation, OOM, and runtime failures for policy routing. */
bool npu_available();
bool gpu_available();
bool ov_device_available(const char* name);
bool ov_matmul(ResourcesData& r, const char* device, const float* a, const float* b, float* c,
               int64_t m, int64_t n, int64_t k, bool trans_b);
bool ov_matmul_compute(ResourcesData& r, const char* device, ovvsDType compute, const float* a,
                       const float* b, float* c, int64_t m, int64_t n, int64_t k, bool trans_b);
void append_lowbit_probe_json(std::ostringstream& o);
bool ov_topk(ResourcesData& r, const char* device, const float* scores, int64_t rows, int64_t cols,
             int64_t k, int64_t* indices, float* values, bool largest);
bool ov_gather_rows(ResourcesData& r, const char* device, const float* src, int64_t src_rows,
                    int64_t dim, const int64_t* idx, int64_t nidx, float* out);
bool npu_gemm(ResourcesData& r, const float* a, const float* b, float* c, int64_t m, int64_t n,
              int64_t k, bool trans_b);
bool npu_gemm_compute(ResourcesData& r, ovvsDType compute, const float* a, const float* b, float* c,
                      int64_t m, int64_t n, int64_t k, bool trans_b);
bool npu_topk(ResourcesData& r, const float* scores, int64_t rows, int64_t cols, int64_t k,
              int64_t* indices, float* values, bool largest);
bool npu_gather_rows(ResourcesData& r, const float* src, int64_t src_rows, int64_t dim,
                     const int64_t* idx, int64_t nidx, float* out);
bool gpu_gemm(ResourcesData& r, const float* a, const float* b, float* c, int64_t m, int64_t n,
              int64_t k, bool trans_b, GpuWorkStats* stats = nullptr);
bool gpu_gemm_compute(ResourcesData& r, ovvsDType compute, const float* a, const float* b, float* c,
                      int64_t m, int64_t n, int64_t k, bool trans_b);
bool gpu_topk(ResourcesData& r, const float* scores, int64_t rows, int64_t cols, int64_t k,
              int64_t* indices, float* values, bool largest, GpuWorkStats* stats = nullptr);
bool gpu_gather_rows(ResourcesData& r, const float* src, int64_t src_rows, int64_t dim,
                     const int64_t* idx, int64_t nidx, float* out,
                     GpuWorkStats* stats = nullptr);
bool gpu_vector_add(const float* a, const float* b, float* c, int64_t n);
ovvsStatus gpu_nndescent_build(ResourcesData& r, const float* dataset, int64_t n,
                               int64_t dim, ovvsMetric metric, int32_t degree,
                               int32_t iters, int32_t* graph,
                               NnDescentBuildStats* stats = nullptr);
OVVS_API ovvsStatus gpu_cagra_optimize_ranked(ResourcesData& r,
                                               const int32_t* initial, int64_t n,
                                               int32_t initial_degree,
                                               int32_t final_degree,
                                               std::vector<int32_t>& output);
bool gpu_cagra_walk(ResourcesData& r, const float* dataset, int64_t n, int64_t dim, ovvsMetric metric,
                    const int32_t* graph, int32_t degree, const float* queries, int64_t nq, int64_t k,
                    int32_t itopk, int32_t search_width, const uint8_t* bitset, int64_t* neighbors,
                    float* distances);
bool gpu_pairwise(ResourcesData& r, ovvsMetric metric, const float* x, int64_t nx, const float* y,
                  int64_t ny, int64_t dim, float* out, float metric_arg,
                  GpuWorkStats* stats = nullptr);
bool npu_pairwise(ResourcesData& r, ovvsMetric metric, const float* x, int64_t nx, const float* y,
                  int64_t ny, int64_t dim, float* out, float metric_arg);
bool npu_pq_adc_batch(ResourcesData& r, const PqAdcChunk* chunks, int64_t chunk_count,
                      int32_t pq_m, int32_t ks, float* out);
ovvsStatus gpu_ivfpq_scan_select(ResourcesData& r, const IvfPqScanTask* tasks,
                                 int64_t task_count, const float* luts,
                                 int64_t lut_elements, const int64_t* packed_ids,
                                 const uint8_t* packed_codes, int64_t packed_rows,
                                 const uint8_t* allow_bitset, int64_t allow_bitset_bytes,
                                 int64_t nq, int32_t pq_m, int32_t ks, int32_t krefine,
                                 int32_t* packed_positions, int32_t* counts,
                                 GpuWorkStats* stats = nullptr);
int32_t sycl_enabled();
bool sycl_gpu_available();
bool mkl_gesvd_components(const float* centered, int64_t n, int64_t dim, int32_t ncomp, float* components);
bool mkl_syev_smallest(float* a, int64_t n, int32_t ncomp, float* embed);

void cpu_gemm(const float* a, const float* b, float* c, int64_t m, int64_t n, int64_t k,
              bool trans_b);
void cpu_topk(const float* scores, int64_t rows, int64_t cols, int64_t k, int64_t* indices,
              float* values, bool largest);
void cpu_gather_rows(const float* src, int64_t src_rows, int64_t dim, const int64_t* idx,
                     int64_t nidx, float* out);
void cpu_pairwise(ovvsMetric metric, const float* x, int64_t nx, const float* y, int64_t ny,
                  int64_t dim, float* out, float metric_arg);

ovvsDevice choose_device(ResourcesData& r, const char* op, int64_t flops_or_elems);

ovvsStatus prim_gemm(ResourcesData& r, const float* a, const float* b, float* c,
                     int64_t m, int64_t n, int64_t k, bool trans_b,
                     GpuWorkStats* stats = nullptr);
ovvsStatus prim_gemm_compute(ResourcesData& r, const float* a, const float* b, float* c, int64_t m,
                             int64_t n, int64_t k, bool trans_b, ovvsDType compute);
ovvsStatus prim_topk(ResourcesData& r, const float* scores, int64_t rows, int64_t cols, int64_t k,
                     int64_t* indices, float* values, bool largest,
                     GpuWorkStats* stats = nullptr);
ovvsStatus prim_gather_rows(ResourcesData& r, const float* src, int64_t src_rows, int64_t dim,
                            const int64_t* idx, int64_t nidx, float* out,
                            GpuWorkStats* stats = nullptr);
ovvsStatus prim_pairwise(ResourcesData& r, ovvsMetric metric, const float* x, int64_t nx,
                         const float* y, int64_t ny, int64_t dim, float* out,
                         float metric_arg, GpuWorkStats* stats = nullptr);
ovvsStatus prim_pq_adc(ResourcesData& r, const float* tables, int32_t pq_m, int32_t ks,
                       const uint8_t* codes, int64_t ncodes, float* out);
/* Private C++ test seams; not part of the installed C ABI. */
OVVS_API int32_t pq_adc_bucket_rows(int64_t remaining);
OVVS_API ovvsStatus plan_pq_adc_chunks(const PqAdcTask* tasks, int64_t task_count,
                                       int32_t pq_m, int64_t output_rows,
                                       std::vector<PqAdcChunk>& chunks);
OVVS_API ovvsStatus prim_pq_adc_batch(ResourcesData& r, const PqAdcTask* tasks,
                                      int64_t task_count, int32_t pq_m, int32_t ks,
                                      float* out, int64_t output_rows);
ovvsStatus prim_ivfpq_scan_select(ResourcesData& r, const IvfPqScanTask* tasks,
                                  int64_t task_count, const float* luts,
                                  int64_t lut_elements, const int64_t* packed_ids,
                                  const uint8_t* packed_codes, int64_t packed_rows,
                                  const uint8_t* allow_bitset, int64_t allow_bitset_bytes,
                                  int64_t nq, int32_t pq_m, int32_t ks, int32_t krefine,
                                  int32_t* packed_positions, int32_t* counts,
                                  GpuWorkStats* stats = nullptr);
ovvsStatus prim_nndescent_build(ResourcesData& r, const float* dataset, int64_t n, int64_t dim,
                                 ovvsMetric metric, int32_t degree, int32_t iterations,
                                 int32_t* graph, NnDescentBuildStats* stats = nullptr);
/* Private C++ test seam; not part of the installed C ABI. */
OVVS_API ovvsStatus prim_cagra_optimize_ranked(ResourcesData& r,
                                                const int32_t* initial, int64_t n,
                                                int32_t initial_degree,
                                                int32_t final_degree,
                                                std::vector<int32_t>& output);
OVVS_API ovvsStatus cagra_optimize_ranked(const int32_t* initial, int64_t n,
                                          int32_t initial_degree, int32_t final_degree,
                                          std::vector<int32_t>& output);
ovvsStatus prim_graph_walk(ResourcesData& r, const float* dataset, int64_t n, int64_t dim,
                           ovvsMetric metric, const int32_t* graph, int32_t degree, const float* queries,
                           int64_t nq, int64_t k, int32_t itopk, int32_t search_width,
                           const uint8_t* bitset, int64_t* neighbors, float* distances);

/* Temporary sweep instrument: OVVS_CAGRA_SEEDS pins the seed count so the recall-vs-QPS
   curve can be measured in one process instead of one rebuild per point. Read once.
   Remove this override once the seed budget is fixed. */
inline int64_t cagra_seed_override() {
  static const int64_t value = []() -> int64_t {
    const char* env = std::getenv("OVVS_CAGRA_SEEDS");
    if (!env || !*env) return -1;
    char* end = nullptr;
    const long long parsed = std::strtoll(env, &end, 10);
    if (end == env || *end != '\0' || parsed < 1) return -1;
    return static_cast<int64_t>(parsed);
  }();
  return value;
}

/* CAGRA is single-layer: random itopk seeds land too far in high-d for greedy routing
   on graph_degree=16. Score a larger sample, then keep itopk.
   Measured on SIFT100K: this phase is 43-50% of ALL candidate distance evaluations, so the
   count is a first-order performance knob and must stay identical on the CPU and GPU paths.
   Both call this function; do not re-derive it at a call site. */
inline int64_t cagra_seed_count(int64_t n, int32_t itopk, int32_t search_width) {
  const int64_t override_seeds = cagra_seed_override();
  int64_t seeds;
  if (override_seeds >= 1) {
    seeds = override_seeds;
  } else {
    /* Measured on SIFT100K, batch 1024, sweeping this count over 512..16 with everything
       else fixed: recall@10 is flat to four decimals (itopk=32/w1: 0.9718 -> 0.9721;
       itopk=64/w2: 0.9947 -> 0.9946) while throughput rises 36-46%. The old
       "score a larger sample" budget of 16*itopk (512-2048 seeds) was 43-50% of ALL
       candidate distance evaluations and bought no measurable recall. Keep enough to give
       the first iteration search_width distinct candidates to expand. */
    seeds = std::max<int64_t>(32, static_cast<int64_t>(std::max(1, search_width)));
  }
  if (n > 0 && seeds > n) seeds = n;
  if (seeds < 1) seeds = n > 0 ? 1 : 0;
  return seeds;
}

inline uint32_t cagra_query_hash(const float* query, int64_t dim) {
  uint32_t hash = 2166136261u;
  for (int64_t i = 0; i < dim; ++i) {
    hash ^= std::bit_cast<uint32_t>(query[i]);
    hash *= 16777619u;
  }
  hash ^= hash >> 16;
  hash *= 0x7feb352du;
  hash ^= hash >> 15;
  hash *= 0x846ca68bu;
  return hash ^ (hash >> 16);
}

inline float f16_to_f32(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u) << 16);
  const uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t man = h & 0x3ffu;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {
      int32_t e = 127 - 15 + 1;
      while ((man & 0x400u) == 0) {
        man <<= 1;
        --e;
      }
      man &= 0x3ffu;
      bits = sign | (static_cast<uint32_t>(e) << 23) | (man << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (man << 13);
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

inline uint16_t f32_to_f16(float f) {
  uint32_t bits = 0;
  std::memcpy(&bits, &f, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  const uint32_t man = bits & 0x7fffffu;
  if (exp <= 0) return static_cast<uint16_t>(sign);
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13));
}

template <typename FloatVec>
inline void convert_to_f32(ovvsDType dtype, const void* src, int64_t n, int64_t dim, FloatVec& out) {
  const int64_t cells = n * dim;
  out.resize(static_cast<size_t>(cells));
  if (dtype == OVVS_DTYPE_F32) {
    std::memcpy(out.data(), src, static_cast<size_t>(cells) * sizeof(float));
    return;
  }
  if (dtype == OVVS_DTYPE_F16) {
    const auto* h = static_cast<const uint16_t*>(src);
    for (int64_t i = 0; i < cells; ++i) out[static_cast<size_t>(i)] = f16_to_f32(h[i]);
    return;
  }
  if (dtype == OVVS_DTYPE_I8) {
    const auto* p = static_cast<const int8_t*>(src);
    for (int64_t i = 0; i < cells; ++i) out[static_cast<size_t>(i)] = static_cast<float>(p[i]);
    return;
  }
  const auto* p = static_cast<const uint8_t*>(src);
  for (int64_t i = 0; i < cells; ++i) out[static_cast<size_t>(i)] = static_cast<float>(p[i]);
}

ovvsStatus brute_search_impl(ResourcesData& r, const float* dataset, int64_t n, int64_t dim,
                             const float* queries, int64_t nq, ovvsMetric metric, int64_t k,
                             const uint8_t* bitset, int64_t* neighbors, float* distances);

ovvsStatus kmeans_fit_impl(ResourcesData& r, const float* x, int64_t n, int64_t dim, int32_t k,
                           int32_t iters, std::vector<float>& centroids);

int64_t brute_force_dim(ovvsBruteForceIndex_t index);

inline uint32_t fnv1a(const void* data, size_t n) {
  const auto* p = static_cast<const uint8_t*>(data);
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 16777619u;
  }
  return h;
}

inline std::mt19937 rng_from(uint32_t seed) { return std::mt19937(seed); }

}  // namespace impl
}  // namespace ovvs
