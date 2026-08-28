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
  int32_t npu_fallbacks = 0;
  std::mutex cagra_transfer_mutex;
  int64_t cagra_walk_calls = 0;
  int64_t cagra_direct_index_calls = 0;
  int64_t cagra_index_upload_calls = 0;
  int64_t cagra_index_upload_bytes = 0;
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
  int64_t ivfpq_filtered_code_copy_bytes = 0;
  bool npu_busy = false;
  ovvsDevice large_gemm_winner = OVVS_DEVICE_AUTO;
  int64_t large_gemm_flops = 100000LL * 32LL * 768LL;
  std::vector<float> scratch;

  float* scratch_f(size_t n) {
    if (scratch.size() < n) scratch.resize(n);
    return scratch.data();
  }
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

/* Device backends. Return false if unavailable / failed (caller falls back). */
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
              int64_t k, bool trans_b);
bool gpu_gemm_compute(ResourcesData& r, ovvsDType compute, const float* a, const float* b, float* c,
                      int64_t m, int64_t n, int64_t k, bool trans_b);
bool gpu_topk(ResourcesData& r, const float* scores, int64_t rows, int64_t cols, int64_t k,
              int64_t* indices, float* values, bool largest);
bool gpu_gather_rows(ResourcesData& r, const float* src, int64_t src_rows, int64_t dim,
                     const int64_t* idx, int64_t nidx, float* out);
bool gpu_vector_add(const float* a, const float* b, float* c, int64_t n);
ovvsStatus gpu_nndescent_build(ResourcesData& r, const float* dataset, int64_t n,
                               int64_t dim, ovvsMetric metric, int32_t degree,
                               int32_t iters, int32_t* graph);
bool gpu_cagra_walk(ResourcesData& r, const float* dataset, int64_t n, int64_t dim, ovvsMetric metric,
                    const int32_t* graph, int32_t degree, const float* queries, int64_t nq, int64_t k,
                    int32_t itopk, int32_t search_width, const uint8_t* bitset, int64_t* neighbors,
                    float* distances);
bool gpu_pairwise(ResourcesData& r, ovvsMetric metric, const float* x, int64_t nx, const float* y,
                  int64_t ny, int64_t dim, float* out, float metric_arg);
bool npu_pairwise(ResourcesData& r, ovvsMetric metric, const float* x, int64_t nx, const float* y,
                  int64_t ny, int64_t dim, float* out, float metric_arg);
bool npu_pq_adc(ResourcesData& r, const float* tables, int32_t pq_m, int32_t ks, const uint8_t* codes,
                int64_t ncodes, float* out);
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

ovvsStatus prim_gemm(ResourcesData& r, const float* a, const float* b, float* c, int64_t m, int64_t n,
                     int64_t k, bool trans_b);
ovvsStatus prim_gemm_compute(ResourcesData& r, const float* a, const float* b, float* c, int64_t m,
                             int64_t n, int64_t k, bool trans_b, ovvsDType compute);
ovvsStatus prim_topk(ResourcesData& r, const float* scores, int64_t rows, int64_t cols, int64_t k,
                     int64_t* indices, float* values, bool largest);
ovvsStatus prim_gather_rows(ResourcesData& r, const float* src, int64_t src_rows, int64_t dim,
                            const int64_t* idx, int64_t nidx, float* out);
ovvsStatus prim_pairwise(ResourcesData& r, ovvsMetric metric, const float* x, int64_t nx,
                         const float* y, int64_t ny, int64_t dim, float* out, float metric_arg);
ovvsStatus prim_pq_adc(ResourcesData& r, const float* tables, int32_t pq_m, int32_t ks,
                       const uint8_t* codes, int64_t ncodes, float* out);
ovvsStatus prim_nndescent_build(ResourcesData& r, const float* dataset, int64_t n, int64_t dim,
                                 ovvsMetric metric, int32_t degree, int32_t iterations,
                                 int32_t* graph);
/* Private C++ test seam; not part of the installed C ABI. */
OVVS_API ovvsStatus cagra_optimize_ranked(const int32_t* initial, int64_t n,
                                          int32_t initial_degree, int32_t final_degree,
                                          std::vector<int32_t>& output);
ovvsStatus prim_graph_walk(ResourcesData& r, const float* dataset, int64_t n, int64_t dim,
                           ovvsMetric metric, const int32_t* graph, int32_t degree, const float* queries,
                           int64_t nq, int64_t k, int32_t itopk, int32_t search_width,
                           const uint8_t* bitset, int64_t* neighbors, float* distances);

/* CAGRA is single-layer: random itopk seeds land too far in high-d for greedy routing
   on graph_degree=16. Score a larger sample, then keep itopk. */
inline int64_t cagra_seed_count(int64_t n, int32_t itopk, int32_t search_width) {
  int64_t seeds = static_cast<int64_t>(std::max(1, itopk)) * 16;
  const int64_t from_sw = static_cast<int64_t>(std::max(1, search_width)) * 32;
  if (from_sw > seeds) seeds = from_sw;
  if (seeds < 512) seeds = 512;
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
