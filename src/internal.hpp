#pragma once

#include "iovs/iovs.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace iovs {
namespace impl {

constexpr float kInf = std::numeric_limits<float>::infinity();

inline bool metric_largest(iovsMetric m) {
  return m == IOVS_METRIC_INNER_PRODUCT || m == IOVS_METRIC_COSINE_EXPANDED;
}

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

inline float distance_one(iovsMetric metric, const float* x, const float* y, int64_t d,
                          float metric_arg) {
  switch (metric) {
    case IOVS_METRIC_L2_EXPANDED:
      return l2sq(x, y, d);
    case IOVS_METRIC_L2_SQRT_EXPANDED:
      return std::sqrt(l2sq(x, y, d));
    case IOVS_METRIC_INNER_PRODUCT:
      return -dot(x, y, d);
    case IOVS_METRIC_COSINE_EXPANDED: {
      const float nx = std::sqrt(std::max(nrm2sq(x, d), 1e-12f));
      const float ny = std::sqrt(std::max(nrm2sq(y, d), 1e-12f));
      return 1.f - dot(x, y, d) / (nx * ny);
    }
    case IOVS_METRIC_BITWISE_HAMMING: {
      int s = 0;
      for (int64_t i = 0; i < d; ++i) {
        const uint32_t ax = static_cast<uint32_t>(x[i] >= 0.f);
        const uint32_t ay = static_cast<uint32_t>(y[i] >= 0.f);
        s += static_cast<int>(ax ^ ay);
      }
      return static_cast<float>(s);
    }
    case IOVS_METRIC_LP_UNEXPANDED: {
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
  iovsDevice device = IOVS_DEVICE_CPU;
  bool npu_attempted = false;
  bool npu_failed = false;
};

struct ResourcesData {
  iovsPolicy policy = IOVS_POLICY_AUTO;
  bool npu_available = false;
  bool gpu_available = false;
  iovsDevice last_device = IOVS_DEVICE_CPU;
  std::string sku = "generic-cpu";
  std::string npu_name;
  std::string gpu_name;
  std::string cache_dir;
  int32_t npu_compile_fails = 0;
  int32_t npu_fallbacks = 0;
  std::vector<float> scratch;

  float* scratch_f(size_t n) {
    if (scratch.size() < n) scratch.resize(n);
    return scratch.data();
  }
};

inline ResourcesData* rd(iovsResources_t r) { return reinterpret_cast<ResourcesData*>(r); }

void probe_fill(ResourcesData& r);
std::string probe_json();

/* Device backends. Return false if unavailable / failed (caller falls back). */
bool npu_available();
bool gpu_available();
bool ov_device_available(const char* name);
bool ov_matmul(ResourcesData& r, const char* device, const float* a, const float* b, float* c,
               int64_t m, int64_t n, int64_t k, bool trans_b);
bool ov_topk(ResourcesData& r, const char* device, const float* scores, int64_t rows, int64_t cols,
             int64_t k, int64_t* indices, float* values, bool largest);
bool ov_gather_rows(ResourcesData& r, const char* device, const float* src, int64_t src_rows,
                    int64_t dim, const int64_t* idx, int64_t nidx, float* out);
bool npu_gemm(ResourcesData& r, const float* a, const float* b, float* c, int64_t m, int64_t n,
              int64_t k, bool trans_b);
bool npu_topk(ResourcesData& r, const float* scores, int64_t rows, int64_t cols, int64_t k,
              int64_t* indices, float* values, bool largest);
bool npu_gather_rows(ResourcesData& r, const float* src, int64_t src_rows, int64_t dim,
                     const int64_t* idx, int64_t nidx, float* out);
bool gpu_gemm(ResourcesData& r, const float* a, const float* b, float* c, int64_t m, int64_t n,
              int64_t k, bool trans_b);
bool gpu_topk(ResourcesData& r, const float* scores, int64_t rows, int64_t cols, int64_t k,
              int64_t* indices, float* values, bool largest);
bool gpu_gather_rows(ResourcesData& r, const float* src, int64_t src_rows, int64_t dim,
                     const int64_t* idx, int64_t nidx, float* out);
bool gpu_vector_add(const float* a, const float* b, float* c, int64_t n);

void cpu_gemm(const float* a, const float* b, float* c, int64_t m, int64_t n, int64_t k,
              bool trans_b);
void cpu_topk(const float* scores, int64_t rows, int64_t cols, int64_t k, int64_t* indices,
              float* values, bool largest);
void cpu_gather_rows(const float* src, int64_t src_rows, int64_t dim, const int64_t* idx,
                     int64_t nidx, float* out);
void cpu_pairwise(iovsMetric metric, const float* x, int64_t nx, const float* y, int64_t ny,
                  int64_t dim, float* out, float metric_arg);

iovsDevice choose_device(ResourcesData& r, const char* op, int64_t flops_or_elems);

iovsStatus prim_gemm(ResourcesData& r, const float* a, const float* b, float* c, int64_t m, int64_t n,
                     int64_t k, bool trans_b);
iovsStatus prim_topk(ResourcesData& r, const float* scores, int64_t rows, int64_t cols, int64_t k,
                     int64_t* indices, float* values, bool largest);
iovsStatus prim_gather_rows(ResourcesData& r, const float* src, int64_t src_rows, int64_t dim,
                            const int64_t* idx, int64_t nidx, float* out);
iovsStatus prim_pairwise(ResourcesData& r, iovsMetric metric, const float* x, int64_t nx,
                         const float* y, int64_t ny, int64_t dim, float* out, float metric_arg);

void brute_search_impl(ResourcesData& r, const float* dataset, int64_t n, int64_t dim,
                       const float* queries, int64_t nq, iovsMetric metric, int64_t k,
                       const uint8_t* bitset, int64_t* neighbors, float* distances);

void kmeans_fit_impl(ResourcesData& r, const float* x, int64_t n, int64_t dim, int32_t k,
                     int32_t iters, std::vector<float>& centroids);

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
}  // namespace iovs
