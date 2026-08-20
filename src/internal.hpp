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

namespace iovs {
namespace impl {

constexpr float kInf = std::numeric_limits<float>::infinity();

/* Pairwise scores are always distances (smaller is better): L2, -IP, 1-cos, Hamming.
   Search must take the minimum. INNER_PRODUCT is converted back to +IP after topk. */
inline bool metric_largest(iovsMetric) { return false; }

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
  bool npu_busy = false;
  iovsDevice large_gemm_winner = IOVS_DEVICE_AUTO;
  int64_t large_gemm_flops = 100000LL * 32LL * 768LL;
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
bool gpu_cagra_walk(ResourcesData& r, const float* dataset, int64_t n, int64_t dim, iovsMetric metric,
                    const int32_t* graph, int32_t degree, const float* queries, int64_t nq, int64_t k,
                    int32_t itopk, int32_t search_width, const uint8_t* bitset, int64_t* neighbors,
                    float* distances);
int32_t sycl_enabled();

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
iovsStatus prim_graph_walk(ResourcesData& r, const float* dataset, int64_t n, int64_t dim,
                           iovsMetric metric, const int32_t* graph, int32_t degree, const float* queries,
                           int64_t nq, int64_t k, int32_t itopk, int32_t search_width,
                           const uint8_t* bitset, int64_t* neighbors, float* distances);

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

inline void convert_to_f32(iovsDType dtype, const void* src, int64_t n, int64_t dim,
                           std::vector<float>& out) {
  const int64_t cells = n * dim;
  out.resize(static_cast<size_t>(cells));
  if (dtype == IOVS_DTYPE_F32) {
    std::memcpy(out.data(), src, static_cast<size_t>(cells) * sizeof(float));
    return;
  }
  if (dtype == IOVS_DTYPE_F16) {
    const auto* h = static_cast<const uint16_t*>(src);
    for (int64_t i = 0; i < cells; ++i) out[static_cast<size_t>(i)] = f16_to_f32(h[i]);
    return;
  }
  if (dtype == IOVS_DTYPE_I8) {
    const auto* p = static_cast<const int8_t*>(src);
    for (int64_t i = 0; i < cells; ++i) out[static_cast<size_t>(i)] = static_cast<float>(p[i]);
    return;
  }
  const auto* p = static_cast<const uint8_t*>(src);
  for (int64_t i = 0; i < cells; ++i) out[static_cast<size_t>(i)] = static_cast<float>(p[i]);
}

void brute_search_impl(ResourcesData& r, const float* dataset, int64_t n, int64_t dim,
                       const float* queries, int64_t nq, iovsMetric metric, int64_t k,
                       const uint8_t* bitset, int64_t* neighbors, float* distances);

void kmeans_fit_impl(ResourcesData& r, const float* x, int64_t n, int64_t dim, int32_t k,
                     int32_t iters, std::vector<float>& centroids);

int64_t brute_force_dim(iovsBruteForceIndex_t index);

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
