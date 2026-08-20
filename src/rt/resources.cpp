#include "internal.hpp"

#include <cstring>

using iovs::impl::probe_fill;
using iovs::impl::rd;
using iovs::impl::ResourcesData;

iovsStatus iovsResourcesCreate(iovsResources_t* res) {
  if (!res) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* r = new (std::nothrow) ResourcesData();
  if (!r) return IOVS_STATUS_OOM;
  probe_fill(*r);
  *res = reinterpret_cast<iovsResources_t>(r);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsResourcesDestroy(iovsResources_t res) {
  delete rd(res);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsResourcesSetPolicy(iovsResources_t res, iovsPolicy policy) {
  if (!res) return IOVS_STATUS_INVALID_ARGUMENT;
  rd(res)->policy = policy;
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsResourcesGetPolicy(iovsResources_t res, iovsPolicy* policy) {
  if (!res || !policy) return IOVS_STATUS_INVALID_ARGUMENT;
  *policy = rd(res)->policy;
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsResourcesNpuAvailable(iovsResources_t res, int32_t* available) {
  if (!res || !available) return IOVS_STATUS_INVALID_ARGUMENT;
  *available = rd(res)->npu_available ? 1 : 0;
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsResourcesGpuAvailable(iovsResources_t res, int32_t* available) {
  if (!res || !available) return IOVS_STATUS_INVALID_ARGUMENT;
  *available = rd(res)->gpu_available ? 1 : 0;
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsResourcesSku(iovsResources_t res, char* buf, int32_t len) {
  if (!res || !buf || len <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  std::snprintf(buf, static_cast<size_t>(len), "%s", rd(res)->sku.c_str());
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsResourcesNpuCompileFails(iovsResources_t res, int32_t* count) {
  if (!res || !count) return IOVS_STATUS_INVALID_ARGUMENT;
  *count = rd(res)->npu_compile_fails;
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsResourcesNpuFallbacks(iovsResources_t res, int32_t* count) {
  if (!res || !count) return IOVS_STATUS_INVALID_ARGUMENT;
  *count = rd(res)->npu_fallbacks;
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsResourcesLastDevice(iovsResources_t res, iovsDevice* device) {
  if (!res || !device) return IOVS_STATUS_INVALID_ARGUMENT;
  *device = rd(res)->last_device;
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsResourcesSetNpuBusy(iovsResources_t res, int32_t busy) {
  if (!res) return IOVS_STATUS_INVALID_ARGUMENT;
  rd(res)->npu_busy = busy != 0;
  return IOVS_STATUS_SUCCESS;
}

int32_t iovsSyclEnabled(void) { return iovs::impl::sycl_enabled(); }

iovsStatus iovsProbeJson(char* buf, int32_t len) {
  if (!buf || len <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  const std::string s = iovs::impl::probe_json();
  if (static_cast<int32_t>(s.size()) + 1 > len) return IOVS_STATUS_SHAPE_MISMATCH;
  std::memcpy(buf, s.c_str(), s.size() + 1);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsGemm(iovsResources_t res, const float* a, const float* b, float* c, int64_t m,
                    int64_t n, int64_t k, int32_t trans_b) {
  if (!res || !a || !b || !c || m <= 0 || n <= 0 || k <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  return iovs::impl::prim_gemm(*rd(res), a, b, c, m, n, k, trans_b != 0);
}

iovsStatus iovsTopk(iovsResources_t res, const float* scores, int64_t rows, int64_t cols, int64_t k,
                    int64_t* indices, float* values, int32_t largest) {
  if (!res || !scores || !indices || !values || rows <= 0 || cols <= 0 || k <= 0) {
    return IOVS_STATUS_INVALID_ARGUMENT;
  }
  return iovs::impl::prim_topk(*rd(res), scores, rows, cols, k, indices, values, largest != 0);
}

iovsStatus iovsGatherRows(iovsResources_t res, const float* src, int64_t src_rows, int64_t dim,
                          const int64_t* idx, int64_t nidx, float* out) {
  if (!res || !src || !idx || !out || src_rows <= 0 || dim <= 0 || nidx <= 0) {
    return IOVS_STATUS_INVALID_ARGUMENT;
  }
  return iovs::impl::prim_gather_rows(*rd(res), src, src_rows, dim, idx, nidx, out);
}

iovsStatus iovsPairwiseDistance(iovsResources_t res, iovsMetric metric, const float* x, int64_t nx,
                                const float* y, int64_t ny, int64_t dim, float* out,
                                float metric_arg) {
  if (!res || !x || !y || !out || nx <= 0 || ny <= 0 || dim <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  return iovs::impl::prim_pairwise(*rd(res), metric, x, nx, y, ny, dim, out, metric_arg);
}

iovsStatus iovsKSelection(iovsResources_t res, const float* scores, int64_t rows, int64_t cols,
                          int64_t k, int64_t* indices, float* values, int32_t largest) {
  return iovsTopk(res, scores, rows, cols, k, indices, values, largest);
}

#include "iovs_shave.h"

void iovsShaveTopkSmallest(const float* scores, int32_t cols, int32_t k, int32_t* idx, float* val) {
  iovs_shave_topk_smallest(scores, cols, k, idx, val);
}

float iovsShavePqAdc(const float* tables, const uint8_t* code, int32_t pq_m, int32_t ks) {
  return iovs_shave_pq_adc(tables, code, pq_m, ks);
}
