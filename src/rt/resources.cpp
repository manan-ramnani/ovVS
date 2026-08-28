#include "internal.hpp"

#include <cstring>

using ovvs::impl::probe_fill;
using ovvs::impl::rd;
using ovvs::impl::ResourcesData;

static_assert(sizeof(ovvsIvfPqSearchStatsV1) == 200,
              "ovvsIvfPqSearchStatsV1 is a frozen 200-byte ABI");
static_assert(sizeof(ovvsCagraBuildStatsV1) == 256,
              "ovvsCagraBuildStatsV1 is a frozen 256-byte ABI");

ovvsStatus ovvsResourcesCreate(ovvsResources_t* res) {
  if (!res) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* r = new (std::nothrow) ResourcesData();
  if (!r) return OVVS_STATUS_OOM;
  probe_fill(*r);
  *res = reinterpret_cast<ovvsResources_t>(r);
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsResourcesDestroy(ovvsResources_t res) {
  delete rd(res);
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsResourcesSetPolicy(ovvsResources_t res, ovvsPolicy policy) {
  if (!res) return OVVS_STATUS_INVALID_ARGUMENT;
  rd(res)->policy = policy;
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsResourcesGetPolicy(ovvsResources_t res, ovvsPolicy* policy) {
  if (!res || !policy) return OVVS_STATUS_INVALID_ARGUMENT;
  *policy = rd(res)->policy;
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsResourcesNpuAvailable(ovvsResources_t res, int32_t* available) {
  if (!res || !available) return OVVS_STATUS_INVALID_ARGUMENT;
  *available = rd(res)->npu_available ? 1 : 0;
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsResourcesGpuAvailable(ovvsResources_t res, int32_t* available) {
  if (!res || !available) return OVVS_STATUS_INVALID_ARGUMENT;
  *available = rd(res)->gpu_available ? 1 : 0;
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsResourcesSku(ovvsResources_t res, char* buf, int32_t len) {
  if (!res || !buf || len <= 0) return OVVS_STATUS_INVALID_ARGUMENT;
  std::snprintf(buf, static_cast<size_t>(len), "%s", rd(res)->sku.c_str());
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsResourcesNpuCompileFails(ovvsResources_t res, int32_t* count) {
  if (!res || !count) return OVVS_STATUS_INVALID_ARGUMENT;
  *count = rd(res)->npu_compile_fails;
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsResourcesNpuFallbacks(ovvsResources_t res, int32_t* count) {
  if (!res || !count) return OVVS_STATUS_INVALID_ARGUMENT;
  *count = rd(res)->npu_fallbacks;
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsResourcesLastDevice(ovvsResources_t res, ovvsDevice* device) {
  if (!res || !device) return OVVS_STATUS_INVALID_ARGUMENT;
  *device = rd(res)->last_device;
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsResourcesLastComputeDtype(ovvsResources_t res, ovvsDType* dtype) {
  if (!res || !dtype) return OVVS_STATUS_INVALID_ARGUMENT;
  *dtype = rd(res)->last_compute_dtype;
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsResourcesCagraTransferStats(ovvsResources_t res, int64_t* walks,
                                           int64_t* direct_walks, int64_t* upload_calls,
                                           int64_t* upload_bytes) {
  if (!res || !walks || !direct_walks || !upload_calls || !upload_bytes) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  auto* resources = rd(res);
  std::lock_guard<std::mutex> lock(resources->cagra_transfer_mutex);
  *walks = resources->cagra_walk_calls;
  *direct_walks = resources->cagra_direct_index_calls;
  *upload_calls = resources->cagra_index_upload_calls;
  *upload_bytes = resources->cagra_index_upload_bytes;
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsResourcesIvfPqSearchStatsV1(ovvsResources_t res,
                                           ovvsIvfPqSearchStatsV1* stats) {
  if (!res || !stats) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* resources = rd(res);
  std::lock_guard<std::mutex> lock(resources->ivfpq_stats_mutex);
  *stats = resources->ivfpq_search_stats;
  stats->abi_version = OVVS_IVF_PQ_SEARCH_STATS_ABI_V1;
  stats->struct_size = static_cast<uint32_t>(sizeof(*stats));
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsResourcesCagraBuildStatsV1(ovvsResources_t res,
                                          ovvsCagraBuildStatsV1* stats) {
  if (!res || !stats) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* resources = rd(res);
  std::lock_guard<std::mutex> lock(resources->cagra_build_stats_mutex);
  *stats = resources->cagra_build_stats;
  stats->abi_version = OVVS_CAGRA_BUILD_STATS_ABI_V1;
  stats->struct_size = static_cast<uint32_t>(sizeof(*stats));
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsResourcesSetNpuBusy(ovvsResources_t res, int32_t busy) {
  if (!res) return OVVS_STATUS_INVALID_ARGUMENT;
  rd(res)->npu_busy = busy != 0;
  return OVVS_STATUS_SUCCESS;
}

int32_t ovvsSyclEnabled(void) { return ovvs::impl::sycl_enabled(); }

ovvsStatus ovvsProbeJson(char* buf, int32_t len) {
  if (!buf || len <= 0) return OVVS_STATUS_INVALID_ARGUMENT;
  const std::string s = ovvs::impl::probe_json();
  if (static_cast<int32_t>(s.size()) + 1 > len) return OVVS_STATUS_SHAPE_MISMATCH;
  std::memcpy(buf, s.c_str(), s.size() + 1);
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsGemm(ovvsResources_t res, const float* a, const float* b, float* c, int64_t m,
                    int64_t n, int64_t k, int32_t trans_b) {
  if (!res || !a || !b || !c || m <= 0 || n <= 0 || k <= 0) return OVVS_STATUS_INVALID_ARGUMENT;
  return ovvs::impl::prim_gemm(*rd(res), a, b, c, m, n, k, trans_b != 0);
}

ovvsStatus ovvsGemmEx(ovvsResources_t res, const float* a, const float* b, float* c, int64_t m,
                      int64_t n, int64_t k, int32_t trans_b, ovvsDType compute) {
  if (!res || !a || !b || !c || m <= 0 || n <= 0 || k <= 0) return OVVS_STATUS_INVALID_ARGUMENT;
  return ovvs::impl::prim_gemm_compute(*rd(res), a, b, c, m, n, k, trans_b != 0, compute);
}

ovvsStatus ovvsTopk(ovvsResources_t res, const float* scores, int64_t rows, int64_t cols, int64_t k,
                    int64_t* indices, float* values, int32_t largest) {
  if (!res || !scores || !indices || !values || rows <= 0 || cols <= 0 || k <= 0) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  return ovvs::impl::prim_topk(*rd(res), scores, rows, cols, k, indices, values, largest != 0);
}

ovvsStatus ovvsGatherRows(ovvsResources_t res, const float* src, int64_t src_rows, int64_t dim,
                          const int64_t* idx, int64_t nidx, float* out) {
  if (!res || !src || !idx || !out || src_rows <= 0 || dim <= 0 || nidx <= 0) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  return ovvs::impl::prim_gather_rows(*rd(res), src, src_rows, dim, idx, nidx, out);
}

ovvsStatus ovvsPairwiseDistance(ovvsResources_t res, ovvsMetric metric, const float* x, int64_t nx,
                                const float* y, int64_t ny, int64_t dim, float* out,
                                float metric_arg) {
  if (!res || !x || !y || !out || nx <= 0 || ny <= 0 || dim <= 0) return OVVS_STATUS_INVALID_ARGUMENT;
  return ovvs::impl::prim_pairwise(*rd(res), metric, x, nx, y, ny, dim, out, metric_arg);
}

ovvsStatus ovvsKSelection(ovvsResources_t res, const float* scores, int64_t rows, int64_t cols,
                          int64_t k, int64_t* indices, float* values, int32_t largest) {
  return ovvsTopk(res, scores, rows, cols, k, indices, values, largest);
}

ovvsStatus ovvsBitsetFromAllowList(int64_t n, const int64_t* ids, int64_t nids, uint8_t* bitset) {
  if (!bitset || n <= 0 || nids < 0) return OVVS_STATUS_INVALID_ARGUMENT;
  const int64_t nbytes = (n + 7) / 8;
  std::memset(bitset, 0, static_cast<size_t>(nbytes));
  if (!ids || nids == 0) return OVVS_STATUS_SUCCESS;
  for (int64_t i = 0; i < nids; ++i) {
    const int64_t id = ids[i];
    if (id < 0 || id >= n) continue;
    bitset[id >> 3] = static_cast<uint8_t>(bitset[id >> 3] | (1u << (id & 7)));
  }
  return OVVS_STATUS_SUCCESS;
}

#include "ovvs_shave.h"

void ovvsShaveTopkSmallest(const float* scores, int32_t cols, int32_t k, int32_t* idx, float* val) {
  ovvs_shave_topk_smallest(scores, cols, k, idx, val);
}

float ovvsShavePqAdc(const float* tables, const uint8_t* code, int32_t pq_m, int32_t ks) {
  return ovvs_shave_pq_adc(tables, code, pq_m, ks);
}

ovvsStatus ovvsPqAdcBatch(ovvsResources_t res, const float* tables, int32_t pq_m, int32_t ks,
                          const uint8_t* codes, int64_t ncodes, float* out) {
  if (!res) return OVVS_STATUS_INVALID_ARGUMENT;
  return ovvs::impl::prim_pq_adc(*rd(res), tables, pq_m, ks, codes, ncodes, out);
}
