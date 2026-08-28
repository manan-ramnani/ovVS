#include "ovvs/ovvs.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

_Static_assert(OVVS_IVF_PQ_SEARCH_STATS_ABI_V1 == 1u, "unexpected IVF-PQ stats ABI version");
_Static_assert(sizeof(ovvsIvfPqSearchStatsV1) == 200, "unexpected IVF-PQ stats ABI size");
_Static_assert(OVVS_CAGRA_BUILD_STATS_ABI_V1 == 1u, "unexpected CAGRA build stats ABI version");
_Static_assert(sizeof(ovvsCagraBuildStatsV1) == 256, "unexpected CAGRA build stats ABI size");
#define OVVS_ASSERT_CAGRA_OFFSET(field, offset) \
  _Static_assert(offsetof(ovvsCagraBuildStatsV1, field) == (offset), \
                 "unexpected CAGRA build stats offset: " #field)
OVVS_ASSERT_CAGRA_OFFSET(abi_version, 0);
OVVS_ASSERT_CAGRA_OFFSET(struct_size, 4);
OVVS_ASSERT_CAGRA_OFFSET(successful_calls, 8);
OVVS_ASSERT_CAGRA_OFFSET(rows, 16);
OVVS_ASSERT_CAGRA_OFFSET(dataset_copy_bytes, 24);
OVVS_ASSERT_CAGRA_OFFSET(initializer_graph_payload_bytes, 32);
OVVS_ASSERT_CAGRA_OFFSET(published_graph_copy_bytes, 40);
OVVS_ASSERT_CAGRA_OFFSET(nndescent_initializer_calls, 48);
OVVS_ASSERT_CAGRA_OFFSET(ivfpq_initializer_calls, 56);
OVVS_ASSERT_CAGRA_OFFSET(iterative_initializer_calls, 64);
OVVS_ASSERT_CAGRA_OFFSET(total_wall_ns, 72);
OVVS_ASSERT_CAGRA_OFFSET(dataset_copy_ns, 80);
OVVS_ASSERT_CAGRA_OFFSET(initializer_ns, 88);
OVVS_ASSERT_CAGRA_OFFSET(optimizer_prune_merge_ns, 96);
OVVS_ASSERT_CAGRA_OFFSET(index_materialize_ns, 104);
OVVS_ASSERT_CAGRA_OFFSET(initializer_final_cpu_calls, 112);
OVVS_ASSERT_CAGRA_OFFSET(initializer_final_gpu_calls, 120);
OVVS_ASSERT_CAGRA_OFFSET(initializer_final_npu_calls, 128);
OVVS_ASSERT_CAGRA_OFFSET(nndescent_gpu_iterations, 136);
OVVS_ASSERT_CAGRA_OFFSET(nndescent_gpu_converged_calls, 144);
OVVS_ASSERT_CAGRA_OFFSET(nndescent_gpu_final_changed_edges, 152);
OVVS_ASSERT_CAGRA_OFFSET(nndescent_gpu_final_pending_new_edges, 160);
OVVS_ASSERT_CAGRA_OFFSET(nndescent_gpu_instrumented_calls, 168);
OVVS_ASSERT_CAGRA_OFFSET(nndescent_gpu_allocation_calls, 176);
OVVS_ASSERT_CAGRA_OFFSET(nndescent_gpu_allocation_bytes, 184);
OVVS_ASSERT_CAGRA_OFFSET(nndescent_gpu_h2d_calls, 192);
OVVS_ASSERT_CAGRA_OFFSET(nndescent_gpu_h2d_bytes, 200);
OVVS_ASSERT_CAGRA_OFFSET(nndescent_gpu_d2h_calls, 208);
OVVS_ASSERT_CAGRA_OFFSET(nndescent_gpu_d2h_bytes, 216);
OVVS_ASSERT_CAGRA_OFFSET(nndescent_gpu_kernel_launches, 224);
OVVS_ASSERT_CAGRA_OFFSET(nndescent_gpu_submission_calls, 232);
OVVS_ASSERT_CAGRA_OFFSET(nndescent_gpu_wait_calls, 240);
OVVS_ASSERT_CAGRA_OFFSET(nndescent_gpu_peak_owned_bytes_max, 248);
#undef OVVS_ASSERT_CAGRA_OFFSET

static float l2sq(const float* a, const float* b, int64_t d) {
  float s = 0.f;
  for (int64_t i = 0; i < d; ++i) {
    float t = a[i] - b[i];
    s += t * t;
  }
  return s;
}

int main(void) {
  ovvsResources_t res = NULL;
  if (ovvsResourcesCreate(&res) != OVVS_STATUS_SUCCESS) {
    fprintf(stderr, "create failed\n");
    return 1;
  }
  ovvsIvfPqSearchStatsV1 ivfpq_stats = {0};
  if (ovvsResourcesIvfPqSearchStatsV1(res, &ivfpq_stats) != OVVS_STATUS_SUCCESS ||
      ivfpq_stats.abi_version != OVVS_IVF_PQ_SEARCH_STATS_ABI_V1 ||
      ivfpq_stats.struct_size != sizeof(ivfpq_stats)) {
    fprintf(stderr, "IVF-PQ stats ABI mismatch\n");
    return 1;
  }
  ovvsCagraBuildStatsV1 cagra_build_stats = {0};
  if (ovvsResourcesCagraBuildStatsV1(res, &cagra_build_stats) != OVVS_STATUS_SUCCESS ||
      cagra_build_stats.abi_version != OVVS_CAGRA_BUILD_STATS_ABI_V1 ||
      cagra_build_stats.struct_size != sizeof(cagra_build_stats) ||
      cagra_build_stats.successful_calls != 0) {
    fprintf(stderr, "CAGRA build stats ABI mismatch\n");
    return 1;
  }
  const int64_t n = 12, dim = 4, k = 3;
  float data[12 * 4];
  float q[4] = {0.1f, -0.2f, 0.3f, 0.0f};
  for (int64_t i = 0; i < n * dim; ++i) data[i] = ((i * 17) % 100) / 50.f - 1.f;

  ovvsBruteForceIndex_t ix = NULL;
  if (ovvsBruteForceBuild(res, data, n, dim, OVVS_METRIC_L2_EXPANDED, &ix) != OVVS_STATUS_SUCCESS) {
    fprintf(stderr, "build failed\n");
    return 1;
  }
  int64_t nb[3];
  float ds[3];
  if (ovvsBruteForceSearch(res, ix, q, 1, k, NULL, nb, ds) != OVVS_STATUS_SUCCESS) {
    fprintf(stderr, "search failed\n");
    return 1;
  }

  /* Oracle computed here from the same input, not a hardcoded matrix. */
  int64_t truth[3] = {-1, -1, -1};
  float td[3] = {1e30f, 1e30f, 1e30f};
  for (int64_t i = 0; i < n; ++i) {
    float d = l2sq(q, data + i * dim, dim);
    for (int t = 0; t < 3; ++t) {
      if (d < td[t]) {
        for (int u = 2; u > t; --u) {
          td[u] = td[u - 1];
          truth[u] = truth[u - 1];
        }
        td[t] = d;
        truth[t] = i;
        break;
      }
    }
  }
  for (int t = 0; t < 3; ++t) {
    if (nb[t] != truth[t]) {
      fprintf(stderr, "mismatch t=%d got=%lld expect=%lld\n", t, (long long)nb[t],
              (long long)truth[t]);
      return 1;
    }
  }
  printf("consumer ok neighbors=%lld,%lld,%lld version=%s\n", (long long)nb[0], (long long)nb[1],
         (long long)nb[2], ovvsGetVersion());
  ovvsBruteForceDestroy(ix);
  ovvsResourcesDestroy(res);
  return 0;
}
