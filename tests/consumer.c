#include "ovvs/ovvs.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

_Static_assert(OVVS_IVF_PQ_SEARCH_STATS_ABI_V1 == 1u, "unexpected IVF-PQ stats ABI version");
_Static_assert(sizeof(ovvsIvfPqSearchStatsV1) == 200, "unexpected IVF-PQ stats ABI size");

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
