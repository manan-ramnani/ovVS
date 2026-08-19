/* SHAVE-portable scalar kernel compiled into libiovs (host + future npu_compiler). */
#include "iovs_shave.h"

void iovs_shave_topk_smallest(const float* scores, int32_t cols, int32_t k, int32_t* idx,
                              float* val) {
  for (int32_t t = 0; t < k; ++t) {
    int32_t best = -1;
    float bv = 3.4e38f;
    for (int32_t j = 0; j < cols; ++j) {
      int used = 0;
      for (int32_t u = 0; u < t; ++u) {
        if (idx[u] == j) {
          used = 1;
          break;
        }
      }
      if (used) continue;
      if (scores[j] < bv) {
        bv = scores[j];
        best = j;
      }
    }
    idx[t] = best;
    val[t] = bv;
  }
}
