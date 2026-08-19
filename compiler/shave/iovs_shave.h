#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void iovs_shave_topk_smallest(const float* scores, int32_t cols, int32_t k, int32_t* idx, float* val);
float iovs_shave_pq_adc(const float* tables, const uint8_t* code, int32_t pq_m, int32_t ks);

#ifdef __cplusplus
}
#endif
