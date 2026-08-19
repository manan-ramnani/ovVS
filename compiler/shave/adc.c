/* SHAVE-portable ADC kernel compiled into libiovs. */
#include "iovs_shave.h"

float iovs_shave_pq_adc(const float* tables, const uint8_t* code, int32_t pq_m, int32_t ks) {
  float s = 0.f;
  for (int32_t m = 0; m < pq_m; ++m) {
    s += tables[m * ks + code[m]];
  }
  return s;
}
