#include "iovs/iovs.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static float l2sq(const float* a, const float* b, int64_t d) {
  float s = 0.f;
  for (int64_t i = 0; i < d; ++i) {
    const float t = a[i] - b[i];
    s += t * t;
  }
  return s;
}

int main() {
  iovs::Resources res;
  const int64_t n = 12, dim = 4, k = 3;
  std::vector<float> data(static_cast<size_t>(n * dim));
  for (int64_t i = 0; i < n * dim; ++i) data[static_cast<size_t>(i)] = ((i * 17) % 100) / 50.f - 1.f;
  const float q[4] = {0.1f, -0.2f, 0.3f, 0.0f};
  iovs::BruteForceIndex ix(res, data.data(), n, dim, IOVS_METRIC_L2_EXPANDED);
  int64_t nb[3];
  float ds[3];
  ix.search(q, 1, k, nb, ds);
  int64_t truth[3] = {-1, -1, -1};
  float td[3] = {1e30f, 1e30f, 1e30f};
  for (int64_t i = 0; i < n; ++i) {
    const float d = l2sq(q, data.data() + i * dim, dim);
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
      std::fprintf(stderr, "cxx mismatch t=%d got=%lld expect=%lld\n", t, static_cast<long long>(nb[t]),
                   static_cast<long long>(truth[t]));
      return 1;
    }
  }
  std::printf("cxx consumer ok neighbors=%lld,%lld,%lld version=%s\n", static_cast<long long>(nb[0]),
              static_cast<long long>(nb[1]), static_cast<long long>(nb[2]), iovsGetVersion());
  return 0;
}
