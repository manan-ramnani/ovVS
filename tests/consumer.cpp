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

  const int64_t allow[4] = {truth[0], truth[1], 0, 1};
  std::vector<uint8_t> bits(static_cast<size_t>((n + 7) / 8), 0);
  iovs::bitset_from_allow_list(n, allow, 4, bits.data());
  int64_t fnb[3];
  float fds[3];
  ix.search(q, 1, k, fnb, fds, bits.data());
  bool allowed = false;
  for (int a = 0; a < 4; ++a)
    if (fnb[0] == allow[a]) allowed = true;
  if (!allowed) {
    std::fprintf(stderr, "cxx allow-list neighbor %lld not in list\n", static_cast<long long>(fnb[0]));
    return 1;
  }

  iovs::BruteForceIndex ham(res, data.data(), n, dim, IOVS_METRIC_BITWISE_HAMMING);
  int64_t hnb[3];
  float hds[3];
  ham.search(q, 1, k, hnb, hds);
  int64_t htruth = -1;
  float hbest = 1e30f;
  for (int64_t i = 0; i < n; ++i) {
    float s = 0.f;
    for (int64_t d = 0; d < dim; ++d) {
      const uint32_t ax = data[static_cast<size_t>(i * dim + d)] >= 0.f;
      const uint32_t ay = q[d] >= 0.f;
      s += static_cast<float>(ax ^ ay);
    }
    if (s < hbest) {
      hbest = s;
      htruth = i;
    }
  }
  auto ham_score = [&](int64_t id) {
    float s = 0.f;
    for (int64_t d = 0; d < dim; ++d) {
      const uint32_t ax = data[static_cast<size_t>(id * dim + d)] >= 0.f;
      const uint32_t ay = q[d] >= 0.f;
      s += static_cast<float>(ax ^ ay);
    }
    return s;
  };
  if (hnb[0] < 0 || hnb[0] >= n || ham_score(hnb[0]) != hbest) {
    std::fprintf(stderr, "cxx hamming mismatch got=%lld expect_id=%lld expect_score=%.0f\n",
                 static_cast<long long>(hnb[0]), static_cast<long long>(htruth), hbest);
    return 1;
  }

  iovs::IvfFlatIndex ivf(res, data.data(), n, dim, IOVS_METRIC_L2_EXPANDED, 4);
  int64_t inb[3];
  float ids[3];
  ivf.search(q, 1, k, 4, inb, ids);
  if (inb[0] < 0 || inb[0] >= n) {
    std::fprintf(stderr, "cxx ivf-flat id out of range\n");
    return 1;
  }

  iovs::CagraIndex cagra(res, data.data(), n, dim, IOVS_METRIC_L2_EXPANDED, 4, 8);
  int64_t cnb[3];
  float cds[3];
  cagra.search(q, 1, k, 8, 2, cnb, cds);
  if (cnb[0] < 0 || cnb[0] >= n) {
    std::fprintf(stderr, "cxx cagra id out of range\n");
    return 1;
  }

  iovs::IvfPqIndex pq(res, data.data(), n, dim, IOVS_METRIC_L2_EXPANDED, 4, 2, 8);
  int64_t pnb[3];
  float pds[3];
  pq.search(q, 1, k, 4, 8, pnb, pds);
  if (pnb[0] < 0 || pnb[0] >= n) {
    std::fprintf(stderr, "cxx ivf-pq id out of range\n");
    return 1;
  }

  std::printf("cxx consumer wrappers ok ivf=%lld cagra=%lld pq=%lld ham=%lld allow=%lld\n",
              static_cast<long long>(inb[0]), static_cast<long long>(cnb[0]),
              static_cast<long long>(pnb[0]), static_cast<long long>(hnb[0]),
              static_cast<long long>(fnb[0]));
  return 0;
}
