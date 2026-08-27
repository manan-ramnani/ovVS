#include "test_harness.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <latch>
#include <numeric>
#include <set>
#include <thread>
#include <vector>

static float oracle_dot(const float* a, const float* b, int64_t d) {
  float s = 0.f;
  for (int64_t i = 0; i < d; ++i) s += a[i] * b[i];
  return s;
}

static float oracle_n2(const float* a, int64_t d) {
  float s = 0.f;
  for (int64_t i = 0; i < d; ++i) s += a[i] * a[i];
  return s;
}

/* Independent of libovvs search: rank dataset rows by a caller-supplied score (higher is better). */
static void rank_oracle(const float* data, int64_t n, int64_t dim, const float* q, int64_t k,
                        int64_t* idx, float* out_score, bool ip) {
  std::vector<int64_t> order(static_cast<size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::vector<float> sc(static_cast<size_t>(n));
  const float qn = std::sqrt(std::max(oracle_n2(q, dim), 1e-12f));
  for (int64_t j = 0; j < n; ++j) {
    const float* x = data + j * dim;
    if (ip) {
      sc[static_cast<size_t>(j)] = oracle_dot(q, x, dim);
    } else {
      const float xn = std::sqrt(std::max(oracle_n2(x, dim), 1e-12f));
      sc[static_cast<size_t>(j)] = oracle_dot(q, x, dim) / (qn * xn);
    }
  }
  std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int64_t a, int64_t b) {
    return sc[static_cast<size_t>(a)] > sc[static_cast<size_t>(b)];
  });
  for (int64_t t = 0; t < k; ++t) {
    idx[t] = order[static_cast<size_t>(t)];
    out_score[t] = sc[static_cast<size_t>(order[static_cast<size_t>(t)])];
  }
}

static void brute_oracle(const float* data, int64_t n, int64_t dim, const float* q, int64_t nq,
                         int64_t k, int64_t* idx, float* dist) {
  for (int64_t qi = 0; qi < nq; ++qi) {
    std::vector<int64_t> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::vector<float> sc(static_cast<size_t>(n));
    const float* qq = q + qi * dim;
    for (int64_t j = 0; j < n; ++j) {
      float s = 0.f;
      const float* x = data + j * dim;
      for (int64_t d = 0; d < dim; ++d) {
        const float t = qq[d] - x[d];
        s += t * t;
      }
      sc[static_cast<size_t>(j)] = s;
    }
    std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int64_t a, int64_t b) {
      return sc[static_cast<size_t>(a)] < sc[static_cast<size_t>(b)];
    });
    for (int64_t t = 0; t < k; ++t) {
      idx[qi * k + t] = order[static_cast<size_t>(t)];
      dist[qi * k + t] = sc[static_cast<size_t>(order[static_cast<size_t>(t)])];
    }
  }
}

OVVS_TEST(brute_force_recall_one) {
  Res res;
  const int64_t n = 40, dim = 8, nq = 5, k = 4;
  auto data = make_data(n, dim, 100);
  auto q = make_data(nq, dim, 101);
  ovvsBruteForceIndex_t ix = nullptr;
  expect_status(ovvsBruteForceBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, &ix), "bf");
  std::vector<int64_t> got(static_cast<size_t>(nq * k)), truth(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k)), td(static_cast<size_t>(nq * k));
  expect_status(ovvsBruteForceSearch(res.r, ix, q.data(), nq, k, nullptr, got.data(), gd.data()),
                "bfs");
  brute_oracle(data.data(), n, dim, q.data(), nq, k, truth.data(), td.data());
  expect(recall_at_k(got.data(), truth.data(), nq, k) > 0.999f, "bf recall");
  ovvsBruteForceDestroy(ix);
}

OVVS_TEST(brute_force_inner_product_vs_dot_oracle) {
  Res res;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  const float data[] = {1.f, 0.f, 0.f, 1.f, -1.f, 0.f};
  const float q[] = {1.f, 0.f};
  ovvsBruteForceIndex_t ix = nullptr;
  expect_status(ovvsBruteForceBuild(res.r, data, 3, 2, OVVS_METRIC_INNER_PRODUCT, &ix), "ip build");
  int64_t nb[2];
  float ds[2];
  expect_status(ovvsBruteForceSearch(res.r, ix, q, 1, 2, nullptr, nb, ds), "ip search");
  int64_t truth[2];
  float tscore[2];
  rank_oracle(data, 3, 2, q, 2, truth, tscore, true);
  expect(nb[0] == truth[0] && nb[1] == truth[1], "ip neighbor ids");
  expect(std::fabs(ds[0] - tscore[0]) < 1e-5f && std::fabs(ds[1] - tscore[1]) < 1e-5f, "ip values");
  expect(nb[0] == 0 && nb[1] == 1, "ip order vs [1,0],[0,1],[-1,0]");
  ovvsBruteForceDestroy(ix);
}

OVVS_TEST(brute_force_cosine_vs_oracle) {
  Res res;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  const float data[] = {1.f, 0.f, 0.f, 1.f, -1.f, 0.f};
  const float q[] = {1.f, 0.f};
  ovvsBruteForceIndex_t ix = nullptr;
  expect_status(ovvsBruteForceBuild(res.r, data, 3, 2, OVVS_METRIC_COSINE_EXPANDED, &ix), "cos build");
  int64_t nb[2];
  float ds[2];
  expect_status(ovvsBruteForceSearch(res.r, ix, q, 1, 2, nullptr, nb, ds), "cos search");
  int64_t truth[2];
  float tscore[2];
  rank_oracle(data, 3, 2, q, 2, truth, tscore, false);
  expect(nb[0] == truth[0] && nb[1] == truth[1], "cosine neighbor ids");
  expect(nb[0] == 0 && nb[1] == 1, "cosine order");
  /* API reports 1-cos distance. */
  expect(std::fabs(ds[0] - (1.f - tscore[0])) < 1e-5f, "cosine dist 0");
  expect(std::fabs(ds[1] - (1.f - tscore[1])) < 1e-5f, "cosine dist 1");
  ovvsBruteForceDestroy(ix);
}

OVVS_TEST(ivf_flat_inner_product_refine_vs_oracle) {
  Res res;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  const float data[] = {1.f, 0.f, 0.f, 1.f, -1.f, 0.f, 0.5f, 0.5f};
  const float q[] = {1.f, 0.f};
  ovvsIvfFlatIndex_t ix = nullptr;
  expect_status(ovvsIvfFlatBuild(res.r, data, 4, 2, OVVS_METRIC_INNER_PRODUCT, 1, &ix), "ivf ip");
  int64_t nb[2];
  float ds[2];
  expect_status(ovvsIvfFlatSearch(res.r, ix, q, 1, 2, 1, nullptr, nb, ds), "ivf ip search");
  int64_t truth[2];
  float tscore[2];
  rank_oracle(data, 4, 2, q, 2, truth, tscore, true);
  expect(nb[0] == truth[0] && nb[1] == truth[1], "ivf ip ids");
  expect(std::fabs(ds[0] - tscore[0]) < 1e-4f, "ivf ip value");
  ovvsIvfFlatDestroy(ix);
}

OVVS_TEST(ivf_pq_cosine_refine_vs_oracle) {
  Res res;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  const int64_t n = 12, dim = 4, k = 3;
  auto data = make_data(n, dim, 222);
  const float q[] = {0.4f, -0.1f, 0.2f, 0.8f};
  ovvsIvfPqIndex_t ix = nullptr;
  expect_status(ovvsIvfPqBuild(res.r, data.data(), n, dim, OVVS_METRIC_COSINE_EXPANDED, 1, 2, 8, &ix),
                "ivfpq cos");
  std::vector<int64_t> nb(static_cast<size_t>(k));
  std::vector<float> ds(static_cast<size_t>(k));
  /* nprobe=1, krefine large: refine is exact over the whole list. */
  expect_status(ovvsIvfPqSearch(res.r, ix, q, 1, k, 1, static_cast<int32_t>(n), nullptr, nb.data(),
                                ds.data()),
                "ivfpq cos search");
  int64_t truth[3];
  float tscore[3];
  rank_oracle(data.data(), n, dim, q, k, truth, tscore, false);
  for (int64_t t = 0; t < k; ++t) expect(nb[static_cast<size_t>(t)] == truth[t], "ivfpq cosine id");
  ovvsIvfPqDestroy(ix);
}

OVVS_TEST(allow_list_bitset_and_hamming_lp) {
  Res res;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  const int64_t n = 16, dim = 4, k = 3;
  auto data = make_data(n, dim, 112);
  auto q = make_data(1, dim, 113);
  const int64_t allow[4] = {2, 5, 9, 12};
  std::vector<uint8_t> bits(static_cast<size_t>((n + 7) / 8), 0);
  expect_status(ovvsBitsetFromAllowList(n, allow, 4, bits.data()), "allow");
  expect((bits[2 >> 3] & (1u << (2 & 7))) != 0, "bit 2");
  expect((bits[0] & 1u) == 0, "bit 0 off");
  ovvsBruteForceIndex_t ix = nullptr;
  expect_status(ovvsBruteForceBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, &ix), "bf");
  std::vector<int64_t> nb(static_cast<size_t>(k));
  std::vector<float> ds(static_cast<size_t>(k));
  expect_status(ovvsBruteForceSearch(res.r, ix, q.data(), 1, k, bits.data(), nb.data(), ds.data()), "s");
  for (int64_t t = 0; t < k; ++t) {
    bool ok = false;
    for (int64_t a : allow)
      if (nb[static_cast<size_t>(t)] == a) ok = true;
    expect(ok, "allow-list neighbor");
  }
  ovvsBruteForceDestroy(ix);

  expect_status(ovvsBruteForceBuild(res.r, data.data(), n, dim, OVVS_METRIC_BITWISE_HAMMING, &ix), "ham");
  expect_status(ovvsBruteForceSearch(res.r, ix, q.data(), 1, k, nullptr, nb.data(), ds.data()), "hams");
  int64_t truth[3] = {-1, -1, -1};
  float td[3] = {1e30f, 1e30f, 1e30f};
  for (int64_t i = 0; i < n; ++i) {
    float s = 0.f;
    for (int64_t d = 0; d < dim; ++d) {
      const uint32_t ax = data[static_cast<size_t>(i * dim + d)] >= 0.f;
      const uint32_t ay = q[static_cast<size_t>(d)] >= 0.f;
      s += static_cast<float>(ax ^ ay);
    }
    for (int t = 0; t < 3; ++t) {
      if (s < td[t]) {
        for (int u = 2; u > t; --u) {
          td[u] = td[u - 1];
          truth[u] = truth[u - 1];
        }
        td[t] = s;
        truth[t] = i;
        break;
      }
    }
  }
  expect(nb[0] == truth[0], "hamming top1");
  ovvsBruteForceDestroy(ix);

  expect_status(ovvsBruteForceBuild(res.r, data.data(), n, dim, OVVS_METRIC_LP_UNEXPANDED, &ix), "lp");
  expect_status(ovvsBruteForceSearch(res.r, ix, q.data(), 1, k, nullptr, nb.data(), ds.data()), "lps");
  expect(nb[0] >= 0 && nb[0] < n, "lp id");
  ovvsBruteForceDestroy(ix);
}

OVVS_TEST(force_gpu_hamming_lp_vs_oracle) {
  Res res;
  int32_t gpu = 0;
  ovvsResourcesGpuAvailable(res.r, &gpu);
  if (!gpu) return;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU);
  const int64_t n = 16, dim = 4, k = 3;
  auto data = make_data(n, dim, 112);
  auto q = make_data(1, dim, 113);
  ovvsBruteForceIndex_t ix = nullptr;
  expect_status(ovvsBruteForceBuild(res.r, data.data(), n, dim, OVVS_METRIC_BITWISE_HAMMING, &ix), "ham");
  std::vector<int64_t> nb(static_cast<size_t>(k));
  std::vector<float> ds(static_cast<size_t>(k));
  const ovvsStatus st = ovvsBruteForceSearch(res.r, ix, q.data(), 1, k, nullptr, nb.data(), ds.data());
  if (st == OVVS_STATUS_DEVICE_UNAVAILABLE) {
    ovvsBruteForceDestroy(ix);
    expect(!ovvsSyclEnabled(), "SYCL GPU Hamming DEVICE_UNAVAILABLE");
    return;
  }
  expect_status(st, "ham gpu");
  ovvsDevice last = OVVS_DEVICE_CPU;
  expect_status(ovvsResourcesLastDevice(res.r, &last), "last");
  expect(last == OVVS_DEVICE_GPU, "FORCE_GPU hamming last_device");
  std::printf("    hamming last_device=gpu\n");
  float hbest = 1e30f;
  for (int64_t i = 0; i < n; ++i) {
    float s = 0.f;
    for (int64_t d = 0; d < dim; ++d) {
      const uint32_t ax = data[static_cast<size_t>(i * dim + d)] >= 0.f;
      const uint32_t ay = q[static_cast<size_t>(d)] >= 0.f;
      s += static_cast<float>(ax ^ ay);
    }
    if (s < hbest) hbest = s;
  }
  expect(nb[0] >= 0 && nb[0] < n, "ham id");
  float got = 0.f;
  for (int64_t d = 0; d < dim; ++d) {
    const uint32_t ax = data[static_cast<size_t>(nb[0] * dim + d)] >= 0.f;
    const uint32_t ay = q[static_cast<size_t>(d)] >= 0.f;
    got += static_cast<float>(ax ^ ay);
  }
  expect(got == hbest, "hamming min score");
  ovvsBruteForceDestroy(ix);

  expect_status(ovvsBruteForceBuild(res.r, data.data(), n, dim, OVVS_METRIC_LP_UNEXPANDED, &ix), "lp");
  expect_status(ovvsBruteForceSearch(res.r, ix, q.data(), 1, k, nullptr, nb.data(), ds.data()), "lps");
  expect_status(ovvsResourcesLastDevice(res.r, &last), "lp last");
  expect(last == OVVS_DEVICE_GPU, "FORCE_GPU lp last_device");
  float pbest = 1e30f;
  int64_t pid = -1;
  const float p = 2.f;
  for (int64_t i = 0; i < n; ++i) {
    float s = 0.f;
    for (int64_t d = 0; d < dim; ++d) {
      const float t = std::fabs(data[static_cast<size_t>(i * dim + d)] - q[static_cast<size_t>(d)]);
      s += t * t;
    }
    s = std::sqrt(s);
    if (s < pbest) {
      pbest = s;
      pid = i;
    }
  }
  (void)pid;
  expect(nb[0] >= 0 && nb[0] < n, "lp id gpu");
  float pg = 0.f;
  for (int64_t d = 0; d < dim; ++d) {
    const float t = std::fabs(data[static_cast<size_t>(nb[0] * dim + d)] - q[static_cast<size_t>(d)]);
    pg += t * t;
  }
  pg = std::sqrt(pg);
  expect(std::fabs(pg - pbest) < 1e-4f, "lp min score");
  std::printf("    lp last_device=gpu\n");
  (void)p;
  ovvsBruteForceDestroy(ix);
}

OVVS_TEST(force_npu_hamming_lp_vs_oracle) {
  Res res;
  int32_t npu = 0;
  ovvsResourcesNpuAvailable(res.r, &npu);
  if (!npu) return;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_NPU);
  const int64_t n = 16, dim = 4, k = 3;
  auto data = make_data(n, dim, 112);
  auto q = make_data(1, dim, 113);
  ovvsBruteForceIndex_t ix = nullptr;
  expect_status(ovvsBruteForceBuild(res.r, data.data(), n, dim, OVVS_METRIC_BITWISE_HAMMING, &ix), "ham");
  std::vector<int64_t> nb(static_cast<size_t>(k));
  std::vector<float> ds(static_cast<size_t>(k));
  expect_status(ovvsBruteForceSearch(res.r, ix, q.data(), 1, k, nullptr, nb.data(), ds.data()), "ham npu");
  ovvsDevice last = OVVS_DEVICE_CPU;
  expect_status(ovvsResourcesLastDevice(res.r, &last), "last");
  expect(last == OVVS_DEVICE_NPU, "FORCE_NPU hamming last_device");
  std::printf("    hamming last_device=npu\n");
  float hbest = 1e30f;
  for (int64_t i = 0; i < n; ++i) {
    float s = 0.f;
    for (int64_t d = 0; d < dim; ++d) {
      const uint32_t ax = data[static_cast<size_t>(i * dim + d)] >= 0.f;
      const uint32_t ay = q[static_cast<size_t>(d)] >= 0.f;
      s += static_cast<float>(ax ^ ay);
    }
    if (s < hbest) hbest = s;
  }
  expect(nb[0] >= 0 && nb[0] < n, "ham id");
  float got = 0.f;
  for (int64_t d = 0; d < dim; ++d) {
    const uint32_t ax = data[static_cast<size_t>(nb[0] * dim + d)] >= 0.f;
    const uint32_t ay = q[static_cast<size_t>(d)] >= 0.f;
    got += static_cast<float>(ax ^ ay);
  }
  expect(got == hbest, "hamming min score npu");
  ovvsBruteForceDestroy(ix);

  expect_status(ovvsBruteForceBuild(res.r, data.data(), n, dim, OVVS_METRIC_LP_UNEXPANDED, &ix), "lp");
  expect_status(ovvsBruteForceSearch(res.r, ix, q.data(), 1, k, nullptr, nb.data(), ds.data()), "lps npu");
  expect_status(ovvsResourcesLastDevice(res.r, &last), "lp last");
  expect(last == OVVS_DEVICE_NPU, "FORCE_NPU lp last_device");
  float pbest = 1e30f;
  const float p = 2.f;
  for (int64_t i = 0; i < n; ++i) {
    float s = 0.f;
    for (int64_t d = 0; d < dim; ++d) {
      const float t = std::fabs(data[static_cast<size_t>(i * dim + d)] - q[static_cast<size_t>(d)]);
      s += t * t;
    }
    s = std::sqrt(s);
    if (s < pbest) pbest = s;
  }
  expect(nb[0] >= 0 && nb[0] < n, "lp id npu");
  float pg = 0.f;
  for (int64_t d = 0; d < dim; ++d) {
    const float t = std::fabs(data[static_cast<size_t>(nb[0] * dim + d)] - q[static_cast<size_t>(d)]);
    pg += t * t;
  }
  pg = std::sqrt(pg);
  expect(std::fabs(pg - pbest) < 2e-2f, "lp min score npu");
  std::printf("    lp last_device=npu\n");
  (void)p;
  ovvsBruteForceDestroy(ix);
}

OVVS_TEST(brute_force_bitset_filter) {
  Res res;
  const int64_t n = 20, dim = 4, k = 3;
  auto data = make_data(n, dim, 110);
  auto q = make_data(1, dim, 111);
  std::vector<uint8_t> bits((n + 7) / 8, 0);
  for (int64_t i = 0; i < n; ++i) {
    if (i % 2 == 0) bits[static_cast<size_t>(i >> 3)] |= static_cast<uint8_t>(1u << (i & 7));
  }
  ovvsBruteForceIndex_t ix = nullptr;
  expect_status(ovvsBruteForceBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, &ix), "bf");
  std::vector<int64_t> nb(static_cast<size_t>(k));
  std::vector<float> ds(static_cast<size_t>(k));
  expect_status(ovvsBruteForceSearch(res.r, ix, q.data(), 1, k, bits.data(), nb.data(), ds.data()),
                "bfs");
  for (int64_t t = 0; t < k; ++t) expect((nb[static_cast<size_t>(t)] % 2) == 0, "filter even");
  ovvsBruteForceDestroy(ix);
}

OVVS_TEST(ivf_flat_recall) {
  Res res;
  const int64_t n = 80, dim = 8, nq = 6, k = 5;
  auto data = make_data(n, dim, 200);
  auto q = make_data(nq, dim, 201);
  ovvsIvfFlatIndex_t ix = nullptr;
  expect_status(ovvsIvfFlatBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 8, &ix), "ivf");
  std::vector<int64_t> got(static_cast<size_t>(nq * k)), truth(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k)), td(static_cast<size_t>(nq * k));
  expect_status(ovvsIvfFlatSearch(res.r, ix, q.data(), nq, k, 8, nullptr, got.data(), gd.data()),
                "ivfs");
  brute_oracle(data.data(), n, dim, q.data(), nq, k, truth.data(), td.data());
  const float rec = recall_at_k(got.data(), truth.data(), nq, k);
  expect(rec >= 0.85f, "ivf-flat recall " + std::to_string(rec));
  ovvsIvfFlatDestroy(ix);
}

OVVS_TEST(ivf_pq_recall) {
  Res res;
  const int64_t n = 64, dim = 8, nq = 5, k = 4;
  auto data = make_data(n, dim, 210);
  auto q = make_data(nq, dim, 211);
  ovvsIvfPqIndex_t ix = nullptr;
  expect_status(ovvsIvfPqBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 8, 4, 8, &ix),
                "ivfpq");
  std::vector<int64_t> got(static_cast<size_t>(nq * k)), truth(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k)), td(static_cast<size_t>(nq * k));
  expect_status(ovvsIvfPqSearch(res.r, ix, q.data(), nq, k, 8, 16, nullptr, got.data(), gd.data()),
                "ivfpqs");
  brute_oracle(data.data(), n, dim, q.data(), nq, k, truth.data(), td.data());
  const float rec = recall_at_k(got.data(), truth.data(), nq, k);
  expect(rec >= 0.5f, "ivf-pq recall " + std::to_string(rec));
  const auto pqpath = std::filesystem::temp_directory_path() / "ovvs_ivfpq.bin";
  expect_status(ovvsIvfPqSerialize(ix, pqpath.string().c_str()), "pqser");
  ovvsIvfPqIndex_t loaded = nullptr;
  expect_status(ovvsIvfPqDeserialize(res.r, pqpath.string().c_str(), &loaded), "pqdes");
  auto extra = make_data(4, dim, 212);
  expect_status(ovvsIvfPqExtend(res.r, loaded, extra.data(), 4), "pqext");
  ovvsIvfPqDestroy(loaded);
  ovvsIvfPqDestroy(ix);
}

OVVS_TEST(ivf_rabitq_recall) {
  Res res;
  const int64_t n = 64, dim = 8, nq = 5, k = 4;
  auto data = make_data(n, dim, 220);
  auto q = make_data(nq, dim, 221);
  ovvsIvfRabitqIndex_t ix = nullptr;
  expect_status(ovvsIvfRabitqBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 8, &ix),
                "rabitq");
  std::vector<int64_t> got(static_cast<size_t>(nq * k)), truth(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k)), td(static_cast<size_t>(nq * k));
  expect_status(
      ovvsIvfRabitqSearch(res.r, ix, q.data(), nq, k, 8, 16, nullptr, got.data(), gd.data()), "rqs");
  brute_oracle(data.data(), n, dim, q.data(), nq, k, truth.data(), td.data());
  const float rec = recall_at_k(got.data(), truth.data(), nq, k);
  expect(rec >= 0.45f, "rabitq recall " + std::to_string(rec));
  const auto rqpath = std::filesystem::temp_directory_path() / "ovvs_rabitq.bin";
  expect_status(ovvsIvfRabitqSerialize(ix, rqpath.string().c_str()), "rqser");
  ovvsIvfRabitqIndex_t loaded = nullptr;
  expect_status(ovvsIvfRabitqDeserialize(res.r, rqpath.string().c_str(), &loaded), "rqdes");
  auto extra = make_data(4, dim, 223);
  expect_status(ovvsIvfRabitqExtend(res.r, loaded, extra.data(), 4), "rqext");
  ovvsIvfRabitqDestroy(loaded);
  ovvsIvfRabitqDestroy(ix);
}

OVVS_TEST(nndescent_graph_overlap) {
  Res res;
  const int64_t n = 30, dim = 6;
  const int32_t deg = 5;
  auto data = make_data(n, dim, 300);
  ovvsNnDescentGraph_t g = nullptr;
  expect_status(ovvsNnDescentBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, deg, 4, &g),
                "nnd");
  const int32_t* ids = nullptr;
  int64_t nn = 0;
  int32_t d = 0;
  expect_status(ovvsNnDescentNeighbors(g, &ids, &nn, &d), "nndn");
  expect(nn == n && d == deg, "nnd shape");
  std::vector<int64_t> truth(static_cast<size_t>(n * deg));
  std::vector<float> td(static_cast<size_t>(n * deg));
  expect_status(ovvsAllNeighbors(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, deg, truth.data(),
                                 td.data()),
                "alln");
  int hit = 0;
  for (int64_t i = 0; i < n; ++i) {
    std::set<int64_t> tset(truth.begin() + i * deg, truth.begin() + (i + 1) * deg);
    for (int32_t t = 0; t < deg; ++t) {
      if (tset.count(ids[i * deg + t])) ++hit;
    }
  }
  const float ov = static_cast<float>(hit) / static_cast<float>(n * deg);
  expect(ov >= 0.6f, "nnd overlap " + std::to_string(ov));
  ovvsNnDescentDestroy(g);
}

OVVS_TEST(cagra_build_search_filter_serialize) {
  Res res;
  const int64_t n = 36, dim = 8, nq = 4, k = 4;
  auto data = make_data(n, dim, 400);
  auto q = make_data(nq, dim, 401);
  ovvsCagraIndex_t ix = nullptr;
  expect_status(ovvsCagraBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 8, 16, &ix),
                "cagra");
  std::vector<int64_t> got(static_cast<size_t>(nq * k)), truth(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k)), td(static_cast<size_t>(nq * k));
  expect_status(ovvsCagraSearch(res.r, ix, q.data(), nq, k, 16, 2, nullptr, got.data(), gd.data()),
                "cags");
  brute_oracle(data.data(), n, dim, q.data(), nq, k, truth.data(), td.data());
  const float rec = recall_at_k(got.data(), truth.data(), nq, k);
  expect(rec >= 0.9f, "cagra recall " + std::to_string(rec));

  std::vector<uint8_t> bits((n + 7) / 8, 0xff);
  bits[0] = 0xfe; /* drop id 0 */
  expect_status(ovvsCagraSearch(res.r, ix, q.data(), nq, k, 16, 2, bits.data(), got.data(), gd.data()),
                "cagf");
  for (int64_t i = 0; i < nq * k; ++i) expect(got[static_cast<size_t>(i)] != 0, "cagra filter");

  const auto path = std::filesystem::temp_directory_path() / "ovvs_cagra_test.bin";
  expect_status(ovvsCagraSerialize(ix, path.string().c_str()), "ser");
  ovvsCagraIndex_t loaded = nullptr;
  expect_status(ovvsCagraDeserialize(res.r, path.string().c_str(), &loaded), "des");
  std::vector<int64_t> g2(static_cast<size_t>(nq * k));
  std::vector<float> d2(static_cast<size_t>(nq * k));
  expect_status(ovvsCagraSearch(res.r, loaded, q.data(), nq, k, 16, 2, nullptr, g2.data(), d2.data()),
                "cags2");
  expect(recall_at_k(g2.data(), truth.data(), nq, k) >= 0.7f, "deser recall");
  auto extra = make_data(4, dim, 409);
  expect_status(ovvsCagraExtend(res.r, loaded, extra.data(), 4), "ext");
  ovvsCagraDestroy(loaded);
  ovvsCagraDestroy(ix);
}

OVVS_TEST(hnsw_from_cagra) {
  Res res;
  const int64_t n = 28, dim = 6, nq = 3, k = 3;
  auto data = make_data(n, dim, 410);
  auto q = make_data(nq, dim, 411);
  ovvsCagraIndex_t cg = nullptr;
  expect_status(ovvsCagraBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 6, 12, &cg),
                "cagra");
  ovvsHnswIndex_t hx = nullptr;
  expect_status(ovvsHnswFromCagra(res.r, cg, &hx), "hnsw");
  std::vector<int64_t> got(static_cast<size_t>(nq * k)), truth(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k)), td(static_cast<size_t>(nq * k));
  expect_status(ovvsHnswSearch(res.r, hx, q.data(), nq, k, 16, got.data(), gd.data()), "hnsws");
  brute_oracle(data.data(), n, dim, q.data(), nq, k, truth.data(), td.data());
  expect(recall_at_k(got.data(), truth.data(), nq, k) >= 0.6f, "hnsw recall");
  ovvsHnswDestroy(hx);
  ovvsCagraDestroy(cg);
}

OVVS_TEST(vamana_and_scann) {
  Res res;
  const int64_t n = 40, dim = 8, nq = 4, k = 4;
  auto data = make_data(n, dim, 500);
  auto q = make_data(nq, dim, 501);
  ovvsVamanaIndex_t vx = nullptr;
  expect_status(ovvsVamanaBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 8, 1.2f, &vx),
                "vam");
  std::vector<int64_t> got(static_cast<size_t>(nq * k)), truth(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k)), td(static_cast<size_t>(nq * k));
  expect_status(ovvsVamanaSearch(res.r, vx, q.data(), nq, k, 16, nullptr, got.data(), gd.data()),
                "vams");
  brute_oracle(data.data(), n, dim, q.data(), nq, k, truth.data(), td.data());
  expect(recall_at_k(got.data(), truth.data(), nq, k) >= 0.55f, "vamana recall");
  ovvsVamanaDestroy(vx);

  ovvsScannIndex_t sx = nullptr;
  expect_status(ovvsScannBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 8, 4, &sx),
                "scann");
  expect_status(ovvsScannSearch(res.r, sx, q.data(), nq, k, 8, 12, got.data(), gd.data()), "scanns");
  expect(recall_at_k(got.data(), truth.data(), nq, k) >= 0.35f, "scann recall");
  ovvsScannDestroy(sx);

  /* Full nprobe + krefine=n: original-space refine must match brute L2 ids. */
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  ovvsScannIndex_t sx2 = nullptr;
  expect_status(ovvsScannBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 4, 4, &sx2),
                "scann2");
  expect_status(ovvsScannSearch(res.r, sx2, q.data(), nq, k, 4, static_cast<int32_t>(n), got.data(),
                                gd.data()),
                "scann-refine");
  expect(recall_at_k(got.data(), truth.data(), nq, k) >= 0.99f, "scann original refine");
  ovvsScannDestroy(sx2);
}

OVVS_TEST(cagra_graph_build_params_and_q) {
  Res res;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  const int64_t n = 32, dim = 8, nq = 4, k = 4;
  auto data = make_data(n, dim, 420);
  auto q = make_data(nq, dim, 421);
  std::vector<int64_t> truth(static_cast<size_t>(nq * k));
  std::vector<float> td(static_cast<size_t>(nq * k));
  brute_oracle(data.data(), n, dim, q.data(), nq, k, truth.data(), td.data());
  const ovvsCagraBuildAlgo algos[] = {OVVS_CAGRA_BUILD_NN_DESCENT, OVVS_CAGRA_BUILD_IVF_PQ,
                                      OVVS_CAGRA_BUILD_ITERATIVE};
  const char* names[] = {"nnd", "ivfpq", "iter"};
  for (int a = 0; a < 3; ++a) {
    ovvsCagraIndex_t ix = nullptr;
    expect_status(ovvsCagraBuildEx(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 8, 16, algos[a],
                                   &ix),
                  names[a]);
    std::vector<int64_t> got(static_cast<size_t>(nq * k));
    std::vector<float> gd(static_cast<size_t>(nq * k));
    expect_status(ovvsCagraSearch(res.r, ix, q.data(), nq, k, 16, 2, nullptr, got.data(), gd.data()),
                  names[a]);
    const float rec = recall_at_k(got.data(), truth.data(), nq, k);
    expect(rec >= 0.45f, std::string(names[a]) + " recall " + std::to_string(rec));
    ovvsCagraDestroy(ix);
  }
  ovvsCagraIndex_t qx = nullptr;
  expect_status(ovvsCagraBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 8, 16, &qx), "qbuild");
  expect_status(ovvsCagraQuantize(res.r, qx, 4, 8), "q");
  std::vector<int64_t> got(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k));
  expect_status(ovvsCagraSearch(res.r, qx, q.data(), nq, k, 16, 2, nullptr, got.data(), gd.data()), "qs");
  const float qrec = recall_at_k(got.data(), truth.data(), nq, k);
  expect(qrec >= 0.35f, "cagra-q recall " + std::to_string(qrec));
  const auto path = std::filesystem::temp_directory_path() / "ovvs_cagra_detach.bin";
  expect_status(ovvsCagraSerializeEx(qx, path.string().c_str(), 0), "ser0");
  expect_status(ovvsCagraDetachDataset(qx), "det");
  expect_status(ovvsCagraSearch(res.r, qx, q.data(), nq, k, 16, 2, nullptr, got.data(), gd.data()),
                "q-detached");
  expect_status(ovvsCagraAttachDataset(qx, data.data(), n, dim), "att");
  ovvsCagraIndex_t loaded = nullptr;
  expect_status(ovvsCagraDeserialize(res.r, path.string().c_str(), &loaded), "des0");
  expect_status(ovvsCagraAttachDataset(loaded, data.data(), n, dim), "att2");
  expect_status(ovvsCagraSearch(res.r, loaded, q.data(), nq, k, 16, 2, nullptr, got.data(), gd.data()),
                "att-search");
  ovvsCagraDestroy(loaded);
  ovvsCagraDestroy(qx);
}

OVVS_TEST(cagra_extend_recall_hold) {
  Res res;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  const int64_t n = 40, extra = 6, dim = 8, nq = 4, k = 4;
  auto all = make_data(n + extra, dim, 430);
  auto q = make_data(nq, dim, 431);
  std::vector<int64_t> truth(static_cast<size_t>(nq * k));
  std::vector<float> td(static_cast<size_t>(nq * k));
  brute_oracle(all.data(), n + extra, dim, q.data(), nq, k, truth.data(), td.data());
  ovvsCagraIndex_t base = nullptr;
  expect_status(ovvsCagraBuild(res.r, all.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 8, 16, &base), "b");
  expect_status(ovvsCagraExtend(res.r, base, all.data() + n * dim, extra), "ext");
  std::vector<int64_t> got(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k));
  expect_status(ovvsCagraSearch(res.r, base, q.data(), nq, k, 20, 2, nullptr, got.data(), gd.data()),
                "es");
  const float rec_ext = recall_at_k(got.data(), truth.data(), nq, k);
  ovvsCagraIndex_t rebuilt = nullptr;
  expect_status(ovvsCagraBuild(res.r, all.data(), n + extra, dim, OVVS_METRIC_L2_EXPANDED, 8, 16, &rebuilt),
                "rb");
  std::vector<int64_t> gr(static_cast<size_t>(nq * k));
  expect_status(ovvsCagraSearch(res.r, rebuilt, q.data(), nq, k, 20, 2, nullptr, gr.data(), gd.data()),
                "rs");
  const float rec_rb = recall_at_k(gr.data(), truth.data(), nq, k);
  expect(rec_ext >= 0.45f, "extend recall " + std::to_string(rec_ext));
  expect(rec_ext + 0.2f >= rec_rb, "extend holds vs rebuild");
  ovvsCagraDestroy(base);
  ovvsCagraDestroy(rebuilt);
}

OVVS_TEST(cagra_force_gpu_last_device) {
  Res res;
  int32_t gpu = 0;
  ovvsResourcesGpuAvailable(res.r, &gpu);
  if (!gpu) return;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU);
  const int64_t n = 24, dim = 8, k = 3;
  auto data = make_data(n, dim, 440);
  auto q = make_data(2, dim, 441);
  ovvsCagraIndex_t ix = nullptr;
  expect_status(ovvsCagraBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 6, 12, &ix), "b");
  std::vector<int64_t> nb(static_cast<size_t>(2 * k));
  std::vector<float> ds(static_cast<size_t>(2 * k));
  expect_status(ovvsCagraSearch(res.r, ix, q.data(), 2, k, 12, 2, nullptr, nb.data(), ds.data()), "s");
  ovvsDevice last = OVVS_DEVICE_CPU;
  ovvsResourcesLastDevice(res.r, &last);
  if (ovvsSyclEnabled()) {
    expect(last == OVVS_DEVICE_GPU, "sycl walk last_device");
  } else {
    expect(last == OVVS_DEVICE_GPU, "openvino-gpu walk last_device");
  }
  ovvsCagraDestroy(ix);
}

OVVS_TEST(cagra_sycl_walk_n_over_4096) {
  if (!ovvsSyclEnabled()) return;
  Res res;
  int32_t gpu = 0;
  ovvsResourcesGpuAvailable(res.r, &gpu);
  if (!gpu) return;
  /* n>4096: old kernel skipped expanding id>=4096 and overflowed slm at search_width>64. */
  const int64_t n = 4200, dim = 8, nq = 4, k = 5;
  auto data = make_data(n, dim, 442);
  auto q = make_data(nq, dim, 443);
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  ovvsCagraIndex_t ix = nullptr;
  expect_status(ovvsCagraBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 8, 16, &ix), "b");
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU);
  std::vector<int64_t> got(static_cast<size_t>(nq * k)), truth(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k)), td(static_cast<size_t>(nq * k));
  expect_status(ovvsCagraSearch(res.r, ix, q.data(), nq, k, 32, 80, nullptr, got.data(), gd.data()), "s");
  ovvsDevice last = OVVS_DEVICE_CPU;
  ovvsResourcesLastDevice(res.r, &last);
  expect(last == OVVS_DEVICE_GPU, "sycl n>4096 last_device");
  brute_oracle(data.data(), n, dim, q.data(), nq, k, truth.data(), td.data());
  for (int64_t i = 0; i < nq * k; ++i) {
    expect(got[static_cast<size_t>(i)] >= 0 && got[static_cast<size_t>(i)] < n, "id range");
  }
  const float rec = recall_at_k(got.data(), truth.data(), nq, k);
  expect(rec >= 0.2f, "sycl n>4096 recall vs independent L2 " + std::to_string(rec));
  ovvsCagraDestroy(ix);
}

OVVS_TEST(ivf_flat_serialize_extend) {
  Res res;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  const int64_t n = 40, dim = 8, extra = 5, nq = 4, k = 4;
  auto all = make_data(n + extra, dim, 450);
  auto q = make_data(nq, dim, 451);
  ovvsIvfFlatIndex_t ix = nullptr;
  expect_status(ovvsIvfFlatBuild(res.r, all.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 8, &ix), "b");
  const auto path = std::filesystem::temp_directory_path() / "ovvs_ivfflat.bin";
  expect_status(ovvsIvfFlatSerialize(ix, path.string().c_str()), "ser");
  ovvsIvfFlatIndex_t loaded = nullptr;
  expect_status(ovvsIvfFlatDeserialize(res.r, path.string().c_str(), &loaded), "des");
  expect_status(ovvsIvfFlatExtend(res.r, loaded, all.data() + n * dim, extra), "ext");
  std::vector<int64_t> got(static_cast<size_t>(nq * k)), truth(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k)), td(static_cast<size_t>(nq * k));
  expect_status(ovvsIvfFlatSearch(res.r, loaded, q.data(), nq, k, 8, nullptr, got.data(), gd.data()),
                "s");
  brute_oracle(all.data(), n + extra, dim, q.data(), nq, k, truth.data(), td.data());
  expect(recall_at_k(got.data(), truth.data(), nq, k) >= 0.7f, "ivf extend recall");
  ovvsIvfFlatDestroy(loaded);
  ovvsIvfFlatDestroy(ix);
}

OVVS_TEST(vamana_serialize_mmap_filter) {
  Res res;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  const int64_t n = 28, dim = 6, nq = 3, k = 3;
  auto data = make_data(n, dim, 460);
  auto q = make_data(nq, dim, 461);
  ovvsVamanaIndex_t vx = nullptr;
  expect_status(ovvsVamanaBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 6, 1.2f, &vx), "b");
  const auto path = std::filesystem::temp_directory_path() / "ovvs_vamana.bin";
  expect_status(ovvsVamanaSerialize(vx, path.string().c_str()), "ser");
  ovvsVamanaIndex_t mapped = nullptr;
  expect_status(ovvsVamanaMmap(res.r, path.string().c_str(), &mapped), "mmap");
  std::vector<uint8_t> bits((n + 7) / 8, 0xff);
  bits[0] = 0xfe;
  std::vector<int64_t> got(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k));
  expect_status(ovvsVamanaSearch(res.r, mapped, q.data(), nq, k, 12, bits.data(), got.data(), gd.data()),
                "ms");
  for (int64_t i = 0; i < nq * k; ++i) expect(got[static_cast<size_t>(i)] != 0, "vamana mmap filter");
  ovvsVamanaDestroy(mapped);
  ovvsVamanaDestroy(vx);
}

OVVS_TEST(hnsw_hnswlib_format_roundtrip) {
  Res res;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  const int64_t n = 24, dim = 6, nq = 3, k = 3;
  auto data = make_data(n, dim, 470);
  auto q = make_data(nq, dim, 471);
  ovvsCagraIndex_t cg = nullptr;
  expect_status(ovvsCagraBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 6, 12, &cg), "c");
  ovvsHnswIndex_t hx = nullptr;
  expect_status(ovvsHnswFromCagra(res.r, cg, &hx), "from");
  std::vector<int64_t> cagra_nb(static_cast<size_t>(nq * k)), hnsw_nb(static_cast<size_t>(nq * k));
  std::vector<float> ds(static_cast<size_t>(nq * k));
  expect_status(ovvsCagraSearch(res.r, cg, q.data(), nq, k, 16, 2, nullptr, cagra_nb.data(), ds.data()),
                "cs");
  expect_status(ovvsHnswSearch(res.r, hx, q.data(), nq, k, 16, hnsw_nb.data(), ds.data()), "hs");
  const float overlap = recall_at_k(hnsw_nb.data(), cagra_nb.data(), nq, k);
  expect(overlap >= 0.5f, "hnsw vs cagra " + std::to_string(overlap));
  const auto path = std::filesystem::temp_directory_path() / "ovvs_hnsw.bin";
  expect_status(ovvsHnswSerialize(hx, path.string().c_str()), "ser");
  ovvsHnswIndex_t loaded = nullptr;
  expect_status(ovvsHnswDeserialize(res.r, path.string().c_str(), &loaded), "des");
  std::vector<int64_t> lnb(static_cast<size_t>(nq * k));
  expect_status(ovvsHnswSearch(res.r, loaded, q.data(), nq, k, 16, lnb.data(), ds.data()), "ls");
  expect(recall_at_k(lnb.data(), hnsw_nb.data(), nq, k) >= 0.7f, "hnsw deser");
  /* Header is hnswlib saveIndex: size_t offsetLevel0 at byte 0 must be 0. */
  std::ifstream hf(path, std::ios::binary);
  size_t offset0 = 1;
  hf.read(reinterpret_cast<char*>(&offset0), sizeof(size_t));
  expect(offset0 == 0, "hnswlib offsetLevel0");
  ovvsHnswDestroy(loaded);
  ovvsHnswDestroy(hx);
  ovvsCagraDestroy(cg);
}

/* Independent IEEE-754 binary16 decoder (not libovvs). Search is compared to L2 on these floats. */
static float oracle_f16_to_f32(uint16_t h) {
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

static uint16_t oracle_f32_to_f16(float f) {
  uint32_t bits = 0;
  std::memcpy(&bits, &f, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  const uint32_t man = bits & 0x7fffffu;
  if (exp <= 0) return static_cast<uint16_t>(sign);
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13));
}

OVVS_TEST(brute_fp16_int8_vs_fp32_oracle) {
  Res res;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  const int64_t n = 20, dim = 8, k = 4;
  auto data = make_data(n, dim, 480);
  auto q = make_data(1, dim, 481);
  std::vector<uint16_t> h(static_cast<size_t>(n * dim));
  std::vector<int8_t> i8(static_cast<size_t>(n * dim));
  std::vector<float> f16host(static_cast<size_t>(n * dim));
  for (size_t i = 0; i < h.size(); ++i) {
    h[i] = oracle_f32_to_f16(data[i]);
    f16host[i] = oracle_f16_to_f32(h[i]);
    float s = std::max(-127.f, std::min(127.f, std::round(data[i] * 50.f)));
    i8[i] = static_cast<int8_t>(s);
  }
  std::vector<int64_t> truth(static_cast<size_t>(k));
  std::vector<float> td(static_cast<size_t>(k));
  brute_oracle(f16host.data(), n, dim, q.data(), 1, k, truth.data(), td.data());
  ovvsBruteForceIndex_t hix = nullptr;
  expect_status(ovvsBruteForceBuildTyped(res.r, h.data(), n, dim, OVVS_METRIC_L2_EXPANDED, OVVS_DTYPE_F16,
                                         &hix),
                "f16");
  std::vector<int64_t> nb(static_cast<size_t>(k));
  std::vector<float> ds(static_cast<size_t>(k));
  expect_status(ovvsBruteForceSearch(res.r, hix, q.data(), 1, k, nullptr, nb.data(), ds.data()), "f16s");
  /* Distances are L2(query_f32, decode(f16(dataset))). Ids must match that oracle.
     Atol 2e-2 covers binary16 mantissa (~1e-3) accumulated over dim=8. */
  constexpr float kF16DistAtol = 2e-2f;
  for (int64_t t = 0; t < k; ++t) {
    expect(nb[static_cast<size_t>(t)] == truth[static_cast<size_t>(t)], "f16 neighbor id");
    expect(std::fabs(ds[static_cast<size_t>(t)] - td[static_cast<size_t>(t)]) < kF16DistAtol, "f16 dist");
  }
  ovvsBruteForceDestroy(hix);

  std::vector<float> i8f(static_cast<size_t>(n * dim));
  for (size_t i = 0; i < i8.size(); ++i) i8f[i] = static_cast<float>(i8[i]);
  brute_oracle(i8f.data(), n, dim, q.data(), 1, k, truth.data(), td.data());
  ovvsBruteForceIndex_t iix = nullptr;
  expect_status(ovvsBruteForceBuildTyped(res.r, i8.data(), n, dim, OVVS_METRIC_L2_EXPANDED, OVVS_DTYPE_I8,
                                         &iix),
                "i8");
  expect_status(ovvsBruteForceSearch(res.r, iix, q.data(), 1, k, nullptr, nb.data(), ds.data()), "i8s");
  for (int64_t t = 0; t < k; ++t) expect(nb[static_cast<size_t>(t)] == truth[static_cast<size_t>(t)], "i8 id");
  ovvsBruteForceDestroy(iix);
}

OVVS_TEST(dynamic_batcher_matches_eager) {
  Res res;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  const int64_t n = 30, dim = 8, nq = 4, k = 3;
  auto data = make_data(n, dim, 490);
  auto q = make_data(nq, dim, 491);
  ovvsBruteForceIndex_t ix = nullptr;
  expect_status(ovvsBruteForceBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, &ix), "b");
  std::vector<int64_t> eager(static_cast<size_t>(nq * k));
  std::vector<float> eds(static_cast<size_t>(nq * k));
  expect_status(ovvsBruteForceSearch(res.r, ix, q.data(), nq, k, nullptr, eager.data(), eds.data()), "e");

  /* Concurrent nq=1 submits: max_batch=4, max_wait=250ms so they merge into one prim search. */
  ovvsBatcher_t b = nullptr;
  expect_status(ovvsBatcherCreate(res.r, ix, 4, 250, &b), "batch");
  std::vector<int64_t> got(static_cast<size_t>(nq * k), -1);
  std::vector<float> gds(static_cast<size_t>(nq * k), 0.f);
  std::vector<ovvsStatus> sts(static_cast<size_t>(nq), OVVS_STATUS_ERROR);
  std::latch go(static_cast<std::ptrdiff_t>(nq));
  std::vector<std::thread> th;
  th.reserve(static_cast<size_t>(nq));
  for (int64_t i = 0; i < nq; ++i) {
    th.emplace_back([&, i] {
      go.arrive_and_wait();
      sts[static_cast<size_t>(i)] = ovvsBatcherSearch(b, q.data() + i * dim, 1, k,
                                                      got.data() + i * k, gds.data() + i * k);
    });
  }
  for (auto& t : th) t.join();
  for (int64_t i = 0; i < nq; ++i) expect_status(sts[static_cast<size_t>(i)], "thread search");
  int32_t coalesced = 0;
  expect_status(ovvsBatcherLastBatchSize(b, &coalesced), "last batch");
  expect(coalesced == static_cast<int32_t>(nq), "concurrent queries must coalesce");
  for (int64_t i = 0; i < nq * k; ++i)
    expect(got[static_cast<size_t>(i)] == eager[static_cast<size_t>(i)], "coalesced vs eager");
  ovvsBatcherDestroy(b);
  ovvsBruteForceDestroy(ix);
}
