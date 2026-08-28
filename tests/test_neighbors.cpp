#include "test_harness.hpp"
#include "internal.hpp"

#include <algorithm>
#include <chrono>
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

static float sampled_graph_l2_overlap(const float* data, int64_t n, int64_t dim,
                                       const int32_t* graph, int32_t degree,
                                       int32_t sample_rows) {
  int64_t hits = 0;
  for (int32_t sample = 0; sample < sample_rows; ++sample) {
    const int64_t row = (static_cast<int64_t>(sample) * 104729 + 17) % n;
    std::vector<int64_t> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::vector<float> scores(static_cast<size_t>(n));
    const float* query = data + row * dim;
    for (int64_t candidate = 0; candidate < n; ++candidate) {
      float score = 0.f;
      for (int64_t d = 0; d < dim; ++d) {
        const float delta = query[d] - data[candidate * dim + d];
        score += delta * delta;
      }
      scores[static_cast<size_t>(candidate)] = candidate == row ? std::numeric_limits<float>::max()
                                                                : score;
    }
    std::partial_sort(order.begin(), order.begin() + degree, order.end(),
                      [&](int64_t a, int64_t b) {
                        if (scores[static_cast<size_t>(a)] != scores[static_cast<size_t>(b)]) {
                          return scores[static_cast<size_t>(a)] < scores[static_cast<size_t>(b)];
                        }
                        return a < b;
                      });
    std::set<int64_t> truth(order.begin(), order.begin() + degree);
    for (int32_t edge = 0; edge < degree; ++edge) {
      if (truth.count(graph[row * degree + edge])) ++hits;
    }
  }
  return static_cast<float>(hits) / static_cast<float>(sample_rows * degree);
}

struct NnDescentStatsSnapshot {
  int32_t iterations_run = 0;
  int64_t changed_edges = 0;
  int64_t pending_new_edges = 0;
  double change_ratio = 0.0;
  int64_t peak_device_bytes = 0;
  bool converged = false;
};

static NnDescentStatsSnapshot nndescent_stats(ovvsResources_t resources) {
  auto* data = ovvs::impl::rd(resources);
  std::lock_guard<std::mutex> lock(data->nndescent_stats_mutex);
  return {data->nndescent_iterations_run, data->nndescent_changed_edges,
          data->nndescent_pending_new_edges, data->nndescent_change_ratio,
          data->nndescent_peak_device_bytes, data->nndescent_converged};
}

static bool same_nndescent_stats(const NnDescentStatsSnapshot& lhs,
                                 const NnDescentStatsSnapshot& rhs) {
  return lhs.iterations_run == rhs.iterations_run &&
         lhs.changed_edges == rhs.changed_edges &&
         lhs.pending_new_edges == rhs.pending_new_edges &&
         lhs.change_ratio == rhs.change_ratio &&
         lhs.peak_device_bytes == rhs.peak_device_bytes && lhs.converged == rhs.converged;
}

static float l2_distance(const float* data, int64_t dim, int64_t lhs, int64_t rhs) {
  float distance = 0.f;
  for (int64_t d = 0; d < dim; ++d) {
    const float delta = data[lhs * dim + d] - data[rhs * dim + d];
    distance += delta * delta;
  }
  return distance;
}

static void expect_graph_l2_rows_sorted(const float* data, int64_t n, int64_t dim,
                                        const int32_t* graph, int32_t degree) {
  for (int64_t row = 0; row < n; ++row) {
    float previous_distance = -1.f;
    int32_t previous_id = -1;
    for (int32_t edge = 0; edge < degree; ++edge) {
      const int32_t id = graph[row * degree + edge];
      const float distance = l2_distance(data, dim, row, id);
      expect(edge == 0 || previous_distance < distance ||
                 (previous_distance == distance && previous_id < id),
             "NN-Descent row must be ordered by (distance, id): row=" +
                 std::to_string(row) + " edge=" + std::to_string(edge) +
                 " previous_id=" + std::to_string(previous_id) + " id=" +
                 std::to_string(id) + " previous_distance=" +
                 std::to_string(previous_distance) + " distance=" + std::to_string(distance));
      previous_distance = distance;
      previous_id = id;
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
  const float overlap = sampled_graph_l2_overlap(data.data(), n, dim, ids, deg,
                                                  static_cast<int32_t>(n));
  expect(overlap == 1.f, "small exact NN-Descent overlap " + std::to_string(overlap));
  ovvsNnDescentDestroy(g);
}

OVVS_TEST(cagra_rank_optimizer_known_detours) {
  constexpr int64_t n = 6;
  constexpr int32_t initial_degree = 4;
  constexpr int32_t final_degree = 2;
  const std::vector<int32_t> initial = {
      1, 2, 3, 4,
      2, 3, 5, 4,
      3, 5, 1, 4,
      5, 1, 2, 4,
      5, 1, 2, 3,
      1, 2, 3, 4,
  };
  std::vector<int32_t> optimized;
  expect_status(ovvs::impl::cagra_optimize_ranked(initial.data(), n, initial_degree,
                                                   final_degree, optimized),
                "known detour-rank optimization");
  expect(optimized.size() == static_cast<size_t>(n * final_degree),
         "known rank optimizer shape");
  /* Row 0 has detour counts [0, 1, 2, 0] and no reverse incoming edges. */
  expect(optimized[0] == 1 && optimized[1] == 4,
         "detour rank must retain the two least-redundant row-0 edges");
  for (int64_t row = 0; row < n; ++row) {
    std::set<int32_t> unique;
    for (int32_t edge = 0; edge < final_degree; ++edge) {
      const int32_t id = optimized[static_cast<size_t>(row * final_degree + edge)];
      expect(id >= 0 && static_cast<int64_t>(id) < n, "rank optimizer ID range");
      expect(id != row, "rank optimizer self edge");
      unique.insert(id);
    }
    expect(static_cast<int32_t>(unique.size()) == final_degree,
           "rank optimizer fixed unique degree");
  }
}

OVVS_TEST(cagra_rank_optimizer_reverse_interleave) {
  constexpr int64_t n = 6;
  constexpr int32_t degree = 2;
  const std::vector<int32_t> initial = {
      1, 2,
      0, 3,
      1, 3,
      0, 4,
      0, 5,
      0, 1,
  };
  std::vector<int32_t> optimized;
  expect_status(ovvs::impl::cagra_optimize_ranked(initial.data(), n, degree, degree, optimized),
                "reverse interleave optimization");
  /* Incoming row-0 sources are 1,3,4,5 at forward rank 0. Source 1 is already forward,
     then source-ID tie breaking makes 3 the first novel reverse edge. */
  expect(optimized[0] == 1 && optimized[1] == 3,
         "rank optimizer must interleave one capped reverse edge");
}

OVVS_TEST(cagra_rank_optimizer_invalid_and_deterministic) {
  constexpr int64_t n = 6;
  constexpr int32_t initial_degree = 2;
  constexpr int32_t final_degree = 2;
  const std::vector<int32_t> initial = {
      1, 2,
      0, 3,
      1, 3,
      0, 4,
      0, 5,
      0, 1,
  };
  std::vector<int32_t> first;
  std::vector<int32_t> second;
  expect_status(ovvs::impl::cagra_optimize_ranked(initial.data(), n, initial_degree,
                                                   final_degree, first),
                "first deterministic rank optimization");
  expect_status(ovvs::impl::cagra_optimize_ranked(initial.data(), n, initial_degree,
                                                   final_degree, second),
                "second deterministic rank optimization");
  expect(first == second, "rank optimizer deterministic output");

  std::vector<int32_t> malformed = initial;
  malformed[1] = malformed[0];
  second = {99};
  expect(ovvs::impl::cagra_optimize_ranked(malformed.data(), n, initial_degree,
                                           final_degree, second) == OVVS_STATUS_ERROR,
         "rank optimizer duplicate row rejection");
  expect(second.empty(), "failed rank optimization clears output");

  malformed = initial;
  malformed[0] = 0;
  expect(ovvs::impl::cagra_optimize_ranked(malformed.data(), n, initial_degree,
                                           final_degree, second) == OVVS_STATUS_ERROR,
         "rank optimizer self-edge rejection");
  malformed = initial;
  malformed[0] = static_cast<int32_t>(n);
  expect(ovvs::impl::cagra_optimize_ranked(malformed.data(), n, initial_degree,
                                           final_degree, second) == OVVS_STATUS_ERROR,
         "rank optimizer out-of-range rejection");
  expect(ovvs::impl::cagra_optimize_ranked(initial.data(), n, initial_degree,
                                           initial_degree + 1, second) ==
             OVVS_STATUS_INVALID_ARGUMENT,
         "rank optimizer invalid degree rejection");
  expect(ovvs::impl::cagra_optimize_ranked(nullptr, n, initial_degree,
                                           final_degree, second) ==
             OVVS_STATUS_INVALID_ARGUMENT,
         "rank optimizer null graph rejection");
}

OVVS_TEST(nndescent_gpu_n_over_4096) {
  if (!ovvsSyclEnabled()) skip_test("SYCL GPU path unavailable");
  Res res;
  int32_t gpu = 0;
  expect_status(ovvsResourcesGpuAvailable(res.r, &gpu), "NN-Descent GPU availability");
  if (!gpu) skip_test("GPU device unavailable");

  const int64_t n = 4097, dim = 8;
  const int32_t degree = 8, iterations = 6;
  auto data = make_data(n, dim, 302);
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU);
  ovvsNnDescentGraph_t graph = nullptr;
  const auto build_started = std::chrono::steady_clock::now();
  expect_status(ovvsNnDescentBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED,
                                   degree, iterations, &graph),
                "large GPU NN-Descent build");
  const double build_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - build_started)
                              .count();
  ovvsDevice last = OVVS_DEVICE_CPU;
  expect_status(ovvsResourcesLastDevice(res.r, &last), "large NN-Descent last device");
  expect(last == OVVS_DEVICE_GPU, "large NN-Descent must run on GPU");

  const int32_t* ids = nullptr;
  int64_t graph_n = 0;
  int32_t graph_degree = 0;
  expect_status(ovvsNnDescentNeighbors(graph, &ids, &graph_n, &graph_degree),
                "large NN-Descent neighbors");
  expect(graph_n == n && graph_degree == degree, "large NN-Descent shape");
  for (int64_t row = 0; row < n; ++row) {
    std::set<int32_t> unique;
    for (int32_t edge = 0; edge < degree; ++edge) {
      const int32_t id = ids[row * degree + edge];
      expect(id >= 0 && static_cast<int64_t>(id) < n, "large NN-Descent ID range");
      expect(id != row, "large NN-Descent self edge");
      unique.insert(id);
    }
    expect(static_cast<int32_t>(unique.size()) == degree, "large NN-Descent unique row");
  }
  expect_graph_l2_rows_sorted(data.data(), n, dim, ids, degree);
  const NnDescentStatsSnapshot first_stats = nndescent_stats(res.r);
  const int64_t edge_count = n * degree;
  expect(first_stats.iterations_run > 0 && first_stats.iterations_run <= iterations,
         "NN-Descent iteration telemetry bounds");
  expect(first_stats.changed_edges >= 0 && first_stats.changed_edges <= edge_count,
         "NN-Descent exact changed-edge bounds");
  expect(first_stats.pending_new_edges >= 0 && first_stats.pending_new_edges <= edge_count,
         "NN-Descent pending-NEW bounds");
  expect(std::isfinite(first_stats.change_ratio) && first_stats.change_ratio >= 0.0 &&
             first_stats.change_ratio <= 1.0,
         "NN-Descent finite change-ratio bounds");
  expect(std::fabs(first_stats.change_ratio -
                   static_cast<double>(first_stats.changed_edges) /
                       static_cast<double>(edge_count)) < 1e-15,
         "NN-Descent exact changed-edge ratio");
  expect(first_stats.peak_device_bytes > 0, "NN-Descent peak owned bytes telemetry");
  expect(!first_stats.converged ||
             (first_stats.change_ratio < 0.0001 && first_stats.pending_new_edges == 0),
         "NN-Descent convergence requires no pending NEW edges");
  expect(first_stats.converged || first_stats.iterations_run == iterations,
         "NN-Descent non-converged build must exhaust its iteration budget");
  const float overlap = sampled_graph_l2_overlap(data.data(), n, dim, ids, degree, 64);
  std::printf("    nndescent_gpu_n_over_4096 build_ms=%.3f sampled_overlap=%.4f "
              "iterations=%d changed_edges=%lld pending_new_edges=%lld converged=%d "
              "peak_device_bytes=%lld\n",
              build_ms, static_cast<double>(overlap), first_stats.iterations_run,
              static_cast<long long>(first_stats.changed_edges),
              static_cast<long long>(first_stats.pending_new_edges), first_stats.converged ? 1 : 0,
              static_cast<long long>(first_stats.peak_device_bytes));
  expect(overlap >= 0.60f, "large NN-Descent sampled exact overlap " + std::to_string(overlap));

  std::vector<int32_t> first(ids, ids + n * degree);
  ovvsNnDescentDestroy(graph);
  graph = nullptr;
  expect_status(ovvsNnDescentBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED,
                                   degree, iterations, &graph),
                "deterministic GPU NN-Descent rebuild");
  expect_status(ovvsNnDescentNeighbors(graph, &ids, &graph_n, &graph_degree),
                "deterministic NN-Descent neighbors");
  expect(std::equal(first.begin(), first.end(), ids), "GPU NN-Descent deterministic graph");
  const NnDescentStatsSnapshot second_stats = nndescent_stats(res.r);
  expect(same_nndescent_stats(first_stats, second_stats),
         "GPU NN-Descent deterministic convergence telemetry");
  ovvsNnDescentDestroy(graph);
}

OVVS_TEST(nndescent_gpu_bounded_scale) {
  if (!ovvsSyclEnabled()) skip_test("SYCL GPU path unavailable");
  Res res;
  int32_t gpu = 0;
  expect_status(ovvsResourcesGpuAvailable(res.r, &gpu), "scale NN-Descent GPU availability");
  if (!gpu) skip_test("GPU device unavailable");

  const int64_t n = 16384, dim = 16;
  const int32_t degree = 16, iterations = 4;
  auto data = make_data(n, dim, 306);
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU);
  ovvsNnDescentGraph_t graph = nullptr;
  const auto build_started = std::chrono::steady_clock::now();
  expect_status(ovvsNnDescentBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED,
                                   degree, iterations, &graph),
                "bounded-scale GPU NN-Descent build");
  const double build_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - build_started)
                              .count();
  ovvsDevice last = OVVS_DEVICE_CPU;
  expect_status(ovvsResourcesLastDevice(res.r, &last), "scale NN-Descent last device");
  expect(last == OVVS_DEVICE_GPU, "bounded-scale NN-Descent must run on GPU");

  const int32_t* ids = nullptr;
  int64_t graph_n = 0;
  int32_t graph_degree = 0;
  expect_status(ovvsNnDescentNeighbors(graph, &ids, &graph_n, &graph_degree),
                "bounded-scale NN-Descent neighbors");
  expect(graph_n == n && graph_degree == degree, "bounded-scale NN-Descent shape");
  for (int64_t row = 0; row < n; ++row) {
    std::set<int32_t> unique;
    for (int32_t edge = 0; edge < degree; ++edge) {
      const int32_t id = ids[row * degree + edge];
      expect(id >= 0 && static_cast<int64_t>(id) < n, "bounded-scale NN-Descent ID range");
      expect(id != row, "bounded-scale NN-Descent self edge");
      unique.insert(id);
    }
    expect(static_cast<int32_t>(unique.size()) == degree,
           "bounded-scale NN-Descent unique row");
  }
  const NnDescentStatsSnapshot stats = nndescent_stats(res.r);
  const float overlap = sampled_graph_l2_overlap(data.data(), n, dim, ids, degree, 32);
  std::printf("    nndescent_gpu_bounded_scale build_ms=%.3f sampled_overlap=%.4f "
              "iterations=%d changed_edges=%lld pending_new_edges=%lld converged=%d "
              "peak_device_bytes=%lld\n",
              build_ms, static_cast<double>(overlap), stats.iterations_run,
              static_cast<long long>(stats.changed_edges),
              static_cast<long long>(stats.pending_new_edges), stats.converged ? 1 : 0,
              static_cast<long long>(stats.peak_device_bytes));
  expect(overlap >= 0.25f,
         "bounded-scale NN-Descent sampled exact overlap " + std::to_string(overlap));
  ovvsNnDescentDestroy(graph);
}

OVVS_TEST(nndescent_gpu_nonfinite_fails_closed) {
  if (!ovvsSyclEnabled()) skip_test("SYCL GPU path unavailable");
  Res res;
  int32_t gpu = 0;
  expect_status(ovvsResourcesGpuAvailable(res.r, &gpu), "nonfinite NN-Descent GPU availability");
  if (!gpu) skip_test("GPU device unavailable");

  const int64_t n = 4097, dim = 4;
  auto data = make_data(n, dim, 307);
  data[data.size() / 2u] = std::numeric_limits<float>::quiet_NaN();
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU);
  ovvsNnDescentGraph_t graph = reinterpret_cast<ovvsNnDescentGraph_t>(uintptr_t{1});
  expect(ovvsNnDescentBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 8, 4,
                            &graph) == OVVS_STATUS_INVALID_ARGUMENT,
         "nonfinite GPU NN-Descent input must fail closed");
  expect(graph == nullptr, "nonfinite GPU NN-Descent must not publish a graph");
  const NnDescentStatsSnapshot stats = nndescent_stats(res.r);
  expect(stats.iterations_run == 0 && stats.changed_edges == 0 &&
             stats.pending_new_edges == 0 && stats.change_ratio == 0.0 &&
             stats.peak_device_bytes == 0 && !stats.converged,
         "failed GPU NN-Descent must clear prior convergence telemetry");
}

OVVS_TEST(nndescent_gpu_structured_iteration_progress) {
  if (!ovvsSyclEnabled()) skip_test("SYCL GPU path unavailable");
  Res res;
  int32_t gpu = 0;
  expect_status(ovvsResourcesGpuAvailable(res.r, &gpu), "structured NN-Descent GPU availability");
  if (!gpu) skip_test("GPU device unavailable");

  const int64_t n = 4097, dim = 2;
  const int32_t degree = 8;
  std::vector<float> data(static_cast<size_t>(n * dim));
  for (int64_t row = 0; row < n; ++row) {
    data[static_cast<size_t>(row * dim)] = static_cast<float>(row) / 4096.f;
    data[static_cast<size_t>(row * dim + 1)] =
        static_cast<float>((row * 37) % 101) * 1e-5f;
  }
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU);

  ovvsNnDescentGraph_t one_iteration = nullptr;
  expect_status(ovvsNnDescentBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED,
                                   degree, 1, &one_iteration),
                "structured one-iteration NN-Descent build");
  const int32_t* ids = nullptr;
  int64_t graph_n = 0;
  int32_t graph_degree = 0;
  expect_status(ovvsNnDescentNeighbors(one_iteration, &ids, &graph_n, &graph_degree),
                "structured one-iteration neighbors");
  std::vector<int32_t> first(ids, ids + n * degree);
  ovvsNnDescentDestroy(one_iteration);

  ovvsNnDescentGraph_t two_iterations = nullptr;
  expect_status(ovvsNnDescentBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED,
                                   degree, 2, &two_iterations),
                "structured two-iteration NN-Descent build");
  expect_status(ovvsNnDescentNeighbors(two_iterations, &ids, &graph_n, &graph_degree),
                "structured two-iteration neighbors");
  const NnDescentStatsSnapshot stats = nndescent_stats(res.r);
  expect(stats.iterations_run == 2 && stats.changed_edges > 0,
         "structured second reverse/proposal join must make topology progress");
  expect(!std::equal(first.begin(), first.end(), ids),
         "structured second iteration must change the graph");
  ovvsNnDescentDestroy(two_iterations);
}

OVVS_TEST(nndescent_large_forced_policy_contract) {
  Res res;
  const int64_t n = 4097, dim = 2;
  auto data = make_data(n, dim, 303);
  auto a = make_data(4, 4, 304);
  auto b = make_data(4, 4, 305);
  std::vector<float> c(16);
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  expect_status(ovvsGemm(res.r, a.data(), b.data(), c.data(), 4, 4, 4, 1),
                "stamp CPU attribution");
  ovvsDevice last = OVVS_DEVICE_AUTO;
  expect_status(ovvsResourcesLastDevice(res.r, &last), "prior NN-Descent attribution");
  expect(last == OVVS_DEVICE_CPU, "CPU stamp");
  int32_t fallback_count = 0;
  expect_status(ovvsResourcesNpuFallbacks(res.r, &fallback_count), "NN-Descent fallback count");

  ovvsNnDescentGraph_t graph = reinterpret_cast<ovvsNnDescentGraph_t>(uintptr_t{1});
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU);
  expect(ovvsNnDescentBuild(res.r, data.data(), n, dim, OVVS_METRIC_LP_UNEXPANDED, 4, 1,
                            &graph) == OVVS_STATUS_DEVICE_UNAVAILABLE,
         "unsupported FORCE_GPU NN-Descent must be unavailable");
  expect(graph == nullptr, "failed FORCE_GPU NN-Descent must not publish a graph");
  int32_t after_gpu = 0;
  expect_status(ovvsResourcesNpuFallbacks(res.r, &after_gpu), "NN-Descent count after GPU");
  expect(after_gpu == fallback_count, "FORCE_GPU NN-Descent must not increment NPU fallback");
  expect_status(ovvsResourcesLastDevice(res.r, &last), "NN-Descent last after GPU failure");
  expect(last == OVVS_DEVICE_CPU, "failed FORCE_GPU NN-Descent preserves last device");

  ovvsCagraIndex_t cagra = reinterpret_cast<ovvsCagraIndex_t>(uintptr_t{1});
  expect(ovvsCagraBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 4, 8,
                        &cagra) == OVVS_STATUS_DEVICE_UNAVAILABLE,
         "FORCE_GPU CAGRA build must reject its host prune stage");
  expect(cagra == nullptr, "failed FORCE_GPU CAGRA build must not publish an index");

  graph = reinterpret_cast<ovvsNnDescentGraph_t>(uintptr_t{1});
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_NPU);
  expect(ovvsNnDescentBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 4, 1,
                            &graph) == OVVS_STATUS_DEVICE_UNAVAILABLE,
         "FORCE_NPU NN-Descent must be unavailable");
  expect(graph == nullptr, "failed FORCE_NPU NN-Descent must not publish a graph");
  int32_t after_npu = 0;
  expect_status(ovvsResourcesNpuFallbacks(res.r, &after_npu), "NN-Descent count after NPU");
  expect(after_npu == fallback_count + 1, "FORCE_NPU NN-Descent increments fallback once");
  expect_status(ovvsResourcesLastDevice(res.r, &last), "NN-Descent last after NPU failure");
  expect(last == OVVS_DEVICE_CPU, "failed FORCE_NPU NN-Descent preserves last device");

  cagra = reinterpret_cast<ovvsCagraIndex_t>(uintptr_t{1});
  expect(ovvsCagraBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 4, 8, &cagra) ==
             OVVS_STATUS_DEVICE_UNAVAILABLE,
         "CAGRA must propagate forced NN-Descent failure");
  expect(cagra == nullptr, "failed CAGRA build must not publish an index");
  expect_status(ovvsResourcesNpuFallbacks(res.r, &after_npu), "CAGRA propagated fallback count");
  expect(after_npu == fallback_count + 2, "CAGRA forced NPU failure increments once");

  cagra = reinterpret_cast<ovvsCagraIndex_t>(uintptr_t{1});
  expect(ovvsCagraBuildEx(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 4, 8,
                          OVVS_CAGRA_BUILD_ITERATIVE, &cagra) ==
             OVVS_STATUS_DEVICE_UNAVAILABLE,
         "iterative CAGRA must propagate forced NN-Descent failure");
  expect(cagra == nullptr, "failed iterative CAGRA build must not publish an index");
  expect_status(ovvsResourcesNpuFallbacks(res.r, &after_npu),
                "iterative CAGRA propagated fallback count");
  expect(after_npu == fallback_count + 3,
         "iterative CAGRA forced NPU failure increments once");

  ovvsVamanaIndex_t vamana = reinterpret_cast<ovvsVamanaIndex_t>(uintptr_t{1});
  expect(ovvsVamanaBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 4,
                         1.2f, &vamana) == OVVS_STATUS_DEVICE_UNAVAILABLE,
         "Vamana must propagate forced NN-Descent failure");
  expect(vamana == nullptr, "failed Vamana build must not publish an index");
  expect_status(ovvsResourcesNpuFallbacks(res.r, &after_npu),
                "Vamana propagated fallback count");
  expect(after_npu == fallback_count + 4,
         "Vamana forced NPU failure increments once");

  cagra = reinterpret_cast<ovvsCagraIndex_t>(uintptr_t{1});
  expect(ovvsCagraBuildEx(res.r, data.data(), n, dim, static_cast<ovvsMetric>(99), 4,
                          8, OVVS_CAGRA_BUILD_NN_DESCENT, &cagra) ==
             OVVS_STATUS_INVALID_ARGUMENT,
         "CAGRA invalid metric must reject");
  expect(cagra == nullptr, "invalid-metric CAGRA must not publish an index");
  cagra = reinterpret_cast<ovvsCagraIndex_t>(uintptr_t{1});
  expect(ovvsCagraBuildEx(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 4,
                          8, static_cast<ovvsCagraBuildAlgo>(99), &cagra) ==
             OVVS_STATUS_INVALID_ARGUMENT,
         "CAGRA invalid build algorithm must reject");
  expect(cagra == nullptr, "invalid-algorithm CAGRA must not publish an index");
  vamana = reinterpret_cast<ovvsVamanaIndex_t>(uintptr_t{1});
  expect(ovvsVamanaBuild(res.r, data.data(), n, dim, static_cast<ovvsMetric>(99), 4,
                         1.2f, &vamana) == OVVS_STATUS_INVALID_ARGUMENT,
         "Vamana invalid metric must reject");
  expect(vamana == nullptr, "invalid-metric Vamana must not publish an index");

  graph = reinterpret_cast<ovvsNnDescentGraph_t>(uintptr_t{1});
  expect(ovvsNnDescentBuild(res.r, data.data(),
                            static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1,
                            dim, OVVS_METRIC_L2_EXPANDED, 4, 1, &graph) ==
             OVVS_STATUS_SHAPE_MISMATCH,
         "NN-Descent int32 graph ID overflow must reject before reading data");
  expect(graph == nullptr, "shape rejection must not publish a graph");

  graph = reinterpret_cast<ovvsNnDescentGraph_t>(uintptr_t{1});
  expect(ovvsNnDescentBuild(nullptr, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED,
                            4, 1, &graph) == OVVS_STATUS_INVALID_ARGUMENT,
         "NN-Descent invalid resources must reject");
  expect(graph == nullptr, "invalid NN-Descent build must clear output");
  cagra = reinterpret_cast<ovvsCagraIndex_t>(uintptr_t{1});
  expect(ovvsCagraBuild(nullptr, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 4, 8,
                        &cagra) == OVVS_STATUS_INVALID_ARGUMENT,
         "CAGRA invalid resources must reject");
  expect(cagra == nullptr, "invalid CAGRA build must clear output");
  vamana = reinterpret_cast<ovvsVamanaIndex_t>(uintptr_t{1});
  expect(ovvsVamanaBuild(nullptr, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 4,
                         1.2f, &vamana) == OVVS_STATUS_INVALID_ARGUMENT,
         "Vamana invalid resources must reject");
  expect(vamana == nullptr, "invalid Vamana build must clear output");
  ovvsIvfPqIndex_t ivfpq = reinterpret_cast<ovvsIvfPqIndex_t>(uintptr_t{1});
  expect(ovvsIvfPqBuild(res.r, data.data(), 4, dim, static_cast<ovvsMetric>(99), 2,
                        1, 8, &ivfpq) == OVVS_STATUS_INVALID_ARGUMENT,
         "IVF-PQ invalid metric must reject");
  expect(ivfpq == nullptr, "invalid-metric IVF-PQ must clear output");
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
  expect_status(ovvsCagraDetachDataset(cg), "detach before HNSW conversion rejection");
  hx = reinterpret_cast<ovvsHnswIndex_t>(uintptr_t{1});
  expect(ovvsHnswFromCagra(res.r, cg, &hx) == OVVS_STATUS_INVALID_ARGUMENT,
         "HNSW conversion must reject detached CAGRA dataset");
  expect(hx == nullptr, "failed HNSW conversion must clear output");
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
  const int64_t n = 24, dim = 8, k = 3;
  auto data = make_data(n, dim, 440);
  auto q = make_data(2, dim, 441);
  ovvsCagraIndex_t ix = nullptr;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  expect_status(ovvsCagraBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 6, 12, &ix), "b");
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU);
  std::vector<int64_t> nb(static_cast<size_t>(2 * k));
  std::vector<float> ds(static_cast<size_t>(2 * k));
  std::vector<uint8_t> bits((n + 7) / 8, 0);
  for (int64_t id = 0; id < n; id += 2) bits[static_cast<size_t>(id >> 3)] |= 1u << (id & 7);
  const ovvsStatus search_status =
      ovvsCagraSearch(res.r, ix, q.data(), 2, k, 12, 2, bits.data(), nb.data(), ds.data());
  if (!gpu || !ovvsSyclEnabled()) {
    expect(search_status == OVVS_STATUS_DEVICE_UNAVAILABLE, "absent GPU walk must be unavailable");
    ovvsCagraDestroy(ix);
    return;
  }
  expect_status(search_status, "s");
  ovvsDevice last = OVVS_DEVICE_CPU;
  ovvsResourcesLastDevice(res.r, &last);
  expect(last == OVVS_DEVICE_GPU, "GPU walk last_device");
  for (size_t i = 0; i < nb.size(); ++i) {
    expect(nb[i] >= 0 && nb[i] < n, "GPU walk ID range");
    expect((bits[static_cast<size_t>(nb[i] >> 3)] & (1u << (nb[i] & 7))) != 0,
           "GPU walk bitset filter");
    expect(std::isfinite(ds[i]), "GPU walk finite distance");
  }
  ovvsCagraDestroy(ix);
}

OVVS_TEST(cagra_force_gpu_l2_sqrt_supported) {
  Res res;
  int32_t gpu = 0;
  expect_status(ovvsResourcesGpuAvailable(res.r, &gpu), "GPU availability");
  const int64_t n = 24, dim = 8, nq = 2, k = 3;
  auto data = make_data(n, dim, 446);
  auto q = make_data(nq, dim, 447);
  ovvsCagraIndex_t ix = nullptr;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  expect_status(
      ovvsCagraBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_SQRT_EXPANDED, 6, 12, &ix),
      "L2 sqrt build");
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU);
  std::vector<int64_t> nb(static_cast<size_t>(nq * k));
  std::vector<float> ds(static_cast<size_t>(nq * k));
  const ovvsStatus status =
      ovvsCagraSearch(res.r, ix, q.data(), nq, k, 12, 2, nullptr, nb.data(), ds.data());
  if (!gpu || !ovvsSyclEnabled()) {
    expect(status == OVVS_STATUS_DEVICE_UNAVAILABLE, "absent GPU walk must be unavailable");
  } else {
    expect_status(status, "L2 sqrt FORCE_GPU search");
    ovvsDevice last = OVVS_DEVICE_CPU;
    expect_status(ovvsResourcesLastDevice(res.r, &last), "L2 sqrt last device");
    expect(last == OVVS_DEVICE_GPU, "L2 sqrt walk last_device");
    expect(std::all_of(ds.begin(), ds.end(), [](float d) { return std::isfinite(d) && d >= 0.f; }),
           "L2 sqrt walk finite non-negative distances");
  }
  ovvsCagraDestroy(ix);
}

OVVS_TEST(cagra_force_gpu_cosine_near_zero_matches_oracle) {
  Res res;
  int32_t gpu = 0;
  expect_status(ovvsResourcesGpuAvailable(res.r, &gpu), "GPU availability");
  if (!gpu || !ovvsSyclEnabled()) return;

  const int64_t n = 4, dim = 2, k = 4;
  const float data[] = {1.f, 0.f, 0.f, 1.f, 1e-8f, 0.f, -1.f, 0.f};
  const float query[] = {1e-8f, 0.f};
  ovvsCagraIndex_t ix = nullptr;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  expect_status(ovvsCagraBuild(res.r, data, n, dim, OVVS_METRIC_COSINE_EXPANDED, 3, 3, &ix),
                "cosine build");

  int64_t got[k] = {};
  float distances[k] = {};
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU);
  expect_status(ovvsCagraSearch(res.r, ix, query, 1, k, 4, 1, nullptr, got, distances),
                "cosine FORCE_GPU search");

  int64_t truth[k] = {};
  float scores[k] = {};
  rank_oracle(data, n, dim, query, k, truth, scores, false);
  for (int64_t i = 0; i < k; ++i) {
    expect(got[i] == truth[i], "near-zero cosine neighbor order");
    expect(std::fabs(distances[i] - (1.f - scores[i])) < 1e-5f,
           "near-zero cosine distance matches CPU contract");
  }
  ovvsCagraDestroy(ix);
}

OVVS_TEST(cagra_forced_policy_does_not_host_fallback) {
  Res res;
  const int64_t n = 24, dim = 8, nq = 2, k = 3;
  auto data = make_data(n, dim, 444);
  auto q = make_data(nq, dim, 445);
  ovvsCagraIndex_t ix = nullptr;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  expect_status(ovvsCagraBuild(res.r, data.data(), n, dim, OVVS_METRIC_LP_UNEXPANDED, 6, 12, &ix),
                "Lp build");

  std::vector<int64_t> nb(static_cast<size_t>(nq * k), -77);
  std::vector<float> ds(static_cast<size_t>(nq * k), -123.f);
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_AUTO);
  expect_status(ovvsCagraSearch(res.r, ix, q.data(), nq, k, 12, 2, nullptr, nb.data(), ds.data()),
                "AUTO host fallback");
  ovvsDevice last = OVVS_DEVICE_GPU;
  expect_status(ovvsResourcesLastDevice(res.r, &last), "AUTO last");
  expect(last == OVVS_DEVICE_CPU, "AUTO unsupported GPU metric must attribute CPU fallback");
  std::vector<int64_t> truth(static_cast<size_t>(nq * k));
  std::vector<float> truth_dist(static_cast<size_t>(nq * k));
  brute_oracle(data.data(), n, dim, q.data(), nq, k, truth.data(), truth_dist.data());
  expect(recall_at_k(nb.data(), truth.data(), nq, k) > 0.99f, "AUTO host fallback recall");

  std::fill(nb.begin(), nb.end(), -77);
  std::fill(ds.begin(), ds.end(), -123.f);
  int32_t npu_fallbacks_before = 0;
  expect_status(ovvsResourcesNpuFallbacks(res.r, &npu_fallbacks_before),
                "fallback count before FORCE_GPU");
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU);
  const ovvsStatus gpu_status =
      ovvsCagraSearch(res.r, ix, q.data(), nq, k, 12, 2, nullptr, nb.data(), ds.data());
  expect(gpu_status == OVVS_STATUS_DEVICE_UNAVAILABLE, "FORCE_GPU must not host-fallback");
  expect(std::all_of(nb.begin(), nb.end(), [](int64_t id) { return id == -77; }),
         "FORCE_GPU rejection must not write IDs");
  expect(std::all_of(ds.begin(), ds.end(), [](float d) { return d == -123.f; }),
         "FORCE_GPU rejection must not write distances");
  expect_status(ovvsResourcesLastDevice(res.r, &last), "FORCE_GPU last");
  expect(last == OVVS_DEVICE_CPU, "FORCE_GPU rejection must not claim a device");
  int32_t npu_fallbacks_after_gpu = 0;
  expect_status(ovvsResourcesNpuFallbacks(res.r, &npu_fallbacks_after_gpu),
                "fallback count after FORCE_GPU");
  expect(npu_fallbacks_after_gpu == npu_fallbacks_before,
         "FORCE_GPU rejection must not increment NPU fallback telemetry");

  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_NPU);
  const ovvsStatus npu_status =
      ovvsCagraSearch(res.r, ix, q.data(), nq, k, 12, 2, nullptr, nb.data(), ds.data());
  expect(npu_status == OVVS_STATUS_DEVICE_UNAVAILABLE, "FORCE_NPU must not host-fallback");
  expect(std::all_of(nb.begin(), nb.end(), [](int64_t id) { return id == -77; }),
         "FORCE_NPU rejection must not write IDs");
  expect(std::all_of(ds.begin(), ds.end(), [](float d) { return d == -123.f; }),
         "FORCE_NPU rejection must not write distances");
  expect_status(ovvsResourcesLastDevice(res.r, &last), "FORCE_NPU last");
  expect(last == OVVS_DEVICE_CPU, "FORCE_NPU rejection must not claim a device");
  int32_t npu_fallbacks_after_npu = 0;
  expect_status(ovvsResourcesNpuFallbacks(res.r, &npu_fallbacks_after_npu),
                "fallback count after FORCE_NPU");
  expect(npu_fallbacks_after_npu == npu_fallbacks_before + 1,
         "FORCE_NPU rejection must increment NPU fallback telemetry once");
  ovvsCagraDestroy(ix);
}

OVVS_TEST(cagra_q_forced_policy_does_not_host_fallback) {
  Res res;
  const int64_t n = 32, dim = 8, nq = 2, k = 3;
  auto data = make_data(n, dim, 448);
  auto queries = make_data(nq, dim, 449);
  ovvsCagraIndex_t ix = nullptr;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  expect_status(ovvsCagraBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 8, 16, &ix),
                "CAGRA-Q build");
  expect_status(ovvsCagraQuantize(res.r, ix, 4, 8), "CAGRA-Q quantize");

  std::vector<int64_t> neighbors(static_cast<size_t>(nq * k), -77);
  std::vector<float> distances(static_cast<size_t>(nq * k), -123.f);
  expect_status(ovvsCagraSearch(res.r, ix, queries.data(), nq, k, 16, 2, nullptr,
                                neighbors.data(), distances.data()),
                "CAGRA-Q FORCE_CPU search");

  ovvsDevice preserved_device = OVVS_DEVICE_CPU;
  int32_t gpu_available = 0;
  expect_status(ovvsResourcesGpuAvailable(res.r, &gpu_available), "CAGRA-Q GPU availability");
  if (gpu_available) {
    auto a = make_data(8, 8, 450);
    auto b = make_data(8, 8, 451);
    std::vector<float> c(64);
    ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU);
    expect_status(ovvsGemm(res.r, a.data(), b.data(), c.data(), 8, 8, 8, 1),
                  "stamp prior GPU attribution");
    expect_status(ovvsResourcesLastDevice(res.r, &preserved_device), "prior GPU attribution");
    expect(preserved_device == OVVS_DEVICE_GPU, "GPU stamp must use GPU");
  }

  int32_t quantize_fallbacks = 0;
  expect_status(ovvsResourcesNpuFallbacks(res.r, &quantize_fallbacks),
                "CAGRA-Q quantize fallback baseline");
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU);
  expect(ovvsCagraQuantize(res.r, ix, 4, 8) == OVVS_STATUS_DEVICE_UNAVAILABLE,
         "CAGRA-Q FORCE_GPU quantize must reject host stages");
  int32_t after_gpu_quantize = 0;
  expect_status(ovvsResourcesNpuFallbacks(res.r, &after_gpu_quantize),
                "CAGRA-Q quantize count after FORCE_GPU");
  expect(after_gpu_quantize == quantize_fallbacks,
         "FORCE_GPU quantize must not increment NPU fallback count");
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_NPU);
  expect(ovvsCagraQuantize(res.r, ix, 4, 8) == OVVS_STATUS_DEVICE_UNAVAILABLE,
         "CAGRA-Q FORCE_NPU quantize must reject host stages");
  int32_t after_npu_quantize = 0;
  expect_status(ovvsResourcesNpuFallbacks(res.r, &after_npu_quantize),
                "CAGRA-Q quantize count after FORCE_NPU");
  expect(after_npu_quantize == quantize_fallbacks + 1,
         "FORCE_NPU quantize must increment fallback once");
  ovvsDevice after_quantize_device = OVVS_DEVICE_AUTO;
  expect_status(ovvsResourcesLastDevice(res.r, &after_quantize_device),
                "CAGRA-Q quantize forced failure last device");
  expect(after_quantize_device == preserved_device,
         "failed CAGRA-Q quantize must preserve prior attribution");

  int32_t fallback_count = 0;
  expect_status(ovvsResourcesNpuFallbacks(res.r, &fallback_count), "CAGRA-Q fallback count");
  std::fill(neighbors.begin(), neighbors.end(), -77);
  std::fill(distances.begin(), distances.end(), -123.f);
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU);
  expect(ovvsCagraSearch(res.r, ix, queries.data(), nq, k, 16, 2, nullptr, neighbors.data(),
                         distances.data()) == OVVS_STATUS_DEVICE_UNAVAILABLE,
         "CAGRA-Q FORCE_GPU must not host-fallback");
  expect(std::all_of(neighbors.begin(), neighbors.end(), [](int64_t id) { return id == -77; }),
         "CAGRA-Q FORCE_GPU rejection must not write IDs");
  expect(std::all_of(distances.begin(), distances.end(), [](float d) { return d == -123.f; }),
         "CAGRA-Q FORCE_GPU rejection must not write distances");
  int32_t after_gpu = 0;
  expect_status(ovvsResourcesNpuFallbacks(res.r, &after_gpu), "CAGRA-Q count after FORCE_GPU");
  expect(after_gpu == fallback_count, "CAGRA-Q FORCE_GPU must not increment NPU fallback count");
  ovvsDevice last = OVVS_DEVICE_AUTO;
  expect_status(ovvsResourcesLastDevice(res.r, &last), "CAGRA-Q FORCE_GPU last device");
  expect(last == preserved_device, "CAGRA-Q FORCE_GPU rejection must preserve prior attribution");

  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_NPU);
  expect(ovvsCagraSearch(res.r, ix, queries.data(), nq, k, 16, 2, nullptr, neighbors.data(),
                         distances.data()) == OVVS_STATUS_DEVICE_UNAVAILABLE,
         "CAGRA-Q FORCE_NPU must not host-fallback");
  expect(std::all_of(neighbors.begin(), neighbors.end(), [](int64_t id) { return id == -77; }),
         "CAGRA-Q FORCE_NPU rejection must not write IDs");
  expect(std::all_of(distances.begin(), distances.end(), [](float d) { return d == -123.f; }),
         "CAGRA-Q FORCE_NPU rejection must not write distances");
  int32_t after_npu = 0;
  expect_status(ovvsResourcesNpuFallbacks(res.r, &after_npu), "CAGRA-Q count after FORCE_NPU");
  expect(after_npu == fallback_count + 1,
         "CAGRA-Q FORCE_NPU must increment NPU fallback count once");
  expect_status(ovvsResourcesLastDevice(res.r, &last), "CAGRA-Q forced failure last device");
  expect(last == preserved_device, "CAGRA-Q FORCE_NPU rejection must preserve prior attribution");

  expect_status(ovvsCagraDetachDataset(ix), "detach CAGRA-Q dataset");
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_AUTO);
  expect_status(ovvsCagraSearch(res.r, ix, queries.data(), nq, k, 16, 2, nullptr,
                                neighbors.data(), distances.data()),
                "detached CAGRA-Q adaptive host search");
  expect_status(ovvsResourcesLastDevice(res.r, &last), "detached CAGRA-Q last device");
  expect(last == OVVS_DEVICE_CPU, "executed CAGRA-Q host walk must report CPU");
  ovvsCagraDestroy(ix);
}

OVVS_TEST(cagra_sycl_walk_n_over_4096) {
  if (!ovvsSyclEnabled()) skip_test("SYCL GPU path unavailable");
  Res res;
  int32_t gpu = 0;
  ovvsResourcesGpuAvailable(res.r, &gpu);
  if (!gpu) skip_test("GPU device unavailable");
  /* n>4096: old kernel skipped expanding id>=4096 and overflowed slm at search_width>64. */
  const int64_t n = 4200, dim = 8, nq = 4, k = 5;
  auto data = make_data(n, dim, 442);
  auto q = make_data(nq, dim, 443);
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_AUTO);
  ovvsCagraIndex_t ix = nullptr;
  expect_status(ovvsCagraBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 8, 16, &ix), "b");
  ovvsDevice last = OVVS_DEVICE_AUTO;
  expect_status(ovvsResourcesLastDevice(res.r, &last), "sycl n>4096 build device");
  expect(last == OVVS_DEVICE_CPU, "mixed CAGRA build must report its final host prune stage");
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU);
  std::vector<int64_t> got(static_cast<size_t>(nq * k)), truth(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k)), td(static_cast<size_t>(nq * k));
  int64_t walks_before = 0, direct_before = 0, uploads_before = 0, bytes_before = 0;
  expect_status(ovvsResourcesCagraTransferStats(res.r, &walks_before, &direct_before,
                                                &uploads_before, &bytes_before),
                "CAGRA transfer stats before searches");
  for (int repeat = 0; repeat < 2; ++repeat) {
    expect_status(ovvsCagraSearch(res.r, ix, q.data(), nq, k, 32, 80, nullptr, got.data(),
                                  gd.data()),
                  "s");
  }
  int64_t walks_after = 0, direct_after = 0, uploads_after = 0, bytes_after = 0;
  expect_status(ovvsResourcesCagraTransferStats(res.r, &walks_after, &direct_after,
                                                &uploads_after, &bytes_after),
                "CAGRA transfer stats after searches");
  expect(walks_after - walks_before == 2, "two GPU searches must record two CAGRA walks");
  expect(direct_after - direct_before == 2,
         "shared-USM dataset and graph must record two direct-index walks");
  expect(uploads_after - uploads_before == 0,
         "shared-USM CAGRA index must not record index uploads");
  expect(bytes_after - bytes_before == 0,
         "shared-USM CAGRA index must not record index-upload bytes");
  ovvsResourcesLastDevice(res.r, &last);
  expect(last == OVVS_DEVICE_GPU, "sycl n>4096 last_device");
  brute_oracle(data.data(), n, dim, q.data(), nq, k, truth.data(), td.data());
  for (int64_t i = 0; i < nq * k; ++i) {
    expect(got[static_cast<size_t>(i)] >= 0 && got[static_cast<size_t>(i)] < n, "id range");
    expect(std::isfinite(gd[static_cast<size_t>(i)]), "finite distance");
  }
  for (int64_t qi = 0; qi < nq; ++qi) {
    std::set<int64_t> unique(got.begin() + qi * k, got.begin() + (qi + 1) * k);
    expect(static_cast<int64_t>(unique.size()) == k, "unique result IDs");
  }
  const float rec = recall_at_k(got.data(), truth.data(), nq, k);
  expect(rec >= 0.2f, "sycl n>4096 recall vs independent L2 " + std::to_string(rec));
  ovvsCagraDestroy(ix);
}

OVVS_TEST(cagra_query_seed_is_batch_invariant) {
  Res res;
  const int64_t n = 640, dim = 8, nq = 6, k = 8;
  constexpr int32_t itopk = 8;
  constexpr int32_t search_width = 1;
  /* Keep the walk non-exhaustive: 512 seeds + at most 48 degree-one expansions
     cover fewer than 640 rows. The legacy call-ordinal seeds changed 4/6 rows. */
  auto data = make_data(n, dim, 446);
  auto queries = make_data(nq, dim, 447);

  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  ovvsCagraIndex_t ix = nullptr;
  expect_status(ovvsCagraBuild(res.r, data.data(), n, dim, OVVS_METRIC_L2_EXPANDED, 1, 1, &ix),
                "batch-invariant build");

  const auto close_distance = [](float a, float b) {
    const float scale = std::max(1.f, std::max(std::fabs(a), std::fabs(b)));
    return std::fabs(a - b) <= 2e-5f * scale;
  };
  const auto run_batch = [&](ovvsPolicy policy, const float* query_data, int64_t query_count,
                             std::vector<int64_t>& ids, std::vector<float>& distances,
                             const char* label) {
    expect_status(ovvsResourcesSetPolicy(res.r, policy), label);
    ids.resize(static_cast<size_t>(query_count * k));
    distances.resize(static_cast<size_t>(query_count * k));
    expect_status(ovvsCagraSearch(res.r, ix, query_data, query_count, k, itopk, search_width,
                                  nullptr, ids.data(), distances.data()),
                  label);
  };
  const auto expect_single_partition = [&](ovvsPolicy policy, const std::vector<int64_t>& batch_ids,
                                           const std::vector<float>& batch_distances,
                                           const char* label) {
    std::vector<int64_t> single_ids(static_cast<size_t>(k));
    std::vector<float> single_distances(static_cast<size_t>(k));
    expect_status(ovvsResourcesSetPolicy(res.r, policy), label);
    for (int64_t qi = 0; qi < nq; ++qi) {
      expect_status(ovvsCagraSearch(res.r, ix, queries.data() + qi * dim, 1, k, itopk,
                                    search_width, nullptr, single_ids.data(),
                                    single_distances.data()),
                    label);
      for (int64_t t = 0; t < k; ++t) {
        const size_t offset = static_cast<size_t>(qi * k + t);
        expect(single_ids[static_cast<size_t>(t)] == batch_ids[offset],
               std::string(label) + " exact ID partition invariance");
        expect(close_distance(single_distances[static_cast<size_t>(t)], batch_distances[offset]),
               std::string(label) + " distance partition invariance");
      }
    }
  };

  std::vector<int64_t> cpu_ids;
  std::vector<float> cpu_distances;
  run_batch(OVVS_POLICY_FORCE_CPU, queries.data(), nq, cpu_ids, cpu_distances, "CPU batch search");
  expect_single_partition(OVVS_POLICY_FORCE_CPU, cpu_ids, cpu_distances, "CPU single search");

  std::vector<float> reversed_queries(static_cast<size_t>(nq * dim));
  for (int64_t qi = 0; qi < nq; ++qi) {
    std::copy_n(queries.data() + (nq - 1 - qi) * dim, dim, reversed_queries.data() + qi * dim);
  }
  std::vector<int64_t> reversed_ids;
  std::vector<float> reversed_distances;
  run_batch(OVVS_POLICY_FORCE_CPU, reversed_queries.data(), nq, reversed_ids, reversed_distances,
            "CPU reordered search");
  for (int64_t qi = 0; qi < nq; ++qi) {
    const int64_t original_qi = nq - 1 - qi;
    for (int64_t t = 0; t < k; ++t) {
      const size_t reordered_offset = static_cast<size_t>(qi * k + t);
      const size_t original_offset = static_cast<size_t>(original_qi * k + t);
      expect(reversed_ids[reordered_offset] == cpu_ids[original_offset],
             "CPU exact ID reorder invariance");
      expect(close_distance(reversed_distances[reordered_offset], cpu_distances[original_offset]),
             "CPU distance reorder invariance");
    }
  }

  int32_t gpu = 0;
  expect_status(ovvsResourcesGpuAvailable(res.r, &gpu), "GPU availability");
  if (gpu && ovvsSyclEnabled()) {
    std::vector<int64_t> gpu_ids;
    std::vector<float> gpu_distances;
    run_batch(OVVS_POLICY_FORCE_GPU, queries.data(), nq, gpu_ids, gpu_distances, "GPU batch search");
    expect_single_partition(OVVS_POLICY_FORCE_GPU, gpu_ids, gpu_distances, "GPU single search");
    for (size_t i = 0; i < cpu_ids.size(); ++i) {
      expect(gpu_ids[i] == cpu_ids[i], "CPU/GPU exact ID parity for deterministic seeds");
      expect(close_distance(gpu_distances[i], cpu_distances[i]),
             "CPU/GPU distance parity for deterministic seeds");
    }

    run_batch(OVVS_POLICY_FORCE_GPU, reversed_queries.data(), nq, reversed_ids, reversed_distances,
              "GPU reordered search");
    for (int64_t qi = 0; qi < nq; ++qi) {
      const int64_t original_qi = nq - 1 - qi;
      for (int64_t t = 0; t < k; ++t) {
        const size_t reordered_offset = static_cast<size_t>(qi * k + t);
        const size_t original_offset = static_cast<size_t>(original_qi * k + t);
        expect(reversed_ids[reordered_offset] == gpu_ids[original_offset],
               "GPU exact ID reorder invariance");
        expect(close_distance(reversed_distances[reordered_offset], gpu_distances[original_offset]),
               "GPU distance reorder invariance");
      }
    }
  }

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
