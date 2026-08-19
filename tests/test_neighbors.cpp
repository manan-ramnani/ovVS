#include "test_harness.hpp"

#include <algorithm>
#include <filesystem>
#include <numeric>
#include <set>
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

/* Independent of libiovs search: rank dataset rows by a caller-supplied score (higher is better). */
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

IOVS_TEST(brute_force_recall_one) {
  Res res;
  const int64_t n = 40, dim = 8, nq = 5, k = 4;
  auto data = make_data(n, dim, 100);
  auto q = make_data(nq, dim, 101);
  iovsBruteForceIndex_t ix = nullptr;
  expect_status(iovsBruteForceBuild(res.r, data.data(), n, dim, IOVS_METRIC_L2_EXPANDED, &ix), "bf");
  std::vector<int64_t> got(static_cast<size_t>(nq * k)), truth(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k)), td(static_cast<size_t>(nq * k));
  expect_status(iovsBruteForceSearch(res.r, ix, q.data(), nq, k, nullptr, got.data(), gd.data()),
                "bfs");
  brute_oracle(data.data(), n, dim, q.data(), nq, k, truth.data(), td.data());
  expect(recall_at_k(got.data(), truth.data(), nq, k) > 0.999f, "bf recall");
  iovsBruteForceDestroy(ix);
}

IOVS_TEST(brute_force_inner_product_vs_dot_oracle) {
  Res res;
  iovsResourcesSetPolicy(res.r, IOVS_POLICY_FORCE_CPU);
  const float data[] = {1.f, 0.f, 0.f, 1.f, -1.f, 0.f};
  const float q[] = {1.f, 0.f};
  iovsBruteForceIndex_t ix = nullptr;
  expect_status(iovsBruteForceBuild(res.r, data, 3, 2, IOVS_METRIC_INNER_PRODUCT, &ix), "ip build");
  int64_t nb[2];
  float ds[2];
  expect_status(iovsBruteForceSearch(res.r, ix, q, 1, 2, nullptr, nb, ds), "ip search");
  int64_t truth[2];
  float tscore[2];
  rank_oracle(data, 3, 2, q, 2, truth, tscore, true);
  expect(nb[0] == truth[0] && nb[1] == truth[1], "ip neighbor ids");
  expect(std::fabs(ds[0] - tscore[0]) < 1e-5f && std::fabs(ds[1] - tscore[1]) < 1e-5f, "ip values");
  expect(nb[0] == 0 && nb[1] == 1, "ip order vs [1,0],[0,1],[-1,0]");
  iovsBruteForceDestroy(ix);
}

IOVS_TEST(brute_force_cosine_vs_oracle) {
  Res res;
  iovsResourcesSetPolicy(res.r, IOVS_POLICY_FORCE_CPU);
  const float data[] = {1.f, 0.f, 0.f, 1.f, -1.f, 0.f};
  const float q[] = {1.f, 0.f};
  iovsBruteForceIndex_t ix = nullptr;
  expect_status(iovsBruteForceBuild(res.r, data, 3, 2, IOVS_METRIC_COSINE_EXPANDED, &ix), "cos build");
  int64_t nb[2];
  float ds[2];
  expect_status(iovsBruteForceSearch(res.r, ix, q, 1, 2, nullptr, nb, ds), "cos search");
  int64_t truth[2];
  float tscore[2];
  rank_oracle(data, 3, 2, q, 2, truth, tscore, false);
  expect(nb[0] == truth[0] && nb[1] == truth[1], "cosine neighbor ids");
  expect(nb[0] == 0 && nb[1] == 1, "cosine order");
  /* API reports 1-cos distance. */
  expect(std::fabs(ds[0] - (1.f - tscore[0])) < 1e-5f, "cosine dist 0");
  expect(std::fabs(ds[1] - (1.f - tscore[1])) < 1e-5f, "cosine dist 1");
  iovsBruteForceDestroy(ix);
}

IOVS_TEST(ivf_flat_inner_product_refine_vs_oracle) {
  Res res;
  iovsResourcesSetPolicy(res.r, IOVS_POLICY_FORCE_CPU);
  const float data[] = {1.f, 0.f, 0.f, 1.f, -1.f, 0.f, 0.5f, 0.5f};
  const float q[] = {1.f, 0.f};
  iovsIvfFlatIndex_t ix = nullptr;
  expect_status(iovsIvfFlatBuild(res.r, data, 4, 2, IOVS_METRIC_INNER_PRODUCT, 1, &ix), "ivf ip");
  int64_t nb[2];
  float ds[2];
  expect_status(iovsIvfFlatSearch(res.r, ix, q, 1, 2, 1, nullptr, nb, ds), "ivf ip search");
  int64_t truth[2];
  float tscore[2];
  rank_oracle(data, 4, 2, q, 2, truth, tscore, true);
  expect(nb[0] == truth[0] && nb[1] == truth[1], "ivf ip ids");
  expect(std::fabs(ds[0] - tscore[0]) < 1e-4f, "ivf ip value");
  iovsIvfFlatDestroy(ix);
}

IOVS_TEST(ivf_pq_cosine_refine_vs_oracle) {
  Res res;
  iovsResourcesSetPolicy(res.r, IOVS_POLICY_FORCE_CPU);
  const int64_t n = 12, dim = 4, k = 3;
  auto data = make_data(n, dim, 222);
  const float q[] = {0.4f, -0.1f, 0.2f, 0.8f};
  iovsIvfPqIndex_t ix = nullptr;
  expect_status(iovsIvfPqBuild(res.r, data.data(), n, dim, IOVS_METRIC_COSINE_EXPANDED, 1, 2, 8, &ix),
                "ivfpq cos");
  std::vector<int64_t> nb(static_cast<size_t>(k));
  std::vector<float> ds(static_cast<size_t>(k));
  /* nprobe=1, krefine large: refine is exact over the whole list. */
  expect_status(iovsIvfPqSearch(res.r, ix, q, 1, k, 1, static_cast<int32_t>(n), nullptr, nb.data(),
                                ds.data()),
                "ivfpq cos search");
  int64_t truth[3];
  float tscore[3];
  rank_oracle(data.data(), n, dim, q, k, truth, tscore, false);
  for (int64_t t = 0; t < k; ++t) expect(nb[static_cast<size_t>(t)] == truth[t], "ivfpq cosine id");
  iovsIvfPqDestroy(ix);
}

IOVS_TEST(brute_force_bitset_filter) {
  Res res;
  const int64_t n = 20, dim = 4, k = 3;
  auto data = make_data(n, dim, 110);
  auto q = make_data(1, dim, 111);
  std::vector<uint8_t> bits((n + 7) / 8, 0);
  for (int64_t i = 0; i < n; ++i) {
    if (i % 2 == 0) bits[static_cast<size_t>(i >> 3)] |= static_cast<uint8_t>(1u << (i & 7));
  }
  iovsBruteForceIndex_t ix = nullptr;
  expect_status(iovsBruteForceBuild(res.r, data.data(), n, dim, IOVS_METRIC_L2_EXPANDED, &ix), "bf");
  std::vector<int64_t> nb(static_cast<size_t>(k));
  std::vector<float> ds(static_cast<size_t>(k));
  expect_status(iovsBruteForceSearch(res.r, ix, q.data(), 1, k, bits.data(), nb.data(), ds.data()),
                "bfs");
  for (int64_t t = 0; t < k; ++t) expect((nb[static_cast<size_t>(t)] % 2) == 0, "filter even");
  iovsBruteForceDestroy(ix);
}

IOVS_TEST(ivf_flat_recall) {
  Res res;
  const int64_t n = 80, dim = 8, nq = 6, k = 5;
  auto data = make_data(n, dim, 200);
  auto q = make_data(nq, dim, 201);
  iovsIvfFlatIndex_t ix = nullptr;
  expect_status(iovsIvfFlatBuild(res.r, data.data(), n, dim, IOVS_METRIC_L2_EXPANDED, 8, &ix), "ivf");
  std::vector<int64_t> got(static_cast<size_t>(nq * k)), truth(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k)), td(static_cast<size_t>(nq * k));
  expect_status(iovsIvfFlatSearch(res.r, ix, q.data(), nq, k, 8, nullptr, got.data(), gd.data()),
                "ivfs");
  brute_oracle(data.data(), n, dim, q.data(), nq, k, truth.data(), td.data());
  const float rec = recall_at_k(got.data(), truth.data(), nq, k);
  expect(rec >= 0.85f, "ivf-flat recall " + std::to_string(rec));
  iovsIvfFlatDestroy(ix);
}

IOVS_TEST(ivf_pq_recall) {
  Res res;
  const int64_t n = 64, dim = 8, nq = 5, k = 4;
  auto data = make_data(n, dim, 210);
  auto q = make_data(nq, dim, 211);
  iovsIvfPqIndex_t ix = nullptr;
  expect_status(iovsIvfPqBuild(res.r, data.data(), n, dim, IOVS_METRIC_L2_EXPANDED, 8, 4, 8, &ix),
                "ivfpq");
  std::vector<int64_t> got(static_cast<size_t>(nq * k)), truth(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k)), td(static_cast<size_t>(nq * k));
  expect_status(iovsIvfPqSearch(res.r, ix, q.data(), nq, k, 8, 16, nullptr, got.data(), gd.data()),
                "ivfpqs");
  brute_oracle(data.data(), n, dim, q.data(), nq, k, truth.data(), td.data());
  const float rec = recall_at_k(got.data(), truth.data(), nq, k);
  expect(rec >= 0.5f, "ivf-pq recall " + std::to_string(rec));
  iovsIvfPqDestroy(ix);
}

IOVS_TEST(ivf_rabitq_recall) {
  Res res;
  const int64_t n = 64, dim = 8, nq = 5, k = 4;
  auto data = make_data(n, dim, 220);
  auto q = make_data(nq, dim, 221);
  iovsIvfRabitqIndex_t ix = nullptr;
  expect_status(iovsIvfRabitqBuild(res.r, data.data(), n, dim, IOVS_METRIC_L2_EXPANDED, 8, &ix),
                "rabitq");
  std::vector<int64_t> got(static_cast<size_t>(nq * k)), truth(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k)), td(static_cast<size_t>(nq * k));
  expect_status(
      iovsIvfRabitqSearch(res.r, ix, q.data(), nq, k, 8, 16, nullptr, got.data(), gd.data()), "rqs");
  brute_oracle(data.data(), n, dim, q.data(), nq, k, truth.data(), td.data());
  const float rec = recall_at_k(got.data(), truth.data(), nq, k);
  expect(rec >= 0.45f, "rabitq recall " + std::to_string(rec));
  iovsIvfRabitqDestroy(ix);
}

IOVS_TEST(nndescent_graph_overlap) {
  Res res;
  const int64_t n = 30, dim = 6;
  const int32_t deg = 5;
  auto data = make_data(n, dim, 300);
  iovsNnDescentGraph_t g = nullptr;
  expect_status(iovsNnDescentBuild(res.r, data.data(), n, dim, IOVS_METRIC_L2_EXPANDED, deg, 4, &g),
                "nnd");
  const int32_t* ids = nullptr;
  int64_t nn = 0;
  int32_t d = 0;
  expect_status(iovsNnDescentNeighbors(g, &ids, &nn, &d), "nndn");
  expect(nn == n && d == deg, "nnd shape");
  std::vector<int64_t> truth(static_cast<size_t>(n * deg));
  std::vector<float> td(static_cast<size_t>(n * deg));
  expect_status(iovsAllNeighbors(res.r, data.data(), n, dim, IOVS_METRIC_L2_EXPANDED, deg, truth.data(),
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
  iovsNnDescentDestroy(g);
}

IOVS_TEST(cagra_build_search_filter_serialize) {
  Res res;
  const int64_t n = 36, dim = 8, nq = 4, k = 4;
  auto data = make_data(n, dim, 400);
  auto q = make_data(nq, dim, 401);
  iovsCagraIndex_t ix = nullptr;
  expect_status(iovsCagraBuild(res.r, data.data(), n, dim, IOVS_METRIC_L2_EXPANDED, 8, 16, &ix),
                "cagra");
  std::vector<int64_t> got(static_cast<size_t>(nq * k)), truth(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k)), td(static_cast<size_t>(nq * k));
  expect_status(iovsCagraSearch(res.r, ix, q.data(), nq, k, 16, 2, nullptr, got.data(), gd.data()),
                "cags");
  brute_oracle(data.data(), n, dim, q.data(), nq, k, truth.data(), td.data());
  const float rec = recall_at_k(got.data(), truth.data(), nq, k);
  expect(rec >= 0.7f, "cagra recall " + std::to_string(rec));

  std::vector<uint8_t> bits((n + 7) / 8, 0xff);
  bits[0] = 0xfe; /* drop id 0 */
  expect_status(iovsCagraSearch(res.r, ix, q.data(), nq, k, 16, 2, bits.data(), got.data(), gd.data()),
                "cagf");
  for (int64_t i = 0; i < nq * k; ++i) expect(got[static_cast<size_t>(i)] != 0, "cagra filter");

  const auto path = std::filesystem::temp_directory_path() / "iovs_cagra_test.bin";
  expect_status(iovsCagraSerialize(ix, path.string().c_str()), "ser");
  iovsCagraIndex_t loaded = nullptr;
  expect_status(iovsCagraDeserialize(res.r, path.string().c_str(), &loaded), "des");
  std::vector<int64_t> g2(static_cast<size_t>(nq * k));
  std::vector<float> d2(static_cast<size_t>(nq * k));
  expect_status(iovsCagraSearch(res.r, loaded, q.data(), nq, k, 16, 2, nullptr, g2.data(), d2.data()),
                "cags2");
  expect(recall_at_k(g2.data(), truth.data(), nq, k) >= 0.7f, "deser recall");
  auto extra = make_data(4, dim, 409);
  expect_status(iovsCagraExtend(res.r, loaded, extra.data(), 4), "ext");
  iovsCagraDestroy(loaded);
  iovsCagraDestroy(ix);
}

IOVS_TEST(hnsw_from_cagra) {
  Res res;
  const int64_t n = 28, dim = 6, nq = 3, k = 3;
  auto data = make_data(n, dim, 410);
  auto q = make_data(nq, dim, 411);
  iovsCagraIndex_t cg = nullptr;
  expect_status(iovsCagraBuild(res.r, data.data(), n, dim, IOVS_METRIC_L2_EXPANDED, 6, 12, &cg),
                "cagra");
  iovsHnswIndex_t hx = nullptr;
  expect_status(iovsHnswFromCagra(res.r, cg, &hx), "hnsw");
  std::vector<int64_t> got(static_cast<size_t>(nq * k)), truth(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k)), td(static_cast<size_t>(nq * k));
  expect_status(iovsHnswSearch(res.r, hx, q.data(), nq, k, 16, got.data(), gd.data()), "hnsws");
  brute_oracle(data.data(), n, dim, q.data(), nq, k, truth.data(), td.data());
  expect(recall_at_k(got.data(), truth.data(), nq, k) >= 0.6f, "hnsw recall");
  iovsHnswDestroy(hx);
  iovsCagraDestroy(cg);
}

IOVS_TEST(vamana_and_scann) {
  Res res;
  const int64_t n = 40, dim = 8, nq = 4, k = 4;
  auto data = make_data(n, dim, 500);
  auto q = make_data(nq, dim, 501);
  iovsVamanaIndex_t vx = nullptr;
  expect_status(iovsVamanaBuild(res.r, data.data(), n, dim, IOVS_METRIC_L2_EXPANDED, 8, 1.2f, &vx),
                "vam");
  std::vector<int64_t> got(static_cast<size_t>(nq * k)), truth(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k)), td(static_cast<size_t>(nq * k));
  expect_status(iovsVamanaSearch(res.r, vx, q.data(), nq, k, 16, nullptr, got.data(), gd.data()),
                "vams");
  brute_oracle(data.data(), n, dim, q.data(), nq, k, truth.data(), td.data());
  expect(recall_at_k(got.data(), truth.data(), nq, k) >= 0.55f, "vamana recall");
  iovsVamanaDestroy(vx);

  iovsScannIndex_t sx = nullptr;
  expect_status(iovsScannBuild(res.r, data.data(), n, dim, IOVS_METRIC_L2_EXPANDED, 8, 4, &sx),
                "scann");
  expect_status(iovsScannSearch(res.r, sx, q.data(), nq, k, 8, 12, got.data(), gd.data()), "scanns");
  expect(recall_at_k(got.data(), truth.data(), nq, k) >= 0.35f, "scann recall");
  iovsScannDestroy(sx);
}
