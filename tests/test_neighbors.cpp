#include "test_harness.hpp"

#include <algorithm>
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

IOVS_TEST(cagra_graph_build_params_and_q) {
  Res res;
  iovsResourcesSetPolicy(res.r, IOVS_POLICY_FORCE_CPU);
  const int64_t n = 32, dim = 8, nq = 4, k = 4;
  auto data = make_data(n, dim, 420);
  auto q = make_data(nq, dim, 421);
  std::vector<int64_t> truth(static_cast<size_t>(nq * k));
  std::vector<float> td(static_cast<size_t>(nq * k));
  brute_oracle(data.data(), n, dim, q.data(), nq, k, truth.data(), td.data());
  const iovsCagraBuildAlgo algos[] = {IOVS_CAGRA_BUILD_NN_DESCENT, IOVS_CAGRA_BUILD_IVF_PQ,
                                      IOVS_CAGRA_BUILD_ITERATIVE};
  const char* names[] = {"nnd", "ivfpq", "iter"};
  for (int a = 0; a < 3; ++a) {
    iovsCagraIndex_t ix = nullptr;
    expect_status(iovsCagraBuildEx(res.r, data.data(), n, dim, IOVS_METRIC_L2_EXPANDED, 8, 16, algos[a],
                                   &ix),
                  names[a]);
    std::vector<int64_t> got(static_cast<size_t>(nq * k));
    std::vector<float> gd(static_cast<size_t>(nq * k));
    expect_status(iovsCagraSearch(res.r, ix, q.data(), nq, k, 16, 2, nullptr, got.data(), gd.data()),
                  names[a]);
    const float rec = recall_at_k(got.data(), truth.data(), nq, k);
    expect(rec >= 0.45f, std::string(names[a]) + " recall " + std::to_string(rec));
    iovsCagraDestroy(ix);
  }
  iovsCagraIndex_t qx = nullptr;
  expect_status(iovsCagraBuild(res.r, data.data(), n, dim, IOVS_METRIC_L2_EXPANDED, 8, 16, &qx), "qbuild");
  expect_status(iovsCagraQuantize(res.r, qx, 4, 8), "q");
  std::vector<int64_t> got(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k));
  expect_status(iovsCagraSearch(res.r, qx, q.data(), nq, k, 16, 2, nullptr, got.data(), gd.data()), "qs");
  const float qrec = recall_at_k(got.data(), truth.data(), nq, k);
  expect(qrec >= 0.35f, "cagra-q recall " + std::to_string(qrec));
  const auto path = std::filesystem::temp_directory_path() / "iovs_cagra_detach.bin";
  expect_status(iovsCagraSerializeEx(qx, path.string().c_str(), 0), "ser0");
  expect_status(iovsCagraDetachDataset(qx), "det");
  expect_status(iovsCagraSearch(res.r, qx, q.data(), nq, k, 16, 2, nullptr, got.data(), gd.data()),
                "q-detached");
  expect_status(iovsCagraAttachDataset(qx, data.data(), n, dim), "att");
  iovsCagraIndex_t loaded = nullptr;
  expect_status(iovsCagraDeserialize(res.r, path.string().c_str(), &loaded), "des0");
  expect_status(iovsCagraAttachDataset(loaded, data.data(), n, dim), "att2");
  expect_status(iovsCagraSearch(res.r, loaded, q.data(), nq, k, 16, 2, nullptr, got.data(), gd.data()),
                "att-search");
  iovsCagraDestroy(loaded);
  iovsCagraDestroy(qx);
}

IOVS_TEST(cagra_extend_recall_hold) {
  Res res;
  iovsResourcesSetPolicy(res.r, IOVS_POLICY_FORCE_CPU);
  const int64_t n = 40, extra = 6, dim = 8, nq = 4, k = 4;
  auto all = make_data(n + extra, dim, 430);
  auto q = make_data(nq, dim, 431);
  std::vector<int64_t> truth(static_cast<size_t>(nq * k));
  std::vector<float> td(static_cast<size_t>(nq * k));
  brute_oracle(all.data(), n + extra, dim, q.data(), nq, k, truth.data(), td.data());
  iovsCagraIndex_t base = nullptr;
  expect_status(iovsCagraBuild(res.r, all.data(), n, dim, IOVS_METRIC_L2_EXPANDED, 8, 16, &base), "b");
  expect_status(iovsCagraExtend(res.r, base, all.data() + n * dim, extra), "ext");
  std::vector<int64_t> got(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k));
  expect_status(iovsCagraSearch(res.r, base, q.data(), nq, k, 20, 2, nullptr, got.data(), gd.data()),
                "es");
  const float rec_ext = recall_at_k(got.data(), truth.data(), nq, k);
  iovsCagraIndex_t rebuilt = nullptr;
  expect_status(iovsCagraBuild(res.r, all.data(), n + extra, dim, IOVS_METRIC_L2_EXPANDED, 8, 16, &rebuilt),
                "rb");
  std::vector<int64_t> gr(static_cast<size_t>(nq * k));
  expect_status(iovsCagraSearch(res.r, rebuilt, q.data(), nq, k, 20, 2, nullptr, gr.data(), gd.data()),
                "rs");
  const float rec_rb = recall_at_k(gr.data(), truth.data(), nq, k);
  expect(rec_ext >= 0.45f, "extend recall " + std::to_string(rec_ext));
  expect(rec_ext + 0.2f >= rec_rb, "extend holds vs rebuild");
  iovsCagraDestroy(base);
  iovsCagraDestroy(rebuilt);
}

IOVS_TEST(cagra_force_gpu_last_device) {
  Res res;
  int32_t gpu = 0;
  iovsResourcesGpuAvailable(res.r, &gpu);
  if (!gpu) return;
  iovsResourcesSetPolicy(res.r, IOVS_POLICY_FORCE_GPU);
  const int64_t n = 24, dim = 8, k = 3;
  auto data = make_data(n, dim, 440);
  auto q = make_data(2, dim, 441);
  iovsCagraIndex_t ix = nullptr;
  expect_status(iovsCagraBuild(res.r, data.data(), n, dim, IOVS_METRIC_L2_EXPANDED, 6, 12, &ix), "b");
  std::vector<int64_t> nb(static_cast<size_t>(2 * k));
  std::vector<float> ds(static_cast<size_t>(2 * k));
  expect_status(iovsCagraSearch(res.r, ix, q.data(), 2, k, 12, 2, nullptr, nb.data(), ds.data()), "s");
  iovsDevice last = IOVS_DEVICE_CPU;
  iovsResourcesLastDevice(res.r, &last);
  if (iovsSyclEnabled()) {
    expect(last == IOVS_DEVICE_GPU, "sycl walk last_device");
  } else {
    expect(last == IOVS_DEVICE_GPU, "openvino-gpu walk last_device");
  }
  iovsCagraDestroy(ix);
}

IOVS_TEST(cagra_sycl_walk_n_over_4096) {
  if (!iovsSyclEnabled()) return;
  Res res;
  int32_t gpu = 0;
  iovsResourcesGpuAvailable(res.r, &gpu);
  if (!gpu) return;
  /* n>4096: old kernel skipped expanding id>=4096 and overflowed slm at search_width>64. */
  const int64_t n = 4200, dim = 8, nq = 4, k = 5;
  auto data = make_data(n, dim, 442);
  auto q = make_data(nq, dim, 443);
  iovsResourcesSetPolicy(res.r, IOVS_POLICY_FORCE_CPU);
  iovsCagraIndex_t ix = nullptr;
  expect_status(iovsCagraBuild(res.r, data.data(), n, dim, IOVS_METRIC_L2_EXPANDED, 8, 16, &ix), "b");
  iovsResourcesSetPolicy(res.r, IOVS_POLICY_FORCE_GPU);
  std::vector<int64_t> got(static_cast<size_t>(nq * k)), truth(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k)), td(static_cast<size_t>(nq * k));
  expect_status(iovsCagraSearch(res.r, ix, q.data(), nq, k, 32, 80, nullptr, got.data(), gd.data()), "s");
  iovsDevice last = IOVS_DEVICE_CPU;
  iovsResourcesLastDevice(res.r, &last);
  expect(last == IOVS_DEVICE_GPU, "sycl n>4096 last_device");
  brute_oracle(data.data(), n, dim, q.data(), nq, k, truth.data(), td.data());
  for (int64_t i = 0; i < nq * k; ++i) {
    expect(got[static_cast<size_t>(i)] >= 0 && got[static_cast<size_t>(i)] < n, "id range");
  }
  const float rec = recall_at_k(got.data(), truth.data(), nq, k);
  expect(rec >= 0.2f, "sycl n>4096 recall vs independent L2 " + std::to_string(rec));
  iovsCagraDestroy(ix);
}

IOVS_TEST(ivf_flat_serialize_extend) {
  Res res;
  iovsResourcesSetPolicy(res.r, IOVS_POLICY_FORCE_CPU);
  const int64_t n = 40, dim = 8, extra = 5, nq = 4, k = 4;
  auto all = make_data(n + extra, dim, 450);
  auto q = make_data(nq, dim, 451);
  iovsIvfFlatIndex_t ix = nullptr;
  expect_status(iovsIvfFlatBuild(res.r, all.data(), n, dim, IOVS_METRIC_L2_EXPANDED, 8, &ix), "b");
  const auto path = std::filesystem::temp_directory_path() / "iovs_ivfflat.bin";
  expect_status(iovsIvfFlatSerialize(ix, path.string().c_str()), "ser");
  iovsIvfFlatIndex_t loaded = nullptr;
  expect_status(iovsIvfFlatDeserialize(res.r, path.string().c_str(), &loaded), "des");
  expect_status(iovsIvfFlatExtend(res.r, loaded, all.data() + n * dim, extra), "ext");
  std::vector<int64_t> got(static_cast<size_t>(nq * k)), truth(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k)), td(static_cast<size_t>(nq * k));
  expect_status(iovsIvfFlatSearch(res.r, loaded, q.data(), nq, k, 8, nullptr, got.data(), gd.data()),
                "s");
  brute_oracle(all.data(), n + extra, dim, q.data(), nq, k, truth.data(), td.data());
  expect(recall_at_k(got.data(), truth.data(), nq, k) >= 0.7f, "ivf extend recall");
  iovsIvfFlatDestroy(loaded);
  iovsIvfFlatDestroy(ix);
}

IOVS_TEST(vamana_serialize_mmap_filter) {
  Res res;
  iovsResourcesSetPolicy(res.r, IOVS_POLICY_FORCE_CPU);
  const int64_t n = 28, dim = 6, nq = 3, k = 3;
  auto data = make_data(n, dim, 460);
  auto q = make_data(nq, dim, 461);
  iovsVamanaIndex_t vx = nullptr;
  expect_status(iovsVamanaBuild(res.r, data.data(), n, dim, IOVS_METRIC_L2_EXPANDED, 6, 1.2f, &vx), "b");
  const auto path = std::filesystem::temp_directory_path() / "iovs_vamana.bin";
  expect_status(iovsVamanaSerialize(vx, path.string().c_str()), "ser");
  iovsVamanaIndex_t mapped = nullptr;
  expect_status(iovsVamanaMmap(res.r, path.string().c_str(), &mapped), "mmap");
  std::vector<uint8_t> bits((n + 7) / 8, 0xff);
  bits[0] = 0xfe;
  std::vector<int64_t> got(static_cast<size_t>(nq * k));
  std::vector<float> gd(static_cast<size_t>(nq * k));
  expect_status(iovsVamanaSearch(res.r, mapped, q.data(), nq, k, 12, bits.data(), got.data(), gd.data()),
                "ms");
  for (int64_t i = 0; i < nq * k; ++i) expect(got[static_cast<size_t>(i)] != 0, "vamana mmap filter");
  iovsVamanaDestroy(mapped);
  iovsVamanaDestroy(vx);
}

IOVS_TEST(hnsw_hnswlib_format_roundtrip) {
  Res res;
  iovsResourcesSetPolicy(res.r, IOVS_POLICY_FORCE_CPU);
  const int64_t n = 24, dim = 6, nq = 3, k = 3;
  auto data = make_data(n, dim, 470);
  auto q = make_data(nq, dim, 471);
  iovsCagraIndex_t cg = nullptr;
  expect_status(iovsCagraBuild(res.r, data.data(), n, dim, IOVS_METRIC_L2_EXPANDED, 6, 12, &cg), "c");
  iovsHnswIndex_t hx = nullptr;
  expect_status(iovsHnswFromCagra(res.r, cg, &hx), "from");
  std::vector<int64_t> cagra_nb(static_cast<size_t>(nq * k)), hnsw_nb(static_cast<size_t>(nq * k));
  std::vector<float> ds(static_cast<size_t>(nq * k));
  expect_status(iovsCagraSearch(res.r, cg, q.data(), nq, k, 16, 2, nullptr, cagra_nb.data(), ds.data()),
                "cs");
  expect_status(iovsHnswSearch(res.r, hx, q.data(), nq, k, 16, hnsw_nb.data(), ds.data()), "hs");
  const float overlap = recall_at_k(hnsw_nb.data(), cagra_nb.data(), nq, k);
  expect(overlap >= 0.5f, "hnsw vs cagra " + std::to_string(overlap));
  const auto path = std::filesystem::temp_directory_path() / "iovs_hnsw.bin";
  expect_status(iovsHnswSerialize(hx, path.string().c_str()), "ser");
  iovsHnswIndex_t loaded = nullptr;
  expect_status(iovsHnswDeserialize(res.r, path.string().c_str(), &loaded), "des");
  std::vector<int64_t> lnb(static_cast<size_t>(nq * k));
  expect_status(iovsHnswSearch(res.r, loaded, q.data(), nq, k, 16, lnb.data(), ds.data()), "ls");
  expect(recall_at_k(lnb.data(), hnsw_nb.data(), nq, k) >= 0.7f, "hnsw deser");
  /* Header is hnswlib saveIndex: size_t offsetLevel0 at byte 0 must be 0. */
  std::ifstream hf(path, std::ios::binary);
  size_t offset0 = 1;
  hf.read(reinterpret_cast<char*>(&offset0), sizeof(size_t));
  expect(offset0 == 0, "hnswlib offsetLevel0");
  iovsHnswDestroy(loaded);
  iovsHnswDestroy(hx);
  iovsCagraDestroy(cg);
}

/* Independent IEEE-754 binary16 decoder (not libiovs). Search is compared to L2 on these floats. */
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

IOVS_TEST(brute_fp16_int8_vs_fp32_oracle) {
  Res res;
  iovsResourcesSetPolicy(res.r, IOVS_POLICY_FORCE_CPU);
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
  iovsBruteForceIndex_t hix = nullptr;
  expect_status(iovsBruteForceBuildTyped(res.r, h.data(), n, dim, IOVS_METRIC_L2_EXPANDED, IOVS_DTYPE_F16,
                                         &hix),
                "f16");
  std::vector<int64_t> nb(static_cast<size_t>(k));
  std::vector<float> ds(static_cast<size_t>(k));
  expect_status(iovsBruteForceSearch(res.r, hix, q.data(), 1, k, nullptr, nb.data(), ds.data()), "f16s");
  /* Distances are L2(query_f32, decode(f16(dataset))). Ids must match that oracle.
     Atol 2e-2 covers binary16 mantissa (~1e-3) accumulated over dim=8. */
  constexpr float kF16DistAtol = 2e-2f;
  for (int64_t t = 0; t < k; ++t) {
    expect(nb[static_cast<size_t>(t)] == truth[static_cast<size_t>(t)], "f16 neighbor id");
    expect(std::fabs(ds[static_cast<size_t>(t)] - td[static_cast<size_t>(t)]) < kF16DistAtol, "f16 dist");
  }
  iovsBruteForceDestroy(hix);

  std::vector<float> i8f(static_cast<size_t>(n * dim));
  for (size_t i = 0; i < i8.size(); ++i) i8f[i] = static_cast<float>(i8[i]);
  brute_oracle(i8f.data(), n, dim, q.data(), 1, k, truth.data(), td.data());
  iovsBruteForceIndex_t iix = nullptr;
  expect_status(iovsBruteForceBuildTyped(res.r, i8.data(), n, dim, IOVS_METRIC_L2_EXPANDED, IOVS_DTYPE_I8,
                                         &iix),
                "i8");
  expect_status(iovsBruteForceSearch(res.r, iix, q.data(), 1, k, nullptr, nb.data(), ds.data()), "i8s");
  for (int64_t t = 0; t < k; ++t) expect(nb[static_cast<size_t>(t)] == truth[static_cast<size_t>(t)], "i8 id");
  iovsBruteForceDestroy(iix);
}

IOVS_TEST(dynamic_batcher_matches_eager) {
  Res res;
  iovsResourcesSetPolicy(res.r, IOVS_POLICY_FORCE_CPU);
  const int64_t n = 30, dim = 8, nq = 4, k = 3;
  auto data = make_data(n, dim, 490);
  auto q = make_data(nq, dim, 491);
  iovsBruteForceIndex_t ix = nullptr;
  expect_status(iovsBruteForceBuild(res.r, data.data(), n, dim, IOVS_METRIC_L2_EXPANDED, &ix), "b");
  std::vector<int64_t> eager(static_cast<size_t>(nq * k));
  std::vector<float> eds(static_cast<size_t>(nq * k));
  expect_status(iovsBruteForceSearch(res.r, ix, q.data(), nq, k, nullptr, eager.data(), eds.data()), "e");

  /* Concurrent nq=1 submits: max_batch=4, max_wait=250ms so they merge into one prim search. */
  iovsBatcher_t b = nullptr;
  expect_status(iovsBatcherCreate(res.r, ix, 4, 250, &b), "batch");
  std::vector<int64_t> got(static_cast<size_t>(nq * k), -1);
  std::vector<float> gds(static_cast<size_t>(nq * k), 0.f);
  std::vector<iovsStatus> sts(static_cast<size_t>(nq), IOVS_STATUS_ERROR);
  std::latch go(static_cast<std::ptrdiff_t>(nq));
  std::vector<std::thread> th;
  th.reserve(static_cast<size_t>(nq));
  for (int64_t i = 0; i < nq; ++i) {
    th.emplace_back([&, i] {
      go.arrive_and_wait();
      sts[static_cast<size_t>(i)] = iovsBatcherSearch(b, q.data() + i * dim, 1, k,
                                                      got.data() + i * k, gds.data() + i * k);
    });
  }
  for (auto& t : th) t.join();
  for (int64_t i = 0; i < nq; ++i) expect_status(sts[static_cast<size_t>(i)], "thread search");
  int32_t coalesced = 0;
  expect_status(iovsBatcherLastBatchSize(b, &coalesced), "last batch");
  expect(coalesced == static_cast<int32_t>(nq), "concurrent queries must coalesce");
  for (int64_t i = 0; i < nq * k; ++i)
    expect(got[static_cast<size_t>(i)] == eager[static_cast<size_t>(i)], "coalesced vs eager");
  iovsBatcherDestroy(b);
  iovsBruteForceDestroy(ix);
}
