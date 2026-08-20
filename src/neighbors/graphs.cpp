#include "internal.hpp"

#include <fstream>

using namespace iovs::impl;

namespace {

constexpr uint32_t kCagraMagic = 0x31475243u; /* 'CRG1' */

struct Dataset {
  std::vector<float> x;
  int64_t n = 0;
  int64_t dim = 0;
  iovsMetric metric = IOVS_METRIC_L2_EXPANDED;
};

void copy_ds(Dataset& d, const float* x, int64_t n, int64_t dim, iovsMetric m) {
  d.n = n;
  d.dim = dim;
  d.metric = m;
  d.x.assign(x, x + n * dim);
}

struct IvfRabitq {
  Dataset ds;
  int32_t nlist = 0;
  std::vector<float> centroids;
  std::vector<int32_t> assign;
  std::vector<float> scales;
  std::vector<uint8_t> signs;
  std::vector<uint8_t> rand_sign;
  std::vector<std::vector<int64_t>> lists;
  int64_t nbytes = 0;
};

struct NnGraph {
  Dataset ds;
  int32_t degree = 0;
  std::vector<int32_t> ids;
};

struct CagraIndex {
  Dataset ds;
  int32_t degree = 0;
  std::vector<int32_t> graph;
  bool has_dataset = true;
  int32_t pq_m = 0;
  int32_t pq_ks = 0;
  int32_t dsub = 0;
  std::vector<float> codebooks;
  std::vector<uint8_t> codes;
};

struct HnswIndex {
  Dataset ds;
  int32_t degree = 0;
  std::vector<int32_t> graph;
  std::vector<int32_t> level;
  int32_t max_level = 0;
  int64_t enter = 0;
};

struct VamanaIndex {
  Dataset ds;
  int32_t degree = 0;
  std::vector<int32_t> graph;
  const int32_t* graph_view = nullptr;
  const float* x_view = nullptr;
  bool mmapped = false;
#ifdef _WIN32
  HANDLE file_handle = INVALID_HANDLE_VALUE;
  HANDLE map_handle = nullptr;
  void* view = nullptr;
#else
  int fd = -1;
  void* view = nullptr;
  size_t map_size = 0;
#endif
};

struct ScannIndex {
  iovsIvfPqIndex_t pq = nullptr;
  std::vector<float> aniso;
};

int64_t nbytes_bits(int64_t dim) { return (dim + 7) / 8; }

void pack_signs(const float* v, int64_t dim, const uint8_t* rand_sign, uint8_t* out) {
  const int64_t nb = nbytes_bits(dim);
  std::fill(out, out + nb, 0);
  for (int64_t i = 0; i < dim; ++i) {
    float x = v[i];
    if (rand_sign && rand_sign[static_cast<size_t>(i)]) x = -x;
    if (x >= 0.f) out[i >> 3] |= static_cast<uint8_t>(1u << (i & 7));
  }
}

float rabitq_ip(const float* qres, const uint8_t* code, const uint8_t* rand_sign, int64_t dim,
                float scale) {
  float s = 0.f;
  const float inv = 1.f / std::sqrt(static_cast<float>(std::max<int64_t>(dim, 1)));
  for (int64_t i = 0; i < dim; ++i) {
    float bit = ((code[i >> 3] >> (i & 7)) & 1u) ? 1.f : -1.f;
    if (rand_sign && rand_sign[static_cast<size_t>(i)]) bit = -bit;
    s += qres[i] * bit;
  }
  return scale * s * inv;
}

void build_ivf_lists(const int32_t* assign, int64_t n, int32_t nlist,
                     std::vector<std::vector<int64_t>>& lists) {
  lists.assign(static_cast<size_t>(nlist), {});
  for (int64_t i = 0; i < n; ++i) {
    const int32_t a = assign[i];
    if (a >= 0 && a < nlist) lists[static_cast<size_t>(a)].push_back(i);
  }
}

void knn_graph_brute(ResourcesData& r, const float* x, int64_t n, int64_t dim, iovsMetric metric,
                     int32_t degree, std::vector<int32_t>& graph) {
  degree = std::min(degree, static_cast<int32_t>(std::max<int64_t>(1, n - 1)));
  graph.assign(static_cast<size_t>(n) * static_cast<size_t>(degree), -1);
  std::vector<float> scores(static_cast<size_t>(n) * static_cast<size_t>(n));
  prim_pairwise(r, metric, x, n, x, n, dim, scores.data(), 2.f);
  for (int64_t i = 0; i < n; ++i) scores[static_cast<size_t>(i * n + i)] = kInf;
  std::vector<int64_t> idx(static_cast<size_t>(n) * static_cast<size_t>(degree));
  std::vector<float> val(static_cast<size_t>(n) * static_cast<size_t>(degree));
  prim_topk(r, scores.data(), n, n, degree, idx.data(), val.data(), false);
  for (int64_t i = 0; i < n; ++i) {
    for (int32_t t = 0; t < degree; ++t) {
      graph[static_cast<size_t>(i * degree + t)] =
          static_cast<int32_t>(idx[static_cast<size_t>(i * degree + t)]);
    }
  }
}

void nndescent_build(ResourcesData& r, const float* x, int64_t n, int64_t dim, iovsMetric metric,
                     int32_t degree, int32_t iters, std::vector<int32_t>& graph) {
  degree = std::min(degree, static_cast<int32_t>(std::max<int64_t>(1, n - 1)));
  if (n * n < 160000) {
    knn_graph_brute(r, x, n, dim, metric, degree, graph);
    return;
  }
  graph.assign(static_cast<size_t>(n) * static_cast<size_t>(degree), -1);
  auto rng = rng_from(7);
  std::uniform_int_distribution<int64_t> pick(0, n - 1);
  for (int64_t i = 0; i < n; ++i) {
    std::vector<int32_t> row;
    while (static_cast<int32_t>(row.size()) < degree) {
      int64_t j = pick(rng);
      if (j == i) continue;
      if (std::find(row.begin(), row.end(), static_cast<int32_t>(j)) != row.end()) continue;
      row.push_back(static_cast<int32_t>(j));
    }
    std::memcpy(graph.data() + i * degree, row.data(), static_cast<size_t>(degree) * sizeof(int32_t));
  }
  for (int it = 0; it < iters; ++it) {
    for (int64_t i = 0; i < n; ++i) {
      std::vector<int64_t> cands;
      cands.reserve(static_cast<size_t>(degree) * (degree + 1));
      for (int32_t a = 0; a < degree; ++a) {
        const int32_t u = graph[static_cast<size_t>(i * degree + a)];
        if (u < 0 || u == static_cast<int32_t>(i)) continue;
        cands.push_back(u);
        for (int32_t b = 0; b < degree; ++b) {
          const int32_t v = graph[static_cast<size_t>(static_cast<int64_t>(u) * degree + b)];
          if (v < 0 || v == static_cast<int32_t>(i)) continue;
          cands.push_back(v);
        }
      }
      std::sort(cands.begin(), cands.end());
      cands.erase(std::unique(cands.begin(), cands.end()), cands.end());
      if (cands.empty()) continue;
      std::vector<float> gathered(cands.size() * static_cast<size_t>(dim));
      std::vector<float> sc(cands.size());
      prim_gather_rows(r, x, n, dim, cands.data(), static_cast<int64_t>(cands.size()), gathered.data());
      prim_pairwise(r, metric, x + i * dim, 1, gathered.data(), static_cast<int64_t>(cands.size()), dim,
                    sc.data(), 2.f);
      const int64_t kk = std::min(static_cast<int64_t>(degree), static_cast<int64_t>(cands.size()));
      std::vector<int64_t> ti(static_cast<size_t>(kk));
      std::vector<float> tv(static_cast<size_t>(kk));
      prim_topk(r, sc.data(), 1, static_cast<int64_t>(cands.size()), kk, ti.data(), tv.data(), false);
      for (int64_t t = 0; t < degree; ++t) {
        if (t < kk) {
          graph[static_cast<size_t>(i * degree + t)] =
              static_cast<int32_t>(cands[static_cast<size_t>(ti[static_cast<size_t>(t)])]);
        } else {
          graph[static_cast<size_t>(i * degree + t)] = -1;
        }
      }
    }
  }
}

void prune_graph(const float* x, int64_t n, int64_t dim, iovsMetric metric, const int32_t* in,
                 int32_t in_deg, int32_t out_deg, std::vector<int32_t>& out) {
  out.assign(static_cast<size_t>(n) * static_cast<size_t>(out_deg), -1);
  std::vector<int32_t> cand;
  for (int64_t i = 0; i < n; ++i) {
    cand.clear();
    for (int32_t t = 0; t < in_deg; ++t) {
      const int32_t nb = in[i * in_deg + t];
      if (nb >= 0 && nb != static_cast<int32_t>(i)) cand.push_back(nb);
    }
    for (int64_t j = 0; j < n && n <= 4096; ++j) {
      if (j == i) continue;
      for (int32_t t = 0; t < in_deg; ++t) {
        if (in[j * in_deg + t] == static_cast<int32_t>(i)) {
          cand.push_back(static_cast<int32_t>(j));
          break;
        }
      }
    }
    std::sort(cand.begin(), cand.end());
    cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
    std::sort(cand.begin(), cand.end(), [&](int32_t a, int32_t b) {
      return distance_one(metric, x + i * dim, x + static_cast<int64_t>(a) * dim, dim, 2.f) <
             distance_one(metric, x + i * dim, x + static_cast<int64_t>(b) * dim, dim, 2.f);
    });
    std::vector<int32_t> kept;
    for (int32_t v : cand) {
      bool ok = true;
      const float dv = distance_one(metric, x + i * dim, x + static_cast<int64_t>(v) * dim, dim, 2.f);
      for (int32_t u : kept) {
        const float duv = distance_one(metric, x + static_cast<int64_t>(u) * dim,
                                       x + static_cast<int64_t>(v) * dim, dim, 2.f);
        if (duv < dv) {
          ok = false;
          break;
        }
      }
      if (ok) kept.push_back(v);
      if (static_cast<int32_t>(kept.size()) >= out_deg) break;
    }
    for (size_t t = 0; t < kept.size() && t < static_cast<size_t>(out_deg); ++t) {
      out[static_cast<size_t>(i * out_deg + static_cast<int64_t>(t))] = kept[t];
    }
  }
}

void graph_search(ResourcesData& r, const float* dataset, int64_t n, int64_t dim, iovsMetric metric,
                  const int32_t* graph, int32_t degree, const float* queries, int64_t nq, int64_t k,
                  int32_t itopk, int32_t search_width, const uint8_t* bitset, int64_t* neighbors,
                  float* distances) {
  prim_graph_walk(r, dataset, n, dim, metric, graph, degree, queries, nq, k, itopk, search_width, bitset,
                  neighbors, distances);
}

void pq_encode_rows(const float* x, int64_t n, int64_t dim, int32_t pq_m, int32_t ks, int32_t dsub,
                    const float* codebooks, uint8_t* codes) {
  for (int64_t i = 0; i < n; ++i) {
    for (int32_t m = 0; m < pq_m; ++m) {
      const float* sub = x + i * dim + static_cast<int64_t>(m) * dsub;
      const float* cb = codebooks + static_cast<size_t>(m) * ks * dsub;
      int best = 0;
      float bd = kInf;
      for (int32_t c = 0; c < ks; ++c) {
        const float d = l2sq(sub, cb + c * dsub, dsub);
        if (d < bd) {
          bd = d;
          best = c;
        }
      }
      codes[i * pq_m + m] = static_cast<uint8_t>(best);
    }
  }
}

float pq_adc(const float* query, int64_t dim, const uint8_t* code, int32_t pq_m, int32_t ks, int32_t dsub,
             const float* codebooks) {
  float s = 0.f;
  for (int32_t m = 0; m < pq_m; ++m) {
    const float* sub = query + static_cast<int64_t>(m) * dsub;
    const float* cb = codebooks + (static_cast<size_t>(m) * ks + code[m]) * dsub;
    s += l2sq(sub, cb, dsub);
  }
  (void)dim;
  return s;
}

void graph_search_pq(const float* query_ds, int64_t n, int64_t dim, const int32_t* graph, int32_t degree,
                     const uint8_t* codes, int32_t pq_m, int32_t ks, int32_t dsub, const float* codebooks,
                     const float* queries, int64_t nq, int64_t k, int32_t itopk, int32_t search_width,
                     const uint8_t* bitset, int64_t* neighbors, float* distances) {
  itopk = std::max(itopk, static_cast<int32_t>(k));
  search_width = std::max(1, search_width);
  struct Node {
    float d;
    int64_t id;
  };
  for (int64_t q = 0; q < nq; ++q) {
    const float* query = queries + q * dim;
    std::vector<uint8_t> seen(static_cast<size_t>(n), 0);
    std::vector<char> expanded(static_cast<size_t>(n), 0);
    std::vector<Node> cand;
    const int64_t nseeds = std::min<int64_t>(search_width, n);
    for (int64_t s = 0; s < nseeds; ++s) {
      int64_t id = (s * 9973 + q * 13) % n;
      if (!allowed(bitset, id) || seen[static_cast<size_t>(id)]) continue;
      seen[static_cast<size_t>(id)] = 1;
      cand.push_back({pq_adc(query, dim, codes + id * pq_m, pq_m, ks, dsub, codebooks), id});
    }
    int iters = 0;
    const int max_iters = std::max(24, itopk * 6);
    while (iters++ < max_iters) {
      int64_t pick = -1;
      float pick_d = kInf;
      for (const auto& c : cand) {
        if (!expanded[static_cast<size_t>(c.id)] && c.d < pick_d) {
          pick_d = c.d;
          pick = c.id;
        }
      }
      if (pick < 0) break;
      expanded[static_cast<size_t>(pick)] = 1;
      const int32_t* nbrs = graph + pick * degree;
      for (int32_t e = 0; e < degree; ++e) {
        const int32_t nb = nbrs[e];
        if (nb < 0 || static_cast<int64_t>(nb) >= n) continue;
        if (seen[static_cast<size_t>(nb)]) continue;
        if (!allowed(bitset, nb)) continue;
        seen[static_cast<size_t>(nb)] = 1;
        cand.push_back({pq_adc(query, dim, codes + static_cast<int64_t>(nb) * pq_m, pq_m, ks, dsub, codebooks),
                        nb});
      }
      if (static_cast<int32_t>(cand.size()) > itopk * 4) {
        std::nth_element(cand.begin(), cand.begin() + itopk, cand.end(),
                         [](const Node& a, const Node& b) { return a.d < b.d; });
        cand.resize(static_cast<size_t>(itopk));
      }
    }
    std::sort(cand.begin(), cand.end(), [](const Node& a, const Node& b) { return a.d < b.d; });
    for (int64_t t = 0; t < k; ++t) {
      if (t < static_cast<int64_t>(cand.size())) {
        neighbors[q * k + t] = cand[static_cast<size_t>(t)].id;
        distances[q * k + t] = cand[static_cast<size_t>(t)].d;
      } else {
        neighbors[q * k + t] = -1;
        distances[q * k + t] = kInf;
      }
    }
  }
  (void)query_ds;
}

void cagra_init_ivfpq(ResourcesData& r, const float* x, int64_t n, int64_t dim, iovsMetric metric,
                      int32_t degree, std::vector<int32_t>& graph) {
  iovsIvfPqIndex_t pq = nullptr;
  const int32_t nlist = std::max(2, std::min(8, static_cast<int32_t>(n / 4)));
  int32_t pq_m = 1;
  for (int32_t c = std::min(4, static_cast<int32_t>(dim)); c >= 1; --c) {
    if (dim % c == 0) {
      pq_m = c;
      break;
    }
  }
  auto* res = reinterpret_cast<iovsResources_t>(&r);
  if (iovsIvfPqBuild(res, x, n, dim, metric, nlist, pq_m, 8, &pq) != IOVS_STATUS_SUCCESS) {
    nndescent_build(r, x, n, dim, metric, degree, 6, graph);
    return;
  }
  graph.assign(static_cast<size_t>(n) * static_cast<size_t>(degree), -1);
  std::vector<int64_t> nb(static_cast<size_t>(degree) + 1);
  std::vector<float> ds(static_cast<size_t>(degree) + 1);
  for (int64_t i = 0; i < n; ++i) {
    iovsIvfPqSearch(res, pq, x + i * dim, 1, degree + 1, nlist, static_cast<int32_t>(degree) + 1, nullptr,
                    nb.data(), ds.data());
    int filled = 0;
    for (int32_t t = 0; t < degree + 1 && filled < degree; ++t) {
      if (nb[static_cast<size_t>(t)] == i || nb[static_cast<size_t>(t)] < 0) continue;
      graph[static_cast<size_t>(i * degree + filled)] = static_cast<int32_t>(nb[static_cast<size_t>(t)]);
      ++filled;
    }
  }
  iovsIvfPqDestroy(pq);
}

void cagra_init_iterative(ResourcesData& r, const float* x, int64_t n, int64_t dim, iovsMetric metric,
                          int32_t degree, std::vector<int32_t>& graph) {
  nndescent_build(r, x, n, dim, metric, degree, 4, graph);
  std::vector<int32_t> next = graph;
  std::vector<int64_t> nb(static_cast<size_t>(degree));
  std::vector<float> ds(static_cast<size_t>(degree));
  for (int it = 0; it < 2; ++it) {
    for (int64_t i = 0; i < n; ++i) {
      graph_search(r, x, n, dim, metric, graph.data(), degree, x + i * dim, 1, degree, degree * 2, 2,
                   nullptr, nb.data(), ds.data());
      int filled = 0;
      for (int32_t t = 0; t < degree && filled < degree; ++t) {
        if (nb[static_cast<size_t>(t)] == i || nb[static_cast<size_t>(t)] < 0) continue;
        next[static_cast<size_t>(i * degree + filled)] = static_cast<int32_t>(nb[static_cast<size_t>(t)]);
        ++filled;
      }
      while (filled < degree) {
        next[static_cast<size_t>(i * degree + filled)] = -1;
        ++filled;
      }
    }
    graph.swap(next);
  }
}

void robust_prune(const float* x, int64_t dim, iovsMetric metric, int64_t p, std::vector<int32_t>& cand,
                  int32_t degree, float alpha, int32_t* out_row);

void cagra_insert_one(CagraIndex* ix, ResourcesData& r, const float* vec) {
  const int64_t old_n = ix->ds.n;
  const int64_t dim = ix->ds.dim;
  ix->ds.x.insert(ix->ds.x.end(), vec, vec + dim);
  ix->ds.n = old_n + 1;
  std::vector<int64_t> nb(static_cast<size_t>(ix->degree));
  std::vector<float> ds(static_cast<size_t>(ix->degree));
  graph_search(r, ix->ds.x.data(), old_n, dim, ix->ds.metric, ix->graph.data(), ix->degree, vec, 1,
               ix->degree, ix->degree * 2, 2, nullptr, nb.data(), ds.data());
  ix->graph.resize(static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->degree), -1);
  for (int32_t t = 0; t < ix->degree; ++t) {
    ix->graph[static_cast<size_t>(old_n * ix->degree + t)] =
        t < static_cast<int32_t>(nb.size()) ? static_cast<int32_t>(nb[static_cast<size_t>(t)]) : -1;
  }
  for (int32_t t = 0; t < ix->degree; ++t) {
    const int32_t u = ix->graph[static_cast<size_t>(old_n * ix->degree + t)];
    if (u < 0) continue;
    std::vector<int32_t> cand;
    for (int32_t e = 0; e < ix->degree; ++e) {
      const int32_t v = ix->graph[static_cast<size_t>(static_cast<int64_t>(u) * ix->degree + e)];
      if (v >= 0) cand.push_back(v);
    }
    cand.push_back(static_cast<int32_t>(old_n));
    robust_prune(ix->ds.x.data(), dim, ix->ds.metric, u, cand, ix->degree, 1.2f,
                 ix->graph.data() + static_cast<int64_t>(u) * ix->degree);
  }
}

void robust_prune(const float* x, int64_t dim, iovsMetric metric, int64_t p, std::vector<int32_t>& cand,
                  int32_t degree, float alpha, int32_t* out_row) {
  std::sort(cand.begin(), cand.end());
  cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
  std::sort(cand.begin(), cand.end(), [&](int32_t a, int32_t b) {
    return distance_one(metric, x + p * dim, x + static_cast<int64_t>(a) * dim, dim, 2.f) <
           distance_one(metric, x + p * dim, x + static_cast<int64_t>(b) * dim, dim, 2.f);
  });
  std::vector<int32_t> kept;
  for (int32_t v : cand) {
    if (v == static_cast<int32_t>(p) || v < 0) continue;
    bool ok = true;
    const float dpv = distance_one(metric, x + p * dim, x + static_cast<int64_t>(v) * dim, dim, 2.f);
    for (int32_t u : kept) {
      const float duv = distance_one(metric, x + static_cast<int64_t>(u) * dim,
                                     x + static_cast<int64_t>(v) * dim, dim, 2.f);
      if (alpha * duv <= dpv) {
        ok = false;
        break;
      }
    }
    if (ok) kept.push_back(v);
    if (static_cast<int32_t>(kept.size()) >= degree) break;
  }
  std::fill(out_row, out_row + degree, -1);
  for (size_t t = 0; t < kept.size(); ++t) out_row[t] = kept[t];
}

}  // namespace

iovsStatus iovsIvfRabitqBuild(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                              iovsMetric metric, int32_t nlist, iovsIvfRabitqIndex_t* index) {
  if (!res || !dataset || !index || n <= 0 || dim <= 0 || nlist <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  nlist = std::min(nlist, static_cast<int32_t>(n));
  auto* ix = new IvfRabitq();
  copy_ds(ix->ds, dataset, n, dim, metric);
  ix->nlist = nlist;
  ix->nbytes = nbytes_bits(dim);
  ix->rand_sign.resize(static_cast<size_t>(dim));
  auto rng = rng_from(99);
  std::bernoulli_distribution coin(0.5);
  for (int64_t i = 0; i < dim; ++i) ix->rand_sign[static_cast<size_t>(i)] = coin(rng) ? 1 : 0;
  kmeans_fit_impl(*rd(res), dataset, n, dim, nlist, 12, ix->centroids);
  std::vector<int64_t> labels(static_cast<size_t>(n));
  std::vector<float> d(static_cast<size_t>(n));
  std::vector<float> scores(static_cast<size_t>(n) * static_cast<size_t>(nlist));
  prim_pairwise(*rd(res), IOVS_METRIC_L2_EXPANDED, dataset, n, ix->centroids.data(), nlist, dim,
                scores.data(), 2.f);
  prim_topk(*rd(res), scores.data(), n, nlist, 1, labels.data(), d.data(), false);
  ix->assign.resize(static_cast<size_t>(n));
  ix->scales.resize(static_cast<size_t>(n));
  ix->signs.assign(static_cast<size_t>(n * ix->nbytes), 0);
  std::vector<float> resid(static_cast<size_t>(dim));
  for (int64_t i = 0; i < n; ++i) {
    const int32_t a = static_cast<int32_t>(labels[static_cast<size_t>(i)]);
    ix->assign[static_cast<size_t>(i)] = a;
    const float* c = ix->centroids.data() + static_cast<size_t>(a) * dim;
    for (int64_t t = 0; t < dim; ++t) resid[static_cast<size_t>(t)] = dataset[i * dim + t] - c[t];
    ix->scales[static_cast<size_t>(i)] = std::sqrt(std::max(nrm2sq(resid.data(), dim), 1e-12f));
    pack_signs(resid.data(), dim, ix->rand_sign.data(), ix->signs.data() + i * ix->nbytes);
  }
  build_ivf_lists(ix->assign.data(), n, nlist, ix->lists);
  *index = reinterpret_cast<iovsIvfRabitqIndex_t>(ix);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsIvfRabitqSearch(iovsResources_t res, iovsIvfRabitqIndex_t index, const float* queries,
                               int64_t nq, int64_t k, int32_t nprobe, int32_t krefine,
                               const uint8_t* bitset, int64_t* neighbors, float* distances) {
  if (!res || !index || !queries || !neighbors || !distances) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<IvfRabitq*>(index);
  nprobe = std::max(1, std::min(nprobe, ix->nlist));
  if (krefine < static_cast<int32_t>(k)) krefine = static_cast<int32_t>(k);
  std::vector<int64_t> cl(static_cast<size_t>(nq) * static_cast<size_t>(nprobe));
  std::vector<float> cd(static_cast<size_t>(nq) * static_cast<size_t>(nprobe));
  std::vector<float> cscores(static_cast<size_t>(nq) * static_cast<size_t>(ix->nlist));
  prim_pairwise(*rd(res), IOVS_METRIC_L2_EXPANDED, queries, nq, ix->centroids.data(), ix->nlist,
                ix->ds.dim, cscores.data(), 2.f);
  prim_topk(*rd(res), cscores.data(), nq, ix->nlist, nprobe, cl.data(), cd.data(), false);
  const int64_t dim = ix->ds.dim;
  for (int64_t q = 0; q < nq; ++q) {
    std::vector<int64_t> ids;
    std::vector<float> adc;
    const float* query = queries + q * dim;
    for (int32_t p = 0; p < nprobe; ++p) {
      const int64_t c = cl[q * nprobe + p];
      if (c < 0 || c >= ix->nlist) continue;
      const float* cent = ix->centroids.data() + c * dim;
      std::vector<float> qres(static_cast<size_t>(dim));
      for (int64_t t = 0; t < dim; ++t) qres[static_cast<size_t>(t)] = query[t] - cent[t];
      const float qn = nrm2sq(qres.data(), dim);
      for (int64_t id : ix->lists[static_cast<size_t>(c)]) {
        if (bitset && !allowed(bitset, id)) continue;
        const float ip =
            rabitq_ip(qres.data(), ix->signs.data() + id * ix->nbytes, ix->rand_sign.data(), dim,
                      ix->scales[static_cast<size_t>(id)]);
        const float sc = qn + ix->scales[static_cast<size_t>(id)] * ix->scales[static_cast<size_t>(id)] -
                         2.f * ip;
        ids.push_back(id);
        adc.push_back(sc);
      }
    }
    if (ids.empty()) {
      for (int64_t t = 0; t < k; ++t) {
        neighbors[q * k + t] = -1;
        distances[q * k + t] = kInf;
      }
      continue;
    }
    const int64_t kr = std::min(static_cast<int64_t>(krefine), static_cast<int64_t>(ids.size()));
    std::vector<int64_t> ti(static_cast<size_t>(kr));
    std::vector<float> tv(static_cast<size_t>(kr));
    prim_topk(*rd(res), adc.data(), 1, static_cast<int64_t>(ids.size()), kr, ti.data(), tv.data(), false);
    std::vector<int64_t> cand(static_cast<size_t>(kr));
    for (int64_t t = 0; t < kr; ++t) cand[static_cast<size_t>(t)] = ids[static_cast<size_t>(ti[static_cast<size_t>(t)])];
    std::vector<float> gathered(static_cast<size_t>(kr) * static_cast<size_t>(dim));
    prim_gather_rows(*rd(res), ix->ds.x.data(), ix->ds.n, dim, cand.data(), kr, gathered.data());
    std::vector<float> sc(static_cast<size_t>(kr));
    prim_pairwise(*rd(res), ix->ds.metric, query, 1, gathered.data(), kr, dim, sc.data(), 2.f);
    const int64_t kk = std::min(k, kr);
    std::vector<int64_t> fi(static_cast<size_t>(kk));
    std::vector<float> fv(static_cast<size_t>(kk));
    prim_topk(*rd(res), sc.data(), 1, kr, kk, fi.data(), fv.data(), metric_largest(ix->ds.metric));
    for (int64_t t = 0; t < k; ++t) {
      if (t < kk) {
        neighbors[q * k + t] = cand[static_cast<size_t>(fi[static_cast<size_t>(t)])];
        float dv = fv[static_cast<size_t>(t)];
        if (ix->ds.metric == IOVS_METRIC_INNER_PRODUCT) dv = -dv;
        distances[q * k + t] = dv;
      } else {
        neighbors[q * k + t] = -1;
        distances[q * k + t] = kInf;
      }
    }
  }
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsIvfRabitqDestroy(iovsIvfRabitqIndex_t index) {
  delete reinterpret_cast<IvfRabitq*>(index);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsNnDescentBuild(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                              iovsMetric metric, int32_t graph_degree, int32_t iterations,
                              iovsNnDescentGraph_t* graph) {
  if (!res || !dataset || !graph || n <= 1 || dim <= 0 || graph_degree <= 0) {
    return IOVS_STATUS_INVALID_ARGUMENT;
  }
  auto* g = new NnGraph();
  copy_ds(g->ds, dataset, n, dim, metric);
  g->degree = std::min(graph_degree, static_cast<int32_t>(n - 1));
  nndescent_build(*rd(res), dataset, n, dim, metric, g->degree, std::max(1, iterations), g->ids);
  *graph = reinterpret_cast<iovsNnDescentGraph_t>(g);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsNnDescentNeighbors(iovsNnDescentGraph_t graph, const int32_t** ids, int64_t* n,
                                  int32_t* degree) {
  if (!graph || !ids || !n || !degree) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* g = reinterpret_cast<NnGraph*>(graph);
  *ids = g->ids.data();
  *n = g->ds.n;
  *degree = g->degree;
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsNnDescentDestroy(iovsNnDescentGraph_t graph) {
  delete reinterpret_cast<NnGraph*>(graph);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsCagraBuildEx(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                            iovsMetric metric, int32_t graph_degree, int32_t intermediate_degree,
                            iovsCagraBuildAlgo algo, iovsCagraIndex_t* index) {
  if (!res || !dataset || !index || n <= 1 || dim <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  graph_degree = std::max(1, std::min(graph_degree, static_cast<int32_t>(n - 1)));
  intermediate_degree = std::max(graph_degree, std::min(intermediate_degree, static_cast<int32_t>(n - 1)));
  auto* ix = new CagraIndex();
  copy_ds(ix->ds, dataset, n, dim, metric);
  ix->degree = graph_degree;
  ix->has_dataset = true;
  std::vector<int32_t> init;
  if (algo == IOVS_CAGRA_BUILD_IVF_PQ) {
    cagra_init_ivfpq(*rd(res), dataset, n, dim, metric, intermediate_degree, init);
  } else if (algo == IOVS_CAGRA_BUILD_ITERATIVE) {
    cagra_init_iterative(*rd(res), dataset, n, dim, metric, intermediate_degree, init);
  } else {
    nndescent_build(*rd(res), dataset, n, dim, metric, intermediate_degree, 6, init);
  }
  prune_graph(dataset, n, dim, metric, init.data(), intermediate_degree, graph_degree, ix->graph);
  *index = reinterpret_cast<iovsCagraIndex_t>(ix);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsCagraBuild(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                          iovsMetric metric, int32_t graph_degree, int32_t intermediate_degree,
                          iovsCagraIndex_t* index) {
  return iovsCagraBuildEx(res, dataset, n, dim, metric, graph_degree, intermediate_degree,
                          IOVS_CAGRA_BUILD_NN_DESCENT, index);
}

iovsStatus iovsCagraSearch(iovsResources_t res, iovsCagraIndex_t index, const float* queries,
                           int64_t nq, int64_t k, int32_t itopk_size, int32_t search_width,
                           const uint8_t* bitset, int64_t* neighbors, float* distances) {
  if (!res || !index || !queries || !neighbors || !distances) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<CagraIndex*>(index);
  if (ix->pq_m > 0 && !ix->codes.empty()) {
    graph_search_pq(ix->ds.x.empty() ? nullptr : ix->ds.x.data(), ix->ds.n, ix->ds.dim, ix->graph.data(),
                    ix->degree, ix->codes.data(), ix->pq_m, ix->pq_ks, ix->dsub, ix->codebooks.data(),
                    queries, nq, k, itopk_size, search_width, bitset, neighbors, distances);
    if (ix->has_dataset && !ix->ds.x.empty()) {
      const int64_t dim = ix->ds.dim;
      for (int64_t q = 0; q < nq; ++q) {
        std::vector<int64_t> cand(static_cast<size_t>(k));
        int64_t nc = 0;
        for (int64_t t = 0; t < k; ++t) {
          if (neighbors[q * k + t] >= 0) cand[static_cast<size_t>(nc++)] = neighbors[q * k + t];
        }
        if (nc == 0) continue;
        std::vector<float> gathered(static_cast<size_t>(nc) * static_cast<size_t>(dim));
        prim_gather_rows(*rd(res), ix->ds.x.data(), ix->ds.n, dim, cand.data(), nc, gathered.data());
        std::vector<float> sc(static_cast<size_t>(nc));
        prim_pairwise(*rd(res), ix->ds.metric, queries + q * dim, 1, gathered.data(), nc, dim, sc.data(),
                      2.f);
        std::vector<int64_t> fi(static_cast<size_t>(nc));
        std::vector<float> fv(static_cast<size_t>(nc));
        prim_topk(*rd(res), sc.data(), 1, nc, nc, fi.data(), fv.data(), metric_largest(ix->ds.metric));
        for (int64_t t = 0; t < k; ++t) {
          if (t < nc) {
            neighbors[q * k + t] = cand[static_cast<size_t>(fi[static_cast<size_t>(t)])];
            float d = fv[static_cast<size_t>(t)];
            if (ix->ds.metric == IOVS_METRIC_INNER_PRODUCT) d = -d;
            distances[q * k + t] = d;
          } else {
            neighbors[q * k + t] = -1;
            distances[q * k + t] = kInf;
          }
        }
      }
    }
    return IOVS_STATUS_SUCCESS;
  }
  if (!ix->has_dataset || ix->ds.x.empty()) return IOVS_STATUS_INVALID_ARGUMENT;
  return prim_graph_walk(*rd(res), ix->ds.x.data(), ix->ds.n, ix->ds.dim, ix->ds.metric, ix->graph.data(),
                         ix->degree, queries, nq, k, itopk_size, search_width, bitset, neighbors,
                         distances);
}

iovsStatus iovsCagraSerializeEx(iovsCagraIndex_t index, const char* path, int32_t include_dataset) {
  if (!index || !path) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<CagraIndex*>(index);
  std::ofstream f(path, std::ios::binary);
  if (!f) return IOVS_STATUS_IO;
  f.write(reinterpret_cast<const char*>(&kCagraMagic), 4);
  int32_t ver = 2;
  f.write(reinterpret_cast<const char*>(&ver), 4);
  int32_t flags = 0;
  if (include_dataset && ix->has_dataset && !ix->ds.x.empty()) flags |= 1;
  if (ix->pq_m > 0 && !ix->codes.empty()) flags |= 2;
  f.write(reinterpret_cast<const char*>(&flags), 4);
  f.write(reinterpret_cast<const char*>(&ix->ds.n), 8);
  f.write(reinterpret_cast<const char*>(&ix->ds.dim), 8);
  f.write(reinterpret_cast<const char*>(&ix->degree), 4);
  int32_t metric = static_cast<int32_t>(ix->ds.metric);
  f.write(reinterpret_cast<const char*>(&metric), 4);
  f.write(reinterpret_cast<const char*>(ix->graph.data()),
          static_cast<std::streamsize>(ix->graph.size() * sizeof(int32_t)));
  if (flags & 1) {
    f.write(reinterpret_cast<const char*>(ix->ds.x.data()),
            static_cast<std::streamsize>(ix->ds.x.size() * sizeof(float)));
  }
  if (flags & 2) {
    f.write(reinterpret_cast<const char*>(&ix->pq_m), 4);
    f.write(reinterpret_cast<const char*>(&ix->pq_ks), 4);
    f.write(reinterpret_cast<const char*>(&ix->dsub), 4);
    f.write(reinterpret_cast<const char*>(ix->codebooks.data()),
            static_cast<std::streamsize>(ix->codebooks.size() * sizeof(float)));
    f.write(reinterpret_cast<const char*>(ix->codes.data()),
            static_cast<std::streamsize>(ix->codes.size()));
  }
  return f.good() ? IOVS_STATUS_SUCCESS : IOVS_STATUS_IO;
}

iovsStatus iovsCagraSerialize(iovsCagraIndex_t index, const char* path) {
  return iovsCagraSerializeEx(index, path, 1);
}

iovsStatus iovsCagraDeserialize(iovsResources_t res, const char* path, iovsCagraIndex_t* index) {
  if (!res || !path || !index) return IOVS_STATUS_INVALID_ARGUMENT;
  std::ifstream f(path, std::ios::binary);
  if (!f) return IOVS_STATUS_IO;
  uint32_t magic = 0;
  f.read(reinterpret_cast<char*>(&magic), 4);
  if (magic != kCagraMagic) return IOVS_STATUS_IO;
  int32_t ver = 0;
  f.read(reinterpret_cast<char*>(&ver), 4);
  auto* ix = new CagraIndex();
  int32_t flags = 1;
  if (ver >= 2) f.read(reinterpret_cast<char*>(&flags), 4);
  f.read(reinterpret_cast<char*>(&ix->ds.n), 8);
  f.read(reinterpret_cast<char*>(&ix->ds.dim), 8);
  f.read(reinterpret_cast<char*>(&ix->degree), 4);
  int32_t metric = 0;
  f.read(reinterpret_cast<char*>(&metric), 4);
  ix->ds.metric = static_cast<iovsMetric>(metric);
  ix->graph.resize(static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->degree));
  f.read(reinterpret_cast<char*>(ix->graph.data()),
         static_cast<std::streamsize>(ix->graph.size() * sizeof(int32_t)));
  if (flags & 1) {
    ix->ds.x.resize(static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->ds.dim));
    f.read(reinterpret_cast<char*>(ix->ds.x.data()),
           static_cast<std::streamsize>(ix->ds.x.size() * sizeof(float)));
    ix->has_dataset = true;
  } else {
    ix->has_dataset = false;
  }
  if (flags & 2) {
    f.read(reinterpret_cast<char*>(&ix->pq_m), 4);
    f.read(reinterpret_cast<char*>(&ix->pq_ks), 4);
    f.read(reinterpret_cast<char*>(&ix->dsub), 4);
    ix->codebooks.resize(static_cast<size_t>(ix->pq_m) * ix->pq_ks * ix->dsub);
    ix->codes.resize(static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->pq_m));
    f.read(reinterpret_cast<char*>(ix->codebooks.data()),
           static_cast<std::streamsize>(ix->codebooks.size() * sizeof(float)));
    f.read(reinterpret_cast<char*>(ix->codes.data()), static_cast<std::streamsize>(ix->codes.size()));
  }
  if (!f) {
    delete ix;
    return IOVS_STATUS_IO;
  }
  *index = reinterpret_cast<iovsCagraIndex_t>(ix);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsCagraExtend(iovsResources_t res, iovsCagraIndex_t index, const float* extra,
                           int64_t nextra) {
  if (!res || !index || !extra || nextra <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<CagraIndex*>(index);
  if (!ix->has_dataset || ix->ds.x.empty()) return IOVS_STATUS_INVALID_ARGUMENT;
  const int64_t dim = ix->ds.dim;
  for (int64_t i = 0; i < nextra; ++i) cagra_insert_one(ix, *rd(res), extra + i * dim);
  if (ix->pq_m > 0) {
    ix->codes.resize(static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->pq_m));
    pq_encode_rows(ix->ds.x.data(), ix->ds.n, dim, ix->pq_m, ix->pq_ks, ix->dsub, ix->codebooks.data(),
                   ix->codes.data());
  }
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsCagraQuantize(iovsResources_t res, iovsCagraIndex_t index, int32_t pq_m, int32_t pq_nbits) {
  if (!res || !index || pq_m <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<CagraIndex*>(index);
  if (!ix->has_dataset || ix->ds.x.empty()) return IOVS_STATUS_INVALID_ARGUMENT;
  if (ix->ds.dim % pq_m != 0) return IOVS_STATUS_SHAPE_MISMATCH;
  ix->pq_m = pq_m;
  ix->pq_ks = 1 << std::min(std::max(pq_nbits, 4), 8);
  ix->dsub = static_cast<int32_t>(ix->ds.dim / pq_m);
  ix->codebooks.assign(static_cast<size_t>(ix->pq_m) * ix->pq_ks * ix->dsub, 0.f);
  std::vector<float> sub(static_cast<size_t>(ix->ds.n) * ix->dsub);
  for (int32_t m = 0; m < ix->pq_m; ++m) {
    for (int64_t i = 0; i < ix->ds.n; ++i) {
      std::memcpy(sub.data() + i * ix->dsub, ix->ds.x.data() + i * ix->ds.dim + static_cast<int64_t>(m) * ix->dsub,
                  static_cast<size_t>(ix->dsub) * sizeof(float));
    }
    std::vector<float> cents;
    kmeans_fit_impl(*rd(res), sub.data(), ix->ds.n, ix->dsub, ix->pq_ks, 8, cents);
    std::memcpy(ix->codebooks.data() + static_cast<size_t>(m) * ix->pq_ks * ix->dsub, cents.data(),
                cents.size() * sizeof(float));
  }
  ix->codes.resize(static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->pq_m));
  pq_encode_rows(ix->ds.x.data(), ix->ds.n, ix->ds.dim, ix->pq_m, ix->pq_ks, ix->dsub, ix->codebooks.data(),
                 ix->codes.data());
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsCagraDetachDataset(iovsCagraIndex_t index) {
  if (!index) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<CagraIndex*>(index);
  ix->ds.x.clear();
  ix->ds.x.shrink_to_fit();
  ix->has_dataset = false;
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsCagraAttachDataset(iovsCagraIndex_t index, const float* dataset, int64_t n, int64_t dim) {
  if (!index || !dataset || n <= 0 || dim <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<CagraIndex*>(index);
  if (n != ix->ds.n || dim != ix->ds.dim) return IOVS_STATUS_SHAPE_MISMATCH;
  ix->ds.x.assign(dataset, dataset + n * dim);
  ix->has_dataset = true;
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsCagraDestroy(iovsCagraIndex_t index) {
  delete reinterpret_cast<CagraIndex*>(index);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsHnswFromCagra(iovsResources_t res, iovsCagraIndex_t cagra, iovsHnswIndex_t* index) {
  if (!res || !cagra || !index) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* cg = reinterpret_cast<CagraIndex*>(cagra);
  auto* hx = new HnswIndex();
  hx->ds = cg->ds;
  hx->degree = cg->degree;
  hx->graph = cg->graph;
  hx->level.assign(static_cast<size_t>(cg->ds.n), 0);
  auto rng = rng_from(3);
  std::uniform_real_distribution<float> u(0.f, 1.f);
  const float ml = 1.f / std::log(static_cast<float>(std::max(2, cg->degree)));
  hx->max_level = 0;
  hx->enter = 0;
  for (int64_t i = 0; i < cg->ds.n; ++i) {
    int lvl = static_cast<int>(-std::log(std::max(u(rng), 1e-6f)) * ml);
    hx->level[static_cast<size_t>(i)] = lvl;
    if (lvl > hx->max_level) {
      hx->max_level = lvl;
      hx->enter = i;
    }
  }
  *index = reinterpret_cast<iovsHnswIndex_t>(hx);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsHnswSearch(iovsResources_t res, iovsHnswIndex_t index, const float* queries, int64_t nq,
                          int64_t k, int32_t ef, int64_t* neighbors, float* distances) {
  if (!res || !index || !queries || !neighbors || !distances) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* hx = reinterpret_cast<HnswIndex*>(index);
  graph_search(*rd(res), hx->ds.x.data(), hx->ds.n, hx->ds.dim, hx->ds.metric, hx->graph.data(),
               hx->degree, queries, nq, k, std::max(ef, static_cast<int32_t>(k)), 1, nullptr, neighbors,
               distances);
  return IOVS_STATUS_SUCCESS;
}

namespace {
template <typename T>
void write_pod(std::ostream& f, const T& v) {
  f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T>
void read_pod(std::istream& f, T& v) {
  f.read(reinterpret_cast<char*>(&v), sizeof(T));
}
}  // namespace

iovsStatus iovsHnswSerialize(iovsHnswIndex_t index, const char* path) {
  /* hnswlib Index::saveIndex layout (little-endian host). Documented in docs/devices.md. */
  if (!index || !path) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* hx = reinterpret_cast<HnswIndex*>(index);
  std::ofstream f(path, std::ios::binary);
  if (!f) return IOVS_STATUS_IO;
  const size_t M = static_cast<size_t>(hx->degree);
  const size_t maxM0 = M;
  const size_t maxM = M;
  const size_t n = static_cast<size_t>(hx->ds.n);
  const size_t dim = static_cast<size_t>(hx->ds.dim);
  const size_t size_links_level0 = sizeof(unsigned int) + maxM0 * sizeof(unsigned int);
  const size_t data_size = dim * sizeof(float);
  const size_t offsetLevel0 = 0;
  const size_t offsetData = size_links_level0;
  const size_t label_offset = size_links_level0 + data_size;
  const size_t size_data_per_element = label_offset + sizeof(size_t);
  const int maxlevel = hx->max_level;
  const unsigned int enter = static_cast<unsigned int>(hx->enter);
  const double mult = 1.0 / std::log(static_cast<double>(std::max<size_t>(M, 2)));
  const size_t ef_construction = 200;
  write_pod(f, offsetLevel0);
  write_pod(f, n); /* max_elements */
  write_pod(f, n); /* cur_element_count */
  write_pod(f, size_data_per_element);
  write_pod(f, label_offset);
  write_pod(f, offsetData);
  write_pod(f, maxlevel);
  write_pod(f, enter);
  write_pod(f, maxM);
  write_pod(f, maxM0);
  write_pod(f, M);
  write_pod(f, mult);
  write_pod(f, ef_construction);
  std::vector<char> row(size_data_per_element, 0);
  for (size_t i = 0; i < n; ++i) {
    std::fill(row.begin(), row.end(), 0);
    unsigned int links = 0;
    auto* linkp = reinterpret_cast<unsigned int*>(row.data() + sizeof(unsigned int));
    for (int32_t t = 0; t < hx->degree; ++t) {
      const int32_t nb = hx->graph[i * static_cast<size_t>(hx->degree) + static_cast<size_t>(t)];
      if (nb < 0) continue;
      linkp[links++] = static_cast<unsigned int>(nb);
    }
    std::memcpy(row.data(), &links, sizeof(unsigned int));
    std::memcpy(row.data() + offsetData, hx->ds.x.data() + i * dim, data_size);
    const size_t label = i;
    std::memcpy(row.data() + label_offset, &label, sizeof(size_t));
    f.write(row.data(), static_cast<std::streamsize>(size_data_per_element));
  }
  for (size_t i = 0; i < n; ++i) {
    const unsigned int linkListSize = 0; /* single-layer CAGRA export */
    write_pod(f, linkListSize);
  }
  return f.good() ? IOVS_STATUS_SUCCESS : IOVS_STATUS_IO;
}

iovsStatus iovsHnswDeserialize(iovsResources_t res, const char* path, iovsHnswIndex_t* index) {
  if (!res || !path || !index) return IOVS_STATUS_INVALID_ARGUMENT;
  std::ifstream f(path, std::ios::binary);
  if (!f) return IOVS_STATUS_IO;
  size_t offsetLevel0 = 0, max_elements = 0, cur = 0, size_data = 0, label_off = 0, offsetData = 0;
  int maxlevel = 0;
  unsigned int enter = 0;
  size_t maxM = 0, maxM0 = 0, M = 0, efc = 0;
  double mult = 0;
  read_pod(f, offsetLevel0);
  read_pod(f, max_elements);
  read_pod(f, cur);
  read_pod(f, size_data);
  read_pod(f, label_off);
  read_pod(f, offsetData);
  read_pod(f, maxlevel);
  read_pod(f, enter);
  read_pod(f, maxM);
  read_pod(f, maxM0);
  read_pod(f, M);
  read_pod(f, mult);
  read_pod(f, efc);
  if (!f || cur == 0 || size_data < offsetData) return IOVS_STATUS_IO;
  const size_t dim = (label_off > offsetData) ? (label_off - offsetData) / sizeof(float) : 0;
  if (dim == 0 || dim > 4096) return IOVS_STATUS_IO;
  auto* hx = new HnswIndex();
  hx->ds.n = static_cast<int64_t>(cur);
  hx->ds.dim = static_cast<int64_t>(dim);
  hx->ds.metric = IOVS_METRIC_L2_EXPANDED;
  hx->degree = static_cast<int32_t>(maxM0);
  hx->max_level = maxlevel;
  hx->enter = enter;
  hx->graph.assign(cur * maxM0, -1);
  hx->level.assign(cur, 0);
  hx->ds.x.resize(cur * dim);
  std::vector<char> row(size_data);
  for (size_t i = 0; i < cur; ++i) {
    f.read(row.data(), static_cast<std::streamsize>(size_data));
    if (!f) {
      delete hx;
      return IOVS_STATUS_IO;
    }
    unsigned int links = 0;
    std::memcpy(&links, row.data(), sizeof(unsigned int));
    const auto* linkp = reinterpret_cast<const unsigned int*>(row.data() + sizeof(unsigned int));
    for (unsigned int t = 0; t < links && t < maxM0; ++t) {
      hx->graph[i * maxM0 + t] = static_cast<int32_t>(linkp[t]);
    }
    std::memcpy(hx->ds.x.data() + i * dim, row.data() + offsetData, dim * sizeof(float));
  }
  *index = reinterpret_cast<iovsHnswIndex_t>(hx);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsHnswDestroy(iovsHnswIndex_t index) {
  delete reinterpret_cast<HnswIndex*>(index);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsVamanaBuild(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                           iovsMetric metric, int32_t graph_degree, float alpha,
                           iovsVamanaIndex_t* index) {
  if (!res || !dataset || !index || n <= 1) return IOVS_STATUS_INVALID_ARGUMENT;
  graph_degree = std::max(1, std::min(graph_degree, static_cast<int32_t>(n - 1)));
  if (alpha <= 0.f) alpha = 1.2f;
  auto* ix = new VamanaIndex();
  copy_ds(ix->ds, dataset, n, dim, metric);
  ix->degree = graph_degree;
  std::vector<int32_t> init;
  nndescent_build(*rd(res), dataset, n, dim, metric, graph_degree * 2, 5, init);
  ix->graph.assign(static_cast<size_t>(n) * static_cast<size_t>(graph_degree), -1);
  for (int64_t i = 0; i < n; ++i) {
    std::vector<int32_t> cand;
    for (int32_t t = 0; t < graph_degree * 2; ++t) {
      const int32_t nb = init[static_cast<size_t>(i * (graph_degree * 2) + t)];
      if (nb >= 0) cand.push_back(nb);
    }
    robust_prune(dataset, dim, metric, i, cand, graph_degree, alpha,
                 ix->graph.data() + i * graph_degree);
  }
  *index = reinterpret_cast<iovsVamanaIndex_t>(ix);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsVamanaSearch(iovsResources_t res, iovsVamanaIndex_t index, const float* queries,
                            int64_t nq, int64_t k, int32_t beam, const uint8_t* bitset,
                            int64_t* neighbors, float* distances) {
  if (!res || !index || !queries || !neighbors || !distances) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<VamanaIndex*>(index);
  const float* x = ix->x_view ? ix->x_view : ix->ds.x.data();
  const int32_t* g = ix->graph_view ? ix->graph_view : ix->graph.data();
  return prim_graph_walk(*rd(res), x, ix->ds.n, ix->ds.dim, ix->ds.metric, g, ix->degree, queries, nq, k,
                         std::max(beam, static_cast<int32_t>(k)), 1, bitset, neighbors, distances);
}

constexpr uint32_t kVamanaMagic = 0x314D4156u; /* 'VAM1' */

iovsStatus iovsVamanaSerialize(iovsVamanaIndex_t index, const char* path) {
  if (!index || !path) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<VamanaIndex*>(index);
  std::ofstream f(path, std::ios::binary);
  if (!f) return IOVS_STATUS_IO;
  f.write(reinterpret_cast<const char*>(&kVamanaMagic), 4);
  int32_t ver = 1;
  f.write(reinterpret_cast<const char*>(&ver), 4);
  f.write(reinterpret_cast<const char*>(&ix->ds.n), 8);
  f.write(reinterpret_cast<const char*>(&ix->ds.dim), 8);
  f.write(reinterpret_cast<const char*>(&ix->degree), 4);
  int32_t metric = static_cast<int32_t>(ix->ds.metric);
  f.write(reinterpret_cast<const char*>(&metric), 4);
  const int32_t* g = ix->graph_view ? ix->graph_view : ix->graph.data();
  const float* x = ix->x_view ? ix->x_view : ix->ds.x.data();
  f.write(reinterpret_cast<const char*>(g),
          static_cast<std::streamsize>(static_cast<size_t>(ix->ds.n * ix->degree) * sizeof(int32_t)));
  f.write(reinterpret_cast<const char*>(x),
          static_cast<std::streamsize>(static_cast<size_t>(ix->ds.n * ix->ds.dim) * sizeof(float)));
  return f.good() ? IOVS_STATUS_SUCCESS : IOVS_STATUS_IO;
}

iovsStatus iovsVamanaDeserialize(iovsResources_t res, const char* path, iovsVamanaIndex_t* index) {
  if (!res || !path || !index) return IOVS_STATUS_INVALID_ARGUMENT;
  std::ifstream f(path, std::ios::binary);
  if (!f) return IOVS_STATUS_IO;
  uint32_t magic = 0;
  f.read(reinterpret_cast<char*>(&magic), 4);
  if (magic != kVamanaMagic) return IOVS_STATUS_IO;
  int32_t ver = 0;
  f.read(reinterpret_cast<char*>(&ver), 4);
  auto* ix = new VamanaIndex();
  f.read(reinterpret_cast<char*>(&ix->ds.n), 8);
  f.read(reinterpret_cast<char*>(&ix->ds.dim), 8);
  f.read(reinterpret_cast<char*>(&ix->degree), 4);
  int32_t metric = 0;
  f.read(reinterpret_cast<char*>(&metric), 4);
  ix->ds.metric = static_cast<iovsMetric>(metric);
  ix->graph.resize(static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->degree));
  ix->ds.x.resize(static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->ds.dim));
  f.read(reinterpret_cast<char*>(ix->graph.data()),
         static_cast<std::streamsize>(ix->graph.size() * sizeof(int32_t)));
  f.read(reinterpret_cast<char*>(ix->ds.x.data()),
         static_cast<std::streamsize>(ix->ds.x.size() * sizeof(float)));
  if (!f) {
    delete ix;
    return IOVS_STATUS_IO;
  }
  *index = reinterpret_cast<iovsVamanaIndex_t>(ix);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsVamanaMmap(iovsResources_t res, const char* path, iovsVamanaIndex_t* index) {
  if (!res || !path || !index) return IOVS_STATUS_INVALID_ARGUMENT;
#ifdef _WIN32
  HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                          FILE_ATTRIBUTE_NORMAL, nullptr);
  if (fh == INVALID_HANDLE_VALUE) return IOVS_STATUS_IO;
  LARGE_INTEGER sz;
  if (!GetFileSizeEx(fh, &sz) || sz.QuadPart < 32) {
    CloseHandle(fh);
    return IOVS_STATUS_IO;
  }
  HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!mh) {
    CloseHandle(fh);
    return IOVS_STATUS_IO;
  }
  void* view = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
  if (!view) {
    CloseHandle(mh);
    CloseHandle(fh);
    return IOVS_STATUS_IO;
  }
  const auto* p = static_cast<const uint8_t*>(view);
  uint32_t magic = 0;
  std::memcpy(&magic, p, 4);
  if (magic != kVamanaMagic) {
    UnmapViewOfFile(view);
    CloseHandle(mh);
    CloseHandle(fh);
    return IOVS_STATUS_IO;
  }
  auto* ix = new VamanaIndex();
  std::memcpy(&ix->ds.n, p + 8, 8);
  std::memcpy(&ix->ds.dim, p + 16, 8);
  std::memcpy(&ix->degree, p + 24, 4);
  int32_t metric = 0;
  std::memcpy(&metric, p + 28, 4);
  ix->ds.metric = static_cast<iovsMetric>(metric);
  const size_t gsz = static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->degree) * sizeof(int32_t);
  ix->graph_view = reinterpret_cast<const int32_t*>(p + 32);
  ix->x_view = reinterpret_cast<const float*>(p + 32 + gsz);
  ix->mmapped = true;
  ix->file_handle = fh;
  ix->map_handle = mh;
  ix->view = view;
  *index = reinterpret_cast<iovsVamanaIndex_t>(ix);
  return IOVS_STATUS_SUCCESS;
#else
  int fd = open(path, O_RDONLY);
  if (fd < 0) return IOVS_STATUS_IO;
  struct stat st {};
  if (fstat(fd, &st) != 0 || st.st_size < 32) {
    close(fd);
    return IOVS_STATUS_IO;
  }
  void* view = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
  if (view == MAP_FAILED) {
    close(fd);
    return IOVS_STATUS_IO;
  }
  const auto* p = static_cast<const uint8_t*>(view);
  uint32_t magic = 0;
  std::memcpy(&magic, p, 4);
  if (magic != kVamanaMagic) {
    munmap(view, static_cast<size_t>(st.st_size));
    close(fd);
    return IOVS_STATUS_IO;
  }
  auto* ix = new VamanaIndex();
  std::memcpy(&ix->ds.n, p + 8, 8);
  std::memcpy(&ix->ds.dim, p + 16, 8);
  std::memcpy(&ix->degree, p + 24, 4);
  int32_t metric = 0;
  std::memcpy(&metric, p + 28, 4);
  ix->ds.metric = static_cast<iovsMetric>(metric);
  const size_t gsz = static_cast<size_t>(ix->ds.n) * static_cast<size_t>(ix->degree) * sizeof(int32_t);
  ix->graph_view = reinterpret_cast<const int32_t*>(p + 32);
  ix->x_view = reinterpret_cast<const float*>(p + 32 + gsz);
  ix->mmapped = true;
  ix->fd = fd;
  ix->view = view;
  ix->map_size = static_cast<size_t>(st.st_size);
  *index = reinterpret_cast<iovsVamanaIndex_t>(ix);
  return IOVS_STATUS_SUCCESS;
#endif
}

iovsStatus iovsVamanaDestroy(iovsVamanaIndex_t index) {
  auto* ix = reinterpret_cast<VamanaIndex*>(index);
  if (!ix) return IOVS_STATUS_SUCCESS;
#ifdef _WIN32
  if (ix->view) UnmapViewOfFile(ix->view);
  if (ix->map_handle) CloseHandle(ix->map_handle);
  if (ix->file_handle != INVALID_HANDLE_VALUE) CloseHandle(ix->file_handle);
#else
  if (ix->view && ix->view != MAP_FAILED) munmap(ix->view, ix->map_size);
  if (ix->fd >= 0) close(ix->fd);
#endif
  delete ix;
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsScannBuild(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                          iovsMetric metric, int32_t nlist, int32_t pq_m, iovsScannIndex_t* index) {
  if (!res || !dataset || !index) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* ix = new ScannIndex();
  ix->aniso.resize(static_cast<size_t>(dim), 1.f);
  std::vector<float> mean(static_cast<size_t>(dim), 0.f);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t d = 0; d < dim; ++d) mean[static_cast<size_t>(d)] += dataset[i * dim + d];
  for (int64_t d = 0; d < dim; ++d) mean[static_cast<size_t>(d)] /= static_cast<float>(n);
  for (int64_t d = 0; d < dim; ++d) {
    float v = 0.f;
    for (int64_t i = 0; i < n; ++i) {
      const float t = dataset[i * dim + d] - mean[static_cast<size_t>(d)];
      v += t * t;
    }
    ix->aniso[static_cast<size_t>(d)] = std::sqrt(v / static_cast<float>(std::max<int64_t>(n, 1)) + 1e-6f);
  }
  std::vector<float> scaled(static_cast<size_t>(n) * static_cast<size_t>(dim));
  for (int64_t i = 0; i < n; ++i)
    for (int64_t d = 0; d < dim; ++d)
      scaled[static_cast<size_t>(i * dim + d)] =
          dataset[i * dim + d] * ix->aniso[static_cast<size_t>(d)];
  const iovsStatus st = iovsIvfPqBuild(res, scaled.data(), n, dim, metric, nlist, pq_m, 8, &ix->pq);
  if (st != IOVS_STATUS_SUCCESS) {
    delete ix;
    return st;
  }
  *index = reinterpret_cast<iovsScannIndex_t>(ix);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsScannSearch(iovsResources_t res, iovsScannIndex_t index, const float* queries,
                           int64_t nq, int64_t k, int32_t nprobe, int32_t krefine,
                           int64_t* neighbors, float* distances) {
  if (!res || !index || !queries) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* ix = reinterpret_cast<ScannIndex*>(index);
  const int64_t dim = static_cast<int64_t>(ix->aniso.size());
  std::vector<float> q(static_cast<size_t>(nq) * static_cast<size_t>(dim));
  for (int64_t i = 0; i < nq; ++i)
    for (int64_t d = 0; d < dim; ++d)
      q[static_cast<size_t>(i * dim + d)] = queries[i * dim + d] * ix->aniso[static_cast<size_t>(d)];
  return iovsIvfPqSearch(res, ix->pq, q.data(), nq, k, nprobe, krefine, nullptr, neighbors, distances);
}

iovsStatus iovsScannDestroy(iovsScannIndex_t index) {
  auto* ix = reinterpret_cast<ScannIndex*>(index);
  if (ix) {
    iovsIvfPqDestroy(ix->pq);
    delete ix;
  }
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsAllNeighbors(iovsResources_t res, const float* dataset, int64_t n, int64_t dim,
                            iovsMetric metric, int32_t k, int64_t* neighbors, float* distances) {
  if (!res || !dataset || !neighbors || !distances || n <= 1 || k <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  k = std::min(k, static_cast<int32_t>(n - 1));
  std::vector<float> scores(static_cast<size_t>(n) * static_cast<size_t>(n));
  prim_pairwise(*rd(res), metric, dataset, n, dataset, n, dim, scores.data(), 2.f);
  for (int64_t i = 0; i < n; ++i) scores[static_cast<size_t>(i * n + i)] = kInf;
  prim_topk(*rd(res), scores.data(), n, n, k, neighbors, distances, metric_largest(metric));
  if (metric == IOVS_METRIC_INNER_PRODUCT) {
    for (int64_t i = 0; i < n * k; ++i) distances[i] = -distances[i];
  }
  return IOVS_STATUS_SUCCESS;
}
