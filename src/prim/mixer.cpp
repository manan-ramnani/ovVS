#include "internal.hpp"

#include <cstring>

namespace ovvs {
namespace impl {

static ovvsDevice policy_force(ovvsPolicy p) {
  switch (p) {
    case OVVS_POLICY_FORCE_NPU:
      return OVVS_DEVICE_NPU;
    case OVVS_POLICY_FORCE_GPU:
      return OVVS_DEVICE_GPU;
    case OVVS_POLICY_FORCE_CPU:
      return OVVS_DEVICE_CPU;
    default:
      return OVVS_DEVICE_AUTO;
  }
}

ovvsDevice choose_device(ResourcesData& r, const char* op, int64_t flops_or_elems) {
  const ovvsDevice forced = policy_force(r.policy);
  if (forced != OVVS_DEVICE_AUTO) return forced;

  /* TopK / Gather: CPU wins every Arrow Lake bakeoff (launch tax). FORCE_* still
     takes the branch above. See tables/arrow-lake/topk_large.json, gather_large.json. */
  if (std::strcmp(op, "topk") == 0 || std::strcmp(op, "gather") == 0) return OVVS_DEVICE_CPU;

  const bool gemm_like = std::strcmp(op, "gemm") == 0 || std::strcmp(op, "pairwise") == 0;
  if (!gemm_like) return OVVS_DEVICE_CPU;

  /* Dense GEMM: oneMKL cblas_sgemm wins tiny through 1e5×32×768 on Arrow Lake
     once CPU is not a triple loop. NPU DPU is faster in isolation; the Parameter
     DMA tax is not. Honour gemm_large.json when the shape is in that class so a
     SKU where NPU/GPU actually wins can flip AUTO. */
  if (flops_or_elems >= r.large_gemm_flops && r.large_gemm_winner != OVVS_DEVICE_AUTO) {
    if (r.large_gemm_winner == OVVS_DEVICE_GPU && r.gpu_available) return OVVS_DEVICE_GPU;
    if (r.large_gemm_winner == OVVS_DEVICE_NPU && r.npu_available && !r.npu_busy) return OVVS_DEVICE_NPU;
    if (r.large_gemm_winner == OVVS_DEVICE_CPU) return OVVS_DEVICE_CPU;
  }
  return OVVS_DEVICE_CPU;
}

static ovvsStatus finish_forced_fail(ResourcesData& r) {
  ++r.npu_fallbacks;
  return OVVS_STATUS_DEVICE_UNAVAILABLE;
}

ovvsStatus prim_gemm_compute(ResourcesData& r, const float* a, const float* b, float* c, int64_t m,
                             int64_t n, int64_t k, bool trans_b, ovvsDType compute) {
  r.last_compute_dtype = OVVS_DTYPE_F32;
  const ovvsDevice forced = policy_force(r.policy);
  auto try_npu = [&]() {
    if (npu_gemm_compute(r, compute, a, b, c, m, n, k, trans_b)) {
      r.last_device = OVVS_DEVICE_NPU;
      return true;
    }
    return false;
  };
  auto try_gpu = [&]() {
    if (gpu_gemm_compute(r, compute, a, b, c, m, n, k, trans_b)) {
      r.last_device = OVVS_DEVICE_GPU;
      return true;
    }
    return false;
  };
  if (forced == OVVS_DEVICE_NPU) {
    if (try_npu()) return OVVS_STATUS_SUCCESS;
    return finish_forced_fail(r);
  }
  if (forced == OVVS_DEVICE_GPU) {
    if (try_gpu()) return OVVS_STATUS_SUCCESS;
    return finish_forced_fail(r);
  }
  if (forced == OVVS_DEVICE_CPU) {
    cpu_gemm(a, b, c, m, n, k, trans_b);
    r.last_device = OVVS_DEVICE_CPU;
    return OVVS_STATUS_SUCCESS;
  }
  /* AUTO: same size ladder as f32. On Arrow Lake that is CPU oneMKL for dense GEMM
     (F16/I8 XMX and NPU FQ lose to cblas_sgemm). FORCE_* still above. */
  const ovvsDevice d = choose_device(r, "gemm", m * n * k);
  if (d == OVVS_DEVICE_NPU && try_npu()) return OVVS_STATUS_SUCCESS;
  if ((d == OVVS_DEVICE_GPU || d == OVVS_DEVICE_NPU) && try_gpu()) return OVVS_STATUS_SUCCESS;
  cpu_gemm(a, b, c, m, n, k, trans_b);
  r.last_device = OVVS_DEVICE_CPU;
  r.last_compute_dtype = OVVS_DTYPE_F32;
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus prim_gemm(ResourcesData& r, const float* a, const float* b, float* c, int64_t m, int64_t n,
                     int64_t k, bool trans_b) {
  r.last_compute_dtype = OVVS_DTYPE_F32;
  const ovvsDevice d = choose_device(r, "gemm", m * n * k);
  if (d == OVVS_DEVICE_NPU) {
    if (npu_gemm(r, a, b, c, m, n, k, trans_b)) {
      r.last_device = OVVS_DEVICE_NPU;
      return OVVS_STATUS_SUCCESS;
    }
    if (r.policy == OVVS_POLICY_FORCE_NPU) return finish_forced_fail(r);
  }
  if (d == OVVS_DEVICE_GPU || (d == OVVS_DEVICE_NPU && r.policy != OVVS_POLICY_FORCE_CPU)) {
    if (gpu_gemm(r, a, b, c, m, n, k, trans_b)) {
      r.last_device = OVVS_DEVICE_GPU;
      return OVVS_STATUS_SUCCESS;
    }
    if (r.policy == OVVS_POLICY_FORCE_GPU) return finish_forced_fail(r);
  }
  if (r.policy == OVVS_POLICY_FORCE_NPU || r.policy == OVVS_POLICY_FORCE_GPU) {
    return finish_forced_fail(r);
  }
  cpu_gemm(a, b, c, m, n, k, trans_b);
  r.last_device = OVVS_DEVICE_CPU;
  r.last_compute_dtype = OVVS_DTYPE_F32;
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus prim_topk(ResourcesData& r, const float* scores, int64_t rows, int64_t cols, int64_t k,
                     int64_t* indices, float* values, bool largest) {
  const ovvsDevice d = choose_device(r, "topk", rows * cols);
  if (d == OVVS_DEVICE_NPU) {
    if (npu_topk(r, scores, rows, cols, k, indices, values, largest)) {
      r.last_device = OVVS_DEVICE_NPU;
      return OVVS_STATUS_SUCCESS;
    }
    if (r.policy == OVVS_POLICY_FORCE_NPU) return finish_forced_fail(r);
  }
  if (d == OVVS_DEVICE_GPU || (d == OVVS_DEVICE_NPU && r.policy != OVVS_POLICY_FORCE_CPU)) {
    if (gpu_topk(r, scores, rows, cols, k, indices, values, largest)) {
      r.last_device = OVVS_DEVICE_GPU;
      return OVVS_STATUS_SUCCESS;
    }
    if (r.policy == OVVS_POLICY_FORCE_GPU) return finish_forced_fail(r);
  }
  if (r.policy == OVVS_POLICY_FORCE_NPU || r.policy == OVVS_POLICY_FORCE_GPU) {
    return finish_forced_fail(r);
  }
  cpu_topk(scores, rows, cols, k, indices, values, largest);
  r.last_device = OVVS_DEVICE_CPU;
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus prim_gather_rows(ResourcesData& r, const float* src, int64_t src_rows, int64_t dim,
                            const int64_t* idx, int64_t nidx, float* out) {
  const ovvsDevice d = choose_device(r, "gather", nidx * dim);
  if (d == OVVS_DEVICE_NPU) {
    if (npu_gather_rows(r, src, src_rows, dim, idx, nidx, out)) {
      r.last_device = OVVS_DEVICE_NPU;
      return OVVS_STATUS_SUCCESS;
    }
    if (r.policy == OVVS_POLICY_FORCE_NPU) return finish_forced_fail(r);
  }
  if (d == OVVS_DEVICE_GPU || (d == OVVS_DEVICE_NPU && r.policy != OVVS_POLICY_FORCE_CPU)) {
    if (gpu_gather_rows(r, src, src_rows, dim, idx, nidx, out)) {
      r.last_device = OVVS_DEVICE_GPU;
      return OVVS_STATUS_SUCCESS;
    }
    if (r.policy == OVVS_POLICY_FORCE_GPU) return finish_forced_fail(r);
  }
  if (r.policy == OVVS_POLICY_FORCE_NPU || r.policy == OVVS_POLICY_FORCE_GPU) {
    return finish_forced_fail(r);
  }
  cpu_gather_rows(src, src_rows, dim, idx, nidx, out);
  r.last_device = OVVS_DEVICE_CPU;
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus prim_pairwise(ResourcesData& r, ovvsMetric metric, const float* x, int64_t nx,
                         const float* y, int64_t ny, int64_t dim, float* out, float metric_arg) {
  if (metric == OVVS_METRIC_L2_EXPANDED || metric == OVVS_METRIC_INNER_PRODUCT ||
      metric == OVVS_METRIC_COSINE_EXPANDED) {
    std::vector<float> xnorm(static_cast<size_t>(nx)), ynorm(static_cast<size_t>(ny));
    for (int64_t i = 0; i < nx; ++i) xnorm[static_cast<size_t>(i)] = nrm2sq(x + i * dim, dim);
    for (int64_t j = 0; j < ny; ++j) ynorm[static_cast<size_t>(j)] = nrm2sq(y + j * dim, dim);
    const ovvsStatus gs = prim_gemm(r, x, y, out, nx, ny, dim, true);
    if (gs != OVVS_STATUS_SUCCESS) return gs;
    for (int64_t i = 0; i < nx; ++i) {
      for (int64_t j = 0; j < ny; ++j) {
        const float ip = out[i * ny + j];
        if (metric == OVVS_METRIC_INNER_PRODUCT) {
          out[i * ny + j] = -ip;
        } else if (metric == OVVS_METRIC_COSINE_EXPANDED) {
          const float nxv = std::sqrt(std::max(xnorm[static_cast<size_t>(i)], 1e-12f));
          const float nyv = std::sqrt(std::max(ynorm[static_cast<size_t>(j)], 1e-12f));
          out[i * ny + j] = 1.f - ip / (nxv * nyv);
        } else {
          out[i * ny + j] = xnorm[static_cast<size_t>(i)] + ynorm[static_cast<size_t>(j)] - 2.f * ip;
        }
      }
    }
    return OVVS_STATUS_SUCCESS;
  }
  const ovvsDevice d = choose_device(r, "pairwise", nx * ny * dim);
  if (d == OVVS_DEVICE_NPU) {
    if (npu_pairwise(r, metric, x, nx, y, ny, dim, out, metric_arg)) {
      r.last_device = OVVS_DEVICE_NPU;
      return OVVS_STATUS_SUCCESS;
    }
    if (r.policy == OVVS_POLICY_FORCE_NPU) return finish_forced_fail(r);
  }
  if (d == OVVS_DEVICE_GPU || (d == OVVS_DEVICE_NPU && r.policy != OVVS_POLICY_FORCE_CPU)) {
    if (gpu_pairwise(r, metric, x, nx, y, ny, dim, out, metric_arg)) {
      r.last_device = OVVS_DEVICE_GPU;
      return OVVS_STATUS_SUCCESS;
    }
    if (r.policy == OVVS_POLICY_FORCE_GPU) return finish_forced_fail(r);
  }
  if (r.policy == OVVS_POLICY_FORCE_NPU || r.policy == OVVS_POLICY_FORCE_GPU) {
    return finish_forced_fail(r);
  }
  cpu_pairwise(metric, x, nx, y, ny, dim, out, metric_arg);
  r.last_device = OVVS_DEVICE_CPU;
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus prim_pq_adc(ResourcesData& r, const float* tables, int32_t pq_m, int32_t ks,
                       const uint8_t* codes, int64_t ncodes, float* out) {
  if (!tables || !codes || !out || pq_m <= 0 || ks <= 0 || ncodes <= 0) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  if (r.policy != OVVS_POLICY_FORCE_CPU) {
    if (npu_pq_adc(r, tables, pq_m, ks, codes, ncodes, out)) {
      r.last_device = OVVS_DEVICE_NPU;
      return OVVS_STATUS_SUCCESS;
    }
    if (r.policy == OVVS_POLICY_FORCE_NPU) return finish_forced_fail(r);
  }
  if (r.policy == OVVS_POLICY_FORCE_NPU || r.policy == OVVS_POLICY_FORCE_GPU) {
    return finish_forced_fail(r);
  }
  for (int64_t i = 0; i < ncodes; ++i) {
    out[i] = ovvsShavePqAdc(tables, codes + i * pq_m, pq_m, ks);
  }
  r.last_device = OVVS_DEVICE_CPU;
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus prim_graph_walk(ResourcesData& r, const float* dataset, int64_t n, int64_t dim,
                           ovvsMetric metric, const int32_t* graph, int32_t degree, const float* queries,
                           int64_t nq, int64_t k, int32_t itopk, int32_t search_width,
                           const uint8_t* bitset, int64_t* neighbors, float* distances) {
  if (r.policy != OVVS_POLICY_FORCE_CPU && r.policy != OVVS_POLICY_FORCE_NPU) {
    if (gpu_cagra_walk(r, dataset, n, dim, metric, graph, degree, queries, nq, k, itopk, search_width,
                       bitset, neighbors, distances)) {
      r.last_device = OVVS_DEVICE_GPU;
      return OVVS_STATUS_SUCCESS;
    }
    if (r.policy == OVVS_POLICY_FORCE_GPU && !gpu_available()) return OVVS_STATUS_DEVICE_UNAVAILABLE;
  }

  itopk = std::max(itopk, static_cast<int32_t>(k));
  search_width = std::max(1, search_width);
  struct Node {
    float d;
    int64_t id;
  };
  auto score_ids = [&](const float* query, const std::vector<int64_t>& ids, std::vector<float>& sc) -> ovvsStatus {
    sc.resize(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
      const int64_t id = ids[i];
      if (id < 0 || id >= n) {
        sc[i] = kInf;
        continue;
      }
      sc[i] = distance_one(metric, query, dataset + id * dim, dim, 2.f);
    }
    return OVVS_STATUS_SUCCESS;
  };
  for (int64_t q = 0; q < nq; ++q) {
    const float* query = queries + q * dim;
    std::vector<uint8_t> seen(static_cast<size_t>(n), 0);
    std::vector<char> expanded(static_cast<size_t>(n), 0);
    std::vector<Node> cand;
    std::vector<int64_t> seed_ids;
    const int64_t nseeds = cagra_seed_count(n, itopk, search_width);
    for (int64_t s = 0; s < nseeds; ++s) {
      int64_t id = (s * 9973 + q * 13) % n;
      if (!allowed(bitset, id) || seen[static_cast<size_t>(id)]) continue;
      seen[static_cast<size_t>(id)] = 1;
      seed_ids.push_back(id);
    }
    std::vector<float> seed_sc;
    const ovvsStatus ss = score_ids(query, seed_ids, seed_sc);
    if (ss != OVVS_STATUS_SUCCESS) return ss;
    for (size_t i = 0; i < seed_ids.size(); ++i) cand.push_back({seed_sc[i], seed_ids[i]});
    if (static_cast<int32_t>(cand.size()) > itopk) {
      std::nth_element(cand.begin(), cand.begin() + itopk, cand.end(),
                       [](const Node& a, const Node& b) { return a.d < b.d; });
      cand.resize(static_cast<size_t>(itopk));
    }
    if (cand.empty()) {
      for (int64_t t = 0; t < k; ++t) {
        neighbors[q * k + t] = -1;
        distances[q * k + t] = kInf;
      }
      continue;
    }
    int iters = 0;
    const int max_iters = std::max(24, itopk * 6);
    while (iters++ < max_iters) {
      std::vector<int64_t> picks;
      picks.reserve(static_cast<size_t>(search_width));
      for (int s = 0; s < search_width; ++s) {
        int64_t pick = -1;
        float pick_d = kInf;
        for (const auto& c : cand) {
          if (expanded[static_cast<size_t>(c.id)]) continue;
          bool already = false;
          for (int64_t p : picks) {
            if (p == c.id) {
              already = true;
              break;
            }
          }
          if (already) continue;
          if (c.d < pick_d) {
            pick_d = c.d;
            pick = c.id;
          }
        }
        if (pick < 0) break;
        picks.push_back(pick);
      }
      if (picks.empty()) break;
      std::vector<int64_t> batch;
      for (int64_t pick : picks) {
        expanded[static_cast<size_t>(pick)] = 1;
        const int32_t* nbrs = graph + pick * degree;
        for (int32_t e = 0; e < degree; ++e) {
          const int32_t nb = nbrs[e];
          if (nb < 0 || static_cast<int64_t>(nb) >= n) continue;
          if (seen[static_cast<size_t>(nb)]) continue;
          if (!allowed(bitset, nb)) continue;
          seen[static_cast<size_t>(nb)] = 1;
          batch.push_back(nb);
        }
      }
      std::vector<float> bsc;
      const ovvsStatus bs = score_ids(query, batch, bsc);
      if (bs != OVVS_STATUS_SUCCESS) return bs;
      for (size_t i = 0; i < batch.size(); ++i) cand.push_back({bsc[i], batch[i]});
      const int32_t keep = itopk * 2;
      if (static_cast<int32_t>(cand.size()) > keep) {
        std::nth_element(cand.begin(), cand.begin() + keep, cand.end(),
                         [](const Node& a, const Node& b) { return a.d < b.d; });
        cand.resize(static_cast<size_t>(keep));
      }
    }
    std::sort(cand.begin(), cand.end(), [](const Node& a, const Node& b) { return a.d < b.d; });
    for (int64_t t = 0; t < k; ++t) {
      if (t < static_cast<int64_t>(cand.size())) {
        neighbors[q * k + t] = cand[static_cast<size_t>(t)].id;
        float d = cand[static_cast<size_t>(t)].d;
        if (metric == OVVS_METRIC_INNER_PRODUCT) d = -d;
        distances[q * k + t] = d;
      } else {
        neighbors[q * k + t] = -1;
        distances[q * k + t] = kInf;
      }
    }
  }
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus brute_search_impl(ResourcesData& r, const float* dataset, int64_t n, int64_t dim,
                             const float* queries, int64_t nq, ovvsMetric metric, int64_t k,
                             const uint8_t* bitset, int64_t* neighbors, float* distances) {
  std::vector<float> scores(static_cast<size_t>(nq * n));
  const ovvsStatus ps = prim_pairwise(r, metric, queries, nq, dataset, n, dim, scores.data(), 2.f);
  if (ps != OVVS_STATUS_SUCCESS) return ps;
  if (bitset) {
    for (int64_t i = 0; i < nq; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        if (!allowed(bitset, j)) {
          scores[static_cast<size_t>(i * n + j)] = kInf;
        }
      }
    }
  }
  const ovvsStatus ts = prim_topk(r, scores.data(), nq, n, k, neighbors, distances, metric_largest(metric));
  if (ts != OVVS_STATUS_SUCCESS) return ts;
  if (metric == OVVS_METRIC_INNER_PRODUCT) {
    for (int64_t i = 0; i < nq * k; ++i) distances[i] = -distances[i];
  }
  return OVVS_STATUS_SUCCESS;
}

void kmeans_fit_impl(ResourcesData& r, const float* x, int64_t n, int64_t dim, int32_t k,
                     int32_t iters, std::vector<float>& centroids) {
  k = std::max(1, std::min(k, static_cast<int32_t>(n)));
  centroids.assign(static_cast<size_t>(k) * static_cast<size_t>(dim), 0.f);
  auto rng = rng_from(42);
  std::uniform_int_distribution<int64_t> pick(0, n - 1);
  std::vector<int64_t> used;
  for (int32_t c = 0; c < k; ++c) {
    int64_t i = pick(rng);
    for (int t = 0; t < 8 && std::find(used.begin(), used.end(), i) != used.end(); ++t) i = pick(rng);
    used.push_back(i);
    std::memcpy(centroids.data() + static_cast<size_t>(c) * dim, x + i * dim,
                static_cast<size_t>(dim) * sizeof(float));
  }
  std::vector<int64_t> labels(static_cast<size_t>(n));
  std::vector<float> dist(static_cast<size_t>(n));
  std::vector<float> scores(static_cast<size_t>(n) * static_cast<size_t>(k));
  for (int it = 0; it < iters; ++it) {
    prim_pairwise(r, OVVS_METRIC_L2_EXPANDED, x, n, centroids.data(), k, dim, scores.data(), 2.f);
    prim_topk(r, scores.data(), n, k, 1, labels.data(), dist.data(), false);
    std::vector<float> sum(static_cast<size_t>(k) * static_cast<size_t>(dim), 0.f);
    std::vector<int32_t> cnt(static_cast<size_t>(k), 0);
    for (int64_t i = 0; i < n; ++i) {
      const int64_t c = labels[static_cast<size_t>(i)];
      if (c < 0 || c >= k) continue;
      ++cnt[static_cast<size_t>(c)];
      float* dst = sum.data() + c * dim;
      const float* src = x + i * dim;
      for (int64_t d = 0; d < dim; ++d) dst[d] += src[d];
    }
    for (int32_t c = 0; c < k; ++c) {
      if (cnt[static_cast<size_t>(c)] == 0) continue;
      float* dst = centroids.data() + static_cast<size_t>(c) * dim;
      const float* src = sum.data() + static_cast<size_t>(c) * dim;
      const float inv = 1.f / static_cast<float>(cnt[static_cast<size_t>(c)]);
      for (int64_t d = 0; d < dim; ++d) dst[d] = src[d] * inv;
    }
  }
}

}  // namespace impl
}  // namespace ovvs
