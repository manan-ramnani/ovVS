#include "internal.hpp"

#include <atomic>
#include <cstring>
#include <thread>

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
  if (r.policy == OVVS_POLICY_FORCE_NPU) ++r.npu_fallbacks;
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

ovvsStatus prim_gemm(ResourcesData& r, const float* a, const float* b, float* c,
                     int64_t m, int64_t n, int64_t k, bool trans_b,
                     GpuWorkStats* stats) {
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
    if (gpu_gemm(r, a, b, c, m, n, k, trans_b, stats)) {
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
                     int64_t* indices, float* values, bool largest,
                     GpuWorkStats* stats) {
  const ovvsDevice d = choose_device(r, "topk", rows * cols);
  if (d == OVVS_DEVICE_NPU) {
    if (npu_topk(r, scores, rows, cols, k, indices, values, largest)) {
      r.last_device = OVVS_DEVICE_NPU;
      return OVVS_STATUS_SUCCESS;
    }
    if (r.policy == OVVS_POLICY_FORCE_NPU) return finish_forced_fail(r);
  }
  if (d == OVVS_DEVICE_GPU || (d == OVVS_DEVICE_NPU && r.policy != OVVS_POLICY_FORCE_CPU)) {
    if (gpu_topk(r, scores, rows, cols, k, indices, values, largest, stats)) {
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
                            const int64_t* idx, int64_t nidx, float* out,
                            GpuWorkStats* stats) {
  const ovvsDevice d = choose_device(r, "gather", nidx * dim);
  if (d == OVVS_DEVICE_NPU) {
    if (npu_gather_rows(r, src, src_rows, dim, idx, nidx, out)) {
      r.last_device = OVVS_DEVICE_NPU;
      return OVVS_STATUS_SUCCESS;
    }
    if (r.policy == OVVS_POLICY_FORCE_NPU) return finish_forced_fail(r);
  }
  if (d == OVVS_DEVICE_GPU || (d == OVVS_DEVICE_NPU && r.policy != OVVS_POLICY_FORCE_CPU)) {
    if (gpu_gather_rows(r, src, src_rows, dim, idx, nidx, out, stats)) {
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
                         const float* y, int64_t ny, int64_t dim, float* out,
                         float metric_arg, GpuWorkStats* stats) {
  if (metric == OVVS_METRIC_L2_EXPANDED || metric == OVVS_METRIC_INNER_PRODUCT ||
      metric == OVVS_METRIC_COSINE_EXPANDED) {
    std::vector<float> xnorm(static_cast<size_t>(nx)), ynorm(static_cast<size_t>(ny));
    for (int64_t i = 0; i < nx; ++i) xnorm[static_cast<size_t>(i)] = nrm2sq(x + i * dim, dim);
    for (int64_t j = 0; j < ny; ++j) ynorm[static_cast<size_t>(j)] = nrm2sq(y + j * dim, dim);
    const ovvsStatus gs = prim_gemm(r, x, y, out, nx, ny, dim, true, stats);
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
    if (gpu_pairwise(r, metric, x, nx, y, ny, dim, out, metric_arg, stats)) {
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

int32_t pq_adc_bucket_rows(int64_t remaining) {
  if (remaining <= 0) return 0;
  if (remaining <= 128) return 128;
  if (remaining <= 256) return 256;
  if (remaining <= 512) return 512;
  if (remaining <= 1024) return 1024;
  return 2048;
}

ovvsStatus plan_pq_adc_chunks(const PqAdcTask* tasks, int64_t task_count,
                              int32_t pq_m, int64_t output_rows,
                              std::vector<PqAdcChunk>& chunks) {
  chunks.clear();
  if (!tasks || task_count <= 0 || pq_m <= 0 || output_rows <= 0) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  try {
    std::vector<PqAdcChunk> planned;
    int64_t expected_output = 0;
    for (int64_t task_index = 0; task_index < task_count; ++task_index) {
      const PqAdcTask& task = tasks[task_index];
      if (!task.tables || !task.codes || task.rows <= 0) {
        return OVVS_STATUS_INVALID_ARGUMENT;
      }
      if (task.output_offset != expected_output || task.rows > output_rows - expected_output) {
        return OVVS_STATUS_SHAPE_MISMATCH;
      }
      if (static_cast<uint64_t>(task.rows) >
          static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) /
              static_cast<uint64_t>(pq_m)) {
        return OVVS_STATUS_SHAPE_MISMATCH;
      }
      int64_t consumed = 0;
      while (consumed < task.rows) {
        const int64_t remaining = task.rows - consumed;
        const int32_t bucket_rows = pq_adc_bucket_rows(remaining);
        if (bucket_rows <= 0) return OVVS_STATUS_SHAPE_MISMATCH;
        const int32_t valid_rows = static_cast<int32_t>(
            std::min<int64_t>(remaining, static_cast<int64_t>(bucket_rows)));
        if (planned.size() == planned.max_size()) return OVVS_STATUS_SHAPE_MISMATCH;
        const size_t code_offset =
            static_cast<size_t>(consumed) * static_cast<size_t>(pq_m);
        planned.push_back({task.tables, task.codes + code_offset, valid_rows, bucket_rows,
                           task.output_offset + consumed});
        consumed += valid_rows;
      }
      expected_output += task.rows;
    }
    if (expected_output != output_rows) return OVVS_STATUS_SHAPE_MISMATCH;
    chunks.swap(planned);
    return OVVS_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return OVVS_STATUS_OOM;
  } catch (const std::length_error&) {
    return OVVS_STATUS_SHAPE_MISMATCH;
  } catch (...) {
    return OVVS_STATUS_ERROR;
  }
}

namespace {

void pq_adc_saturating_add(int64_t& value, int64_t delta) noexcept {
  if (delta <= 0) return;
  if (value > std::numeric_limits<int64_t>::max() - delta) {
    value = std::numeric_limits<int64_t>::max();
  } else {
    value += delta;
  }
}

void record_pq_adc_batch(ResourcesData& r, int64_t task_count, int64_t chunk_count,
                         int64_t valid_rows, int64_t padded_rows) noexcept {
  try {
    std::lock_guard<std::mutex> lock(r.pq_adc_stats_mutex);
    pq_adc_saturating_add(r.pq_adc_calls, 1);
    pq_adc_saturating_add(r.pq_adc_logical_tasks, task_count);
    pq_adc_saturating_add(r.pq_adc_chunks, chunk_count);
    pq_adc_saturating_add(r.pq_adc_valid_rows, valid_rows);
    pq_adc_saturating_add(r.pq_adc_padded_rows, padded_rows);
  } catch (...) {
  }
}

void record_pq_adc_cpu_rows(ResourcesData& r, int64_t rows) noexcept {
  try {
    std::lock_guard<std::mutex> lock(r.pq_adc_stats_mutex);
    pq_adc_saturating_add(r.pq_adc_cpu_rows, rows);
  } catch (...) {
  }
}

}  // namespace

ovvsStatus prim_pq_adc_batch(ResourcesData& r, const PqAdcTask* tasks, int64_t task_count,
                             int32_t pq_m, int32_t ks, float* out, int64_t output_rows) {
  if (!tasks || !out || task_count <= 0 || pq_m <= 0 || ks <= 0 || output_rows <= 0) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  if (pq_m > std::numeric_limits<int32_t>::max() / ks ||
      static_cast<uint64_t>(pq_m) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
              static_cast<uint64_t>(ks) ||
      static_cast<uint64_t>(output_rows) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / sizeof(float)) {
    return OVVS_STATUS_SHAPE_MISMATCH;
  }

  try {
    std::vector<PqAdcChunk> chunks;
    ovvsStatus status = plan_pq_adc_chunks(tasks, task_count, pq_m, output_rows, chunks);
    if (status != OVVS_STATUS_SUCCESS) return status;

    int64_t padded_rows = 0;
    for (int64_t task_index = 0; task_index < task_count; ++task_index) {
      const PqAdcTask& task = tasks[task_index];
      if (static_cast<uint64_t>(task.rows) >
          static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) /
              static_cast<uint64_t>(pq_m)) {
        return OVVS_STATUS_SHAPE_MISMATCH;
      }
      const size_t code_elements =
          static_cast<size_t>(task.rows) * static_cast<size_t>(pq_m);
      for (size_t element = 0; element < code_elements; ++element) {
        if (static_cast<int32_t>(task.codes[element]) >= ks) {
          return OVVS_STATUS_INVALID_ARGUMENT;
        }
      }
    }
    for (const PqAdcChunk& chunk : chunks) {
      pq_adc_saturating_add(
          padded_rows,
          static_cast<int64_t>(chunk.bucket_rows) - static_cast<int64_t>(chunk.valid_rows));
    }

    std::vector<float> staged(static_cast<size_t>(output_rows));
    record_pq_adc_batch(r, task_count, static_cast<int64_t>(chunks.size()), output_rows,
                        padded_rows);

    /* There is no iGPU ADC backend. FORCE_GPU must not punch through to NPU. */
    if (r.policy == OVVS_POLICY_FORCE_GPU) return finish_forced_fail(r);
    const bool npu_allowed = r.policy == OVVS_POLICY_FORCE_NPU || !r.npu_busy;
    if (r.policy != OVVS_POLICY_FORCE_CPU && npu_allowed) {
      if (npu_pq_adc_batch(r, chunks.data(), static_cast<int64_t>(chunks.size()), pq_m,
                           ks, staged.data())) {
        std::memcpy(out, staged.data(), staged.size() * sizeof(float));
        r.last_device = OVVS_DEVICE_NPU;
        return OVVS_STATUS_SUCCESS;
      }
      if (r.policy == OVVS_POLICY_FORCE_NPU) return finish_forced_fail(r);
    }
    if (r.policy == OVVS_POLICY_FORCE_NPU || r.policy == OVVS_POLICY_FORCE_GPU) {
      return finish_forced_fail(r);
    }

    for (int64_t task_index = 0; task_index < task_count; ++task_index) {
      const PqAdcTask& task = tasks[task_index];
      for (int64_t row = 0; row < task.rows; ++row) {
        const uint8_t* code =
            task.codes + static_cast<size_t>(row) * static_cast<size_t>(pq_m);
        float score = 0.0f;
        for (int32_t m = 0; m < pq_m; ++m) {
          score += task.tables[static_cast<size_t>(m) * static_cast<size_t>(ks) +
                               static_cast<size_t>(code[m])];
        }
        staged[static_cast<size_t>(task.output_offset + row)] = score;
      }
    }
    record_pq_adc_cpu_rows(r, output_rows);
    std::memcpy(out, staged.data(), staged.size() * sizeof(float));
    r.last_device = OVVS_DEVICE_CPU;
    return OVVS_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return OVVS_STATUS_OOM;
  } catch (const std::length_error&) {
    return OVVS_STATUS_SHAPE_MISMATCH;
  } catch (...) {
    return OVVS_STATUS_ERROR;
  }
}

ovvsStatus prim_pq_adc(ResourcesData& r, const float* tables, int32_t pq_m, int32_t ks,
                       const uint8_t* codes, int64_t ncodes, float* out) {
  const PqAdcTask task{tables, codes, ncodes, 0};
  return prim_pq_adc_batch(r, &task, 1, pq_m, ks, out, ncodes);
}

ovvsStatus prim_ivfpq_scan_select(ResourcesData& r, const IvfPqScanTask* tasks,
                                  int64_t task_count, const float* luts,
                                  int64_t lut_elements, const int64_t* packed_ids,
                                  const uint8_t* packed_codes, int64_t packed_rows,
                                  const uint8_t* allow_bitset, int64_t allow_bitset_bytes,
                                  int64_t nq, int32_t pq_m, int32_t ks, int32_t krefine,
                                  int32_t* packed_positions, int32_t* counts,
                                  GpuWorkStats* stats) {
  /* No SKU/shape table promotes this fused path yet. FORCE_CPU, FORCE_NPU,
     NPU_IF_FASTER, AUTO, GPU_IF_FASTER, and HETERO keep the existing ADC/select
     route until matched end-to-end evidence names the GPU as winner. */
  if (r.policy != OVVS_POLICY_FORCE_GPU) return OVVS_STATUS_UNSUPPORTED;

  const ovvsStatus status = gpu_ivfpq_scan_select(
      r, tasks, task_count, luts, lut_elements, packed_ids, packed_codes,
      packed_rows, allow_bitset, allow_bitset_bytes, nq, pq_m, ks, krefine,
      packed_positions, counts, stats);
  if (status == OVVS_STATUS_SUCCESS) {
    r.last_device = OVVS_DEVICE_GPU;
    return OVVS_STATUS_SUCCESS;
  }
  if (status == OVVS_STATUS_UNSUPPORTED || status == OVVS_STATUS_DEVICE_UNAVAILABLE) {
    return finish_forced_fail(r);
  }
  return status;
}

ovvsStatus prim_nndescent_build(ResourcesData& r, const float* dataset, int64_t n, int64_t dim,
                                 ovvsMetric metric, int32_t degree, int32_t iterations,
                                 int32_t* graph, NnDescentBuildStats* stats) {
  if (r.policy == OVVS_POLICY_FORCE_NPU) return finish_forced_fail(r);

  const bool gpu_metric_supported = metric == OVVS_METRIC_L2_EXPANDED ||
                                    metric == OVVS_METRIC_L2_SQRT_EXPANDED ||
                                    metric == OVVS_METRIC_INNER_PRODUCT ||
                                    metric == OVVS_METRIC_COSINE_EXPANDED;
  const bool gpu_policy = r.policy == OVVS_POLICY_AUTO ||
                          r.policy == OVVS_POLICY_GPU_IF_FASTER ||
                          r.policy == OVVS_POLICY_HETERO ||
                          r.policy == OVVS_POLICY_FORCE_GPU;
  const bool gpu_ready = gpu_policy && gpu_metric_supported && sycl_gpu_available();
  if (gpu_ready) {
    const ovvsStatus gpu_status =
        gpu_nndescent_build(r, dataset, n, dim, metric, degree, iterations, graph, stats);
    if (gpu_status == OVVS_STATUS_SUCCESS) {
      r.last_device = OVVS_DEVICE_GPU;
      return OVVS_STATUS_SUCCESS;
    }
    /* A present, supported GPU path that fails preflight or execution must not
       silently enter the O(n*degree^2) host loop at production scale. */
    return gpu_status;
  }
  if (r.policy == OVVS_POLICY_FORCE_GPU) return finish_forced_fail(r);

  /* The algorithm owns the existing CPU reference loop. UNSUPPORTED is an
     internal adaptive-fallback signal and is never returned through the C ABI. */
  return OVVS_STATUS_UNSUPPORTED;
}

ovvsStatus prim_cagra_optimize_ranked(ResourcesData& r, const int32_t* initial,
                                      int64_t n, int32_t initial_degree,
                                      int32_t final_degree,
                                      std::vector<int32_t>& output) {
  output.clear();
  if (r.policy == OVVS_POLICY_FORCE_NPU) return finish_forced_fail(r);
  const bool gpu_policy = r.policy == OVVS_POLICY_AUTO ||
                          r.policy == OVVS_POLICY_GPU_IF_FASTER ||
                          r.policy == OVVS_POLICY_HETERO ||
                          r.policy == OVVS_POLICY_FORCE_GPU;
  if (!gpu_policy) return OVVS_STATUS_UNSUPPORTED;
  if (!sycl_gpu_available()) {
    return r.policy == OVVS_POLICY_FORCE_GPU ? finish_forced_fail(r)
                                             : OVVS_STATUS_UNSUPPORTED;
  }
  const ovvsStatus status = gpu_cagra_optimize_ranked(
      r, initial, n, initial_degree, final_degree, output);
  if (status == OVVS_STATUS_SUCCESS) {
    r.last_device = OVVS_DEVICE_GPU;
    return status;
  }
  if (status == OVVS_STATUS_UNSUPPORTED || status == OVVS_STATUS_DEVICE_UNAVAILABLE) {
    return r.policy == OVVS_POLICY_FORCE_GPU ? finish_forced_fail(r)
                                             : OVVS_STATUS_UNSUPPORTED;
  }
  return status;
}

ovvsStatus prim_graph_walk(ResourcesData& r, const float* dataset, int64_t n, int64_t dim,
                           ovvsMetric metric, const int32_t* graph, int32_t degree, const float* queries,
                           int64_t nq, int64_t k, int32_t itopk, int32_t search_width,
                           const uint8_t* bitset, int64_t* neighbors, float* distances) {
  /* The graph walk has no NPU implementation. A forced device must either run
     the complete walk or fail; it must never cross into the scalar host path. */
  if (r.policy == OVVS_POLICY_FORCE_NPU) return finish_forced_fail(r);

  const bool gpu_metric_supported = metric == OVVS_METRIC_L2_EXPANDED ||
                                    metric == OVVS_METRIC_L2_SQRT_EXPANDED ||
                                    metric == OVVS_METRIC_INNER_PRODUCT ||
                                    metric == OVVS_METRIC_COSINE_EXPANDED;
  const bool gpu_policy = r.policy == OVVS_POLICY_AUTO ||
                          r.policy == OVVS_POLICY_GPU_IF_FASTER ||
                          r.policy == OVVS_POLICY_HETERO ||
                          r.policy == OVVS_POLICY_FORCE_GPU;
  /* Hybrid split: run the iGPU and the CPU workers on disjoint query ranges of the same
     batch, one wall clock. Per-query CPU/GPU parity is a tested invariant, so the merged
     output is bit-identical to either engine alone at any split. A forced device stays
     exclusive; the GPU leg failing falls back to the CPU for its share, preserving the
     existing fallback semantics.
     TEMPORARY knob until OVVS_POLICY_HETERO adopts it as default behaviour:
     OVVS_HYBRID_WALK = fraction of queries sent to the GPU, 0 < f < 1 (0/unset = off;
     measured contended engine rates put the optimum near 0.5). */
  const double hybrid_frac = [&]() -> double {
    const char* env = std::getenv("OVVS_HYBRID_WALK");
    if (!env || !*env) return 0.0;
    const double parsed = std::strtod(env, nullptr);
    if (!(parsed > 0.0) || parsed >= 1.0) return 0.0;
    return parsed;
  }();
  const bool hybrid_ok = hybrid_frac > 0.0 && gpu_metric_supported && nq >= 2 &&
                         (r.policy == OVVS_POLICY_AUTO || r.policy == OVVS_POLICY_GPU_IF_FASTER ||
                          r.policy == OVVS_POLICY_HETERO);
  int64_t cpu_begin = 0;
  bool hybrid_active = false;
  bool gpu_leg_ok = false;
  std::thread gpu_worker;

  if (gpu_policy && gpu_metric_supported && !hybrid_ok) {
    if (gpu_cagra_walk(r, dataset, n, dim, metric, graph, degree, queries, nq, k, itopk, search_width,
                       bitset, neighbors, distances)) {
      r.last_device = OVVS_DEVICE_GPU;
      return OVVS_STATUS_SUCCESS;
    }
  }
  if (r.policy == OVVS_POLICY_FORCE_GPU) return finish_forced_fail(r);
  if (hybrid_ok) {
    cpu_begin = static_cast<int64_t>(hybrid_frac * static_cast<double>(nq) + 0.5);
    cpu_begin = std::max<int64_t>(1, std::min<int64_t>(cpu_begin, nq - 1));
    hybrid_active = true;
    gpu_worker = std::thread([&, cpu_begin]() {
      try {
        gpu_leg_ok = gpu_cagra_walk(r, dataset, n, dim, metric, graph, degree, queries, cpu_begin,
                                    k, itopk, search_width, bitset, neighbors, distances);
      } catch (...) {
        gpu_leg_ok = false; /* the CPU picks this range up after the join */
      }
    });
  }

  r.last_device = OVVS_DEVICE_CPU;

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
  /* Each query's walk is fully independent -- every scratch buffer below is per-iteration
     and the output rows are disjoint -- so queries fan out across worker threads with
     bit-identical results at any thread count.
     TEMPORARY default: all hardware threads; OVVS_CPU_WALK_THREADS overrides (1 = the old
     serial behaviour) until a thread-count parameter earns a place in the ABI. */
  auto walk_one = [&](int64_t q) -> ovvsStatus {
    const float* query = queries + q * dim;
    std::vector<uint8_t> seen(static_cast<size_t>(n), 0);
    std::vector<char> expanded(static_cast<size_t>(n), 0);
    std::vector<Node> cand;
    std::vector<int64_t> seed_ids;
    const int64_t nseeds = cagra_seed_count(n, itopk, search_width);
    const uint32_t query_hash = cagra_query_hash(query, dim);
    for (int64_t s = 0; s < nseeds; ++s) {
      const uint64_t mixed = static_cast<uint64_t>(s) * 9973u + static_cast<uint64_t>(query_hash) * 13u;
      const int64_t id = static_cast<int64_t>(mixed % static_cast<uint64_t>(n));
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
      return OVVS_STATUS_SUCCESS; /* this query is done (was `continue` in the loop) */
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
    return OVVS_STATUS_SUCCESS;
  };

  const int walk_threads = [&]() -> int {
    const char* env = std::getenv("OVVS_CPU_WALK_THREADS");
    long parsed = 0;
    if (env && *env) parsed = std::strtol(env, nullptr, 10);
    if (parsed <= 0) parsed = static_cast<long>(std::thread::hardware_concurrency());
    if (parsed < 1) parsed = 1;
    return static_cast<int>(std::min<long>(parsed, 256));
  }();
  /* Work-stealing over a query range: per-query cost varies, so workers claim the next
     index from an atomic counter rather than taking fixed chunks. First failure wins;
     workers drain out on it. */
  auto run_cpu_range = [&](int64_t lo, int64_t hi) -> ovvsStatus {
    if (hi <= lo) return OVVS_STATUS_SUCCESS;
    if (walk_threads <= 1 || hi - lo <= 1) {
      for (int64_t q = lo; q < hi; ++q) {
        const ovvsStatus s = walk_one(q);
        if (s != OVVS_STATUS_SUCCESS) return s;
      }
      return OVVS_STATUS_SUCCESS;
    }
    std::atomic<ovvsStatus> walk_status{OVVS_STATUS_SUCCESS};
    std::atomic<int64_t> next_query{lo};
    const int worker_count = static_cast<int>(std::min<int64_t>(walk_threads, hi - lo));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(worker_count));
    for (int t = 0; t < worker_count; ++t) {
      workers.emplace_back([&]() {
        for (;;) {
          const int64_t q = next_query.fetch_add(1, std::memory_order_relaxed);
          if (q >= hi) return;
          if (walk_status.load(std::memory_order_relaxed) != OVVS_STATUS_SUCCESS) return;
          const ovvsStatus s = walk_one(q);
          if (s != OVVS_STATUS_SUCCESS) {
            ovvsStatus expected = OVVS_STATUS_SUCCESS;
            walk_status.compare_exchange_strong(expected, s);
            return;
          }
        }
      });
    }
    for (auto& w : workers) w.join();
    return walk_status.load();
  };

  const ovvsStatus cpu_status = run_cpu_range(cpu_begin, nq);
  if (hybrid_active) {
    gpu_worker.join();
    if (!gpu_leg_ok) {
      /* The accelerator leg failed: finish its share on the CPU so the call keeps the
         all-or-nothing contract the non-hybrid fallback already provides. */
      const ovvsStatus fallback = run_cpu_range(0, cpu_begin);
      if (fallback != OVVS_STATUS_SUCCESS) return fallback;
    }
  }
  return cpu_status;
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

ovvsStatus kmeans_fit_impl(ResourcesData& r, const float* x, int64_t n, int64_t dim, int32_t k,
                           int32_t iters, std::vector<float>& centroids) {
  if (!x || n <= 0 || dim <= 0 || k <= 0 || iters < 0 ||
      n > std::numeric_limits<int32_t>::max()) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  k = std::max(1, std::min(k, static_cast<int32_t>(n)));
  const size_t max_float_elements = std::numeric_limits<size_t>::max() / sizeof(float);
  if (static_cast<uint64_t>(k) >
          static_cast<uint64_t>(max_float_elements) / static_cast<uint64_t>(dim) ||
      static_cast<uint64_t>(n) >
          static_cast<uint64_t>(max_float_elements) / static_cast<uint64_t>(k)) {
    return OVVS_STATUS_SHAPE_MISMATCH;
  }
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
    const ovvsStatus pairwise_status =
        prim_pairwise(r, OVVS_METRIC_L2_EXPANDED, x, n, centroids.data(), k, dim,
                      scores.data(), 2.f);
    if (pairwise_status != OVVS_STATUS_SUCCESS) return pairwise_status;
    const ovvsStatus topk_status =
        prim_topk(r, scores.data(), n, k, 1, labels.data(), dist.data(), false);
    if (topk_status != OVVS_STATUS_SUCCESS) return topk_status;
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
  return OVVS_STATUS_SUCCESS;
}

}  // namespace impl
}  // namespace ovvs
