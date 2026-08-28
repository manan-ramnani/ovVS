#include "test_harness.hpp"
#include "internal.hpp"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <limits>
#include <numeric>

namespace {

bool ref_topk_before(float lhs, int64_t lhs_ordinal, float rhs, int64_t rhs_ordinal,
                     bool largest) {
  const bool lhs_nan = std::isnan(lhs);
  const bool rhs_nan = std::isnan(rhs);
  if (lhs_nan != rhs_nan) return !lhs_nan;
  if (!lhs_nan) {
    if (lhs < rhs) return !largest;
    if (lhs > rhs) return largest;
  }
  return lhs_ordinal < rhs_ordinal;
}

void expect_cpu_topk_matches_total_order(ovvsResources_t res, const std::vector<float>& scores,
                                         int64_t rows, int64_t cols, int64_t k, bool largest,
                                         const std::string& label) {
  std::vector<int64_t> indices(static_cast<size_t>(rows * k), -1);
  std::vector<float> values(static_cast<size_t>(rows * k), 0.0f);
  expect_status(ovvsTopk(res, scores.data(), rows, cols, k, indices.data(), values.data(), largest),
                label.c_str());

  for (int64_t row_index = 0; row_index < rows; ++row_index) {
    const float* row = scores.data() + row_index * cols;
    std::vector<int64_t> expected(static_cast<size_t>(cols));
    std::iota(expected.begin(), expected.end(), 0);
    std::sort(expected.begin(), expected.end(), [&](int64_t lhs, int64_t rhs) {
      return ref_topk_before(row[lhs], lhs, row[rhs], rhs, largest);
    });
    for (int64_t rank = 0; rank < k; ++rank) {
      const size_t out_offset = static_cast<size_t>(row_index * k + rank);
      const int64_t expected_index = expected[static_cast<size_t>(rank)];
      expect(indices[out_offset] == expected_index,
             label + " index at row " + std::to_string(row_index) + " rank " +
                 std::to_string(rank));
      const float expected_value = row[expected_index];
      if (std::isnan(expected_value)) {
        expect(std::isnan(values[out_offset]), label + " NaN value");
      } else {
        expect(std::bit_cast<uint32_t>(values[out_offset]) ==
                   std::bit_cast<uint32_t>(expected_value),
               label + " exact value at row " + std::to_string(row_index) + " rank " +
                   std::to_string(rank));
      }
    }
  }
}

}  // namespace

OVVS_TEST(gemm_matches_cpu_reference) {
  Res res;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  const int64_t m = 7, n = 11, k = 5;
  auto A = make_data(m, k, 1);
  auto B = make_data(n, k, 2);
  std::vector<float> C(static_cast<size_t>(m * n)), R(static_cast<size_t>(m * n));
  expect_status(ovvsGemm(res.r, A.data(), B.data(), C.data(), m, n, k, 1), "gemm");
  ref_gemm(A.data(), B.data(), R.data(), m, n, k, true);
  for (size_t i = 0; i < C.size(); ++i) {
    expect(std::fabs(C[i] - R[i]) < 1e-4f, "gemm mismatch");
  }
}

OVVS_TEST(gemm_no_transpose) {
  Res res;
  const int64_t m = 4, n = 3, k = 6;
  auto A = make_data(m, k, 3);
  auto Bt = make_data(k, n, 4);
  std::vector<float> C(static_cast<size_t>(m * n)), R(static_cast<size_t>(m * n));
  expect_status(ovvsGemm(res.r, A.data(), Bt.data(), C.data(), m, n, k, 0), "gemm");
  ref_gemm(A.data(), Bt.data(), R.data(), m, n, k, false);
  for (size_t i = 0; i < C.size(); ++i) expect(std::fabs(C[i] - R[i]) < 1e-4f, "gemm nt");
}

OVVS_TEST(topk_matches_partial_sort) {
  Res res;
  const int64_t rows = 5, cols = 17, k = 4;
  auto scores = make_data(rows, cols, 9);
  std::vector<int64_t> idx(static_cast<size_t>(rows * k));
  std::vector<float> val(static_cast<size_t>(rows * k));
  expect_status(ovvsTopk(res.r, scores.data(), rows, cols, k, idx.data(), val.data(), 0), "topk");
  for (int64_t r = 0; r < rows; ++r) {
    std::vector<int64_t> order(static_cast<size_t>(cols));
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int64_t a, int64_t b) {
      return ref_topk_before(scores[static_cast<size_t>(r * cols + a)], a,
                             scores[static_cast<size_t>(r * cols + b)], b, false);
    });
    for (int64_t t = 0; t < k; ++t) {
      expect(idx[static_cast<size_t>(r * k + t)] == order[static_cast<size_t>(t)], "topk idx");
      expect(std::fabs(val[static_cast<size_t>(r * k + t)] -
                       scores[static_cast<size_t>(r * cols + order[static_cast<size_t>(t)])]) < 1e-6f,
             "topk val");
    }
  }
}

OVVS_TEST(topk_cpu_small_k_has_deterministic_total_order) {
  Res res;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  constexpr int64_t rows = 3;
  constexpr int64_t cols = 97;
  std::vector<float> scores(static_cast<size_t>(rows * cols));
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      scores[static_cast<size_t>(row * cols + col)] =
          static_cast<float>(((col * 11 + row * 7) % 19) - 9);
    }
    float* current = scores.data() + row * cols;
    current[0] = std::numeric_limits<float>::infinity();
    current[1] = -std::numeric_limits<float>::infinity();
    current[2] = 0.0f;
    current[3] = -0.0f;
    current[4] = std::numeric_limits<float>::quiet_NaN();
    current[5] = std::numeric_limits<float>::quiet_NaN();
    current[6] = current[7] = 3.5f;
  }

  for (const bool largest : {false, true}) {
    for (const int64_t k : {int64_t{1}, int64_t{10}, int64_t{32}, int64_t{64}}) {
      expect_cpu_topk_matches_total_order(res.r, scores, rows, cols, k, largest,
                                          std::string("small-k ") +
                                              (largest ? "largest" : "smallest") + " k=" +
                                              std::to_string(k));
    }
  }
}

OVVS_TEST(topk_cpu_nonfinite_zero_and_equal_score_order) {
  Res res;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  const std::vector<float> scores = {
      std::numeric_limits<float>::quiet_NaN(), 0.0f,
      -0.0f,                                     std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),  1.0f,
      1.0f,                                      std::numeric_limits<float>::quiet_NaN(),
  };
  expect_cpu_topk_matches_total_order(res.r, scores, 1, 8, 8, false, "nonfinite smallest");
  expect_cpu_topk_matches_total_order(res.r, scores, 1, 8, 8, true, "nonfinite largest");
}

OVVS_TEST(topk_cpu_k64_k65_boundary_matches) {
  Res res;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  constexpr int64_t rows = 2;
  constexpr int64_t cols = 71;
  std::vector<float> scores(static_cast<size_t>(rows * cols));
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      float value = static_cast<float>(((col * 5 + row * 3) % 13) - 6);
      if ((col % 17) == 0) value = (col % 34) == 0 ? -0.0f : 0.0f;
      scores[static_cast<size_t>(row * cols + col)] = value;
    }
  }

  for (const bool largest : {false, true}) {
    expect_cpu_topk_matches_total_order(res.r, scores, rows, cols, 64, largest,
                                        largest ? "boundary largest k64" : "boundary smallest k64");
    expect_cpu_topk_matches_total_order(res.r, scores, rows, cols, 65, largest,
                                        largest ? "boundary largest k65" : "boundary smallest k65");

    std::vector<int64_t> k64_indices(static_cast<size_t>(rows * 64));
    std::vector<int64_t> k65_indices(static_cast<size_t>(rows * 65));
    std::vector<float> k64_values(static_cast<size_t>(rows * 64));
    std::vector<float> k65_values(static_cast<size_t>(rows * 65));
    expect_status(ovvsTopk(res.r, scores.data(), rows, cols, 64, k64_indices.data(),
                           k64_values.data(), largest),
                  "boundary k64");
    expect_status(ovvsTopk(res.r, scores.data(), rows, cols, 65, k65_indices.data(),
                           k65_values.data(), largest),
                  "boundary k65");
    for (int64_t row = 0; row < rows; ++row) {
      for (int64_t rank = 0; rank < 64; ++rank) {
        expect(k64_indices[static_cast<size_t>(row * 64 + rank)] ==
                   k65_indices[static_cast<size_t>(row * 65 + rank)],
               "k64/k65 prefix mismatch");
      }
    }
  }
}

OVVS_TEST(kselection_is_topk) {
  Res res;
  auto scores = make_data(3, 8, 12);
  std::vector<int64_t> a(9), b(9);
  std::vector<float> va(9), vb(9);
  expect_status(ovvsTopk(res.r, scores.data(), 3, 8, 3, a.data(), va.data(), 1), "topk");
  expect_status(ovvsKSelection(res.r, scores.data(), 3, 8, 3, b.data(), vb.data(), 1), "ksel");
  expect(a == b, "kselection != topk");
}

OVVS_TEST(gemm_npu_f16_i8_vs_cpu) {
  Res res;
  int32_t npu = 0;
  ovvsResourcesNpuAvailable(res.r, &npu);
  if (!npu) return;
  const int64_t m = 32, n = 32, k = 64;
  auto A = make_data(m, k, 91);
  auto B = make_data(n, k, 92);
  std::vector<float> cpu(static_cast<size_t>(m * n)), got(static_cast<size_t>(m * n));
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  expect_status(ovvsGemm(res.r, A.data(), B.data(), cpu.data(), m, n, k, 1), "cpu");
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_NPU);
  const ovvsStatus s16 = ovvsGemmEx(res.r, A.data(), B.data(), got.data(), m, n, k, 1, OVVS_DTYPE_F16);
  if (s16 == OVVS_STATUS_SUCCESS) {
    ovvsDType dt = OVVS_DTYPE_F32;
    ovvsResourcesLastComputeDtype(res.r, &dt);
    expect(dt == OVVS_DTYPE_F16, "npu f16 compute dtype");
    float maxe = 0.f;
    for (size_t i = 0; i < cpu.size(); ++i) maxe = std::max(maxe, std::fabs(cpu[i] - got[i]));
    expect(maxe < 0.15f, "npu f16 vs cpu " + std::to_string(maxe));
  } else {
    expect(s16 == OVVS_STATUS_DEVICE_UNAVAILABLE, "f16 unavailable or success");
  }
  const ovvsStatus s8 = ovvsGemmEx(res.r, A.data(), B.data(), got.data(), m, n, k, 1, OVVS_DTYPE_I8);
  if (s8 == OVVS_STATUS_SUCCESS) {
    ovvsDType dt = OVVS_DTYPE_F32;
    ovvsResourcesLastComputeDtype(res.r, &dt);
    expect(dt == OVVS_DTYPE_I8, "npu i8 compute dtype");
    float maxe = 0.f;
    for (size_t i = 0; i < cpu.size(); ++i) {
      const float denom = std::max(1.f, std::fabs(cpu[i]));
      maxe = std::max(maxe, std::fabs(cpu[i] - got[i]) / denom);
    }
    expect(maxe < 0.25f, "npu i8 rel err " + std::to_string(maxe));
  } else {
    expect(s8 == OVVS_STATUS_DEVICE_UNAVAILABLE, "i8 unavailable or success");
  }
}

OVVS_TEST(gemm_gpu_f16_vs_cpu) {
  Res res;
  int32_t gpu = 0;
  ovvsResourcesGpuAvailable(res.r, &gpu);
  if (!gpu) return;
  const int64_t m = 32, n = 32, k = 64;
  auto A = make_data(m, k, 93);
  auto B = make_data(n, k, 94);
  std::vector<float> cpu(static_cast<size_t>(m * n)), got(static_cast<size_t>(m * n));
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  expect_status(ovvsGemm(res.r, A.data(), B.data(), cpu.data(), m, n, k, 1), "cpu");
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU);
  const ovvsStatus st = ovvsGemmEx(res.r, A.data(), B.data(), got.data(), m, n, k, 1, OVVS_DTYPE_F16);
  if (st != OVVS_STATUS_SUCCESS) {
    expect(st == OVVS_STATUS_DEVICE_UNAVAILABLE, "gpu f16");
    return;
  }
  ovvsDType dt = OVVS_DTYPE_F32;
  ovvsResourcesLastComputeDtype(res.r, &dt);
  expect(dt == OVVS_DTYPE_F16, "gpu f16 compute dtype");
  float maxe = 0.f;
  for (size_t i = 0; i < cpu.size(); ++i) maxe = std::max(maxe, std::fabs(cpu[i] - got[i]));
  expect(maxe < 0.15f, "gpu f16 vs cpu " + std::to_string(maxe));
}

OVVS_TEST(gemm_gpu_i8_vs_cpu) {
  Res res;
  int32_t gpu = 0;
  ovvsResourcesGpuAvailable(res.r, &gpu);
  if (!gpu) return;
  const int64_t m = 32, n = 32, k = 64;
  auto A = make_data(m, k, 95);
  auto B = make_data(n, k, 96);
  std::vector<float> cpu(static_cast<size_t>(m * n)), got(static_cast<size_t>(m * n));
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  expect_status(ovvsGemm(res.r, A.data(), B.data(), cpu.data(), m, n, k, 1), "cpu");
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU);
  const ovvsStatus st = ovvsGemmEx(res.r, A.data(), B.data(), got.data(), m, n, k, 1, OVVS_DTYPE_I8);
  if (st != OVVS_STATUS_SUCCESS) {
    expect(st == OVVS_STATUS_DEVICE_UNAVAILABLE, "gpu i8");
    return;
  }
  ovvsDType dt = OVVS_DTYPE_F32;
  ovvsResourcesLastComputeDtype(res.r, &dt);
  expect(dt == OVVS_DTYPE_I8, "gpu i8 compute dtype");
  float maxe = 0.f;
  for (size_t i = 0; i < cpu.size(); ++i) {
    const float denom = std::max(1.f, std::fabs(cpu[i]));
    maxe = std::max(maxe, std::fabs(cpu[i] - got[i]) / denom);
  }
  expect(maxe < 0.25f, "gpu i8 rel err " + std::to_string(maxe));
}

OVVS_TEST(gemm_npu_matches_cpu_when_present) {
  Res res;
  int32_t npu = 0;
  ovvsResourcesNpuAvailable(res.r, &npu);
  if (!npu) return;
  const int64_t m = 8, n = 12, k = 16;
  auto A = make_data(m, k, 77);
  auto B = make_data(n, k, 78);
  std::vector<float> cpu(static_cast<size_t>(m * n)), npuo(static_cast<size_t>(m * n));
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  expect_status(ovvsGemm(res.r, A.data(), B.data(), cpu.data(), m, n, k, 1), "cpu gemm");
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_NPU);
  expect_status(ovvsGemm(res.r, A.data(), B.data(), npuo.data(), m, n, k, 1), "npu gemm");
  for (size_t i = 0; i < cpu.size(); ++i) {
    expect(std::fabs(cpu[i] - npuo[i]) < 2e-2f, "npu vs cpu gemm");
  }
}

OVVS_TEST(gemm_npu_rejects_unsafe_fp16_range_when_present) {
  Res res;
  int32_t npu = 0;
  ovvsResourcesNpuAvailable(res.r, &npu);
  if (!npu) return;

  const int64_t m = 8, n = 12, k = 16;
  std::vector<float> A(static_cast<size_t>(m * k), 16.0f);
  std::vector<float> B(static_cast<size_t>(n * k), 255.0f);
  std::vector<float> out(static_cast<size_t>(m * n));
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_NPU);
  expect_status(ovvsGemm(res.r, A.data(), B.data(), out.data(), m, n, k, 1),
                "npu finite-boundary gemm");
  for (float value : out) {
    expect(std::isfinite(value), "npu finite-boundary gemm finite");
    expect(std::fabs(value - 65280.0f) <= 64.0f, "npu finite-boundary gemm value");
  }

  std::fill(B.begin(), B.end(), 256.0f);
  expect(ovvsGemm(res.r, A.data(), B.data(), out.data(), m, n, k, 1) ==
             OVVS_STATUS_DEVICE_UNAVAILABLE,
         "npu gemm must reject fp16 overflow bound");

  std::fill(B.begin(), B.end(), 0.01f);
  A[0] = 70000.0f;
  expect(ovvsGemm(res.r, A.data(), B.data(), out.data(), m, n, k, 1) ==
             OVVS_STATUS_DEVICE_UNAVAILABLE,
         "npu gemm must reject an out-of-range operand even when the dot bound is small");

  A[0] = std::numeric_limits<float>::infinity();
  expect(ovvsGemm(res.r, A.data(), B.data(), out.data(), m, n, k, 1) ==
             OVVS_STATUS_DEVICE_UNAVAILABLE,
         "npu gemm must reject non-finite input");
}

OVVS_TEST(gemm_gpu_matches_cpu_when_present) {
  Res res;
  int32_t gpu = 0;
  ovvsResourcesGpuAvailable(res.r, &gpu);
  if (!gpu) return;
  const int64_t m = 8, n = 12, k = 16;
  auto A = make_data(m, k, 81);
  auto B = make_data(n, k, 82);
  std::vector<float> cpu(static_cast<size_t>(m * n)), g(static_cast<size_t>(m * n));
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  expect_status(ovvsGemm(res.r, A.data(), B.data(), cpu.data(), m, n, k, 1), "cpu gemm");
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU);
  expect_status(ovvsGemm(res.r, A.data(), B.data(), g.data(), m, n, k, 1), "gpu gemm");
  for (size_t i = 0; i < cpu.size(); ++i) {
    expect(std::fabs(cpu[i] - g[i]) < 2e-2f, "gpu vs cpu gemm");
  }
}

OVVS_TEST(gemm_gpu_matches_ref_gemm) {
  Res res;
  int32_t gpu = 0;
  ovvsResourcesGpuAvailable(res.r, &gpu);
  if (!gpu) return;
  const int64_t m = 6, n = 9, k = 7;
  auto A = make_data(m, k, 83);
  auto B = make_data(n, k, 84);
  std::vector<float> g(static_cast<size_t>(m * n)), R(static_cast<size_t>(m * n));
  ref_gemm(A.data(), B.data(), R.data(), m, n, k, true);
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU);
  expect_status(ovvsGemm(res.r, A.data(), B.data(), g.data(), m, n, k, 1), "gpu ref gemm");
  ovvsDevice last = OVVS_DEVICE_CPU;
  expect_status(ovvsResourcesLastDevice(res.r, &last), "last");
  expect(last == OVVS_DEVICE_GPU, "FORCE_GPU last_device");
  std::printf("    gemm_gpu_matches_ref_gemm last_device=gpu\n");
  for (size_t i = 0; i < g.size(); ++i) expect(std::fabs(g[i] - R[i]) < 2e-2f, "gpu vs ref_gemm");
}

static float independent_pq_adc(const float* tables, const uint8_t* code, int32_t pq_m,
                                int32_t ks) {
  float score = 0.0f;
  for (int32_t m = 0; m < pq_m; ++m) {
    score += tables[static_cast<size_t>(m * ks + static_cast<int32_t>(code[m]))];
  }
  return score;
}

OVVS_TEST(pq_adc_fixed_bucket_planner_boundaries_and_splits) {
  using ovvs::impl::PqAdcChunk;
  using ovvs::impl::PqAdcTask;

  struct BucketCase {
    int64_t rows;
    int32_t bucket;
  };
  const std::vector<BucketCase> bucket_cases{
      {0, 0},       {1, 128},     {127, 128},   {128, 128},   {129, 256},
      {255, 256},   {256, 256},   {257, 512},   {511, 512},   {512, 512},
      {513, 1024},  {1023, 1024}, {1024, 1024}, {1025, 2048}, {2047, 2048},
      {2048, 2048},
  };
  for (const BucketCase& item : bucket_cases) {
    expect(ovvs::impl::pq_adc_bucket_rows(item.rows) == item.bucket,
           "fixed ADC bucket for rows=" + std::to_string(item.rows));
  }

  float dummy_table = 0.0f;
  uint8_t dummy_codes[6]{};
  const std::vector<int64_t> rows{127, 128, 129, 255, 256, 257};
  const std::vector<int32_t> expected_buckets{128, 128, 256, 256, 256, 512};
  std::vector<PqAdcTask> tasks;
  int64_t output_rows = 0;
  for (size_t i = 0; i < rows.size(); ++i) {
    tasks.push_back({&dummy_table, &dummy_codes[i], rows[i], output_rows});
    output_rows += rows[i];
  }
  expect(output_rows == 1152, "fixed ADC boundary fixture real rows");

  std::vector<PqAdcChunk> chunks;
  expect_status(ovvs::impl::plan_pq_adc_chunks(tasks.data(),
                                                static_cast<int64_t>(tasks.size()),
                                                1, output_rows, chunks),
                "fixed ADC boundary plan");
  expect(chunks.size() == tasks.size(), "one chunk per sub-512 boundary task");
  int64_t planned_rows = 0;
  int64_t planned_capacity = 0;
  int32_t bucket_128 = 0, bucket_256 = 0, bucket_512 = 0;
  for (size_t i = 0; i < chunks.size(); ++i) {
    const PqAdcChunk& chunk = chunks[i];
    expect(chunk.tables == tasks[i].tables && chunk.codes == tasks[i].codes,
           "fixed ADC planner preserves task payload pointers");
    expect(chunk.valid_rows == rows[i], "fixed ADC planner preserves valid rows");
    expect(chunk.bucket_rows == expected_buckets[i], "fixed ADC planner selects bucket");
    expect(chunk.output_offset == tasks[i].output_offset,
           "fixed ADC planner preserves dense output order");
    planned_rows += chunk.valid_rows;
    planned_capacity += chunk.bucket_rows;
    bucket_128 += chunk.bucket_rows == 128 ? 1 : 0;
    bucket_256 += chunk.bucket_rows == 256 ? 1 : 0;
    bucket_512 += chunk.bucket_rows == 512 ? 1 : 0;
  }
  expect(planned_rows == 1152 && planned_capacity == 1536,
         "fixed ADC boundary real and padded row totals");
  expect(planned_capacity - planned_rows == 384, "fixed ADC boundary padding total");
  expect(bucket_128 == 2 && bucket_256 == 3 && bucket_512 == 1,
         "fixed ADC boundary bucket histogram");

  struct SplitCase {
    int64_t rows;
    std::vector<int32_t> valid;
    std::vector<int32_t> buckets;
  };
  const std::vector<SplitCase> split_cases{
      {2049, {2048, 1}, {2048, 128}},
      {2177, {2048, 129}, {2048, 256}},
      {4096, {2048, 2048}, {2048, 2048}},
      {4097, {2048, 2048, 1}, {2048, 2048, 128}},
  };
  for (const SplitCase& item : split_cases) {
    constexpr int32_t split_pq_m = 3;
    std::vector<uint8_t> split_codes(static_cast<size_t>(item.rows * split_pq_m));
    const PqAdcTask task{&dummy_table, split_codes.data(), item.rows, 0};
    chunks.clear();
    expect_status(ovvs::impl::plan_pq_adc_chunks(&task, 1, split_pq_m, item.rows, chunks),
                  "fixed ADC split plan");
    expect(chunks.size() == item.valid.size(), "fixed ADC split chunk count");
    int64_t offset = 0;
    int64_t valid_sum = 0;
    for (size_t i = 0; i < chunks.size(); ++i) {
      expect(chunks[i].valid_rows == item.valid[i] &&
                 chunks[i].bucket_rows == item.buckets[i],
             "fixed ADC split shape");
      expect(chunks[i].output_offset == offset, "fixed ADC split output continuity");
      expect(chunks[i].codes == split_codes.data() + offset * split_pq_m,
             "fixed ADC split advances the code pointer");
      offset += chunks[i].valid_rows;
      valid_sum += chunks[i].valid_rows;
    }
    expect(valid_sum == item.rows, "fixed ADC split covers every real row");
  }
}

OVVS_TEST(pq_adc_cpu_multitask_matches_oracle_and_rejects_bad_codes_atomically) {
  using ovvs::impl::PqAdcTask;
  Res res;
  expect_status(ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU),
                "CPU multi-task ADC policy");
  constexpr int32_t pq_m = 4, ks = 16;
  const std::vector<int64_t> rows{3, 129, 5};
  std::vector<std::vector<float>> tables(rows.size(),
                                         std::vector<float>(static_cast<size_t>(pq_m * ks)));
  std::vector<std::vector<uint8_t>> codes(rows.size());
  std::vector<PqAdcTask> tasks;
  int64_t output_rows = 0;
  for (size_t task = 0; task < rows.size(); ++task) {
    for (int32_t m = 0; m < pq_m; ++m) {
      for (int32_t code = 0; code < ks; ++code) {
        tables[task][static_cast<size_t>(m * ks + code)] =
            10.0f * static_cast<float>(task + 1) + 0.25f * static_cast<float>(m) +
            0.01f * static_cast<float>(code);
      }
    }
    codes[task].resize(static_cast<size_t>(rows[task] * pq_m));
    for (int64_t row = 0; row < rows[task]; ++row) {
      for (int32_t m = 0; m < pq_m; ++m) {
        codes[task][static_cast<size_t>(row * pq_m + m)] =
            static_cast<uint8_t>((row * 7 + m * 3 + static_cast<int64_t>(task)) % ks);
      }
    }
    tasks.push_back({tables[task].data(), codes[task].data(), rows[task], output_rows});
    output_rows += rows[task];
  }

  std::vector<float> expected(static_cast<size_t>(output_rows));
  for (size_t task = 0; task < tasks.size(); ++task) {
    for (int64_t row = 0; row < tasks[task].rows; ++row) {
      expected[static_cast<size_t>(tasks[task].output_offset + row)] =
          independent_pq_adc(tasks[task].tables, tasks[task].codes + row * pq_m, pq_m, ks);
    }
  }
  std::vector<float> actual(static_cast<size_t>(output_rows + 2), -12345.0f);
  expect_status(ovvs::impl::prim_pq_adc_batch(
                    *ovvs::impl::rd(res.r), tasks.data(), static_cast<int64_t>(tasks.size()),
                    pq_m, ks, actual.data() + 1, output_rows),
                "CPU multi-task ADC");
  expect(actual.front() == -12345.0f && actual.back() == -12345.0f,
         "CPU multi-task ADC canaries");
  for (int64_t row = 0; row < output_rows; ++row) {
    const float tolerance =
        1e-5f * std::max(1.0f, std::fabs(expected[static_cast<size_t>(row)]));
    expect(std::fabs(actual[static_cast<size_t>(row + 1)] -
                     expected[static_cast<size_t>(row)]) <= tolerance,
           "CPU multi-task ADC oracle/order row=" + std::to_string(row));
  }
  ovvsDevice last = OVVS_DEVICE_NPU;
  expect_status(ovvsResourcesLastDevice(res.r, &last), "CPU multi-task ADC last device");
  expect(last == OVVS_DEVICE_CPU, "CPU multi-task ADC attribution");

  codes[1][static_cast<size_t>(17 * pq_m + 2)] = static_cast<uint8_t>(ks);
  std::fill(actual.begin(), actual.end(), -12345.0f);
  expect(ovvs::impl::prim_pq_adc_batch(
             *ovvs::impl::rd(res.r), tasks.data(), static_cast<int64_t>(tasks.size()), pq_m,
             ks, actual.data() + 1, output_rows) == OVVS_STATUS_INVALID_ARGUMENT,
         "multi-task ADC rejects a code outside the LUT");
  for (float value : actual) {
    expect(value == -12345.0f, "invalid multi-task ADC publishes no output");
  }
}

OVVS_TEST(pq_adc_multitask_forced_policies_fail_atomically) {
  using ovvs::impl::PqAdcTask;
  Res res;
  int32_t npu = 0;
  expect_status(ovvsResourcesNpuAvailable(res.r, &npu),
                "multi-task ADC NPU availability");
  constexpr int32_t pq_m = 8, ks = 16;
  const std::vector<int64_t> rows{129, 257};
  std::vector<float> safe_tables(static_cast<size_t>(pq_m * ks), 0.25f);
  std::vector<float> unsafe_tables(static_cast<size_t>(pq_m * ks), 8192.0f);
  std::vector<uint8_t> first_codes(static_cast<size_t>(rows[0] * pq_m), 1);
  std::vector<uint8_t> second_codes(static_cast<size_t>(rows[1] * pq_m), 2);
  const int64_t output_rows = rows[0] + rows[1];
  const PqAdcTask tasks[]{{safe_tables.data(), first_codes.data(), rows[0], 0},
                          {unsafe_tables.data(), second_codes.data(), rows[1], rows[0]}};

  expect_status(ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU),
                "forced-policy ADC CPU seed");
  std::vector<float> seeded(static_cast<size_t>(output_rows));
  expect_status(ovvs::impl::prim_pq_adc_batch(*ovvs::impl::rd(res.r), tasks, 2, pq_m, ks,
                                               seeded.data(), output_rows),
                "forced-policy ADC CPU seed call");

  int32_t fallbacks_before = 0;
  int32_t compile_fails_before = 0;
  expect_status(ovvsResourcesNpuFallbacks(res.r, &fallbacks_before),
                "forced-policy ADC fallback count before");
  expect_status(ovvsResourcesNpuCompileFails(res.r, &compile_fails_before),
                "forced-policy ADC compile-fail count before");

  std::vector<float> actual(static_cast<size_t>(output_rows + 2), -12345.0f);
  expect_status(ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU),
                "multi-task ADC FORCE_GPU policy");
  expect(ovvs::impl::prim_pq_adc_batch(*ovvs::impl::rd(res.r), tasks, 2, pq_m, ks,
                                       actual.data() + 1, output_rows) ==
             OVVS_STATUS_DEVICE_UNAVAILABLE,
         "multi-task ADC FORCE_GPU fails closed");
  for (float value : actual) expect(value == -12345.0f, "FORCE_GPU ADC output is atomic");
  int32_t fallbacks_after_gpu = 0;
  expect_status(ovvsResourcesNpuFallbacks(res.r, &fallbacks_after_gpu),
                "multi-task ADC FORCE_GPU fallback count");
  expect(fallbacks_after_gpu == fallbacks_before,
         "FORCE_GPU multi-task ADC does not increment NPU fallback count");

  expect_status(ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_NPU),
                "multi-task ADC FORCE_NPU policy");
  const ovvsStatus npu_status = ovvs::impl::prim_pq_adc_batch(
      *ovvs::impl::rd(res.r), tasks, 2, pq_m, ks, actual.data() + 1, output_rows);
  int32_t fallbacks_after_npu = 0;
  int32_t compile_fails_after = 0;
  expect_status(ovvsResourcesNpuFallbacks(res.r, &fallbacks_after_npu),
                "multi-task ADC FORCE_NPU fallback count");
  expect_status(ovvsResourcesNpuCompileFails(res.r, &compile_fails_after),
                "multi-task ADC FORCE_NPU compile-fail count");
  if (!npu) {
    expect(npu_status == OVVS_STATUS_DEVICE_UNAVAILABLE,
           "multi-task FORCE_NPU rejects an absent device");
    for (float value : actual) expect(value == -12345.0f, "absent NPU output is atomic");
    expect(fallbacks_after_npu == fallbacks_before + 1,
           "absent multi-task FORCE_NPU records one logical fallback");
  } else {
    expect_status(npu_status, "centered multi-task FORCE_NPU ADC");
    expect(actual.front() == -12345.0f && actual.back() == -12345.0f,
           "centered multi-task FORCE_NPU ADC canaries");
    for (int64_t row = 0; row < output_rows; ++row) {
      const float tolerance =
          2e-4f * std::max(1.0f, std::fabs(seeded[static_cast<size_t>(row)]));
      expect(std::fabs(actual[static_cast<size_t>(row + 1)] -
                       seeded[static_cast<size_t>(row)]) <= tolerance,
             "centered multi-task FORCE_NPU ADC oracle row=" + std::to_string(row));
    }
    expect(fallbacks_after_npu == fallbacks_before,
           "scaled multi-task FORCE_NPU needs no fallback");
    ovvsDevice last = OVVS_DEVICE_CPU;
    expect_status(ovvsResourcesLastDevice(res.r, &last),
                  "scaled multi-task ADC last device");
    expect(last == OVVS_DEVICE_NPU, "scaled multi-task ADC attributes NPU");
  }
  expect(compile_fails_after == compile_fails_before,
         "scaled or unavailable ADC is not a compile failure");
}

OVVS_TEST(npu_pq_adc_true_multitask_matches_reference_when_present) {
  using ovvs::impl::PqAdcTask;
  Res res;
  int32_t npu = 0;
  expect_status(ovvsResourcesNpuAvailable(res.r, &npu), "multi-task ADC NPU probe");
  if (!npu) skip_test("NPU not available");

  constexpr int32_t pq_m = 8, ks = 256;
  const std::vector<int64_t> rows{127, 128, 129, 255, 256, 257};
  std::vector<std::vector<float>> tables(rows.size(),
                                         std::vector<float>(static_cast<size_t>(pq_m * ks)));
  std::vector<std::vector<uint8_t>> codes(rows.size());
  std::vector<PqAdcTask> tasks;
  int64_t output_rows = 0;
  for (size_t task = 0; task < rows.size(); ++task) {
    for (int32_t m = 0; m < pq_m; ++m) {
      for (int32_t code = 0; code < ks; ++code) {
        tables[task][static_cast<size_t>(m * ks + code)] =
            0.03125f * static_cast<float>(task + 1) +
            0.0005f * static_cast<float>(m + 1) + 0.0001f * static_cast<float>(code);
      }
    }
    codes[task].resize(static_cast<size_t>(rows[task] * pq_m));
    for (int64_t row = 0; row < rows[task]; ++row) {
      for (int32_t m = 0; m < pq_m; ++m) {
        codes[task][static_cast<size_t>(row * pq_m + m)] =
            static_cast<uint8_t>((row * 37 + m * 53 + static_cast<int64_t>(task) * 97) % ks);
      }
    }
    tasks.push_back({tables[task].data(), codes[task].data(), rows[task], output_rows});
    output_rows += rows[task];
  }
  expect(output_rows == 1152, "true multi-task ADC real row count");

  expect_status(ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_NPU),
                "true multi-task ADC FORCE_NPU policy");
  auto telemetry = [&]() {
    auto* resources = ovvs::impl::rd(res.r);
    std::lock_guard<std::mutex> lock(resources->pq_adc_stats_mutex);
    return std::vector<int64_t>{resources->pq_adc_calls,
                                resources->pq_adc_logical_tasks,
                                resources->pq_adc_chunks,
                                resources->pq_adc_valid_rows,
                                resources->pq_adc_padded_rows,
                                resources->pq_adc_npu_requests,
                                resources->pq_adc_npu_rows,
                                resources->pq_adc_npu_capacity_rows,
                                resources->pq_adc_cpu_rows};
  };
  std::vector<int64_t> before = telemetry();
  for (int pass = 0; pass < 2; ++pass) {
    if (pass == 1) {
      for (size_t task = 0; task < tables.size(); ++task) {
        for (size_t i = 0; i < tables[task].size(); ++i) {
          tables[task][i] += 0.00001f * static_cast<float>((i + task) % 13 + 1);
        }
      }
    }
    std::vector<float> expected(static_cast<size_t>(output_rows));
    for (const PqAdcTask& task : tasks) {
      for (int64_t row = 0; row < task.rows; ++row) {
        expected[static_cast<size_t>(task.output_offset + row)] =
            independent_pq_adc(task.tables, task.codes + row * pq_m, pq_m, ks);
      }
    }
    std::vector<float> actual(static_cast<size_t>(output_rows + 2), -12345.0f);
    const ovvsStatus status = ovvs::impl::prim_pq_adc_batch(
        *ovvs::impl::rd(res.r), tasks.data(), static_cast<int64_t>(tasks.size()), pq_m, ks,
        actual.data() + 1, output_rows);
    expect(status != OVVS_STATUS_DEVICE_UNAVAILABLE,
           "NPU present but true multi-task ADC unavailable");
    expect_status(status, "true multi-task NPU ADC");
    expect(actual.front() == -12345.0f && actual.back() == -12345.0f,
           "true multi-task NPU ADC canaries");
    float max_abs_error = 0.0f;
    for (int64_t row = 0; row < output_rows; ++row) {
      max_abs_error = std::max(
          max_abs_error,
          std::fabs(actual[static_cast<size_t>(row + 1)] - expected[static_cast<size_t>(row)]));
    }
    expect(max_abs_error < 2e-2f,
           "true multi-task NPU ADC max error " + std::to_string(max_abs_error) +
               " pass=" + std::to_string(pass));
    ovvsDevice last = OVVS_DEVICE_CPU;
    expect_status(ovvsResourcesLastDevice(res.r, &last), "true multi-task ADC last device");
    expect(last == OVVS_DEVICE_NPU, "true multi-task FORCE_NPU ADC attribution");
    const std::vector<int64_t> after = telemetry();
    expect(after[0] - before[0] == 1, "true multi-task ADC records one logical call");
    expect(after[1] - before[1] == 6 && after[2] - before[2] == 6,
           "true multi-task ADC records six tasks and chunks");
    expect(after[3] - before[3] == 1152 && after[4] - before[4] == 384,
           "true multi-task ADC records valid and padded rows");
    expect(after[5] - before[5] == 3,
           "true multi-task ADC groups six chunks into three NPU requests");
    expect(after[6] - before[6] == 1152 && after[7] - before[7] == 1792 &&
               after[8] == before[8],
           "true multi-task ADC records real and submitted NPU rows without CPU fallback");
    before = after;
  }
}

OVVS_TEST(npu_pq_adc_split_and_bucket_regrouping_matches_reference_when_present) {
  using ovvs::impl::PqAdcTask;
  Res res;
  int32_t npu = 0;
  expect_status(ovvsResourcesNpuAvailable(res.r, &npu), "split ADC NPU probe");
  if (!npu) skip_test("NPU not available");

  constexpr int32_t pq_m = 8, ks = 256;
  /* Alternating bucket classes plus a >2048 task exercise both bucket regrouping and
     the planner's code-pointer advance for the split tail. Three 256-row chunks also
     force a four-slot request, making inactive-slot work visible in telemetry. */
  const std::vector<int64_t> rows{129, 3, 2049, 257, 255, 130};
  std::vector<std::vector<float>> tables(
      rows.size(), std::vector<float>(static_cast<size_t>(pq_m * ks)));
  std::vector<std::vector<uint8_t>> codes(rows.size());
  std::vector<PqAdcTask> tasks;
  int64_t output_rows = 0;
  for (size_t task = 0; task < rows.size(); ++task) {
    for (int32_t m = 0; m < pq_m; ++m) {
      for (int32_t code = 0; code < ks; ++code) {
        tables[task][static_cast<size_t>(m * ks + code)] =
            0.02f * static_cast<float>(task + 1) +
            0.0007f * static_cast<float>(m + 1) +
            0.00003f * static_cast<float>(code);
      }
    }
    codes[task].resize(static_cast<size_t>(rows[task] * pq_m));
    for (int64_t row = 0; row < rows[task]; ++row) {
      for (int32_t m = 0; m < pq_m; ++m) {
        codes[task][static_cast<size_t>(row * pq_m + m)] = static_cast<uint8_t>(
            (row * 71 + m * 29 + static_cast<int64_t>(task) * 43) % ks);
      }
    }
    tasks.push_back({tables[task].data(), codes[task].data(), rows[task], output_rows});
    output_rows += rows[task];
  }
  expect(output_rows == 2823, "split ADC real row count");

  auto telemetry = [&]() {
    auto* resources = ovvs::impl::rd(res.r);
    std::lock_guard<std::mutex> lock(resources->pq_adc_stats_mutex);
    return std::vector<int64_t>{resources->pq_adc_calls,
                                resources->pq_adc_logical_tasks,
                                resources->pq_adc_chunks,
                                resources->pq_adc_valid_rows,
                                resources->pq_adc_padded_rows,
                                resources->pq_adc_npu_requests,
                                resources->pq_adc_npu_rows,
                                resources->pq_adc_npu_capacity_rows,
                                resources->pq_adc_cpu_rows};
  };
  const std::vector<int64_t> before = telemetry();
  std::vector<float> expected(static_cast<size_t>(output_rows));
  for (const PqAdcTask& task : tasks) {
    for (int64_t row = 0; row < task.rows; ++row) {
      expected[static_cast<size_t>(task.output_offset + row)] =
          independent_pq_adc(task.tables, task.codes + row * pq_m, pq_m, ks);
    }
  }
  std::vector<float> actual(static_cast<size_t>(output_rows + 2), -12345.0f);
  expect_status(ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_NPU),
                "split ADC FORCE_NPU policy");
  const ovvsStatus status = ovvs::impl::prim_pq_adc_batch(
      *ovvs::impl::rd(res.r), tasks.data(), static_cast<int64_t>(tasks.size()), pq_m, ks,
      actual.data() + 1, output_rows);
  expect(status != OVVS_STATUS_DEVICE_UNAVAILABLE,
         "NPU present but split/regroup ADC unavailable");
  expect_status(status, "split/regroup NPU ADC");
  expect(actual.front() == -12345.0f && actual.back() == -12345.0f,
         "split/regroup NPU ADC canaries");
  float max_abs_error = 0.0f;
  for (int64_t row = 0; row < output_rows; ++row) {
    max_abs_error = std::max(
        max_abs_error,
        std::fabs(actual[static_cast<size_t>(row + 1)] - expected[static_cast<size_t>(row)]));
  }
  expect(max_abs_error < 2e-2f,
         "split/regroup NPU ADC max error " + std::to_string(max_abs_error));
  const std::vector<int64_t> after = telemetry();
  expect(after[0] - before[0] == 1 && after[1] - before[1] == 6 &&
             after[2] - before[2] == 7,
         "split/regroup ADC records one call, six tasks, and seven chunks");
  expect(after[3] - before[3] == 2823 && after[4] - before[4] == 761,
         "split/regroup ADC records real and intra-bucket padded rows");
  expect(after[5] - before[5] == 4 && after[6] - before[6] == 2823 &&
             after[7] - before[7] == 3840 && after[8] == before[8],
         "split/regroup ADC records four requests and actual device capacity");
}

OVVS_TEST(npu_pq_adc_matches_shave_when_present) {
  Res res;
  int32_t npu = 0;
  ovvsResourcesNpuAvailable(res.r, &npu);
  if (!npu) return;
  const int32_t pq_m = 4, ks = 8;
  const int64_t ncodes = 6;
  auto tables = make_data(pq_m, ks, 91);
  std::vector<uint8_t> codes(static_cast<size_t>(ncodes * pq_m));
  for (size_t i = 0; i < codes.size(); ++i) codes[i] = static_cast<uint8_t>(i % ks);
  std::vector<float> host(static_cast<size_t>(ncodes)), npuo(static_cast<size_t>(ncodes));
  for (int64_t i = 0; i < ncodes; ++i)
    host[static_cast<size_t>(i)] = ovvsShavePqAdc(tables.data(), codes.data() + i * pq_m, pq_m, ks);
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_NPU);
  const ovvsStatus st =
      ovvsPqAdcBatch(res.r, tables.data(), pq_m, ks, codes.data(), ncodes, npuo.data());
  expect(st != OVVS_STATUS_DEVICE_UNAVAILABLE, "NPU present but PQ ADC DEVICE_UNAVAILABLE");
  expect_status(st, "npu adc");
  ovvsDevice last = OVVS_DEVICE_CPU;
  expect_status(ovvsResourcesLastDevice(res.r, &last), "adc last");
  expect(last == OVVS_DEVICE_NPU, "FORCE_NPU adc last_device");
  std::printf("    npu_pq_adc last_device=npu\n");
  for (int64_t i = 0; i < ncodes; ++i) {
    expect(std::fabs(host[static_cast<size_t>(i)] - npuo[static_cast<size_t>(i)]) < 2e-2f, "adc vs shave");
  }
}

OVVS_TEST(npu_pq_adc_tile_boundaries_match_reference_when_present) {
  Res res;
  int32_t npu = 0;
  ovvsResourcesNpuAvailable(res.r, &npu);
  if (!npu) skip_test("NPU not available");

  constexpr int32_t pq_m = 8;
  constexpr int32_t ks = 256;
  std::vector<float> tables(static_cast<size_t>(pq_m * ks));
  for (int32_t m = 0; m < pq_m; ++m) {
    for (int32_t code = 0; code < ks; ++code) {
      tables[static_cast<size_t>(m * ks + code)] =
          0.03125f * static_cast<float>(m + 1) + 0.001f * static_cast<float>(code);
    }
  }

  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_NPU);
  const std::vector<int64_t> counts{127, 128, 129, 255, 256, 257};
  for (int pass = 0; pass < 2; ++pass) {
    if (pass == 1) {
      for (size_t i = 0; i < tables.size(); ++i) {
        tables[i] += 0.00025f * static_cast<float>((i % 11) + 1);
      }
    }
    for (int64_t ncodes : counts) {
      std::vector<uint8_t> codes(static_cast<size_t>(ncodes * pq_m));
      for (int64_t i = 0; i < ncodes; ++i) {
        for (int32_t m = 0; m < pq_m; ++m) {
          codes[static_cast<size_t>(i * pq_m + m)] =
              static_cast<uint8_t>((i * 37 + m * 53 + pass * 97) % ks);
        }
      }
      const uint8_t special_codes[] = {0, 127, 128, 255};
      for (size_t row = 0; row < 4 && row < static_cast<size_t>(ncodes); ++row) {
        for (int32_t m = 0; m < pq_m; ++m) {
          codes[row * static_cast<size_t>(pq_m) + static_cast<size_t>(m)] = special_codes[row];
        }
      }
      for (int32_t m = 0; m < pq_m; ++m) {
        codes[static_cast<size_t>((ncodes - 1) * pq_m + m)] = 255;
      }

      std::vector<float> host(static_cast<size_t>(ncodes));
      std::vector<float> npuo(static_cast<size_t>(ncodes + 2), -12345.0f);
      for (int64_t i = 0; i < ncodes; ++i) {
        float sum = 0.0f;
        for (int32_t m = 0; m < pq_m; ++m) {
          sum += tables[static_cast<size_t>(m * ks + codes[static_cast<size_t>(i * pq_m + m)])];
        }
        host[static_cast<size_t>(i)] = sum;
      }

      const ovvsStatus st =
          ovvsPqAdcBatch(res.r, tables.data(), pq_m, ks, codes.data(), ncodes, npuo.data() + 1);
      expect(st != OVVS_STATUS_DEVICE_UNAVAILABLE,
             "NPU tile-boundary PQ ADC DEVICE_UNAVAILABLE at n=" + std::to_string(ncodes));
      const std::string what = "NPU tile-boundary PQ ADC at n=" + std::to_string(ncodes);
      expect_status(st, what.c_str());
      ovvsDevice last = OVVS_DEVICE_CPU;
      expect_status(ovvsResourcesLastDevice(res.r, &last), "tile-boundary ADC last device");
      expect(last == OVVS_DEVICE_NPU, "tile-boundary FORCE_NPU ADC last_device");
      expect(npuo.front() == -12345.0f && npuo.back() == -12345.0f,
             "tile-boundary ADC output canaries");

      float max_abs_error = 0.0f;
      for (int64_t i = 0; i < ncodes; ++i) {
        max_abs_error = std::max(
            max_abs_error,
            std::fabs(host[static_cast<size_t>(i)] - npuo[static_cast<size_t>(i + 1)]));
      }
      expect(max_abs_error < 2e-2f,
             "tile-boundary ADC max error " + std::to_string(max_abs_error) +
                 " at n=" + std::to_string(ncodes) + " pass=" + std::to_string(pass));
    }
  }
}

OVVS_TEST(npu_pq_adc_center_scales_unsafe_accumulation_when_present) {
  Res res;
  int32_t npu = 0;
  ovvsResourcesNpuAvailable(res.r, &npu);
  if (!npu) skip_test("NPU not available");

  constexpr int32_t pq_m = 8;
  constexpr int32_t ks = 256;
  constexpr int64_t ncodes = 129;
  std::vector<float> tables(static_cast<size_t>(pq_m * ks), 8192.0f);
  std::vector<uint8_t> codes(static_cast<size_t>(ncodes * pq_m));
  for (size_t i = 0; i < codes.size(); ++i) codes[i] = static_cast<uint8_t>(i % ks);
  std::vector<float> expected(static_cast<size_t>(ncodes));
  std::vector<float> actual(static_cast<size_t>(ncodes), -1.0f);
  std::fill(expected.begin(), expected.end(), 65536.0f);

  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  expect_status(ovvsPqAdcBatch(res.r, tables.data(), pq_m, ks, codes.data(), ncodes,
                               actual.data()),
                "unsafe CPU ADC reference");
  for (int64_t i = 0; i < ncodes; ++i) {
    expect(actual[static_cast<size_t>(i)] == expected[static_cast<size_t>(i)],
           "unsafe CPU ADC reference result");
  }
  std::fill(actual.begin(), actual.end(), -1.0f);
  int32_t fallbacks_before = 0;
  int32_t compile_fails_before = 0;
  expect_status(ovvsResourcesNpuFallbacks(res.r, &fallbacks_before),
                "unsafe NPU ADC fallback count before");
  expect_status(ovvsResourcesNpuCompileFails(res.r, &compile_fails_before),
                "unsafe NPU ADC compile-fail count before");
  auto* resources = ovvs::impl::rd(res.r);
  int64_t scaled_chunks_before = 0;
  int64_t scaled_rows_before = 0;
  {
    std::lock_guard<std::mutex> lock(resources->pq_adc_stats_mutex);
    scaled_chunks_before = resources->pq_adc_npu_transformed_chunks;
    scaled_rows_before = resources->pq_adc_npu_transformed_rows;
  }

  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_NPU);
  const ovvsStatus forced =
      ovvsPqAdcBatch(res.r, tables.data(), pq_m, ks, codes.data(), ncodes, actual.data());
  expect_status(forced, "centered constant-offset NPU ADC");
  for (int64_t i = 0; i < ncodes; ++i) {
    const float tolerance = 2e-4f * expected[static_cast<size_t>(i)];
    expect(std::fabs(actual[static_cast<size_t>(i)] - expected[static_cast<size_t>(i)]) <=
               tolerance,
           "centered constant-offset NPU ADC result");
  }
  ovvsDevice last = OVVS_DEVICE_CPU;
  expect_status(ovvsResourcesLastDevice(res.r, &last), "centered ADC last device");
  expect(last == OVVS_DEVICE_NPU, "centered ADC must attribute NPU");
  int32_t fallbacks_after = 0;
  int32_t compile_fails_after = 0;
  expect_status(ovvsResourcesNpuFallbacks(res.r, &fallbacks_after),
                "unsafe NPU ADC fallback count after");
  expect_status(ovvsResourcesNpuCompileFails(res.r, &compile_fails_after),
                "unsafe NPU ADC compile-fail count after");
  expect(fallbacks_after == fallbacks_before, "centered forced ADC fallback count");
  expect(compile_fails_after == compile_fails_before,
         "centered forced ADC must not count as a compile failure");
  {
    std::lock_guard<std::mutex> lock(resources->pq_adc_stats_mutex);
    expect(resources->pq_adc_npu_transformed_chunks == scaled_chunks_before + 1,
           "centered forced ADC records one transformed chunk");
    expect(resources->pq_adc_npu_transformed_rows == scaled_rows_before + ncodes,
           "centered forced ADC records transformed rows");
  }

  std::fill(actual.begin(), actual.end(), -1.0f);
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_AUTO);
  expect_status(ovvsPqAdcBatch(res.r, tables.data(), pq_m, ks, codes.data(), ncodes,
                               actual.data()),
                "unsafe AUTO ADC CPU fallback");
  last = OVVS_DEVICE_NPU;
  expect_status(ovvsResourcesLastDevice(res.r, &last), "unsafe AUTO ADC last device");
  expect(last == OVVS_DEVICE_CPU, "unsafe AUTO ADC must attribute CPU fallback");
  for (int64_t i = 0; i < ncodes; ++i) {
    expect(actual[static_cast<size_t>(i)] == expected[static_cast<size_t>(i)],
           "unsafe AUTO ADC CPU result");
  }
}

OVVS_TEST(npu_pq_adc_scaled_span_matches_reference_when_present) {
  Res res;
  int32_t npu = 0;
  expect_status(ovvsResourcesNpuAvailable(res.r, &npu), "scaled-span NPU probe");
  if (!npu) skip_test("NPU not available");

  constexpr int32_t pq_m = 8, ks = 256;
  constexpr int64_t ncodes = 257;
  std::vector<float> tables(static_cast<size_t>(pq_m * ks));
  for (int32_t m = 0; m < pq_m; ++m) {
    const float base = -200000.0f + 1000.0f * static_cast<float>(m);
    for (int32_t code = 0; code < ks; ++code) {
      tables[static_cast<size_t>(m * ks + code)] =
          base + 50.0f * static_cast<float>(code);
    }
  }
  std::vector<uint8_t> codes(static_cast<size_t>(ncodes * pq_m));
  for (int64_t row = 0; row < ncodes; ++row) {
    for (int32_t m = 0; m < pq_m; ++m) {
      codes[static_cast<size_t>(row * pq_m + m)] =
          static_cast<uint8_t>((row * 37 + m * 29) % ks);
    }
  }
  std::vector<float> expected(static_cast<size_t>(ncodes));
  std::vector<float> actual(static_cast<size_t>(ncodes + 2), -12345.0f);
  for (int64_t row = 0; row < ncodes; ++row) {
    expected[static_cast<size_t>(row)] =
        independent_pq_adc(tables.data(), codes.data() + row * pq_m, pq_m, ks);
  }

  expect_status(ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_NPU),
                "scaled-span FORCE_NPU policy");
  expect_status(ovvsPqAdcBatch(res.r, tables.data(), pq_m, ks, codes.data(), ncodes,
                               actual.data() + 1),
                "scaled-span NPU ADC");
  expect(actual.front() == -12345.0f && actual.back() == -12345.0f,
         "scaled-span ADC canaries");
  float max_abs_error = 0.0f;
  for (int64_t row = 0; row < ncodes; ++row) {
    max_abs_error = std::max(
        max_abs_error,
        std::fabs(actual[static_cast<size_t>(row + 1)] -
                  expected[static_cast<size_t>(row)]));
  }
  constexpr int64_t shortlist = 32;
  std::vector<int64_t> expected_ids(static_cast<size_t>(shortlist));
  std::vector<int64_t> actual_ids(static_cast<size_t>(shortlist));
  std::vector<int64_t> expected_order(static_cast<size_t>(ncodes));
  std::vector<int64_t> actual_order(static_cast<size_t>(ncodes));
  std::iota(expected_order.begin(), expected_order.end(), 0);
  std::iota(actual_order.begin(), actual_order.end(), 0);
  std::partial_sort(expected_order.begin(), expected_order.begin() + shortlist,
                    expected_order.end(), [&](int64_t a, int64_t b) {
                      return expected[static_cast<size_t>(a)] <
                             expected[static_cast<size_t>(b)];
                    });
  std::partial_sort(actual_order.begin(), actual_order.begin() + shortlist,
                    actual_order.end(), [&](int64_t a, int64_t b) {
                      return actual[static_cast<size_t>(a + 1)] <
                             actual[static_cast<size_t>(b + 1)];
                    });
  std::copy_n(expected_order.begin(), shortlist, expected_ids.begin());
  std::copy_n(actual_order.begin(), shortlist, actual_ids.begin());
  int32_t overlap = 0;
  for (int64_t expected_id : expected_ids) {
    if (std::find(actual_ids.begin(), actual_ids.end(), expected_id) != actual_ids.end()) {
      ++overlap;
    }
  }
  std::printf("    scaled_span max_abs_error=%.3f top32_overlap=%d/32\n",
              max_abs_error, overlap);
  expect(max_abs_error < 128.0f,
         "scaled-span ADC max error " + std::to_string(max_abs_error));
  expect(overlap >= 31, "scaled-span ADC top-32 overlap " + std::to_string(overlap));

  for (int32_t m = 0; m < pq_m; ++m) {
    const float base = -200000.0f + 1000.0f * static_cast<float>(m);
    for (int32_t code = 0; code < ks; ++code) {
      tables[static_cast<size_t>(m * ks + code)] =
          base + 1000.0f * static_cast<float>(code);
    }
  }
  std::fill(actual.begin(), actual.end(), -12345.0f);
  int32_t fallbacks_before = 0;
  int32_t compile_fails_before = 0;
  expect_status(ovvsResourcesNpuFallbacks(res.r, &fallbacks_before),
                "excessive-scale fallback count before");
  expect_status(ovvsResourcesNpuCompileFails(res.r, &compile_fails_before),
                "excessive-scale compile count before");
  expect(ovvsPqAdcBatch(res.r, tables.data(), pq_m, ks, codes.data(), ncodes,
                        actual.data() + 1) == OVVS_STATUS_DEVICE_UNAVAILABLE,
         "excessive PQ scaling fails closed");
  for (float value : actual) {
    expect(value == -12345.0f, "excessive PQ scaling publishes no output");
  }
  int32_t fallbacks_after = 0;
  int32_t compile_fails_after = 0;
  expect_status(ovvsResourcesNpuFallbacks(res.r, &fallbacks_after),
                "excessive-scale fallback count after");
  expect_status(ovvsResourcesNpuCompileFails(res.r, &compile_fails_after),
                "excessive-scale compile count after");
  expect(fallbacks_after == fallbacks_before + 1,
         "excessive PQ scaling records one forced fallback");
  expect(compile_fails_after == compile_fails_before,
         "excessive PQ scaling is not a compile failure");

  const float huge_offset = std::numeric_limits<float>::max() / 4.0f;
  std::fill(tables.begin(), tables.end(), huge_offset);
  std::fill(actual.begin(), actual.end(), -12345.0f);
  auto* resources = ovvs::impl::rd(res.r);
  const int32_t runtime_fails_before = resources->npu_runtime_fails;
  int64_t requests_before = 0;
  int64_t rows_before = 0;
  int64_t scaled_chunks_before = 0;
  int64_t scaled_rows_before = 0;
  {
    std::lock_guard<std::mutex> lock(resources->pq_adc_stats_mutex);
    requests_before = resources->pq_adc_npu_requests;
    rows_before = resources->pq_adc_npu_rows;
    scaled_chunks_before = resources->pq_adc_npu_transformed_chunks;
    scaled_rows_before = resources->pq_adc_npu_transformed_rows;
  }
  fallbacks_before = fallbacks_after;
  compile_fails_before = compile_fails_after;
  expect(ovvsPqAdcBatch(res.r, tables.data(), pq_m, ks, codes.data(), ncodes,
                        actual.data() + 1) == OVVS_STATUS_DEVICE_UNAVAILABLE,
         "restored-f32 overflow fails closed after inference");
  for (float value : actual) {
    expect(value == -12345.0f, "restored-f32 overflow publishes no output");
  }
  expect_status(ovvsResourcesNpuFallbacks(res.r, &fallbacks_after),
                "restored-f32 overflow fallback count after");
  expect_status(ovvsResourcesNpuCompileFails(res.r, &compile_fails_after),
                "restored-f32 overflow compile count after");
  expect(fallbacks_after == fallbacks_before + 1,
         "restored-f32 overflow records one forced fallback");
  expect(compile_fails_after == compile_fails_before,
         "restored-f32 overflow is not a compile failure");
  expect(resources->npu_runtime_fails == runtime_fails_before + 1,
         "restored-f32 overflow records one runtime failure");
  {
    std::lock_guard<std::mutex> lock(resources->pq_adc_stats_mutex);
    expect(resources->pq_adc_npu_requests == requests_before + 1,
           "restored-f32 overflow records the executed request");
    expect(resources->pq_adc_npu_rows == rows_before + ncodes,
           "restored-f32 overflow records executed rows");
    expect(resources->pq_adc_npu_transformed_chunks == scaled_chunks_before + 1,
           "restored-f32 overflow records transformed chunk");
    expect(resources->pq_adc_npu_transformed_rows == scaled_rows_before + ncodes,
           "restored-f32 overflow records transformed rows");
  }
  ovvsDevice last = OVVS_DEVICE_CPU;
  expect_status(ovvsResourcesLastDevice(res.r, &last),
                "restored-f32 overflow last device");
  expect(last == OVVS_DEVICE_NPU,
         "restored-f32 overflow preserves prior successful attribution");
}

OVVS_TEST(npu_pq_adc_multitile_output_is_atomic_when_present) {
  Res res;
  int32_t npu = 0;
  ovvsResourcesNpuAvailable(res.r, &npu);
  if (!npu) skip_test("NPU not available");

  constexpr int32_t pq_m = 8;
  constexpr int32_t ks = 256;
  constexpr int64_t ncodes = 256;
  std::vector<float> tables(static_cast<size_t>(pq_m * ks), 1.0f);
  for (int32_t m = 0; m < pq_m; ++m) {
    tables[static_cast<size_t>(m * ks + 255)] = 8187.999f;
  }
  std::vector<uint8_t> codes(static_cast<size_t>(ncodes * pq_m), 0);
  for (int64_t i = 128; i < ncodes; ++i) {
    for (int32_t m = 0; m < pq_m; ++m) {
      codes[static_cast<size_t>(i * pq_m + m)] = 255;
    }
  }
  std::vector<float> actual(static_cast<size_t>(ncodes + 2), -12345.0f);

  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_NPU);
  const ovvsStatus status =
      ovvsPqAdcBatch(res.r, tables.data(), pq_m, ks, codes.data(), ncodes, actual.data() + 1);
  expect(actual.front() == -12345.0f && actual.back() == -12345.0f,
         "multi-tile ADC output canaries");
  if (status == OVVS_STATUS_DEVICE_UNAVAILABLE) {
    for (int64_t i = 0; i < ncodes; ++i) {
      expect(actual[static_cast<size_t>(i + 1)] == -12345.0f,
             "failed multi-tile ADC must publish no partial output");
    }
    return;
  }
  expect_status(status, "multi-tile ADC atomic success");
  for (int64_t i = 0; i < ncodes; ++i) {
    expect(actual[static_cast<size_t>(i + 1)] != -12345.0f,
           "successful multi-tile ADC must publish every output");
  }
}

static const char* dev_name(ovvsDevice d) {
  switch (d) {
    case OVVS_DEVICE_NPU:
      return "npu";
    case OVVS_DEVICE_GPU:
      return "gpu";
    case OVVS_DEVICE_CPU:
      return "cpu";
    default:
      return "other";
  }
}

OVVS_TEST(topk_npu_matches_cpu_when_present) {
  Res res;
  int32_t npu = 0;
  ovvsResourcesNpuAvailable(res.r, &npu);
  if (!npu) return;
  auto scores = make_data(4, 16, 91);
  std::vector<int64_t> ic(40), in(40);
  std::vector<float> vc(40), vn(40);
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  expect_status(ovvsTopk(res.r, scores.data(), 4, 16, 5, ic.data(), vc.data(), 0), "cpu topk");
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_NPU);
  expect_status(ovvsTopk(res.r, scores.data(), 4, 16, 5, in.data(), vn.data(), 0), "npu topk");
  ovvsDevice last = OVVS_DEVICE_CPU;
  ovvsResourcesLastDevice(res.r, &last);
  expect(last == OVVS_DEVICE_NPU, std::string("topk last device ") + dev_name(last));
  for (int i = 0; i < 20; ++i) {
    expect(ic[static_cast<size_t>(i)] == in[static_cast<size_t>(i)], "npu topk idx");
    expect(std::fabs(vc[static_cast<size_t>(i)] - vn[static_cast<size_t>(i)]) < 2e-2f, "npu topk val");
  }
}

OVVS_TEST(topk_npu_rejects_unsafe_fp16_range_when_present) {
  Res res;
  int32_t npu = 0;
  ovvsResourcesNpuAvailable(res.r, &npu);
  if (!npu) return;

  const int64_t rows = 4, cols = 16, k = 5;
  std::vector<float> scores(static_cast<size_t>(rows * cols));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      scores[static_cast<size_t>(r * cols + c)] = 64768.0f + static_cast<float>(c * 32);
    }
  }
  std::vector<int64_t> indices(static_cast<size_t>(rows * k));
  std::vector<float> values(static_cast<size_t>(rows * k));
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_NPU);
  expect_status(ovvsTopk(res.r, scores.data(), rows, cols, k, indices.data(), values.data(), 1),
                "npu finite-boundary topk");
  for (float value : values) {
    expect(std::isfinite(value), "npu finite-boundary topk finite");
    expect(std::fabs(value) < 65504.0f, "npu finite-boundary topk range");
  }

  scores[0] = 65536.0f;
  expect(ovvsTopk(res.r, scores.data(), rows, cols, k, indices.data(), values.data(), 1) ==
             OVVS_STATUS_DEVICE_UNAVAILABLE,
         "npu topk must reject fp16 overflow input");

  scores[0] = std::numeric_limits<float>::quiet_NaN();
  expect(ovvsTopk(res.r, scores.data(), rows, cols, k, indices.data(), values.data(), 1) ==
             OVVS_STATUS_DEVICE_UNAVAILABLE,
         "npu topk must reject non-finite input");
}

OVVS_TEST(topk_gpu_matches_cpu_when_present) {
  Res res;
  int32_t gpu = 0;
  ovvsResourcesGpuAvailable(res.r, &gpu);
  if (!gpu) return;
  auto scores = make_data(4, 16, 92);
  std::vector<int64_t> ic(40), ig(40);
  std::vector<float> vc(40), vg(40);
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  expect_status(ovvsTopk(res.r, scores.data(), 4, 16, 5, ic.data(), vc.data(), 0), "cpu topk");
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU);
  expect_status(ovvsTopk(res.r, scores.data(), 4, 16, 5, ig.data(), vg.data(), 0), "gpu topk");
  ovvsDevice last = OVVS_DEVICE_CPU;
  ovvsResourcesLastDevice(res.r, &last);
  expect(last == OVVS_DEVICE_GPU, std::string("gpu topk last ") + dev_name(last));
  for (int i = 0; i < 20; ++i) {
    expect(ic[static_cast<size_t>(i)] == ig[static_cast<size_t>(i)], "gpu topk idx");
  }
}

OVVS_TEST(gather_npu_matches_cpu_when_present) {
  Res res;
  int32_t npu = 0;
  ovvsResourcesNpuAvailable(res.r, &npu);
  if (!npu) return;
  const int64_t n = 10, dim = 8, nidx = 4;
  auto src = make_data(n, dim, 93);
  const int64_t idxv[4] = {1, 4, 9, 0};
  std::vector<float> cpu(static_cast<size_t>(nidx * dim)), np(static_cast<size_t>(nidx * dim));
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  expect_status(ovvsGatherRows(res.r, src.data(), n, dim, idxv, nidx, cpu.data()), "cpu gather");
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_NPU);
  expect_status(ovvsGatherRows(res.r, src.data(), n, dim, idxv, nidx, np.data()), "npu gather");
  ovvsDevice last = OVVS_DEVICE_CPU;
  ovvsResourcesLastDevice(res.r, &last);
  expect(last == OVVS_DEVICE_NPU, std::string("gather last ") + dev_name(last));
  for (size_t i = 0; i < cpu.size(); ++i) expect(std::fabs(cpu[i] - np[i]) < 2e-2f, "npu gather");
}

OVVS_TEST(gather_gpu_matches_cpu_when_present) {
  Res res;
  int32_t gpu = 0;
  ovvsResourcesGpuAvailable(res.r, &gpu);
  if (!gpu) return;
  const int64_t n = 10, dim = 8, nidx = 4;
  auto src = make_data(n, dim, 94);
  const int64_t idxv[4] = {2, 3, 7, 1};
  std::vector<float> cpu(static_cast<size_t>(nidx * dim)), g(static_cast<size_t>(nidx * dim));
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  expect_status(ovvsGatherRows(res.r, src.data(), n, dim, idxv, nidx, cpu.data()), "cpu gather");
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU);
  expect_status(ovvsGatherRows(res.r, src.data(), n, dim, idxv, nidx, g.data()), "gpu gather");
  ovvsDevice last = OVVS_DEVICE_CPU;
  ovvsResourcesLastDevice(res.r, &last);
  expect(last == OVVS_DEVICE_GPU, std::string("gpu gather last ") + dev_name(last));
  for (size_t i = 0; i < cpu.size(); ++i) expect(std::fabs(cpu[i] - g[i]) < 2e-2f, "gpu gather");
}

OVVS_TEST(shave_topk_matches_cpu_row) {
  auto scores = make_data(1, 12, 95);
  std::vector<int32_t> idx(4);
  std::vector<float> val(4);
  ovvsShaveTopkSmallest(scores.data(), 12, 4, idx.data(), val.data());
  Res res;
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  std::vector<int64_t> ic(4);
  std::vector<float> vc(4);
  expect_status(ovvsTopk(res.r, scores.data(), 1, 12, 4, ic.data(), vc.data(), 0), "cpu");
  for (int i = 0; i < 4; ++i) {
    expect(idx[static_cast<size_t>(i)] == static_cast<int32_t>(ic[static_cast<size_t>(i)]), "shave idx");
  }
}

OVVS_TEST(gather_rows_matches_index) {
  Res res;
  const int64_t n = 9, dim = 6, nidx = 4;
  auto src = make_data(n, dim, 21);
  const int64_t idxv[4] = {0, 3, 8, 1};
  std::vector<float> out(static_cast<size_t>(nidx * dim));
  expect_status(ovvsGatherRows(res.r, src.data(), n, dim, idxv, nidx, out.data()), "gather");
  for (int64_t i = 0; i < nidx; ++i) {
    for (int64_t d = 0; d < dim; ++d) {
      expect(out[static_cast<size_t>(i * dim + d)] == src[static_cast<size_t>(idxv[i] * dim + d)],
             "gather");
    }
  }
}

OVVS_TEST(force_devices_last_device_honest) {
  Res res;
  const int64_t m = 8, n = 8, k = 8;
  auto A = make_data(m, k, 5);
  auto B = make_data(n, k, 6);
  std::vector<float> C(static_cast<size_t>(m * n));
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_CPU);
  expect_status(ovvsGemm(res.r, A.data(), B.data(), C.data(), m, n, k, 1), "cpu");
  ovvsDevice last = OVVS_DEVICE_AUTO;
  ovvsResourcesLastDevice(res.r, &last);
  expect(last == OVVS_DEVICE_CPU, "force cpu last");
  int32_t npu = 0, gpu = 0;
  ovvsResourcesNpuAvailable(res.r, &npu);
  ovvsResourcesGpuAvailable(res.r, &gpu);
  if (npu) {
    ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_NPU);
    expect_status(ovvsGemm(res.r, A.data(), B.data(), C.data(), m, n, k, 1), "npu");
    ovvsResourcesLastDevice(res.r, &last);
    expect(last == OVVS_DEVICE_NPU, "force npu last");
  }
  if (gpu) {
    ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_GPU);
    expect_status(ovvsGemm(res.r, A.data(), B.data(), C.data(), m, n, k, 1), "gpu");
    ovvsResourcesLastDevice(res.r, &last);
    expect(last == OVVS_DEVICE_GPU, "force gpu last");
  }
}

OVVS_TEST(pairwise_l2_matches_definition) {
  Res res;
  const int64_t nx = 3, ny = 4, dim = 5;
  auto X = make_data(nx, dim, 31);
  auto Y = make_data(ny, dim, 32);
  std::vector<float> out(static_cast<size_t>(nx * ny));
  expect_status(ovvsPairwiseDistance(res.r, OVVS_METRIC_L2_EXPANDED, X.data(), nx, Y.data(), ny, dim,
                                     out.data(), 2.f),
                "pair");
  for (int64_t i = 0; i < nx; ++i) {
    for (int64_t j = 0; j < ny; ++j) {
      float s = 0.f;
      for (int64_t d = 0; d < dim; ++d) {
        const float t = X[static_cast<size_t>(i * dim + d)] - Y[static_cast<size_t>(j * dim + d)];
        s += t * t;
      }
      expect(std::fabs(out[static_cast<size_t>(i * ny + j)] - s) < 1e-3f, "pair l2");
    }
  }
}
