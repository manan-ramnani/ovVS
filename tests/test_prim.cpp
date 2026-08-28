#include "test_harness.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <numeric>

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
      return scores[static_cast<size_t>(r * cols + a)] < scores[static_cast<size_t>(r * cols + b)];
    });
    for (int64_t t = 0; t < k; ++t) {
      expect(idx[static_cast<size_t>(r * k + t)] == order[static_cast<size_t>(t)], "topk idx");
      expect(std::fabs(val[static_cast<size_t>(r * k + t)] -
                       scores[static_cast<size_t>(r * cols + order[static_cast<size_t>(t)])]) < 1e-6f,
             "topk val");
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

OVVS_TEST(npu_pq_adc_rejects_unsafe_accumulation_when_present) {
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

  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_FORCE_NPU);
  const ovvsStatus forced =
      ovvsPqAdcBatch(res.r, tables.data(), pq_m, ks, codes.data(), ncodes, actual.data());
  expect(forced == OVVS_STATUS_DEVICE_UNAVAILABLE,
         "unsafe NPU ADC accumulation must fail closed");
  for (float value : actual) expect(value == -1.0f, "failed NPU ADC must not write output");
  ovvsDevice last = OVVS_DEVICE_NPU;
  expect_status(ovvsResourcesLastDevice(res.r, &last), "unsafe rejected ADC last device");
  expect(last == OVVS_DEVICE_CPU, "rejected NPU ADC must preserve prior device attribution");
  int32_t fallbacks_after = 0;
  int32_t compile_fails_after = 0;
  expect_status(ovvsResourcesNpuFallbacks(res.r, &fallbacks_after),
                "unsafe NPU ADC fallback count after");
  expect_status(ovvsResourcesNpuCompileFails(res.r, &compile_fails_after),
                "unsafe NPU ADC compile-fail count after");
  expect(fallbacks_after == fallbacks_before + 1, "unsafe forced ADC fallback count");
  expect(compile_fails_after == compile_fails_before,
         "unsafe forced ADC must not count as a compile failure");

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
