#include "ovvs/ovvs.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static const char* last_name(ovvsDevice d) {
  if (d == OVVS_DEVICE_NPU) return "npu";
  if (d == OVVS_DEVICE_GPU) return "gpu";
  return "cpu";
}

int main(int argc, char** argv) {
  std::string op = "gemm";
  int64_t M = 64, N = 128, K = 32;
  if (argc > 1) op = argv[1];
  if (op == "large" || op == "gemm-large") {
    op = "gemm";
    M = 100000;
    N = 32;
    K = 768;
  }
  if (op == "search" || op == "gemm-search") {
    op = "gemm";
    M = 32;
    N = 100000;
    K = 768;
  }
  if (op == "gemm-f16" || op == "gemm-i8") {
    if (argc <= 4) {
      M = 256;
      N = 256;
      K = 128;
    }
  }
  if (op == "pq-adc-scale" && argc <= 4) {
    M = 131072; /* codes */
    N = 8;      /* PQ subquantizers */
    K = 256;    /* codes per subquantizer */
  }
  if (argc > 4) {
    M = std::strtoll(argv[2], nullptr, 10);
    N = std::strtoll(argv[3], nullptr, 10);
    K = std::strtoll(argv[4], nullptr, 10);
  }
  ovvsResources_t res = nullptr;
  if (ovvsResourcesCreate(&res) != OVVS_STATUS_SUCCESS) return 1;

  char sku[64] = {0};
  ovvsResourcesSku(res, sku, 64);
  int32_t npu = 0, gpu = 0;
  ovvsResourcesNpuAvailable(res, &npu);
  ovvsResourcesGpuAvailable(res, &gpu);

  int64_t e0 = 0, e1 = 0;
  const ovvsStatus es0 = ovvsResourcesEnergyUj(res, &e0);
  std::printf(
      "{\n  \"sku\": \"%s\",\n  \"op\": \"%s\",\n  \"shape\": [%lld, %lld, %lld],\n  \"npu\": %s,\n  "
      "\"gpu\": %s,\n  \"sycl\": %s,\n  \"energy_probe\": \"%s\",\n  \"runs\": [\n",
      sku, op.c_str(), static_cast<long long>(M), static_cast<long long>(N), static_cast<long long>(K),
      npu ? "true" : "false", gpu ? "true" : "false", ovvsSyclEnabled() ? "true" : "false",
      es0 == OVVS_STATUS_SUCCESS ? "success" : "unsupported");

  const ovvsPolicy policies[] = {OVVS_POLICY_FORCE_CPU, OVVS_POLICY_FORCE_NPU, OVVS_POLICY_FORCE_GPU};
  const char* names[] = {"cpu", "npu", "gpu"};
  bool first = true;
  for (int p = 0; p < 3; ++p) {
    ovvsResourcesSetPolicy(res, policies[p]);
    std::vector<float> A, B, C;
    std::vector<uint8_t> pq_codes;
    if (op == "pq-adc-scale") {
      if (M <= 0 || N <= 0 || N > INT32_MAX || K <= 0 || K > 256 ||
          M > std::numeric_limits<int64_t>::max() / N ||
          N > std::numeric_limits<int64_t>::max() / K ||
          static_cast<uint64_t>(M) >
              static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
                  static_cast<uint64_t>(N) ||
          static_cast<uint64_t>(N) >
              static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
                  static_cast<uint64_t>(K)) {
        ovvsResourcesDestroy(res);
        return 2;
      }
      B.resize(static_cast<size_t>(N * K));
      pq_codes.resize(static_cast<size_t>(M * N));
      C.resize(static_cast<size_t>(M));
      for (int64_t m = 0; m < N; ++m) {
        const float base = 20000.0f + 1000.0f * static_cast<float>(m);
        for (int64_t code = 0; code < K; ++code) {
          B[static_cast<size_t>(m * K + code)] =
              base + 50.0f * static_cast<float>(code);
        }
      }
      for (int64_t row = 0; row < M; ++row) {
        for (int64_t m = 0; m < N; ++m) {
          pq_codes[static_cast<size_t>(row * N + m)] =
              static_cast<uint8_t>((row * 37 + m * 29) % K);
        }
      }
    } else {
      A.assign(static_cast<size_t>(M * K), 0.1f);
      B.assign(static_cast<size_t>(N * K), 0.2f);
      C.assign(static_cast<size_t>(M * N), 0.f);
      for (int64_t i = 0; i < M * K && i < 100000; ++i)
        A[static_cast<size_t>(i)] = 0.01f * static_cast<float>(i % 17);
      for (int64_t i = 0; i < N * K; ++i)
        B[static_cast<size_t>(i)] = 0.02f * static_cast<float>(i % 13);
    }

    auto run_op = [&]() {
      ovvsStatus st = OVVS_STATUS_SUCCESS;
      if (op == "gemm") {
        st = ovvsGemm(res, A.data(), B.data(), C.data(), M, N, K, 1);
      } else if (op == "gemm-f16") {
        st = ovvsGemmEx(res, A.data(), B.data(), C.data(), M, N, K, 1, OVVS_DTYPE_F16);
      } else if (op == "gemm-i8") {
        st = ovvsGemmEx(res, A.data(), B.data(), C.data(), M, N, K, 1, OVVS_DTYPE_I8);
      } else if (op == "topk") {
        std::vector<int64_t> idx(static_cast<size_t>(M * 10));
        std::vector<float> val(static_cast<size_t>(M * 10));
        st = ovvsTopk(res, C.data(), M, N, 10, idx.data(), val.data(), 0);
      } else if (op == "gather") {
        const int64_t nidx = std::min<int64_t>(4096, M);
        std::vector<int64_t> idx(static_cast<size_t>(nidx));
        for (int64_t i = 0; i < nidx; ++i) idx[static_cast<size_t>(i)] = i % M;
        std::vector<float> out(static_cast<size_t>(nidx * K));
        st = ovvsGatherRows(res, A.data(), M, K, idx.data(), nidx, out.data());
      } else if (op == "pq-adc-scale") {
        st = ovvsPqAdcBatch(res, B.data(), static_cast<int32_t>(N),
                            static_cast<int32_t>(K), pq_codes.data(), M, C.data());
      }
      return st;
    };
    (void)run_op();
    const double t0 = now_ms();
    ovvsStatus st = run_op();
    const double t1 = now_ms();
    const ovvsStatus st_hot = run_op();
    const double t2 = now_ms();
    if (st == OVVS_STATUS_SUCCESS) st = st_hot;
    if (!first) std::printf(",\n");
    first = false;
    ovvsDevice last = OVVS_DEVICE_CPU;
    ovvsResourcesLastDevice(res, &last);
    if (st == OVVS_STATUS_SUCCESS) {
      std::printf(
          "    {\"requested\": \"%s\", \"last_device\": \"%s\", \"ms\": %.4f, "
          "\"ms_hot\": %.4f, \"status\": \"%s\"}",
          names[p], last_name(last), t1 - t0, t2 - t1, ovvsStatusString(st));
    } else {
      std::printf(
          "    {\"requested\": \"%s\", \"last_device\": null, \"ms\": %.4f, "
          "\"ms_hot\": %.4f, \"status\": \"%s\"}",
          names[p], t1 - t0, t2 - t1, ovvsStatusString(st));
    }
  }
  const ovvsStatus es1 = ovvsResourcesEnergyUj(res, &e1);
  std::printf("\n  ],\n  \"energy_uj_before\": %lld,\n  \"energy_uj_after\": %lld,\n  \"energy_status\": \"%s\"\n}\n",
              static_cast<long long>(e0), static_cast<long long>(e1),
              ovvsStatusString(es1 == OVVS_STATUS_SUCCESS ? es1 : es0));
  ovvsResourcesDestroy(res);
  return 0;
}
