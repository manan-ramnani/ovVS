#include "ovvs/ovvs.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
  if (op == "gemm-f16" || op == "gemm-i8") {
    if (argc <= 4) {
      M = 256;
      N = 256;
      K = 128;
    }
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
    std::vector<float> A(static_cast<size_t>(M * K), 0.1f), B(static_cast<size_t>(N * K), 0.2f),
        C(static_cast<size_t>(M * N), 0.f);
    for (int64_t i = 0; i < M * K && i < 100000; ++i)
      A[static_cast<size_t>(i)] = 0.01f * static_cast<float>(i % 17);
    for (int64_t i = 0; i < N * K; ++i) B[static_cast<size_t>(i)] = 0.02f * static_cast<float>(i % 13);

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
      }
      return st;
    };
    (void)run_op();
    const double t0 = now_ms();
    ovvsStatus st = run_op();
    const double t1 = now_ms();
    if (!first) std::printf(",\n");
    first = false;
    ovvsDevice last = OVVS_DEVICE_CPU;
    ovvsResourcesLastDevice(res, &last);
    std::printf(
        "    {\"requested\": \"%s\", \"last_device\": \"%s\", \"ms\": %.4f, \"status\": \"%s\"}",
        names[p], last_name(last), t1 - t0, ovvsStatusString(st));
  }
  const ovvsStatus es1 = ovvsResourcesEnergyUj(res, &e1);
  std::printf("\n  ],\n  \"energy_uj_before\": %lld,\n  \"energy_uj_after\": %lld,\n  \"energy_status\": \"%s\"\n}\n",
              static_cast<long long>(e0), static_cast<long long>(e1),
              ovvsStatusString(es1 == OVVS_STATUS_SUCCESS ? es1 : es0));
  ovvsResourcesDestroy(res);
  return 0;
}
