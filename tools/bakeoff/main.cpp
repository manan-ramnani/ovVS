#include "iovs/iovs.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  std::string op = "gemm";
  if (argc > 1) op = argv[1];
  iovsResources_t res = nullptr;
  if (iovsResourcesCreate(&res) != IOVS_STATUS_SUCCESS) return 1;

  char sku[64] = {0};
  iovsResourcesSku(res, sku, 64);
  int32_t npu = 0, gpu = 0;
  iovsResourcesNpuAvailable(res, &npu);
  iovsResourcesGpuAvailable(res, &gpu);

  std::printf("{\n  \"sku\": \"%s\",\n  \"op\": \"%s\",\n  \"npu\": %s,\n  \"gpu\": %s,\n  \"runs\": [\n",
              sku, op.c_str(), npu ? "true" : "false", gpu ? "true" : "false");

  const iovsPolicy policies[] = {IOVS_POLICY_FORCE_CPU, IOVS_POLICY_FORCE_NPU, IOVS_POLICY_FORCE_GPU};
  const char* names[] = {"cpu", "npu", "gpu"};
  bool first = true;
  for (int p = 0; p < 3; ++p) {
    iovsResourcesSetPolicy(res, policies[p]);
    const int64_t m = 64, n = 128, k = 32;
    std::vector<float> A(m * k, 0.1f), B(n * k, 0.2f), C(m * n, 0.f);
    for (int64_t i = 0; i < m * k; ++i) A[static_cast<size_t>(i)] = 0.01f * static_cast<float>(i % 17);
    for (int64_t i = 0; i < n * k; ++i) B[static_cast<size_t>(i)] = 0.02f * static_cast<float>(i % 13);

    const double t0 = now_ms();
    iovsStatus st = IOVS_STATUS_SUCCESS;
    if (op == "gemm") {
      st = iovsGemm(res, A.data(), B.data(), C.data(), m, n, k, 1);
    } else if (op == "topk") {
      std::vector<int64_t> idx(m * 10);
      std::vector<float> val(m * 10);
      st = iovsTopk(res, C.data(), m, n, 10, idx.data(), val.data(), 0);
    } else if (op == "gather") {
      std::vector<int64_t> idx(32);
      for (int i = 0; i < 32; ++i) idx[static_cast<size_t>(i)] = i % n;
      std::vector<float> out(32 * k);
      st = iovsGatherRows(res, B.data(), n, k, idx.data(), 32, out.data());
    }
    const double t1 = now_ms();
    if (!first) std::printf(",\n");
    first = false;
    std::printf("    {\"device\": \"%s\", \"ms\": %.4f, \"status\": \"%s\"}", names[p], t1 - t0,
                iovsStatusString(st));
  }
  std::printf("\n  ]\n}\n");
  iovsResourcesDestroy(res);
  return 0;
}
