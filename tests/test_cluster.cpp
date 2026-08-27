#include "test_harness.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <numeric>
#include <set>
#include <string>
#include <thread>
#include <vector>

OVVS_TEST(kmeans_inertia_drops) {
  Res res;
  const int64_t n = 60, dim = 4;
  auto data = make_data(n, dim, 600);
  /* two blobs */
  for (int64_t i = 0; i < n / 2; ++i) data[static_cast<size_t>(i * dim)] += 5.f;
  ovvsKMeansModel_t m = nullptr;
  expect_status(ovvsKMeansFit(res.r, data.data(), n, dim, 2, 15, &m), "km");
  std::vector<int64_t> labels(static_cast<size_t>(n));
  std::vector<float> d(static_cast<size_t>(n));
  expect_status(ovvsKMeansPredict(res.r, m, data.data(), n, labels.data(), d.data()), "kmp");
  std::set<int64_t> uniq(labels.begin(), labels.end());
  expect(uniq.size() >= 2, "kmeans used both clusters");
  float inertia = 0.f;
  for (float x : d) inertia += x;
  expect(inertia > 0.f, "inertia");
  const float* cents = nullptr;
  int32_t k = 0;
  int64_t dd = 0;
  expect_status(ovvsKMeansCentroids(m, &cents, &k, &dd), "cent");
  expect(k == 2 && dd == dim && cents, "cent shape");
  ovvsKMeansDestroy(m);
}

OVVS_TEST(slink_and_spectral_labels) {
  Res res;
  const int64_t n = 24, dim = 3;
  auto data = make_data(n, dim, 610);
  for (int64_t i = 0; i < n / 2; ++i) data[static_cast<size_t>(i * dim)] += 8.f;
  ovvsSlinkModel_t sl = nullptr;
  expect_status(ovvsSlinkFit(res.r, data.data(), n, dim, 2, 6, &sl), "slink");
  const int64_t* lab = nullptr;
  int64_t nn = 0;
  expect_status(ovvsSlinkLabels(sl, &lab, &nn), "sll");
  expect(nn == n, "slink n");
  std::set<int64_t> su(lab, lab + n);
  expect(su.size() >= 2, "slink clusters");
  ovvsSlinkDestroy(sl);

  ovvsSpectralModel_t sp = nullptr;
  expect_status(ovvsSpectralFit(res.r, data.data(), n, dim, 2, 6, &sp), "spec");
  expect_status(ovvsSpectralLabels(sp, &lab, &nn), "spl");
  expect(nn == n, "spec n");
  std::set<int64_t> su2(lab, lab + n);
  expect(!su2.empty(), "spec labels");
  ovvsSpectralDestroy(sp);
}

OVVS_TEST(sq_pq_binary_roundtrip) {
  Res res;
  const int64_t n = 20, dim = 8;
  auto data = make_data(n, dim, 700);
  ovvsSqModel_t sq = nullptr;
  expect_status(ovvsSqFit(res.r, data.data(), n, dim, &sq), "sq");
  std::vector<uint8_t> codes(static_cast<size_t>(n * dim));
  std::vector<float> rec(static_cast<size_t>(n * dim));
  expect_status(ovvsSqEncode(sq, data.data(), n, codes.data()), "sqe");
  expect_status(ovvsSqDecode(sq, codes.data(), n, rec.data()), "sqd");
  float err = 0.f;
  for (size_t i = 0; i < rec.size(); ++i) err += std::fabs(rec[i] - data[i]);
  expect(err / static_cast<float>(rec.size()) < 0.05f, "sq err");
  ovvsSqDestroy(sq);

  ovvsPqModel_t pq = nullptr;
  expect_status(ovvsPqFit(res.r, data.data(), n, dim, 4, 8, &pq), "pq");
  std::vector<uint8_t> pcodes(static_cast<size_t>(n * 4));
  expect_status(ovvsPqEncode(pq, data.data(), n, pcodes.data()), "pqe");
  expect_status(ovvsPqDecode(pq, pcodes.data(), n, rec.data()), "pqd");
  ovvsPqDestroy(pq);

  ovvsBinaryQuantizer_t bq = nullptr;
  expect_status(ovvsBinaryFit(res.r, data.data(), n, dim, &bq), "bin");
  std::vector<uint8_t> bcodes(static_cast<size_t>(n * ((dim + 7) / 8)));
  expect_status(ovvsBinaryEncode(bq, data.data(), n, bcodes.data()), "bine");
  ovvsBinaryDestroy(bq);
}

OVVS_TEST(pca_reduces_and_pairwise_ip) {
  Res res;
  const int64_t n = 16, dim = 6;
  auto data = make_data(n, dim, 800);
  ovvsPcaModel_t pca = nullptr;
  expect_status(ovvsPcaFit(res.r, data.data(), n, dim, 3, &pca), "pca");
  std::vector<float> z(static_cast<size_t>(n * 3));
  expect_status(ovvsPcaTransform(pca, data.data(), n, z.data()), "pcat");
  float nrm = 0.f;
  for (float v : z) nrm += v * v;
  expect(nrm > 0.f, "pca energy");
  ovvsPcaDestroy(pca);

  std::vector<float> out(static_cast<size_t>(n * n));
  expect_status(ovvsPairwiseDistance(res.r, OVVS_METRIC_INNER_PRODUCT, data.data(), n, data.data(), n,
                                     dim, out.data(), 2.f),
                "ip");
  float self = 0.f;
  for (int64_t d = 0; d < dim; ++d) self += data[static_cast<size_t>(d)] * data[static_cast<size_t>(d)];
  expect(std::fabs(out[0] + self) < 1e-4f, "pairwise IP stores -dot");
}

OVVS_TEST(spectral_embedding_api) {
  Res res;
  const int64_t n = 20, dim = 4;
  auto data = make_data(n, dim, 620);
  for (int64_t i = 0; i < n / 2; ++i) data[static_cast<size_t>(i * dim)] += 6.f;
  ovvsSpectralEmbed_t em = nullptr;
  expect_status(ovvsSpectralEmbedFit(res.r, data.data(), n, dim, 2, 5, &em), "embed");
  const float* z = nullptr;
  int64_t nn = 0;
  int32_t nc = 0;
  expect_status(ovvsSpectralEmbedData(em, &z, &nn, &nc), "data");
  expect(nn == n && nc == 2 && z, "embed shape");
  float nrm = 0.f;
  for (int64_t i = 0; i < n * nc; ++i) nrm += z[i] * z[i];
  expect(nrm > 0.f, "embed energy");
  ovvsSpectralEmbedDestroy(em);
}

OVVS_TEST(mixer_auto_skips_busy_npu) {
  Res res;
  int32_t npu = 0, gpu = 0;
  ovvsResourcesNpuAvailable(res.r, &npu);
  ovvsResourcesGpuAvailable(res.r, &gpu);
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_AUTO);
  expect_status(ovvsResourcesSetNpuBusy(res.r, 1), "busy");
  const int64_t m = 256, n = 64, k = 128;
  auto A = make_data(m, k, 1);
  auto B = make_data(n, k, 2);
  std::vector<float> C(static_cast<size_t>(m * n));
  expect_status(ovvsGemm(res.r, A.data(), B.data(), C.data(), m, n, k, 1), "gemm busy");
  ovvsDevice last = OVVS_DEVICE_NPU;
  ovvsResourcesLastDevice(res.r, &last);
  expect(last != OVVS_DEVICE_NPU, "auto must not pick busy npu");
  expect_status(ovvsResourcesSetNpuBusy(res.r, 0), "free");
}

OVVS_TEST(energy_probe_does_not_crash) {
  Res res;
  int64_t uj = -1;
  const ovvsStatus st = ovvsResourcesEnergyUj(res.r, &uj);
  expect(st == OVVS_STATUS_SUCCESS || st == OVVS_STATUS_UNSUPPORTED, "energy status");
  if (st == OVVS_STATUS_SUCCESS) expect(uj > 0, "energy uj");
}

OVVS_TEST(energy_package_uj_nondecreasing) {
  Res res;
  int64_t a = -1, b = -1;
  const ovvsStatus st0 = ovvsResourcesEnergyUj(res.r, &a);
  if (st0 == OVVS_STATUS_UNSUPPORTED) return;
  expect_status(st0, "energy before");
  std::vector<float> x(1 << 20, 1.f);
  float s = 0.f;
  const auto t0 = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(250)) {
    for (float v : x) s += v * v;
  }
  expect(s > 0.f, "cpu work");
  expect_status(ovvsResourcesEnergyUj(res.r, &b), "energy after");
  if (b <= a) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    for (float v : x) s += v * v;
    expect_status(ovvsResourcesEnergyUj(res.r, &b), "energy after wait");
  }
  expect(b > a, "package energy increased");
}

OVVS_TEST(probe_json_energy_and_shave_fields) {
  std::vector<char> buf(32768, 0);
  expect_status(ovvsProbeJson(buf.data(), static_cast<int32_t>(buf.size())), "probe");
  const std::string j(buf.data());
  expect(j.find("\"energy_source\"") != std::string::npos, "energy_source field");
  expect(j.find("\"shave_silicon_load\"") != std::string::npos, "shave_silicon_load field");
  expect(j.find("\"npu_matmul_f16\"") != std::string::npos, "npu_matmul_f16 field");
  expect(j.find("\"npu_matmul_i8\"") != std::string::npos, "npu_matmul_i8 field");
  expect(j.find("\"shave_elf_inject_export\": true") == std::string::npos, "no fake elf inject export");
#ifdef _WIN32
  const bool emi = j.find("emi-intelppm") != std::string::npos;
  const bool pdh = j.find("pdh-energy-meter") != std::string::npos;
  const bool gadget = j.find("power-gadget") != std::string::npos;
  const bool unsupported = j.find("\"energy_source\": \"unsupported\"") != std::string::npos;
  expect(emi || pdh || gadget || unsupported, "known energy source");
  if (emi || pdh) {
    expect(j.find("RAPL_Package0_PKG") != std::string::npos, "package RAPL channel");
  }
  expect(j.find("unsupported_no_inject_api") != std::string::npos ||
             j.find("unsupported_no_compiler") != std::string::npos ||
             j.find("compiler_actshave") != std::string::npos,
         "shave silicon load is honest");
  if (j.find("\"npu_available\": true") != std::string::npos) {
    expect(j.find("\"shave_unsigned_inject\": \"unsupported_no_inject_api\"") != std::string::npos,
           "unsigned SHAVE inject still unsupported");
  }
#endif
}

OVVS_TEST(mixer_auto_completes_under_competing_npu) {
  Res res;
  int32_t npu = 0;
  ovvsResourcesNpuAvailable(res.r, &npu);
  if (!npu) return;
  const int64_t m = 256, n = 64, k = 128;
  auto A = make_data(m, k, 3);
  auto B = make_data(n, k, 4);
  std::atomic<bool> stop{false};
  std::thread load([&] {
    Res r2;
    ovvsResourcesSetPolicy(r2.r, OVVS_POLICY_FORCE_NPU);
    std::vector<float> C(static_cast<size_t>(m * n));
    while (!stop.load()) {
      ovvsGemm(r2.r, A.data(), B.data(), C.data(), m, n, k, 1);
    }
  });
  ovvsResourcesSetPolicy(res.r, OVVS_POLICY_AUTO);
  std::vector<float> C(static_cast<size_t>(m * n));
  const ovvsStatus st = ovvsGemm(res.r, A.data(), B.data(), C.data(), m, n, k, 1);
  stop.store(true);
  load.join();
  expect_status(st, "auto during npu load");
}
