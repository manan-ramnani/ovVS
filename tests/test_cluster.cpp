#include "test_harness.hpp"

#include <atomic>
#include <cmath>
#include <numeric>
#include <set>
#include <thread>

IOVS_TEST(kmeans_inertia_drops) {
  Res res;
  const int64_t n = 60, dim = 4;
  auto data = make_data(n, dim, 600);
  /* two blobs */
  for (int64_t i = 0; i < n / 2; ++i) data[static_cast<size_t>(i * dim)] += 5.f;
  iovsKMeansModel_t m = nullptr;
  expect_status(iovsKMeansFit(res.r, data.data(), n, dim, 2, 15, &m), "km");
  std::vector<int64_t> labels(static_cast<size_t>(n));
  std::vector<float> d(static_cast<size_t>(n));
  expect_status(iovsKMeansPredict(res.r, m, data.data(), n, labels.data(), d.data()), "kmp");
  std::set<int64_t> uniq(labels.begin(), labels.end());
  expect(uniq.size() >= 2, "kmeans used both clusters");
  float inertia = 0.f;
  for (float x : d) inertia += x;
  expect(inertia > 0.f, "inertia");
  const float* cents = nullptr;
  int32_t k = 0;
  int64_t dd = 0;
  expect_status(iovsKMeansCentroids(m, &cents, &k, &dd), "cent");
  expect(k == 2 && dd == dim && cents, "cent shape");
  iovsKMeansDestroy(m);
}

IOVS_TEST(slink_and_spectral_labels) {
  Res res;
  const int64_t n = 24, dim = 3;
  auto data = make_data(n, dim, 610);
  for (int64_t i = 0; i < n / 2; ++i) data[static_cast<size_t>(i * dim)] += 8.f;
  iovsSlinkModel_t sl = nullptr;
  expect_status(iovsSlinkFit(res.r, data.data(), n, dim, 2, 6, &sl), "slink");
  const int64_t* lab = nullptr;
  int64_t nn = 0;
  expect_status(iovsSlinkLabels(sl, &lab, &nn), "sll");
  expect(nn == n, "slink n");
  std::set<int64_t> su(lab, lab + n);
  expect(su.size() >= 2, "slink clusters");
  iovsSlinkDestroy(sl);

  iovsSpectralModel_t sp = nullptr;
  expect_status(iovsSpectralFit(res.r, data.data(), n, dim, 2, 6, &sp), "spec");
  expect_status(iovsSpectralLabels(sp, &lab, &nn), "spl");
  expect(nn == n, "spec n");
  std::set<int64_t> su2(lab, lab + n);
  expect(!su2.empty(), "spec labels");
  iovsSpectralDestroy(sp);
}

IOVS_TEST(sq_pq_binary_roundtrip) {
  Res res;
  const int64_t n = 20, dim = 8;
  auto data = make_data(n, dim, 700);
  iovsSqModel_t sq = nullptr;
  expect_status(iovsSqFit(res.r, data.data(), n, dim, &sq), "sq");
  std::vector<uint8_t> codes(static_cast<size_t>(n * dim));
  std::vector<float> rec(static_cast<size_t>(n * dim));
  expect_status(iovsSqEncode(sq, data.data(), n, codes.data()), "sqe");
  expect_status(iovsSqDecode(sq, codes.data(), n, rec.data()), "sqd");
  float err = 0.f;
  for (size_t i = 0; i < rec.size(); ++i) err += std::fabs(rec[i] - data[i]);
  expect(err / static_cast<float>(rec.size()) < 0.05f, "sq err");
  iovsSqDestroy(sq);

  iovsPqModel_t pq = nullptr;
  expect_status(iovsPqFit(res.r, data.data(), n, dim, 4, 8, &pq), "pq");
  std::vector<uint8_t> pcodes(static_cast<size_t>(n * 4));
  expect_status(iovsPqEncode(pq, data.data(), n, pcodes.data()), "pqe");
  expect_status(iovsPqDecode(pq, pcodes.data(), n, rec.data()), "pqd");
  iovsPqDestroy(pq);

  iovsBinaryQuantizer_t bq = nullptr;
  expect_status(iovsBinaryFit(res.r, data.data(), n, dim, &bq), "bin");
  std::vector<uint8_t> bcodes(static_cast<size_t>(n * ((dim + 7) / 8)));
  expect_status(iovsBinaryEncode(bq, data.data(), n, bcodes.data()), "bine");
  iovsBinaryDestroy(bq);
}

IOVS_TEST(pca_reduces_and_pairwise_ip) {
  Res res;
  const int64_t n = 16, dim = 6;
  auto data = make_data(n, dim, 800);
  iovsPcaModel_t pca = nullptr;
  expect_status(iovsPcaFit(res.r, data.data(), n, dim, 3, &pca), "pca");
  std::vector<float> z(static_cast<size_t>(n * 3));
  expect_status(iovsPcaTransform(pca, data.data(), n, z.data()), "pcat");
  float nrm = 0.f;
  for (float v : z) nrm += v * v;
  expect(nrm > 0.f, "pca energy");
  iovsPcaDestroy(pca);

  std::vector<float> out(static_cast<size_t>(n * n));
  expect_status(iovsPairwiseDistance(res.r, IOVS_METRIC_INNER_PRODUCT, data.data(), n, data.data(), n,
                                     dim, out.data(), 2.f),
                "ip");
  float self = 0.f;
  for (int64_t d = 0; d < dim; ++d) self += data[static_cast<size_t>(d)] * data[static_cast<size_t>(d)];
  expect(std::fabs(out[0] + self) < 1e-4f, "pairwise IP stores -dot");
}

IOVS_TEST(spectral_embedding_api) {
  Res res;
  const int64_t n = 20, dim = 4;
  auto data = make_data(n, dim, 620);
  for (int64_t i = 0; i < n / 2; ++i) data[static_cast<size_t>(i * dim)] += 6.f;
  iovsSpectralEmbed_t em = nullptr;
  expect_status(iovsSpectralEmbedFit(res.r, data.data(), n, dim, 2, 5, &em), "embed");
  const float* z = nullptr;
  int64_t nn = 0;
  int32_t nc = 0;
  expect_status(iovsSpectralEmbedData(em, &z, &nn, &nc), "data");
  expect(nn == n && nc == 2 && z, "embed shape");
  float nrm = 0.f;
  for (int64_t i = 0; i < n * nc; ++i) nrm += z[i] * z[i];
  expect(nrm > 0.f, "embed energy");
  iovsSpectralEmbedDestroy(em);
}

IOVS_TEST(mixer_auto_skips_busy_npu) {
  Res res;
  int32_t npu = 0, gpu = 0;
  iovsResourcesNpuAvailable(res.r, &npu);
  iovsResourcesGpuAvailable(res.r, &gpu);
  iovsResourcesSetPolicy(res.r, IOVS_POLICY_AUTO);
  expect_status(iovsResourcesSetNpuBusy(res.r, 1), "busy");
  const int64_t m = 256, n = 64, k = 128;
  auto A = make_data(m, k, 1);
  auto B = make_data(n, k, 2);
  std::vector<float> C(static_cast<size_t>(m * n));
  expect_status(iovsGemm(res.r, A.data(), B.data(), C.data(), m, n, k, 1), "gemm busy");
  iovsDevice last = IOVS_DEVICE_NPU;
  iovsResourcesLastDevice(res.r, &last);
  expect(last != IOVS_DEVICE_NPU, "auto must not pick busy npu");
  expect_status(iovsResourcesSetNpuBusy(res.r, 0), "free");
}

IOVS_TEST(energy_probe_does_not_crash) {
  Res res;
  int64_t uj = -1;
  const iovsStatus st = iovsResourcesEnergyUj(res.r, &uj);
  expect(st == IOVS_STATUS_SUCCESS || st == IOVS_STATUS_UNSUPPORTED, "energy status");
  if (st == IOVS_STATUS_SUCCESS) expect(uj >= 0, "energy uj");
}

IOVS_TEST(mixer_auto_completes_under_competing_npu) {
  Res res;
  int32_t npu = 0;
  iovsResourcesNpuAvailable(res.r, &npu);
  if (!npu) return;
  const int64_t m = 256, n = 64, k = 128;
  auto A = make_data(m, k, 3);
  auto B = make_data(n, k, 4);
  std::atomic<bool> stop{false};
  std::thread load([&] {
    Res r2;
    iovsResourcesSetPolicy(r2.r, IOVS_POLICY_FORCE_NPU);
    std::vector<float> C(static_cast<size_t>(m * n));
    while (!stop.load()) {
      iovsGemm(r2.r, A.data(), B.data(), C.data(), m, n, k, 1);
    }
  });
  iovsResourcesSetPolicy(res.r, IOVS_POLICY_AUTO);
  std::vector<float> C(static_cast<size_t>(m * n));
  const iovsStatus st = iovsGemm(res.r, A.data(), B.data(), C.data(), m, n, k, 1);
  stop.store(true);
  load.join();
  expect_status(st, "auto during npu load");
}
