// Geometry gate for CAGRA kernel step 2.
// Compares the CURRENT distance-engine shape against the proposed one, in isolation.
//   OLD: one 128-item work-group per query, candidates scored strictly one at a time,
//        each lane owns ONE dimension, full work-group reduce_over_group per candidate.
//   NEW: same 128-item work-group = 8 sub-groups of 16; each sub-group owns a whole
//        candidate, 8 dims per lane via float4 loads, sub-group butterfly reduce.
// If NEW is not >=4x OLD, the premise behind the rewrite is wrong and it should not be built.

#include <sycl/sycl.hpp>

#include <cstdio>
#include <chrono>
#include <vector>
#include <algorithm>

namespace {

constexpr int kDim = 128;        // SIFT
constexpr int kRows = 100000;    // dataset rows resident
constexpr int kCands = 2048;     // candidates scored per query (≈ measured evals/query)
constexpr int kGroups = 32;      // queries in flight = batch 32
constexpr int kWg = 128;         // work-group size, unchanged
constexpr int kSg = 16;          // required sub-group size for NEW

double bench(sycl::queue& q, const char* name, int reps,
             const std::function<sycl::event()>& launch) {
  launch().wait_and_throw();  // warm: absorbs JIT
  std::vector<double> ms;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    launch().wait_and_throw();
    const auto t1 = std::chrono::steady_clock::now();
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(ms.begin(), ms.end());
  const double med = ms[ms.size() / 2];
  std::printf("  %-10s median %8.3f ms   (min %8.3f, max %8.3f)\n", name, med, ms.front(), ms.back());
  return med;
}

}  // namespace

int main() {
  sycl::queue q{sycl::gpu_selector_v};
  std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

  const size_t ds_count = size_t(kRows) * kDim;
  float* DS = sycl::malloc_device<float>(ds_count, q);
  float* QY = sycl::malloc_device<float>(size_t(kGroups) * kDim, q);
  float* OUT = sycl::malloc_device<float>(size_t(kGroups) * kCands, q);
  // Per-group candidate ids: 32 x 2048 distinct rows = ~33 MB touched, well past the 4 MB L2,
  // so the gather is DRAM-bound like the real walk instead of cache-resident.
  int32_t* IDS = sycl::malloc_device<int32_t>(size_t(kGroups) * kCands, q);
  // INT8 mirror of the same data. SIFT values are integers in [0,255], so the int32 L2 of
  // uint8 rows is EXACT, not an approximation. 4x fewer bytes per candidate.
  uint8_t* DS8 = sycl::malloc_device<uint8_t>(ds_count, q);
  uint8_t* QY8 = sycl::malloc_device<uint8_t>(size_t(kGroups) * kDim, q);
  if (!DS || !QY || !OUT || !IDS || !DS8 || !QY8) { std::printf("alloc failed\n"); return 1; }

  // Deterministic fill; candidate ids are scattered so the gather pattern is realistic.
  q.parallel_for(ds_count, [=](sycl::id<1> i) {
     DS[i] = float((i * 1103515245u + 12345u) % 256u);
   }).wait();
  q.parallel_for(size_t(kGroups) * kDim, [=](sycl::id<1> i) {
     QY[i] = float((i * 22695477u + 1u) % 256u);
   }).wait();
  q.parallel_for(size_t(kGroups) * kCands, [=](sycl::id<1> i) {
     IDS[i] = int32_t((i * 9973u + 7u) % kRows);
   }).wait();

  q.parallel_for(ds_count, [=](sycl::id<1> i) {
     DS8[i] = uint8_t((i * 1103515245u + 12345u) % 256u);
   }).wait();
  q.parallel_for(size_t(kGroups) * kDim, [=](sycl::id<1> i) {
     QY8[i] = uint8_t((i * 22695477u + 1u) % 256u);
   }).wait();

  const int reps = 7;

  // ---- OLD shape: whole work-group cooperates on ONE candidate at a time ----
  auto launch_old = [&]() {
    return q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::nd_range<1>(sycl::range<1>(size_t(kGroups) * kWg), sycl::range<1>(kWg)),
                     [=](sycl::nd_item<1> it) {
        const size_t lid = it.get_local_linear_id();
        const size_t g = it.get_group_linear_id();
        auto grp = it.get_group();
        const float* qv = QY + g * kDim;
        for (int c = 0; c < kCands; ++c) {
          const int32_t id = IDS[g * kCands + c];
          float acc = 0.f;
          for (int d = int(lid); d < kDim; d += kWg) {
            const float diff = qv[d] - DS[size_t(id) * kDim + d];
            acc += diff * diff;
          }
          acc = sycl::reduce_over_group(grp, acc, sycl::plus<float>());
          if (lid == 0) OUT[g * kCands + c] = acc;
        }
      });
    });
  };

  // ---- NEW shape: one sub-group per candidate, 8 candidates in flight per work-group ----
  auto launch_new = [&]() {
    return q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::nd_range<1>(sycl::range<1>(size_t(kGroups) * kWg), sycl::range<1>(kWg)),
                     [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
        auto sg = it.get_sub_group();
        const size_t g = it.get_group_linear_id();
        const int sg_id = int(sg.get_group_linear_id());
        const int lane = int(sg.get_local_linear_id());
        const int sg_count = kWg / kSg;
        const float* qv = QY + g * kDim;
        for (int c = sg_id; c < kCands; c += sg_count) {
          const int32_t id = IDS[g * kCands + c];
          const float* row = DS + size_t(id) * kDim;
          float acc = 0.f;
          // 16 lanes x float4 = 64 dims per pass, coalesced; two passes for D=128.
          for (int base = 0; base < kDim; base += kSg * 4) {
            const int off = base + lane * 4;
            const sycl::vec<float, 4> a{qv[off], qv[off + 1], qv[off + 2], qv[off + 3]};
            const sycl::vec<float, 4> b{row[off], row[off + 1], row[off + 2], row[off + 3]};
            const sycl::vec<float, 4> dv = a - b;
            acc += dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2] + dv[3] * dv[3];
          }
          acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
          if (lane == 0) OUT[g * kCands + c] = acc;
        }
      });
    });
  };

  // ---- NEW8: same sub-group geometry, int8 rows. 128 B/candidate instead of 512 B. ----
  auto launch_new8 = [&]() {
    return q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::nd_range<1>(sycl::range<1>(size_t(kGroups) * kWg), sycl::range<1>(kWg)),
                     [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
        auto sg = it.get_sub_group();
        const size_t g = it.get_group_linear_id();
        const int sg_id = int(sg.get_group_linear_id());
        const int lane = int(sg.get_local_linear_id());
        const int sg_count = kWg / kSg;
        const uint8_t* qv = QY8 + g * kDim;
        for (int c = sg_id; c < kCands; c += sg_count) {
          const int32_t id = IDS[g * kCands + c];
          const uint8_t* row = DS8 + size_t(id) * kDim;
          int32_t acc = 0;
          // 16 lanes x 8 contiguous bytes = 128 dims in one pass: 2 cache lines per candidate.
          const int off = lane * 8;
          for (int j = 0; j < 8; ++j) {
            const int32_t d = int32_t(qv[off + j]) - int32_t(row[off + j]);
            acc += d * d;
          }
          acc = sycl::reduce_over_group(sg, acc, sycl::plus<int32_t>());
          if (lane == 0) OUT[g * kCands + c] = float(acc);
        }
      });
    });
  };

  // ---- OLD8: CURRENT geometry unchanged, only the rows become int8. This is the
  //      bit-exact, low-risk change. If it captures most of NEW8's win, the sub-group
  //      rewrite (which changes output bits) is not needed. ----
  auto launch_old8 = [&]() {
    return q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::nd_range<1>(sycl::range<1>(size_t(kGroups) * kWg), sycl::range<1>(kWg)),
                     [=](sycl::nd_item<1> it) {
        const size_t lid = it.get_local_linear_id();
        const size_t g = it.get_group_linear_id();
        auto grp = it.get_group();
        const uint8_t* qv = QY8 + g * kDim;
        for (int c = 0; c < kCands; ++c) {
          const int32_t id = IDS[g * kCands + c];
          const uint8_t* row = DS8 + size_t(id) * kDim;
          int32_t acc = 0;
          for (int d = int(lid); d < kDim; d += kWg) {
            const int32_t diff = int32_t(qv[d]) - int32_t(row[d]);
            acc += diff * diff;
          }
          acc = sycl::reduce_over_group(grp, acc, sycl::plus<int32_t>());
          if (lid == 0) OUT[g * kCands + c] = float(acc);
        }
      });
    });
  };

  std::printf("shape: %d groups x %d wg, %d candidates/query, D=%d\n", kGroups, kWg, kCands, kDim);
  const double old_ms = bench(q, "OLD", reps, launch_old);
  const double old8_ms = bench(q, "OLD+int8", reps, launch_old8);
  const double new_ms = bench(q, "NEW", reps, launch_new);
  const double new8_ms = bench(q, "NEW+int8", reps, launch_new8);
  std::printf("\n  OLD8/OLD  = %.2fx   (int8 ALONE, geometry unchanged, BIT-EXACT)\n", old_ms / old8_ms);
  std::printf("  NEW/OLD   = %.2fx   (sub-group geometry alone)\n", old_ms / new_ms);
  std::printf("  NEW8/OLD  = %.2fx   (geometry + int8 rows)\n", old_ms / new8_ms);
  std::printf("  NEW8/NEW  = %.2fx   (int8 on top of geometry)\n", new_ms / new8_ms);

  // Correctness: both shapes must agree to fp tolerance on the same inputs.
  std::vector<float> a(kCands), b(kCands);
  launch_old().wait_and_throw();
  q.memcpy(a.data(), OUT, kCands * sizeof(float)).wait();
  launch_new().wait_and_throw();
  q.memcpy(b.data(), OUT, kCands * sizeof(float)).wait();
  double worst = 0.0;
  for (int i = 0; i < kCands; ++i) {
    const double rel = std::abs(double(a[i]) - double(b[i])) / std::max(1e-6, double(std::abs(a[i])));
    worst = std::max(worst, rel);
  }
  std::printf("  max relative distance difference OLD vs NEW = %.3e\n", worst);

  /* The bit-exactness claim: for integer-valued data in [0,255] every partial sum is at most
     128 * 255^2 = 8,323,200 < 2^24, so the fp32 result IS the exact integer and the int8 path
     must reproduce it BITWISE, whatever order the reduction happens in. */
  std::vector<float> c8(kCands);
  launch_new8().wait_and_throw();
  q.memcpy(c8.data(), OUT, kCands * sizeof(float)).wait();
  size_t exact = 0;
  for (int i = 0; i < kCands; ++i) {
    if (a[i] == c8[i]) ++exact;
  }
  std::printf("  OLD(fp32) vs NEW8(int8) bitwise-equal: %zu / %d\n", exact, kCands);

  sycl::free(DS, q); sycl::free(QY, q); sycl::free(OUT, q); sycl::free(IDS, q);
  sycl::free(DS8, q); sycl::free(QY8, q);
  return 0;
}
