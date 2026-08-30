// req_gate: measures the iGPU's sustainable RANDOM ROW-READ RATE in the exact access
// shape of the CAGRA walk's distance evals -- one sub-group of 16 reads one row, each
// lane an 8-byte slice -- as a function of:
//   - resident sub-groups        (--wgs 4..64, x8 sub-groups each)
//   - outstanding rows per chain (--burst 1|2|4; 1 = dependent pointer chase)
//   - footprint                  (--rows-log2: 14=2MiB .. 20=128MiB at 128B rows)
//   - row size                   (--row-bytes 64|128)
// The walk measured ~60M (qpw1) and ~75M (qpw8) rows/s at ~9GB/s, far under the byte
// roofline; this gate says whether that is the fabric's request-rate ceiling or kernel
// overhead, whether MLP (burst>1) raises it, and how much L2 residency (small footprint)
// buys -- i.e. what a graph-locality reorder could recover.
//
// Build: build_env.cmd icx -fsycl -O2 out\req_gate.cpp -o out\req_gate.exe
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

static size_t arg(int argc, char** argv, const char* name, size_t fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return static_cast<size_t>(std::atoll(argv[i + 1]));
  }
  return fallback;
}

int main(int argc, char** argv) {
  const size_t wgs = arg(argc, argv, "--wgs", 64);
  const size_t rows_log2 = arg(argc, argv, "--rows-log2", 17); // 2^17 rows * 128B = 16 MiB
  const size_t burst = arg(argc, argv, "--burst", 1);
  const size_t row_bytes = arg(argc, argv, "--row-bytes", 128);
  const size_t steps = arg(argc, argv, "--steps", 20000);
  const size_t reps = arg(argc, argv, "--reps", 3);
  // Burst-locality probe: each dependent step reads a CLUSTER of `cluster` rows -- the
  // base row jumps uniformly (dependent, chase-style; deliberately NOT a bounded walk,
  // whose 1-D recurrence manufactures L2 revisit hits the dedup'd graph walk never gets),
  // and the cluster's other rows land within +/-window rows of the base (0 = uniform,
  // i.e. no clustering). Models what a graph-locality reorder actually changes: the tile
  // gather's rows arriving page-together, at unchanged footprint and zero revisits.
  const size_t window = arg(argc, argv, "--window", 0);
  const size_t cluster = arg(argc, argv, "--cluster", 1);

  const size_t rows = size_t{1} << rows_log2;
  const size_t dpr = row_bytes / 4;   // dwords per row
  const size_t lpw = dpr / 16;        // dwords per lane (1 at 64B, 2 at 128B)
  if (lpw == 0 || (burst != 1 && burst != 2 && burst != 4)) {
    std::fprintf(stderr, "bad args\n");
    return 1;
  }

  sycl::queue q{sycl::gpu_selector_v};
  const size_t sub_groups = wgs * 8;

  // Single-cycle permutation for the dependent chase, stored in dword 0 of each row.
  std::vector<uint32_t> host(rows * dpr);
  {
    std::vector<uint32_t> order(rows);
    std::iota(order.begin(), order.end(), 0u);
    std::mt19937 rng(7);
    std::shuffle(order.begin(), order.end(), rng);
    uint32_t junk = 0x9e3779b9u;
    for (size_t r = 0; r < rows; ++r) {
      for (size_t d = 0; d < dpr; ++d) {
        junk = junk * 1664525u + 1013904223u;
        host[r * dpr + d] = junk;
      }
    }
    for (size_t i = 0; i < rows; ++i) {
      host[static_cast<size_t>(order[i]) * dpr] = order[(i + 1) % rows];
    }
  }
  uint32_t* buf = sycl::malloc_device<uint32_t>(rows * dpr, q);
  uint32_t* out = sycl::malloc_device<uint32_t>(sub_groups, q);
  q.memcpy(buf, host.data(), host.size() * sizeof(uint32_t)).wait();

  const uint32_t mask = static_cast<uint32_t>(rows - 1);
  const int K = static_cast<int>(steps);
  const int B = static_cast<int>(burst);
  const int G = static_cast<int>(cluster);
  const uint32_t win = static_cast<uint32_t>(window);

  auto launch = [&]() {
    return q.submit([&](sycl::handler& h) {
      h.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(wgs * 128), sycl::range<1>(128)),
          [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            auto sg = item.get_sub_group();
            const uint32_t lane = static_cast<uint32_t>(sg.get_local_linear_id());
            const uint32_t gsg = static_cast<uint32_t>(item.get_group_linear_id() * 8 +
                                                       sg.get_group_linear_id());
            uint32_t acc = 0;
            if (B == 1) {
              // Dependent chase: exactly one row outstanding per sub-group, the walk's
              // score-loop shape. Chain latency = elapsed / K.
              uint32_t row = (gsg * 2654435761u) & mask;
              uint32_t lcg = gsg * 747796405u + 2891336453u;
              for (int k = 0; k < K; ++k) {
                uint32_t v0first = 0;
                for (int g = 0; g < G; ++g) {
                  uint32_t r = row;
                  if (g != 0) {
                    lcg = lcg * 1664525u + 1013904223u;
                    r = (win == 0u) ? (lcg & mask)
                                    : ((row + (lcg % (2u * win + 1u)) - win) & mask);
                  }
                  const size_t base = static_cast<size_t>(r) * dpr + lane * lpw;
                  const uint32_t v0 = buf[base];
                  acc += v0;
                  if (lpw > 1) acc += buf[base + 1];
                  if (g == 0) v0first = v0;
                }
                /* the base always jumps uniformly: dependent, recurrence-free */
                row = sycl::group_broadcast(sg, v0first, 0) & mask;
              }
            } else if (B == 2) {
              uint32_t x0 = gsg * 2654435761u + 1u, x1 = gsg * 2246822519u + 2u;
              uint32_t a0 = 0, a1 = 0;
              for (int k = 0; k < K; ++k) {
                x0 = x0 * 1664525u + 1013904223u;
                x1 = x1 * 1664525u + 1013904223u;
                const size_t b0 = static_cast<size_t>(x0 & mask) * dpr + lane * lpw;
                const size_t b1 = static_cast<size_t>(x1 & mask) * dpr + lane * lpw;
                a0 += buf[b0];
                a1 += buf[b1];
                if (lpw > 1) {
                  a0 += buf[b0 + 1];
                  a1 += buf[b1 + 1];
                }
              }
              acc = a0 + a1;
            } else {
              uint32_t x0 = gsg * 2654435761u + 1u, x1 = gsg * 2246822519u + 2u;
              uint32_t x2 = gsg * 3266489917u + 3u, x3 = gsg * 668265263u + 4u;
              uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
              for (int k = 0; k < K; ++k) {
                x0 = x0 * 1664525u + 1013904223u;
                x1 = x1 * 1664525u + 1013904223u;
                x2 = x2 * 1664525u + 1013904223u;
                x3 = x3 * 1664525u + 1013904223u;
                const size_t b0 = static_cast<size_t>(x0 & mask) * dpr + lane * lpw;
                const size_t b1 = static_cast<size_t>(x1 & mask) * dpr + lane * lpw;
                const size_t b2 = static_cast<size_t>(x2 & mask) * dpr + lane * lpw;
                const size_t b3 = static_cast<size_t>(x3 & mask) * dpr + lane * lpw;
                a0 += buf[b0];
                a1 += buf[b1];
                a2 += buf[b2];
                a3 += buf[b3];
                if (lpw > 1) {
                  a0 += buf[b0 + 1];
                  a1 += buf[b1 + 1];
                  a2 += buf[b2 + 1];
                  a3 += buf[b3 + 1];
                }
              }
              acc = a0 + a1 + a2 + a3;
            }
            const uint32_t total =
                sycl::reduce_over_group(sg, acc, sycl::plus<uint32_t>());
            if (lane == 0) out[gsg] = total;
          });
    });
  };

  launch().wait(); // JIT + warm
  double best_rate = 0.0;
  for (size_t rep = 0; rep < reps; ++rep) {
    const auto t0 = std::chrono::steady_clock::now();
    launch().wait();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double reads =
        static_cast<double>(sub_groups) * K * B * (B == 1 ? cluster : 1u);
    best_rate = std::max(best_rate, reads / s);
  }
  const double gbps = best_rate * static_cast<double>(row_bytes) / 1e9;
  std::printf(
      "wgs=%zu sgs=%zu fp=%zuMiB row=%zuB burst=%zu clu=%zu win=%zu  rows/s=%.1fM  GB/s=%.2f\n",
      wgs, sub_groups, rows * row_bytes >> 20, row_bytes, burst, cluster, window,
      best_rate / 1e6, gbps);
  sycl::free(buf, q);
  sycl::free(out, q);
  return 0;
}
