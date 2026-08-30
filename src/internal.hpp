#pragma once

#include "ovvs/ovvs.h"

#include <algorithm>
#include <atomic>
#include <shared_mutex>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ovvs {
namespace impl {

constexpr float kInf = std::numeric_limits<float>::infinity();

/* Pairwise scores are always distances (smaller is better): L2, -IP, 1-cos, Hamming.
   Search must take the minimum. INNER_PRODUCT is converted back to +IP after topk. */
inline bool metric_largest(ovvsMetric) { return false; }

inline float l2sq(const float* a, const float* b, int64_t d) {
  float s = 0.f;
  for (int64_t i = 0; i < d; ++i) {
    const float t = a[i] - b[i];
    s += t * t;
  }
  return s;
}

/* Exact integer sum of squared byte differences. Every partial sum is bounded by
   dim*255^2 < 2^24 for dim <= 256, and the fp32 l2sq over the same integer-valued data
   has exactly representable partials for the same reason -- so this is BITWISE equal to
   l2sq() on any dataset/query pair that qualified for the int8 mirror, while touching a
   quarter of the bytes. icx vectorizes the u8->i32 square-difference reduction well. */
inline float l2sq_u8(const uint8_t* a, const uint8_t* b, int64_t d) {
  int32_t s = 0;
  for (int64_t i = 0; i < d; ++i) {
    const int32_t t = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
    s += t * t;
  }
  return static_cast<float>(s);
}

/* IEEE binary16 <-> binary32, round-to-nearest-even, no FPU environment involvement.
   Integers up to 2048 convert exactly in both directions, so SIFT-class data stored as
   fp16 loses nothing and distance results stay bitwise-equal to the fp32 path. */
inline uint16_t f32_to_f16_bits(float value) {
  uint32_t f;
  std::memcpy(&f, &value, sizeof(f));
  const uint16_t sign = static_cast<uint16_t>((f >> 16) & 0x8000u);
  const uint32_t abs = f & 0x7FFFFFFFu;
  if (abs > 0x7F800000u) return static_cast<uint16_t>(sign | 0x7E00u); /* NaN */
  if (abs >= 0x47800000u) return static_cast<uint16_t>(sign | 0x7C00u); /* inf / overflow */
  if (abs >= 0x38800000u) { /* normal half */
    const uint32_t mant = abs & 0x7FFFFFu;
    const uint32_t exp = (abs >> 23) - 112u;
    uint32_t half = (exp << 10) | (mant >> 13);
    const uint32_t rest = mant & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && (half & 1u))) ++half; /* RNE; carry is fine */
    return static_cast<uint16_t>(sign | half);
  }
  if (abs <= 0x33000000u) return sign; /* underflows to +-0 (2^-25 rounds to even 0) */
  const uint32_t mant = (abs & 0x7FFFFFu) | 0x800000u; /* subnormal half */
  const uint32_t shift = 126u - (abs >> 23);
  uint32_t half = mant >> shift;
  const uint32_t rest = mant & ((1u << shift) - 1u);
  const uint32_t halfway = 1u << (shift - 1u);
  if (rest > halfway || (rest == halfway && (half & 1u))) ++half;
  return static_cast<uint16_t>(sign | half);
}

inline float f16_bits_to_f32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x3FFu;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      uint32_t e = 113u;
      uint32_t m = mant;
      while (!(m & 0x400u)) {
        m <<= 1;
        --e;
      }
      f = sign | (e << 23) | ((m & 0x3FFu) << 13);
    }
  } else if (exp == 31u) {
    f = sign | 0x7F800000u | (mant << 13);
  } else {
    f = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

inline void f16_row_to_f32(const uint16_t* src, float* dst, int64_t d) {
  for (int64_t i = 0; i < d; ++i) dst[i] = f16_bits_to_f32(src[i]);
}

inline void f32_row_to_f16(const float* src, uint16_t* dst, int64_t d) {
  for (int64_t i = 0; i < d; ++i) dst[i] = f32_to_f16_bits(src[i]);
}

/* Same accumulation order as l2sq, so on integer-valued data (exact halves, partial
   sums under 2^24) the result is bitwise-equal to the fp32 walk. Half the bytes. */
inline float l2sq_f16(const float* q, const uint16_t* row, int64_t d) {
  float s = 0.f;
  for (int64_t i = 0; i < d; ++i) {
    const float t = q[i] - f16_bits_to_f32(row[i]);
    s += t * t;
  }
  return s;
}

#if (defined(_M_X64) || defined(__x86_64__)) && defined(__clang__)
/* Hardware half->float conversion: without it the scalar bit-twiddle turns the walk
   compute-bound and eats the bandwidth win (measured 22K vs 38K fp32 at 100K). The
   8-lane accumulation order differs from the scalar loop, which only matters on data
   where fp16 already rounds -- on integer-valued corpora every partial sum is exact and
   the result is still bitwise-equal to fp32. One path is chosen once per process. */
__attribute__((target("avx2,fma,f16c"))) inline float l2sq_f16_avx2(const float* q,
                                                                    const uint16_t* row,
                                                                    int64_t d) {
  __m256 acc = _mm256_setzero_ps();
  int64_t i = 0;
  for (; i + 8 <= d; i += 8) {
    const __m256 b =
        _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(row + i)));
    const __m256 a = _mm256_loadu_ps(q + i);
    const __m256 t = _mm256_sub_ps(a, b);
    acc = _mm256_fmadd_ps(t, t, acc);
  }
  __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
  lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
  lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
  float s = _mm_cvtss_f32(lo);
  for (; i < d; ++i) {
    const float t = q[i] - f16_bits_to_f32(row[i]);
    s += t * t;
  }
  return s;
}

inline bool cpu_has_avx2_f16c() {
  int regs[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
  __cpuid(regs, 1);
#else
  __asm__ volatile("cpuid" : "=a"(regs[0]), "=b"(regs[1]), "=c"(regs[2]), "=d"(regs[3])
                   : "a"(1), "c"(0));
#endif
  const bool f16c = (regs[2] & (1 << 29)) != 0;
  const bool fma = (regs[2] & (1 << 12)) != 0;
  const bool osxsave = (regs[2] & (1 << 27)) != 0;
  if (!f16c || !fma || !osxsave) return false;
  if ((_xgetbv(0) & 0x6) != 0x6) return false; /* OS saves YMM state */
#if defined(_MSC_VER)
  __cpuidex(regs, 7, 0);
#else
  __asm__ volatile("cpuid" : "=a"(regs[0]), "=b"(regs[1]), "=c"(regs[2]), "=d"(regs[3])
                   : "a"(7), "c"(0));
#endif
  return (regs[1] & (1 << 5)) != 0; /* AVX2 */
}

inline float l2sq_f16_dispatch(const float* q, const uint16_t* row, int64_t d) {
  static const bool fast = cpu_has_avx2_f16c();
  return fast ? l2sq_f16_avx2(q, row, d) : l2sq_f16(q, row, d);
}
#else
inline float l2sq_f16_dispatch(const float* q, const uint16_t* row, int64_t d) {
  return l2sq_f16(q, row, d);
}
#endif

inline float dot(const float* a, const float* b, int64_t d) {
  float s = 0.f;
  for (int64_t i = 0; i < d; ++i) s += a[i] * b[i];
  return s;
}

inline float nrm2sq(const float* a, int64_t d) {
  float s = 0.f;
  for (int64_t i = 0; i < d; ++i) s += a[i] * a[i];
  return s;
}

inline int hamming_u8(const uint8_t* a, const uint8_t* b, int64_t nbytes) {
  int s = 0;
  for (int64_t i = 0; i < nbytes; ++i) {
    unsigned v = static_cast<unsigned>(a[i] ^ b[i]);
#if defined(_MSC_VER)
    s += static_cast<int>(__popcnt(v));
#else
    s += __builtin_popcount(v);
#endif
  }
  return s;
}

inline float distance_one(ovvsMetric metric, const float* x, const float* y, int64_t d,
                          float metric_arg) {
  switch (metric) {
    case OVVS_METRIC_L2_EXPANDED:
      return l2sq(x, y, d);
    case OVVS_METRIC_L2_SQRT_EXPANDED:
      return std::sqrt(l2sq(x, y, d));
    case OVVS_METRIC_INNER_PRODUCT:
      return -dot(x, y, d);
    case OVVS_METRIC_COSINE_EXPANDED: {
      const float nx = std::sqrt(std::max(nrm2sq(x, d), 1e-12f));
      const float ny = std::sqrt(std::max(nrm2sq(y, d), 1e-12f));
      return 1.f - dot(x, y, d) / (nx * ny);
    }
    case OVVS_METRIC_BITWISE_HAMMING: {
      int s = 0;
      for (int64_t i = 0; i < d; ++i) {
        const uint32_t ax = static_cast<uint32_t>(x[i] >= 0.f);
        const uint32_t ay = static_cast<uint32_t>(y[i] >= 0.f);
        s += static_cast<int>(ax ^ ay);
      }
      return static_cast<float>(s);
    }
    case OVVS_METRIC_LP_UNEXPANDED: {
      const float p = metric_arg > 0.f ? metric_arg : 2.f;
      float s = 0.f;
      for (int64_t i = 0; i < d; ++i) s += std::pow(std::fabs(x[i] - y[i]), p);
      return std::pow(s, 1.f / p);
    }
    default:
      return l2sq(x, y, d);
  }
}

inline bool allowed(const uint8_t* bitset, int64_t i) {
  if (!bitset) return true;
  return (bitset[i >> 3] >> (i & 7)) & 1u;
}

struct MixerDecision {
  ovvsDevice device = OVVS_DEVICE_CPU;
  bool npu_attempted = false;
  bool npu_failed = false;
};

struct PqAdcTask {
  const float* tables = nullptr;
  const uint8_t* codes = nullptr;
  int64_t rows = 0;
  int64_t output_offset = 0;
};

struct PqAdcChunk {
  const float* tables = nullptr;
  const uint8_t* codes = nullptr;
  int32_t valid_rows = 0;
  int32_t bucket_rows = 0;
  int64_t output_offset = 0;
};

/* Offset-only descriptor for the fused IVF-PQ GPU scan/select path. Offsets are
   expressed in elements/rows of the flattened inputs, never as transient host
   pointers. query_dense_offset preserves query -> probe -> list-row tie order. */
struct IvfPqScanTask {
  int32_t query_index = 0;
  int64_t list_begin = 0;
  int64_t list_count = 0;
  int64_t lut_offset = 0;
  int64_t query_dense_offset = 0;
};

/* Call-local accounting for GPU work explicitly issued by ovVS. OpenVINO owns
   its internal transfers and submissions, so those deliberately remain opaque. */
struct GpuWorkStats {
  int64_t allocation_calls = 0;
  int64_t allocation_bytes = 0;
  int64_t h2d_calls = 0;
  int64_t h2d_bytes = 0;
  int64_t d2h_calls = 0;
  int64_t d2h_bytes = 0;
  int64_t kernel_launches = 0;
  int64_t wait_calls = 0;

  static void add(int64_t& value, uint64_t delta) noexcept {
    const uint64_t room = static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - value);
    value = delta > room ? std::numeric_limits<int64_t>::max()
                         : value + static_cast<int64_t>(delta);
  }

  static uint64_t bytes(size_t count, size_t element_size) noexcept {
    if (element_size != 0 && count > std::numeric_limits<size_t>::max() / element_size) {
      return std::numeric_limits<uint64_t>::max();
    }
    return static_cast<uint64_t>(count * element_size);
  }

  void allocation(size_t count, size_t element_size) noexcept {
    add(allocation_calls, 1);
    add(allocation_bytes, bytes(count, element_size));
  }
  void h2d(size_t count, size_t element_size = 1) noexcept {
    add(h2d_calls, 1);
    add(h2d_bytes, bytes(count, element_size));
  }
  void d2h(size_t count, size_t element_size = 1) noexcept {
    add(d2h_calls, 1);
    add(d2h_bytes, bytes(count, element_size));
  }
  void kernel() noexcept { add(kernel_launches, 1); }
  void wait() noexcept { add(wait_calls, 1); }
};

/* Per-call GPU NN-Descent accounting. This object is threaded through the
   CAGRA initializer so build telemetry never snapshots resource-global
   last-call diagnostics. */
struct NnDescentBuildStats {
  GpuWorkStats gpu;
  int64_t iterations = 0;
  int64_t final_changed_edges = 0;
  int64_t final_pending_new_edges = 0;
  int64_t submission_calls = 0;
  int64_t peak_owned_bytes = 0;
  bool converged = false;
  bool gpu_instrumented = false;

  void submission() noexcept { GpuWorkStats::add(submission_calls, 1); }
};

/* Persistent CPU work-stealing pool. Spawning a fresh std::thread set per call cost
   6-10% of a b1024 walk and made small parallel regions net losses; the pool parks its
   workers on a condition variable instead. One pool per Resources, matching the
   documented one-Resources-per-worker concurrency model; `run` is serialized by a
   submitter mutex, so a second concurrent caller waits instead of corrupting job state.
   Lazily created on first use (size fixed then); joined by the destructor, which runs
   from the explicit ovvsResourcesDestroy call, not DllMain -- no loader-lock hazard. */
struct CpuWorkPool {
  explicit CpuWorkPool(int nthreads) {
    workers.reserve(static_cast<size_t>(nthreads > 1 ? nthreads - 1 : 0));
    for (int t = 1; t < nthreads; ++t) {
      workers.emplace_back([this]() { worker_loop(); });
    }
  }
  ~CpuWorkPool() {
    {
      std::lock_guard<std::mutex> lk(m);
      shutdown = true;
    }
    cv.notify_all();
    for (auto& w : workers) w.join();
  }

  int size() const { return static_cast<int>(workers.size()) + 1; }

  /* Runs fn(i) for i in [lo, hi); the calling thread participates. fn must not throw. */
  void run(int64_t lo, int64_t hi, const std::function<void(int64_t)>& fn) {
    if (hi <= lo) return;
    if (workers.empty() || hi - lo == 1) {
      for (int64_t i = lo; i < hi; ++i) fn(i);
      return;
    }
    std::lock_guard<std::mutex> submit(run_mutex);
    {
      std::lock_guard<std::mutex> lk(m);
      body = &fn;
      next.store(lo, std::memory_order_relaxed);
      end = hi;
      active.store(static_cast<int>(workers.size()), std::memory_order_relaxed);
      ++generation;
    }
    cv.notify_all();
    for (int64_t i = next.fetch_add(1, std::memory_order_relaxed); i < hi;
         i = next.fetch_add(1, std::memory_order_relaxed)) {
      fn(i);
    }
    std::unique_lock<std::mutex> lk(m);
    done_cv.wait(lk, [&]() { return active.load(std::memory_order_acquire) == 0; });
    body = nullptr;
  }

 private:
  void worker_loop() {
    uint64_t seen = 0;
    for (;;) {
      const std::function<void(int64_t)>* fn = nullptr;
      int64_t hi = 0;
      {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&]() { return shutdown || generation != seen; });
        if (shutdown) return;
        seen = generation;
        fn = body;
        hi = end;
      }
      for (int64_t i = next.fetch_add(1, std::memory_order_relaxed); i < hi;
           i = next.fetch_add(1, std::memory_order_relaxed)) {
        (*fn)(i);
      }
      bool last = false;
      {
        std::lock_guard<std::mutex> lk(m);
        last = active.fetch_sub(1, std::memory_order_acq_rel) == 1;
      }
      if (last) done_cv.notify_all();
    }
  }

  std::mutex run_mutex;
  std::mutex m;
  std::condition_variable cv;
  std::condition_variable done_cv;
  uint64_t generation = 0;
  bool shutdown = false;
  const std::function<void(int64_t)>* body = nullptr;
  std::atomic<int64_t> next{0};
  int64_t end = 0;
  std::atomic<int> active{0};
  std::vector<std::thread> workers;
};

struct ResourcesData {
  ovvsPolicy policy = OVVS_POLICY_AUTO;
  bool npu_available = false;
  bool gpu_available = false;
  /* Telemetry that concurrent searches may race on; atomics keep the races defined.
     last_device answers "what ran most recently", which is inherently fuzzy under
     concurrency -- callers wanting per-call attribution must serialize themselves. */
  std::atomic<ovvsDevice> last_device{OVVS_DEVICE_CPU};
  ovvsDType last_compute_dtype = OVVS_DTYPE_F32;
  std::string sku = "generic-cpu";
  std::string npu_name;
  std::string gpu_name;
  std::string cache_dir;
  int32_t npu_compile_fails = 0;
  int32_t npu_runtime_fails = 0;
  std::atomic<int32_t> npu_fallbacks{0};
  std::mutex cagra_transfer_mutex;
  int64_t cagra_walk_calls = 0;
  int64_t cagra_direct_index_calls = 0;
  int64_t cagra_index_upload_calls = 0;
  int64_t cagra_index_upload_bytes = 0;
  // Opt-in graph-walk work counters. Off by default so the production kernel is
  // byte-identical and pays nothing; the walk branches on a uniform null pointer.
  // Index order must match OVVS_CAGRA_WALK_COUNTER_* in ovvs.h.
  std::mutex cagra_walk_counter_mutex;
  bool cagra_walk_counters_enabled = false;
  int64_t cagra_walk_counters[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  std::mutex cagra_build_stats_mutex;
  ovvsCagraBuildStatsV1 cagra_build_stats{};
  std::mutex nndescent_stats_mutex;
  int32_t nndescent_iterations_run = 0;
  int64_t nndescent_changed_edges = 0;
  int64_t nndescent_pending_new_edges = 0;
  double nndescent_change_ratio = 0.0;
  int64_t nndescent_peak_device_bytes = 0;
  bool nndescent_converged = false;
  std::mutex ivfpq_stats_mutex;
  int64_t ivfpq_packed_rebuilds = 0;
  int64_t ivfpq_packed_rebuild_rows = 0;
  int64_t ivfpq_unfiltered_direct_rows = 0;
  int64_t ivfpq_unfiltered_id_copy_bytes_avoided = 0;
  int64_t ivfpq_selected_id_resolutions = 0;
  int64_t ivfpq_filtered_code_copy_bytes = 0;
  ovvsIvfPqSearchStatsV1 ivfpq_search_stats{};
  std::mutex pq_adc_stats_mutex;
  int64_t pq_adc_calls = 0;
  int64_t pq_adc_logical_tasks = 0;
  int64_t pq_adc_chunks = 0;
  int64_t pq_adc_valid_rows = 0;
  int64_t pq_adc_padded_rows = 0;
  int64_t pq_adc_npu_requests = 0;
  int64_t pq_adc_npu_rows = 0;
  int64_t pq_adc_npu_capacity_rows = 0;
  int64_t pq_adc_npu_transformed_chunks = 0;
  int64_t pq_adc_npu_transformed_rows = 0;
  int64_t pq_adc_cpu_rows = 0;
  bool npu_busy = false;
  ovvsDevice large_gemm_winner = OVVS_DEVICE_AUTO;
  int64_t large_gemm_flops = 100000LL * 32LL * 768LL;
  std::vector<float> scratch;

  float* scratch_f(size_t n) {
    if (scratch.size() < n) scratch.resize(n);
    return scratch.data();
  }

  /* Lazily created; sized by the first caller (the walk reads OVVS_CPU_WALK_THREADS
     once, at that moment) and fixed for the life of the Resources. */
  std::mutex cpu_pool_mutex;
  std::unique_ptr<CpuWorkPool> cpu_pool;
  CpuWorkPool& pool(int nthreads) {
    std::lock_guard<std::mutex> lk(cpu_pool_mutex);
    if (!cpu_pool) cpu_pool = std::make_unique<CpuWorkPool>(nthreads);
    return *cpu_pool;
  }

  /* Device-local scratch reused across graph-walk calls. Allocating and freeing the visited
     table and the query/output staging on every search cost four device allocations, four
     frees and four queue drains per call -- a fixed tax that dominates batch-one latency.
     Owned per Resources because the documented concurrency model is one Resources per worker. */
  void* gpu_walk_workspace = nullptr;
  size_t gpu_walk_workspace_bytes = 0;

  /* Device-side uint8 mirror of a graph-walk dataset. Only built when every element is an
     integer in [0,255] and the dimension is small enough that the exact integer distance
     still fits a float mantissa, in which case the int8 walk is BITWISE identical to the
     fp32 walk while touching a quarter of the cache lines per candidate. Keyed on the
     source pointer plus a sampled fingerprint so a freed-and-reallocated dataset landing
     on the same address cannot be mistaken for a cache hit. */
  std::mutex gpu_int8_mutex;
  /* Serializes GPU walks that can touch the Resources-cached fp32 int8 mirror: a
     rebuild (first touch, or after a mutation-time invalidate) frees the old device
     buffer, and another in-flight walk may still be reading it. fp16-storage walks
     have no cached GPU state and run without this lock. */
  std::mutex gpu_walk_mutex;
  const void* gpu_int8_source = nullptr;
  size_t gpu_int8_rows = 0;
  size_t gpu_int8_dim = 0;
  uint64_t gpu_int8_fingerprint = 0;
  void* gpu_int8_data = nullptr;
  bool gpu_int8_usable = false;

  ~ResourcesData();
};

inline ResourcesData* rd(ovvsResources_t r) { return reinterpret_cast<ResourcesData*>(r); }

void probe_fill(ResourcesData& r);
std::string probe_json();
void append_energy_probe_json(std::ostringstream& o);
void append_shave_elf_probe_json(std::ostringstream& o);
bool npu_shave_profile_adc(int* shave_tasks, int* dpu_tasks, std::vector<uint8_t>* blob,
                           std::string* exec_types);

/* Shared USM (SYCL) or heap. Dataset/graph vectors use UsmAllocator so iGPU binds these pointers. */
void* ovvs_usm_malloc(size_t bytes);
void ovvs_usm_free(void* p);
bool ovvs_usm_is_shared(const void* p);

/* Device-local (not shared) allocations for reusable GPU scratch. No-ops without SYCL. */
void* ovvs_gpu_workspace_alloc(size_t bytes);
void ovvs_gpu_workspace_free(void* p);

template <typename T>
struct UsmAllocator {
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using propagate_on_container_move_assignment = std::true_type;
  UsmAllocator() noexcept = default;
  template <typename U>
  UsmAllocator(const UsmAllocator<U>&) noexcept {}
  T* allocate(std::size_t n) {
    void* p = ovvs_usm_malloc(n == 0 ? sizeof(T) : n * sizeof(T));
    if (!p) throw std::bad_alloc();
    return static_cast<T*>(p);
  }
  void deallocate(T* p, std::size_t) noexcept { ovvs_usm_free(p); }
  template <typename U>
  bool operator==(const UsmAllocator<U>&) const noexcept {
    return true;
  }
  template <typename U>
  bool operator!=(const UsmAllocator<U>&) const noexcept {
    return false;
  }
};

using UsmFloatVec = std::vector<float, UsmAllocator<float>>;
using UsmI32Vec = std::vector<int32_t, UsmAllocator<int32_t>>;
using UsmU16Vec = std::vector<uint16_t, UsmAllocator<uint16_t>>;
using UsmI64Vec = std::vector<int64_t, UsmAllocator<int64_t>>;
using UsmU8Vec = std::vector<uint8_t, UsmAllocator<uint8_t>>;

/* Device backends. Boolean paths return false when unavailable/failed; status
   paths preserve validation, OOM, and runtime failures for policy routing. */
bool npu_available();
bool gpu_available();
bool ov_device_available(const char* name);
bool ov_matmul(ResourcesData& r, const char* device, const float* a, const float* b, float* c,
               int64_t m, int64_t n, int64_t k, bool trans_b);
bool ov_matmul_compute(ResourcesData& r, const char* device, ovvsDType compute, const float* a,
                       const float* b, float* c, int64_t m, int64_t n, int64_t k, bool trans_b);
void append_lowbit_probe_json(std::ostringstream& o);
bool ov_topk(ResourcesData& r, const char* device, const float* scores, int64_t rows, int64_t cols,
             int64_t k, int64_t* indices, float* values, bool largest);
bool ov_gather_rows(ResourcesData& r, const char* device, const float* src, int64_t src_rows,
                    int64_t dim, const int64_t* idx, int64_t nidx, float* out);
bool npu_gemm(ResourcesData& r, const float* a, const float* b, float* c, int64_t m, int64_t n,
              int64_t k, bool trans_b);
bool npu_gemm_compute(ResourcesData& r, ovvsDType compute, const float* a, const float* b, float* c,
                      int64_t m, int64_t n, int64_t k, bool trans_b);
bool npu_topk(ResourcesData& r, const float* scores, int64_t rows, int64_t cols, int64_t k,
              int64_t* indices, float* values, bool largest);
bool npu_gather_rows(ResourcesData& r, const float* src, int64_t src_rows, int64_t dim,
                     const int64_t* idx, int64_t nidx, float* out);
bool gpu_gemm(ResourcesData& r, const float* a, const float* b, float* c, int64_t m, int64_t n,
              int64_t k, bool trans_b, GpuWorkStats* stats = nullptr);
bool gpu_gemm_compute(ResourcesData& r, ovvsDType compute, const float* a, const float* b, float* c,
                      int64_t m, int64_t n, int64_t k, bool trans_b);
bool gpu_topk(ResourcesData& r, const float* scores, int64_t rows, int64_t cols, int64_t k,
              int64_t* indices, float* values, bool largest, GpuWorkStats* stats = nullptr);
bool gpu_gather_rows(ResourcesData& r, const float* src, int64_t src_rows, int64_t dim,
                     const int64_t* idx, int64_t nidx, float* out,
                     GpuWorkStats* stats = nullptr);
bool gpu_vector_add(const float* a, const float* b, float* c, int64_t n);
ovvsStatus gpu_nndescent_build(ResourcesData& r, const float* dataset, int64_t n,
                               int64_t dim, ovvsMetric metric, int32_t degree,
                               int32_t iters, int32_t* graph,
                               NnDescentBuildStats* stats = nullptr);
OVVS_API ovvsStatus gpu_cagra_optimize_ranked(ResourcesData& r,
                                               const int32_t* initial, int64_t n,
                                               int32_t initial_degree,
                                               int32_t final_degree,
                                               std::vector<int32_t>& output);
const uint8_t* gpu_cagra_int8_mirror_host(ResourcesData& r, const float* dataset, int64_t rows,
                                          int64_t dim);
void ovvs_gpu_mirror_invalidate();
/* dataset may be null when dataset_f16 is provided (fp16 primary storage). In that
   mode the GPU walk runs ONLY via dataset_u8 (the int8 mirror) and the caller must
   guarantee every query qualifies for the int8 path -- the kernel never touches the
   fp32 pointer then. dataset_f16 itself is reserved for the phase-2 fp16 kernel. */
bool gpu_cagra_walk(ResourcesData& r, const float* dataset, int64_t n, int64_t dim, ovvsMetric metric,
                    const int32_t* graph, int32_t degree, const float* queries, int64_t nq, int64_t k,
                    int32_t itopk, int32_t search_width, const uint8_t* bitset, int64_t* neighbors,
                    float* distances, const uint16_t* dataset_f16 = nullptr,
                    const uint8_t* dataset_u8 = nullptr);
bool gpu_pairwise(ResourcesData& r, ovvsMetric metric, const float* x, int64_t nx, const float* y,
                  int64_t ny, int64_t dim, float* out, float metric_arg,
                  GpuWorkStats* stats = nullptr);
bool npu_pairwise(ResourcesData& r, ovvsMetric metric, const float* x, int64_t nx, const float* y,
                  int64_t ny, int64_t dim, float* out, float metric_arg);
bool npu_pq_adc_batch(ResourcesData& r, const PqAdcChunk* chunks, int64_t chunk_count,
                      int32_t pq_m, int32_t ks, float* out);
ovvsStatus gpu_ivfpq_scan_select(ResourcesData& r, const IvfPqScanTask* tasks,
                                 int64_t task_count, const float* luts,
                                 int64_t lut_elements, const int64_t* packed_ids,
                                 const uint8_t* packed_codes, int64_t packed_rows,
                                 const uint8_t* allow_bitset, int64_t allow_bitset_bytes,
                                 int64_t nq, int32_t pq_m, int32_t ks, int32_t krefine,
                                 int32_t* packed_positions, int32_t* counts,
                                 GpuWorkStats* stats = nullptr);
int32_t sycl_enabled();
int32_t gpu_has_xmx();
bool sycl_gpu_available();
bool mkl_gesvd_components(const float* centered, int64_t n, int64_t dim, int32_t ncomp, float* components);
bool mkl_syev_smallest(float* a, int64_t n, int32_t ncomp, float* embed);

void cpu_gemm(const float* a, const float* b, float* c, int64_t m, int64_t n, int64_t k,
              bool trans_b);
void cpu_topk(const float* scores, int64_t rows, int64_t cols, int64_t k, int64_t* indices,
              float* values, bool largest);
void cpu_gather_rows(const float* src, int64_t src_rows, int64_t dim, const int64_t* idx,
                     int64_t nidx, float* out);
void cpu_pairwise(ovvsMetric metric, const float* x, int64_t nx, const float* y, int64_t ny,
                  int64_t dim, float* out, float metric_arg);

ovvsDevice choose_device(ResourcesData& r, const char* op, int64_t flops_or_elems);

ovvsStatus prim_gemm(ResourcesData& r, const float* a, const float* b, float* c,
                     int64_t m, int64_t n, int64_t k, bool trans_b,
                     GpuWorkStats* stats = nullptr);
ovvsStatus prim_gemm_compute(ResourcesData& r, const float* a, const float* b, float* c, int64_t m,
                             int64_t n, int64_t k, bool trans_b, ovvsDType compute);
ovvsStatus prim_topk(ResourcesData& r, const float* scores, int64_t rows, int64_t cols, int64_t k,
                     int64_t* indices, float* values, bool largest,
                     GpuWorkStats* stats = nullptr);
ovvsStatus prim_gather_rows(ResourcesData& r, const float* src, int64_t src_rows, int64_t dim,
                            const int64_t* idx, int64_t nidx, float* out,
                            GpuWorkStats* stats = nullptr);
ovvsStatus prim_pairwise(ResourcesData& r, ovvsMetric metric, const float* x, int64_t nx,
                         const float* y, int64_t ny, int64_t dim, float* out,
                         float metric_arg, GpuWorkStats* stats = nullptr);
ovvsStatus prim_pq_adc(ResourcesData& r, const float* tables, int32_t pq_m, int32_t ks,
                       const uint8_t* codes, int64_t ncodes, float* out);
/* Private C++ test seams; not part of the installed C ABI. */
OVVS_API int32_t pq_adc_bucket_rows(int64_t remaining);
OVVS_API ovvsStatus plan_pq_adc_chunks(const PqAdcTask* tasks, int64_t task_count,
                                       int32_t pq_m, int64_t output_rows,
                                       std::vector<PqAdcChunk>& chunks);
OVVS_API ovvsStatus prim_pq_adc_batch(ResourcesData& r, const PqAdcTask* tasks,
                                      int64_t task_count, int32_t pq_m, int32_t ks,
                                      float* out, int64_t output_rows);
ovvsStatus prim_ivfpq_scan_select(ResourcesData& r, const IvfPqScanTask* tasks,
                                  int64_t task_count, const float* luts,
                                  int64_t lut_elements, const int64_t* packed_ids,
                                  const uint8_t* packed_codes, int64_t packed_rows,
                                  const uint8_t* allow_bitset, int64_t allow_bitset_bytes,
                                  int64_t nq, int32_t pq_m, int32_t ks, int32_t krefine,
                                  int32_t* packed_positions, int32_t* counts,
                                  GpuWorkStats* stats = nullptr);
ovvsStatus prim_nndescent_build(ResourcesData& r, const float* dataset, int64_t n, int64_t dim,
                                 ovvsMetric metric, int32_t degree, int32_t iterations,
                                 int32_t* graph, NnDescentBuildStats* stats = nullptr);
/* Private C++ test seam; not part of the installed C ABI. */
OVVS_API ovvsStatus prim_cagra_optimize_ranked(ResourcesData& r,
                                                const int32_t* initial, int64_t n,
                                                int32_t initial_degree,
                                                int32_t final_degree,
                                                std::vector<int32_t>& output);
OVVS_API ovvsStatus cagra_optimize_ranked(const int32_t* initial, int64_t n,
                                          int32_t initial_degree, int32_t final_degree,
                                          std::vector<int32_t>& output);
/* policy_override (-1 = use r.policy): mutation-time repair searches pass their
   engine choice explicitly so they never mutate r.policy, which concurrent readers
   are consulting. */
ovvsStatus prim_graph_walk(ResourcesData& r, const float* dataset, int64_t n, int64_t dim,
                           ovvsMetric metric, const int32_t* graph, int32_t degree, const float* queries,
                           int64_t nq, int64_t k, int32_t itopk, int32_t search_width,
                           const uint8_t* bitset, int64_t* neighbors, float* distances,
                           const uint16_t* dataset_f16 = nullptr, const uint8_t* dataset_u8 = nullptr,
                           int32_t policy_override = -1);

/* Temporary sweep instrument: OVVS_CAGRA_SEEDS pins the seed count so the recall-vs-QPS
   curve can be measured in one process instead of one rebuild per point. Read once.
   Remove this override once the seed budget is fixed. */
inline int64_t cagra_seed_override() {
  static const int64_t value = []() -> int64_t {
    const char* env = std::getenv("OVVS_CAGRA_SEEDS");
    if (!env || !*env) return -1;
    char* end = nullptr;
    const long long parsed = std::strtoll(env, &end, 10);
    if (end == env || *end != '\0' || parsed < 1) return -1;
    return static_cast<int64_t>(parsed);
  }();
  return value;
}

/* CAGRA is single-layer: random itopk seeds land too far in high-d for greedy routing
   on graph_degree=16. Score a larger sample, then keep itopk.
   Measured on SIFT100K: this phase is 43-50% of ALL candidate distance evaluations, so the
   count is a first-order performance knob and must stay identical on the CPU and GPU paths.
   Both call this function; do not re-derive it at a call site. */
inline int64_t cagra_seed_count(int64_t n, int32_t itopk, int32_t search_width) {
  const int64_t override_seeds = cagra_seed_override();
  int64_t seeds;
  if (override_seeds >= 1) {
    seeds = override_seeds;
  } else {
    /* Measured on SIFT100K, batch 1024, sweeping this count over 512..16 with everything
       else fixed: recall@10 is flat to four decimals (itopk=32/w1: 0.9718 -> 0.9721;
       itopk=64/w2: 0.9947 -> 0.9946) while throughput rises 36-46%. The old
       "score a larger sample" budget of 16*itopk (512-2048 seeds) was 43-50% of ALL
       candidate distance evaluations and bought no measurable recall. Keep enough to give
       the first iteration search_width distinct candidates to expand. */
    seeds = std::max<int64_t>(32, static_cast<int64_t>(std::max(1, search_width)));
  }
  if (n > 0 && seeds > n) seeds = n;
  if (seeds < 1) seeds = n > 0 ? 1 : 0;
  return seeds;
}

inline uint32_t cagra_query_hash(const float* query, int64_t dim) {
  uint32_t hash = 2166136261u;
  for (int64_t i = 0; i < dim; ++i) {
    hash ^= std::bit_cast<uint32_t>(query[i]);
    hash *= 16777619u;
  }
  hash ^= hash >> 16;
  hash *= 0x7feb352du;
  hash ^= hash >> 15;
  hash *= 0x846ca68bu;
  return hash ^ (hash >> 16);
}

inline float f16_to_f32(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u) << 16);
  const uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t man = h & 0x3ffu;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {
      int32_t e = 127 - 15 + 1;
      while ((man & 0x400u) == 0) {
        man <<= 1;
        --e;
      }
      man &= 0x3ffu;
      bits = sign | (static_cast<uint32_t>(e) << 23) | (man << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (man << 13);
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

inline uint16_t f32_to_f16(float f) {
  uint32_t bits = 0;
  std::memcpy(&bits, &f, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  const uint32_t man = bits & 0x7fffffu;
  if (exp <= 0) return static_cast<uint16_t>(sign);
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13));
}

template <typename FloatVec>
inline void convert_to_f32(ovvsDType dtype, const void* src, int64_t n, int64_t dim, FloatVec& out) {
  const int64_t cells = n * dim;
  out.resize(static_cast<size_t>(cells));
  if (dtype == OVVS_DTYPE_F32) {
    std::memcpy(out.data(), src, static_cast<size_t>(cells) * sizeof(float));
    return;
  }
  if (dtype == OVVS_DTYPE_F16) {
    const auto* h = static_cast<const uint16_t*>(src);
    for (int64_t i = 0; i < cells; ++i) out[static_cast<size_t>(i)] = f16_to_f32(h[i]);
    return;
  }
  if (dtype == OVVS_DTYPE_I8) {
    const auto* p = static_cast<const int8_t*>(src);
    for (int64_t i = 0; i < cells; ++i) out[static_cast<size_t>(i)] = static_cast<float>(p[i]);
    return;
  }
  const auto* p = static_cast<const uint8_t*>(src);
  for (int64_t i = 0; i < cells; ++i) out[static_cast<size_t>(i)] = static_cast<float>(p[i]);
}

ovvsStatus brute_search_impl(ResourcesData& r, const float* dataset, int64_t n, int64_t dim,
                             const float* queries, int64_t nq, ovvsMetric metric, int64_t k,
                             const uint8_t* bitset, int64_t* neighbors, float* distances);

ovvsStatus kmeans_fit_impl(ResourcesData& r, const float* x, int64_t n, int64_t dim, int32_t k,
                           int32_t iters, std::vector<float>& centroids);

int64_t brute_force_dim(ovvsBruteForceIndex_t index);

inline uint32_t fnv1a(const void* data, size_t n) {
  const auto* p = static_cast<const uint8_t*>(data);
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 16777619u;
  }
  return h;
}

inline std::mt19937 rng_from(uint32_t seed) { return std::mt19937(seed); }

}  // namespace impl
}  // namespace ovvs
