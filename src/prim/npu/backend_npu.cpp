#include "internal.hpp"

#if defined(OVVS_WITH_OPENVINO)
#include <openvino/openvino.hpp>
#include <openvino/op/abs.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/fake_quantize.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/negative.hpp>
#include <openvino/op/greater_eq.hpp>
#include <openvino/op/logical_xor.hpp>
#include <openvino/op/power.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/topk.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/opsets/opset8.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace ovvs {
namespace impl {

namespace {

constexpr float kNpuFp16FiniteMax = 65504.0f;
constexpr double kNpuPqAdcScaledHeadroom = 60000.0;
/* The first scaled lane is deliberately narrow: at most 2x de-scaling. Wider
   transforms remain unavailable until candidate-ranking error is measured. */
constexpr double kNpuPqAdcMinScale = 0.5;

bool finite_below_npu_limit(const float* values, int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    if (!std::isfinite(values[i]) || std::fabs(values[i]) >= kNpuFp16FiniteMax) return false;
  }
  return true;
}

struct PqAdcTransform {
  bool active = false;
  float scale = 1.0f;
  double bias = 0.0;
  double scaled_bound = 0.0;
};

bool plan_npu_pq_adc_transform(const float* tables, int32_t pq_m, int32_t ks,
                               bool allow_transform, float* offsets,
                               PqAdcTransform& transform) {
  double accumulation_bound = 0.0;
  for (int32_t m = 0; m < pq_m; ++m) {
    double subspace_bound = 0.0;
    for (int32_t code = 0; code < ks; ++code) {
      const float value = tables[static_cast<size_t>(m) * static_cast<size_t>(ks) +
                                 static_cast<size_t>(code)];
      if (!std::isfinite(value)) return false;
      subspace_bound = std::max(subspace_bound, static_cast<double>(std::fabs(value)));
    }
    accumulation_bound += subspace_bound;
    if (!std::isfinite(accumulation_bound) ||
        accumulation_bound >= static_cast<double>(kNpuFp16FiniteMax)) {
      break;
    }
  }
  if (accumulation_bound < static_cast<double>(kNpuFp16FiniteMax)) return true;
  if (!allow_transform || !offsets) return false;

  double span_bound = 0.0;
  double bias = 0.0;
  for (int32_t m = 0; m < pq_m; ++m) {
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for (int32_t code = 0; code < ks; ++code) {
      const double value = static_cast<double>(
          tables[static_cast<size_t>(m) * static_cast<size_t>(ks) +
                 static_cast<size_t>(code)]);
      if (!std::isfinite(value)) return false;
      lo = std::min(lo, value);
      hi = std::max(hi, value);
    }
    const double span = hi - lo;
    if (!std::isfinite(lo) || !std::isfinite(span) || span < 0.0) return false;
    offsets[m] = static_cast<float>(lo);
    bias += lo;
    span_bound += span;
    if (!std::isfinite(bias) || !std::isfinite(span_bound)) return false;
  }

  double scale = 1.0;
  if (span_bound >= kNpuPqAdcScaledHeadroom) {
    scale = kNpuPqAdcScaledHeadroom / span_bound;
  }
  if (!std::isfinite(scale) || scale < kNpuPqAdcMinScale || scale > 1.0) {
    return false;
  }
  transform.active = true;
  transform.scale = static_cast<float>(scale);
  transform.bias = bias;
  transform.scaled_bound = span_bound * static_cast<double>(transform.scale);
  return std::isfinite(transform.scale) && transform.scale > 0.0f &&
         std::isfinite(transform.scaled_bound) && transform.scaled_bound >= 0.0 &&
         transform.scaled_bound < static_cast<double>(kNpuFp16FiniteMax);
}

bool transformed_npu_adc_output_valid(const float* values, int64_t count,
                                      const PqAdcTransform& transform,
                                      int32_t pq_m) {
  if (!transform.active || !values || count < 0 || pq_m <= 0) return false;
  /* The graph may accumulate in fp16 even with f32 IO. Allow a conservative
     reduction-rounding envelope, but never accept a negative centered sum or
     cross the device's observed finite boundary. */
  constexpr double kFp16RelativeSpacing = 1.0 / 1024.0;
  const double rounding_slack =
      transform.scaled_bound == 0.0
          ? 0.0
          : std::max(1.0, transform.scaled_bound *
                              (static_cast<double>(pq_m) + 1.0) *
                              kFp16RelativeSpacing);
  const double upper = std::min(
      static_cast<double>(kNpuFp16FiniteMax),
      transform.scaled_bound + rounding_slack);
  for (int64_t i = 0; i < count; ++i) {
    const double value = static_cast<double>(values[i]);
    if (!std::isfinite(value) || value < 0.0 ||
        value >= static_cast<double>(kNpuFp16FiniteMax) || value > upper) {
      return false;
    }
  }
  return true;
}

bool npu_gemm_range_safe(const float* a, int64_t na, const float* b, int64_t nb, int64_t k) {
  float amax = 0.0f;
  float bmax = 0.0f;
  for (int64_t i = 0; i < na; ++i) {
    const float magnitude = std::fabs(a[i]);
    if (!std::isfinite(a[i]) || magnitude >= kNpuFp16FiniteMax) return false;
    amax = std::max(amax, magnitude);
  }
  for (int64_t i = 0; i < nb; ++i) {
    const float magnitude = std::fabs(b[i]);
    if (!std::isfinite(b[i]) || magnitude >= kNpuFp16FiniteMax) return false;
    bmax = std::max(bmax, magnitude);
  }

  const double dot_bound = static_cast<double>(amax) * static_cast<double>(bmax) *
                           static_cast<double>(k);
  return std::isfinite(dot_bound) && dot_bound < static_cast<double>(kNpuFp16FiniteMax);
}

void saturating_add_i64(int64_t& dst, int64_t value) {
  if (value <= 0) return;
  if (dst > std::numeric_limits<int64_t>::max() - value) {
    dst = std::numeric_limits<int64_t>::max();
  } else {
    dst += value;
  }
}

void record_npu_adc_execution(ResourcesData& r, int64_t requests, int64_t rows,
                              int64_t capacity_rows, int64_t scaled_chunks,
                              int64_t scaled_rows) noexcept {
  if (requests <= 0 && rows <= 0 && capacity_rows <= 0 && scaled_chunks <= 0 &&
      scaled_rows <= 0) {
    return;
  }
  try {
    std::lock_guard<std::mutex> lock(r.pq_adc_stats_mutex);
    saturating_add_i64(r.pq_adc_npu_requests, requests);
    saturating_add_i64(r.pq_adc_npu_rows, rows);
    saturating_add_i64(r.pq_adc_npu_capacity_rows, capacity_rows);
    saturating_add_i64(r.pq_adc_npu_transformed_chunks, scaled_chunks);
    saturating_add_i64(r.pq_adc_npu_transformed_rows, scaled_rows);
  } catch (...) {
  }
}

}  // namespace

#if defined(OVVS_WITH_OPENVINO)
static ov::Core& ov_core() {
  static ov::Core& core = *new ov::Core();
  return core;
}

struct CachedReq {
  ov::CompiledModel compiled;
  ov::InferRequest req;
  std::mutex mu;
  const void* fp_ptr[4]{};
  size_t fp_n[4]{};
  uint64_t fp_h[4]{};
};

/* Cheap content fingerprint: ends + 32 strided samples. Catches k-means/ScaNN
   rewriting centroids (first row changes). In-place mutation that misses every
   sample is not detected — reallocate the buffer if you mutate a large operand. */
static uint64_t content_fp(const void* p, size_t n) {
  const auto* b = static_cast<const uint8_t*>(p);
  uint64_t h = 0x9e3779b97f4a7c15ull ^ n;
  auto mix = [&](size_t off) {
    uint64_t w = 0;
    const size_t o = off > n - 8 ? n - 8 : off;
    std::memcpy(&w, b + o, 8);
    h ^= w + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
  };
  if (n < 8) {
    uint64_t w = 0;
    if (n) std::memcpy(&w, b, n);
    return h ^ w;
  }
  mix(0);
  mix(n - 8);
  for (int i = 1; i < 32; ++i) mix((n - 8) * static_cast<size_t>(i) / 32);
  return h;
}

/* NPU plugin L0 tensors are allocated at create_infer_request. get_input_tensor /
   get_output_tensor populate those buffers with no extra SHAVE copy. set_tensor on a
   malloc/USM host pointer forces a copy (OpenVINO intel_npu README). Skip memcpy
   when the operand is already in the L0 tensor or the fingerprint matches (search
   dataset / k-means X). Tiny payloads always copy — hash is not cheaper. */
static void fill_in(CachedReq& slot, size_t idx, const void* src, size_t nbytes) {
  ov::Tensor t = slot.req.get_input_tensor(idx);
  if (t.get_byte_size() < nbytes) throw std::runtime_error("npu input smaller than payload");
  void* dst = t.data();
  if (dst == src || src == nullptr || nbytes == 0) return;
  if (nbytes >= 65536 && slot.fp_ptr[idx] == src && slot.fp_n[idx] == nbytes) {
    const uint64_t h = content_fp(src, nbytes);
    if (h == slot.fp_h[idx]) return;
    slot.fp_h[idx] = h;
  } else if (nbytes >= 65536) {
    slot.fp_ptr[idx] = src;
    slot.fp_n[idx] = nbytes;
    slot.fp_h[idx] = content_fp(src, nbytes);
  } else {
    slot.fp_ptr[idx] = nullptr;
    slot.fp_n[idx] = 0;
    slot.fp_h[idx] = 0;
  }
  std::memcpy(dst, src, nbytes);
}

static void read_out(CachedReq& slot, size_t idx, void* dst, size_t nbytes) {
  ov::Tensor t = slot.req.get_output_tensor(idx);
  if (t.get_byte_size() < nbytes) throw std::runtime_error("npu output smaller than payload");
  const void* src = t.data();
  if (src != dst && dst != nullptr && nbytes > 0) std::memcpy(dst, src, nbytes);
}

static ov::CompiledModel compile_on(const char* device, const std::shared_ptr<ov::Model>& model,
                                    const std::string& cache_dir) {
  ov::AnyMap props;
  if (!cache_dir.empty()) props["CACHE_DIR"] = cache_dir;
  if (std::strcmp(device, "NPU") == 0) {
    props["PERFORMANCE_HINT"] = "LATENCY";
    props["NPU_TURBO"] = true;
    props["NPU_COMPILATION_MODE_PARAMS"] =
        std::string("optimization-level=2 performance-hint-override=latency");
  }
  try {
    return ov_core().compile_model(model, device, props);
  } catch (...) {
    props.erase("NPU_COMPILATION_MODE_PARAMS");
    try {
      return ov_core().compile_model(model, device, props);
    } catch (...) {
      props.erase("NPU_TURBO");
      return ov_core().compile_model(model, device, props);
    }
  }
}

static CachedReq& cached_req(const std::string& key, const std::function<ov::CompiledModel()>& make) {
  static auto& g = *new std::unordered_map<std::string, std::unique_ptr<CachedReq>>();
  static std::mutex gmu;
  std::lock_guard<std::mutex> lk(gmu);
  auto found = g.find(key);
  if (found != g.end()) return *found->second;
  auto slot = std::make_unique<CachedReq>();
  slot->compiled = make();
  slot->req = slot->compiled.create_infer_request();
  CachedReq* published = slot.get();
  g.emplace(key, std::move(slot));
  return *published;
}

#endif

static bool ov_has_device(const char* name) {
#if defined(OVVS_WITH_OPENVINO)
  try {
    for (const auto& d : ov_core().get_available_devices()) {
      if (d == name || d.rfind(name, 0) == 0) return true;
    }
  } catch (...) {
  }
#else
  (void)name;
#endif
  return false;
}

bool ov_device_available(const char* name) { return ov_has_device(name); }

bool npu_available() {
#if defined(OVVS_WITH_OPENVINO)
  return ov_has_device("NPU");
#else
  return false;
#endif
}

bool ov_matmul(ResourcesData& r, const char* device, const float* a, const float* b, float* c,
               int64_t m, int64_t n, int64_t k, bool trans_b) {
#if defined(OVVS_WITH_OPENVINO)
  if (!ov_has_device(device)) return false;
  const bool npu = std::strcmp(device, "NPU") == 0;
  const int64_t na_elems = m * k;
  const int64_t nb_elems = trans_b ? n * k : k * n;
  if (npu && !npu_gemm_range_safe(a, na_elems, b, nb_elems, k)) return false;
  try {
    using namespace ov;
    std::ostringstream key;
    key << device << "_gemm_m" << m << "_n" << n << "_k" << k << "_tb" << int(trans_b);
    std::filesystem::create_directories(r.cache_dir);
    CachedReq& slot = cached_req(key.str(), [&] {
      auto pA = std::make_shared<opset8::Parameter>(element::f32, Shape{static_cast<size_t>(m), static_cast<size_t>(k)});
      auto pB = std::make_shared<opset8::Parameter>(
          element::f32, trans_b ? Shape{static_cast<size_t>(n), static_cast<size_t>(k)}
                                : Shape{static_cast<size_t>(k), static_cast<size_t>(n)});
      auto mm = std::make_shared<opset8::MatMul>(pA, pB, false, trans_b);
      auto model = std::make_shared<Model>(OutputVector{mm}, ParameterVector{pA, pB});
      return compile_on(device, model, r.cache_dir);
    });
    std::lock_guard<std::mutex> lk(slot.mu);
    const size_t na = static_cast<size_t>(m * k) * sizeof(float);
    const size_t nb = static_cast<size_t>((trans_b ? n * k : k * n)) * sizeof(float);
    const size_t nc = static_cast<size_t>(m * n) * sizeof(float);
    fill_in(slot, 0, a, na);
    fill_in(slot, 1, b, nb);
    slot.req.infer();
    read_out(slot, 0, c, nc);
    if (npu && !finite_below_npu_limit(c, m * n)) return false;
    return true;
  } catch (...) {
    ++r.npu_compile_fails;
    return false;
  }
#else
  (void)r;
  (void)device;
  (void)a;
  (void)b;
  (void)c;
  (void)m;
  (void)n;
  (void)k;
  (void)trans_b;
  return false;
#endif
}

#if defined(OVVS_WITH_OPENVINO)
static const char* compute_tag(ovvsDType d) {
  switch (d) {
    case OVVS_DTYPE_F16:
      return "f16";
    case OVVS_DTYPE_I8:
      return "i8";
    case OVVS_DTYPE_F8E4M3:
      return "f8e4m3";
    case OVVS_DTYPE_F8E5M2:
      return "f8e5m2";
    case OVVS_DTYPE_F4E2M1:
      return "f4e2m1";
    default:
      return "f32";
  }
}

static ov::element::Type ov_compute_type(ovvsDType d) {
  using ov::element::Type;
  switch (d) {
    case OVVS_DTYPE_F16:
      return ov::element::f16;
    case OVVS_DTYPE_I8:
      return ov::element::i8;
    case OVVS_DTYPE_F8E4M3:
      return ov::element::f8e4m3;
    case OVVS_DTYPE_F8E5M2:
      return ov::element::f8e5m2;
    case OVVS_DTYPE_F4E2M1:
      return ov::element::f4e2m1;
    default:
      return ov::element::f32;
  }
}

static float max_abs(const float* p, int64_t n) {
  float m = 0.f;
  for (int64_t i = 0; i < n; ++i) {
    const float a = std::fabs(p[i]);
    if (a > m) m = a;
  }
  return m;
}

#endif

bool ov_matmul_compute(ResourcesData& r, const char* device, ovvsDType compute, const float* a,
                       const float* b, float* c, int64_t m, int64_t n, int64_t k, bool trans_b) {
  if (compute == OVVS_DTYPE_F32 || compute == OVVS_DTYPE_U8) {
    const bool ok = ov_matmul(r, device, a, b, c, m, n, k, trans_b);
    if (ok) r.last_compute_dtype = OVVS_DTYPE_F32;
    return ok;
  }
#if defined(OVVS_WITH_OPENVINO)
  if (!ov_has_device(device)) return false;
  const bool npu = std::strcmp(device, "NPU") == 0;
  const int64_t na_elems = m * k;
  const int64_t nb_elems = trans_b ? n * k : k * n;
  if (npu && !npu_gemm_range_safe(a, na_elems, b, nb_elems, k)) return false;
  try {
    using namespace ov;
    const Shape sa{static_cast<size_t>(m), static_cast<size_t>(k)};
    const Shape sb = trans_b ? Shape{static_cast<size_t>(n), static_cast<size_t>(k)}
                             : Shape{static_cast<size_t>(k), static_cast<size_t>(n)};
    std::ostringstream key;
    key << device << "_gemm_" << compute_tag(compute) << "_m" << m << "_n" << n << "_k" << k << "_tb"
        << int(trans_b);
    std::filesystem::create_directories(r.cache_dir);
    const bool i8 = compute == OVVS_DTYPE_I8;
    CachedReq& slot = cached_req(key.str(), [&] {
      if (i8) {
        auto pA = std::make_shared<opset8::Parameter>(element::f32, sa);
        auto pB = std::make_shared<opset8::Parameter>(element::f32, sb);
        auto pRa = std::make_shared<opset8::Parameter>(element::f32, Shape{1});
        auto pRb = std::make_shared<opset8::Parameter>(element::f32, Shape{1});
        auto loA = std::make_shared<opset8::Negative>(pRa);
        auto loB = std::make_shared<opset8::Negative>(pRb);
        auto fqA = std::make_shared<opset8::FakeQuantize>(pA, loA, pRa, loA, pRa, 256);
        auto fqB = std::make_shared<opset8::FakeQuantize>(pB, loB, pRb, loB, pRb, 256);
        auto mm = std::make_shared<opset8::MatMul>(fqA, fqB, false, trans_b);
        auto model = std::make_shared<Model>(OutputVector{mm}, ParameterVector{pA, pB, pRa, pRb});
        return compile_on(device, model, r.cache_dir);
      }
      auto pA = std::make_shared<opset8::Parameter>(element::f32, sa);
      auto pB = std::make_shared<opset8::Parameter>(element::f32, sb);
      const auto et = ov_compute_type(compute);
      auto a_c = std::make_shared<opset8::Convert>(pA, et);
      auto b_c = std::make_shared<opset8::Convert>(pB, et);
      auto mm = std::make_shared<opset8::MatMul>(a_c, b_c, false, trans_b);
      auto out = std::make_shared<opset8::Convert>(mm, element::f32);
      auto model = std::make_shared<Model>(OutputVector{out}, ParameterVector{pA, pB});
      return compile_on(device, model, r.cache_dir);
    });
    std::lock_guard<std::mutex> lk(slot.mu);
    const size_t na = static_cast<size_t>(m * k) * sizeof(float);
    const size_t nb = static_cast<size_t>((trans_b ? n * k : k * n)) * sizeof(float);
    const size_t nc = static_cast<size_t>(m * n) * sizeof(float);
    fill_in(slot, 0, a, na);
    fill_in(slot, 1, b, nb);
    if (i8) {
      const int64_t n_a = m * k;
      const int64_t n_b = trans_b ? n * k : k * n;
      float ra = std::max(max_abs(a, n_a), 1e-8f);
      float rb = std::max(max_abs(b, n_b), 1e-8f);
      fill_in(slot, 2, &ra, sizeof(ra));
      fill_in(slot, 3, &rb, sizeof(rb));
    }
    slot.req.infer();
    read_out(slot, 0, c, nc);
    if (npu && !finite_below_npu_limit(c, m * n)) return false;
    r.last_compute_dtype = compute;
    return true;
  } catch (...) {
    ++r.npu_compile_fails;
    return false;
  }
#else
  (void)r;
  (void)device;
  (void)compute;
  (void)a;
  (void)b;
  (void)c;
  (void)m;
  (void)n;
  (void)k;
  (void)trans_b;
  return false;
#endif
}

bool npu_gemm_compute(ResourcesData& r, ovvsDType compute, const float* a, const float* b, float* c,
                      int64_t m, int64_t n, int64_t k, bool trans_b) {
  if (ov_matmul_compute(r, "NPU", compute, a, b, c, m, n, k, trans_b)) return true;
  constexpr int64_t kTile = 256;
  if (m <= kTile) return false;
  for (int64_t i = 0; i < m; i += kTile) {
    const int64_t rows = std::min(kTile, m - i);
    if (!ov_matmul_compute(r, "NPU", compute, a + i * k, b, c + i * n, rows, n, k, trans_b)) return false;
  }
  return true;
}

bool npu_gemm(ResourcesData& r, const float* a, const float* b, float* c, int64_t m, int64_t n,
              int64_t k, bool trans_b) {
  /* Default API is fp32. INT8/FP16 only via npu_gemm_compute / ovvsGemmEx — do not silently
     swap in FakeQuantize on the hot f32 path (it lost to f32 DPU+DMA on this SKU). */
  return npu_gemm_compute(r, OVVS_DTYPE_F32, a, b, c, m, n, k, trans_b);
}

#if defined(OVVS_WITH_OPENVINO)
static bool try_compile_lowbit(const char* device, ovvsDType compute) {
  ResourcesData tmp;
  const float a[16] = {0.1f, 0.2f, -0.3f, 0.4f, 0.5f, -0.1f, 0.2f, 0.3f,
                       0.0f, 0.1f, 0.2f, -0.2f, 0.3f, 0.1f, -0.4f, 0.2f};
  const float b[16] = {0.2f, -0.1f, 0.3f, 0.1f, 0.0f, 0.4f, -0.2f, 0.1f,
                       0.3f, 0.2f, -0.1f, 0.2f, 0.1f, 0.0f, 0.2f, -0.3f};
  float c[16] = {0};
  return ov_matmul_compute(tmp, device, compute, a, b, c, 4, 4, 4, true);
}
#endif

void append_lowbit_probe_json(std::ostringstream& o) {
  static std::once_flag once;
  static std::string cached;
  std::call_once(once, [] {
    std::ostringstream s;
#if defined(OVVS_WITH_OPENVINO)
    const bool npu = ov_has_device("NPU");
    const bool gpu = ov_has_device("GPU");
    auto one = [&](const char* dev, bool present, ovvsDType dt) {
      if (!present) return std::string("absent");
      return try_compile_lowbit(dev, dt) ? std::string("ok") : std::string("compile_fail");
    };
    s << "  \"npu_matmul_f16\": \"" << one("NPU", npu, OVVS_DTYPE_F16) << "\",\n";
    s << "  \"npu_matmul_i8\": \"" << one("NPU", npu, OVVS_DTYPE_I8) << "\",\n";
    s << "  \"npu_matmul_f8e4m3\": \"" << one("NPU", npu, OVVS_DTYPE_F8E4M3) << "\",\n";
    s << "  \"npu_matmul_f4e2m1\": \"" << one("NPU", npu, OVVS_DTYPE_F4E2M1) << "\",\n";
    s << "  \"gpu_matmul_f16\": \"" << one("GPU", gpu, OVVS_DTYPE_F16) << "\",\n";
    s << "  \"gpu_matmul_i8\": \"" << one("GPU", gpu, OVVS_DTYPE_I8) << "\",\n";
#else
    s << "  \"npu_matmul_f16\": \"no_openvino\",\n";
    s << "  \"npu_matmul_i8\": \"no_openvino\",\n";
    s << "  \"npu_matmul_f8e4m3\": \"no_openvino\",\n";
    s << "  \"npu_matmul_f4e2m1\": \"no_openvino\",\n";
    s << "  \"gpu_matmul_f16\": \"no_openvino\",\n";
    s << "  \"gpu_matmul_i8\": \"no_openvino\",\n";
#endif
    cached = s.str();
  });
  o << cached;
}

bool ov_topk(ResourcesData& r, const char* device, const float* scores, int64_t rows, int64_t cols,
             int64_t k, int64_t* indices, float* values, bool largest) {
#if defined(OVVS_WITH_OPENVINO)
  if (!ov_has_device(device)) return false;
  k = std::min(k, cols);
  const bool npu = std::strcmp(device, "NPU") == 0;
  if (npu && !finite_below_npu_limit(scores, rows * cols)) return false;
  try {
    using namespace ov;
    std::ostringstream key;
    key << device << "_topk_r" << rows << "_c" << cols << "_k" << k << "_lg" << int(largest);
    std::filesystem::create_directories(r.cache_dir);
    CachedReq& slot = cached_req(key.str(), [&] {
      auto pScores = std::make_shared<opset8::Parameter>(
          element::f32, Shape{static_cast<size_t>(rows), static_cast<size_t>(cols)});
      auto kconst = opset8::Constant::create(element::i32, Shape{}, {static_cast<int32_t>(k)});
      auto topk = std::make_shared<ov::op::v3::TopK>(pScores, kconst, /*axis*/ 1,
                                                     largest ? "max" : "min", "value", element::i32);
      auto model = std::make_shared<Model>(OutputVector{topk->output(0), topk->output(1)},
                                           ParameterVector{pScores});
      return compile_on(device, model, r.cache_dir);
    });
    std::lock_guard<std::mutex> lk(slot.mu);
    fill_in(slot, 0, scores, static_cast<size_t>(rows * cols) * sizeof(float));
    slot.req.infer();
    read_out(slot, 0, values, static_cast<size_t>(rows * k) * sizeof(float));
    Tensor tI = slot.req.get_output_tensor(1);
    if (tI.get_element_type() == element::i32) {
      const int32_t* pi = tI.data<int32_t>();
      for (int64_t i = 0; i < rows * k; ++i) indices[i] = pi[i];
    } else if (tI.get_element_type() == element::i64) {
      std::memcpy(indices, tI.data<int64_t>(), static_cast<size_t>(rows * k) * sizeof(int64_t));
    } else {
      return false;
    }
    if (npu) {
      if (!finite_below_npu_limit(values, rows * k)) return false;
      for (int64_t i = 0; i < rows * k; ++i) {
        if (indices[i] < 0 || indices[i] >= cols) return false;
      }
    }
    return true;
  } catch (...) {
    ++r.npu_compile_fails;
    return false;
  }
#else
  (void)r;
  (void)device;
  (void)scores;
  (void)rows;
  (void)cols;
  (void)k;
  (void)indices;
  (void)values;
  (void)largest;
  return false;
#endif
}

bool ov_gather_rows(ResourcesData& r, const char* device, const float* src, int64_t src_rows,
                    int64_t dim, const int64_t* idx, int64_t nidx, float* out) {
#if defined(OVVS_WITH_OPENVINO)
  if (!ov_has_device(device)) return false;
  try {
    using namespace ov;
    std::ostringstream key;
    key << device << "_gather_sr" << src_rows << "_d" << dim << "_n" << nidx;
    std::filesystem::create_directories(r.cache_dir);
    CachedReq& slot = cached_req(key.str(), [&] {
      auto pSrc = std::make_shared<opset8::Parameter>(
          element::f32, Shape{static_cast<size_t>(src_rows), static_cast<size_t>(dim)});
      auto pIdx = std::make_shared<opset8::Parameter>(element::i32, Shape{static_cast<size_t>(nidx)});
      auto axis = opset8::Constant::create(element::i64, Shape{}, {0});
      auto g = std::make_shared<ov::op::v8::Gather>(pSrc, pIdx, axis);
      auto model = std::make_shared<Model>(OutputVector{g}, ParameterVector{pSrc, pIdx});
      return compile_on(device, model, r.cache_dir);
    });
    std::lock_guard<std::mutex> lk(slot.mu);
    fill_in(slot, 0, src, static_cast<size_t>(src_rows * dim) * sizeof(float));
    Tensor tI = slot.req.get_input_tensor(1);
    if (tI.get_byte_size() < static_cast<size_t>(nidx) * sizeof(int32_t)) return false;
    int32_t* i32 = tI.data<int32_t>();
    for (int64_t i = 0; i < nidx; ++i) i32[static_cast<size_t>(i)] = static_cast<int32_t>(idx[i]);
    slot.req.infer();
    read_out(slot, 0, out, static_cast<size_t>(nidx * dim) * sizeof(float));
    return true;
  } catch (...) {
    ++r.npu_compile_fails;
    return false;
  }
#else
  (void)r;
  (void)device;
  (void)src;
  (void)src_rows;
  (void)dim;
  (void)idx;
  (void)nidx;
  (void)out;
  return false;
#endif
}

bool npu_topk(ResourcesData& r, const float* scores, int64_t rows, int64_t cols, int64_t k,
              int64_t* indices, float* values, bool largest) {
  if (ov_topk(r, "NPU", scores, rows, cols, k, indices, values, largest)) return true;
  constexpr int64_t kTile = 32;
  if (rows <= kTile) return false;
  for (int64_t i = 0; i < rows; i += kTile) {
    const int64_t rr = std::min(kTile, rows - i);
    if (!ov_topk(r, "NPU", scores + i * cols, rr, cols, k, indices + i * k, values + i * k, largest)) {
      return false;
    }
  }
  return true;
}

bool npu_gather_rows(ResourcesData& r, const float* src, int64_t src_rows, int64_t dim,
                     const int64_t* idx, int64_t nidx, float* out) {
  if (ov_gather_rows(r, "NPU", src, src_rows, dim, idx, nidx, out)) return true;
  constexpr int64_t kTile = 128;
  if (nidx <= kTile) return false;
  for (int64_t i = 0; i < nidx; i += kTile) {
    const int64_t nn = std::min(kTile, nidx - i);
    if (!ov_gather_rows(r, "NPU", src, src_rows, dim, idx + i, nn, out + i * dim)) return false;
  }
  return true;
}

bool npu_pq_adc_batch(ResourcesData& r, const PqAdcChunk* chunks, int64_t chunk_count,
                      int32_t pq_m, int32_t ks, float* out) {
#if defined(OVVS_WITH_OPENVINO)
  if (!ov_has_device("NPU")) return false;
  if (!chunks || !out || chunk_count <= 0 || pq_m <= 0 || ks <= 0 || ks > 256) return false;
  if (pq_m > std::numeric_limits<int32_t>::max() / ks) return false;
  if (static_cast<int64_t>(pq_m) > std::numeric_limits<int64_t>::max() / ks) return false;
  const int64_t lut_n = static_cast<int64_t>(pq_m) * ks;
  if (static_cast<uint64_t>(lut_n) >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(float))) {
    return false;
  }

  constexpr int32_t kBuckets[] = {128, 256, 512, 1024, 2048};
  int64_t executed_requests = 0;
  int64_t executed_rows = 0;
  int64_t executed_capacity_rows = 0;
  int64_t executed_scaled_chunks = 0;
  int64_t executed_scaled_rows = 0;
  bool creating_request = false;
  try {
    using namespace ov;
    std::vector<int64_t> bucket_chunks[5];
    std::vector<std::pair<int64_t, int64_t>> output_ranges;
    output_ranges.reserve(static_cast<size_t>(chunk_count));
    const bool allow_transform = r.policy == OVVS_POLICY_FORCE_NPU;
    std::vector<PqAdcTransform> transforms(static_cast<size_t>(chunk_count));
    std::vector<float> transform_offsets;
    if (allow_transform) {
      if (static_cast<uint64_t>(chunk_count) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
              static_cast<uint64_t>(pq_m)) {
        return false;
      }
      transform_offsets.resize(static_cast<size_t>(chunk_count) *
                               static_cast<size_t>(pq_m));
    }
    int64_t max_output_end = 0;

    /* Validate the complete logical call before compiling or submitting any request.
       FORCE_NPU relies on this to fail without partially executing a safe prefix. */
    for (int64_t ci = 0; ci < chunk_count; ++ci) {
      const PqAdcChunk& chunk = chunks[ci];
      if (!chunk.tables || !chunk.codes || chunk.valid_rows <= 0 || chunk.bucket_rows <= 0 ||
          chunk.valid_rows > chunk.bucket_rows || chunk.output_offset < 0 ||
          chunk.output_offset > std::numeric_limits<int64_t>::max() - chunk.valid_rows) {
        return false;
      }
      int bucket_index = -1;
      for (int bi = 0; bi < 5; ++bi) {
        if (chunk.bucket_rows == kBuckets[bi]) {
          bucket_index = bi;
          break;
        }
      }
      float* offsets =
          allow_transform
              ? transform_offsets.data() +
                    static_cast<size_t>(ci) * static_cast<size_t>(pq_m)
              : nullptr;
      if (bucket_index < 0 ||
          !plan_npu_pq_adc_transform(chunk.tables, pq_m, ks,
                                     allow_transform, offsets,
                                     transforms[static_cast<size_t>(ci)])) {
        return false;
      }
      for (int32_t row = 0; row < chunk.valid_rows; ++row) {
        const uint8_t* code =
            chunk.codes + static_cast<size_t>(row) * static_cast<size_t>(pq_m);
        for (int32_t m = 0; m < pq_m; ++m) {
          if (static_cast<int32_t>(code[m]) >= ks) return false;
        }
      }
      const int64_t output_end = chunk.output_offset + chunk.valid_rows;
      max_output_end = std::max(max_output_end, output_end);
      output_ranges.emplace_back(chunk.output_offset, output_end);
      bucket_chunks[bucket_index].push_back(ci);
    }
    std::sort(output_ranges.begin(), output_ranges.end());
    for (size_t i = 1; i < output_ranges.size(); ++i) {
      if (output_ranges[i].first < output_ranges[i - 1].second) return false;
    }
    if (static_cast<uint64_t>(max_output_end) >
        static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(float))) {
      return false;
    }

    std::vector<float> staged(static_cast<size_t>(max_output_end));
    std::filesystem::create_directories(r.cache_dir);
    for (int bucket_index = 0; bucket_index < 5; ++bucket_index) {
      const std::vector<int64_t>& group = bucket_chunks[bucket_index];
      if (group.empty()) continue;
      const int32_t bucket = kBuckets[bucket_index];
      const int64_t index_elems_per_chunk = static_cast<int64_t>(bucket) * pq_m;
      const int32_t raw_capacity_limit = static_cast<int32_t>(
          std::max<int64_t>(1, std::min<int64_t>(32, 65536 / index_elems_per_chunk)));
      int32_t capacity_limit = 1;
      while (capacity_limit <= raw_capacity_limit / 2) capacity_limit *= 2;

      for (size_t base = 0; base < group.size();) {
        const size_t remaining = group.size() - base;
        const size_t target =
            std::min(remaining, static_cast<size_t>(capacity_limit));
        int32_t capacity = 1;
        while (static_cast<size_t>(capacity) < target) capacity *= 2;
        const size_t active = std::min(static_cast<size_t>(capacity), remaining);
        std::ostringstream key;
        key << "NPU_adc_v3_m" << pq_m << "_ks" << ks << "_b" << bucket << "_cap"
            << capacity << "_latency";
        creating_request = true;
        CachedReq& slot = cached_req(key.str(), [&] {
          auto pLut = std::make_shared<opset8::Parameter>(
              element::f32,
              Shape{static_cast<size_t>(capacity), static_cast<size_t>(lut_n)});
          auto pIdx = std::make_shared<opset8::Parameter>(
              element::i32,
              Shape{static_cast<size_t>(capacity), static_cast<size_t>(bucket),
                    static_cast<size_t>(pq_m)});
          auto axis = opset8::Constant::create(element::i64, Shape{}, {1});
          auto g = std::make_shared<ov::op::v8::Gather>(pLut, pIdx, axis, 1);
          auto red_axes = opset8::Constant::create(element::i64, Shape{1}, {2});
          auto red = std::make_shared<ov::op::v1::ReduceSum>(g, red_axes, false);
          auto model = std::make_shared<Model>(OutputVector{red}, ParameterVector{pLut, pIdx});
          return compile_on("NPU", model, r.cache_dir);
        });
        creating_request = false;
        std::lock_guard<std::mutex> lk(slot.mu);
        Tensor tLut = slot.req.get_input_tensor(0);
        Tensor tIdx = slot.req.get_input_tensor(1);
        const size_t lut_elems = static_cast<size_t>(capacity) * static_cast<size_t>(lut_n);
        const size_t idx_elems = static_cast<size_t>(capacity) * static_cast<size_t>(bucket) *
                                 static_cast<size_t>(pq_m);
        if (tLut.get_byte_size() < lut_elems * sizeof(float) ||
            tIdx.get_byte_size() < idx_elems * sizeof(int32_t)) {
          throw std::runtime_error("npu ADC request tensor smaller than fixed batch");
        }
        float* lutp = tLut.data<float>();
        int32_t* idxp = tIdx.data<int32_t>();
        std::fill(lutp, lutp + lut_elems, 0.0f);
        for (int32_t batch = 0; batch < capacity; ++batch) {
          for (int32_t row = 0; row < bucket; ++row) {
            int32_t* dst = idxp +
                           (static_cast<size_t>(batch) * static_cast<size_t>(bucket) +
                            static_cast<size_t>(row)) *
                               static_cast<size_t>(pq_m);
            for (int32_t m = 0; m < pq_m; ++m) dst[m] = m * ks;
          }
        }

        int64_t request_rows = 0;
        int64_t request_scaled_chunks = 0;
        int64_t request_scaled_rows = 0;
        for (size_t batch = 0; batch < active; ++batch) {
          const int64_t chunk_index = group[base + batch];
          const PqAdcChunk& chunk = chunks[chunk_index];
          const PqAdcTransform& transform =
              transforms[static_cast<size_t>(chunk_index)];
          float* lut_dst = lutp + batch * static_cast<size_t>(lut_n);
          if (!transform.active) {
            std::memcpy(lut_dst, chunk.tables,
                        static_cast<size_t>(lut_n) * sizeof(float));
          } else {
            const float* offsets =
                transform_offsets.data() +
                static_cast<size_t>(chunk_index) * static_cast<size_t>(pq_m);
            for (int32_t m = 0; m < pq_m; ++m) {
              const double offset = static_cast<double>(offsets[m]);
              for (int32_t code = 0; code < ks; ++code) {
                const size_t lut_index =
                    static_cast<size_t>(m) * static_cast<size_t>(ks) +
                    static_cast<size_t>(code);
                const double centered =
                    static_cast<double>(chunk.tables[lut_index]) - offset;
                const double scaled = centered * static_cast<double>(transform.scale);
                if (!std::isfinite(scaled) || scaled < 0.0 ||
                    scaled >= static_cast<double>(kNpuFp16FiniteMax)) {
                  throw std::runtime_error("npu ADC transformed LUT outside finite range");
                }
                lut_dst[lut_index] = static_cast<float>(scaled);
              }
            }
            ++request_scaled_chunks;
            saturating_add_i64(request_scaled_rows, chunk.valid_rows);
          }
          for (int32_t row = 0; row < chunk.valid_rows; ++row) {
            const uint8_t* code =
                chunk.codes + static_cast<size_t>(row) * static_cast<size_t>(pq_m);
            int32_t* dst = idxp +
                           (batch * static_cast<size_t>(bucket) + static_cast<size_t>(row)) *
                               static_cast<size_t>(pq_m);
            for (int32_t m = 0; m < pq_m; ++m) {
              dst[m] = m * ks + static_cast<int32_t>(code[m]);
            }
          }
          saturating_add_i64(request_rows, chunk.valid_rows);
        }

        slot.req.infer();
        ++executed_requests;
        saturating_add_i64(executed_rows, request_rows);
        saturating_add_i64(executed_capacity_rows,
                           static_cast<int64_t>(capacity) * bucket);
        saturating_add_i64(executed_scaled_chunks, request_scaled_chunks);
        saturating_add_i64(executed_scaled_rows, request_scaled_rows);
        Tensor result = slot.req.get_output_tensor(0);
        const size_t result_elems = static_cast<size_t>(capacity) * static_cast<size_t>(bucket);
        if (result.get_byte_size() < result_elems * sizeof(float)) {
          throw std::runtime_error("npu ADC output tensor smaller than fixed batch");
        }
        const float* result_data = result.data<const float>();
        for (size_t batch = 0; batch < active; ++batch) {
          const int64_t chunk_index = group[base + batch];
          const PqAdcChunk& chunk = chunks[chunk_index];
          const PqAdcTransform& transform =
              transforms[static_cast<size_t>(chunk_index)];
          const float* src = result_data + batch * static_cast<size_t>(bucket);
          if ((!transform.active && !finite_below_npu_limit(src, chunk.valid_rows)) ||
              (transform.active &&
               !transformed_npu_adc_output_valid(src, chunk.valid_rows, transform,
                                                 pq_m))) {
            throw std::runtime_error("npu ADC output outside finite range");
          }
          float* dst = staged.data() + chunk.output_offset;
          if (!transform.active) {
            std::memcpy(dst, src, static_cast<size_t>(chunk.valid_rows) * sizeof(float));
          } else {
            const double inverse_scale = 1.0 / static_cast<double>(transform.scale);
            for (int32_t row = 0; row < chunk.valid_rows; ++row) {
              const double restored = static_cast<double>(src[row]) * inverse_scale +
                                      transform.bias;
              if (!std::isfinite(restored) ||
                  std::fabs(restored) >
                      static_cast<double>(std::numeric_limits<float>::max())) {
                throw std::runtime_error("npu ADC restored output is not finite f32");
              }
              dst[row] = static_cast<float>(restored);
            }
          }
        }
        base += active;
      }
    }
    for (int64_t ci = 0; ci < chunk_count; ++ci) {
      const PqAdcChunk& chunk = chunks[ci];
      std::memcpy(out + chunk.output_offset, staged.data() + chunk.output_offset,
                  static_cast<size_t>(chunk.valid_rows) * sizeof(float));
    }
    record_npu_adc_execution(r, executed_requests, executed_rows, executed_capacity_rows,
                             executed_scaled_chunks, executed_scaled_rows);
    return true;
  } catch (...) {
    record_npu_adc_execution(r, executed_requests, executed_rows, executed_capacity_rows,
                             executed_scaled_chunks, executed_scaled_rows);
    int32_t& failure_counter = creating_request ? r.npu_compile_fails : r.npu_runtime_fails;
    if (failure_counter < std::numeric_limits<int32_t>::max()) ++failure_counter;
    return false;
  }
#else
  (void)r;
  (void)chunks;
  (void)chunk_count;
  (void)pq_m;
  (void)ks;
  (void)out;
  return false;
#endif
}

bool ov_pairwise_npu(ResourcesData& r, ovvsMetric metric, const float* x, int64_t nx, const float* y,
                     int64_t ny, int64_t dim, float* out, float metric_arg) {
#if defined(OVVS_WITH_OPENVINO)
  if (!ov_has_device("NPU")) return false;
  if (metric != OVVS_METRIC_BITWISE_HAMMING && metric != OVVS_METRIC_LP_UNEXPANDED) return false;
  try {
    using namespace ov;
    const float p = metric_arg > 0.f ? metric_arg : 2.f;
    std::ostringstream key;
    key << "NPU_pair_m" << static_cast<int>(metric) << "_nx" << nx << "_ny" << ny << "_d" << dim << "_p" << p;
    std::filesystem::create_directories(r.cache_dir);
    CachedReq& slot = cached_req(key.str(), [&] {
      auto pX = std::make_shared<opset8::Parameter>(element::f32,
                                                    Shape{static_cast<size_t>(nx), static_cast<size_t>(dim)});
      auto pY = std::make_shared<opset8::Parameter>(element::f32,
                                                    Shape{static_cast<size_t>(ny), static_cast<size_t>(dim)});
      auto uns1 = opset8::Constant::create(element::i64, Shape{1}, {1});
      auto uns0 = opset8::Constant::create(element::i64, Shape{1}, {0});
      auto x3 = std::make_shared<ov::op::v0::Unsqueeze>(pX, uns1);
      auto y3 = std::make_shared<ov::op::v0::Unsqueeze>(pY, uns0);
      auto axes = opset8::Constant::create(element::i64, Shape{1}, {2});
      std::shared_ptr<ov::Node> scores;
      if (metric == OVVS_METRIC_BITWISE_HAMMING) {
        auto zero = opset8::Constant::create(element::f32, Shape{}, {0.f});
        auto bx = std::make_shared<ov::op::v1::GreaterEqual>(x3, zero);
        auto by = std::make_shared<ov::op::v1::GreaterEqual>(y3, zero);
        auto lx = std::make_shared<ov::op::v1::LogicalXor>(bx, by);
        auto fx = std::make_shared<ov::op::v0::Convert>(lx, element::f32);
        scores = std::make_shared<ov::op::v1::ReduceSum>(fx, axes, false);
      } else {
        auto diff = std::make_shared<ov::op::v1::Subtract>(x3, y3);
        auto ad = std::make_shared<ov::op::v0::Abs>(diff);
        auto pconst = opset8::Constant::create(element::f32, Shape{}, {p});
        auto pw = std::make_shared<ov::op::v1::Power>(ad, pconst);
        auto sm = std::make_shared<ov::op::v1::ReduceSum>(pw, axes, false);
        auto invp = opset8::Constant::create(element::f32, Shape{}, {1.f / p});
        scores = std::make_shared<ov::op::v1::Power>(sm, invp);
      }
      auto model = std::make_shared<Model>(OutputVector{scores}, ParameterVector{pX, pY});
      return compile_on("NPU", model, r.cache_dir);
    });
    std::lock_guard<std::mutex> lk(slot.mu);
    fill_in(slot, 0, x, static_cast<size_t>(nx * dim) * sizeof(float));
    fill_in(slot, 1, y, static_cast<size_t>(ny * dim) * sizeof(float));
    slot.req.infer();
    read_out(slot, 0, out, static_cast<size_t>(nx * ny) * sizeof(float));
    return true;
  } catch (...) {
    ++r.npu_compile_fails;
    return false;
  }
#else
  (void)r;
  (void)metric;
  (void)x;
  (void)nx;
  (void)y;
  (void)ny;
  (void)dim;
  (void)out;
  (void)metric_arg;
  return false;
#endif
}

bool npu_pairwise(ResourcesData& r, ovvsMetric metric, const float* x, int64_t nx, const float* y,
                  int64_t ny, int64_t dim, float* out, float metric_arg) {
  if (ov_pairwise_npu(r, metric, x, nx, y, ny, dim, out, metric_arg)) return true;
  constexpr int64_t kTile = 32;
  if (nx <= kTile) return false;
  for (int64_t i = 0; i < nx; i += kTile) {
    const int64_t rows = std::min(kTile, nx - i);
    if (!ov_pairwise_npu(r, metric, x + i * dim, rows, y, ny, dim, out + i * ny, metric_arg)) return false;
  }
  return true;
}

bool npu_shave_profile_adc(int* shave_tasks, int* dpu_tasks, std::vector<uint8_t>* blob,
                           std::string* exec_types) {
#if defined(OVVS_WITH_OPENVINO)
  if (!ov_has_device("NPU")) return false;
  try {
    using namespace ov;
    constexpr int32_t pq_m = 8, ks = 16, nn = 4;
    const int64_t lut_n = static_cast<int64_t>(pq_m) * ks;
    auto pLut = std::make_shared<opset8::Parameter>(element::f32, Shape{static_cast<size_t>(lut_n)});
    auto pIdx = std::make_shared<opset8::Parameter>(
        element::i32, Shape{static_cast<size_t>(nn), static_cast<size_t>(pq_m)});
    auto axis = opset8::Constant::create(element::i64, Shape{}, {0});
    auto g = std::make_shared<ov::op::v8::Gather>(pLut, pIdx, axis);
    auto red_axes = opset8::Constant::create(element::i64, Shape{1}, {1});
    auto red = std::make_shared<ov::op::v1::ReduceSum>(g, red_axes, false);
    auto model = std::make_shared<Model>(OutputVector{red}, ParameterVector{pLut, pIdx});
    auto compiled = ov_core().compile_model(model, "NPU", ov::AnyMap{{"PERF_COUNT", true}});
    std::vector<float> lut(static_cast<size_t>(lut_n), 1.f);
    std::vector<int32_t> idx(static_cast<size_t>(nn * pq_m), 0);
    auto req = compiled.create_infer_request();
    req.set_input_tensor(0, Tensor(element::f32, Shape{static_cast<size_t>(lut_n)}, lut.data()));
    req.set_input_tensor(1, Tensor(element::i32, Shape{static_cast<size_t>(nn), static_cast<size_t>(pq_m)},
                                   idx.data()));
    req.infer();
    int shave = 0, dpu = 0;
    std::string types;
    for (const auto& p : req.get_profiling_info()) {
      if (!types.empty()) types += ",";
      types += p.exec_type;
      if (p.exec_type.find("have") != std::string::npos || p.exec_type.find("SHAVE") != std::string::npos) {
        ++shave;
      } else if (p.exec_type.find("DPU") != std::string::npos) {
        ++dpu;
      }
    }
    if (shave_tasks) *shave_tasks = shave;
    if (dpu_tasks) *dpu_tasks = dpu;
    if (exec_types) *exec_types = types;
    if (blob) {
      std::stringstream ss;
      compiled.export_model(ss);
      const std::string s = ss.str();
      blob->assign(s.begin(), s.end());
    }
    return shave > 0;
  } catch (...) {
    return false;
  }
#else
  (void)shave_tasks;
  (void)dpu_tasks;
  (void)blob;
  (void)exec_types;
  return false;
#endif
}

}  // namespace impl
}  // namespace ovvs
