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

#if defined(OVVS_WITH_OPENVINO)
static ov::Core& ov_core() {
  static ov::Core& core = *new ov::Core();
  return core;
}

struct CachedReq {
  ov::CompiledModel compiled;
  ov::InferRequest req;
  std::mutex mu;
};

/* NPU plugin L0 tensors are allocated at create_infer_request. get_input_tensor /
   get_output_tensor populate those buffers with no extra SHAVE copy. set_tensor on a
   malloc/USM host pointer forces a copy (OpenVINO intel_npu README). Always memcpy:
   k-means / ScaNN mutate centroids in place behind a stable pointer. */
static void fill_in(CachedReq& slot, size_t idx, const void* src, size_t nbytes) {
  ov::Tensor t = slot.req.get_input_tensor(idx);
  if (t.get_byte_size() < nbytes) throw std::runtime_error("npu input smaller than payload");
  void* dst = t.data();
  if (dst != src && src != nullptr && nbytes > 0) std::memcpy(dst, src, nbytes);
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
  auto& slot = g[key];
  if (!slot) {
    slot = std::make_unique<CachedReq>();
    slot->compiled = make();
    slot->req = slot->compiled.create_infer_request();
  }
  return *slot;
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

bool npu_pq_adc(ResourcesData& r, const float* tables, int32_t pq_m, int32_t ks, const uint8_t* codes,
                int64_t ncodes, float* out) {
#if defined(OVVS_WITH_OPENVINO)
  if (!ov_has_device("NPU")) return false;
  constexpr int64_t kTile = 128;
  try {
    using namespace ov;
    const int64_t lut_n = static_cast<int64_t>(pq_m) * ks;
    std::filesystem::create_directories(r.cache_dir);
    for (int64_t off = 0; off < ncodes; off += kTile) {
      const int64_t nn = std::min(kTile, ncodes - off);
      std::ostringstream key;
      key << "NPU_adc_m" << pq_m << "_ks" << ks << "_n" << nn;
      CachedReq& slot = cached_req(key.str(), [&] {
        auto pLut = std::make_shared<opset8::Parameter>(element::f32, Shape{static_cast<size_t>(lut_n)});
        auto pIdx = std::make_shared<opset8::Parameter>(element::i32, Shape{static_cast<size_t>(nn),
                                                                            static_cast<size_t>(pq_m)});
        auto axis = opset8::Constant::create(element::i64, Shape{}, {0});
        auto g = std::make_shared<ov::op::v8::Gather>(pLut, pIdx, axis);
        auto red_axes = opset8::Constant::create(element::i64, Shape{1}, {1});
        auto red = std::make_shared<ov::op::v1::ReduceSum>(g, red_axes, false);
        auto model = std::make_shared<Model>(OutputVector{red}, ParameterVector{pLut, pIdx});
        return compile_on("NPU", model, r.cache_dir);
      });
      std::lock_guard<std::mutex> lk(slot.mu);
      fill_in(slot, 0, tables, static_cast<size_t>(lut_n) * sizeof(float));
      Tensor tI = slot.req.get_input_tensor(1);
      int32_t* idxp = tI.data<int32_t>();
      for (int64_t i = 0; i < nn; ++i) {
        const uint8_t* code = codes + (off + i) * pq_m;
        for (int32_t m = 0; m < pq_m; ++m) {
          idxp[static_cast<size_t>(i * pq_m + m)] = m * ks + static_cast<int32_t>(code[m]);
        }
      }
      slot.req.infer();
      read_out(slot, 0, out + off, static_cast<size_t>(nn) * sizeof(float));
    }
    return true;
  } catch (...) {
    ++r.npu_compile_fails;
    return false;
  }
#else
  (void)r;
  (void)tables;
  (void)pq_m;
  (void)ks;
  (void)codes;
  (void)ncodes;
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
