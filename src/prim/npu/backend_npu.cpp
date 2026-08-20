#include "internal.hpp"

#if defined(IOVS_WITH_OPENVINO)
#include <openvino/openvino.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/topk.hpp>
#include <openvino/opsets/opset8.hpp>
#endif

#include <filesystem>
#include <sstream>
#include <unordered_map>

namespace iovs {
namespace impl {

#if defined(IOVS_WITH_OPENVINO)
static ov::Core& ov_core() {
  static ov::Core& core = *new ov::Core();
  return core;
}

#endif

static bool ov_has_device(const char* name) {
#if defined(IOVS_WITH_OPENVINO)
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
#if defined(IOVS_WITH_OPENVINO)
  return ov_has_device("NPU");
#else
  return false;
#endif
}

bool ov_matmul(ResourcesData& r, const char* device, const float* a, const float* b, float* c,
               int64_t m, int64_t n, int64_t k, bool trans_b) {
#if defined(IOVS_WITH_OPENVINO)
  if (!ov_has_device(device)) return false;
  try {
    using namespace ov;
    std::ostringstream key;
    key << device << "_gemm_m" << m << "_n" << n << "_k" << k << "_tb" << int(trans_b);
    static auto& cache = *new std::unordered_map<std::string, CompiledModel>();
    auto it = cache.find(key.str());
    if (it == cache.end()) {
      auto pA = std::make_shared<opset8::Parameter>(element::f32, Shape{static_cast<size_t>(m), static_cast<size_t>(k)});
      auto pB = std::make_shared<opset8::Parameter>(
          element::f32, trans_b ? Shape{static_cast<size_t>(n), static_cast<size_t>(k)}
                                : Shape{static_cast<size_t>(k), static_cast<size_t>(n)});
      auto mm = std::make_shared<opset8::MatMul>(pA, pB, false, trans_b);
      auto model = std::make_shared<Model>(OutputVector{mm}, ParameterVector{pA, pB});
      std::filesystem::create_directories(r.cache_dir);
      cache[key.str()] = ov_core().compile_model(model, device);
      it = cache.find(key.str());
    }
    auto req = it->second.create_infer_request();
    const Shape sa{static_cast<size_t>(m), static_cast<size_t>(k)};
    const Shape sb = trans_b ? Shape{static_cast<size_t>(n), static_cast<size_t>(k)}
                             : Shape{static_cast<size_t>(k), static_cast<size_t>(n)};
    const Shape scs{static_cast<size_t>(m), static_cast<size_t>(n)};
    Tensor tA(element::f32, sa, const_cast<float*>(a));
    Tensor tB(element::f32, sb, const_cast<float*>(b));
    Tensor tC(element::f32, scs, c);
    req.set_input_tensor(0, tA);
    req.set_input_tensor(1, tB);
    req.set_output_tensor(tC);
    req.infer();
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

bool npu_gemm(ResourcesData& r, const float* a, const float* b, float* c, int64_t m, int64_t n,
              int64_t k, bool trans_b) {
  if (ov_matmul(r, "NPU", a, b, c, m, n, k, trans_b)) return true;
  /* HostCompile tiles: static M=256 blobs over a dynamic leading dimension. */
  constexpr int64_t kTile = 256;
  if (m <= kTile) return false;
  for (int64_t i = 0; i < m; i += kTile) {
    const int64_t rows = std::min(kTile, m - i);
    if (!ov_matmul(r, "NPU", a + i * k, b, c + i * n, rows, n, k, trans_b)) return false;
  }
  return true;
}

bool ov_topk(ResourcesData& r, const char* device, const float* scores, int64_t rows, int64_t cols,
             int64_t k, int64_t* indices, float* values, bool largest) {
#if defined(IOVS_WITH_OPENVINO)
  if (!ov_has_device(device)) return false;
  k = std::min(k, cols);
  try {
    using namespace ov;
    std::ostringstream key;
    key << device << "_topk_r" << rows << "_c" << cols << "_k" << k << "_lg" << int(largest);
    static auto& cache = *new std::unordered_map<std::string, CompiledModel>();
    auto it = cache.find(key.str());
    if (it == cache.end()) {
      auto pScores = std::make_shared<opset8::Parameter>(
          element::f32, Shape{static_cast<size_t>(rows), static_cast<size_t>(cols)});
      auto kconst = opset8::Constant::create(element::i32, Shape{}, {static_cast<int32_t>(k)});
      auto topk = std::make_shared<ov::op::v3::TopK>(pScores, kconst, /*axis*/ 1,
                                                     largest ? "max" : "min", "value", element::i32);
      auto model = std::make_shared<Model>(OutputVector{topk->output(0), topk->output(1)},
                                           ParameterVector{pScores});
      cache[key.str()] = ov_core().compile_model(model, device);
      it = cache.find(key.str());
    }
    auto req = it->second.create_infer_request();
    Tensor tS(element::f32, Shape{static_cast<size_t>(rows), static_cast<size_t>(cols)},
              const_cast<float*>(scores));
    req.set_input_tensor(tS);
    req.infer();
    Tensor tV = req.get_output_tensor(0);
    Tensor tI = req.get_output_tensor(1);
    const float* pv = tV.data<float>();
    std::memcpy(values, pv, static_cast<size_t>(rows * k) * sizeof(float));
    if (tI.get_element_type() == element::i32) {
      const int32_t* pi = tI.data<int32_t>();
      for (int64_t i = 0; i < rows * k; ++i) indices[i] = pi[i];
    } else if (tI.get_element_type() == element::i64) {
      const int64_t* pi = tI.data<int64_t>();
      std::memcpy(indices, pi, static_cast<size_t>(rows * k) * sizeof(int64_t));
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
#if defined(IOVS_WITH_OPENVINO)
  if (!ov_has_device(device)) return false;
  try {
    using namespace ov;
    std::ostringstream key;
    key << device << "_gather_sr" << src_rows << "_d" << dim << "_n" << nidx;
    static auto& cache = *new std::unordered_map<std::string, CompiledModel>();
    auto it = cache.find(key.str());
    if (it == cache.end()) {
      auto pSrc = std::make_shared<opset8::Parameter>(
          element::f32, Shape{static_cast<size_t>(src_rows), static_cast<size_t>(dim)});
      auto pIdx = std::make_shared<opset8::Parameter>(element::i32, Shape{static_cast<size_t>(nidx)});
      auto axis = opset8::Constant::create(element::i64, Shape{}, {0});
      auto g = std::make_shared<ov::op::v8::Gather>(pSrc, pIdx, axis);
      auto model = std::make_shared<Model>(OutputVector{g}, ParameterVector{pSrc, pIdx});
      cache[key.str()] = ov_core().compile_model(model, device);
      it = cache.find(key.str());
    }
    std::vector<int32_t> i32(static_cast<size_t>(nidx));
    for (int64_t i = 0; i < nidx; ++i) i32[static_cast<size_t>(i)] = static_cast<int32_t>(idx[i]);
    auto req = it->second.create_infer_request();
    Tensor tS(element::f32, Shape{static_cast<size_t>(src_rows), static_cast<size_t>(dim)},
              const_cast<float*>(src));
    Tensor tI(element::i32, Shape{static_cast<size_t>(nidx)}, i32.data());
    req.set_input_tensor(0, tS);
    req.set_input_tensor(1, tI);
    req.infer();
    Tensor tO = req.get_output_tensor();
    const size_t nbytes = static_cast<size_t>(nidx * dim) * sizeof(float);
    if (tO.get_byte_size() < nbytes) return false;
    std::memcpy(out, tO.data<float>(), nbytes);
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
  return ov_topk(r, "NPU", scores, rows, cols, k, indices, values, largest);
}

bool npu_gather_rows(ResourcesData& r, const float* src, int64_t src_rows, int64_t dim,
                     const int64_t* idx, int64_t nidx, float* out) {
  return ov_gather_rows(r, "NPU", src, src_rows, dim, idx, nidx, out);
}

}  // namespace impl
}  // namespace iovs
