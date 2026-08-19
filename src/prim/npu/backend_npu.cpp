#include "internal.hpp"

#if defined(IOVS_WITH_OPENVINO)
#include <openvino/openvino.hpp>
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
  return ov_matmul(r, "NPU", a, b, c, m, n, k, trans_b);
}

}  // namespace impl
}  // namespace iovs
