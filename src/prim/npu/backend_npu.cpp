#include "internal.hpp"

#if defined(IOVS_WITH_OPENVINO)
#include <openvino/openvino.hpp>
#include <openvino/op/abs.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/greater_eq.hpp>
#include <openvino/op/logical_xor.hpp>
#include <openvino/op/power.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/topk.hpp>
#include <openvino/op/unsqueeze.hpp>
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
#if defined(IOVS_WITH_OPENVINO)
  if (!ov_has_device("NPU")) return false;
  constexpr int64_t kTile = 128;
  try {
    using namespace ov;
    const int64_t lut_n = static_cast<int64_t>(pq_m) * ks;
    std::vector<float> lut(tables, tables + lut_n);
    for (int64_t off = 0; off < ncodes; off += kTile) {
      const int64_t nn = std::min(kTile, ncodes - off);
      std::vector<int32_t> idx(static_cast<size_t>(nn * pq_m));
      for (int64_t i = 0; i < nn; ++i) {
        const uint8_t* code = codes + (off + i) * pq_m;
        for (int32_t m = 0; m < pq_m; ++m) {
          idx[static_cast<size_t>(i * pq_m + m)] = m * ks + static_cast<int32_t>(code[m]);
        }
      }
      std::ostringstream key;
      key << "NPU_adc_m" << pq_m << "_ks" << ks << "_n" << nn;
      static auto& cache = *new std::unordered_map<std::string, CompiledModel>();
      auto it = cache.find(key.str());
      if (it == cache.end()) {
        auto pLut = std::make_shared<opset8::Parameter>(element::f32, Shape{static_cast<size_t>(lut_n)});
        auto pIdx = std::make_shared<opset8::Parameter>(element::i32, Shape{static_cast<size_t>(nn),
                                                                            static_cast<size_t>(pq_m)});
        auto axis = opset8::Constant::create(element::i64, Shape{}, {0});
        auto g = std::make_shared<ov::op::v8::Gather>(pLut, pIdx, axis);
        auto red_axes = opset8::Constant::create(element::i64, Shape{1}, {1});
        auto red = std::make_shared<ov::op::v1::ReduceSum>(g, red_axes, false);
        auto model = std::make_shared<Model>(OutputVector{red}, ParameterVector{pLut, pIdx});
        cache[key.str()] = ov_core().compile_model(model, "NPU");
        it = cache.find(key.str());
      }
      auto req = it->second.create_infer_request();
      Tensor tL(element::f32, Shape{static_cast<size_t>(lut_n)}, lut.data());
      Tensor tI(element::i32, Shape{static_cast<size_t>(nn), static_cast<size_t>(pq_m)}, idx.data());
      req.set_input_tensor(0, tL);
      req.set_input_tensor(1, tI);
      req.infer();
      Tensor tO = req.get_output_tensor();
      std::memcpy(out + off, tO.data<float>(), static_cast<size_t>(nn) * sizeof(float));
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

bool ov_pairwise_npu(ResourcesData& r, iovsMetric metric, const float* x, int64_t nx, const float* y,
                     int64_t ny, int64_t dim, float* out, float metric_arg) {
#if defined(IOVS_WITH_OPENVINO)
  if (!ov_has_device("NPU")) return false;
  if (metric != IOVS_METRIC_BITWISE_HAMMING && metric != IOVS_METRIC_LP_UNEXPANDED) return false;
  try {
    using namespace ov;
    const float p = metric_arg > 0.f ? metric_arg : 2.f;
    std::ostringstream key;
    key << "NPU_pair_m" << static_cast<int>(metric) << "_nx" << nx << "_ny" << ny << "_d" << dim << "_p" << p;
    static auto& cache = *new std::unordered_map<std::string, CompiledModel>();
    auto it = cache.find(key.str());
    if (it == cache.end()) {
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
      if (metric == IOVS_METRIC_BITWISE_HAMMING) {
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
      cache[key.str()] = ov_core().compile_model(model, "NPU");
      it = cache.find(key.str());
    }
    auto req = it->second.create_infer_request();
    Tensor tX(element::f32, Shape{static_cast<size_t>(nx), static_cast<size_t>(dim)}, const_cast<float*>(x));
    Tensor tY(element::f32, Shape{static_cast<size_t>(ny), static_cast<size_t>(dim)}, const_cast<float*>(y));
    req.set_input_tensor(0, tX);
    req.set_input_tensor(1, tY);
    req.infer();
    Tensor tO = req.get_output_tensor();
    const size_t nbytes = static_cast<size_t>(nx * ny) * sizeof(float);
    if (tO.get_byte_size() < nbytes) return false;
    std::memcpy(out, tO.data<float>(), nbytes);
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

bool npu_pairwise(ResourcesData& r, iovsMetric metric, const float* x, int64_t nx, const float* y,
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
#if defined(IOVS_WITH_OPENVINO)
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
}  // namespace iovs
