// Copyright (C) 2026 ovVS contributors
// SPDX-License-Identifier: Apache-2.0

#define NOMINMAX
#include <windows.h>
#include <psapi.h>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset8.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "level_zero_graph_abi.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using ovvs::npu_escape::l0x::GraphArgumentProperties;
using ovvs::npu_escape::l0x::GraphDdiTable;
using ovvs::npu_escape::l0x::GraphDesc2;
using ovvs::npu_escape::l0x::GraphHandle;
using ovvs::npu_escape::l0x::GraphProperties;
using ovvs::npu_escape::l0x::GraphProperties2;

constexpr std::size_t kPage = 4096;
constexpr std::size_t kPqM = 8;
constexpr std::size_t kKs = 256;
constexpr int kMaxDepth = 4;
constexpr int kMaxFixtureEpoch = 30;

class Failure : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

std::string json_escape(std::string_view s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (const unsigned char c : s) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          std::ostringstream h;
          h << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << static_cast<unsigned>(c);
          out += h.str();
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  return out;
}

std::string hex_u32(std::uint32_t v) {
  std::ostringstream out;
  out << "0x" << std::hex << std::setw(8) << std::setfill('0') << v;
  return out.str();
}

void ze_check(ze_result_t result, std::string_view operation) {
  if (result == ZE_RESULT_SUCCESS) return;
  throw Failure(std::string(operation) + " failed with " +
                hex_u32(static_cast<std::uint32_t>(result)));
}

std::size_t checked_mul(std::size_t a, std::size_t b, std::string_view what) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
    throw Failure(std::string(what) + " size overflow");
  }
  return a * b;
}

std::size_t round_page(std::size_t n) {
  if (n == 0 || n > std::numeric_limits<std::size_t>::max() - (kPage - 1)) {
    throw Failure("invalid page allocation size");
  }
  return (n + kPage - 1) & ~(kPage - 1);
}

double elapsed_ms(Clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

std::uint64_t fnv1a(const void* data, std::size_t bytes) {
  auto* p = static_cast<const std::uint8_t*>(data);
  std::uint64_t h = 1469598103934665603ull;
  for (std::size_t i = 0; i < bytes; ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}

std::string hex_u64(std::uint64_t v) {
  std::ostringstream out;
  out << "0x" << std::hex << std::setw(16) << std::setfill('0') << v;
  return out.str();
}

struct Options {
  std::size_t rows = 131072;
  int repeats = 5;
  int warmups = 1;
  std::vector<int> depths{1, 2, 4};
};

std::size_t parse_size(const char* text, std::string_view option) {
  if (!text || !*text || *text == '-') throw Failure(std::string(option) + " needs a positive integer");
  std::size_t used = 0;
  unsigned long long v = 0;
  try {
    v = std::stoull(text, &used, 10);
  } catch (...) {
    throw Failure(std::string(option) + " needs a positive integer");
  }
  if (used != std::strlen(text) || v == 0 || v > std::numeric_limits<std::size_t>::max()) {
    throw Failure(std::string(option) + " is out of range");
  }
  return static_cast<std::size_t>(v);
}

std::vector<int> parse_depths(std::string text) {
  std::vector<int> out;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t end = text.find(',', start);
    const std::string token = text.substr(start, end == std::string::npos ? end : end - start);
    const std::size_t d = parse_size(token.c_str(), "--depths");
    if (d != 1 && d != 2 && d != 4) throw Failure("--depths only accepts 1,2,4");
    out.push_back(static_cast<int>(d));
    if (end == std::string::npos) break;
    start = end + 1;
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  if (out.empty()) throw Failure("--depths cannot be empty");
  return out;
}

Options parse_options(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&](std::string_view name) -> const char* {
      if (++i >= argc) throw Failure(std::string(name) + " needs a value");
      return argv[i];
    };
    if (arg == "--rows") {
      o.rows = parse_size(value(arg), arg);
    } else if (arg == "--repeats") {
      const std::size_t v = parse_size(value(arg), arg);
      if (v > 1000) throw Failure("--repeats must be <= 1000");
      o.repeats = static_cast<int>(v);
    } else if (arg == "--warmups") {
      const std::size_t v = parse_size(value(arg), arg);
      if (v > 100) throw Failure("--warmups must be <= 100");
      o.warmups = static_cast<int>(v);
    } else if (arg == "--depths") {
      o.depths = parse_depths(value(arg));
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: ovvs_npu_l0_escape [--rows N] [--repeats N] "
                   "[--warmups N] [--depths 1,2,4]\n";
      std::exit(0);
    } else {
      throw Failure("unknown option: " + arg);
    }
  }
  if (o.rows > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw Failure("--rows exceeds the graph ABI dimension range");
  }
  if (o.warmups + o.repeats > kMaxFixtureEpoch) {
    throw Failure("--warmups plus --repeats must be <= 30 for the exact range-safe fixture");
  }
  checked_mul(checked_mul(o.rows, kPqM, "indices"), sizeof(std::int32_t), "indices");
  return o;
}

std::size_t fixture_tag(int slot, int epoch) {
  if (slot < 0 || slot >= kMaxDepth || epoch < 0 || epoch > kMaxFixtureEpoch) {
    throw Failure("fixture slot or epoch is out of range");
  }
  return static_cast<std::size_t>(epoch * kMaxDepth + slot);
}

float fixture_lut_value(std::size_t m, std::size_t code, int slot, int epoch) {
  const std::size_t tag = fixture_tag(slot, epoch);
  if (m == 0) return static_cast<float>(code * 32);
  if (m == 1) return static_cast<float>(tag * 256);
  return static_cast<float>(((m * 17 + code * 3) & 1u) * 32u);
}

std::size_t fixture_code(std::size_t row, std::size_t m, int slot, int epoch) {
  if (m == 0) return fixture_tag(slot, epoch);
  return (row * 17 + m * 31) & (kKs - 1);
}

struct Fixture {
  std::size_t rows;
  std::vector<float> lut;
  std::vector<std::int32_t> indices;
  std::vector<float> oracle;
  std::shared_ptr<ov::Model> model;
};

Fixture make_fixture(std::size_t rows) {
  Fixture f;
  f.rows = rows;
  f.lut.resize(kPqM * kKs);
  for (std::size_t m = 0; m < kPqM; ++m) {
    for (std::size_t code = 0; code < kKs; ++code) {
      f.lut[m * kKs + code] = fixture_lut_value(m, code, 0, 0);
    }
  }
  f.indices.resize(checked_mul(rows, kPqM, "fixture indices"));
  f.oracle.assign(rows, 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    float sum = 0.0f;
    for (std::size_t m = 0; m < kPqM; ++m) {
      const std::size_t code = fixture_code(row, m, 0, 0);
      const std::int32_t idx = static_cast<std::int32_t>(m * kKs + code);
      f.indices[row * kPqM + m] = idx;
      sum += f.lut[static_cast<std::size_t>(idx)];
    }
    f.oracle[row] = sum;
  }

  auto lut = std::make_shared<ov::opset8::Parameter>(ov::element::f32,
                                                     ov::Shape{kPqM * kKs});
  auto indices = std::make_shared<ov::opset8::Parameter>(
      ov::element::i32, ov::Shape{rows, kPqM});
  lut->set_friendly_name("lut");
  indices->set_friendly_name("indices");
  lut->output(0).get_tensor().set_names({"lut"});
  indices->output(0).get_tensor().set_names({"indices"});
  auto axis = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {0});
  auto gathered = std::make_shared<ov::op::v8::Gather>(lut, indices, axis);
  auto reduce_axes = ov::opset8::Constant::create(ov::element::i64, ov::Shape{1}, {1});
  auto scores = std::make_shared<ov::op::v1::ReduceSum>(gathered, reduce_axes, false);
  scores->set_friendly_name("scores");
  scores->output(0).get_tensor().set_names({"scores"});
  f.model = std::make_shared<ov::Model>(ov::OutputVector{scores},
                                        ov::ParameterVector{lut, indices}, "ovvs_pq_adc_escape");
  return f;
}

void fill_slot_inputs(float* lut, std::int32_t* indices, const Fixture& f, int slot, int epoch) {
  if (!lut || !indices) throw Failure("cannot fill a null input buffer");
  for (std::size_t m = 0; m < kPqM; ++m) {
    for (std::size_t code = 0; code < kKs; ++code) {
      lut[m * kKs + code] = fixture_lut_value(m, code, slot, epoch);
    }
  }
  for (std::size_t row = 0; row < f.rows; ++row) {
    for (std::size_t m = 0; m < kPqM; ++m) {
      indices[row * kPqM + m] =
          static_cast<std::int32_t>(m * kKs + fixture_code(row, m, slot, epoch));
    }
  }
}

void poison_output(float* output, const Fixture& f) {
  if (!output) throw Failure("cannot poison a null output buffer");
  std::fill_n(output, f.rows, std::numeric_limits<float>::quiet_NaN());
}

float oracle_value(std::size_t row, int slot, int epoch) {
  float sum = 0.0f;
  for (std::size_t m = 0; m < kPqM; ++m) {
    const std::size_t code = fixture_code(row, m, slot, epoch);
    sum += fixture_lut_value(m, code, slot, epoch);
  }
  return sum;
}

void validate_output(const float* actual, const Fixture& f, std::string_view lane, int slot,
                     int epoch) {
  if (!actual) throw Failure(std::string(lane) + " returned a null output pointer");
  for (std::size_t row = 0; row < f.rows; ++row) {
    const float expected = oracle_value(row, slot, epoch);
    if (!std::isfinite(actual[row]) || actual[row] != expected) {
      std::ostringstream msg;
      msg << lane << " correctness failure in slot " << slot << " row " << row
          << " epoch " << epoch << ": got " << actual[row] << ", expected " << expected;
      throw Failure(msg.str());
    }
  }
}

struct Samples {
  std::vector<double> wall_ms;
};

double percentile(const std::vector<double>& samples, double p) {
  if (samples.empty()) throw Failure("cannot summarize empty samples");
  std::vector<double> sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  const std::size_t rank = static_cast<std::size_t>(std::ceil(p * sorted.size()));
  return sorted[std::max<std::size_t>(1, rank) - 1];
}

struct DepthResult {
  int depth = 0;
  Samples samples;
};

struct OpenVinoPool {
  const Fixture& fixture;
  ov::CompiledModel compiled;
  std::vector<ov::InferRequest> requests;
  std::size_t tensor_payload_bytes = 0;
  double request_setup_ms = 0.0;

  OpenVinoPool(const Fixture& fixture_in, ov::CompiledModel compiled_in, int max_depth)
      : fixture(fixture_in), compiled(std::move(compiled_in)) {
    const auto start = Clock::now();
    requests.reserve(static_cast<std::size_t>(max_depth));
    for (int slot = 0; slot < max_depth; ++slot) {
      auto request = compiled.create_infer_request();
      ov::Tensor lut = request.get_input_tensor(0);
      ov::Tensor indices = request.get_input_tensor(1);
      ov::Tensor scores = request.get_output_tensor(0);
      if (lut.get_element_type() != ov::element::f32 || lut.get_shape() != ov::Shape{kPqM * kKs} ||
          indices.get_element_type() != ov::element::i32 ||
          indices.get_shape() != ov::Shape{fixture.rows, kPqM} ||
          scores.get_element_type() != ov::element::f32 ||
          scores.get_shape() != ov::Shape{fixture.rows}) {
        throw Failure("OpenVINO compiled-model tensor contract changed");
      }
      fill_slot_inputs(lut.data<float>(), indices.data<std::int32_t>(), fixture, slot, 0);
      poison_output(scores.data<float>(), fixture);
      tensor_payload_bytes += lut.get_byte_size() + indices.get_byte_size() + scores.get_byte_size();
      requests.emplace_back(std::move(request));
    }
    request_setup_ms = elapsed_ms(start);
  }

  double run_once(int depth) {
    poison(depth);
    const auto start = Clock::now();
    for (int slot = 0; slot < depth; ++slot) requests[static_cast<std::size_t>(slot)].start_async();
    for (int slot = 0; slot < depth; ++slot) requests[static_cast<std::size_t>(slot)].wait();
    return elapsed_ms(start);
  }

  void validate(int depth, int epoch) {
    for (int slot = 0; slot < depth; ++slot) {
      ov::Tensor scores = requests[static_cast<std::size_t>(slot)].get_output_tensor(0);
      validate_output(scores.data<float>(), fixture, "OpenVINO", slot, epoch);
    }
  }

  void prepare(int depth, int epoch) {
    for (int slot = 0; slot < depth; ++slot) {
      ov::Tensor lut = requests[static_cast<std::size_t>(slot)].get_input_tensor(0);
      ov::Tensor indices = requests[static_cast<std::size_t>(slot)].get_input_tensor(1);
      fill_slot_inputs(lut.data<float>(), indices.data<std::int32_t>(), fixture, slot, epoch);
    }
  }

  void poison(int depth) {
    for (int slot = 0; slot < depth; ++slot) {
      ov::Tensor scores = requests[static_cast<std::size_t>(slot)].get_output_tensor(0);
      poison_output(scores.data<float>(), fixture);
    }
  }

  double run_end_to_end(int depth, int epoch) {
    poison(depth);
    const auto start = Clock::now();
    prepare(depth, epoch);
    for (int slot = 0; slot < depth; ++slot) requests[static_cast<std::size_t>(slot)].start_async();
    for (int slot = 0; slot < depth; ++slot) requests[static_cast<std::size_t>(slot)].wait();
    validate(depth, epoch);
    return elapsed_ms(start);
  }
};

class L0Api {
 public:
  decltype(&zeInit) init = nullptr;
  decltype(&zeDriverGet) driver_get = nullptr;
  decltype(&zeDriverGetExtensionProperties) driver_get_extension_properties = nullptr;
  decltype(&zeDriverGetExtensionFunctionAddress) driver_get_extension_function_address = nullptr;
  decltype(&zeDeviceGet) device_get = nullptr;
  decltype(&zeDeviceGetProperties) device_get_properties = nullptr;
  decltype(&zeDeviceGetCommandQueueGroupProperties) device_get_queue_groups = nullptr;
  decltype(&zeContextCreate) context_create = nullptr;
  decltype(&zeContextDestroy) context_destroy = nullptr;
  decltype(&zeMemAllocHost) mem_alloc_host = nullptr;
  decltype(&zeMemFree) mem_free = nullptr;
  decltype(&zeCommandQueueCreate) queue_create = nullptr;
  decltype(&zeCommandQueueDestroy) queue_destroy = nullptr;
  decltype(&zeCommandQueueExecuteCommandLists) queue_execute = nullptr;
  decltype(&zeCommandQueueSynchronize) queue_synchronize = nullptr;
  decltype(&zeCommandListCreate) list_create = nullptr;
  decltype(&zeCommandListDestroy) list_destroy = nullptr;
  decltype(&zeCommandListClose) list_close = nullptr;
  decltype(&zeFenceCreate) fence_create = nullptr;
  decltype(&zeFenceDestroy) fence_destroy = nullptr;
  decltype(&zeFenceHostSynchronize) fence_synchronize = nullptr;
  decltype(&zeFenceReset) fence_reset = nullptr;
  decltype(&zeEventPoolCreate) event_pool_create = nullptr;
  decltype(&zeEventPoolDestroy) event_pool_destroy = nullptr;
  decltype(&zeEventCreate) event_create = nullptr;
  decltype(&zeEventDestroy) event_destroy = nullptr;
  decltype(&zeEventQueryStatus) event_query = nullptr;
  decltype(&zeEventHostReset) event_reset = nullptr;

  L0Api() {
    module_ = LoadLibraryW(L"ze_loader.dll");
    if (!module_) throw Failure("ze_loader.dll is not available");
    init = load<decltype(init)>("zeInit");
    driver_get = load<decltype(driver_get)>("zeDriverGet");
    driver_get_extension_properties =
        load<decltype(driver_get_extension_properties)>("zeDriverGetExtensionProperties");
    driver_get_extension_function_address =
        load<decltype(driver_get_extension_function_address)>("zeDriverGetExtensionFunctionAddress");
    device_get = load<decltype(device_get)>("zeDeviceGet");
    device_get_properties = load<decltype(device_get_properties)>("zeDeviceGetProperties");
    device_get_queue_groups =
        load<decltype(device_get_queue_groups)>("zeDeviceGetCommandQueueGroupProperties");
    context_create = load<decltype(context_create)>("zeContextCreate");
    context_destroy = load<decltype(context_destroy)>("zeContextDestroy");
    mem_alloc_host = load<decltype(mem_alloc_host)>("zeMemAllocHost");
    mem_free = load<decltype(mem_free)>("zeMemFree");
    queue_create = load<decltype(queue_create)>("zeCommandQueueCreate");
    queue_destroy = load<decltype(queue_destroy)>("zeCommandQueueDestroy");
    queue_execute = load<decltype(queue_execute)>("zeCommandQueueExecuteCommandLists");
    queue_synchronize = load<decltype(queue_synchronize)>("zeCommandQueueSynchronize");
    list_create = load<decltype(list_create)>("zeCommandListCreate");
    list_destroy = load<decltype(list_destroy)>("zeCommandListDestroy");
    list_close = load<decltype(list_close)>("zeCommandListClose");
    fence_create = load<decltype(fence_create)>("zeFenceCreate");
    fence_destroy = load<decltype(fence_destroy)>("zeFenceDestroy");
    fence_synchronize = load<decltype(fence_synchronize)>("zeFenceHostSynchronize");
    fence_reset = load<decltype(fence_reset)>("zeFenceReset");
    event_pool_create = load<decltype(event_pool_create)>("zeEventPoolCreate");
    event_pool_destroy = load<decltype(event_pool_destroy)>("zeEventPoolDestroy");
    event_create = load<decltype(event_create)>("zeEventCreate");
    event_destroy = load<decltype(event_destroy)>("zeEventDestroy");
    event_query = load<decltype(event_query)>("zeEventQueryStatus");
    event_reset = load<decltype(event_reset)>("zeEventHostReset");
  }

  ~L0Api() {
    if (module_) FreeLibrary(module_);
  }

  L0Api(const L0Api&) = delete;
  L0Api& operator=(const L0Api&) = delete;

  void retain_for_process_teardown() noexcept { module_ = nullptr; }

 private:
  template <typename T>
  T load(const char* name) {
    auto address = GetProcAddress(module_, name);
    if (!address) throw Failure(std::string("ze_loader.dll is missing ") + name);
    return reinterpret_cast<T>(address);
  }

  HMODULE module_ = nullptr;
};

struct HostAllocation {
  void* pointer = nullptr;
  std::size_t payload_bytes = 0;
  std::size_t allocated_bytes = 0;
};

struct DirectSlot {
  HostAllocation lut;
  HostAllocation indices;
  HostAllocation scores;
  ze_command_list_handle_t list = nullptr;
  ze_event_handle_t event = nullptr;
};

class DirectGraphPool {
 public:
  DirectGraphPool(const Fixture& fixture_in, const std::vector<std::uint8_t>& blob,
                  const std::vector<int>& depths)
      : fixture_(fixture_in), blob_source_(blob), depths_(depths) {
    const auto start = Clock::now();
    try {
      initialize();
      setup_ms = elapsed_ms(start);
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~DirectGraphPool() { cleanup(); }
  DirectGraphPool(const DirectGraphPool&) = delete;
  DirectGraphPool& operator=(const DirectGraphPool&) = delete;

  double run_once(int depth) {
    if (depth <= 0 || depth > max_depth_) throw Failure("invalid direct Level Zero depth");
    poison(depth);
    ze_fence_handle_t fence = fences_.at(static_cast<std::size_t>(depth));
    if (!fence) throw Failure("direct Level Zero fence was not created for requested depth");
    std::array<ze_command_list_handle_t, kMaxDepth> lists{};
    for (int slot = 0; slot < depth; ++slot) lists[static_cast<std::size_t>(slot)] = slots_[slot].list;
    const auto start = Clock::now();
    ze_check(api_.queue_execute(queue_, static_cast<std::uint32_t>(depth), lists.data(), fence),
             "zeCommandQueueExecuteCommandLists");
    ze_check(api_.fence_synchronize(fence, std::numeric_limits<std::uint64_t>::max()),
             "zeFenceHostSynchronize");
    for (int slot = 0; slot < depth; ++slot) {
      ze_check(api_.event_query(slots_[static_cast<std::size_t>(slot)].event),
               "zeEventQueryStatus");
      ze_check(api_.event_reset(slots_[static_cast<std::size_t>(slot)].event),
               "zeEventHostReset");
    }
    ze_check(api_.fence_reset(fence), "zeFenceReset");
    return elapsed_ms(start);
  }

  void validate(int depth, int epoch) const {
    for (int slot = 0; slot < depth; ++slot) {
      validate_output(static_cast<const float*>(slots_[slot].scores.pointer), fixture_,
                      "direct Level Zero", slot, epoch);
    }
  }

  void prepare(int depth, int epoch) {
    if (depth <= 0 || depth > max_depth_) throw Failure("invalid direct Level Zero depth");
    for (int slot = 0; slot < depth; ++slot) {
      fill_slot_inputs(static_cast<float*>(slots_[slot].lut.pointer),
                       static_cast<std::int32_t*>(slots_[slot].indices.pointer), fixture_, slot,
                       epoch);
    }
  }

  void poison(int depth) {
    if (depth <= 0 || depth > max_depth_) throw Failure("invalid direct Level Zero depth");
    for (int slot = 0; slot < depth; ++slot) {
      poison_output(static_cast<float*>(slots_[static_cast<std::size_t>(slot)].scores.pointer),
                    fixture_);
    }
  }

  double run_end_to_end(int depth, int epoch) {
    poison(depth);
    const auto start = Clock::now();
    prepare(depth, epoch);
    ze_fence_handle_t fence = fences_.at(static_cast<std::size_t>(depth));
    std::array<ze_command_list_handle_t, kMaxDepth> lists{};
    for (int slot = 0; slot < depth; ++slot) lists[static_cast<std::size_t>(slot)] = slots_[slot].list;
    ze_check(api_.queue_execute(queue_, static_cast<std::uint32_t>(depth), lists.data(), fence),
             "zeCommandQueueExecuteCommandLists(end-to-end)");
    ze_check(api_.fence_synchronize(fence, std::numeric_limits<std::uint64_t>::max()),
             "zeFenceHostSynchronize(end-to-end)");
    for (int slot = 0; slot < depth; ++slot) {
      ze_check(api_.event_query(slots_[static_cast<std::size_t>(slot)].event),
               "zeEventQueryStatus(end-to-end)");
      ze_check(api_.event_reset(slots_[static_cast<std::size_t>(slot)].event),
               "zeEventHostReset(end-to-end)");
    }
    ze_check(api_.fence_reset(fence), "zeFenceReset(end-to-end)");
    validate(depth, epoch);
    return elapsed_ms(start);
  }

  std::string device_name;
  std::string device_uuid;
  std::uint32_t extension_version = 0;
  std::uint32_t queue_ordinal = 0;
  std::uint32_t queue_count = 0;
  std::size_t owned_host_bytes = 0;
  std::size_t blob_payload_bytes = 0;
  std::size_t blob_allocated_bytes = 0;
  bool persistent_blob = false;
  double setup_ms = 0.0;

 private:
  HostAllocation allocate(std::size_t payload) {
    HostAllocation allocation;
    allocation.payload_bytes = payload;
    allocation.allocated_bytes = round_page(payload);
    ze_host_mem_alloc_desc_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    ze_check(api_.mem_alloc_host(context_, &desc, allocation.allocated_bytes, kPage,
                                 &allocation.pointer),
             "zeMemAllocHost");
    if (!allocation.pointer || reinterpret_cast<std::uintptr_t>(allocation.pointer) % kPage != 0) {
      if (allocation.pointer) api_.mem_free(context_, allocation.pointer);
      throw Failure("zeMemAllocHost did not honor 4096-byte alignment");
    }
    std::memset(allocation.pointer, 0, allocation.allocated_bytes);
    owned_host_bytes += allocation.allocated_bytes;
    return allocation;
  }

  static bool dims_match(const GraphArgumentProperties& p,
                         std::initializer_list<std::uint32_t> expected) {
    std::size_t i = 0;
    for (const std::uint32_t dim : expected) {
      if (i >= ovvs::npu_escape::l0x::kMaxGraphArgumentDimensions || p.dims[i] != dim) return false;
      ++i;
    }
    for (; i < ovvs::npu_escape::l0x::kMaxGraphArgumentDimensions; ++i) {
      if (p.dims[i] != 0 && p.dims[i] != 1) return false;
    }
    return true;
  }

  void find_device() {
    ze_check(api_.init(0), "zeInit");
    std::uint32_t driver_count = 0;
    ze_check(api_.driver_get(&driver_count, nullptr), "zeDriverGet(count)");
    if (driver_count == 0 || driver_count > 32) throw Failure("no bounded Level Zero driver set");
    std::vector<ze_driver_handle_t> drivers(driver_count);
    ze_check(api_.driver_get(&driver_count, drivers.data()), "zeDriverGet");

    for (const ze_driver_handle_t driver : drivers) {
      std::uint32_t ext_count = 0;
      if (api_.driver_get_extension_properties(driver, &ext_count, nullptr) != ZE_RESULT_SUCCESS ||
          ext_count == 0 || ext_count > 4096) {
        continue;
      }
      std::vector<ze_driver_extension_properties_t> extensions(ext_count);
      if (api_.driver_get_extension_properties(driver, &ext_count, extensions.data()) !=
          ZE_RESULT_SUCCESS) {
        continue;
      }
      std::optional<std::uint32_t> graph_version;
      for (const auto& ext : extensions) {
        if (std::string_view(ext.name) == ovvs::npu_escape::l0x::kGraphExtensionName) {
          graph_version = ext.version;
          break;
        }
      }
      if (!graph_version || ZE_MAJOR_VERSION(*graph_version) != 1 ||
          ZE_MINOR_VERSION(*graph_version) < 13) {
        continue;
      }

      std::uint32_t device_count = 0;
      if (api_.device_get(driver, &device_count, nullptr) != ZE_RESULT_SUCCESS || device_count == 0 ||
          device_count > 64) {
        continue;
      }
      std::vector<ze_device_handle_t> devices(device_count);
      if (api_.device_get(driver, &device_count, devices.data()) != ZE_RESULT_SUCCESS) continue;
      for (const ze_device_handle_t device : devices) {
        ze_device_properties_t properties{};
        properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        if (api_.device_get_properties(device, &properties) != ZE_RESULT_SUCCESS ||
            properties.type != ZE_DEVICE_TYPE_VPU) {
          continue;
        }
        void* table = nullptr;
        ze_check(api_.driver_get_extension_function_address(
                     driver, ovvs::npu_escape::l0x::kGraphExtensionName, &table),
                 "zeDriverGetExtensionFunctionAddress(ZE_extension_graph)");
        if (!table) throw Failure("graph extension returned a null DDI table");
        driver_ = driver;
        device_ = device;
        graph_ddi_ = static_cast<GraphDdiTable*>(table);
        extension_version = *graph_version;
        device_name = properties.name;
        std::ostringstream uuid;
        for (const std::uint8_t byte : properties.uuid.id) {
          uuid << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(byte);
        }
        device_uuid = uuid.str();
        return;
      }
    }
    throw Failure("no VPU device with ZE_extension_graph >= 1.13 was found");
  }

  void validate_ddi() const {
    if (!graph_ddi_->pfnDestroy || !graph_ddi_->pfnGetProperties ||
        !graph_ddi_->pfnGetArgumentProperties || !graph_ddi_->pfnSetArgumentValue ||
        !graph_ddi_->pfnAppendGraphInitialize || !graph_ddi_->pfnAppendGraphExecute ||
        !graph_ddi_->pfnCreate2 || !graph_ddi_->pfnGetProperties2 ||
        !graph_ddi_->pfnGraphInitialize) {
      throw Failure("graph extension DDI is missing a required function through version 1.8");
    }
  }

  void find_queue_group() {
    std::uint32_t count = 0;
    ze_check(api_.device_get_queue_groups(device_, &count, nullptr),
             "zeDeviceGetCommandQueueGroupProperties(count)");
    if (count == 0 || count > 256) throw Failure("no bounded Level Zero queue-group set");
    std::vector<ze_command_queue_group_properties_t> groups(count);
    for (auto& group : groups) group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    ze_check(api_.device_get_queue_groups(device_, &count, groups.data()),
             "zeDeviceGetCommandQueueGroupProperties");
    for (std::uint32_t ordinal = 0; ordinal < count; ++ordinal) {
      if ((groups[ordinal].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) != 0 &&
          groups[ordinal].numQueues > 0) {
        queue_ordinal = ordinal;
        queue_count = groups[ordinal].numQueues;
        return;
      }
    }
    throw Failure("NPU exposes no compute command queue group");
  }

  void create_queue() {
    ze_command_queue_desc_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    desc.ordinal = queue_ordinal;
    // The public core API requires index zero; numQueues is reported as
    // capability evidence, not used as permission to select another index.
    desc.index = 0;
    desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ze_check(api_.queue_create(context_, device_, &desc, &queue_), "zeCommandQueueCreate");
  }

  void initialize_graph() {
    GraphProperties2 properties{};
    properties.stype = ovvs::npu_escape::l0x::kStructureTypeGraphProperties2;
    ze_check(graph_ddi_->pfnGetProperties2(graph_, &properties), "pfnGetProperties2");
    const std::uint32_t known = ovvs::npu_escape::l0x::kGraphStageCommandListInitialize |
                                ovvs::npu_escape::l0x::kGraphStageInitialize;
    if ((properties.initStageRequired & ~known) != 0) {
      throw Failure("graph reports an unknown initialization stage bit");
    }
    if ((properties.initStageRequired & ovvs::npu_escape::l0x::kGraphStageInitialize) != 0) {
      ze_check(graph_ddi_->pfnGraphInitialize(graph_), "pfnGraphInitialize");
    }
    if ((properties.initStageRequired &
         ovvs::npu_escape::l0x::kGraphStageCommandListInitialize) != 0) {
      ze_command_list_handle_t list = nullptr;
      ze_fence_handle_t fence = nullptr;
      ze_command_list_desc_t list_desc{};
      list_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
      list_desc.commandQueueGroupOrdinal = queue_ordinal;
      ze_check(api_.list_create(context_, device_, &list_desc, &list),
               "zeCommandListCreate(graph initialize)");
      try {
        ze_fence_desc_t fence_desc{};
        fence_desc.stype = ZE_STRUCTURE_TYPE_FENCE_DESC;
        ze_check(api_.fence_create(queue_, &fence_desc, &fence),
                 "zeFenceCreate(graph initialize)");
        ze_check(graph_ddi_->pfnAppendGraphInitialize(list, graph_, nullptr, 0, nullptr),
                 "pfnAppendGraphInitialize");
        ze_check(api_.list_close(list), "zeCommandListClose(graph initialize)");
        ze_check(api_.queue_execute(queue_, 1, &list, fence),
                 "zeCommandQueueExecuteCommandLists(graph initialize)");
        ze_check(api_.fence_synchronize(fence, std::numeric_limits<std::uint64_t>::max()),
                 "zeFenceHostSynchronize(graph initialize)");
      } catch (...) {
        if (fence) api_.fence_destroy(fence);
        if (list) api_.list_destroy(list);
        throw;
      }
      api_.fence_destroy(fence);
      api_.list_destroy(list);
    }
  }

  void discover_arguments() {
    GraphProperties properties{};
    properties.stype = ovvs::npu_escape::l0x::kStructureTypeGraphProperties;
    ze_check(graph_ddi_->pfnGetProperties(graph_, &properties), "pfnGetProperties");
    if (properties.numGraphArgs != 3) {
      throw Failure("native graph must expose exactly two inputs and one output");
    }
    bool found_lut = false;
    bool found_indices = false;
    bool found_scores = false;
    for (std::uint32_t i = 0; i < properties.numGraphArgs; ++i) {
      GraphArgumentProperties p{};
      p.stype = ovvs::npu_escape::l0x::kStructureTypeGraphArgumentProperties;
      ze_check(graph_ddi_->pfnGetArgumentProperties(graph_, i, &p),
               "pfnGetArgumentProperties");
      const bool fp32 = p.networkPrecision == ovvs::npu_escape::l0x::kGraphArgumentPrecisionFp32 &&
                        p.devicePrecision == ovvs::npu_escape::l0x::kGraphArgumentPrecisionFp32;
      const bool int32 = p.networkPrecision == ovvs::npu_escape::l0x::kGraphArgumentPrecisionInt32 &&
                         p.devicePrecision == ovvs::npu_escape::l0x::kGraphArgumentPrecisionInt32;
      if (p.type == ovvs::npu_escape::l0x::kGraphArgumentTypeInput && fp32 &&
          dims_match(p, {static_cast<std::uint32_t>(kPqM * kKs)})) {
        if (found_lut) throw Failure("ambiguous native graph LUT argument");
        lut_arg_ = i;
        found_lut = true;
      } else if (p.type == ovvs::npu_escape::l0x::kGraphArgumentTypeInput && int32 &&
                 dims_match(p, {static_cast<std::uint32_t>(fixture_.rows),
                                static_cast<std::uint32_t>(kPqM)})) {
        if (found_indices) throw Failure("ambiguous native graph indices argument");
        indices_arg_ = i;
        found_indices = true;
      } else if (p.type == ovvs::npu_escape::l0x::kGraphArgumentTypeOutput && fp32 &&
                 dims_match(p, {static_cast<std::uint32_t>(fixture_.rows)})) {
        if (found_scores) throw Failure("ambiguous native graph output argument");
        scores_arg_ = i;
        found_scores = true;
      } else {
        std::ostringstream msg;
        msg << "unsupported native graph argument " << i << " name=" << p.name
            << " type=" << p.type << " networkPrecision=" << p.networkPrecision
            << " devicePrecision=" << p.devicePrecision;
        throw Failure(msg.str());
      }
    }
    if (!found_lut || !found_indices || !found_scores) {
      throw Failure("native graph argument contract is incomplete");
    }
  }

  void create_slots() {
    slots_.resize(static_cast<std::size_t>(max_depth_));
    ze_event_pool_desc_t pool_desc{};
    pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    pool_desc.count = static_cast<std::uint32_t>(max_depth_);
    ze_check(api_.event_pool_create(context_, &pool_desc, 1, &device_, &event_pool_),
             "zeEventPoolCreate");
    const std::size_t lut_bytes = fixture_.lut.size() * sizeof(float);
    const std::size_t indices_bytes = fixture_.indices.size() * sizeof(std::int32_t);
    const std::size_t scores_bytes = fixture_.oracle.size() * sizeof(float);
    for (int index = 0; index < max_depth_; ++index) {
      DirectSlot& slot = slots_[static_cast<std::size_t>(index)];
      slot.lut = allocate(lut_bytes);
      slot.indices = allocate(indices_bytes);
      slot.scores = allocate(scores_bytes);
      fill_slot_inputs(static_cast<float*>(slot.lut.pointer),
                       static_cast<std::int32_t*>(slot.indices.pointer), fixture_, index, 0);

      ze_event_desc_t event_desc{};
      event_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
      event_desc.index = static_cast<std::uint32_t>(index);
      // The NPU driver accepts the OpenVINO plugin's zero-scope event
      // contract; HOST scope is rejected as INVALID_ENUMERATION on this stack.
      event_desc.signal = 0;
      event_desc.wait = 0;
      ze_check(api_.event_create(event_pool_, &event_desc, &slot.event), "zeEventCreate");

      ze_check(graph_ddi_->pfnSetArgumentValue(graph_, lut_arg_, slot.lut.pointer),
               "pfnSetArgumentValue(lut)");
      ze_check(graph_ddi_->pfnSetArgumentValue(graph_, indices_arg_, slot.indices.pointer),
               "pfnSetArgumentValue(indices)");
      ze_check(graph_ddi_->pfnSetArgumentValue(graph_, scores_arg_, slot.scores.pointer),
               "pfnSetArgumentValue(scores)");

      ze_command_list_desc_t desc{};
      desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
      desc.commandQueueGroupOrdinal = queue_ordinal;
      ze_check(api_.list_create(context_, device_, &desc, &slot.list), "zeCommandListCreate");
      ze_check(graph_ddi_->pfnAppendGraphExecute(slot.list, graph_, nullptr, slot.event, 0, nullptr),
               "pfnAppendGraphExecute");
      ze_check(api_.list_close(slot.list), "zeCommandListClose");
    }
    fences_.resize(static_cast<std::size_t>(max_depth_ + 1), nullptr);
    for (const int depth : depths_) {
      ze_fence_desc_t desc{};
      desc.stype = ZE_STRUCTURE_TYPE_FENCE_DESC;
      ze_check(api_.fence_create(queue_, &desc, &fences_[static_cast<std::size_t>(depth)]),
               "zeFenceCreate");
    }
  }

  void initialize() {
    if (blob_source_.empty()) throw Failure("OpenVINO exported an empty native blob");
    max_depth_ = *std::max_element(depths_.begin(), depths_.end());
    find_device();
    validate_ddi();
    find_queue_group();

    ze_context_desc_t context_desc{};
    context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    ze_check(api_.context_create(driver_, &context_desc, &context_), "zeContextCreate");
    create_queue();

    blob_payload_bytes = blob_source_.size();
    blob_ = allocate(blob_payload_bytes);
    blob_allocated_bytes = blob_.allocated_bytes;
    std::memcpy(blob_.pointer, blob_source_.data(), blob_source_.size());
    persistent_blob = blob_payload_bytes % kPage == 0;

    GraphDesc2 desc{};
    desc.stype = ovvs::npu_escape::l0x::kStructureTypeGraphDesc2;
    desc.format = ovvs::npu_escape::l0x::kGraphFormatNative;
    desc.inputSize = blob_payload_bytes;
    desc.pInput = static_cast<const std::uint8_t*>(blob_.pointer);
    desc.flags = persistent_blob ? ovvs::npu_escape::l0x::kGraphFlagInputGraphPersistent : 0;
    ze_check(graph_ddi_->pfnCreate2(context_, device_, &desc, &graph_), "pfnCreate2(native)");
    if (!graph_) throw Failure("pfnCreate2 returned a null graph");

    initialize_graph();
    discover_arguments();
    create_slots();
  }

  void free_allocation(HostAllocation& allocation) noexcept {
    if (allocation.pointer && context_) api_.mem_free(context_, allocation.pointer);
    allocation = {};
  }

  void cleanup() noexcept {
    if (queue_ &&
        api_.queue_synchronize(queue_, std::numeric_limits<std::uint64_t>::max()) !=
            ZE_RESULT_SUCCESS) {
      // Referenced buffers and handles must outlive any work whose completion is
      // unknown. The process owns this experiment, so retain them for OS teardown.
      api_.retain_for_process_teardown();
      return;
    }
    for (auto& fence : fences_) {
      if (fence) api_.fence_destroy(fence);
      fence = nullptr;
    }
    for (auto& slot : slots_) {
      if (slot.list) api_.list_destroy(slot.list);
      slot.list = nullptr;
    }
    for (auto& slot : slots_) {
      if (slot.event) api_.event_destroy(slot.event);
      slot.event = nullptr;
    }
    if (event_pool_) api_.event_pool_destroy(event_pool_);
    event_pool_ = nullptr;
    if (graph_ && graph_ddi_ && graph_ddi_->pfnDestroy) graph_ddi_->pfnDestroy(graph_);
    graph_ = nullptr;
    for (auto& slot : slots_) {
      free_allocation(slot.scores);
      free_allocation(slot.indices);
      free_allocation(slot.lut);
    }
    free_allocation(blob_);
    if (queue_) api_.queue_destroy(queue_);
    queue_ = nullptr;
    if (context_) api_.context_destroy(context_);
    context_ = nullptr;
  }

  const Fixture& fixture_;
  const std::vector<std::uint8_t>& blob_source_;
  std::vector<int> depths_;
  L0Api api_;
  ze_driver_handle_t driver_ = nullptr;
  ze_device_handle_t device_ = nullptr;
  ze_context_handle_t context_ = nullptr;
  ze_command_queue_handle_t queue_ = nullptr;
  GraphDdiTable* graph_ddi_ = nullptr;
  GraphHandle graph_ = nullptr;
  HostAllocation blob_;
  ze_event_pool_handle_t event_pool_ = nullptr;
  std::vector<DirectSlot> slots_;
  std::vector<ze_fence_handle_t> fences_;
  std::uint32_t lut_arg_ = 0;
  std::uint32_t indices_arg_ = 0;
  std::uint32_t scores_arg_ = 0;
  int max_depth_ = 0;
};

struct ScopeResults {
  std::vector<DepthResult> prefilled;
  std::vector<DepthResult> refill_consume;
};

ScopeResults measure_openvino(OpenVinoPool& pool, const Options& options) {
  ScopeResults results;
  for (const int depth : options.depths) {
    pool.prepare(depth, 0);
    for (int i = 0; i < options.warmups; ++i) {
      pool.run_once(depth);
      pool.validate(depth, 0);
    }
    DepthResult result;
    result.depth = depth;
    result.samples.wall_ms.reserve(static_cast<std::size_t>(options.repeats));
    for (int i = 0; i < options.repeats; ++i) {
      result.samples.wall_ms.push_back(pool.run_once(depth));
      pool.validate(depth, 0);
    }
    results.prefilled.push_back(std::move(result));

    for (int i = 0; i < options.warmups; ++i) pool.run_end_to_end(depth, i + 1);
    DepthResult e2e;
    e2e.depth = depth;
    e2e.samples.wall_ms.reserve(static_cast<std::size_t>(options.repeats));
    for (int i = 0; i < options.repeats; ++i) {
      e2e.samples.wall_ms.push_back(pool.run_end_to_end(depth, i + options.warmups + 1));
    }
    results.refill_consume.push_back(std::move(e2e));
  }
  return results;
}

ScopeResults measure_direct(DirectGraphPool& pool, const Options& options) {
  ScopeResults results;
  for (const int depth : options.depths) {
    pool.prepare(depth, 0);
    for (int i = 0; i < options.warmups; ++i) {
      pool.run_once(depth);
      pool.validate(depth, 0);
    }
    DepthResult result;
    result.depth = depth;
    result.samples.wall_ms.reserve(static_cast<std::size_t>(options.repeats));
    for (int i = 0; i < options.repeats; ++i) {
      result.samples.wall_ms.push_back(pool.run_once(depth));
      pool.validate(depth, 0);
    }
    results.prefilled.push_back(std::move(result));

    for (int i = 0; i < options.warmups; ++i) pool.run_end_to_end(depth, i + 1);
    DepthResult e2e;
    e2e.depth = depth;
    e2e.samples.wall_ms.reserve(static_cast<std::size_t>(options.repeats));
    for (int i = 0; i < options.repeats; ++i) {
      e2e.samples.wall_ms.push_back(pool.run_end_to_end(depth, i + options.warmups + 1));
    }
    results.refill_consume.push_back(std::move(e2e));
  }
  return results;
}

void write_samples(std::ostream& out, const std::vector<DepthResult>& results,
                   std::size_t rows, std::string_view indent, bool direct, bool refill_consume) {
  for (std::size_t r = 0; r < results.size(); ++r) {
    const auto& result = results[r];
    const double p50 = percentile(result.samples.wall_ms, 0.50);
    const double p95 = percentile(result.samples.wall_ms, 0.95);
    const double p99 = percentile(result.samples.wall_ms, 0.99);
    out << indent << "{\"depth\": " << result.depth
        << ", \"graph_executions_per_repeat\": " << result.depth
        << ", \"api_submissions_per_repeat\": " << (direct ? 1 : result.depth)
        << ", \"api_waits_per_repeat\": " << (direct ? 1 : result.depth)
        << ", \"api_event_completion_queries_per_repeat\": " << (direct ? result.depth : 0)
        << ", \"application_input_fill_bytes_per_repeat\": "
        << (refill_consume ? result.depth * (kPqM * kKs * sizeof(float) +
                                             rows * kPqM * sizeof(std::int32_t))
                           : 0)
        << ", \"application_output_consumed_bytes_per_repeat\": "
        << (refill_consume ? result.depth * rows * sizeof(float) : 0)
        << ", \"p50_wall_ms\": " << p50 << ", \"p95_wall_ms\": " << p95
        << ", \"p99_wall_ms\": " << p99
        << ", \"requests_per_second_at_p50\": " << (1000.0 * result.depth / p50)
        << ", \"candidate_rows_per_second_at_p50\": "
        << (1000.0 * result.depth * static_cast<double>(rows) / p50) << ", \"samples_ms\": [";
    for (std::size_t i = 0; i < result.samples.wall_ms.size(); ++i) {
      if (i) out << ", ";
      out << result.samples.wall_ms[i];
    }
    out << "]}" << (r + 1 == results.size() ? "\n" : ",\n");
  }
}

std::uint64_t process_peak_working_set() {
  PROCESS_MEMORY_COUNTERS_EX counters{};
  counters.cb = sizeof(counters);
  if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                            sizeof(counters))) {
    return 0;
  }
  return static_cast<std::uint64_t>(counters.PeakWorkingSetSize);
}

ov::CompiledModel compile_npu(ov::Core& core, const std::shared_ptr<ov::Model>& model,
                              std::string& effective_profile, int& attempts) {
  ov::AnyMap props;
  props["PERFORMANCE_HINT"] = "LATENCY";
  props["NPU_TURBO"] = true;
  props["NPU_COMPILATION_MODE_PARAMS"] =
      std::string("optimization-level=2 performance-hint-override=latency");
  attempts = 1;
  try {
    auto compiled = core.compile_model(model, "NPU", props);
    effective_profile = "LATENCY+NPU_TURBO+optimization-level=2";
    return compiled;
  } catch (...) {
    props.erase("NPU_COMPILATION_MODE_PARAMS");
    attempts = 2;
    try {
      auto compiled = core.compile_model(model, "NPU", props);
      effective_profile = "LATENCY+NPU_TURBO";
      return compiled;
    } catch (...) {
      props.erase("NPU_TURBO");
      attempts = 3;
      auto compiled = core.compile_model(model, "NPU", props);
      effective_profile = "LATENCY";
      return compiled;
    }
  }
}

int run(const Options& options) {
  Fixture fixture = make_fixture(options.rows);
  ov::Core core;
  const auto devices = core.get_available_devices();
  if (std::find(devices.begin(), devices.end(), "NPU") == devices.end()) {
    throw Failure("OpenVINO does not expose an NPU device");
  }
  std::string openvino_device = "NPU";
  try {
    openvino_device = core.get_property("NPU", "FULL_DEVICE_NAME").as<std::string>();
  } catch (...) {
  }

  const int max_depth = *std::max_element(options.depths.begin(), options.depths.end());
  double compile_ms = 0.0;
  double export_ms = 0.0;
  double openvino_request_setup_ms = 0.0;
  std::size_t openvino_tensor_payload_bytes = 0;
  std::string compile_profile;
  int compile_attempts = 0;
  std::vector<std::uint8_t> blob;
  ScopeResults openvino_results;
  {
    const auto compile_start = Clock::now();
    ov::CompiledModel compiled =
        compile_npu(core, fixture.model, compile_profile, compile_attempts);
    compile_ms = elapsed_ms(compile_start);

    const auto export_start = Clock::now();
    std::stringstream blob_stream(std::ios::in | std::ios::out | std::ios::binary);
    compiled.export_model(blob_stream);
    const std::string blob_string = blob_stream.str();
    blob.assign(blob_string.begin(), blob_string.end());
    export_ms = elapsed_ms(export_start);
    if (blob.empty()) throw Failure("OpenVINO exported an empty native blob");

    OpenVinoPool openvino_pool(fixture, std::move(compiled), max_depth);
    openvino_results = measure_openvino(openvino_pool, options);
    openvino_request_setup_ms = openvino_pool.request_setup_ms;
    openvino_tensor_payload_bytes = openvino_pool.tensor_payload_bytes;
  }

  std::unique_ptr<DirectGraphPool> direct;
  ScopeResults direct_results;
  std::string direct_error;
  try {
    direct = std::make_unique<DirectGraphPool>(fixture, blob, options.depths);
    direct_results = measure_direct(*direct, options);
  } catch (const std::exception& e) {
    direct_error = e.what();
    direct.reset();
  }

  const ov::Version version = ov::get_openvino_version();
  const std::size_t application_setup_copy_bytes =
      static_cast<std::size_t>(max_depth) *
      (fixture.lut.size() * sizeof(float) + fixture.indices.size() * sizeof(std::int32_t));

  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "{\n"
      << "  \"schema\": \"ovvs.npu_escape.level_zero_graph.v1\",\n"
      << "  \"status\": \"" << (direct ? "ok" : "direct_unavailable") << "\",\n"
      << "  \"scope\": \"Gather+ReduceSum orchestration and refill/consume microexperiment\",\n"
      << "  \"acceleration_claim\": false,\n"
      << "  \"measurement_repeat_gate_met\": " << (options.repeats >= 5 ? "true" : "false")
      << ",\n"
      << "  \"limitations\": [\"no Level Zero graph profiling query; device execution time is not separated\", "
          "\"prefilled scope removes per-repeat application copies from both lanes\", "
          "\"OpenVINO is measured before direct Level Zero and driver-cache state is uncontrolled\", "
          "\"not an end-to-end IVF-PQ or recall measurement\"],\n"
      << "  \"fixture\": {\"rows_per_graph\": " << fixture.rows << ", \"pq_m\": " << kPqM
      << ", \"ks\": " << kKs << ", \"oracle\": \"CPU exact integer-sum\", "
      << "\"oracle_hash_fnv1a64\": \""
      << hex_u64(fnv1a(fixture.oracle.data(), fixture.oracle.size() * sizeof(float)))
      << "\", \"correctness\": \"all rows and every measured slot/epoch exact and finite\", "
         "\"output_validation\": \"NaN poison before every submission; validate after every completion\"},\n"
      << "  \"software\": {\"openvino_build\": \"" << json_escape(version.buildNumber)
      << "\", \"openvino_description\": \"" << json_escape(version.description) << "\"},\n"
      << "  \"compile\": {\"compile_model_call_ms\": " << compile_ms
      << ", \"attempts\": " << compile_attempts << ", \"effective_profile\": \""
      << json_escape(compile_profile)
      << "\", \"fresh_process\": true, \"driver_cache_controlled\": false, \"native_export_ms\": "
      << export_ms << ", \"native_blob_bytes\": " << blob.size()
      << ", \"native_blob_hash_fnv1a64\": \"" << hex_u64(fnv1a(blob.data(), blob.size()))
      << "\"},\n"
      << "  \"openvino_request_pool\": {\n"
      << "    \"status\": \"ok\", \"device\": \"" << json_escape(openvino_device)
      << "\", \"persistent_compiled_model\": true, \"persistent_requests\": " << max_depth
      << ", \"released_before_direct_setup\": true"
      << ", \"request_setup_and_fill_ms\": " << openvino_request_setup_ms
      << ", \"request_tensor_payload_bytes\": " << openvino_tensor_payload_bytes
      << ", \"application_setup_copied_bytes\": " << application_setup_copy_bytes
      << ", \"queue_wait_ms\": null, \"npu_execution_ms\": null,\n"
      << "    \"prefilled_orchestration\": {\"depths\": [\n";
  write_samples(out, openvino_results.prefilled, fixture.rows, "      ", false, false);
  out << "    ]},\n"
      << "    \"refill_execute_consume\": {\"depths\": [\n";
  write_samples(out, openvino_results.refill_consume, fixture.rows, "      ", false, true);
  out << "    ]}\n  },\n";

  if (direct) {
    out << "  \"direct_level_zero\": {\n"
        << "    \"status\": \"ok\", \"device\": \"" << json_escape(direct->device_name)
        << "\", \"device_uuid\": \"" << direct->device_uuid
        << "\", \"graph_extension_version\": \"" << ZE_MAJOR_VERSION(direct->extension_version)
        << "." << ZE_MINOR_VERSION(direct->extension_version)
        << "\", \"queue_group_ordinal\": " << direct->queue_ordinal
        << ", \"reported_queue_count\": " << direct->queue_count
        << ", \"selected_queue_index\": 0,\n"
        << "    \"persistent_graph\": true, \"persistent_async_queue\": true, "
           "\"persistent_command_lists\": " << max_depth
        << ", \"persistent_fences\": " << options.depths.size()
        << ", \"persistent_events\": " << max_depth
        << ", \"native_blob_persistent_flag\": " << (direct->persistent_blob ? "true" : "false")
        << ",\n"
        << "    \"argument_binding_validation\": \"slot/epoch-distinct LUTs, codes, and expected "
           "outputs; every output is NaN-poisoned and validated after completion\",\n"
        << "    \"setup_ms\": " << direct->setup_ms
        << ", \"native_blob_payload_bytes\": " << direct->blob_payload_bytes
        << ", \"native_blob_allocation_bytes\": " << direct->blob_allocated_bytes
        << ", \"owned_l0_host_buffer_bytes\": " << direct->owned_host_bytes
        << ", \"application_setup_copied_bytes\": "
        << application_setup_copy_bytes + direct->blob_payload_bytes
        << ", \"queue_wait_ms\": null, \"npu_execution_ms\": null,\n"
        << "    \"prefilled_orchestration\": {\"depths\": [\n";
    write_samples(out, direct_results.prefilled, fixture.rows, "      ", true, false);
    out << "    ]},\n"
        << "    \"refill_execute_consume\": {\"depths\": [\n";
    write_samples(out, direct_results.refill_consume, fixture.rows, "      ", true, true);
    out << "    ]}\n  },\n";
  } else {
    out << "  \"direct_level_zero\": {\"status\": \"unavailable\", \"reason\": \""
        << json_escape(direct_error) << "\"},\n";
  }
  out << "  \"process_peak_working_set_bytes\": " << process_peak_working_set() << "\n}\n";
  std::cout << out.str();
  return direct ? 0 : 3;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parse_options(argc, argv));
  } catch (const std::exception& e) {
    std::cerr << "{\"schema\":\"ovvs.npu_escape.level_zero_graph.v1\",\"status\":\"error\","
                 "\"error\":\""
              << json_escape(e.what()) << "\"}\n";
    return 2;
  }
}
