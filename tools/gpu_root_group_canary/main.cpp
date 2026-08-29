#include <sycl/feature_test.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kWorkGroups = 4;
constexpr std::uint32_t kLocalSize = 128;
/* Conservative gpu_cagra_walk admission footprint for 128/4: a 256-slot
   beam, four picks, the 64-ID frontier tile, and bounded metadata. */
constexpr std::uint32_t kLocalBytes =
    256u * (sizeof(std::int32_t) + sizeof(float) + sizeof(std::uint8_t)) +
    4u * sizeof(std::int32_t) + 64u * sizeof(std::int32_t) + 64u;
static_assert(kLocalBytes == 2640u);
constexpr std::uint32_t kRepeats = 5;
constexpr std::uint32_t kGroupState = 0;
constexpr std::uint32_t kReadyMaskState = kGroupState + kWorkGroups;
constexpr std::uint32_t kParticipantsState = kReadyMaskState + 1;
constexpr std::uint32_t kFinalState = kParticipantsState + 1;
constexpr std::uint32_t kStateWords = kFinalState + 1;
constexpr std::uint32_t kReadyMask = (1u << kWorkGroups) - 1u;
constexpr std::uint32_t kSuccessMarker = 0xc001cafeu;

class RootGroupCanaryKernel;

constexpr std::uint32_t scratch_checksum(std::uint32_t group) {
  std::uint32_t sum = 0;
  for (std::uint32_t offset = 0; offset < kLocalBytes; ++offset) {
    sum += (offset + group * 17u) & 0xffu;
  }
  return sum;
}

std::string json_string(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char ch : value) {
    switch (ch) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (ch < 0x20u) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<unsigned>(ch) << std::dec << std::setfill(' ');
        } else {
          out << static_cast<char>(ch);
        }
    }
  }
  out << '"';
  return out.str();
}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
bool has_unified_host_memory(const sycl::device& device) {
  return device.get_info<sycl::info::device::host_unified_memory>();
}
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

std::optional<sycl::device> select_integrated_intel_level_zero_gpu(
    std::string& reason) {
  std::vector<sycl::device> candidates;
  try {
    for (const sycl::platform& platform : sycl::platform::get_platforms()) {
      std::string platform_name =
          platform.get_info<sycl::info::platform::name>();
      std::transform(platform_name.begin(), platform_name.end(),
                     platform_name.begin(), [](unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                     });
      if (platform_name.find("level-zero") == std::string::npos &&
          platform_name.find("level zero") == std::string::npos) {
        continue;
      }
      for (const sycl::device& device :
           platform.get_devices(sycl::info::device_type::gpu)) {
        if (device.get_info<sycl::info::device::vendor_id>() == 0x8086u &&
            has_unified_host_memory(device)) {
          candidates.push_back(device);
        }
      }
    }
  } catch (const sycl::exception& exception) {
    reason = std::string("device_enumeration_failed: ") + exception.what();
    return std::nullopt;
  }
  if (candidates.empty()) {
    reason = "no Intel integrated Level Zero GPU";
    return std::nullopt;
  }
  if (candidates.size() != 1) {
    reason = "multiple Intel integrated Level Zero GPUs; selection is ambiguous";
    return std::nullopt;
  }
  return candidates.front();
}

void print_unavailable(const std::string& reason) {
  std::ostringstream out;
  out << "{\"schema\":\"ovvs.gpu-root-group-canary.v1\","
      << "\"status\":\"device_unavailable\",\"canary_admitted\":false,"
      << "\"fallback\":false,\"scope\":\"capability_only_not_search\","
      << "\"acceleration_claim\":false,"
      << "\"integration_status\":\"standalone_not_routed\","
      << "\"reason\":" << json_string(reason) << "}\n";
  std::fputs(out.str().c_str(), stdout);
}

struct Observation {
  std::array<std::uint32_t, kStateWords> state{};
  double wall_ms = 0.0;
  bool valid = false;
};

bool validate(const std::uint32_t* state) {
  for (std::uint32_t group = 0; group < kWorkGroups; ++group) {
    if (state[kGroupState + group] != scratch_checksum(group)) return false;
  }
  return state[kReadyMaskState] == kReadyMask &&
         state[kParticipantsState] == kWorkGroups * kLocalSize &&
         state[kFinalState] == kSuccessMarker;
}

}  // namespace

int main() {
#if !defined(SYCL_EXT_ONEAPI_ROOT_GROUP) || \
    !defined(SYCL_EXT_ONEAPI_MAX_WORK_GROUP_QUERY)
  print_unavailable(
      "required SYCL root-group or maximum-work-group query API is absent");
  return 77;
#else
  namespace syclex = sycl::ext::oneapi::experimental;

  std::string selection_reason;
  const std::optional<sycl::device> selected =
      select_integrated_intel_level_zero_gpu(selection_reason);
  if (!selected) {
    print_unavailable(selection_reason);
    return 77;
  }

  const sycl::device device = *selected;
  const std::string device_name =
      device.get_info<sycl::info::device::name>();
  const std::string driver =
      device.get_info<sycl::info::device::driver_version>();
  const std::string platform =
      device.get_platform().get_info<sycl::info::platform::name>();
  const std::size_t max_work_group_size =
      device.get_info<sycl::info::device::max_work_group_size>();
  const std::uint64_t local_mem_bytes =
      device.get_info<sycl::info::device::local_mem_size>();
  const bool shared_usm = device.has(sycl::aspect::usm_shared_allocations);

  if (max_work_group_size < kLocalSize || local_mem_bytes < kLocalBytes ||
      !shared_usm) {
    print_unavailable(
        "required work-group, local-memory, or shared-USM capability missing");
    return 77;
  }

  std::uint32_t* state = nullptr;
  try {
    sycl::queue queue(device, sycl::property::queue::in_order{});
    const auto bundle =
        sycl::get_kernel_bundle<RootGroupCanaryKernel,
                                sycl::bundle_state::executable>(
            queue.get_context(), std::vector<sycl::device>{device});
    const sycl::kernel kernel = bundle.get_kernel<RootGroupCanaryKernel>();
    const std::size_t max_num_work_groups =
        kernel.ext_oneapi_get_info<
            syclex::info::kernel_queue_specific::max_num_work_groups>(
            queue, sycl::range<1>(kLocalSize), kLocalBytes);

    if (max_num_work_groups < kWorkGroups) {
      std::ostringstream out;
      out << "{\"schema\":\"ovvs.gpu-root-group-canary.v1\","
          << "\"status\":\"unsupported\",\"canary_admitted\":false,"
          << "\"fallback\":false,\"scope\":\"capability_only_not_search\","
          << "\"acceleration_claim\":false,"
          << "\"integration_status\":\"standalone_not_routed\","
          << "\"device\":" << json_string(device_name)
          << ",\"driver\":" << json_string(driver)
          << ",\"platform\":" << json_string(platform)
          << ",\"requested\":{\"work_groups\":" << kWorkGroups
          << ",\"local_size\":" << kLocalSize
          << ",\"local_memory_bytes_per_group\":" << kLocalBytes
          << "},\"query\":{\"max_num_work_groups\":"
          << max_num_work_groups
          << "},\"reason\":\"cooperative capacity is below four work-groups\"}\n";
      std::fputs(out.str().c_str(), stdout);
      return 77;
    }

    state = sycl::malloc_shared<std::uint32_t>(kStateWords, queue);
    if (!state) throw std::bad_alloc();

    std::array<Observation, kRepeats> observations;
    bool all_valid = true;
    for (std::uint32_t repeat = 0; repeat < kRepeats; ++repeat) {
      std::fill(state, state + kStateWords, 0u);
      const auto start = std::chrono::steady_clock::now();
      sycl::event event = queue.submit([&](sycl::handler& handler) {
        sycl::local_accessor<std::uint8_t, 1> scratch(
            sycl::range<1>(kLocalBytes), handler);
        handler.parallel_for<RootGroupCanaryKernel>(
            sycl::nd_range<1>(sycl::range<1>(kWorkGroups * kLocalSize),
                              sycl::range<1>(kLocalSize)),
            syclex::properties{syclex::use_root_sync},
            [=](sycl::nd_item<1> item) {
              const std::uint32_t local =
                  static_cast<std::uint32_t>(item.get_local_linear_id());
              const std::uint32_t group =
                  static_cast<std::uint32_t>(item.get_group_linear_id());
              for (std::uint32_t offset = local; offset < kLocalBytes;
                   offset += kLocalSize) {
                scratch[offset] = static_cast<std::uint8_t>(
                    (offset + group * 17u) & 0xffu);
              }
              item.barrier(sycl::access::fence_space::local_space);
              if (local == 0) {
                std::uint32_t checksum = 0;
                for (std::uint32_t offset = 0; offset < kLocalBytes; ++offset) {
                  checksum += scratch[offset];
                }
                state[kGroupState + group] = checksum;
              }

              const auto root = item.ext_oneapi_get_root_group();
              sycl::group_barrier(root);
              if (item.get_global_linear_id() == 0) {
                std::uint32_t ready_mask = 0;
                for (std::uint32_t candidate = 0; candidate < kWorkGroups;
                     ++candidate) {
                  if (state[kGroupState + candidate] ==
                      scratch_checksum(candidate)) {
                    ready_mask |= 1u << candidate;
                  }
                }
                state[kReadyMaskState] = ready_mask;
              }

              sycl::group_barrier(root);
              if (state[kReadyMaskState] == kReadyMask) {
                sycl::atomic_ref<
                    std::uint32_t, sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>
                    participants(state[kParticipantsState]);
                participants.fetch_add(1u);
              }

              sycl::group_barrier(root);
              if (item.get_global_linear_id() == 0) {
                state[kFinalState] =
                    state[kReadyMaskState] == kReadyMask &&
                            state[kParticipantsState] ==
                                kWorkGroups * kLocalSize
                        ? kSuccessMarker
                        : 0u;
              }
            });
      });
      event.wait_and_throw();
      Observation& observation = observations[repeat];
      observation.wall_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - start)
              .count();
      std::copy(state, state + kStateWords, observation.state.begin());
      observation.valid = validate(state);
      all_valid = all_valid && observation.valid;
    }

    sycl::free(state, queue);
    state = nullptr;

    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << "{\"schema\":\"ovvs.gpu-root-group-canary.v1\","
        << "\"status\":" << (all_valid ? "\"success\"" : "\"failed\"")
        << ",\"canary_admitted\":" << (all_valid ? "true" : "false")
        << ",\"fallback\":false,\"scope\":\"capability_only_not_search\","
        << "\"acceleration_claim\":false,"
        << "\"integration_status\":\"standalone_not_routed\","
        << "\"device\":"
        << json_string(device_name) << ",\"driver\":" << json_string(driver)
        << ",\"platform\":" << json_string(platform)
        << ",\"requested\":{\"work_groups\":" << kWorkGroups
        << ",\"local_size\":" << kLocalSize
        << ",\"local_memory_bytes_per_group\":" << kLocalBytes
        << ",\"root_barriers_per_launch\":3,\"launches\":" << kRepeats
        << "},\"device_capabilities\":{\"max_work_group_size\":"
        << max_work_group_size << ",\"local_memory_bytes\":"
        << local_mem_bytes << ",\"shared_usm\":true},"
        << "\"query\":{\"max_num_work_groups\":" << max_num_work_groups
        << "},\"observations\":[";
    for (std::uint32_t repeat = 0; repeat < kRepeats; ++repeat) {
      if (repeat != 0) out << ',';
      const Observation& observation = observations[repeat];
      out << "{\"valid\":" << (observation.valid ? "true" : "false")
          << ",\"wall_ms\":" << observation.wall_ms
          << ",\"group_checksums\":[";
      for (std::uint32_t group = 0; group < kWorkGroups; ++group) {
        if (group != 0) out << ',';
        out << observation.state[kGroupState + group];
      }
      out << "],\"ready_mask\":" << observation.state[kReadyMaskState]
          << ",\"participants\":"
          << observation.state[kParticipantsState]
          << ",\"final_marker\":" << observation.state[kFinalState] << '}';
    }
    out << "]}\n";
    std::fputs(out.str().c_str(), stdout);
    return all_valid ? 0 : 1;
  } catch (const sycl::exception& exception) {
    std::ostringstream out;
    out << "{\"schema\":\"ovvs.gpu-root-group-canary.v1\","
        << "\"status\":\"failed\",\"canary_admitted\":false,"
        << "\"fallback\":false,\"scope\":\"capability_only_not_search\","
        << "\"acceleration_claim\":false,"
        << "\"integration_status\":\"standalone_not_routed\","
        << "\"device\":" << json_string(device_name)
        << ",\"driver\":" << json_string(driver)
        << ",\"platform\":" << json_string(platform)
        << ",\"reason\":" << json_string(exception.what()) << "}\n";
    std::fputs(out.str().c_str(), stdout);
  } catch (const std::exception& exception) {
    std::ostringstream out;
    out << "{\"schema\":\"ovvs.gpu-root-group-canary.v1\","
        << "\"status\":\"failed\",\"canary_admitted\":false,"
        << "\"fallback\":false,\"scope\":\"capability_only_not_search\","
        << "\"acceleration_claim\":false,"
        << "\"integration_status\":\"standalone_not_routed\","
        << "\"device\":" << json_string(device_name)
        << ",\"driver\":" << json_string(driver)
        << ",\"platform\":" << json_string(platform)
        << ",\"reason\":" << json_string(exception.what()) << "}\n";
    std::fputs(out.str().c_str(), stdout);
  }
  return 1;
#endif
}
