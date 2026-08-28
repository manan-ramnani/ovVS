#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

/*
 * Standalone B3 feasibility lane. One work-group scans a bounded list block,
 * computes ADC directly from u8 codes, and emits only that block's exact K
 * best (score, dense-id) pairs. The union of every block's K best contains the
 * global K best, so a bounded host merge is exact while avoiding both NPU-style
 * i32 index expansion and full score materialization/readback. This executable
 * has no CPU execution fallback: the scalar path below is an oracle only.
 */

using Clock = std::chrono::steady_clock;

constexpr uint32_t kMaxLocalK = 64;
constexpr uint32_t kDefaultBlockRows = 256;
constexpr uint32_t kInvalidId = std::numeric_limits<uint32_t>::max();
constexpr float kInfinity = std::numeric_limits<float>::infinity();

struct Task {
  uint64_t code_offset = 0;
  uint32_t rows = 0;
  uint32_t table_index = 0;
  uint32_t first_id = 0;
};

struct Block {
  uint64_t code_offset = 0;
  uint32_t rows = 0;
  uint32_t table_index = 0;
  uint32_t first_id = 0;
};

struct HostTopK {
  std::vector<float> scores;
  std::vector<uint32_t> ids;
};

struct StageTiming {
  double wall_ms = 0.0;
  double reset_ms = 0.0;
  double table_upload_ms = 0.0;
  double scan_select_ms = 0.0;
  double host_final_select_ms = 0.0;
  double readback_ms = 0.0;
  double unattributed_ms = 0.0;
  uint32_t queue_submissions = 0;
  uint32_t host_synchronizations = 0;
};

struct RunResult {
  bool success = false;
  std::string error;
  HostTopK published;
  StageTiming timing;
};

struct FixtureSlice {
  uint64_t global_begin = 0;
  std::vector<float> tables;
  std::vector<uint8_t> codes;
  std::vector<Task> tasks;
};

template <typename T>
class DeviceAllocation {
 public:
  DeviceAllocation() = default;

  DeviceAllocation(sycl::queue& queue, size_t count) : queue_(&queue) {
    if (count == 0) return;
    pointer_ = sycl::malloc_device<T>(count, queue);
    if (!pointer_) throw std::bad_alloc();
  }

  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;

  DeviceAllocation(DeviceAllocation&& other) noexcept
      : queue_(other.queue_), pointer_(other.pointer_) {
    other.queue_ = nullptr;
    other.pointer_ = nullptr;
  }

  DeviceAllocation& operator=(DeviceAllocation&& other) noexcept {
    if (this == &other) return *this;
    release();
    queue_ = other.queue_;
    pointer_ = other.pointer_;
    other.queue_ = nullptr;
    other.pointer_ = nullptr;
    return *this;
  }

  ~DeviceAllocation() { release(); }

  T* get() const { return pointer_; }

 private:
  void release() noexcept {
    if (!pointer_ || !queue_) return;
    try {
      queue_->wait_and_throw();
    } catch (...) {
    }
    try {
      sycl::free(pointer_, *queue_);
    } catch (...) {
    }
    pointer_ = nullptr;
  }

  sycl::queue* queue_ = nullptr;
  T* pointer_ = nullptr;
};

bool checked_multiply(size_t lhs, size_t rhs, size_t& result) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) return false;
  result = lhs * rhs;
  return true;
}

bool better(float score, uint32_t id, float incumbent_score, uint32_t incumbent_id) {
  return score < incumbent_score ||
         (score == incumbent_score && id < incumbent_id);
}

void insert_topk(std::vector<float>& scores, std::vector<uint32_t>& ids, float score,
                 uint32_t id) {
  if (!better(score, id, scores.back(), ids.back())) return;
  size_t position = scores.size() - 1;
  while (position > 0 && better(score, id, scores[position - 1], ids[position - 1])) {
    scores[position] = scores[position - 1];
    ids[position] = ids[position - 1];
    --position;
  }
  scores[position] = score;
  ids[position] = id;
}

bool preflight(const std::vector<float>& tables, const std::vector<uint8_t>& codes,
               const std::vector<Task>& tasks, uint32_t pq_m, uint32_t ks,
               uint32_t local_k, uint64_t candidate_count, std::string& error) {
  if (pq_m == 0 || ks == 0 || ks > 256 || local_k == 0 || local_k > kMaxLocalK ||
      candidate_count == 0 || candidate_count > kInvalidId || local_k > candidate_count) {
    error = "invalid_shape";
    return false;
  }
  size_t table_stride = 0;
  if (!checked_multiply(static_cast<size_t>(pq_m), static_cast<size_t>(ks), table_stride) ||
      tasks.size() > std::numeric_limits<size_t>::max() / table_stride ||
      tables.size() != tasks.size() * table_stride) {
    error = "table_shape_mismatch";
    return false;
  }
  if (candidate_count > std::numeric_limits<size_t>::max() / pq_m ||
      codes.size() != static_cast<size_t>(candidate_count) * pq_m) {
    error = "code_shape_mismatch";
    return false;
  }
  for (float value : tables) {
    if (!std::isfinite(value)) {
      error = "non_finite_table";
      return false;
    }
  }
  uint64_t expected_row = 0;
  for (const Task& task : tasks) {
    if (task.rows == 0 || task.first_id != expected_row ||
        task.table_index >= tasks.size() || task.code_offset != expected_row * pq_m ||
        task.rows > candidate_count - expected_row) {
      error = "task_descriptor_mismatch";
      return false;
    }
    expected_row += task.rows;
  }
  if (expected_row != candidate_count) {
    error = "task_coverage_mismatch";
    return false;
  }
  for (uint8_t code : codes) {
    if (static_cast<uint32_t>(code) >= ks) {
      error = "code_out_of_range";
      return false;
    }
  }
  return true;
}

std::vector<Task> make_tasks(uint64_t candidate_count, uint32_t task_count,
                             uint32_t pq_m) {
  if (task_count == 0 || candidate_count < task_count || candidate_count > kInvalidId) {
    throw std::invalid_argument("invalid task geometry");
  }
  std::vector<uint64_t> weights(task_count);
  for (uint32_t task = 0; task < task_count; ++task) {
    weights[task] = 1u + ((task * 5u + 3u) % 11u);
  }
  const uint64_t total_weight = std::accumulate(weights.begin(), weights.end(), uint64_t{0});
  std::vector<Task> tasks;
  tasks.reserve(task_count);
  uint64_t first = 0;
  uint64_t cumulative_weight = 0;
  for (uint32_t task = 0; task < task_count; ++task) {
    cumulative_weight += weights[task];
    uint64_t end = task + 1 == task_count
                       ? candidate_count
                       : candidate_count * cumulative_weight / total_weight;
    const uint64_t tasks_left = static_cast<uint64_t>(task_count - task - 1);
    end = std::max(end, first + 1);
    end = std::min(end, candidate_count - tasks_left);
    const uint64_t rows = end - first;
    tasks.push_back({first * pq_m, static_cast<uint32_t>(rows), task,
                     static_cast<uint32_t>(first)});
    first = end;
  }
  return tasks;
}

std::vector<Block> make_blocks(const std::vector<Task>& tasks, uint32_t pq_m,
                               uint32_t block_rows) {
  std::vector<Block> blocks;
  for (const Task& task : tasks) {
    uint32_t consumed = 0;
    while (consumed < task.rows) {
      const uint32_t rows = std::min(block_rows, task.rows - consumed);
      blocks.push_back({task.code_offset + static_cast<uint64_t>(consumed) * pq_m,
                        rows, task.table_index, task.first_id + consumed});
      consumed += rows;
    }
  }
  return blocks;
}

HostTopK scalar_oracle(const std::vector<float>& tables,
                       const std::vector<uint8_t>& codes,
                       const std::vector<Task>& tasks, uint32_t pq_m, uint32_t ks,
                       uint32_t local_k) {
  HostTopK result{std::vector<float>(local_k, kInfinity),
                  std::vector<uint32_t>(local_k, kInvalidId)};
  const size_t table_stride = static_cast<size_t>(pq_m) * ks;
  for (const Task& task : tasks) {
    const float* table = tables.data() + static_cast<size_t>(task.table_index) * table_stride;
    for (uint32_t row = 0; row < task.rows; ++row) {
      const uint8_t* code = codes.data() + task.code_offset + static_cast<uint64_t>(row) * pq_m;
      float score = 0.0f;
      for (uint32_t m = 0; m < pq_m; ++m) {
        score += table[static_cast<size_t>(m) * ks + code[m]];
      }
      insert_topk(result.scores, result.ids, score, task.first_id + row);
    }
  }
  return result;
}

FixtureSlice slice_fixture(const std::vector<float>& tables,
                           const std::vector<uint8_t>& codes,
                           const std::vector<Task>& tasks, uint64_t global_begin,
                           uint64_t global_end, uint32_t pq_m, uint32_t ks) {
  if (global_begin >= global_end) throw std::invalid_argument("empty fixture slice");
  FixtureSlice slice;
  slice.global_begin = global_begin;
  const size_t table_stride = static_cast<size_t>(pq_m) * ks;
  uint64_t local_row = 0;
  for (const Task& task : tasks) {
    const uint64_t task_begin = task.first_id;
    const uint64_t task_end = task_begin + task.rows;
    const uint64_t overlap_begin = std::max(global_begin, task_begin);
    const uint64_t overlap_end = std::min(global_end, task_end);
    if (overlap_begin >= overlap_end) continue;
    const size_t source_table = static_cast<size_t>(task.table_index) * table_stride;
    slice.tables.insert(slice.tables.end(), tables.begin() + source_table,
                        tables.begin() + source_table + table_stride);
    const uint64_t source_code = overlap_begin * pq_m;
    const uint64_t rows = overlap_end - overlap_begin;
    const size_t code_count = static_cast<size_t>(rows) * pq_m;
    slice.codes.insert(slice.codes.end(), codes.begin() + source_code,
                       codes.begin() + source_code + code_count);
    slice.tasks.push_back({local_row * pq_m, static_cast<uint32_t>(rows),
                           static_cast<uint32_t>(slice.tasks.size()),
                           static_cast<uint32_t>(local_row)});
    local_row += rows;
  }
  if (local_row != global_end - global_begin) {
    throw std::runtime_error("fixture slice did not preserve dense coverage");
  }
  return slice;
}

HostTopK merge_partition_topk(const RunResult& gpu, uint64_t gpu_global_begin,
                              const HostTopK& cpu, uint64_t cpu_global_begin,
                              uint32_t local_k) {
  HostTopK merged{std::vector<float>(local_k, kInfinity),
                  std::vector<uint32_t>(local_k, kInvalidId)};
  for (uint32_t rank = 0; rank < local_k; ++rank) {
    const uint64_t gpu_id = gpu_global_begin + gpu.published.ids[rank + 1];
    const uint64_t cpu_id = cpu_global_begin + cpu.ids[rank];
    if (gpu_id >= kInvalidId || cpu_id >= kInvalidId) {
      throw std::overflow_error("partition result id exceeds bounded u32 fixture");
    }
    insert_topk(merged.scores, merged.ids, gpu.published.scores[rank + 1],
                static_cast<uint32_t>(gpu_id));
    insert_topk(merged.scores, merged.ids, cpu.scores[rank],
                static_cast<uint32_t>(cpu_id));
  }
  return merged;
}

std::optional<double> event_ms(const sycl::event& event) {
  try {
    const uint64_t start =
        event.get_profiling_info<sycl::info::event_profiling::command_start>();
    const uint64_t end =
        event.get_profiling_info<sycl::info::event_profiling::command_end>();
    if (end < start) return std::nullopt;
    return static_cast<double>(end - start) / 1.0e6;
  } catch (...) {
    return std::nullopt;
  }
}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
bool has_unified_host_memory(const sycl::device& device) {
  /* SYCL 2020 has no replacement that distinguishes integrated from discrete
     GPUs: shared-USM is supported by both. This read-only property is retained
     solely as a fail-closed device-selection guard for the experiment. */
  return device.get_info<sycl::info::device::host_unified_memory>();
}
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

std::optional<sycl::device> select_intel_integrated_level_zero_gpu(std::string& reason) {
  std::vector<sycl::device> candidates;
  try {
    for (const sycl::platform& platform : sycl::platform::get_platforms()) {
      std::string platform_name = platform.get_info<sycl::info::platform::name>();
      std::transform(platform_name.begin(), platform_name.end(), platform_name.begin(),
                     [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
      if (platform_name.find("level-zero") == std::string::npos &&
          platform_name.find("level zero") == std::string::npos) {
        continue;
      }
      for (const sycl::device& device : platform.get_devices(sycl::info::device_type::gpu)) {
        if (device.get_info<sycl::info::device::vendor_id>() != 0x8086u ||
            !has_unified_host_memory(device)) {
          continue;
        }
        candidates.push_back(device);
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

class FusedGpuRunner {
 public:
  FusedGpuRunner(sycl::queue& queue, const std::vector<float>& tables,
                 const std::vector<uint8_t>& codes, const std::vector<Block>& blocks,
                 uint32_t pq_m, uint32_t ks, uint32_t local_k)
      : queue_(queue),
        pq_m_(pq_m),
        ks_(ks),
        local_k_(local_k),
        table_elements_(tables.size()),
        code_elements_(codes.size()),
        block_count_(blocks.size()),
        partial_elements_(blocks.size() * static_cast<size_t>(local_k)),
        tables_(queue, table_elements_),
        codes_(queue, code_elements_),
        blocks_(queue, block_count_),
        partial_scores_(queue, partial_elements_),
        partial_ids_(queue, partial_elements_),
        error_(queue, 1) {
    const auto start = Clock::now();
    queue_.memcpy(codes_.get(), codes.data(), code_elements_ * sizeof(uint8_t));
    queue_.memcpy(blocks_.get(), blocks.data(), block_count_ * sizeof(Block));
    queue_.wait_and_throw();
    persistent_upload_ms_ =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  }

  RunResult run(const std::vector<float>& tables, const HostTopK& oracle) {
    RunResult result;
    const auto wall_start = Clock::now();
    try {
      result.published.scores.assign(local_k_ + 2, -12345.0f);
      result.published.ids.assign(local_k_ + 2, 0xdeadbeefu);
    } catch (const std::exception& exception) {
      result.error = std::string("output_staging_failed: ") + exception.what();
      return result;
    } catch (...) {
      result.error = "output_staging_failed";
      return result;
    }
    if (tables.size() != table_elements_) {
      result.error = "table_shape_changed";
      return result;
    }
    for (float value : tables) {
      if (!std::isfinite(value)) {
        result.error = "non_finite_table";
        return result;
      }
    }

    std::vector<float> staged_scores;
    std::vector<uint32_t> staged_ids;
    int32_t staged_error = 0;
    try {
      staged_scores.resize(partial_elements_);
      staged_ids.resize(partial_elements_);
      const sycl::event reset_event = queue_.memset(error_.get(), 0, sizeof(int32_t));
      const sycl::event table_event =
          queue_.memcpy(tables_.get(), tables.data(), table_elements_ * sizeof(float));

      const float* device_tables = tables_.get();
      const uint8_t* device_codes = codes_.get();
      const Block* device_blocks = blocks_.get();
      float* partial_scores = partial_scores_.get();
      uint32_t* partial_ids = partial_ids_.get();
      int32_t* error = error_.get();
      const uint32_t pq_m = pq_m_;
      const uint32_t ks = ks_;
      const uint32_t local_k = local_k_;

      const sycl::event scan_event = queue_.submit([&](sycl::handler& handler) {
        sycl::local_accessor<float, 1> local_scores(
            sycl::range<1>(kDefaultBlockRows), handler);
        sycl::local_accessor<uint32_t, 1> local_ids(
            sycl::range<1>(kDefaultBlockRows), handler);
        handler.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(block_count_ * kDefaultBlockRows),
                              sycl::range<1>(kDefaultBlockRows)),
            [=](sycl::nd_item<1> item) {
            const size_t block_index = item.get_group_linear_id();
            const uint32_t lane = static_cast<uint32_t>(item.get_local_linear_id());
            const Block block = device_blocks[block_index];
            const size_t table_stride = static_cast<size_t>(pq_m) * ks;
            const float* table =
                device_tables + static_cast<size_t>(block.table_index) * table_stride;
            float score = kInfinity;
            uint32_t id = kInvalidId;
            if (lane < block.rows) {
              const uint8_t* code =
                  device_codes + block.code_offset + static_cast<uint64_t>(lane) * pq_m;
              score = 0.0f;
              bool valid = true;
              for (uint32_t m = 0; m < pq_m; ++m) {
                const uint32_t centroid = code[m];
                if (centroid >= ks) {
                  valid = false;
                  break;
                }
                score += table[static_cast<size_t>(m) * ks + centroid];
              }
              if (!valid || !sycl::isfinite(score)) {
                sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    error_ref(*error);
                error_ref.store(1);
                score = kInfinity;
              } else {
                id = block.first_id + lane;
              }
            }
            local_scores[lane] = score;
            local_ids[lane] = id;
            item.barrier(sycl::access::fence_space::local_space);

            for (uint32_t width = 2; width <= kDefaultBlockRows; width <<= 1u) {
              for (uint32_t stride = width >> 1u; stride > 0; stride >>= 1u) {
                const uint32_t peer = lane ^ stride;
                if (peer > lane) {
                  const bool ascending = (lane & width) == 0;
                  const float lhs_score = local_scores[lane];
                  const uint32_t lhs_id = local_ids[lane];
                  const float rhs_score = local_scores[peer];
                  const uint32_t rhs_id = local_ids[peer];
                  const bool rhs_better =
                      rhs_score < lhs_score || (rhs_score == lhs_score && rhs_id < lhs_id);
                  const bool lhs_better =
                      lhs_score < rhs_score || (lhs_score == rhs_score && lhs_id < rhs_id);
                  if ((ascending && rhs_better) || (!ascending && lhs_better)) {
                    local_scores[lane] = rhs_score;
                    local_ids[lane] = rhs_id;
                    local_scores[peer] = lhs_score;
                    local_ids[peer] = lhs_id;
                  }
                }
                item.barrier(sycl::access::fence_space::local_space);
              }
            }
            const size_t output = block_index * local_k;
            if (lane < local_k) {
              partial_scores[output + lane] = local_scores[lane];
              partial_ids[output + lane] = local_ids[lane];
            }
          });
      });

      const sycl::event score_readback_event =
          queue_.memcpy(staged_scores.data(), partial_scores_.get(),
                        partial_elements_ * sizeof(float));
      const sycl::event id_readback_event =
          queue_.memcpy(staged_ids.data(), partial_ids_.get(),
                        partial_elements_ * sizeof(uint32_t));
      sycl::event error_readback_event =
          queue_.memcpy(&staged_error, error_.get(), sizeof(staged_error));
      error_readback_event.wait_and_throw();

      const auto select_start = Clock::now();
      HostTopK selected{std::vector<float>(local_k_, kInfinity),
                        std::vector<uint32_t>(local_k_, kInvalidId)};
      for (size_t candidate = 0; candidate < partial_elements_; ++candidate) {
        const float score = staged_scores[candidate];
        const uint32_t id = staged_ids[candidate];
        if (id == kInvalidId) continue;
        if (!std::isfinite(score)) {
          staged_error = 1;
          break;
        }
        insert_topk(selected.scores, selected.ids, score, id);
      }
      result.timing.host_final_select_ms =
          std::chrono::duration<double, std::milli>(Clock::now() - select_start).count();
      const std::optional<double> reset_ms = event_ms(reset_event);
      const std::optional<double> table_upload_ms = event_ms(table_event);
      const std::optional<double> scan_select_ms = event_ms(scan_event);
      const std::optional<double> score_readback_ms = event_ms(score_readback_event);
      const std::optional<double> id_readback_ms = event_ms(id_readback_event);
      const std::optional<double> error_readback_ms = event_ms(error_readback_event);
      if (!reset_ms || !table_upload_ms || !scan_select_ms || !score_readback_ms ||
          !id_readback_ms || !error_readback_ms) {
        result.error = "queue_profiling_unavailable";
        return result;
      }
      result.timing.reset_ms = *reset_ms;
      result.timing.table_upload_ms = *table_upload_ms;
      result.timing.scan_select_ms = *scan_select_ms;
      result.timing.readback_ms =
          *score_readback_ms + *id_readback_ms + *error_readback_ms;
      result.timing.queue_submissions = 6;
      result.timing.host_synchronizations = 1;

      if (staged_error == 0) {
        for (uint32_t rank = 0; rank < local_k_; ++rank) {
          staged_scores[rank] = selected.scores[rank];
          staged_ids[rank] = selected.ids[rank];
        }
      }
    } catch (const sycl::exception& exception) {
      try {
        queue_.wait();
      } catch (...) {
      }
      result.error = std::string("sycl_exception: ") + exception.what();
      return result;
    } catch (const std::exception& exception) {
      try {
        queue_.wait();
      } catch (...) {
      }
      result.error = std::string("exception: ") + exception.what();
      return result;
    } catch (...) {
      try {
        queue_.wait();
      } catch (...) {
      }
      result.error = "unknown_exception";
      return result;
    }

    if (staged_error != 0) {
      result.error = "device_validation_failed";
      return result;
    }
    for (uint32_t rank = 0; rank < local_k_; ++rank) {
      if (staged_ids[rank] == kInvalidId || !std::isfinite(staged_scores[rank])) {
        result.error = "incomplete_or_non_finite_output";
        return result;
      }
      if (rank > 0 && !better(staged_scores[rank - 1], staged_ids[rank - 1],
                              staged_scores[rank], staged_ids[rank])) {
        result.error = "unordered_output";
        return result;
      }
      if (staged_ids[rank] != oracle.ids[rank] ||
          std::fabs(staged_scores[rank] - oracle.scores[rank]) > 1.0e-5f) {
        result.error = "oracle_mismatch";
        return result;
      }
    }

    for (uint32_t rank = 0; rank < local_k_; ++rank) {
      result.published.scores[rank + 1] = staged_scores[rank];
      result.published.ids[rank + 1] = staged_ids[rank];
    }
    result.success = result.published.scores.front() == -12345.0f &&
                     result.published.scores.back() == -12345.0f &&
                     result.published.ids.front() == 0xdeadbeefu &&
                     result.published.ids.back() == 0xdeadbeefu;
    if (!result.success) result.error = "publication_canary_changed";
    result.timing.wall_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - wall_start).count();
    const double attributed = result.timing.reset_ms + result.timing.table_upload_ms +
                              result.timing.scan_select_ms +
                              result.timing.host_final_select_ms +
                              result.timing.readback_ms;
    result.timing.unattributed_ms = std::max(0.0, result.timing.wall_ms - attributed);
    return result;
  }

  uint64_t persistent_upload_bytes() const {
    return static_cast<uint64_t>(code_elements_) * sizeof(uint8_t) +
           static_cast<uint64_t>(block_count_) * sizeof(Block);
  }

  uint64_t per_search_input_bytes() const {
    return static_cast<uint64_t>(table_elements_) * sizeof(float) + sizeof(int32_t);
  }

  uint64_t per_search_output_bytes() const {
    return static_cast<uint64_t>(partial_elements_) *
               (sizeof(float) + sizeof(uint32_t)) +
           sizeof(int32_t);
  }

  uint64_t full_score_readback_bytes_avoided(uint64_t candidate_count) const {
    const uint64_t full_scores = candidate_count * sizeof(float);
    const uint64_t partials = per_search_output_bytes();
    return full_scores > partials ? full_scores - partials : 0;
  }

  uint64_t expanded_i32_bytes_avoided(uint64_t candidate_count) const {
    return candidate_count * static_cast<uint64_t>(pq_m_) * sizeof(int32_t);
  }

  uint64_t peak_device_bytes() const {
    return static_cast<uint64_t>(table_elements_) * sizeof(float) +
           static_cast<uint64_t>(code_elements_) * sizeof(uint8_t) +
           static_cast<uint64_t>(block_count_) * sizeof(Block) +
           static_cast<uint64_t>(partial_elements_) *
               (sizeof(float) + sizeof(uint32_t)) +
           sizeof(int32_t);
  }

  double persistent_upload_ms() const { return persistent_upload_ms_; }

 private:
  sycl::queue& queue_;
  uint32_t pq_m_ = 0;
  uint32_t ks_ = 0;
  uint32_t local_k_ = 0;
  size_t table_elements_ = 0;
  size_t code_elements_ = 0;
  size_t block_count_ = 0;
  size_t partial_elements_ = 0;
  DeviceAllocation<float> tables_;
  DeviceAllocation<uint8_t> codes_;
  DeviceAllocation<Block> blocks_;
  DeviceAllocation<float> partial_scores_;
  DeviceAllocation<uint32_t> partial_ids_;
  DeviceAllocation<int32_t> error_;
  double persistent_upload_ms_ = 0.0;
};

struct HeteroSample {
  bool success = false;
  std::string error;
  double wall_ms = 0.0;
  double cpu_partition_ms = 0.0;
  double gpu_partition_ms = 0.0;
  double final_merge_ms = 0.0;
  double overlap_efficiency = 0.0;
};

struct HeteroLane {
  uint32_t requested_gpu_percent = 0;
  uint64_t gpu_rows = 0;
  uint64_t cpu_rows = 0;
  double actual_gpu_fraction = 0.0;
  bool success = false;
  std::string error;
  StageTiming cold_gpu{};
  uint64_t gpu_persistent_upload_bytes = 0;
  uint64_t gpu_per_search_input_bytes = 0;
  uint64_t gpu_per_search_output_bytes = 0;
  uint64_t gpu_peak_device_bytes = 0;
  std::vector<HeteroSample> samples;
};

HeteroLane run_hetero_lane(sycl::queue& queue, uint32_t requested_gpu_percent,
                           const std::vector<float>& tables,
                           const std::vector<uint8_t>& codes,
                           const std::vector<Task>& tasks, const HostTopK& global_oracle,
                           uint64_t candidate_count, uint32_t pq_m, uint32_t ks,
                           uint32_t local_k, uint32_t repeats) {
  HeteroLane lane;
  lane.requested_gpu_percent = requested_gpu_percent;
  lane.gpu_rows = candidate_count * requested_gpu_percent / 100u;
  lane.gpu_rows = std::max<uint64_t>(lane.gpu_rows, local_k);
  lane.gpu_rows = std::min<uint64_t>(lane.gpu_rows, candidate_count - local_k);
  lane.cpu_rows = candidate_count - lane.gpu_rows;
  lane.actual_gpu_fraction = static_cast<double>(lane.gpu_rows) / candidate_count;

  try {
    FixtureSlice gpu_fixture =
        slice_fixture(tables, codes, tasks, 0, lane.gpu_rows, pq_m, ks);
    FixtureSlice cpu_fixture =
        slice_fixture(tables, codes, tasks, lane.gpu_rows, candidate_count, pq_m, ks);
    std::string preflight_error;
    if (!preflight(gpu_fixture.tables, gpu_fixture.codes, gpu_fixture.tasks, pq_m, ks,
                   local_k, lane.gpu_rows, preflight_error) ||
        !preflight(cpu_fixture.tables, cpu_fixture.codes, cpu_fixture.tasks, pq_m, ks,
                   local_k, lane.cpu_rows, preflight_error)) {
      lane.error = std::string("partition_preflight_failed: ") + preflight_error;
      return lane;
    }
    const HostTopK gpu_oracle = scalar_oracle(gpu_fixture.tables, gpu_fixture.codes,
                                              gpu_fixture.tasks, pq_m, ks, local_k);
    const std::vector<Block> gpu_blocks =
        make_blocks(gpu_fixture.tasks, pq_m, kDefaultBlockRows);
    FusedGpuRunner gpu_runner(queue, gpu_fixture.tables, gpu_fixture.codes, gpu_blocks,
                              pq_m, ks, local_k);
    lane.gpu_persistent_upload_bytes = gpu_runner.persistent_upload_bytes();
    lane.gpu_per_search_input_bytes = gpu_runner.per_search_input_bytes();
    lane.gpu_per_search_output_bytes = gpu_runner.per_search_output_bytes();
    lane.gpu_peak_device_bytes = gpu_runner.peak_device_bytes();

    const RunResult cold = gpu_runner.run(gpu_fixture.tables, gpu_oracle);
    lane.cold_gpu = cold.timing;
    if (!cold.success) {
      lane.error = std::string("cold_gpu_failed: ") + cold.error;
      return lane;
    }
    const RunResult warmup = gpu_runner.run(gpu_fixture.tables, gpu_oracle);
    if (!warmup.success) {
      lane.error = std::string("gpu_warmup_failed: ") + warmup.error;
      return lane;
    }

    lane.samples.reserve(repeats);
    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
      HeteroSample sample;
      HostTopK cpu_result;
      double cpu_ms = 0.0;
      std::exception_ptr cpu_exception;
      const auto wall_start = Clock::now();
      std::jthread cpu_worker([&]() {
        try {
          const auto cpu_start = Clock::now();
          cpu_result = scalar_oracle(cpu_fixture.tables, cpu_fixture.codes,
                                     cpu_fixture.tasks, pq_m, ks, local_k);
          cpu_ms =
              std::chrono::duration<double, std::milli>(Clock::now() - cpu_start).count();
        } catch (...) {
          cpu_exception = std::current_exception();
        }
      });
      RunResult gpu_result = gpu_runner.run(gpu_fixture.tables, gpu_oracle);
      cpu_worker.join();
      sample.cpu_partition_ms = cpu_ms;
      sample.gpu_partition_ms = gpu_result.timing.wall_ms;
      if (cpu_exception) {
        sample.error = "cpu_partition_failed";
      } else if (!gpu_result.success) {
        sample.error = std::string("gpu_failed: ") + gpu_result.error;
      } else {
        const auto merge_start = Clock::now();
        HostTopK merged = merge_partition_topk(gpu_result, gpu_fixture.global_begin,
                                               cpu_result, cpu_fixture.global_begin,
                                               local_k);
        sample.final_merge_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - merge_start).count();
        sample.success = true;
        for (uint32_t rank = 0; rank < local_k; ++rank) {
          if (merged.ids[rank] != global_oracle.ids[rank] ||
              !std::isfinite(merged.scores[rank]) ||
              std::fabs(merged.scores[rank] - global_oracle.scores[rank]) > 1.0e-5f) {
            sample.success = false;
            sample.error = "global_oracle_mismatch";
            break;
          }
        }
      }
      sample.wall_ms =
          std::chrono::duration<double, std::milli>(Clock::now() - wall_start).count();
      if (sample.wall_ms > 0.0) {
        sample.overlap_efficiency =
            (sample.cpu_partition_ms + sample.gpu_partition_ms) / sample.wall_ms;
      }
      lane.samples.push_back(std::move(sample));
    }
    lane.success = std::all_of(lane.samples.begin(), lane.samples.end(),
                               [](const HeteroSample& sample) { return sample.success; });
    if (!lane.success) lane.error = "one_or_more_measured_samples_failed";
    return lane;
  } catch (const sycl::exception& exception) {
    lane.error = std::string("sycl_exception: ") + exception.what();
  } catch (const std::exception& exception) {
    lane.error = std::string("exception: ") + exception.what();
  } catch (...) {
    lane.error = "unknown_exception";
  }
  return lane;
}

double percentile(std::vector<double> samples, double quantile) {
  if (samples.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(samples.begin(), samples.end());
  const double position = quantile * static_cast<double>(samples.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return samples[lower] * (1.0 - fraction) + samples[upper] * fraction;
}

bool parse_positive(const char* text, uint64_t maximum, uint64_t& value) {
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0' || parsed == 0 || parsed > maximum) return false;
  value = static_cast<uint64_t>(parsed);
  return true;
}

std::string escape_json(const std::string& text) {
  std::ostringstream out;
  out << '"';
  for (unsigned char ch : text) {
    switch (ch) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (ch < 0x20) {
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

void append_timing(std::ostringstream& out, const StageTiming& timing) {
  out << "{\"wall_ms\":" << timing.wall_ms
      << ",\"reset_ms\":" << timing.reset_ms
      << ",\"table_upload_ms\":" << timing.table_upload_ms
      << ",\"scan_select_ms\":" << timing.scan_select_ms
      << ",\"host_final_select_ms\":" << timing.host_final_select_ms
      << ",\"readback_ms\":" << timing.readback_ms
      << ",\"unattributed_ms\":" << timing.unattributed_ms
      << ",\"queue_submissions\":" << timing.queue_submissions
      << ",\"host_synchronizations\":" << timing.host_synchronizations << '}';
}

void append_hetero_lane(std::ostringstream& out, const HeteroLane& lane,
                        double full_cpu_p50_ms) {
  std::vector<double> wall_samples;
  std::vector<double> cpu_samples;
  std::vector<double> gpu_samples;
  std::vector<double> merge_samples;
  std::vector<double> overlap_samples;
  for (const HeteroSample& sample : lane.samples) {
    if (!sample.success) continue;
    wall_samples.push_back(sample.wall_ms);
    cpu_samples.push_back(sample.cpu_partition_ms);
    gpu_samples.push_back(sample.gpu_partition_ms);
    merge_samples.push_back(sample.final_merge_ms);
    overlap_samples.push_back(sample.overlap_efficiency);
  }
  out << "{\"requested_gpu_percent\":" << lane.requested_gpu_percent
      << ",\"actual_gpu_fraction\":" << lane.actual_gpu_fraction
      << ",\"gpu_rows\":" << lane.gpu_rows << ",\"cpu_rows\":" << lane.cpu_rows
      << ",\"status\":" << escape_json(lane.success ? "success" : "failed")
      << ",\"reason\":" << (lane.error.empty() ? "null" : escape_json(lane.error))
      << ",\"first_partition_gpu_timing_jit_already_warm\":";
  append_timing(out, lane.cold_gpu);
  out << ",\"samples\":[";
  for (size_t index = 0; index < lane.samples.size(); ++index) {
    if (index) out << ',';
    const HeteroSample& sample = lane.samples[index];
    out << "{\"status\":" << escape_json(sample.success ? "success" : "failed")
        << ",\"reason\":"
        << (sample.error.empty() ? "null" : escape_json(sample.error))
        << ",\"wall_ms\":" << sample.wall_ms
        << ",\"cpu_partition_ms\":" << sample.cpu_partition_ms
        << ",\"gpu_partition_ms\":" << sample.gpu_partition_ms
        << ",\"final_union_merge_ms\":" << sample.final_merge_ms
        << ",\"overlap_efficiency\":" << sample.overlap_efficiency << '}';
  }
  out << "],\"summary\":{\"valid_repetitions\":" << wall_samples.size();
  if (!lane.success || wall_samples.empty()) {
    out << ",\"p50_ms\":null,\"p95_ms\":null,\"p99_ms\":null,"
        << "\"searches_per_second\":null,\"speedup_vs_full_scalar_oracle\":null,"
        << "\"cpu_partition_p50_ms\":null,\"gpu_partition_p50_ms\":null,"
        << "\"final_union_merge_p50_ms\":null,\"overlap_efficiency_p50\":null";
  } else {
    const double wall_p50 = percentile(wall_samples, 0.50);
    out << ",\"p50_ms\":" << wall_p50
        << ",\"p95_ms\":" << percentile(wall_samples, 0.95)
        << ",\"p99_ms\":" << percentile(wall_samples, 0.99)
        << ",\"searches_per_second\":" << 1000.0 / wall_p50
        << ",\"speedup_vs_full_scalar_oracle\":" << full_cpu_p50_ms / wall_p50
        << ",\"cpu_partition_p50_ms\":" << percentile(cpu_samples, 0.50)
        << ",\"gpu_partition_p50_ms\":" << percentile(gpu_samples, 0.50)
        << ",\"final_union_merge_p50_ms\":" << percentile(merge_samples, 0.50)
        << ",\"overlap_efficiency_p50\":" << percentile(overlap_samples, 0.50);
  }
  out << "},\"traffic\":{\"gpu_persistent_upload_bytes\":"
      << lane.gpu_persistent_upload_bytes
      << ",\"gpu_per_search_input_bytes\":" << lane.gpu_per_search_input_bytes
      << ",\"gpu_per_search_output_bytes\":" << lane.gpu_per_search_output_bytes
      << "},\"memory\":{\"gpu_peak_owned_device_bytes\":"
      << lane.gpu_peak_device_bytes << "}}";
}

}  // namespace

int main(int argc, char** argv) {
  uint64_t candidate_count = 131072;
  uint64_t local_k64 = 32;
  uint64_t task_count64 = 16;
  uint64_t repeats64 = 5;
  uint64_t pq_m64 = 8;
  uint64_t ks64 = 256;
  if ((argc > 1 && !parse_positive(argv[1], kInvalidId - 1, candidate_count)) ||
      (argc > 2 && !parse_positive(argv[2], kMaxLocalK, local_k64)) ||
      (argc > 3 && !parse_positive(argv[3], 4096, task_count64)) ||
      (argc > 4 && !parse_positive(argv[4], 1000, repeats64)) ||
      (argc > 5 && !parse_positive(argv[5], 256, pq_m64)) ||
      (argc > 6 && !parse_positive(argv[6], 256, ks64)) || argc > 7 ||
      repeats64 < 5 || task_count64 > candidate_count || local_k64 > candidate_count) {
    std::fprintf(stderr,
                 "usage: ovvs_pq_gpu_fused_bench [candidates] [local_k<=64] "
                 "[tasks] [repeats>=5] [pq_m] [ks<=256]\n");
    return 2;
  }
  if (candidate_count < 2u * local_k64) {
    std::fprintf(stderr,
                 "candidate count must be at least twice local_k for the CPU+iGPU split sweep\n");
    return 2;
  }
  const uint32_t local_k = static_cast<uint32_t>(local_k64);
  const uint32_t task_count = static_cast<uint32_t>(task_count64);
  const uint32_t repeats = static_cast<uint32_t>(repeats64);
  const uint32_t pq_m = static_cast<uint32_t>(pq_m64);
  const uint32_t ks = static_cast<uint32_t>(ks64);

  try {
    const std::vector<Task> tasks = make_tasks(candidate_count, task_count, pq_m);
    const std::vector<Block> blocks = make_blocks(tasks, pq_m, kDefaultBlockRows);
    size_t table_elements = 0;
    size_t code_elements = 0;
    if (!checked_multiply(static_cast<size_t>(task_count),
                          static_cast<size_t>(pq_m) * ks, table_elements) ||
        !checked_multiply(static_cast<size_t>(candidate_count), pq_m, code_elements)) {
      throw std::length_error("fixture size overflow");
    }
    std::vector<float> tables(table_elements);
    std::vector<uint8_t> codes(code_elements);
    for (uint32_t task = 0; task < task_count; ++task) {
      for (uint32_t m = 0; m < pq_m; ++m) {
        for (uint32_t code = 0; code < ks; ++code) {
          const uint32_t mixed =
              (code * 37u + task * 17u + m * 13u + code * code * 3u) % 1009u;
          tables[(static_cast<size_t>(task) * pq_m + m) * ks + code] =
              static_cast<float>(mixed) * 0.001f + static_cast<float>(m) * 0.00001f;
        }
      }
    }
    for (const Task& task : tasks) {
      for (uint32_t row = 0; row < task.rows; ++row) {
        const uint64_t global_row = static_cast<uint64_t>(task.first_id) + row;
        for (uint32_t m = 0; m < pq_m; ++m) {
          codes[task.code_offset + static_cast<uint64_t>(row) * pq_m + m] =
              static_cast<uint8_t>((global_row * 37u + m * 29u + task.table_index * 11u +
                                    (global_row >> 7u)) %
                                   ks);
        }
      }
    }

    std::string preflight_error;
    if (!preflight(tables, codes, tasks, pq_m, ks, local_k, candidate_count,
                   preflight_error)) {
      std::fprintf(stderr, "fixture preflight failed: %s\n", preflight_error.c_str());
      return 1;
    }

    const uint32_t negative_ks = std::min<uint32_t>(ks, 255);
    std::vector<Task> bad_tasks{{0, 1, 0, 0}};
    std::vector<float> bad_tables(static_cast<size_t>(pq_m) * negative_ks, 0.0f);
    std::vector<uint8_t> bad_row(pq_m, 0);
    bad_row[0] = static_cast<uint8_t>(negative_ks);
    std::string negative_error;
    const bool invalid_code_rejected =
        !preflight(bad_tables, bad_row, bad_tasks, pq_m, negative_ks, 1, 1,
                   negative_error) &&
        negative_error == "code_out_of_range";
    bad_tables[0] = std::numeric_limits<float>::quiet_NaN();
    negative_error.clear();
    const bool nonfinite_table_rejected =
        !preflight(bad_tables, std::vector<uint8_t>(pq_m, 0), bad_tasks, pq_m,
                   negative_ks, 1, 1, negative_error) &&
        negative_error == "non_finite_table";

    std::vector<double> cpu_samples;
    HostTopK oracle;
    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
      const auto start = Clock::now();
      HostTopK current = scalar_oracle(tables, codes, tasks, pq_m, ks, local_k);
      cpu_samples.push_back(
          std::chrono::duration<double, std::milli>(Clock::now() - start).count());
      if (repeat == 0) oracle = std::move(current);
    }

    std::string selection_reason;
    const std::optional<sycl::device> selected_device =
        select_intel_integrated_level_zero_gpu(selection_reason);
    if (!selected_device) {
      std::ostringstream unavailable;
      unavailable << "{\"schema\":\"ovvs.pq-gpu-fused-bakeoff.v1\","
                  << "\"requested_policy\":\"force_gpu\","
                  << "\"status\":\"device_unavailable\",\"reason\":"
                  << escape_json(selection_reason) << ",\"fallback\":false}\n";
      std::fputs(unavailable.str().c_str(), stdout);
      return 77;
    }
    const size_t max_work_group_size =
        selected_device->get_info<sycl::info::device::max_work_group_size>();
    const size_t max_work_item_size =
        selected_device->get_info<sycl::info::device::max_work_item_sizes<1>>()[0];
    const uint64_t local_mem_size =
        selected_device->get_info<sycl::info::device::local_mem_size>();
    const uint64_t required_local_mem =
        static_cast<uint64_t>(kDefaultBlockRows) *
        (sizeof(float) + sizeof(uint32_t));
    if (max_work_group_size < kDefaultBlockRows ||
        max_work_item_size < kDefaultBlockRows || local_mem_size < required_local_mem ||
        !selected_device->has(sycl::aspect::queue_profiling) ||
        !selected_device->has(sycl::aspect::usm_device_allocations)) {
      std::ostringstream unavailable;
      unavailable << "{\"schema\":\"ovvs.pq-gpu-fused-bakeoff.v1\","
                  << "\"requested_policy\":\"force_gpu\","
                  << "\"status\":\"device_unavailable\","
                  << "\"reason\":\"required work-group, local-memory, profiling, or device-USM capability missing\","
                  << "\"fallback\":false}\n";
      std::fputs(unavailable.str().c_str(), stdout);
      return 77;
    }

    std::unique_ptr<sycl::queue> queue;
    try {
      queue = std::make_unique<sycl::queue>(
          *selected_device,
          sycl::property_list{sycl::property::queue::in_order{},
                              sycl::property::queue::enable_profiling{}});
    } catch (const sycl::exception& exception) {
      std::ostringstream unavailable;
      unavailable << "{\"schema\":\"ovvs.pq-gpu-fused-bakeoff.v1\","
                  << "\"requested_policy\":\"force_gpu\","
                  << "\"status\":\"device_unavailable\",\"reason\":"
                  << escape_json(exception.what())
                  << ",\"fallback\":false}\n";
      std::fputs(unavailable.str().c_str(), stdout);
      return 77;
    }

    FusedGpuRunner runner(*queue, tables, codes, blocks, pq_m, ks, local_k);
    bool per_run_nonfinite_table_rejected = false;
    {
      std::vector<float> runtime_bad_tables = tables;
      runtime_bad_tables.back() = std::numeric_limits<float>::infinity();
      const RunResult runtime_bad_table_result = runner.run(runtime_bad_tables, oracle);
      per_run_nonfinite_table_rejected =
          !runtime_bad_table_result.success &&
          runtime_bad_table_result.error == "non_finite_table" &&
          runtime_bad_table_result.published.scores.front() == -12345.0f &&
          runtime_bad_table_result.published.scores.back() == -12345.0f &&
          runtime_bad_table_result.published.ids.front() == 0xdeadbeefu &&
          runtime_bad_table_result.published.ids.back() == 0xdeadbeefu;
    }
    const RunResult cold = runner.run(tables, oracle);
    if (!cold.success) {
      std::fprintf(stderr, "cold FORCE_GPU run failed without fallback: %s\n",
                   cold.error.c_str());
      return 1;
    }
    const RunResult warmup = runner.run(tables, oracle);
    if (!warmup.success) {
      std::fprintf(stderr, "warmup FORCE_GPU run failed without fallback: %s\n",
                   warmup.error.c_str());
      return 1;
    }

    std::vector<RunResult> measured;
    measured.reserve(repeats);
    std::vector<double> gpu_samples;
    double max_abs_error = 0.0;
    uint32_t overlap = local_k;
    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
      RunResult current = runner.run(tables, oracle);
      if (!current.success) {
        std::fprintf(stderr, "measured FORCE_GPU run failed without fallback: %s\n",
                     current.error.c_str());
        return 1;
      }
      uint32_t current_overlap = 0;
      for (uint32_t rank = 0; rank < local_k; ++rank) {
        const float actual = current.published.scores[rank + 1];
        max_abs_error = std::max(
            max_abs_error,
            static_cast<double>(std::fabs(actual - oracle.scores[rank])));
        if (std::find(oracle.ids.begin(), oracle.ids.end(),
                      current.published.ids[rank + 1]) != oracle.ids.end()) {
          ++current_overlap;
        }
      }
      overlap = std::min(overlap, current_overlap);
      gpu_samples.push_back(current.timing.wall_ms);
      measured.push_back(std::move(current));
    }

    const std::string device_name =
        queue->get_device().get_info<sycl::info::device::name>();
    const std::string driver_version =
        queue->get_device().get_info<sycl::info::device::driver_version>();
    const double gpu_p50 = percentile(gpu_samples, 0.50);
    const double cpu_p50 = percentile(cpu_samples, 0.50);
    const bool fail_closed_gate_passed =
        invalid_code_rejected && nonfinite_table_rejected &&
        per_run_nonfinite_table_rejected;
    const uint32_t hetero_fractions[] = {25, 50, 75};
    std::vector<HeteroLane> hetero_lanes;
    hetero_lanes.reserve(std::size(hetero_fractions));
    for (uint32_t fraction : hetero_fractions) {
      hetero_lanes.push_back(run_hetero_lane(
          *queue, fraction, tables, codes, tasks, oracle, candidate_count, pq_m, ks,
          local_k, repeats));
    }
    const bool hetero_gate_passed =
        std::all_of(hetero_lanes.begin(), hetero_lanes.end(),
                    [](const HeteroLane& lane) { return lane.success; });
    const bool overall_gate_passed = fail_closed_gate_passed && hetero_gate_passed;
    const std::string platform_name =
        queue->get_device().get_platform().get_info<sycl::info::platform::name>();

    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << "{\"schema\":\"ovvs.pq-gpu-fused-bakeoff.v1\","
        << "\"requested_policy\":\"force_gpu\",\"status\":"
        << (overall_gate_passed ? "\"success\"" : "\"failed\"") << ','
        << "\"fallback\":false,\"device\":" << escape_json(device_name)
        << ",\"driver\":" << escape_json(driver_version)
        << ",\"platform\":" << escape_json(platform_name)
        << ",\"device_capabilities\":{\"vendor_id\":32902,"
        << "\"host_unified_memory\":true,\"max_work_group_size\":"
        << max_work_group_size << ",\"max_work_item_size_1d\":"
        << max_work_item_size << ",\"local_mem_bytes\":" << local_mem_size
        << ",\"queue_profiling\":true,\"device_usm\":true}"
        << ",\"fixture\":{\"candidates\":" << candidate_count
        << ",\"tasks\":" << task_count << ",\"task_geometry\":\"variable_weighted\","
        << "\"pq_m\":" << pq_m << ",\"ks\":" << ks
        << ",\"local_k\":" << local_k << ",\"block_rows\":"
        << kDefaultBlockRows << ",\"blocks\":" << blocks.size()
        << ",\"table_formula\":\"((code*37+task*17+m*13+code*code*3)%1009)*0.001+m*0.00001\","
        << "\"code_formula\":\"(global_row*37+m*29+task*11+(global_row>>7))%ks\","
        << "\"task_rows\":[";
    for (size_t task = 0; task < tasks.size(); ++task) {
      if (task) out << ',';
      out << tasks[task].rows;
    }
    out << "]},"
        << "\"oracle\":{\"kind\":\"independent_scalar_fused_topk\","
        << "\"samples_ms\":[";
    for (size_t index = 0; index < cpu_samples.size(); ++index) {
      if (index) out << ',';
      out << cpu_samples[index];
    }
    out << "],\"p50_ms\":" << cpu_p50
        << ",\"p95_ms\":" << percentile(cpu_samples, 0.95)
        << ",\"p99_ms\":" << percentile(cpu_samples, 0.99) << "},"
        << "\"cold\":{\"scope\":\"first_kernel_submission_in_fresh_process_includes_jit\","
        << "\"timing\":";
    append_timing(out, cold.timing);
    out << "},\"warmup_calls\":1,\"warm_runs\":[";
    for (size_t index = 0; index < measured.size(); ++index) {
      if (index) out << ',';
      append_timing(out, measured[index].timing);
    }
    out << "],\"summary\":{\"repetitions\":" << repeats
        << ",\"p50_ms\":" << gpu_p50
        << ",\"p95_ms\":" << percentile(gpu_samples, 0.95)
        << ",\"p99_ms\":" << percentile(gpu_samples, 0.99)
        << ",\"searches_per_second\":" << 1000.0 / gpu_p50
        << ",\"candidates_per_second\":"
        << static_cast<double>(candidate_count) * 1000.0 / gpu_p50
        << ",\"speedup_vs_scalar_fused_oracle\":" << cpu_p50 / gpu_p50 << "},"
        << "\"correctness\":{\"finite\":true,\"canaries_preserved\":true,"
        << "\"exact_id_order\":true,\"topk_overlap\":" << overlap
        << ",\"recall_at_k\":" << static_cast<double>(overlap) / local_k
        << ",\"max_abs_score_error\":" << max_abs_error << "},"
        << "\"negative_preflight\":{\"invalid_code_rejected\":"
        << (invalid_code_rejected ? "true" : "false")
        << ",\"nonfinite_table_rejected\":"
        << (nonfinite_table_rejected ? "true" : "false")
        << ",\"per_run_nonfinite_table_rejected\":"
        << (per_run_nonfinite_table_rejected ? "true" : "false")
        << ",\"output_publication_before_validation\":false},"
        << "\"traffic\":{\"persistent_upload_bytes\":"
        << runner.persistent_upload_bytes()
        << ",\"persistent_upload_ms\":" << runner.persistent_upload_ms()
        << ",\"per_search_input_bytes\":" << runner.per_search_input_bytes()
        << ",\"per_search_output_bytes\":" << runner.per_search_output_bytes()
        << ",\"expanded_i32_bytes_avoided\":"
        << runner.expanded_i32_bytes_avoided(candidate_count)
        << ",\"full_score_readback_bytes_avoided\":"
        << runner.full_score_readback_bytes_avoided(candidate_count) << "},"
        << "\"memory\":{\"peak_owned_device_bytes\":" << runner.peak_device_bytes()
        << ",\"host_fixture_bytes\":"
        << (static_cast<uint64_t>(tables.size()) * sizeof(float) + codes.size() +
            static_cast<uint64_t>(tasks.size()) * sizeof(Task) +
            static_cast<uint64_t>(blocks.size()) * sizeof(Block))
        << ",\"host_readback_staging_bytes\":" << runner.per_search_output_bytes()
        << "},\"hetero_sweep\":[";
    for (size_t lane = 0; lane < hetero_lanes.size(); ++lane) {
      if (lane) out << ',';
      append_hetero_lane(out, hetero_lanes[lane], cpu_p50);
    }
    out << "],\"hetero_contract\":{\"execution\":\"concurrent_cpu_suffix_and_igpu_prefix\","
        << "\"partition_union_topk_exact\":true,\"cpu_hot_loop_fallback\":false,"
        << "\"npu_share\":0.0},"
        << "\"integration_status\":\"standalone_experiment_not_routed\"}\n";
    std::fputs(out.str().c_str(), stdout);
    return overall_gate_passed ? 0 : 1;
  } catch (const sycl::exception& exception) {
    std::fprintf(stderr, "SYCL failure (no fallback): %s\n", exception.what());
    return 1;
  } catch (const std::exception& exception) {
    std::fprintf(stderr, "failure: %s\n", exception.what());
    return 1;
  }
}
