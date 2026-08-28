#include "ovvs/ovvs.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr int32_t kPqM = 8;
constexpr int32_t kKs = 256;
constexpr int32_t kMaxTopK = 64;
constexpr int64_t kTileRows = 2048;
constexpr float kCanary = -12345.0f;

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

double percentile(std::vector<double> samples, double quantile) {
  if (samples.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(samples.begin(), samples.end());
  const double position = quantile * static_cast<double>(samples.size() - 1);
  const size_t lo = static_cast<size_t>(position);
  const size_t hi = std::min(lo + 1, samples.size() - 1);
  const double fraction = position - static_cast<double>(lo);
  return samples[lo] + (samples[hi] - samples[lo]) * fraction;
}

std::string json_string(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char ch : value) {
    switch (ch) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
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

bool parse_positive(const char* text, int64_t maximum, int64_t& value) {
  char* end = nullptr;
  const long long parsed = std::strtoll(text, &end, 10);
  if (end == text || *end != '\0' || parsed <= 0 || parsed > maximum) return false;
  value = static_cast<int64_t>(parsed);
  return true;
}

struct Pair {
  float score = std::numeric_limits<float>::infinity();
  int64_t id = -1;
};

bool pair_less(const Pair& lhs, const Pair& rhs) {
  return lhs.score < rhs.score || (lhs.score == rhs.score && lhs.id < rhs.id);
}

std::vector<Pair> topk_pairs(std::vector<Pair> candidates, int32_t k) {
  const size_t keep = std::min(candidates.size(), static_cast<size_t>(k));
  if (keep == 0) return {};
  std::partial_sort(candidates.begin(), candidates.begin() + keep, candidates.end(), pair_less);
  candidates.resize(keep);
  return candidates;
}

std::vector<float> scalar_scores(const std::vector<float>& tables,
                                 const std::vector<uint8_t>& codes,
                                 int64_t row_begin, int64_t rows) {
  std::vector<float> scores(static_cast<size_t>(rows));
  for (int64_t local = 0; local < rows; ++local) {
    const int64_t row = row_begin + local;
    float sum = 0.0f;
    for (int32_t m = 0; m < kPqM; ++m) {
      const uint8_t code = codes[static_cast<size_t>(row) * kPqM + m];
      sum += tables[static_cast<size_t>(m) * kKs + code];
    }
    scores[static_cast<size_t>(local)] = sum;
  }
  return scores;
}

std::vector<Pair> score_topk(const std::vector<float>& scores, int64_t row_begin, int32_t k) {
  std::vector<Pair> pairs(scores.size());
  for (size_t i = 0; i < scores.size(); ++i) {
    pairs[i] = {scores[i], row_begin + static_cast<int64_t>(i)};
  }
  return topk_pairs(std::move(pairs), k);
}

struct GpuSelection {
  bool success = false;
  std::string reason;
  std::vector<Pair> topk;
  double wall_ms = 0.0;
  uint64_t explicit_input_copy_bytes = 0;
  uint64_t partial_readback_bytes = 0;
};

class GpuSelector {
 public:
  GpuSelector(const std::vector<float>& tables, const std::vector<uint8_t>& codes,
              int64_t max_rows, int32_t topk)
      : queue_(sycl::gpu_selector_v,
               sycl::property_list{sycl::property::queue::enable_profiling{}}),
        max_rows_(max_rows), topk_(topk),
        max_tiles_(static_cast<size_t>((max_rows + kTileRows - 1) / kTileRows)) {
    try {
      if (topk <= 0 || topk > kMaxTopK) {
        throw std::invalid_argument("topk outside [1,64]");
      }
      const size_t code_count = static_cast<size_t>(max_rows) * kPqM;
      d_tables_ = sycl::malloc_device<float>(tables.size(), queue_);
      d_codes_ = sycl::malloc_device<uint8_t>(code_count, queue_);
      d_scores_ = sycl::malloc_device<float>(static_cast<size_t>(max_rows), queue_);
      d_partial_scores_ = sycl::malloc_device<float>(max_tiles_ * topk_, queue_);
      d_partial_ids_ = sycl::malloc_device<int64_t>(max_tiles_ * topk_, queue_);
      if (!d_tables_ || !d_codes_ || !d_scores_ || !d_partial_scores_ || !d_partial_ids_) {
        throw std::bad_alloc();
      }
      queue_.memcpy(d_tables_, tables.data(), tables.size() * sizeof(float));
      queue_.memcpy(d_codes_, codes.data(), code_count * sizeof(uint8_t));
      queue_.wait_and_throw();
      persistent_upload_bytes_ = tables.size() * sizeof(float) + code_count * sizeof(uint8_t);
    } catch (...) {
      release_noexcept();
      throw;
    }
  }

  GpuSelector(const GpuSelector&) = delete;
  GpuSelector& operator=(const GpuSelector&) = delete;

  ~GpuSelector() noexcept { release_noexcept(); }

  std::string device_name() const {
    return queue_.get_device().get_info<sycl::info::device::name>();
  }

  std::string device_vendor() const {
    return queue_.get_device().get_info<sycl::info::device::vendor>();
  }

  std::string driver_version() const {
    return queue_.get_device().get_info<sycl::info::device::driver_version>();
  }

  uint32_t vendor_id() const {
    return queue_.get_device().get_info<sycl::info::device::vendor_id>();
  }

  bool intel_gpu() const {
    return queue_.get_device().is_gpu() && vendor_id() == 0x8086u;
  }

  uint64_t persistent_upload_bytes() const { return persistent_upload_bytes_; }

  GpuSelection direct_adc_select(int64_t row_begin, int64_t rows) {
    GpuSelection result;
    if (rows <= 0) {
      result.success = true;
      return result;
    }
    if (row_begin < 0 || rows > max_rows_ - row_begin) {
      result.reason = "direct ADC range outside persistent code buffer";
      return result;
    }
    try {
      const auto started = Clock::now();
      const size_t tiles = static_cast<size_t>((rows + kTileRows - 1) / kTileRows);
      const uint8_t* codes = d_codes_;
      const float* tables = d_tables_;
      float* partial_scores = d_partial_scores_;
      int64_t* partial_ids = d_partial_ids_;
      const int32_t keep = topk_;
      auto event = queue_.parallel_for(sycl::range<1>(tiles), [=](sycl::id<1> item) {
        float best_scores[kMaxTopK];
        int64_t best_ids[kMaxTopK];
        for (int32_t slot = 0; slot < kMaxTopK; ++slot) {
          best_scores[slot] = std::numeric_limits<float>::infinity();
          best_ids[slot] = -1;
        }
        const int64_t local_begin = static_cast<int64_t>(item[0]) * kTileRows;
        const int64_t local_end =
            local_begin + kTileRows < rows ? local_begin + kTileRows : rows;
        for (int64_t local = local_begin; local < local_end; ++local) {
          const int64_t global = row_begin + local;
          float score = 0.0f;
          for (int32_t m = 0; m < kPqM; ++m) {
            const uint8_t code = codes[static_cast<size_t>(global) * kPqM + m];
            score += tables[static_cast<size_t>(m) * kKs + code];
          }
          int32_t insert = keep;
          for (int32_t slot = 0; slot < keep; ++slot) {
            if (score < best_scores[slot] ||
                (score == best_scores[slot] && global < best_ids[slot])) {
              insert = slot;
              break;
            }
          }
          if (insert < keep) {
            for (int32_t slot = keep - 1; slot > insert; --slot) {
              best_scores[slot] = best_scores[slot - 1];
              best_ids[slot] = best_ids[slot - 1];
            }
            best_scores[insert] = score;
            best_ids[insert] = global;
          }
        }
        const size_t output = item[0] * static_cast<size_t>(keep);
        for (int32_t slot = 0; slot < keep; ++slot) {
          partial_scores[output + slot] = best_scores[slot];
          partial_ids[output + slot] = best_ids[slot];
        }
      });
      result = finish_partials(event, tiles);
      result.wall_ms = elapsed_ms(started, Clock::now());
      return result;
    } catch (const std::exception& error) {
      drain_noexcept();
      result.reason = error.what();
      return result;
    } catch (...) {
      drain_noexcept();
      result.reason = "unknown SYCL direct ADC/select failure";
      return result;
    }
  }

 private:
  void drain_noexcept() noexcept {
    try { queue_.wait(); } catch (...) {}
  }

  void release_noexcept() noexcept {
    try { queue_.wait_and_throw(); } catch (...) {}
    try { if (d_partial_ids_) sycl::free(d_partial_ids_, queue_); } catch (...) {}
    d_partial_ids_ = nullptr;
    try { if (d_partial_scores_) sycl::free(d_partial_scores_, queue_); } catch (...) {}
    d_partial_scores_ = nullptr;
    try { if (d_scores_) sycl::free(d_scores_, queue_); } catch (...) {}
    d_scores_ = nullptr;
    try { if (d_codes_) sycl::free(d_codes_, queue_); } catch (...) {}
    d_codes_ = nullptr;
    try { if (d_tables_) sycl::free(d_tables_, queue_); } catch (...) {}
    d_tables_ = nullptr;
  }

 public:
  GpuSelection staged_score_select(const float* host_scores, int64_t row_begin, int64_t rows) {
    GpuSelection result;
    if (rows <= 0) {
      result.success = true;
      return result;
    }
    if (!host_scores || row_begin < 0 || rows > max_rows_ - row_begin) {
      result.reason = "invalid staged-score range";
      return result;
    }
    try {
      const auto started = Clock::now();
      const size_t bytes = static_cast<size_t>(rows) * sizeof(float);
      const auto copy = queue_.memcpy(d_scores_, host_scores, bytes);
      const size_t tiles = static_cast<size_t>((rows + kTileRows - 1) / kTileRows);
      const float* staged_scores = d_scores_;
      float* partial_scores = d_partial_scores_;
      int64_t* partial_ids = d_partial_ids_;
      const int32_t keep = topk_;
      const auto event = queue_.submit([&](sycl::handler& handler) {
        handler.depends_on(copy);
        handler.parallel_for(sycl::range<1>(tiles), [=](sycl::id<1> item) {
          float best_scores[kMaxTopK];
          int64_t best_ids[kMaxTopK];
          for (int32_t slot = 0; slot < kMaxTopK; ++slot) {
            best_scores[slot] = std::numeric_limits<float>::infinity();
            best_ids[slot] = -1;
          }
          const int64_t local_begin = static_cast<int64_t>(item[0]) * kTileRows;
          const int64_t local_end =
              local_begin + kTileRows < rows ? local_begin + kTileRows : rows;
          for (int64_t local = local_begin; local < local_end; ++local) {
            const int64_t global = row_begin + local;
            const float score = staged_scores[static_cast<size_t>(local)];
            int32_t insert = keep;
            for (int32_t slot = 0; slot < keep; ++slot) {
              if (score < best_scores[slot] ||
                  (score == best_scores[slot] && global < best_ids[slot])) {
                insert = slot;
                break;
              }
            }
            if (insert < keep) {
              for (int32_t slot = keep - 1; slot > insert; --slot) {
                best_scores[slot] = best_scores[slot - 1];
                best_ids[slot] = best_ids[slot - 1];
              }
              best_scores[insert] = score;
              best_ids[insert] = global;
            }
          }
          const size_t output = item[0] * static_cast<size_t>(keep);
          for (int32_t slot = 0; slot < keep; ++slot) {
            partial_scores[output + slot] = best_scores[slot];
            partial_ids[output + slot] = best_ids[slot];
          }
        });
      });
      result = finish_partials(event, tiles);
      result.explicit_input_copy_bytes = bytes;
      result.wall_ms = elapsed_ms(started, Clock::now());
      return result;
    } catch (const std::exception& error) {
      drain_noexcept();
      result.reason = error.what();
      return result;
    } catch (...) {
      drain_noexcept();
      result.reason = "unknown SYCL staged-score select failure";
      return result;
    }
  }

 private:
  GpuSelection finish_partials(const sycl::event& producer, size_t tiles) {
    GpuSelection result;
    const size_t count = tiles * static_cast<size_t>(topk_);
    std::vector<float> scores(count);
    std::vector<int64_t> ids(count);
    sycl::event score_copy;
    sycl::event id_copy;
    try {
      score_copy = queue_.submit([&](sycl::handler& handler) {
        handler.depends_on(producer);
        handler.memcpy(scores.data(), d_partial_scores_, count * sizeof(float));
      });
      id_copy = queue_.submit([&](sycl::handler& handler) {
        handler.depends_on(producer);
        handler.memcpy(ids.data(), d_partial_ids_, count * sizeof(int64_t));
      });
      score_copy.wait();
      id_copy.wait();
      score_copy.wait_and_throw();
      id_copy.wait_and_throw();
    } catch (...) {
      try { queue_.wait(); } catch (...) {}
      throw;
    }
    std::vector<Pair> candidates;
    candidates.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      if (ids[i] >= 0) candidates.push_back({scores[i], ids[i]});
    }
    result.topk = topk_pairs(std::move(candidates), topk_);
    result.partial_readback_bytes = count * (sizeof(float) + sizeof(int64_t));
    result.success = true;
    return result;
  }

  sycl::queue queue_;
  int64_t max_rows_ = 0;
  int32_t topk_ = 0;
  size_t max_tiles_ = 0;
  float* d_tables_ = nullptr;
  uint8_t* d_codes_ = nullptr;
  float* d_scores_ = nullptr;
  float* d_partial_scores_ = nullptr;
  int64_t* d_partial_ids_ = nullptr;
  uint64_t persistent_upload_bytes_ = 0;
};

struct Sample {
  double total_ms = 0.0;
  double concurrent_phase_ms = 0.0;
  double npu_ms = 0.0;
  double gpu_direct_ms = 0.0;
  double npu_to_gpu_ms = 0.0;
  double exact_rerank_ms = 0.0;
  double final_merge_ms = 0.0;
  double observed_overlap_ms = 0.0;
};

struct CpuGpuSample {
  double total_ms = 0.0;
  double concurrent_phase_ms = 0.0;
  double cpu_ms = 0.0;
  double gpu_ms = 0.0;
  double final_merge_ms = 0.0;
  double observed_overlap_ms = 0.0;
};

struct CpuGpuLane {
  int32_t gpu_percent = 0;
  int64_t cpu_rows = 0;
  int64_t gpu_rows = 0;
  int32_t topk = 0;
  std::string status = "failed";
  std::string reason;
  double first_call_ms = 0.0;
  std::vector<CpuGpuSample> samples;
  int32_t topk_overlap = 0;
  uint64_t cpu_compact_code_bytes = 0;
  uint64_t cpu_score_materialization_bytes = 0;
  uint64_t gpu_compact_code_bytes = 0;
  uint64_t gpu_partial_readback_bytes = 0;
};

struct Lane {
  int32_t npu_percent = 0;
  int64_t npu_rows = 0;
  int64_t gpu_rows = 0;
  std::string status = "failed";
  std::string reason;
  double first_call_ms = 0.0;
  std::vector<Sample> samples;
  bool has_failed_attempt = false;
  Sample failed_attempt;
  bool exact_union_rerank = false;
  bool finite = true;
  bool canaries = true;
  bool fully_published = true;
  double npu_max_abs_error = 0.0;
  double npu_max_relative_error = 0.0;
  int32_t topk_overlap = 0;
  int32_t topk = 0;
  uint64_t npu_compressed_code_bytes = 0;
  uint64_t npu_expanded_index_logical_bytes = 0;
  uint64_t npu_score_bytes = 0;
  uint64_t npu_to_gpu_bytes = 0;
  uint64_t gpu_compact_code_bytes = 0;
  uint64_t gpu_partial_readback_bytes = 0;
  int32_t calls_attempted = 0;
  int32_t npu_calls_succeeded = 0;
  int32_t staged_calls_completed = 0;
  int32_t full_pipeline_calls_completed = 0;
  int32_t warmup_calls_completed = 0;
  int32_t rerank_candidates = 0;
  uint64_t actual_npu_compressed_code_bytes = 0;
  uint64_t actual_npu_expanded_index_logical_bytes = 0;
  uint64_t actual_npu_score_bytes = 0;
  uint64_t actual_npu_to_gpu_bytes = 0;
  uint64_t actual_gpu_compact_code_bytes = 0;
  uint64_t actual_gpu_partial_readback_bytes = 0;
};

struct LaneContext {
  ovvsResources_t resources = nullptr;
  explicit LaneContext(bool needed) {
    if (!needed) return;
    if (ovvsResourcesCreate(&resources) != OVVS_STATUS_SUCCESS) {
      throw std::runtime_error("ovvsResourcesCreate failed");
    }
    const ovvsStatus status = ovvsResourcesSetPolicy(resources, OVVS_POLICY_FORCE_NPU);
    if (status != OVVS_STATUS_SUCCESS) {
      ovvsResourcesDestroy(resources);
      resources = nullptr;
      throw std::runtime_error("failed to set FORCE_NPU");
    }
  }
  ~LaneContext() { if (resources) ovvsResourcesDestroy(resources); }
  LaneContext(const LaneContext&) = delete;
  LaneContext& operator=(const LaneContext&) = delete;
};

struct TopkAssessment {
  int32_t overlap = 0;
  bool ordered_ids = false;
  bool scores_within_tolerance = false;
  double max_abs_error = 0.0;
};

TopkAssessment assess_topk(const std::vector<Pair>& actual,
                           const std::vector<Pair>& oracle,
                           double score_tolerance) {
  TopkAssessment result;
  for (const Pair& expected : oracle) {
    if (std::find_if(actual.begin(), actual.end(), [&](const Pair& value) {
          return value.id == expected.id;
        }) != actual.end()) {
      ++result.overlap;
    }
  }
  if (actual.size() != oracle.size()) return result;
  result.ordered_ids = true;
  result.scores_within_tolerance = true;
  for (size_t i = 0; i < oracle.size(); ++i) {
    result.ordered_ids = result.ordered_ids && actual[i].id == oracle[i].id;
    const double error = std::fabs(static_cast<double>(actual[i].score) - oracle[i].score);
    if (!std::isfinite(error)) {
      result.scores_within_tolerance = false;
    } else {
      result.max_abs_error = std::max(result.max_abs_error, error);
      result.scores_within_tolerance =
          result.scores_within_tolerance && error <= score_tolerance;
    }
  }
  return result;
}

struct CpuPartitionResult {
  std::vector<Pair> topk;
  double wall_ms = 0.0;
  std::string error;
};

class CpuPartitionWorker {
 public:
  CpuPartitionWorker(const std::vector<float>& tables,
                     const std::vector<uint8_t>& codes, int64_t rows,
                     int32_t topk)
      : tables_(tables), codes_(codes), rows_(rows), topk_(topk),
        worker_([this] { loop(); }) {}

  ~CpuPartitionWorker() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    task_ready_.notify_one();
    if (worker_.joinable()) worker_.join();
  }

  CpuPartitionWorker(const CpuPartitionWorker&) = delete;
  CpuPartitionWorker& operator=(const CpuPartitionWorker&) = delete;

  void start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (task_pending_ || result_ready_) {
      throw std::logic_error("CPU partition worker already has outstanding work");
    }
    task_pending_ = true;
    task_ready_.notify_one();
  }

  CpuPartitionResult finish() {
    std::unique_lock<std::mutex> lock(mutex_);
    result_ready_cv_.wait(lock, [&] { return result_ready_; });
    result_ready_ = false;
    return std::move(result_);
  }

 private:
  void loop() noexcept {
    while (true) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        task_ready_.wait(lock, [&] { return task_pending_ || stopping_; });
        if (stopping_ && !task_pending_) return;
        task_pending_ = false;
      }
      CpuPartitionResult next;
      const auto begin = Clock::now();
      try {
        const std::vector<float> scores = scalar_scores(tables_, codes_, 0, rows_);
        next.topk = score_topk(scores, 0, topk_);
      } catch (const std::exception& error) {
        next.error = error.what();
      } catch (...) {
        next.error = "unknown CPU partition failure";
      }
      next.wall_ms = elapsed_ms(begin, Clock::now());
      {
        std::lock_guard<std::mutex> lock(mutex_);
        result_ = std::move(next);
        result_ready_ = true;
      }
      result_ready_cv_.notify_one();
    }
  }

  const std::vector<float>& tables_;
  const std::vector<uint8_t>& codes_;
  int64_t rows_ = 0;
  int32_t topk_ = 0;
  std::mutex mutex_;
  std::condition_variable task_ready_;
  std::condition_variable result_ready_cv_;
  bool task_pending_ = false;
  bool result_ready_ = false;
  bool stopping_ = false;
  CpuPartitionResult result_;
  std::thread worker_;
};

CpuGpuLane run_cpu_gpu_lane(int32_t gpu_percent, int64_t rows, int32_t repeats,
                            const std::vector<float>& tables,
                            const std::vector<uint8_t>& codes,
                            const std::vector<Pair>& oracle_topk, GpuSelector& gpu) {
  CpuGpuLane lane;
  lane.gpu_percent = gpu_percent;
  lane.gpu_rows = rows * gpu_percent / 100;
  lane.cpu_rows = rows - lane.gpu_rows;
  lane.topk = static_cast<int32_t>(oracle_topk.size());
  lane.cpu_compact_code_bytes = static_cast<uint64_t>(lane.cpu_rows) * kPqM;
  lane.cpu_score_materialization_bytes =
      static_cast<uint64_t>(lane.cpu_rows) * sizeof(float);
  lane.gpu_compact_code_bytes = static_cast<uint64_t>(lane.gpu_rows) * kPqM;

  const int32_t total_runs = repeats + 2;
  std::unique_ptr<CpuPartitionWorker> worker;
  if (lane.cpu_rows > 0 && lane.gpu_rows > 0) {
    worker = std::make_unique<CpuPartitionWorker>(tables, codes, lane.cpu_rows,
                                                   lane.topk);
  }

  auto run_once = [&]() -> std::pair<bool, CpuGpuSample> {
    CpuGpuSample sample;
    std::vector<Pair> cpu_topk;
    const auto total_begin = Clock::now();
    const auto concurrent_begin = Clock::now();
    GpuSelection gpu_result;
    if (lane.cpu_rows > 0 && lane.gpu_rows > 0) {
      worker->start();
      gpu_result = gpu.direct_adc_select(lane.cpu_rows, lane.gpu_rows);
      sample.gpu_ms = gpu_result.wall_ms;
      CpuPartitionResult cpu_result = worker->finish();
      sample.cpu_ms = cpu_result.wall_ms;
      cpu_topk = std::move(cpu_result.topk);
      if (!cpu_result.error.empty()) {
        lane.reason = std::string("CPU partition failed: ") +
                      cpu_result.error;
        return {false, sample};
      }
    } else if (lane.cpu_rows > 0) {
      const auto begin = Clock::now();
      const std::vector<float> scores = scalar_scores(tables, codes, 0, lane.cpu_rows);
      cpu_topk = score_topk(scores, 0, lane.topk);
      sample.cpu_ms = elapsed_ms(begin, Clock::now());
      gpu_result.success = true;
    } else {
      gpu_result = gpu.direct_adc_select(0, lane.gpu_rows);
      sample.gpu_ms = gpu_result.wall_ms;
    }
    sample.concurrent_phase_ms = elapsed_ms(concurrent_begin, Clock::now());
    sample.observed_overlap_ms = std::max(
        0.0, sample.cpu_ms + sample.gpu_ms - sample.concurrent_phase_ms);
    if (!gpu_result.success) {
      lane.reason = std::string("iGPU partition failed: ") + gpu_result.reason;
      return {false, sample};
    }

    const auto merge_begin = Clock::now();
    std::vector<Pair> candidates = std::move(cpu_topk);
    candidates.insert(candidates.end(), gpu_result.topk.begin(), gpu_result.topk.end());
    const std::vector<Pair> merged = topk_pairs(std::move(candidates), lane.topk);
    sample.final_merge_ms = elapsed_ms(merge_begin, Clock::now());
    sample.total_ms = elapsed_ms(total_begin, Clock::now());
    lane.gpu_partial_readback_bytes = gpu_result.partial_readback_bytes;
    const TopkAssessment assessment = assess_topk(merged, oracle_topk, 1e-3);
    lane.topk_overlap = assessment.overlap;
    if (!assessment.ordered_ids || !assessment.scores_within_tolerance) {
      lane.reason = "CPU+iGPU merged shortlist does not match the CPU oracle";
      return {false, sample};
    }
    return {true, sample};
  };

  bool succeeded = true;
  std::vector<CpuGpuSample> all_samples;
  all_samples.reserve(static_cast<size_t>(total_runs));
  try {
    for (int32_t run = 0; run < total_runs; ++run) {
      auto result = run_once();
      succeeded = succeeded && result.first;
      all_samples.push_back(result.second);
    }
  } catch (const std::exception& error) {
    lane.reason = std::string("CPU+iGPU lane exception: ") + error.what();
    lane.status = "failed";
    return lane;
  } catch (...) {
    lane.reason = "unknown CPU+iGPU lane exception";
    lane.status = "failed";
    return lane;
  }
  lane.first_call_ms = all_samples.front().total_ms;
  if (all_samples.size() > 2) {
    lane.samples.assign(all_samples.begin() + 2, all_samples.end());
  }
  if (!succeeded) {
    lane.status = "failed";
    return lane;
  }
  lane.status = "success";
  return lane;
}

Lane run_lane(int32_t npu_percent, int64_t rows, int32_t repeats,
              const std::vector<float>& tables, const std::vector<uint8_t>& codes,
              const std::vector<float>& oracle_scores, const std::vector<Pair>& oracle_topk,
              GpuSelector& gpu, bool npu_available, bool exact_union_rerank) {
  Lane lane;
  lane.npu_percent = npu_percent;
  lane.exact_union_rerank = exact_union_rerank;
  lane.topk = static_cast<int32_t>(oracle_topk.size());
  lane.npu_rows = rows * npu_percent / 100;
  lane.gpu_rows = rows - lane.npu_rows;
  lane.npu_compressed_code_bytes = static_cast<uint64_t>(lane.npu_rows) * kPqM;
  lane.npu_expanded_index_logical_bytes =
      static_cast<uint64_t>(lane.npu_rows) * kPqM * sizeof(int32_t);
  lane.npu_score_bytes = static_cast<uint64_t>(lane.npu_rows) * sizeof(float);
  lane.npu_to_gpu_bytes = lane.npu_score_bytes;
  lane.gpu_compact_code_bytes = static_cast<uint64_t>(lane.gpu_rows) * kPqM;

  if (lane.npu_rows > 0 && !npu_available) {
    lane.status = "unavailable";
    lane.reason = "NPU share requested but the ovVS resource probe reports no NPU";
    return lane;
  }

  try {
    LaneContext context(lane.npu_rows > 0);
    std::vector<float> guarded_npu(static_cast<size_t>(lane.npu_rows) + 2, kCanary);
    auto execute = [&]() -> std::pair<bool, Sample> {
      ++lane.calls_attempted;
      Sample sample;
      guarded_npu.front() = kCanary;
      guarded_npu.back() = kCanary;
      std::fill(guarded_npu.begin() + (lane.npu_rows > 0 ? 1 : 0),
                guarded_npu.end() - (lane.npu_rows > 0 ? 1 : 0), kCanary);
      ovvsStatus npu_status = OVVS_STATUS_SUCCESS;
      GpuSelection direct;
      GpuSelection staged;
      const auto total_begin = Clock::now();
      const auto concurrent_begin = Clock::now();
      const auto failed = [&]() {
        sample.total_ms = elapsed_ms(total_begin, Clock::now());
        return std::pair<bool, Sample>{false, sample};
      };

      if (lane.npu_rows > 0 && lane.gpu_rows > 0) {
        std::barrier start_line(2);
        std::jthread npu_thread([&] {
          start_line.arrive_and_wait();
          const auto begin = Clock::now();
          npu_status = ovvsPqAdcBatch(context.resources, tables.data(), kPqM, kKs,
                                     codes.data(), lane.npu_rows,
                                     guarded_npu.data() + 1);
          sample.npu_ms = elapsed_ms(begin, Clock::now());
        });
        start_line.arrive_and_wait();
        direct = gpu.direct_adc_select(lane.npu_rows, lane.gpu_rows);
        sample.gpu_direct_ms = direct.wall_ms;
        npu_thread.join();
      } else if (lane.npu_rows > 0) {
        const auto begin = Clock::now();
        npu_status = ovvsPqAdcBatch(context.resources, tables.data(), kPqM, kKs,
                                   codes.data(), lane.npu_rows,
                                   guarded_npu.data() + 1);
        sample.npu_ms = elapsed_ms(begin, Clock::now());
        direct.success = true;
      } else {
        direct = gpu.direct_adc_select(0, lane.gpu_rows);
        sample.gpu_direct_ms = direct.wall_ms;
      }
      sample.concurrent_phase_ms = elapsed_ms(concurrent_begin, Clock::now());
      sample.observed_overlap_ms = std::max(
          0.0, sample.npu_ms + sample.gpu_direct_ms - sample.concurrent_phase_ms);
      if (npu_status != OVVS_STATUS_SUCCESS) {
        lane.reason = std::string("FORCE_NPU PQ-ADC failed: ") + ovvsStatusString(npu_status);
        return failed();
      }
      if (!direct.success) {
        lane.reason = std::string("direct iGPU ADC/select failed: ") + direct.reason;
        return failed();
      }
      if (lane.npu_rows > 0) {
        ++lane.npu_calls_succeeded;
        lane.actual_npu_compressed_code_bytes += lane.npu_compressed_code_bytes;
        lane.actual_npu_expanded_index_logical_bytes +=
            lane.npu_expanded_index_logical_bytes;
        lane.actual_npu_score_bytes += lane.npu_score_bytes;
      }
      if (lane.gpu_rows > 0) {
        lane.actual_gpu_compact_code_bytes += lane.gpu_compact_code_bytes;
        lane.actual_gpu_partial_readback_bytes += direct.partial_readback_bytes;
      }

      if (lane.npu_rows > 0) {
        staged = gpu.staged_score_select(guarded_npu.data() + 1, 0, lane.npu_rows);
        sample.npu_to_gpu_ms = staged.wall_ms;
        if (!staged.success) {
          lane.reason = std::string("NPU-to-iGPU score staging/select failed: ") + staged.reason;
          return failed();
        }
        ++lane.staged_calls_completed;
        lane.actual_npu_to_gpu_bytes += staged.explicit_input_copy_bytes;
        lane.actual_gpu_partial_readback_bytes += staged.partial_readback_bytes;
      } else {
        staged.success = true;
      }

      const auto merge_begin = Clock::now();
      std::vector<Pair> union_candidates = direct.topk;
      union_candidates.insert(union_candidates.end(), staged.topk.begin(), staged.topk.end());
      if (exact_union_rerank) {
        const auto rerank_begin = Clock::now();
        lane.rerank_candidates = static_cast<int32_t>(union_candidates.size());
        for (Pair& candidate : union_candidates) {
          if (candidate.id < 0 || candidate.id >= rows) {
            lane.reason = "device shortlist contains an out-of-range candidate ID";
            return failed();
          }
          float exact = 0.0f;
          for (int32_t m = 0; m < kPqM; ++m) {
            const uint8_t code =
                codes[static_cast<size_t>(candidate.id) * kPqM + m];
            exact += tables[static_cast<size_t>(m) * kKs + code];
          }
          candidate.score = exact;
        }
        sample.exact_rerank_ms = elapsed_ms(rerank_begin, Clock::now());
      }
      const std::vector<Pair> merged = topk_pairs(std::move(union_candidates),
                                                  static_cast<int32_t>(oracle_topk.size()));
      sample.final_merge_ms = elapsed_ms(merge_begin, Clock::now());
      sample.total_ms = elapsed_ms(total_begin, Clock::now());
      lane.gpu_partial_readback_bytes =
          direct.partial_readback_bytes + staged.partial_readback_bytes;

      lane.canaries = lane.canaries &&
                      (lane.npu_rows == 0 ||
                       (guarded_npu.front() == kCanary && guarded_npu.back() == kCanary));
      if (lane.npu_rows > 0) {
        for (int64_t row = 0; row < lane.npu_rows; ++row) {
          const float actual = guarded_npu[static_cast<size_t>(row + 1)];
          const float expected = oracle_scores[static_cast<size_t>(row)];
          lane.finite = lane.finite && std::isfinite(actual);
          lane.fully_published = lane.fully_published && actual != kCanary;
          if (std::isfinite(actual)) {
            const double error = std::fabs(static_cast<double>(actual) - expected);
            lane.npu_max_abs_error = std::max(lane.npu_max_abs_error, error);
            lane.npu_max_relative_error =
                std::max(lane.npu_max_relative_error,
                         error / std::max(1.0, std::fabs(static_cast<double>(expected))));
          }
        }
      }
      if (!lane.finite || !lane.canaries || !lane.fully_published) {
        lane.reason = "NPU output is non-finite, incomplete, or overwrote a canary";
        return failed();
      }
      const double score_tolerance = exact_union_rerank ? 1e-3 : 128.0;
      const TopkAssessment assessment = assess_topk(merged, oracle_topk,
                                                     score_tolerance);
      lane.topk_overlap = assessment.overlap;
      if (!assessment.ordered_ids || !assessment.scores_within_tolerance) {
        lane.reason = "merged device shortlist does not match the CPU oracle";
        return failed();
      }
      ++lane.full_pipeline_calls_completed;
      return {true, sample};
    };

    const auto first_begin = Clock::now();
    const auto first = execute();
    lane.first_call_ms = elapsed_ms(first_begin, Clock::now());
    if (!first.first) {
      lane.has_failed_attempt = true;
      lane.failed_attempt = first.second;
      lane.status = "failed";
      return lane;
    }
    const auto warm = execute();
    if (!warm.first) {
      lane.has_failed_attempt = true;
      lane.failed_attempt = warm.second;
      lane.status = "failed";
      return lane;
    }
    lane.warmup_calls_completed = 1;
    for (int32_t repeat = 0; repeat < repeats; ++repeat) {
      auto measured = execute();
      if (!measured.first) {
        lane.has_failed_attempt = true;
        lane.failed_attempt = measured.second;
        lane.status = "failed";
        return lane;
      }
      lane.samples.push_back(measured.second);
    }
    lane.status = "success";
    return lane;
  } catch (const std::exception& error) {
    lane.status = "failed";
    lane.reason = error.what();
    return lane;
  } catch (...) {
    lane.status = "failed";
    lane.reason = "unknown heterogeneous lane failure";
    return lane;
  }
}

void append_samples(std::ostringstream& out, const Lane& lane) {
  std::vector<double> total;
  std::vector<double> npu;
  std::vector<double> gpu;
  std::vector<double> handoff;
  std::vector<double> overlap;
  out << '[';
  for (size_t i = 0; i < lane.samples.size(); ++i) {
    if (i) out << ',';
    const Sample& sample = lane.samples[i];
    total.push_back(sample.total_ms);
    npu.push_back(sample.npu_ms);
    gpu.push_back(sample.gpu_direct_ms);
    handoff.push_back(sample.npu_to_gpu_ms);
    overlap.push_back(sample.observed_overlap_ms);
    out << "{\"total_ms\":" << sample.total_ms
        << ",\"concurrent_phase_ms\":" << sample.concurrent_phase_ms
        << ",\"npu_public_api_wall_ms\":" << sample.npu_ms
        << ",\"gpu_direct_pipeline_wall_ms\":" << sample.gpu_direct_ms
        << ",\"npu_to_gpu_staging_select_pipeline_wall_ms\":" << sample.npu_to_gpu_ms
        << ",\"cpu_exact_union_rerank_ms\":" << sample.exact_rerank_ms
        << ",\"final_merge_ms\":" << sample.final_merge_ms
        << ",\"observed_host_overlap_ms\":" << sample.observed_overlap_ms << '}';
  }
  out << "],\"p50_ms\":";
  if (total.empty()) {
    out << "null,\"p95_ms\":null,\"p99_ms\":null";
  } else {
    out << percentile(total, 0.50)
        << ",\"p95_ms\":" << percentile(total, 0.95)
        << ",\"p99_ms\":" << percentile(total, 0.99);
  }
}

void append_cpu_gpu_lane(std::ostringstream& out, const CpuGpuLane& lane) {
  std::vector<double> totals;
  out << "{\"lane\":\"scalar_cpu_parallel_intel_gpu_compact_adc_select\""
      << ",\"requested_igpu_percent\":" << lane.gpu_percent
      << ",\"actual_cpu_rows\":" << lane.cpu_rows
      << ",\"actual_igpu_rows\":" << lane.gpu_rows
      << ",\"status\":" << json_string(lane.status)
      << ",\"reason\":" << (lane.reason.empty() ? "null" : json_string(lane.reason))
      << ",\"first_process_call_ms\":" << lane.first_call_ms
      << ",\"warmup_calls\":1,\"samples\":[";
  for (size_t i = 0; i < lane.samples.size(); ++i) {
    if (i) out << ',';
    const CpuGpuSample& sample = lane.samples[i];
    totals.push_back(sample.total_ms);
    out << "{\"total_ms\":" << sample.total_ms
        << ",\"concurrent_phase_ms\":" << sample.concurrent_phase_ms
        << ",\"scalar_cpu_pipeline_wall_ms\":" << sample.cpu_ms
        << ",\"gpu_pipeline_wall_ms\":" << sample.gpu_ms
        << ",\"final_merge_ms\":" << sample.final_merge_ms
        << ",\"observed_host_overlap_ms\":" << sample.observed_overlap_ms << '}';
  }
  out << "],\"p50_ms\":";
  if (totals.empty()) {
    out << "null,\"p95_ms\":null,\"p99_ms\":null";
  } else {
    out << percentile(totals, 0.50)
        << ",\"p95_ms\":" << percentile(totals, 0.95)
        << ",\"p99_ms\":" << percentile(totals, 0.99);
  }
  out << ",\"correctness\":{\"oracle_gate_passed\":"
      << (lane.status == "success" ? "true" : "false")
      << ",\"ordered_ids_and_scores_checked\":true,\"score_abs_tolerance\":0.001"
      << ",\"topk_overlap\":" << lane.topk_overlap
      << ",\"topk\":" << lane.topk
      << "},\"planned_bytes_per_call\":{\"cpu_compact_code_reads_logical\":"
      << lane.cpu_compact_code_bytes
      << ",\"cpu_score_materialization\":" << lane.cpu_score_materialization_bytes
      << ",\"igpu_compact_code_reads_logical\":" << lane.gpu_compact_code_bytes
      << ",\"igpu_partial_readback\":" << lane.gpu_partial_readback_bytes << "}}";
}

void append_lane(std::ostringstream& out, const Lane& lane) {
  out << "{\"lane\":"
      << json_string(lane.exact_union_rerank
                         ? "npu_parallel_intel_gpu_host_stage_cpu_exact_union_rerank"
                         : "npu_parallel_intel_gpu_raw_score_direct_merge_negative")
      << ",\"requested_npu_percent\":" << lane.npu_percent
      << ",\"actual_npu_rows\":" << lane.npu_rows
      << ",\"actual_igpu_rows\":" << lane.gpu_rows
      << ",\"status\":" << json_string(lane.status)
      << ",\"reason\":" << (lane.reason.empty() ? "null" : json_string(lane.reason))
      << ",\"first_process_call_ms\":" << lane.first_call_ms
      << ",\"calls_attempted\":" << lane.calls_attempted
      << ",\"npu_calls_succeeded\":" << lane.npu_calls_succeeded
      << ",\"staged_calls_completed\":" << lane.staged_calls_completed
      << ",\"full_pipeline_calls_completed\":" << lane.full_pipeline_calls_completed
      << ",\"warmup_calls_completed\":" << lane.warmup_calls_completed
      << ",\"local_shortlist_per_partition\":" << lane.topk
      << ",\"cpu_exact_rerank_candidates\":" << lane.rerank_candidates
      << ",\"samples\":";
  append_samples(out, lane);
  if (lane.has_failed_attempt) {
    out << ",\"failed_attempt\":{\"total_ms\":" << lane.failed_attempt.total_ms
        << ",\"concurrent_phase_ms\":" << lane.failed_attempt.concurrent_phase_ms
        << ",\"npu_public_api_wall_ms\":" << lane.failed_attempt.npu_ms
        << ",\"gpu_direct_pipeline_wall_ms\":" << lane.failed_attempt.gpu_direct_ms
        << ",\"npu_to_gpu_staging_select_pipeline_wall_ms\":"
        << lane.failed_attempt.npu_to_gpu_ms
        << ",\"cpu_exact_union_rerank_ms\":" << lane.failed_attempt.exact_rerank_ms
        << '}';
  } else {
    out << ",\"failed_attempt\":null";
  }
  out << ",\"correctness\":{\"oracle_gate_passed\":"
      << (lane.status == "success" ? "true" : "false")
      << ",\"ordered_ids_checked\":true,\"score_abs_tolerance\":"
      << (lane.exact_union_rerank ? 0.001 : 128.0)
      << ",\"finite\":" << (lane.finite ? "true" : "false")
      << ",\"canaries_preserved\":" << (lane.canaries ? "true" : "false")
      << ",\"all_outputs_published\":" << (lane.fully_published ? "true" : "false")
      << ",\"npu_max_abs_error\":";
  if (!lane.finite || lane.npu_calls_succeeded == 0) {
    out << "null,\"npu_max_relative_error\":null";
  } else {
    out << lane.npu_max_abs_error
        << ",\"npu_max_relative_error\":" << lane.npu_max_relative_error;
  }
  out << ",\"topk_overlap\":" << lane.topk_overlap << ",\"topk\":" << lane.topk
      << "},\"planned_bytes_per_full_call\":{\"npu_compressed_codes\":" << lane.npu_compressed_code_bytes
      << ",\"npu_expanded_indices_logical_lower_bound\":"
      << lane.npu_expanded_index_logical_bytes
      << ",\"npu_full_scores\":" << lane.npu_score_bytes
      << ",\"npu_to_igpu_explicit_staging\":" << lane.npu_to_gpu_bytes
      << ",\"igpu_compact_code_reads_logical\":" << lane.gpu_compact_code_bytes
      << ",\"igpu_partial_readback\":" << lane.gpu_partial_readback_bytes
      << "},\"actual_bytes_all_attempts\":{\"npu_compressed_codes\":"
      << lane.actual_npu_compressed_code_bytes
      << ",\"npu_expanded_indices_logical_lower_bound\":"
      << lane.actual_npu_expanded_index_logical_bytes
      << ",\"npu_full_scores\":" << lane.actual_npu_score_bytes
      << ",\"npu_to_igpu_explicit_staging\":" << lane.actual_npu_to_gpu_bytes
      << ",\"igpu_compact_code_reads_logical\":" << lane.actual_gpu_compact_code_bytes
      << ",\"igpu_partial_readback\":" << lane.actual_gpu_partial_readback_bytes
      << "}}";
}

}  // namespace

int main(int argc, char** argv) {
  int64_t rows = 131072;
  int64_t repeats64 = 5;
  int64_t topk64 = 32;
  if ((argc > 1 && !parse_positive(argv[1], 8 * 1024 * 1024, rows)) ||
      (argc > 2 && !parse_positive(argv[2], 1000, repeats64)) ||
      (argc > 3 && !parse_positive(argv[3], kMaxTopK, topk64)) || argc > 5) {
    std::cerr << "usage: ovvs_hetero_ivfpq_bench [rows<=8388608] [repeats] "
                 "[topk<=64] [output.json]\n";
    return 2;
  }
  const int32_t repeats = static_cast<int32_t>(repeats64);
  const int32_t topk = static_cast<int32_t>(topk64);
  const std::string output_path = argc > 4 ? argv[4] : "";
  const auto emit = [&](const std::string& json) {
    if (output_path.empty()) {
      std::cout << json;
      return true;
    }
    std::ofstream file(output_path, std::ios::binary | std::ios::trunc);
    if (!file) return false;
    file.write(json.data(), static_cast<std::streamsize>(json.size()));
    return static_cast<bool>(file);
  };

  if (static_cast<uint64_t>(rows) >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / kPqM) {
    std::cerr << "shape overflows host size_t\n";
    return 2;
  }

  std::ostringstream output;
  output << std::fixed << std::setprecision(6);
  try {
  std::vector<float> tables(static_cast<size_t>(kPqM) * kKs);
  std::vector<uint8_t> codes(static_cast<size_t>(rows) * kPqM);
  for (int32_t m = 0; m < kPqM; ++m) {
    const float base = 20000.0f + 1000.0f * static_cast<float>(m);
    for (int32_t code = 0; code < kKs; ++code) {
      tables[static_cast<size_t>(m) * kKs + code] =
          base + 50.0f * static_cast<float>(code);
    }
  }
  for (int64_t row = 0; row < rows; ++row) {
    for (int32_t m = 0; m < kPqM; ++m) {
      codes[static_cast<size_t>(row) * kPqM + m] =
          static_cast<uint8_t>((row * 37 + m * 29) % kKs);
    }
  }

  const auto oracle_begin = Clock::now();
  const std::vector<float> oracle_scores = scalar_scores(tables, codes, 0, rows);
  const std::vector<Pair> oracle_topk = score_topk(oracle_scores, 0, topk);
  const double oracle_ms = elapsed_ms(oracle_begin, Clock::now());

  ovvsResources_t probe = nullptr;
  int32_t npu_available = 0;
  char sku[128] = {};
  std::string probe_error;
  if (ovvsResourcesCreate(&probe) == OVVS_STATUS_SUCCESS) {
    const ovvsStatus available_status =
        ovvsResourcesNpuAvailable(probe, &npu_available);
    const ovvsStatus sku_status =
        ovvsResourcesSku(probe, sku, static_cast<int32_t>(sizeof(sku)));
    if (available_status != OVVS_STATUS_SUCCESS || sku_status != OVVS_STATUS_SUCCESS) {
      npu_available = 0;
      probe_error = "NPU availability or SKU probe failed";
    }
    ovvsResourcesDestroy(probe);
  } else {
    probe_error = "ovvsResourcesCreate failed for device probe";
  }

    GpuSelector gpu(tables, codes, rows, topk);
    if (!gpu.intel_gpu()) {
      throw std::runtime_error("selected SYCL GPU is not an Intel GPU");
    }
    const std::array<int32_t, 5> shares = {0, 25, 50, 75, 100};
    std::vector<CpuGpuLane> cpu_gpu_lanes;
    cpu_gpu_lanes.reserve(shares.size());
    bool cpu_gpu_success = true;
    for (const int32_t share : shares) {
      CpuGpuLane lane = run_cpu_gpu_lane(share, rows, repeats, tables, codes,
                                         oracle_topk, gpu);
      cpu_gpu_success = cpu_gpu_success && lane.status == "success";
      cpu_gpu_lanes.push_back(std::move(lane));
    }
    std::vector<Lane> lanes;
    lanes.reserve(shares.size());
    bool all_success = cpu_gpu_success;
    for (const int32_t share : shares) {
      Lane lane = run_lane(share, rows, repeats, tables, codes, oracle_scores,
                           oracle_topk, gpu, npu_available != 0, false);
      all_success = all_success && lane.status == "success";
      lanes.push_back(std::move(lane));
    }
    const std::array<int32_t, 3> mixed_shares = {25, 50, 75};
    std::vector<Lane> exact_rerank_lanes;
    exact_rerank_lanes.reserve(mixed_shares.size());
    for (const int32_t share : mixed_shares) {
      Lane lane = run_lane(share, rows, repeats, tables, codes, oracle_scores,
                           oracle_topk, gpu, npu_available != 0, true);
      all_success = all_success && lane.status == "success";
      exact_rerank_lanes.push_back(std::move(lane));
    }

    output << "{\"schema_version\":2,\"status\":"
           << json_string(all_success ? "complete" : "partial")
           << ",\"experiment\":\"ivfpq_npu_igpu_partition_and_handoff\""
           << ",\"sku\":" << json_string(sku)
           << ",\"fixture\":{\"rows\":" << rows
           << ",\"pq_m\":" << kPqM << ",\"ks\":" << kKs
           << ",\"topk\":" << topk
           << ",\"table\":\"scale_0.588_affine_fixture\"}"
           << ",\"device_probe\":{\"npu_available\":"
           << (npu_available ? "true" : "false")
           << ",\"sycl_gpu_available\":true,\"selected_gpu\":{\"name\":"
           << json_string(gpu.device_name())
           << ",\"vendor\":" << json_string(gpu.device_vendor())
           << ",\"vendor_id\":" << gpu.vendor_id()
           << ",\"driver_version\":" << json_string(gpu.driver_version())
           << ",\"intel_gpu\":true,\"integrated_classification\":\"not asserted by the generic SYCL selector; correlate with the reported Arrow Lake SKU\"}"
           << ",\"error\":"
           << (probe_error.empty() ? "null" : json_string(probe_error)) << "}"
           << ",\"measurement_contract\":{"
           << "\"cold_compile\":\"first process call is reported but share lanes reuse the process-wide graph cache; it is not an isolated cold comparison\","
           << "\"warm\":\"one untimed warmup after the first call, followed by complete pipeline samples\","
           << "\"cross_device\":\"OpenVINO NPU completion is host-fenced; scores are explicitly copied into a distinct Intel GPU buffer; no cross-driver zero-copy is claimed\","
           << "\"selection\":\"experimental one-work-item-per-2048-row tile partial top-k followed by CPU merge; not the production fused kernel\","
           << "\"timing_scope\":\"NPU is complete public API wall; GPU pipeline walls include submission, kernel, readback, and CPU tile-partial merge; no device-execution timing is claimed\","
           << "\"cpu_igpu_worker\":\"mixed CPU+iGPU lanes use one persistent CPU worker for first, warmup, and measured calls\","
           << "\"npu_worker\":\"the diagnostic mixed lane creates one host thread per invocation and includes that orchestration in full wall\","
           << "\"cpu_baseline_scope\":\"scalar fixture oracle and partition probe, not the production FORCE_CPU competitor\","
           << "\"tails\":\"p95 and p99 are interpolated diagnostics over the requested raw samples and are not promotion-grade with five repeats\","
           << "\"depth\":1,\"failed_lanes_retained\":true}"
           << ",\"cpu_oracle\":{\"wall_ms\":" << oracle_ms
           << ",\"finite\":true,\"full_scores_materialized\":true}"
           << ",\"gpu_persistent_upload_bytes\":" << gpu.persistent_upload_bytes()
           << ",\"cpu_igpu_lanes\":[";
    for (size_t i = 0; i < cpu_gpu_lanes.size(); ++i) {
      if (i) output << ',';
      append_cpu_gpu_lane(output, cpu_gpu_lanes[i]);
    }
    output << "],\"npu_gpu_raw_direct_merge_lanes\":[";
    for (size_t i = 0; i < lanes.size(); ++i) {
      if (i) output << ',';
      append_lane(output, lanes[i]);
    }
    output << "],\"npu_gpu_exact_union_rerank_lanes\":[";
    for (size_t i = 0; i < exact_rerank_lanes.size(); ++i) {
      if (i) output << ',';
      append_lane(output, exact_rerank_lanes[i]);
    }
    output << "]}\n";
    if (!emit(output.str())) {
      std::cerr << "failed to write JSON artifact: " << output_path << '\n';
      return 3;
    }
    return 0;
  } catch (const std::exception& error) {
    std::ostringstream failed;
    failed << "{\"schema_version\":2,\"status\":\"failed\","
           << "\"experiment\":\"ivfpq_npu_igpu_partition_and_handoff\","
           << "\"reason\":" << json_string(error.what()) << "}\n";
    if (!emit(failed.str())) return 3;
    return 1;
  } catch (...) {
    const std::string failed =
        "{\"schema_version\":2,\"status\":\"failed\","
        "\"experiment\":\"ivfpq_npu_igpu_partition_and_handoff\","
        "\"reason\":\"unknown tool failure\"}\n";
    if (!emit(failed)) return 3;
    return 1;
  }
}
