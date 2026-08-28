// Copyright (C) 2026 ovVS contributors
// SPDX-License-Identifier: Apache-2.0

#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <intrin.h>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset8.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
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

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kDim = 128;
constexpr std::uint32_t kPqM = 16;
constexpr std::uint32_t kSubDim = kDim / kPqM;
constexpr std::uint32_t kKs = 256;
constexpr std::uint32_t kLists = 32;
constexpr std::uint32_t kTopK = 32;
constexpr std::uint32_t kBlockRows = 256;
constexpr std::uint32_t kInvalidId = std::numeric_limits<std::uint32_t>::max();
constexpr float kInfinity = std::numeric_limits<float>::infinity();

static_assert(kDim % kPqM == 0);
static_assert((kBlockRows & (kBlockRows - 1)) == 0);

class Failure : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::size_t checked_mul(std::size_t a, std::size_t b, std::string_view what) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
    throw Failure(std::string(what) + " size overflow");
  }
  return a * b;
}

std::string json_escape(std::string_view value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    switch (c) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<unsigned>(c) << std::dec << std::setfill(' ');
        } else {
          out << static_cast<char>(c);
        }
    }
  }
  out << '"';
  return out.str();
}

std::uint64_t fnv1a(const void* data, std::size_t bytes) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  std::uint64_t hash = 1469598103934665603ull;
  for (std::size_t i = 0; i < bytes; ++i) {
    hash ^= p[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

std::string hex_u64(std::uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
  return out.str();
}

std::uint64_t process_peak_working_set() {
  PROCESS_MEMORY_COUNTERS_EX counters{};
  counters.cb = sizeof(counters);
  if (!GetProcessMemoryInfo(GetCurrentProcess(),
                            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                            sizeof(counters))) {
    return 0;
  }
  return static_cast<std::uint64_t>(counters.PeakWorkingSetSize);
}

std::string cpu_brand() {
  std::array<int, 4> regs{};
  __cpuid(regs.data(), 0x80000000);
  if (static_cast<unsigned>(regs[0]) < 0x80000004u) return "unknown x86 host CPU";
  std::array<int, 12> brand{};
  __cpuid(brand.data() + 0, 0x80000002);
  __cpuid(brand.data() + 4, 0x80000003);
  __cpuid(brand.data() + 8, 0x80000004);
  std::string value(reinterpret_cast<const char*>(brand.data()), 48);
  while (!value.empty() && (value.back() == '\0' || value.back() == ' ')) value.pop_back();
  const auto first = value.find_first_not_of(' ');
  return first == std::string::npos ? "unknown x86 host CPU" : value.substr(first);
}

struct Options {
  std::size_t candidates = 131072;
  int repeats = 5;
  int warmups = 1;
  std::filesystem::path cache_base = "out/npu-escape-cache/pq-lut-factor";
  std::optional<std::filesystem::path> output;
};

std::size_t parse_positive(const char* text, std::string_view option) {
  if (!text || !*text || *text == '-') throw Failure(std::string(option) + " needs a positive integer");
  std::size_t used = 0;
  unsigned long long value = 0;
  try {
    value = std::stoull(text, &used, 10);
  } catch (...) {
    throw Failure(std::string(option) + " needs a positive integer");
  }
  if (used != std::strlen(text) || value == 0 || value > std::numeric_limits<std::size_t>::max()) {
    throw Failure(std::string(option) + " is out of range");
  }
  return static_cast<std::size_t>(value);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> const char* {
      if (++i >= argc) throw Failure(arg + " needs a value");
      return argv[i];
    };
    if (arg == "--candidates") {
      options.candidates = parse_positive(next(), arg);
    } else if (arg == "--repeats") {
      options.repeats = static_cast<int>(parse_positive(next(), arg));
    } else if (arg == "--warmups") {
      options.warmups = static_cast<int>(parse_positive(next(), arg));
    } else if (arg == "--cache-base") {
      options.cache_base = next();
    } else if (arg == "--output") {
      options.output = std::filesystem::path(next());
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: ovvs_pq_lut_factor_bench [--candidates N] [--repeats N>=5] "
                   "[--warmups N] [--cache-base PATH] [--output NEW_PATH]\n";
      std::exit(0);
    } else {
      throw Failure("unknown option: " + arg);
    }
  }
  if (options.repeats < 5 || options.repeats > 1000 || options.warmups > 100 ||
      options.candidates <= kTopK || options.candidates > 0x00ffffffu) {
    throw Failure("invalid measurement geometry (repeats >=5; fixture candidates must fit 24-bit code)");
  }
  checked_mul(options.candidates, kPqM, "candidate codes");
  return options;
}

bool better(float score, std::uint32_t id, float other_score, std::uint32_t other_id) {
  return score < other_score || (score == other_score && id < other_id);
}

struct TopK {
  std::array<float, kTopK> scores{};
  std::array<std::uint32_t, kTopK> ids{};

  TopK() {
    scores.fill(kInfinity);
    ids.fill(kInvalidId);
  }
};

void insert_topk(TopK& topk, float score, std::uint32_t id) {
  if (!better(score, id, topk.scores.back(), topk.ids.back())) return;
  std::size_t pos = kTopK - 1;
  while (pos > 0 && better(score, id, topk.scores[pos - 1], topk.ids[pos - 1])) {
    topk.scores[pos] = topk.scores[pos - 1];
    topk.ids[pos] = topk.ids[pos - 1];
    --pos;
  }
  topk.scores[pos] = score;
  topk.ids[pos] = id;
}

bool equal_topk(const TopK& actual, const TopK& expected, float tolerance,
                float* max_abs_error = nullptr) {
  float max_error = 0.0f;
  for (std::size_t rank = 0; rank < kTopK; ++rank) {
    if (!std::isfinite(actual.scores[rank]) || actual.ids[rank] != expected.ids[rank]) return false;
    max_error = std::max(max_error, std::fabs(actual.scores[rank] - expected.scores[rank]));
  }
  if (max_abs_error) *max_abs_error = max_error;
  return max_error <= tolerance;
}

struct ListSpan {
  std::uint32_t first_id = 0;
  std::uint32_t rows = 0;
};

struct Block {
  std::uint64_t code_offset = 0;
  std::uint32_t rows = 0;
  std::uint32_t list = 0;
  std::uint32_t first_id = 0;
  std::uint32_t reserved = 0;
};

struct Pair {
  float score = kInfinity;
  std::uint32_t id = kInvalidId;
};

struct FactorizationGate {
  bool finite = true;
  bool bit_equal = true;
  bool exact_topk = true;
  std::uint64_t compared_scores = 0;
  std::uint64_t bit_equal_scores = 0;
  float max_abs_error = 0.0f;
};

struct Fixture {
  std::size_t candidates = 0;
  std::vector<float> centroids;   // [list, dim]
  std::vector<float> codebooks;   // [m, k, subdim]
  std::vector<float> persistent_t;  // [list, m, k]
  std::vector<std::uint8_t> codes;  // list-major [candidate, m]
  std::array<std::vector<float>, 2> queries;
  std::array<std::vector<float>, 2> coarse;
  std::array<std::vector<float>, 2> q_tables;
  std::array<TopK, 2> oracle_topk;
  std::array<float, 2> oracle_cutoff_next_score{};
  std::array<float, 2> cutoff_gap{};
  std::array<float, 2> min_local_cutoff_gap{};
  std::array<float, 2> min_selected_adjacent_gap{};
  std::vector<ListSpan> lists;
  std::vector<Block> blocks;
  FactorizationGate gate;
  float max_abs_q_term = 0.0f;
  float max_abs_t_term = 0.0f;
  float max_abs_coarse = 0.0f;
  float max_abs_factor_score = 0.0f;
  float max_abs_accumulator_bound = 0.0f;
  bool integer_exact_under_f32 = true;
};

std::vector<float> make_q_table(const std::vector<float>& query,
                                const std::vector<float>& codebooks) {
  std::vector<float> q(static_cast<std::size_t>(kPqM) * kKs, 0.0f);
  for (std::uint32_t m = 0; m < kPqM; ++m) {
    for (std::uint32_t code = 0; code < kKs; ++code) {
      float dot = 0.0f;
      for (std::uint32_t j = 0; j < kSubDim; ++j) {
        dot += query[static_cast<std::size_t>(m) * kSubDim + j] *
               codebooks[(static_cast<std::size_t>(m) * kKs + code) * kSubDim + j];
      }
      q[static_cast<std::size_t>(m) * kKs + code] = -2.0f * dot;
    }
  }
  return q;
}

std::uint64_t mix64(std::uint64_t value) {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31u;
  return value;
}

std::array<int, 4> four_squares(std::uint32_t value) {
  for (int a = 0; a * a <= static_cast<int>(value); ++a) {
    for (int b = 0; a * a + b * b <= static_cast<int>(value); ++b) {
      for (int c = 0; a * a + b * b + c * c <= static_cast<int>(value); ++c) {
        const int remainder = static_cast<int>(value) - a * a - b * b - c * c;
        const int d = static_cast<int>(std::sqrt(static_cast<double>(remainder)));
        if (d * d == remainder) return {a, b, c, d};
      }
    }
  }
  throw Failure("four-square construction failed");
}

Fixture make_fixture(std::size_t candidates) {
  Fixture f;
  f.candidates = candidates;
  f.centroids.resize(static_cast<std::size_t>(kLists) * kDim);
  f.codebooks.resize(static_cast<std::size_t>(kPqM) * kKs * kSubDim);
  f.persistent_t.resize(static_cast<std::size_t>(kLists) * kPqM * kKs);
  f.codes.resize(checked_mul(candidates, kPqM, "fixture codes"));
  for (auto& query : f.queries) query.resize(kDim);
  for (auto& coarse : f.coarse) coarse.resize(kLists);

  for (std::uint32_t list = 0; list < kLists; ++list) {
    for (std::uint32_t d = 0; d < kDim; ++d) {
      f.centroids[static_cast<std::size_t>(list) * kDim + d] = 0.0f;
    }
  }
  for (std::uint32_t m = 0; m < kPqM; ++m) {
    for (std::uint32_t code = 0; code < kKs; ++code) {
      for (std::uint32_t j = 0; j < kSubDim; ++j) {
        int value = 0;
        if (m == 0) {
          const std::array<int, 4> squares = four_squares(code);
          const int scale = 64;  // norm squared is 4096*code
          if (j < squares.size()) value = squares[j] * scale;
        } else {
          const std::uint64_t hash = mix64(static_cast<std::uint64_t>(m) * 0x1000003dull +
                                           static_cast<std::uint64_t>(code) * 0x9e3779b1ull +
                                           static_cast<std::uint64_t>(j) * 0x85ebca6bull);
          value = static_cast<int>(hash % 5u) - 2;
        }
        f.codebooks[(static_cast<std::size_t>(m) * kKs + code) * kSubDim + j] =
            static_cast<float>(value);
      }
    }
  }
  for (std::uint32_t epoch = 0; epoch < 2; ++epoch) {
    for (std::uint32_t d = 0; d < kDim; ++d) {
      const std::uint32_t m = d / kSubDim;
      const int value = m == 0 ? 0 :
          static_cast<int>(mix64(static_cast<std::uint64_t>(d) * 131u +
                                 epoch * 0x9e3779b9ull + 7u) % 5u) - 2;
      f.queries[epoch][d] = static_cast<float>(value);
    }
  }

  std::array<std::uint64_t, kLists> weights{};
  for (std::uint32_t list = 0; list < kLists; ++list) weights[list] = 1u + ((list * 5u + 3u) % 11u);
  const std::uint64_t total_weight = std::accumulate(weights.begin(), weights.end(), std::uint64_t{0});
  std::uint64_t first = 0;
  std::uint64_t cumulative = 0;
  for (std::uint32_t list = 0; list < kLists; ++list) {
    cumulative += weights[list];
    std::uint64_t end = list + 1 == kLists ? candidates : candidates * cumulative / total_weight;
    const std::uint64_t lists_left = kLists - list - 1;
    end = std::max(end, first + 1);
    end = std::min(end, candidates - lists_left);
    const auto rows = static_cast<std::uint32_t>(end - first);
    f.lists.push_back({static_cast<std::uint32_t>(first), rows});
    std::uint32_t consumed = 0;
    while (consumed < rows) {
      const std::uint32_t block_rows = std::min(kBlockRows, rows - consumed);
      const std::uint32_t block_first = static_cast<std::uint32_t>(first) + consumed;
      f.blocks.push_back({static_cast<std::uint64_t>(block_first) * kPqM, block_rows, list,
                          block_first, 0});
      consumed += block_rows;
    }
    for (std::uint64_t row = first; row < end; ++row) {
      for (std::uint32_t m = 0; m < kPqM; ++m) {
        if (m == 0) {
          // A strict, bounded separator protects the top-33 cutoff while the
          // remaining 15 subspaces retain full deterministic code entropy.
          f.codes[row * kPqM + m] =
              static_cast<std::uint8_t>(row < (kTopK + 1) ? row : 255u);
        } else {
          const std::uint64_t hash = mix64(row * 0x9e3779b97f4a7c15ull +
                                           static_cast<std::uint64_t>(m) * 0xbf58476d1ce4e5b9ull +
                                           static_cast<std::uint64_t>(list) * 0x94d049bb133111ebull);
          f.codes[row * kPqM + m] = static_cast<std::uint8_t>(hash & 0xffu);
        }
      }
    }
    first = end;
  }
  if (first != candidates) throw Failure("list geometry does not cover the fixture");

  for (std::uint32_t list = 0; list < kLists; ++list) {
    for (std::uint32_t m = 0; m < kPqM; ++m) {
      for (std::uint32_t code = 0; code < kKs; ++code) {
        float t = 0.0f;
        for (std::uint32_t j = 0; j < kSubDim; ++j) {
          const float c = f.centroids[static_cast<std::size_t>(list) * kDim + m * kSubDim + j];
          const float w = f.codebooks[(static_cast<std::size_t>(m) * kKs + code) * kSubDim + j];
          t += w * w + 2.0f * c * w;
        }
        f.persistent_t[(static_cast<std::size_t>(list) * kPqM + m) * kKs + code] = t;
        f.max_abs_t_term = std::max(f.max_abs_t_term, std::fabs(t));
        f.integer_exact_under_f32 =
            f.integer_exact_under_f32 && t == std::trunc(t);
      }
    }
  }

  for (std::uint32_t epoch = 0; epoch < 2; ++epoch) {
    f.q_tables[epoch] = make_q_table(f.queries[epoch], f.codebooks);
    for (float value : f.q_tables[epoch]) {
      f.max_abs_q_term = std::max(f.max_abs_q_term, std::fabs(value));
      f.integer_exact_under_f32 =
          f.integer_exact_under_f32 && value == std::trunc(value);
    }
    for (std::uint32_t list = 0; list < kLists; ++list) {
      float coarse = 0.0f;
      for (std::uint32_t d = 0; d < kDim; ++d) {
        const float diff = f.queries[epoch][d] -
                           f.centroids[static_cast<std::size_t>(list) * kDim + d];
        coarse += diff * diff;
      }
      f.coarse[epoch][list] = coarse;
      f.max_abs_coarse = std::max(f.max_abs_coarse, std::fabs(coarse));
      f.integer_exact_under_f32 =
          f.integer_exact_under_f32 && coarse == std::trunc(coarse);
    }

    TopK direct_topk;
    TopK factor_topk;
    std::vector<Pair> direct_scores;
    direct_scores.reserve(candidates);
    std::vector<float> direct_score_by_id(candidates);
    for (std::uint32_t list = 0; list < kLists; ++list) {
      const ListSpan span = f.lists[list];
      for (std::uint32_t local = 0; local < span.rows; ++local) {
        const std::uint32_t row = span.first_id + local;
        float direct_score = 0.0f;
        float factor_partial = 0.0f;
        for (std::uint32_t m = 0; m < kPqM; ++m) {
          const std::uint32_t code = f.codes[static_cast<std::size_t>(row) * kPqM + m];
          float subspace = 0.0f;
          for (std::uint32_t j = 0; j < kSubDim; ++j) {
            const float diff = f.queries[epoch][m * kSubDim + j] -
                               f.centroids[static_cast<std::size_t>(list) * kDim + m * kSubDim + j] -
                               f.codebooks[(static_cast<std::size_t>(m) * kKs + code) * kSubDim + j];
            subspace += diff * diff;
          }
          direct_score += subspace;
          factor_partial += f.q_tables[epoch][static_cast<std::size_t>(m) * kKs + code] +
                            f.persistent_t[(static_cast<std::size_t>(list) * kPqM + m) * kKs + code];
          f.max_abs_accumulator_bound =
              std::max(f.max_abs_accumulator_bound, std::fabs(factor_partial));
        }
        const float factor_score = factor_partial + f.coarse[epoch][list];
        f.gate.finite = f.gate.finite && std::isfinite(direct_score) && std::isfinite(factor_score);
        const float error = std::fabs(direct_score - factor_score);
        f.gate.max_abs_error = std::max(f.gate.max_abs_error, error);
        f.max_abs_factor_score = std::max(f.max_abs_factor_score, std::fabs(factor_score));
        f.max_abs_accumulator_bound =
            std::max(f.max_abs_accumulator_bound, std::fabs(factor_score));
        f.integer_exact_under_f32 =
            f.integer_exact_under_f32 && factor_score == std::trunc(factor_score);
        ++f.gate.compared_scores;
        if (std::bit_cast<std::uint32_t>(direct_score) == std::bit_cast<std::uint32_t>(factor_score)) {
          ++f.gate.bit_equal_scores;
        } else {
          f.gate.bit_equal = false;
        }
        insert_topk(direct_topk, direct_score, row);
        insert_topk(factor_topk, factor_score, row);
        direct_scores.push_back({direct_score, row});
        direct_score_by_id[row] = direct_score;
      }
    }
    std::partial_sort(
        direct_scores.begin(), direct_scores.begin() + (kTopK + 1), direct_scores.end(),
        [](const Pair& lhs, const Pair& rhs) {
          return better(lhs.score, lhs.id, rhs.score, rhs.id);
        });
    f.oracle_cutoff_next_score[epoch] = direct_scores[kTopK].score;
    f.cutoff_gap[epoch] = direct_scores[kTopK].score - direct_scores[kTopK - 1].score;
    f.min_selected_adjacent_gap[epoch] = kInfinity;
    for (std::size_t rank = 1; rank < kTopK; ++rank) {
      f.min_selected_adjacent_gap[epoch] =
          std::min(f.min_selected_adjacent_gap[epoch],
                   direct_scores[rank].score - direct_scores[rank - 1].score);
    }
    f.min_local_cutoff_gap[epoch] = kInfinity;
    for (const Block& block : f.blocks) {
      if (block.rows <= kTopK) continue;
      std::vector<Pair> local_scores;
      local_scores.reserve(block.rows);
      for (std::uint32_t row = 0; row < block.rows; ++row) {
        const std::uint32_t id = block.first_id + row;
        local_scores.push_back({direct_score_by_id[id], id});
      }
      std::partial_sort(local_scores.begin(), local_scores.begin() + (kTopK + 1),
                        local_scores.end(), [](const Pair& lhs, const Pair& rhs) {
                          return better(lhs.score, lhs.id, rhs.score, rhs.id);
                        });
      f.min_local_cutoff_gap[epoch] =
          std::min(f.min_local_cutoff_gap[epoch],
                   local_scores[kTopK].score - local_scores[kTopK - 1].score);
    }
    f.gate.exact_topk = f.gate.exact_topk && equal_topk(factor_topk, direct_topk, 0.0f);
    f.oracle_topk[epoch] = direct_topk;
  }
  constexpr float kExactIntegerLimit = 16777216.0f;
  f.max_abs_accumulator_bound =
      std::max(f.max_abs_accumulator_bound, f.max_abs_factor_score);
  f.integer_exact_under_f32 = f.integer_exact_under_f32 &&
                              f.max_abs_q_term < kExactIntegerLimit &&
                              f.max_abs_t_term < kExactIntegerLimit &&
                              f.max_abs_coarse < kExactIntegerLimit &&
                              f.max_abs_accumulator_bound < kExactIntegerLimit;
  return f;
}

double percentile(const std::vector<double>& samples, double p) {
  if (samples.empty()) throw Failure("cannot summarize an empty sample set");
  std::vector<double> sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  const std::size_t rank = static_cast<std::size_t>(std::ceil(p * sorted.size()));
  return sorted[std::max<std::size_t>(1, rank) - 1];
}

struct Distribution {
  std::vector<double> values;

  double p50() const { return percentile(values, 0.50); }
  double p95() const { return percentile(values, 0.95); }
  double p99() const { return percentile(values, 0.99); }
};

std::optional<double> event_ms(const sycl::event& event) {
  try {
    const auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    if (end < start) return std::nullopt;
    return static_cast<double>(end - start) / 1.0e6;
  } catch (...) {
    return std::nullopt;
  }
}

template <typename T>
class DeviceAllocation {
 public:
  DeviceAllocation() = default;
  DeviceAllocation(sycl::queue& queue, std::size_t count) : queue_(&queue), count_(count) {
    if (count == 0) return;
    pointer_ = sycl::malloc_device<T>(count, queue);
    if (!pointer_) throw std::bad_alloc();
  }
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  DeviceAllocation(DeviceAllocation&& other) noexcept
      : queue_(other.queue_), pointer_(other.pointer_), count_(other.count_) {
    other.queue_ = nullptr;
    other.pointer_ = nullptr;
    other.count_ = 0;
  }
  DeviceAllocation& operator=(DeviceAllocation&& other) noexcept {
    if (this == &other) return *this;
    release();
    queue_ = other.queue_;
    pointer_ = other.pointer_;
    count_ = other.count_;
    other.queue_ = nullptr;
    other.pointer_ = nullptr;
    other.count_ = 0;
    return *this;
  }
  ~DeviceAllocation() { release(); }

  T* get() const { return pointer_; }
  std::uint64_t bytes() const { return static_cast<std::uint64_t>(count_) * sizeof(T); }

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
    queue_ = nullptr;
    count_ = 0;
  }

  sycl::queue* queue_ = nullptr;
  T* pointer_ = nullptr;
  std::size_t count_ = 0;
};

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

std::optional<sycl::device> select_integrated_intel_l0_gpu(std::string& reason) {
  std::vector<sycl::device> candidates;
  try {
    for (const sycl::platform& platform : sycl::platform::get_platforms()) {
      std::string name = platform.get_info<sycl::info::platform::name>();
      std::transform(name.begin(), name.end(), name.begin(),
                     [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      if (name.find("level-zero") == std::string::npos &&
          name.find("level zero") == std::string::npos) {
        continue;
      }
      for (const sycl::device& device : platform.get_devices(sycl::info::device_type::gpu)) {
        if (device.get_info<sycl::info::device::vendor_id>() == 0x8086u) {
          candidates.push_back(device);
        }
      }
    }
  } catch (const sycl::exception& exception) {
    reason = std::string("device_enumeration_failed: ") + exception.what();
    return std::nullopt;
  }
  if (candidates.empty()) {
    reason = "no Intel Level Zero GPU";
    return std::nullopt;
  }
  if (candidates.size() != 1) {
    reason = "multiple Intel Level Zero GPUs; selection is ambiguous";
    return std::nullopt;
  }
  const sycl::device& device = candidates.front();
  if (!device.has(sycl::aspect::queue_profiling) ||
      !device.has(sycl::aspect::usm_device_allocations) ||
      device.get_info<sycl::info::device::max_work_group_size>() < kBlockRows ||
      device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0] < kBlockRows ||
      device.get_info<sycl::info::device::local_mem_size>() <
          kBlockRows * (sizeof(float) + sizeof(std::uint32_t))) {
    reason = "selected Intel Level Zero GPU lacks required profiling, USM, work-group, or local-memory capability";
    return std::nullopt;
  }
  return device;
}

struct ProjectionSample {
  double wall_ms = 0.0;
  double input_ms = 0.0;
  double execution_ms = 0.0;
  double output_ms = 0.0;
  float max_abs_error = 0.0f;
  bool finite = false;
};

struct ProjectionSummary {
  std::string status = "unavailable";
  std::string reason;
  double compile_ms = 0.0;
  double setup_ms = 0.0;
  ProjectionSample first;
  std::vector<ProjectionSample> warm;
  std::uint64_t persistent_bytes = 0;
  std::uint64_t per_call_input_bytes = 0;
  std::uint64_t per_call_output_bytes = 0;
  std::uint32_t requests_per_call = 0;
  std::uint32_t launches_per_call = 0;
  bool persistent_request = false;
  bool persistent_compiled_graph = false;
};

bool validate_q(const float* actual, const std::vector<float>& expected, float tolerance,
                float& max_abs_error) {
  if (!actual || expected.size() != static_cast<std::size_t>(kPqM) * kKs) return false;
  max_abs_error = 0.0f;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (!std::isfinite(actual[i])) return false;
    max_abs_error = std::max(max_abs_error, std::fabs(actual[i] - expected[i]));
  }
  return max_abs_error <= tolerance;
}

ProjectionSample cpu_project_once(const Fixture& fixture, std::uint32_t epoch,
                                  std::vector<float>& output) {
  ProjectionSample sample;
  const auto start = Clock::now();
  output = make_q_table(fixture.queries[epoch], fixture.codebooks);
  sample.execution_ms = elapsed_ms(start);
  sample.wall_ms = sample.execution_ms;
  sample.finite = validate_q(output.data(), fixture.q_tables[epoch], 0.0f,
                             sample.max_abs_error);
  return sample;
}

ProjectionSummary measure_cpu_projection(const Fixture& fixture, const Options& options) {
  ProjectionSummary result;
  result.status = "ok";
  result.per_call_input_bytes = kDim * sizeof(float);
  result.per_call_output_bytes = static_cast<std::uint64_t>(kPqM) * kKs * sizeof(float);
  std::vector<float> output;
  result.first = cpu_project_once(fixture, 0, output);
  if (!result.first.finite) throw Failure("CPU Q projection failed its exact oracle");
  for (int i = 0; i < options.warmups; ++i) {
    if (!cpu_project_once(fixture, static_cast<std::uint32_t>(i & 1), output).finite) {
      throw Failure("CPU Q projection warmup failed");
    }
  }
  for (int i = 0; i < options.repeats; ++i) {
    ProjectionSample sample = cpu_project_once(fixture, static_cast<std::uint32_t>(i & 1), output);
    if (!sample.finite) throw Failure("CPU Q projection measured sample failed");
    result.warm.push_back(sample);
  }
  return result;
}

class NpuProjector {
 public:
  NpuProjector(const Fixture& fixture, const Options& options) : fixture_(fixture) {
    const auto devices = core_.get_available_devices();
    if (std::find(devices.begin(), devices.end(), "NPU") == devices.end()) {
      throw Failure("OpenVINO does not expose an NPU device");
    }
    try {
      device_name_ = core_.get_property("NPU", "FULL_DEVICE_NAME").as<std::string>();
    } catch (...) {
      device_name_ = "NPU";
    }

    cache_dir_ = options.cache_base /
                 ("run-" + std::to_string(static_cast<unsigned long>(GetCurrentProcessId())));
    std::error_code ec;
    if (std::filesystem::exists(cache_dir_, ec)) {
      throw Failure("isolated NPU cache path already exists: " + cache_dir_.string());
    }
    if (!std::filesystem::create_directories(cache_dir_, ec) || ec) {
      throw Failure("cannot create isolated NPU cache path: " + cache_dir_.string());
    }

    std::vector<float> weights(static_cast<std::size_t>(kPqM) * kSubDim * kKs);
    for (std::uint32_t m = 0; m < kPqM; ++m) {
      for (std::uint32_t j = 0; j < kSubDim; ++j) {
        for (std::uint32_t code = 0; code < kKs; ++code) {
          weights[(static_cast<std::size_t>(m) * kSubDim + j) * kKs + code] =
              fixture_.codebooks[(static_cast<std::size_t>(m) * kKs + code) * kSubDim + j];
        }
      }
    }
    persistent_weight_bytes_ = static_cast<std::uint64_t>(weights.size()) * sizeof(float);

    auto query = std::make_shared<ov::opset8::Parameter>(
        ov::element::f32, ov::Shape{kPqM, 1, kSubDim});
    query->set_friendly_name("query_subspaces");
    query->output(0).get_tensor().set_names({"query_subspaces"});
    auto stationary_weights = std::make_shared<ov::opset8::Parameter>(
        ov::element::f32, ov::Shape{kPqM, kSubDim, kKs});
    stationary_weights->set_friendly_name("stationary_codebooks");
    stationary_weights->output(0).get_tensor().set_names({"stationary_codebooks"});
    auto projected =
        std::make_shared<ov::opset8::MatMul>(query, stationary_weights, false, false);
    auto scale = ov::opset8::Constant::create(ov::element::f32, ov::Shape{}, {-2.0f});
    auto scaled = std::make_shared<ov::opset8::Multiply>(projected, scale);
    auto flat_shape = ov::opset8::Constant::create(
        ov::element::i64, ov::Shape{1}, {static_cast<std::int64_t>(kPqM * kKs)});
    auto flat = std::make_shared<ov::opset8::Reshape>(scaled, flat_shape, false);
    flat->set_friendly_name("q_table");
    flat->output(0).get_tensor().set_names({"q_table"});
    model_ = std::make_shared<ov::Model>(ov::OutputVector{flat},
                                         ov::ParameterVector{query, stationary_weights},
                                         "ovvs_residual_pq_q_projection");

    ov::AnyMap properties;
    properties["PERFORMANCE_HINT"] = "LATENCY";
    properties["NPU_TURBO"] = true;
    properties["NPU_COMPILATION_MODE_PARAMS"] =
        std::string("optimization-level=2 performance-hint-override=latency");
    properties["CACHE_DIR"] = cache_dir_.string();
    const auto compile_start = Clock::now();
    try {
      compiled_ = core_.compile_model(model_, "NPU", properties);
      compile_property_scope_ = "turbo+optimization-level=2";
    } catch (const std::exception& first) {
      first_compile_error_ = first.what();
      properties.erase("NPU_COMPILATION_MODE_PARAMS");
      try {
        compiled_ = core_.compile_model(model_, "NPU", properties);
        compile_property_scope_ = "turbo_without_compilation_mode_params";
      } catch (const std::exception& second) {
        second_compile_error_ = second.what();
        properties.erase("NPU_TURBO");
        compiled_ = core_.compile_model(model_, "NPU", properties);
        compile_property_scope_ = "latency_without_turbo_or_compilation_mode_params";
      }
    }
    compile_ms_ = elapsed_ms(compile_start);

    const auto setup_start = Clock::now();
    request_ = compiled_.create_infer_request();
    input_ = request_.get_input_tensor(0);
    weight_input_ = request_.get_input_tensor(1);
    output_ = request_.get_output_tensor(0);
    if (input_.get_element_type() != ov::element::f32 ||
        input_.get_shape() != ov::Shape{kPqM, 1, kSubDim} ||
        weight_input_.get_element_type() != ov::element::f32 ||
        weight_input_.get_shape() != ov::Shape{kPqM, kSubDim, kKs} ||
        output_.get_element_type() != ov::element::f32 ||
        output_.get_shape() != ov::Shape{kPqM * kKs}) {
      throw Failure("NPU Q projection tensor contract changed");
    }
    // The request owns the L0 input buffer. Fill stationary weights once and
    // keep this request alive; measured calls update only the 512-byte query.
    std::memcpy(weight_input_.data<float>(), weights.data(), weight_input_.get_byte_size());
    setup_ms_ = elapsed_ms(setup_start);
  }

  ProjectionSample project(std::uint32_t epoch) {
    if (epoch >= fixture_.queries.size()) throw Failure("invalid NPU projection epoch");
    ProjectionSample sample;
    const auto wall_start = Clock::now();
    const auto input_start = Clock::now();
    std::memcpy(input_.data<float>(), fixture_.queries[epoch].data(), input_.get_byte_size());
    sample.input_ms = elapsed_ms(input_start);
    const auto execute_start = Clock::now();
    request_.start_async();
    request_.wait();
    sample.execution_ms = elapsed_ms(execute_start);
    const auto output_start = Clock::now();
    sample.finite = validate_q(output_.data<float>(), fixture_.q_tables[epoch], 1.0e-3f,
                               sample.max_abs_error);
    sample.output_ms = elapsed_ms(output_start);
    sample.wall_ms = elapsed_ms(wall_start);
    if (!sample.finite) throw Failure("NPU Q projection failed finite/epsilon oracle");
    return sample;
  }

  const float* output_data() { return output_.data<float>(); }
  std::uint64_t input_bytes() const { return input_.get_byte_size(); }
  std::uint64_t output_bytes() const { return output_.get_byte_size(); }
  std::uint64_t persistent_weight_bytes() const { return persistent_weight_bytes_; }
  double compile_ms() const { return compile_ms_; }
  double setup_ms() const { return setup_ms_; }
  const std::string& device_name() const { return device_name_; }
  const std::filesystem::path& cache_dir() const { return cache_dir_; }
  const std::string& compile_property_scope() const { return compile_property_scope_; }
  const std::string& first_compile_error() const { return first_compile_error_; }
  const std::string& second_compile_error() const { return second_compile_error_; }

 private:
  const Fixture& fixture_;
  ov::Core core_;
  std::shared_ptr<ov::Model> model_;
  ov::CompiledModel compiled_;
  ov::InferRequest request_;
  ov::Tensor input_;
  ov::Tensor weight_input_;
  ov::Tensor output_;
  std::filesystem::path cache_dir_;
  std::string device_name_;
  std::string compile_property_scope_;
  std::string first_compile_error_;
  std::string second_compile_error_;
  double compile_ms_ = 0.0;
  double setup_ms_ = 0.0;
  std::uint64_t persistent_weight_bytes_ = 0;
};

ProjectionSummary measure_npu_projection(NpuProjector& projector, const Options& options) {
  ProjectionSummary result;
  result.status = "ok";
  result.compile_ms = projector.compile_ms();
  result.setup_ms = projector.setup_ms();
  result.persistent_bytes = projector.persistent_weight_bytes();
  result.per_call_input_bytes = projector.input_bytes();
  result.per_call_output_bytes = projector.output_bytes();
  result.requests_per_call = 1;
  result.persistent_request = true;
  result.persistent_compiled_graph = true;
  result.first = projector.project(0);
  for (int i = 0; i < options.warmups; ++i) projector.project(static_cast<std::uint32_t>(i & 1));
  for (int i = 0; i < options.repeats; ++i) {
    result.warm.push_back(projector.project(static_cast<std::uint32_t>(i & 1)));
  }
  return result;
}

enum class QProducer { kCpu, kGpu, kNpu };

const char* producer_name(QProducer producer) {
  switch (producer) {
    case QProducer::kCpu: return "cpu_q_plus_igpu_scan";
    case QProducer::kGpu: return "igpu_q_plus_igpu_scan";
    case QProducer::kNpu: return "npu_q_plus_igpu_scan";
  }
  return "unknown";
}

struct JoinedSample {
  bool success = false;
  std::string reason;
  double wall_ms = 0.0;
  double q_input_upload_ms = 0.0;
  double q_execution_ms = 0.0;
  double q_output_upload_ms = 0.0;
  double coarse_upload_ms = 0.0;
  double reset_ms = 0.0;
  double scan_local_topk_ms = 0.0;
  double partial_readback_ms = 0.0;
  double final_select_ms = 0.0;
  double direct_selected_rescore_ms = 0.0;
  double direct_fallback_ms = 0.0;
  double unattributed_ms = 0.0;
  std::uint32_t npu_requests = 0;
  std::uint32_t gpu_kernel_launches = 0;
  std::uint32_t gpu_queue_submissions = 0;
  std::uint32_t host_synchronizations = 0;
  std::uint32_t gpu_host_synchronizations = 0;
  std::uint32_t npu_host_waits = 0;
  float q_max_abs_error = 0.0f;
  float topk_max_abs_error = 0.0f;
  float measured_host_factor_vs_direct_band = 0.0f;
  float score_error_band = 0.0f;
  float required_gap = 0.0f;
  float global_cutoff_gap = 0.0f;
  float factor_global_cutoff_gap = 0.0f;
  float min_local_cutoff_gap = 0.0f;
  float factor_min_local_cutoff_gap = 0.0f;
  float min_selected_adjacent_gap = 0.0f;
  double recall_at_k = 0.0;
  double factor_candidate_recall_at_k = 0.0;
  bool finite = false;
  bool ordered_ids_exact = false;
  bool factor_candidate_ordered_ids_exact = false;
  bool direct_selected_rescore_exact = false;
  bool local_k_plus_one_complete = false;
  bool q_per_call_validated = false;
  bool bounded_integer_analytic_band = false;
  bool cutoff_certificate = false;
  bool direct_fallback_activated = false;
  bool direct_fallback_exact = false;
  bool publication_canaries = false;
};

struct JoinedSummary {
  QProducer producer = QProducer::kCpu;
  std::string status = "unavailable";
  std::string reason;
  JoinedSample first;
  std::vector<JoinedSample> warm;
};

bool finite_vector(const std::vector<float>& values) {
  return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

TopK direct_residual_lut_topk(const Fixture& fixture, std::uint32_t epoch) {
  std::vector<float> lut(static_cast<std::size_t>(kLists) * kPqM * kKs);
  for (std::uint32_t list = 0; list < kLists; ++list) {
    for (std::uint32_t m = 0; m < kPqM; ++m) {
      for (std::uint32_t code = 0; code < kKs; ++code) {
        float distance = 0.0f;
        for (std::uint32_t j = 0; j < kSubDim; ++j) {
          const float diff = fixture.queries[epoch][m * kSubDim + j] -
                             fixture.centroids[static_cast<std::size_t>(list) * kDim +
                                               m * kSubDim + j] -
                             fixture.codebooks[(static_cast<std::size_t>(m) * kKs + code) *
                                               kSubDim + j];
          distance += diff * diff;
        }
        lut[(static_cast<std::size_t>(list) * kPqM + m) * kKs + code] = distance;
      }
    }
  }
  TopK selected;
  for (std::uint32_t list = 0; list < kLists; ++list) {
    const ListSpan span = fixture.lists[list];
    for (std::uint32_t local = 0; local < span.rows; ++local) {
      const std::uint32_t row = span.first_id + local;
      float score = 0.0f;
      for (std::uint32_t m = 0; m < kPqM; ++m) {
        const std::uint32_t code = fixture.codes[static_cast<std::size_t>(row) * kPqM + m];
        score += lut[(static_cast<std::size_t>(list) * kPqM + m) * kKs + code];
      }
      insert_topk(selected, score, row);
    }
  }
  return selected;
}

float direct_residual_score(const Fixture& fixture, std::uint32_t epoch,
                            std::uint32_t id) {
  if (id >= fixture.candidates) throw Failure("direct selected rescore ID out of range");
  std::uint32_t list = 0;
  while (list < kLists &&
         id >= fixture.lists[list].first_id + fixture.lists[list].rows) {
    ++list;
  }
  if (list >= kLists || id < fixture.lists[list].first_id) {
    throw Failure("direct selected rescore could not resolve list");
  }
  float score = 0.0f;
  for (std::uint32_t m = 0; m < kPqM; ++m) {
    const std::uint32_t code = fixture.codes[static_cast<std::size_t>(id) * kPqM + m];
    float subspace = 0.0f;
    for (std::uint32_t j = 0; j < kSubDim; ++j) {
      const float diff = fixture.queries[epoch][m * kSubDim + j] -
                         fixture.centroids[static_cast<std::size_t>(list) * kDim +
                                           m * kSubDim + j] -
                         fixture.codebooks[(static_cast<std::size_t>(m) * kKs + code) *
                                           kSubDim + j];
      subspace += diff * diff;
    }
    score += subspace;
  }
  return score;
}

class GpuFactorRunner {
 public:
  GpuFactorRunner(sycl::queue& queue, const Fixture& fixture)
      : queue_(queue),
        fixture_(fixture),
        codebooks_(queue, fixture.codebooks.size()),
        t_(queue, fixture.persistent_t.size()),
        codes_(queue, fixture.codes.size()),
        blocks_(queue, fixture.blocks.size()),
        query_(queue, kDim),
        q_(queue, static_cast<std::size_t>(kPqM) * kKs),
        coarse_(queue, kLists),
        partials_(queue, fixture.blocks.size() * (kTopK + 1)),
        error_(queue, 1) {
    const auto start = Clock::now();
    queue_.memcpy(codebooks_.get(), fixture.codebooks.data(), codebooks_.bytes());
    queue_.memcpy(t_.get(), fixture.persistent_t.data(), t_.bytes());
    queue_.memcpy(codes_.get(), fixture.codes.data(), codes_.bytes());
    queue_.memcpy(blocks_.get(), fixture.blocks.data(), blocks_.bytes());
    queue_.wait_and_throw();
    persistent_upload_ms_ = elapsed_ms(start);
  }

  ProjectionSample project_gpu(std::uint32_t epoch) {
    if (epoch >= fixture_.queries.size() || !finite_vector(fixture_.queries[epoch])) {
      throw Failure("iGPU Q projection rejected a non-finite query");
    }
    ProjectionSample sample;
    std::vector<float> output(static_cast<std::size_t>(kPqM) * kKs);
    const auto wall_start = Clock::now();
    const sycl::event input_event =
        queue_.memcpy(query_.get(), fixture_.queries[epoch].data(), query_.bytes());
    const sycl::event projection_event = submit_projection();
    sycl::event output_event = queue_.memcpy(output.data(), q_.get(), q_.bytes());
    output_event.wait_and_throw();
    const auto input_ms = event_ms(input_event);
    const auto execution_ms = event_ms(projection_event);
    const auto output_ms = event_ms(output_event);
    if (!input_ms || !execution_ms || !output_ms) {
      throw Failure("iGPU projection profiling timestamps unavailable");
    }
    sample.wall_ms = elapsed_ms(wall_start);
    sample.input_ms = *input_ms;
    sample.execution_ms = *execution_ms;
    sample.output_ms = *output_ms;
    sample.finite = validate_q(output.data(), fixture_.q_tables[epoch], 0.0f,
                               sample.max_abs_error);
    if (!sample.finite) throw Failure("iGPU Q projection failed exact oracle");
    return sample;
  }

  JoinedSample run(QProducer producer, std::uint32_t epoch, NpuProjector* npu) {
    JoinedSample sample;
    std::array<Pair, kTopK + 2> publication;
    for (Pair& pair : publication) pair = {-12345.0f, 0xdeadbeefu};
    const auto wall_start = Clock::now();
    if (epoch >= fixture_.queries.size() || !finite_vector(fixture_.queries[epoch]) ||
        !finite_vector(fixture_.coarse[epoch])) {
      sample.reason = "non_finite_or_invalid_query";
      sample.publication_canaries = publication.front().id == 0xdeadbeefu &&
                                    publication.back().id == 0xdeadbeefu;
      return sample;
    }

    std::vector<float> cpu_q;
    std::optional<sycl::event> q_input_event;
    std::optional<sycl::event> q_execution_event;
    std::optional<sycl::event> q_output_event;
    try {
      if (producer == QProducer::kCpu) {
        const auto q_start = Clock::now();
        cpu_q = make_q_table(fixture_.queries[epoch], fixture_.codebooks);
        sample.q_execution_ms = elapsed_ms(q_start);
        if (!validate_q(cpu_q.data(), fixture_.q_tables[epoch], 0.0f,
                        sample.q_max_abs_error)) {
          sample.reason = "cpu_q_oracle_mismatch";
          return sample;
        }
        sample.q_per_call_validated = true;
        sample.bounded_integer_analytic_band = fixture_.integer_exact_under_f32;
        q_output_event = queue_.memcpy(q_.get(), cpu_q.data(), q_.bytes());
        sample.gpu_queue_submissions += 1;
      } else if (producer == QProducer::kGpu) {
        q_input_event = queue_.memcpy(query_.get(), fixture_.queries[epoch].data(), query_.bytes());
        q_execution_event = submit_projection();
        sample.gpu_queue_submissions += 2;
        sample.gpu_kernel_launches += 1;
        // The joined lane does not read Q back. The fixed integer fixture and
        // the separately exercised bit-exact projection kernel provide the
        // bounded-fixture analytic E=0; this is not a general f32 certificate.
        sample.q_per_call_validated = false;
        sample.bounded_integer_analytic_band = fixture_.integer_exact_under_f32;
      } else {
        if (!npu) {
          sample.reason = "npu_projector_unavailable";
          return sample;
        }
        const ProjectionSample projection = npu->project(epoch);
        sample.q_input_upload_ms = projection.input_ms;
        sample.q_execution_ms = projection.execution_ms;
        sample.q_output_upload_ms = projection.output_ms;
        sample.q_max_abs_error = projection.max_abs_error;
        sample.q_per_call_validated = true;
        sample.bounded_integer_analytic_band =
            fixture_.integer_exact_under_f32 && projection.max_abs_error == 0.0f;
        sample.npu_requests = 1;
        sample.npu_host_waits = 1;
        q_output_event = queue_.memcpy(q_.get(), npu->output_data(), q_.bytes());
        sample.gpu_queue_submissions += 1;
      }

      const sycl::event coarse_event =
          queue_.memcpy(coarse_.get(), fixture_.coarse[epoch].data(), coarse_.bytes());
      const sycl::event reset_event = queue_.memset(error_.get(), 0, sizeof(std::int32_t));
      sample.gpu_queue_submissions += 2;
      const sycl::event scan_event = submit_scan();
      sample.gpu_queue_submissions += 1;
      sample.gpu_kernel_launches += 1;

      std::vector<Pair> staged(fixture_.blocks.size() * (kTopK + 1));
      std::int32_t staged_error = 0;
      const sycl::event partial_event =
          queue_.memcpy(staged.data(), partials_.get(), partials_.bytes());
      sycl::event error_event =
          queue_.memcpy(&staged_error, error_.get(), sizeof(staged_error));
      sample.gpu_queue_submissions += 2;
      error_event.wait_and_throw();
      sample.gpu_host_synchronizations = 1;
      sample.host_synchronizations =
          sample.gpu_host_synchronizations + sample.npu_host_waits;

      if (q_input_event) {
        const auto ms = event_ms(*q_input_event);
        if (!ms) throw Failure("query upload profiling unavailable");
        sample.q_input_upload_ms += *ms;
      }
      if (q_execution_event) {
        const auto ms = event_ms(*q_execution_event);
        if (!ms) throw Failure("Q projection profiling unavailable");
        sample.q_execution_ms += *ms;
      }
      if (q_output_event) {
        const auto ms = event_ms(*q_output_event);
        if (!ms) throw Failure("Q output upload profiling unavailable");
        sample.q_output_upload_ms += *ms;
      }
      const auto coarse_ms = event_ms(coarse_event);
      const auto reset_ms = event_ms(reset_event);
      const auto scan_ms = event_ms(scan_event);
      const auto partial_ms = event_ms(partial_event);
      const auto error_ms = event_ms(error_event);
      if (!coarse_ms || !reset_ms || !scan_ms || !partial_ms || !error_ms) {
        throw Failure("joined GPU stage profiling unavailable");
      }
      sample.coarse_upload_ms = *coarse_ms;
      sample.reset_ms = *reset_ms;
      sample.scan_local_topk_ms = *scan_ms;
      sample.partial_readback_ms = *partial_ms + *error_ms;
      if (staged_error != 0) {
        sample.reason = "device_non_finite_or_code_validation_failed";
        return sample;
      }

      const auto select_start = Clock::now();
      sample.factor_min_local_cutoff_gap = kInfinity;
      sample.local_k_plus_one_complete = true;
      for (std::size_t block = 0; block < fixture_.blocks.size(); ++block) {
        if (fixture_.blocks[block].rows <= kTopK) continue;
        const Pair& kth = staged[block * (kTopK + 1) + (kTopK - 1)];
        const Pair& next = staged[block * (kTopK + 1) + kTopK];
        if (kth.id == kInvalidId || next.id == kInvalidId ||
            !std::isfinite(kth.score) || !std::isfinite(next.score)) {
          sample.local_k_plus_one_complete = false;
          sample.reason = "incomplete_local_k_plus_one";
          return sample;
        }
        sample.factor_min_local_cutoff_gap =
            std::min(sample.factor_min_local_cutoff_gap, next.score - kth.score);
      }
      std::vector<Pair> valid_partials;
      valid_partials.reserve(staged.size());
      for (const Pair& pair : staged) {
        if (pair.id == kInvalidId) continue;
        if (!std::isfinite(pair.score)) {
          sample.reason = "non_finite_partial_topk";
          return sample;
        }
        valid_partials.push_back(pair);
      }
      if (valid_partials.size() <= kTopK) {
        sample.reason = "incomplete_global_k_plus_one";
        return sample;
      }
      std::partial_sort(valid_partials.begin(), valid_partials.begin() + (kTopK + 1),
                        valid_partials.end(), [](const Pair& lhs, const Pair& rhs) {
                          return better(lhs.score, lhs.id, rhs.score, rhs.id);
                        });
      TopK factor_selected;
      for (std::size_t rank = 0; rank < kTopK; ++rank) {
        factor_selected.scores[rank] = valid_partials[rank].score;
        factor_selected.ids[rank] = valid_partials[rank].id;
      }
      sample.factor_global_cutoff_gap =
          valid_partials[kTopK].score - valid_partials[kTopK - 1].score;
      sample.final_select_ms = elapsed_ms(select_start);

      std::size_t factor_matching_ids = 0;
      for (std::size_t rank = 0; rank < kTopK; ++rank) {
        if (factor_selected.ids[rank] == fixture_.oracle_topk[epoch].ids[rank]) {
          ++factor_matching_ids;
        }
      }
      sample.factor_candidate_recall_at_k = static_cast<double>(factor_matching_ids) / kTopK;
      const float tolerance = producer == QProducer::kNpu ? 0.05f : 0.0f;
      sample.factor_candidate_ordered_ids_exact =
          equal_topk(factor_selected, fixture_.oracle_topk[epoch], tolerance,
                     &sample.topk_max_abs_error);
      sample.finite = std::all_of(factor_selected.scores.begin(), factor_selected.scores.end(),
                                  [](float score) { return std::isfinite(score); });
      if (!sample.finite) {
        sample.reason = "joined_factor_candidate_non_finite";
        return sample;
      }

      const auto rescore_start = Clock::now();
      std::array<Pair, kTopK + 1> directly_rescored{};
      for (std::size_t rank = 0; rank <= kTopK; ++rank) {
        directly_rescored[rank] =
            {direct_residual_score(fixture_, epoch, valid_partials[rank].id),
             valid_partials[rank].id};
      }
      std::sort(directly_rescored.begin(), directly_rescored.end(),
                [](const Pair& lhs, const Pair& rhs) {
                  return better(lhs.score, lhs.id, rhs.score, rhs.id);
                });
      TopK rescored_selected;
      for (std::size_t rank = 0; rank < kTopK; ++rank) {
        rescored_selected.scores[rank] = directly_rescored[rank].score;
        rescored_selected.ids[rank] = directly_rescored[rank].id;
      }
      sample.direct_selected_rescore_ms = elapsed_ms(rescore_start);
      float rescore_error = 0.0f;
      sample.direct_selected_rescore_exact =
          equal_topk(rescored_selected, fixture_.oracle_topk[epoch], 0.0f,
                     &rescore_error);

      sample.measured_host_factor_vs_direct_band = fixture_.gate.max_abs_error;
      sample.score_error_band = fixture_.gate.max_abs_error +
                                static_cast<float>(kPqM) * sample.q_max_abs_error;
      sample.required_gap = 2.0f * sample.score_error_band;
      sample.global_cutoff_gap = fixture_.cutoff_gap[epoch];
      sample.min_local_cutoff_gap = fixture_.min_local_cutoff_gap[epoch];
      sample.min_selected_adjacent_gap = fixture_.min_selected_adjacent_gap[epoch];
      sample.cutoff_certificate = sample.local_k_plus_one_complete &&
                                  sample.bounded_integer_analytic_band &&
                                  std::isfinite(sample.score_error_band) &&
                                  sample.score_error_band >= 0.0f &&
                                  std::isfinite(sample.required_gap) &&
                                  sample.required_gap >= 0.0f &&
                                  std::isfinite(sample.factor_global_cutoff_gap) &&
                                  sample.factor_global_cutoff_gap >= 0.0f &&
                                  sample.factor_global_cutoff_gap > sample.required_gap;

      TopK published = rescored_selected;
      if (!sample.cutoff_certificate || !sample.direct_selected_rescore_exact) {
        sample.direct_fallback_activated = true;
        const auto fallback_start = Clock::now();
        published = direct_residual_lut_topk(fixture_, epoch);
        sample.direct_fallback_ms = elapsed_ms(fallback_start);
        float fallback_error = 0.0f;
        sample.direct_fallback_exact =
            equal_topk(published, fixture_.oracle_topk[epoch], 0.0f, &fallback_error);
        if (!sample.direct_fallback_exact) {
          sample.reason = "direct_residual_lut_fallback_failed_oracle";
          return sample;
        }
      }
      std::size_t matching_ids = 0;
      for (std::size_t rank = 0; rank < kTopK; ++rank) {
        if (published.ids[rank] == fixture_.oracle_topk[epoch].ids[rank]) ++matching_ids;
      }
      sample.recall_at_k = static_cast<double>(matching_ids) / kTopK;
      sample.ordered_ids_exact =
          equal_topk(published, fixture_.oracle_topk[epoch], 0.0f,
                     &sample.topk_max_abs_error);
      if (!sample.ordered_ids_exact) {
        sample.reason = "published_topk_oracle_mismatch";
        return sample;
      }
      for (std::size_t rank = 0; rank < kTopK; ++rank) publication[rank + 1] =
          {published.scores[rank], published.ids[rank]};
      sample.publication_canaries = publication.front().score == -12345.0f &&
                                    publication.back().score == -12345.0f &&
                                    publication.front().id == 0xdeadbeefu &&
                                    publication.back().id == 0xdeadbeefu;
      if (!sample.publication_canaries) {
        sample.reason = "publication_canary_changed";
        return sample;
      }
      sample.success = true;
      sample.wall_ms = elapsed_ms(wall_start);
      const double attributed = sample.q_input_upload_ms + sample.q_execution_ms +
                                sample.q_output_upload_ms + sample.coarse_upload_ms +
                                sample.reset_ms + sample.scan_local_topk_ms +
                                sample.partial_readback_ms + sample.final_select_ms +
                                sample.direct_selected_rescore_ms + sample.direct_fallback_ms;
      sample.unattributed_ms = std::max(0.0, sample.wall_ms - attributed);
      return sample;
    } catch (const std::exception& exception) {
      try {
        queue_.wait();
      } catch (...) {
      }
      sample.reason = exception.what();
      sample.wall_ms = elapsed_ms(wall_start);
      sample.publication_canaries = publication.front().id == 0xdeadbeefu &&
                                    publication.back().id == 0xdeadbeefu;
      return sample;
    }
  }

  std::uint64_t persistent_upload_bytes() const {
    return codebooks_.bytes() + t_.bytes() + codes_.bytes() + blocks_.bytes();
  }
  std::uint64_t peak_owned_device_bytes() const {
    return persistent_upload_bytes() + query_.bytes() + q_.bytes() + coarse_.bytes() +
           partials_.bytes() + error_.bytes();
  }
  std::uint64_t partial_readback_bytes() const { return partials_.bytes() + error_.bytes(); }
  std::uint64_t expanded_i32_bytes_avoided() const {
    return static_cast<std::uint64_t>(fixture_.candidates) * kPqM * sizeof(std::int32_t);
  }
  std::uint64_t full_score_readback_bytes_avoided() const {
    const std::uint64_t full = static_cast<std::uint64_t>(fixture_.candidates) * sizeof(float);
    return full > partial_readback_bytes() ? full - partial_readback_bytes() : 0;
  }
  double persistent_upload_ms() const { return persistent_upload_ms_; }
  std::size_t candidate_count() const { return fixture_.candidates; }

 private:
  sycl::event submit_projection() {
    const float* query = query_.get();
    const float* codebooks = codebooks_.get();
    float* q = q_.get();
    return queue_.parallel_for(
        sycl::range<1>(static_cast<std::size_t>(kPqM) * kKs), [=](sycl::id<1> index) {
          const std::size_t flat = index[0];
          const std::uint32_t m = static_cast<std::uint32_t>(flat / kKs);
          const std::uint32_t code = static_cast<std::uint32_t>(flat % kKs);
          float dot = 0.0f;
          for (std::uint32_t j = 0; j < kSubDim; ++j) {
            dot += query[static_cast<std::size_t>(m) * kSubDim + j] *
                   codebooks[(static_cast<std::size_t>(m) * kKs + code) * kSubDim + j];
          }
          q[flat] = -2.0f * dot;
        });
  }

  sycl::event submit_scan() {
    const float* q = q_.get();
    const float* t = t_.get();
    const float* coarse = coarse_.get();
    const std::uint8_t* codes = codes_.get();
    const Block* blocks = blocks_.get();
    Pair* partials = partials_.get();
    std::int32_t* error = error_.get();
    const std::size_t block_count = fixture_.blocks.size();
    return queue_.submit([&](sycl::handler& handler) {
      sycl::local_accessor<float, 1> local_scores(sycl::range<1>(kBlockRows), handler);
      sycl::local_accessor<std::uint32_t, 1> local_ids(sycl::range<1>(kBlockRows), handler);
      handler.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(block_count * kBlockRows),
                            sycl::range<1>(kBlockRows)),
          [=](sycl::nd_item<1> item) {
            const std::size_t block_index = item.get_group_linear_id();
            const std::uint32_t lane = static_cast<std::uint32_t>(item.get_local_linear_id());
            const Block block = blocks[block_index];
            float score = kInfinity;
            std::uint32_t id = kInvalidId;
            if (lane < block.rows) {
              // Coarse distance is constant within a list. Scan and sort only
              // the factorized residual contribution, then restore coarse on
              // the bounded partials instead of once per candidate.
              score = 0.0f;
              const std::uint8_t* row_codes =
                  codes + block.code_offset + static_cast<std::uint64_t>(lane) * kPqM;
              bool valid = block.list < kLists;
              for (std::uint32_t m = 0; m < kPqM && valid; ++m) {
                const std::uint32_t code = row_codes[m];
                const std::size_t mk = static_cast<std::size_t>(m) * kKs + code;
                const std::size_t lmk =
                    (static_cast<std::size_t>(block.list) * kPqM + m) * kKs + code;
                score += q[mk] + t[lmk];
              }
              if (!valid || !sycl::isfinite(score)) {
                sycl::atomic_ref<std::int32_t, sycl::memory_order::relaxed,
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
            for (std::uint32_t width = 2; width <= kBlockRows; width <<= 1u) {
              for (std::uint32_t stride = width >> 1u; stride > 0; stride >>= 1u) {
                const std::uint32_t peer = lane ^ stride;
                if (peer > lane) {
                  const bool ascending = (lane & width) == 0;
                  const float lhs_score = local_scores[lane];
                  const std::uint32_t lhs_id = local_ids[lane];
                  const float rhs_score = local_scores[peer];
                  const std::uint32_t rhs_id = local_ids[peer];
                  const bool rhs_better = rhs_score < lhs_score ||
                                          (rhs_score == lhs_score && rhs_id < lhs_id);
                  const bool lhs_better = lhs_score < rhs_score ||
                                          (lhs_score == rhs_score && lhs_id < rhs_id);
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
            if (lane <= kTopK) {
              const float restored = local_ids[lane] == kInvalidId
                                         ? local_scores[lane]
                                         : local_scores[lane] + coarse[block.list];
              partials[block_index * (kTopK + 1) + lane] = {restored, local_ids[lane]};
            }
          });
    });
  }

  sycl::queue& queue_;
  const Fixture& fixture_;
  DeviceAllocation<float> codebooks_;
  DeviceAllocation<float> t_;
  DeviceAllocation<std::uint8_t> codes_;
  DeviceAllocation<Block> blocks_;
  DeviceAllocation<float> query_;
  DeviceAllocation<float> q_;
  DeviceAllocation<float> coarse_;
  DeviceAllocation<Pair> partials_;
  DeviceAllocation<std::int32_t> error_;
  double persistent_upload_ms_ = 0.0;
};

ProjectionSummary measure_gpu_projection(GpuFactorRunner& runner, const Options& options) {
  ProjectionSummary result;
  result.status = "ok";
  result.persistent_bytes = static_cast<std::uint64_t>(kPqM) * kKs * kSubDim * sizeof(float);
  result.per_call_input_bytes = kDim * sizeof(float);
  result.per_call_output_bytes = static_cast<std::uint64_t>(kPqM) * kKs * sizeof(float);
  result.launches_per_call = 1;
  result.first = runner.project_gpu(0);
  for (int i = 0; i < options.warmups; ++i) runner.project_gpu(static_cast<std::uint32_t>(i & 1));
  for (int i = 0; i < options.repeats; ++i) {
    result.warm.push_back(runner.project_gpu(static_cast<std::uint32_t>(i & 1)));
  }
  return result;
}

JoinedSummary measure_joined(GpuFactorRunner& runner, QProducer producer,
                             NpuProjector* npu, const Options& options) {
  JoinedSummary result;
  result.producer = producer;
  if (producer == QProducer::kNpu && !npu) {
    result.reason = "NPU Q projection unavailable";
    return result;
  }
  result.first = runner.run(producer, 0, npu);
  if (!result.first.success) {
    result.reason = "first_joined_call_failed: " + result.first.reason;
    return result;
  }
  for (int i = 0; i < options.warmups; ++i) {
    const JoinedSample warmup = runner.run(producer, static_cast<std::uint32_t>(i & 1), npu);
    if (!warmup.success) {
      result.reason = "joined_warmup_failed: " + warmup.reason;
      return result;
    }
  }
  for (int i = 0; i < options.repeats; ++i) {
    JoinedSample sample = runner.run(producer, static_cast<std::uint32_t>(i & 1), npu);
    if (!sample.success) {
      result.reason = "joined_measured_call_failed: " + sample.reason;
      return result;
    }
    result.warm.push_back(std::move(sample));
  }
  result.status = "ok";
  return result;
}

struct CpuDirectSample {
  bool success = false;
  double wall_ms = 0.0;
  double direct_lut_ms = 0.0;
  double scan_select_ms = 0.0;
  float max_abs_error = 0.0f;
  double recall_at_k = 0.0;
};

struct CpuDirectSummary {
  CpuDirectSample first;
  std::vector<CpuDirectSample> warm;
};

CpuDirectSample run_cpu_direct(const Fixture& fixture, std::uint32_t epoch) {
  CpuDirectSample sample;
  const auto wall_start = Clock::now();
  const auto lut_start = Clock::now();
  std::vector<float> lut(static_cast<std::size_t>(kLists) * kPqM * kKs);
  for (std::uint32_t list = 0; list < kLists; ++list) {
    for (std::uint32_t m = 0; m < kPqM; ++m) {
      for (std::uint32_t code = 0; code < kKs; ++code) {
        float distance = 0.0f;
        for (std::uint32_t j = 0; j < kSubDim; ++j) {
          const float diff = fixture.queries[epoch][m * kSubDim + j] -
                             fixture.centroids[static_cast<std::size_t>(list) * kDim +
                                               m * kSubDim + j] -
                             fixture.codebooks[(static_cast<std::size_t>(m) * kKs + code) *
                                               kSubDim + j];
          distance += diff * diff;
        }
        lut[(static_cast<std::size_t>(list) * kPqM + m) * kKs + code] = distance;
      }
    }
  }
  sample.direct_lut_ms = elapsed_ms(lut_start);
  const auto scan_start = Clock::now();
  TopK selected;
  for (std::uint32_t list = 0; list < kLists; ++list) {
    const ListSpan span = fixture.lists[list];
    for (std::uint32_t local = 0; local < span.rows; ++local) {
      const std::uint32_t row = span.first_id + local;
      float score = 0.0f;
      for (std::uint32_t m = 0; m < kPqM; ++m) {
        const std::uint32_t code = fixture.codes[static_cast<std::size_t>(row) * kPqM + m];
        score += lut[(static_cast<std::size_t>(list) * kPqM + m) * kKs + code];
      }
      insert_topk(selected, score, row);
    }
  }
  sample.scan_select_ms = elapsed_ms(scan_start);
  std::size_t matches = 0;
  for (std::size_t rank = 0; rank < kTopK; ++rank) {
    if (selected.ids[rank] == fixture.oracle_topk[epoch].ids[rank]) ++matches;
  }
  sample.recall_at_k = static_cast<double>(matches) / kTopK;
  sample.success = equal_topk(selected, fixture.oracle_topk[epoch], 0.0f,
                              &sample.max_abs_error);
  sample.wall_ms = elapsed_ms(wall_start);
  return sample;
}

CpuDirectSummary measure_cpu_direct(const Fixture& fixture, const Options& options) {
  CpuDirectSummary result;
  result.first = run_cpu_direct(fixture, 0);
  if (!result.first.success) throw Failure("direct residual LUT CPU first call failed oracle");
  for (int i = 0; i < options.warmups; ++i) {
    if (!run_cpu_direct(fixture, static_cast<std::uint32_t>(i & 1)).success) {
      throw Failure("direct residual LUT CPU warmup failed oracle");
    }
  }
  for (int i = 0; i < options.repeats; ++i) {
    CpuDirectSample sample = run_cpu_direct(fixture, static_cast<std::uint32_t>(i & 1));
    if (!sample.success) throw Failure("direct residual LUT CPU measured call failed oracle");
    result.warm.push_back(sample);
  }
  return result;
}

struct AmbiguousFallbackEvidence {
  bool activated = false;
  bool exact = false;
  bool publication_canaries = false;
  bool no_partial_publication = false;
  float score_error_band = 0.0f;
  float cutoff_gap = 0.0f;
  float required_gap = 0.0f;
  std::uint32_t fallback_count = 0;
  std::string reason;
};

AmbiguousFallbackEvidence exercise_ambiguous_fallback() {
  // One residual subspace, q=c=0. IDs 31,32,33 deliberately share w=31,
  // so factor and direct LUTs are exact but the k/k+1 membership boundary is tied.
  std::array<float, kTopK + 2> direct_lut{};
  std::array<std::uint8_t, kTopK + 2> codes{};
  std::vector<Pair> factor_candidates;
  factor_candidates.reserve(kTopK + 2);
  for (std::uint32_t id = 0; id < kTopK + 2; ++id) {
    const std::uint32_t code = id;
    codes[id] = static_cast<std::uint8_t>(code);
    const float w = static_cast<float>(std::min<std::uint32_t>(id, kTopK - 1));
    direct_lut[code] = w * w;
    const float q = 0.0f;
    const float t = w * w;
    factor_candidates.push_back({q + t, id});
  }
  std::partial_sort(factor_candidates.begin(), factor_candidates.begin() + (kTopK + 1),
                    factor_candidates.end(), [](const Pair& lhs, const Pair& rhs) {
                      return better(lhs.score, lhs.id, rhs.score, rhs.id);
                    });
  AmbiguousFallbackEvidence evidence;
  evidence.score_error_band = 0.0f;
  evidence.required_gap = 0.0f;
  evidence.cutoff_gap = factor_candidates[kTopK].score -
                        factor_candidates[kTopK - 1].score;
  const bool certified = evidence.cutoff_gap > evidence.required_gap;

  std::array<Pair, kTopK + 2> publication;
  for (Pair& pair : publication) pair = {-12345.0f, 0xdeadbeefu};
  evidence.no_partial_publication = publication[1].id == 0xdeadbeefu;
  if (!certified) {
    evidence.activated = true;
    evidence.fallback_count = 1;
    evidence.reason = "ambiguous_k_k_plus_one_cutoff_gap_not_strictly_above_2E";
    TopK direct;
    for (std::uint32_t id = 0; id < kTopK + 2; ++id) {
      insert_topk(direct, direct_lut[codes[id]], id);
    }
    for (std::size_t rank = 0; rank < kTopK; ++rank) {
      publication[rank + 1] = {direct.scores[rank], direct.ids[rank]};
    }
    evidence.exact = true;
    for (std::size_t rank = 0; rank < kTopK; ++rank) {
      if (publication[rank + 1].id != rank ||
          publication[rank + 1].score != direct_lut[rank]) {
        evidence.exact = false;
      }
    }
  }
  evidence.publication_canaries = publication.front().score == -12345.0f &&
                                  publication.back().score == -12345.0f &&
                                  publication.front().id == 0xdeadbeefu &&
                                  publication.back().id == 0xdeadbeefu;
  return evidence;
}

template <typename Sample, typename Accessor>
Distribution distribution_of(const std::vector<Sample>& samples, Accessor accessor) {
  Distribution result;
  result.values.reserve(samples.size());
  for (const Sample& sample : samples) result.values.push_back(accessor(sample));
  return result;
}

void append_distribution(std::ostringstream& out, const Distribution& distribution) {
  out << "{\"p50\":" << distribution.p50() << ",\"p95\":" << distribution.p95()
      << ",\"p99\":" << distribution.p99() << ",\"samples\":[";
  for (std::size_t i = 0; i < distribution.values.size(); ++i) {
    if (i) out << ',';
    out << distribution.values[i];
  }
  out << "]}";
}

void append_projection_sample(std::ostringstream& out, const ProjectionSample& sample) {
  out << "{\"wall_ms\":" << sample.wall_ms
      << ",\"input_or_upload_ms\":" << sample.input_ms
      << ",\"execution_ms\":" << sample.execution_ms
      << ",\"output_or_readback_ms\":" << sample.output_ms
      << ",\"finite\":" << (sample.finite ? "true" : "false")
      << ",\"max_abs_error\":" << sample.max_abs_error << '}';
}

void append_projection_summary(std::ostringstream& out, const ProjectionSummary& summary) {
  out << "{\"status\":" << json_escape(summary.status);
  if (!summary.reason.empty()) out << ",\"reason\":" << json_escape(summary.reason);
  if (summary.status != "ok") {
    out << '}';
    return;
  }
  out << ",\"cold_compile_ms\":" << summary.compile_ms
      << ",\"request_setup_ms\":" << summary.setup_ms
      << ",\"persistent_compiled_graph\":"
      << (summary.persistent_compiled_graph ? "true" : "false")
      << ",\"persistent_reused_request\":"
      << (summary.persistent_request ? "true" : "false")
      << ",\"first_process_call\":";
  append_projection_sample(out, summary.first);
  out << ",\"warm\":{";
  out << "\"wall_ms\":";
  append_distribution(out, distribution_of(summary.warm,
                                             [](const ProjectionSample& s) { return s.wall_ms; }));
  out << ",\"input_or_upload_ms\":";
  append_distribution(out, distribution_of(summary.warm,
                                             [](const ProjectionSample& s) { return s.input_ms; }));
  out << ",\"execution_ms\":";
  append_distribution(out, distribution_of(summary.warm,
                                             [](const ProjectionSample& s) { return s.execution_ms; }));
  out << ",\"output_or_readback_ms\":";
  append_distribution(out, distribution_of(summary.warm,
                                             [](const ProjectionSample& s) { return s.output_ms; }));
  float max_error = summary.first.max_abs_error;
  bool finite = summary.first.finite;
  for (const ProjectionSample& sample : summary.warm) {
    max_error = std::max(max_error, sample.max_abs_error);
    finite = finite && sample.finite;
  }
  out << "},\"correctness\":{\"finite_all\":" << (finite ? "true" : "false")
      << ",\"max_abs_error\":" << max_error
      << "},\"traffic\":{\"persistent_bytes\":" << summary.persistent_bytes
      << ",\"per_call_input_bytes\":" << summary.per_call_input_bytes
      << ",\"per_call_output_bytes\":" << summary.per_call_output_bytes
      << "},\"calls\":{\"requests_per_call\":" << summary.requests_per_call
      << ",\"kernel_launches_per_call\":" << summary.launches_per_call << "}}";
}

void append_joined_sample(std::ostringstream& out, const JoinedSample& sample) {
  out << "{\"success\":" << (sample.success ? "true" : "false");
  if (!sample.reason.empty()) out << ",\"reason\":" << json_escape(sample.reason);
  out << ",\"wall_ms\":" << sample.wall_ms
      << ",\"q_input_upload_ms\":" << sample.q_input_upload_ms
      << ",\"q_execution_ms\":" << sample.q_execution_ms
      << ",\"q_output_upload_ms\":" << sample.q_output_upload_ms
      << ",\"coarse_upload_ms\":" << sample.coarse_upload_ms
      << ",\"reset_ms\":" << sample.reset_ms
      << ",\"scan_local_topk_ms\":" << sample.scan_local_topk_ms
      << ",\"partial_readback_ms\":" << sample.partial_readback_ms
      << ",\"final_select_ms\":" << sample.final_select_ms
      << ",\"direct_selected_rescore_ms\":" << sample.direct_selected_rescore_ms
      << ",\"direct_fallback_ms\":" << sample.direct_fallback_ms
      << ",\"unattributed_ms\":" << sample.unattributed_ms
      << ",\"npu_requests\":" << sample.npu_requests
      << ",\"gpu_kernel_launches\":" << sample.gpu_kernel_launches
      << ",\"gpu_queue_submissions\":" << sample.gpu_queue_submissions
      << ",\"host_synchronizations\":" << sample.host_synchronizations
      << ",\"gpu_host_synchronizations\":" << sample.gpu_host_synchronizations
      << ",\"npu_host_waits\":" << sample.npu_host_waits
      << ",\"q_max_abs_error\":" << sample.q_max_abs_error
      << ",\"topk_max_abs_error\":" << sample.topk_max_abs_error
      << ",\"measured_host_factor_vs_direct_band\":"
      << sample.measured_host_factor_vs_direct_band
      << ",\"score_error_band\":" << sample.score_error_band
      << ",\"score_error_band_basis\":\"bounded_integer_exact_ieee_f32_only_not_general_f32; npu certification additionally requires per-call bit-exact full Q\""
      << ",\"required_gap_2E\":" << sample.required_gap
      << ",\"direct_global_cutoff_gap\":" << sample.global_cutoff_gap
      << ",\"factor_global_cutoff_gap\":" << sample.factor_global_cutoff_gap
      << ",\"direct_min_block_local_cutoff_gap\":" << sample.min_local_cutoff_gap
      << ",\"factor_min_block_local_cutoff_gap\":"
      << sample.factor_min_local_cutoff_gap
      << ",\"direct_min_selected_adjacent_gap\":"
      << sample.min_selected_adjacent_gap
      << ",\"recall_at_32\":" << sample.recall_at_k
      << ",\"factor_candidate_recall_at_32\":" << sample.factor_candidate_recall_at_k
      << ",\"finite\":" << (sample.finite ? "true" : "false")
      << ",\"ordered_ids_exact\":" << (sample.ordered_ids_exact ? "true" : "false")
      << ",\"factor_candidate_ordered_ids_exact\":"
      << (sample.factor_candidate_ordered_ids_exact ? "true" : "false")
      << ",\"q_per_call_full_table_validated\":"
      << (sample.q_per_call_validated ? "true" : "false")
      << ",\"bounded_integer_analytic_band\":"
      << (sample.bounded_integer_analytic_band ? "true" : "false")
      << ",\"certificate_band_available\":"
      << (sample.bounded_integer_analytic_band ? "true" : "false")
      << ",\"local_k_plus_one_complete\":"
      << (sample.local_k_plus_one_complete ? "true" : "false")
      << ",\"cutoff_certified\":" << (sample.cutoff_certificate ? "true" : "false")
      << ",\"direct_selected_rescore_exact\":"
      << (sample.direct_selected_rescore_exact ? "true" : "false")
      << ",\"direct_fallback_activated\":"
      << (sample.direct_fallback_activated ? "true" : "false")
      << ",\"direct_fallback_exact\":"
      << (sample.direct_fallback_exact ? "true" : "false")
      << ",\"publication_canaries\":" << (sample.publication_canaries ? "true" : "false")
      << '}';
}

void append_joined_summary(std::ostringstream& out, const JoinedSummary& summary,
                           const GpuFactorRunner& runner) {
  out << "{\"lane\":" << json_escape(producer_name(summary.producer))
      << ",\"status\":" << json_escape(summary.status);
  if (!summary.reason.empty()) out << ",\"reason\":" << json_escape(summary.reason);
  if (summary.status != "ok") {
    out << '}';
    return;
  }
  out << ",\"first_joined_call_after_q_microbench\":";
  append_joined_sample(out, summary.first);
  out << ",\"warm\":{";
  const auto emit = [&](std::string_view name, auto accessor, bool comma) {
    if (comma) out << ',';
    out << json_escape(name) << ':';
    append_distribution(out, distribution_of(summary.warm, accessor));
  };
  emit("wall_ms", [](const JoinedSample& s) { return s.wall_ms; }, false);
  emit("q_input_upload_ms", [](const JoinedSample& s) { return s.q_input_upload_ms; }, true);
  emit("q_execution_ms", [](const JoinedSample& s) { return s.q_execution_ms; }, true);
  emit("q_output_upload_ms", [](const JoinedSample& s) { return s.q_output_upload_ms; }, true);
  emit("coarse_upload_ms", [](const JoinedSample& s) { return s.coarse_upload_ms; }, true);
  emit("scan_local_topk_ms", [](const JoinedSample& s) { return s.scan_local_topk_ms; }, true);
  emit("partial_readback_ms", [](const JoinedSample& s) { return s.partial_readback_ms; }, true);
  emit("final_select_ms", [](const JoinedSample& s) { return s.final_select_ms; }, true);
  emit("direct_selected_rescore_ms",
       [](const JoinedSample& s) { return s.direct_selected_rescore_ms; }, true);
  emit("direct_fallback_ms", [](const JoinedSample& s) { return s.direct_fallback_ms; }, true);
  emit("unattributed_ms", [](const JoinedSample& s) { return s.unattributed_ms; }, true);
  out << "},\"throughput\":{\"candidates_per_second_at_p50\":"
      << (1000.0 * static_cast<double>(runner.candidate_count()) /
          distribution_of(summary.warm, [](const JoinedSample& s) { return s.wall_ms; }).p50())
      << "}";

  bool finite = summary.first.finite;
  bool canaries = summary.first.publication_canaries;
  bool exact = summary.first.ordered_ids_exact;
  bool certified = summary.first.cutoff_certificate;
  bool direct_rescore_exact = summary.first.direct_selected_rescore_exact;
  bool factor_candidate_exact = summary.first.factor_candidate_ordered_ids_exact;
  std::uint32_t fallback_count = summary.first.direct_fallback_activated ? 1u : 0u;
  double min_recall = summary.first.recall_at_k;
  float max_error = summary.first.topk_max_abs_error;
  for (const JoinedSample& sample : summary.warm) {
    finite = finite && sample.finite;
    canaries = canaries && sample.publication_canaries;
    exact = exact && sample.ordered_ids_exact;
    certified = certified && sample.cutoff_certificate;
    direct_rescore_exact = direct_rescore_exact && sample.direct_selected_rescore_exact;
    factor_candidate_exact =
        factor_candidate_exact && sample.factor_candidate_ordered_ids_exact;
    if (sample.direct_fallback_activated) ++fallback_count;
    min_recall = std::min(min_recall, sample.recall_at_k);
    max_error = std::max(max_error, sample.topk_max_abs_error);
  }
  out << ",\"correctness\":{\"finite_all\":" << (finite ? "true" : "false")
      << ",\"publication_canaries_all\":" << (canaries ? "true" : "false")
      << ",\"ordered_ids_exact_all\":" << (exact ? "true" : "false")
      << ",\"cutoff_certified_all\":" << (certified ? "true" : "false")
      << ",\"direct_selected_rescore_exact_all\":"
      << (direct_rescore_exact ? "true" : "false")
      << ",\"factor_candidate_ordered_ids_exact_all\":"
      << (factor_candidate_exact ? "true" : "false")
      << ",\"direct_fallback_count_first_plus_warm\":" << fallback_count
      << ",\"min_recall_at_32\":" << min_recall
      << ",\"max_distance_abs_error\":" << max_error << "},\"traffic\":{";
  const std::uint64_t query_bytes = kDim * sizeof(float);
  const std::uint64_t q_bytes = static_cast<std::uint64_t>(kPqM) * kKs * sizeof(float);
  const std::uint64_t coarse_bytes = kLists * sizeof(float);
  std::uint64_t copied = coarse_bytes + runner.partial_readback_bytes();
  if (summary.producer == QProducer::kCpu) copied += q_bytes;
  if (summary.producer == QProducer::kGpu) copied += query_bytes;
  if (summary.producer == QProducer::kNpu) copied += query_bytes + q_bytes;
  out << "\"per_call_visible_copied_bytes\":" << copied
      << ",\"partial_readback_bytes\":" << runner.partial_readback_bytes()
      << ",\"expanded_i32_bytes_avoided\":" << runner.expanded_i32_bytes_avoided()
      << ",\"full_score_readback_bytes_avoided\":"
      << runner.full_score_readback_bytes_avoided() << "}}";
}

void append_cpu_direct_summary(std::ostringstream& out, const CpuDirectSummary& summary,
                               std::size_t candidates) {
  out << "{\"status\":\"ok\",\"label\":\"scalar_direct_residual_lut_oracle_not_optimized_cpu_baseline\",";
  out << "\"first_process_call\":{\"wall_ms\":" << summary.first.wall_ms
      << ",\"direct_lut_ms\":" << summary.first.direct_lut_ms
      << ",\"scan_select_ms\":" << summary.first.scan_select_ms << "},\"warm\":{";
  out << "\"wall_ms\":";
  append_distribution(out, distribution_of(summary.warm,
                                             [](const CpuDirectSample& s) { return s.wall_ms; }));
  out << ",\"direct_lut_ms\":";
  append_distribution(out, distribution_of(summary.warm,
                                             [](const CpuDirectSample& s) { return s.direct_lut_ms; }));
  out << ",\"scan_select_ms\":";
  append_distribution(out, distribution_of(summary.warm,
                                             [](const CpuDirectSample& s) { return s.scan_select_ms; }));
  const auto wall = distribution_of(summary.warm, [](const CpuDirectSample& s) { return s.wall_ms; });
  float max_error = summary.first.max_abs_error;
  double min_recall = summary.first.recall_at_k;
  for (const CpuDirectSample& sample : summary.warm) {
    max_error = std::max(max_error, sample.max_abs_error);
    min_recall = std::min(min_recall, sample.recall_at_k);
  }
  out << "},\"candidates_per_second_at_p50\":"
      << (1000.0 * static_cast<double>(candidates) / wall.p50())
      << ",\"correctness\":{\"min_recall_at_32\":" << min_recall
      << ",\"max_distance_abs_error\":" << max_error
      << ",\"finite\":true,\"ordered_ids_exact\":true}}";
}

std::filesystem::path prepare_output_path(const Options& options) {
  if (!options.output) return {};
  std::error_code ec;
  const std::filesystem::path absolute = std::filesystem::absolute(*options.output, ec);
  if (ec || absolute.empty()) throw Failure("cannot resolve --output path");
  if (std::filesystem::exists(absolute, ec) || ec) {
    throw Failure("--output already exists or cannot be inspected: " + absolute.string());
  }
  const std::filesystem::path parent = absolute.parent_path();
  if (parent.empty()) throw Failure("--output has no resolvable parent");
  if (!std::filesystem::exists(parent, ec)) {
    if (!std::filesystem::create_directories(parent, ec) || ec) {
      throw Failure("cannot create --output parent: " + parent.string());
    }
  }
  if (!std::filesystem::is_directory(parent, ec) || ec) {
    throw Failure("--output parent is not a directory: " + parent.string());
  }
  return absolute;
}

void write_output_exclusive(const std::filesystem::path& path, const std::string& payload) {
  if (path.empty()) return;
  HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    throw Failure("cannot create exclusive output file (win32=" +
                  std::to_string(static_cast<unsigned long>(GetLastError())) + "): " +
                  path.string());
  }
  bool ok = true;
  std::size_t offset = 0;
  while (offset < payload.size()) {
    const DWORD chunk = static_cast<DWORD>(
        std::min<std::size_t>(payload.size() - offset, std::numeric_limits<DWORD>::max()));
    DWORD written = 0;
    if (!WriteFile(file, payload.data() + offset, chunk, &written, nullptr) || written != chunk) {
      ok = false;
      break;
    }
    offset += written;
  }
  if (!FlushFileBuffers(file)) ok = false;
  CloseHandle(file);
  if (!ok) {
    DeleteFileW(path.c_str());
    throw Failure("failed to write complete output file: " + path.string());
  }
}

int run(const Options& options) {
  const std::filesystem::path output_path = prepare_output_path(options);
  const auto fixture_start = Clock::now();
  Fixture fixture = make_fixture(options.candidates);
  const double fixture_setup_ms = elapsed_ms(fixture_start);
  const bool algebra_gate = fixture.gate.finite && fixture.gate.exact_topk &&
                            fixture.gate.max_abs_error == 0.0f;
  if (!algebra_gate) throw Failure("bounded residual-PQ factorization gate failed");
  const AmbiguousFallbackEvidence ambiguous_fallback = exercise_ambiguous_fallback();
  if (!ambiguous_fallback.activated || !ambiguous_fallback.exact ||
      !ambiguous_fallback.publication_canaries ||
      !ambiguous_fallback.no_partial_publication) {
    throw Failure("deliberate ambiguous-cutoff fallback gate failed");
  }

  const ProjectionSummary cpu_projection = measure_cpu_projection(fixture, options);
  const CpuDirectSummary cpu_direct = measure_cpu_direct(fixture, options);

  ProjectionSummary gpu_projection;
  JoinedSummary cpu_gpu_joined;
  cpu_gpu_joined.producer = QProducer::kCpu;
  JoinedSummary gpu_gpu_joined;
  gpu_gpu_joined.producer = QProducer::kGpu;
  JoinedSummary npu_gpu_joined;
  npu_gpu_joined.producer = QProducer::kNpu;
  std::string gpu_reason;
  std::string gpu_name;
  std::string gpu_driver;
  std::string gpu_platform;
  bool gpu_host_unified_memory = false;
  std::unique_ptr<sycl::queue> queue;
  std::unique_ptr<GpuFactorRunner> gpu_runner;
  const std::optional<sycl::device> selected_gpu =
      select_integrated_intel_l0_gpu(gpu_reason);
  if (selected_gpu) {
    try {
      queue = std::make_unique<sycl::queue>(
          *selected_gpu,
          sycl::property_list{sycl::property::queue::in_order{},
                              sycl::property::queue::enable_profiling{}});
      gpu_name = selected_gpu->get_info<sycl::info::device::name>();
      gpu_driver = selected_gpu->get_info<sycl::info::device::driver_version>();
      gpu_platform = selected_gpu->get_platform().get_info<sycl::info::platform::name>();
      gpu_host_unified_memory = has_unified_host_memory(*selected_gpu);
      gpu_runner = std::make_unique<GpuFactorRunner>(*queue, fixture);
      gpu_projection = measure_gpu_projection(*gpu_runner, options);
      cpu_gpu_joined = measure_joined(*gpu_runner, QProducer::kCpu, nullptr, options);
      gpu_gpu_joined = measure_joined(*gpu_runner, QProducer::kGpu, nullptr, options);
    } catch (const std::exception& exception) {
      gpu_reason = exception.what();
      gpu_projection.status = "unavailable";
      gpu_projection.reason = gpu_reason;
      if (cpu_gpu_joined.status != "ok") cpu_gpu_joined.reason = gpu_reason;
      if (gpu_gpu_joined.status != "ok") gpu_gpu_joined.reason = gpu_reason;
    }
  } else {
    gpu_projection.status = "unavailable";
    gpu_projection.reason = gpu_reason;
    cpu_gpu_joined.reason = gpu_reason;
    gpu_gpu_joined.reason = gpu_reason;
  }

  // Initialize and exercise the SYCL Level Zero path before OpenVINO's NPU
  // plugin. On this Windows stack, NPU-first Level Zero initialization can
  // hide the GPU driver from a later Unified Runtime enumeration.
  ProjectionSummary npu_projection;
  std::unique_ptr<NpuProjector> npu;
  std::string npu_device;
  std::string npu_cache;
  std::string npu_compile_scope;
  std::string npu_first_compile_error;
  std::string npu_second_compile_error;
  try {
    npu = std::make_unique<NpuProjector>(fixture, options);
    npu_device = npu->device_name();
    npu_cache = npu->cache_dir().string();
    npu_compile_scope = npu->compile_property_scope();
    npu_first_compile_error = npu->first_compile_error();
    npu_second_compile_error = npu->second_compile_error();
    npu_projection = measure_npu_projection(*npu, options);
  } catch (const std::exception& exception) {
    npu_projection.status = "unavailable";
    npu_projection.reason = exception.what();
    npu.reset();
  }
  if (gpu_runner) {
    npu_gpu_joined = measure_joined(*gpu_runner, QProducer::kNpu, npu.get(), options);
  } else {
    npu_gpu_joined.reason = npu ? gpu_reason : npu_projection.reason;
  }

  const bool gpu_gate = gpu_projection.status == "ok" &&
                        cpu_gpu_joined.status == "ok" && gpu_gpu_joined.status == "ok";
  const bool overall_gate = algebra_gate && gpu_gate;
  const ov::Version openvino_version = ov::get_openvino_version();

  std::uint64_t host_fixture_bytes =
      static_cast<std::uint64_t>(fixture.centroids.size() + fixture.codebooks.size() +
                                 fixture.persistent_t.size()) * sizeof(float) +
      fixture.codes.size() +
      static_cast<std::uint64_t>(fixture.lists.size()) * sizeof(ListSpan) +
      static_cast<std::uint64_t>(fixture.blocks.size()) * sizeof(Block);
  for (std::size_t epoch = 0; epoch < 2; ++epoch) {
    host_fixture_bytes += static_cast<std::uint64_t>(fixture.queries[epoch].size() +
                                                     fixture.coarse[epoch].size() +
                                                     fixture.q_tables[epoch].size()) * sizeof(float);
  }

  std::vector<float> negative_query = fixture.queries[0];
  negative_query[0] = std::numeric_limits<float>::quiet_NaN();
  const bool nonfinite_preflight_rejected = !finite_vector(negative_query);

  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "{\"schema\":\"ovvs.pq_lut_factor_bench.v1\",";
  out << "\"status\":" << json_escape(overall_gate ?
      (npu_projection.status == "ok" ? "ok" : "ok_with_npu_unavailable") : "failed") << ',';
  out << "\"scope\":\"synthetic_joined_residual_pq_stage_microbenchmark_not_end_to_end_ivf_search\",";
  out << "\"acceleration_claim\":false,\"integration_status\":\"standalone_not_routed\",";
  out << "\"limitations\":["
         "\"coarse assignment and exact vector refinement are excluded\","
         "\"CPU direct lane is a scalar oracle rather than the optimized ovVS CPU baseline\","
         "\"integer-bounded score equivalence is not a general f32 numerical proof\","
         "\"device stage timestamps are queue/request walls rather than a common cross-device clock\"],";
  out << "\"fixture\":{\"candidates\":" << options.candidates
      << ",\"queries\":2,\"dim\":" << kDim << ",\"pq_m\":" << kPqM
      << ",\"dsub\":" << kSubDim << ",\"ks\":" << kKs
      << ",\"lists\":" << kLists << ",\"blocks\":" << fixture.blocks.size()
      << ",\"topk\":" << kTopK << ",\"setup_ms\":" << fixture_setup_ms
      << ",\"layout\":\"list-major contiguous codes; dense ordinal equals synthetic ID\"},";
  out << "\"algebra_gate\":{\"identity\":"
      << json_escape("d=B(q,l)+sum_m[Q(q,m,k)+T(l,m,k)]")
      << ",\"bounded_integer_fixture\":true,\"finite\":"
      << (fixture.gate.finite ? "true" : "false")
      << ",\"compared_scores\":" << fixture.gate.compared_scores
      << ",\"bit_equal_scores\":" << fixture.gate.bit_equal_scores
      << ",\"bit_equal_all\":" << (fixture.gate.bit_equal ? "true" : "false")
      << ",\"max_abs_error\":" << fixture.gate.max_abs_error
      << ",\"bounded_integer_analytic_exact_under_ieee_f32\":"
      << (fixture.integer_exact_under_f32 ? "true" : "false")
      << ",\"exact_integer_limit\":16777216.0"
      << ",\"max_abs_q_term\":" << fixture.max_abs_q_term
      << ",\"max_abs_t_term\":" << fixture.max_abs_t_term
      << ",\"max_abs_coarse\":" << fixture.max_abs_coarse
      << ",\"max_abs_factor_score\":" << fixture.max_abs_factor_score
      << ",\"max_abs_accumulator_bound\":" << fixture.max_abs_accumulator_bound
      << ",\"exact_ordered_top32\":" << (fixture.gate.exact_topk ? "true" : "false")
      << ",\"query0_direct_global_cutoff_gap\":" << fixture.cutoff_gap[0]
      << ",\"query1_direct_global_cutoff_gap\":" << fixture.cutoff_gap[1]
      << ",\"query0_min_block_local_cutoff_gap\":" << fixture.min_local_cutoff_gap[0]
      << ",\"query1_min_block_local_cutoff_gap\":" << fixture.min_local_cutoff_gap[1]
      << ",\"query0_min_selected_adjacent_gap\":"
      << fixture.min_selected_adjacent_gap[0]
      << ",\"query1_min_selected_adjacent_gap\":"
      << fixture.min_selected_adjacent_gap[1]
      << ",\"cpu_igpu_distance_scope\":\"factorized-score equivalent only on this bounded integer fixture\",";
  out << "\"anchor_formulation_tested\":false,"
         "\"anchor_disposition\":\"raw bounded f32 Q/T/coarse passed exact IDs, recall 1.0, and <=0.001 parity gate; conditional anchor lane not required\",";
  out << "\"oracle_query0_ids_hash_fnv1a64\":"
      << json_escape(hex_u64(fnv1a(fixture.oracle_topk[0].ids.data(),
                                   sizeof(fixture.oracle_topk[0].ids))))
      << ",\"oracle_query0_scores_hash_fnv1a64\":"
      << json_escape(hex_u64(fnv1a(fixture.oracle_topk[0].scores.data(),
                                   sizeof(fixture.oracle_topk[0].scores)))) << "},";
  out << "\"measurement\":{\"warmups\":" << options.warmups
      << ",\"repetitions\":" << options.repeats
      << ",\"repeat_gate_met\":" << (options.repeats >= 5 ? "true" : "false")
      << ",\"build_contract\":{\"compiler\":\"Intel icx 2025.1.1 MSVC target\","
         "\"compile_flags\":\"/fp:precise /clang:-fno-fast-math /clang:-fno-associative-math /clang:-ffp-contract=off\","
         "\"fp_model\":\"precise\",\"fast_math\":false,"
         "\"reassociation_disabled\":true,\"fp_contraction_disabled\":true}"
      << ",\"cold_separation\":{\"npu_compile_model_call_reported_separately\":true,"
         "\"igpu_device_jit_in_first_projection_call\":true,"
         "\"cpu_executable_aot_compile_not_measured\":true}},";
  out << "\"devices\":{\"cpu\":{\"name\":" << json_escape(cpu_brand())
      << ",\"attribution\":\"host scalar C++20\"},\"igpu\":{\"status\":"
      << json_escape(gpu_projection.status);
  if (gpu_projection.status == "ok") {
    out << ",\"name\":" << json_escape(gpu_name) << ",\"driver\":"
        << json_escape(gpu_driver) << ",\"platform\":" << json_escape(gpu_platform)
        << ",\"backend\":\"SYCL Unified Runtime Level Zero\","
           "\"selection\":\"single Intel vendor 0x8086 Level Zero GPU; integration inferred from Arrow Lake device identity\","
           "\"host_unified_memory_reported\":"
        << (gpu_host_unified_memory ? "true" : "false") << ','
        << "\"integration_attribution_is_inference\":true,"
           "\"queue_in_order\":true,\"queue_profiling\":true";
  } else {
    out << ",\"reason\":" << json_escape(gpu_projection.reason);
  }
  out << "},\"npu\":{\"status\":" << json_escape(npu_projection.status);
  if (npu_projection.status == "ok") {
    out << ",\"name\":" << json_escape(npu_device)
        << ",\"openvino_build\":" << json_escape(openvino_version.buildNumber)
        << ",\"openvino_description\":" << json_escape(openvino_version.description)
        << ",\"isolated_fresh_cache_dir\":" << json_escape(npu_cache)
        << ",\"compile_property_scope\":" << json_escape(npu_compile_scope)
        << ",\"stationary_weights_parameter_filled_once\":true,"
           "\"constant_weight_graph_not_used\":true,\"request_owned_l0_tensors\":true";
    if (!npu_first_compile_error.empty()) {
      out << ",\"first_property_attempt_error\":" << json_escape(npu_first_compile_error);
    }
    if (!npu_second_compile_error.empty()) {
      out << ",\"second_property_attempt_error\":" << json_escape(npu_second_compile_error);
    }
  } else {
    out << ",\"reason\":" << json_escape(npu_projection.reason);
  }
  out << "}},";
  out << "\"q_projection_only\":{\"cpu\":";
  append_projection_summary(out, cpu_projection);
  out << ",\"igpu\":";
  append_projection_summary(out, gpu_projection);
  out << ",\"npu\":";
  append_projection_summary(out, npu_projection);
  out << "},\"cpu_direct_residual_oracle\":";
  append_cpu_direct_summary(out, cpu_direct, options.candidates);
  out << ",\"joined_factorized_scan_select\":{\"coarse_distance_contract\":"
      << "\"upstream-provided per-list B uploaded each call; coarse assignment excluded\",";
  out << "\"coarse_added_only_to_bounded_block_partials\":true,"
         "\"persistent_t_and_codes\":true,\"full_scores_materialized\":false,"
         "\"block_emission_contract\":\"top min(k+1, block_rows)\","
         "\"blocks_above_k_emit_k_plus_one_boundary_witness\":true,"
         "\"blocks_at_or_below_k_emit_all_candidates\":true,"
         "\"fixed_oversampling_used_as_proof\":false,"
         "\"block_topk_union_is_exact_for_factor_scores\":true,\"lanes\":[";
  if (gpu_runner) {
    append_joined_summary(out, cpu_gpu_joined, *gpu_runner);
    out << ',';
    append_joined_summary(out, gpu_gpu_joined, *gpu_runner);
    out << ',';
    append_joined_summary(out, npu_gpu_joined, *gpu_runner);
  } else {
    out << "{\"lane\":\"cpu_q_plus_igpu_scan\",\"status\":\"unavailable\",\"reason\":"
        << json_escape(gpu_reason) << "},{\"lane\":\"igpu_q_plus_igpu_scan\","
           "\"status\":\"unavailable\",\"reason\":" << json_escape(gpu_reason)
        << "},{\"lane\":\"npu_q_plus_igpu_scan\",\"status\":\"unavailable\","
           "\"reason\":" << json_escape(npu ? gpu_reason : npu_projection.reason) << '}';
  }
  out << "]},";
  out << "\"safety\":{\"nonfinite_query_preflight_rejected\":"
      << (nonfinite_preflight_rejected ? "true" : "false")
      << ",\"fail_before_publication\":true,\"memory_lifetimes_raii\":true,"
         "\"no_firmware_driver_or_runtime_modification\":true,"
         "\"deliberate_ambiguous_cutoff_fallback\":{\"activated\":"
      << (ambiguous_fallback.activated ? "true" : "false")
      << ",\"fallback_count\":" << ambiguous_fallback.fallback_count
      << ",\"reason\":" << json_escape(ambiguous_fallback.reason)
      << ",\"band_basis\":\"fixture_oracle_measured_not_deployable\","
         "\"score_error_band\":" << ambiguous_fallback.score_error_band
      << ",\"cutoff_gap\":" << ambiguous_fallback.cutoff_gap
      << ",\"required_gap_2E\":" << ambiguous_fallback.required_gap
      << ",\"certified\":false,\"direct_lut_exact\":"
      << (ambiguous_fallback.exact ? "true" : "false")
      << ",\"no_partial_publication\":"
      << (ambiguous_fallback.no_partial_publication ? "true" : "false")
      << ",\"publication_canaries\":"
      << (ambiguous_fallback.publication_canaries ? "true" : "false") << "}},";
  out << "\"memory\":{\"host_fixture_owned_bytes\":" << host_fixture_bytes
      << ",\"igpu_peak_owned_device_bytes\":"
      << (gpu_runner ? gpu_runner->peak_owned_device_bytes() : 0)
      << ",\"igpu_persistent_upload_bytes\":"
      << (gpu_runner ? gpu_runner->persistent_upload_bytes() : 0)
      << ",\"igpu_persistent_upload_ms\":"
      << (gpu_runner ? gpu_runner->persistent_upload_ms() : 0.0)
      << ",\"process_peak_working_set_bytes\":" << process_peak_working_set() << "},";
  out << "\"policy\":{\"npu_share_zero_allowed\":true,"
         "\"k_plus_one_is_exact_cutoff_witness_not_heuristic_oversampling\":true,"
         "\"certificate_rule\":\"F_(k+1)-F_k > 2E\","
         "\"equal_or_touching_intervals_and_score_ties_fall_back\":true,"
         "\"direct_fallback_order\":\"exact direct-LUT f32 score then dense ordinal/ID\","
         "\"promotion_requires_true_end_to_end_ivf_gain_and_unchanged_recall\":true},";
  out << "\"output_path\":" << (output_path.empty() ? "null" : json_escape(output_path.string()))
      << "}\n";

  const std::string payload = out.str();
  write_output_exclusive(output_path, payload);
  std::cout << payload;
  return overall_gate ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parse_options(argc, argv));
  } catch (const std::exception& exception) {
    std::cerr << "{\"schema\":\"ovvs.pq_lut_factor_bench.v1\",\"status\":\"error\","
                 "\"error\":" << json_escape(exception.what()) << "}\n";
    return 2;
  }
}
