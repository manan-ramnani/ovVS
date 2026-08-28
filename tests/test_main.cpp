#include "test_harness.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

struct TestRec {
  const char* name;
  void (*fn)();
};

static std::vector<TestRec>& tests() {
  static std::vector<TestRec> t;
  return t;
}

int ovvs_register_test(const char* name, void (*fn)()) {
  tests().push_back({name, fn});
  return 0;
}

static bool contains(const std::vector<std::string>& names, const char* name) {
  return std::find(names.begin(), names.end(), name) != names.end();
}

static bool registered(const std::string& name) {
  return std::any_of(tests().begin(), tests().end(), [&](const TestRec& test) { return name == test.name; });
}

static int usage(const char* program, const char* error = nullptr) {
  if (error) std::fprintf(stderr, "%s\n", error);
  std::fprintf(stderr,
               "usage: %s [--filter EXACT_TEST_NAME]... [--exclude EXACT_TEST_NAME]...\n"
               "       %s --list-tests\n",
               program, program);
  return error ? 2 : 0;
}

int main(int argc, char** argv) {
  std::vector<std::string> filters;
  std::vector<std::string> excludes;
  bool list_tests = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") return usage(argv[0]);
    if (arg == "--list-tests") {
      list_tests = true;
      continue;
    }
    if (arg != "--filter" && arg != "--exclude") {
      const std::string error = "unknown argument: " + arg;
      return usage(argv[0], error.c_str());
    }
    if (++i >= argc) {
      const std::string error = "missing exact test name after " + arg;
      return usage(argv[0], error.c_str());
    }
    (arg == "--filter" ? filters : excludes).emplace_back(argv[i]);
  }

  if (list_tests) {
    if (!filters.empty() || !excludes.empty()) return usage(argv[0], "--list-tests cannot be combined with filters");
    for (const auto& test : tests()) std::printf("%s\n", test.name);
    return 0;
  }

  for (const auto& name : filters) {
    if (!registered(name)) {
      const std::string error = "unknown test in --filter: " + name;
      return usage(argv[0], error.c_str());
    }
    if (contains(excludes, name.c_str())) {
      const std::string error = "test appears in both --filter and --exclude: " + name;
      return usage(argv[0], error.c_str());
    }
  }
  for (const auto& name : excludes) {
    if (!registered(name)) {
      const std::string error = "unknown test in --exclude: " + name;
      return usage(argv[0], error.c_str());
    }
  }

  const auto selected = [&](const TestRec& test) {
    if (!filters.empty() && !contains(filters, test.name)) return false;
    return !contains(excludes, test.name);
  };
  const size_t selected_count = static_cast<size_t>(
      std::count_if(tests().begin(), tests().end(), [&](const TestRec& test) { return selected(test); }));
  if (selected_count == 0) return usage(argv[0], "test selection is empty");

  setvbuf(stdout, nullptr, _IONBF, 0);
  int failed = 0;
  size_t skipped = 0;
  std::printf("running %zu tests\n", selected_count);
  for (const auto& t : tests()) {
    if (!selected(t)) continue;
    try {
      t.fn();
      std::printf("  PASS %s\n", t.name);
    } catch (const OvvsSkipTest& e) {
      std::printf("  SKIP %s: %s\n", t.name, e.what());
      ++skipped;
    } catch (const std::exception& e) {
      std::printf("  FAIL %s: %s\n", t.name, e.what());
      ++failed;
    } catch (...) {
      std::printf("  FAIL %s: unknown\n", t.name);
      ++failed;
    }
  }
  std::printf("%d failed / %zu total (%zu skipped)\n", failed, selected_count, skipped);
  if (!failed && skipped == selected_count) return 77;
  return failed ? 1 : 0;
}
