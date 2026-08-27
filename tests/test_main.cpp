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

#define OVVS_TEST(name)                                          \
  static void test_##name();                                     \
  static int reg_##name = ovvs_register_test(#name, test_##name); \
  static void test_##name()

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  int failed = 0;
  std::printf("running %zu tests\n", tests().size());
  for (const auto& t : tests()) {
    try {
      t.fn();
      std::printf("  PASS %s\n", t.name);
    } catch (const std::exception& e) {
      std::printf("  FAIL %s: %s\n", t.name, e.what());
      ++failed;
    } catch (...) {
      std::printf("  FAIL %s: unknown\n", t.name);
      ++failed;
    }
  }
  std::printf("%d failed / %zu total\n", failed, tests().size());
  return failed ? 1 : 0;
}
