#pragma once

#include "ovvs/ovvs.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

int ovvs_register_test(const char* name, void (*fn)());

struct OvvsSkipTest : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

[[noreturn]] inline void skip_test(const std::string& reason) {
  throw OvvsSkipTest(reason);
}

#define OVVS_TEST(name)                                          \
  static void test_##name();                                     \
  static int reg_##name = ovvs_register_test(#name, test_##name); \
  static void test_##name()

inline void expect(bool cond, const std::string& msg) {
  if (!cond) throw std::runtime_error(msg);
}

inline void expect_status(ovvsStatus s, const char* what) {
  if (s != OVVS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(what) + ": " + ovvsStatusString(s));
  }
}

inline float ref_dot(const float* a, const float* b, int64_t d) {
  float s = 0.f;
  for (int64_t i = 0; i < d; ++i) s += a[i] * b[i];
  return s;
}

inline void ref_gemm(const float* a, const float* b, float* c, int64_t m, int64_t n, int64_t k,
                     bool trans_b) {
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      float s = 0.f;
      if (!trans_b) {
        for (int64_t t = 0; t < k; ++t) s += a[i * k + t] * b[t * n + j];
      } else {
        for (int64_t t = 0; t < k; ++t) s += a[i * k + t] * b[j * k + t];
      }
      c[i * n + j] = s;
    }
  }
}

inline float recall_at_k(const int64_t* got, const int64_t* truth, int64_t nq, int64_t k) {
  int hit = 0;
  for (int64_t q = 0; q < nq; ++q) {
    for (int64_t i = 0; i < k; ++i) {
      for (int64_t j = 0; j < k; ++j) {
        if (got[q * k + i] == truth[q * k + j]) {
          ++hit;
          break;
        }
      }
    }
  }
  return static_cast<float>(hit) / static_cast<float>(nq * k);
}

inline std::vector<float> make_data(int64_t n, int64_t dim, uint32_t seed) {
  std::vector<float> x(static_cast<size_t>(n * dim));
  uint32_t s = seed;
  for (size_t i = 0; i < x.size(); ++i) {
    s = s * 1664525u + 1013904223u;
    x[i] = (static_cast<int>(s >> 16) % 2000) / 1000.f - 1.f;
  }
  return x;
}

struct Res {
  ovvsResources_t r = nullptr;
  Res() { expect_status(ovvsResourcesCreate(&r), "ResourcesCreate"); }
  ~Res() { ovvsResourcesDestroy(r); }
};
