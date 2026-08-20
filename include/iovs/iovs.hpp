#pragma once

#include "iovs.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace iovs {

inline void check(iovsStatus s) {
  if (s != IOVS_STATUS_SUCCESS) {
    throw std::runtime_error(iovsStatusString(s));
  }
}

class Resources {
 public:
  Resources() { check(iovsResourcesCreate(&res_)); }
  ~Resources() {
    if (res_) iovsResourcesDestroy(res_);
  }
  Resources(const Resources&) = delete;
  Resources& operator=(const Resources&) = delete;
  iovsResources_t get() const { return res_; }
  void set_policy(iovsPolicy p) { check(iovsResourcesSetPolicy(res_, p)); }
  iovsDevice last_device() const {
    iovsDevice d = IOVS_DEVICE_CPU;
    check(iovsResourcesLastDevice(res_, &d));
    return d;
  }

 private:
  iovsResources_t res_ = nullptr;
};

class BruteForceIndex {
 public:
  BruteForceIndex(Resources& res, const float* dataset, int64_t n, int64_t dim, iovsMetric metric)
      : res_(&res) {
    check(iovsBruteForceBuild(res.get(), dataset, n, dim, metric, &ix_));
  }
  ~BruteForceIndex() {
    if (ix_) iovsBruteForceDestroy(ix_);
  }
  BruteForceIndex(const BruteForceIndex&) = delete;
  BruteForceIndex& operator=(const BruteForceIndex&) = delete;

  void search(const float* queries, int64_t nq, int64_t k, int64_t* neighbors, float* distances,
              const uint8_t* bitset = nullptr) {
    check(iovsBruteForceSearch(res_->get(), ix_, queries, nq, k, bitset, neighbors, distances));
  }

 private:
  Resources* res_ = nullptr;
  iovsBruteForceIndex_t ix_ = nullptr;
};

}  // namespace iovs
