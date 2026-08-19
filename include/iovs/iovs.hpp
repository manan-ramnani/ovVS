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

 private:
  iovsResources_t res_ = nullptr;
};

}  // namespace iovs
