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

class IvfFlatIndex {
 public:
  IvfFlatIndex(Resources& res, const float* dataset, int64_t n, int64_t dim, iovsMetric metric, int32_t nlist)
      : res_(&res) {
    check(iovsIvfFlatBuild(res.get(), dataset, n, dim, metric, nlist, &ix_));
  }
  IvfFlatIndex(Resources& res, const char* path) : res_(&res) {
    check(iovsIvfFlatDeserialize(res.get(), path, &ix_));
  }
  ~IvfFlatIndex() {
    if (ix_) iovsIvfFlatDestroy(ix_);
  }
  IvfFlatIndex(const IvfFlatIndex&) = delete;
  IvfFlatIndex& operator=(const IvfFlatIndex&) = delete;
  void search(const float* queries, int64_t nq, int64_t k, int32_t nprobe, int64_t* neighbors,
              float* distances, const uint8_t* bitset = nullptr) {
    check(iovsIvfFlatSearch(res_->get(), ix_, queries, nq, k, nprobe, bitset, neighbors, distances));
  }
  void serialize(const char* path) { check(iovsIvfFlatSerialize(ix_, path)); }
  void extend(const float* extra, int64_t nextra) {
    check(iovsIvfFlatExtend(res_->get(), ix_, extra, nextra));
  }

 private:
  Resources* res_ = nullptr;
  iovsIvfFlatIndex_t ix_ = nullptr;
};

class IvfPqIndex {
 public:
  IvfPqIndex(Resources& res, const float* dataset, int64_t n, int64_t dim, iovsMetric metric, int32_t nlist,
             int32_t pq_m, int32_t pq_nbits)
      : res_(&res) {
    check(iovsIvfPqBuild(res.get(), dataset, n, dim, metric, nlist, pq_m, pq_nbits, &ix_));
  }
  IvfPqIndex(Resources& res, const char* path) : res_(&res) {
    check(iovsIvfPqDeserialize(res.get(), path, &ix_));
  }
  ~IvfPqIndex() {
    if (ix_) iovsIvfPqDestroy(ix_);
  }
  IvfPqIndex(const IvfPqIndex&) = delete;
  IvfPqIndex& operator=(const IvfPqIndex&) = delete;
  void search(const float* queries, int64_t nq, int64_t k, int32_t nprobe, int32_t krefine,
              int64_t* neighbors, float* distances, const uint8_t* bitset = nullptr) {
    check(iovsIvfPqSearch(res_->get(), ix_, queries, nq, k, nprobe, krefine, bitset, neighbors,
                          distances));
  }
  void serialize(const char* path) { check(iovsIvfPqSerialize(ix_, path)); }
  void extend(const float* extra, int64_t nextra) {
    check(iovsIvfPqExtend(res_->get(), ix_, extra, nextra));
  }

 private:
  Resources* res_ = nullptr;
  iovsIvfPqIndex_t ix_ = nullptr;
};

class IvfRabitqIndex {
 public:
  IvfRabitqIndex(Resources& res, const float* dataset, int64_t n, int64_t dim, iovsMetric metric,
                 int32_t nlist)
      : res_(&res) {
    check(iovsIvfRabitqBuild(res.get(), dataset, n, dim, metric, nlist, &ix_));
  }
  IvfRabitqIndex(Resources& res, const char* path) : res_(&res) {
    check(iovsIvfRabitqDeserialize(res.get(), path, &ix_));
  }
  ~IvfRabitqIndex() {
    if (ix_) iovsIvfRabitqDestroy(ix_);
  }
  IvfRabitqIndex(const IvfRabitqIndex&) = delete;
  IvfRabitqIndex& operator=(const IvfRabitqIndex&) = delete;
  void search(const float* queries, int64_t nq, int64_t k, int32_t nprobe, int32_t krefine,
              int64_t* neighbors, float* distances, const uint8_t* bitset = nullptr) {
    check(iovsIvfRabitqSearch(res_->get(), ix_, queries, nq, k, nprobe, krefine, bitset, neighbors,
                              distances));
  }
  void serialize(const char* path) { check(iovsIvfRabitqSerialize(ix_, path)); }
  void extend(const float* extra, int64_t nextra) {
    check(iovsIvfRabitqExtend(res_->get(), ix_, extra, nextra));
  }

 private:
  Resources* res_ = nullptr;
  iovsIvfRabitqIndex_t ix_ = nullptr;
};

class CagraIndex {
 public:
  CagraIndex(Resources& res, const float* dataset, int64_t n, int64_t dim, iovsMetric metric,
             int32_t graph_degree, int32_t intermediate_degree)
      : res_(&res) {
    check(iovsCagraBuild(res.get(), dataset, n, dim, metric, graph_degree, intermediate_degree, &ix_));
  }
  CagraIndex(Resources& res, const char* path) : res_(&res) {
    check(iovsCagraDeserialize(res.get(), path, &ix_));
  }
  ~CagraIndex() {
    if (ix_) iovsCagraDestroy(ix_);
  }
  CagraIndex(const CagraIndex&) = delete;
  CagraIndex& operator=(const CagraIndex&) = delete;
  void search(const float* queries, int64_t nq, int64_t k, int32_t itopk, int32_t search_width,
              int64_t* neighbors, float* distances, const uint8_t* bitset = nullptr) {
    check(iovsCagraSearch(res_->get(), ix_, queries, nq, k, itopk, search_width, bitset, neighbors,
                          distances));
  }
  void serialize(const char* path) { check(iovsCagraSerialize(ix_, path)); }
  void extend(const float* extra, int64_t nextra) {
    check(iovsCagraExtend(res_->get(), ix_, extra, nextra));
  }

 private:
  Resources* res_ = nullptr;
  iovsCagraIndex_t ix_ = nullptr;
};

inline void bitset_from_allow_list(int64_t n, const int64_t* ids, int64_t nids, uint8_t* bitset) {
  check(iovsBitsetFromAllowList(n, ids, nids, bitset));
}

}  // namespace iovs
