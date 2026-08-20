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
  iovsBruteForceIndex_t get() const { return ix_; }

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
  iovsCagraIndex_t get() const { return ix_; }

 private:
  Resources* res_ = nullptr;
  iovsCagraIndex_t ix_ = nullptr;
};

class VamanaIndex {
 public:
  VamanaIndex(Resources& res, const float* dataset, int64_t n, int64_t dim, iovsMetric metric,
              int32_t graph_degree, float alpha)
      : res_(&res) {
    check(iovsVamanaBuild(res.get(), dataset, n, dim, metric, graph_degree, alpha, &ix_));
  }
  VamanaIndex(Resources& res, const char* path) : res_(&res) {
    check(iovsVamanaDeserialize(res.get(), path, &ix_));
  }
  ~VamanaIndex() {
    if (ix_) iovsVamanaDestroy(ix_);
  }
  VamanaIndex(const VamanaIndex&) = delete;
  VamanaIndex& operator=(const VamanaIndex&) = delete;
  void search(const float* queries, int64_t nq, int64_t k, int32_t beam, int64_t* neighbors,
              float* distances, const uint8_t* bitset = nullptr) {
    check(iovsVamanaSearch(res_->get(), ix_, queries, nq, k, beam, bitset, neighbors, distances));
  }

 private:
  Resources* res_ = nullptr;
  iovsVamanaIndex_t ix_ = nullptr;
};

class ScannIndex {
 public:
  ScannIndex(Resources& res, const float* dataset, int64_t n, int64_t dim, iovsMetric metric, int32_t nlist,
             int32_t pq_m)
      : res_(&res) {
    check(iovsScannBuild(res.get(), dataset, n, dim, metric, nlist, pq_m, &ix_));
  }
  ~ScannIndex() {
    if (ix_) iovsScannDestroy(ix_);
  }
  ScannIndex(const ScannIndex&) = delete;
  ScannIndex& operator=(const ScannIndex&) = delete;
  void search(const float* queries, int64_t nq, int64_t k, int32_t nprobe, int32_t krefine,
              int64_t* neighbors, float* distances) {
    check(iovsScannSearch(res_->get(), ix_, queries, nq, k, nprobe, krefine, neighbors, distances));
  }

 private:
  Resources* res_ = nullptr;
  iovsScannIndex_t ix_ = nullptr;
};

class HnswIndex {
 public:
  HnswIndex(Resources& res, CagraIndex& cagra) : res_(&res) {
    check(iovsHnswFromCagra(res.get(), cagra.get(), &ix_));
  }
  ~HnswIndex() {
    if (ix_) iovsHnswDestroy(ix_);
  }
  HnswIndex(const HnswIndex&) = delete;
  HnswIndex& operator=(const HnswIndex&) = delete;
  void search(const float* queries, int64_t nq, int64_t k, int32_t ef, int64_t* neighbors, float* distances) {
    check(iovsHnswSearch(res_->get(), ix_, queries, nq, k, ef, neighbors, distances));
  }

 private:
  Resources* res_ = nullptr;
  iovsHnswIndex_t ix_ = nullptr;
};

class KMeans {
 public:
  KMeans(Resources& res, const float* dataset, int64_t n, int64_t dim, int32_t nclusters, int32_t iters)
      : res_(&res) {
    check(iovsKMeansFit(res.get(), dataset, n, dim, nclusters, iters, &m_));
  }
  ~KMeans() {
    if (m_) iovsKMeansDestroy(m_);
  }
  KMeans(const KMeans&) = delete;
  KMeans& operator=(const KMeans&) = delete;
  void predict(const float* x, int64_t n, int64_t* labels, float* distances) {
    check(iovsKMeansPredict(res_->get(), m_, x, n, labels, distances));
  }

 private:
  Resources* res_ = nullptr;
  iovsKMeansModel_t m_ = nullptr;
};

class Batcher {
 public:
  Batcher(Resources& res, BruteForceIndex& index, int32_t max_batch, int32_t max_wait_ms) {
    check(iovsBatcherCreate(res.get(), index.get(), max_batch, max_wait_ms, &b_));
  }
  ~Batcher() {
    if (b_) iovsBatcherDestroy(b_);
  }
  Batcher(const Batcher&) = delete;
  Batcher& operator=(const Batcher&) = delete;
  void search(const float* queries, int64_t nq, int64_t k, int64_t* neighbors, float* distances) {
    check(iovsBatcherSearch(b_, queries, nq, k, neighbors, distances));
  }

 private:
  iovsBatcher_t b_ = nullptr;
};

inline void bitset_from_allow_list(int64_t n, const int64_t* ids, int64_t nids, uint8_t* bitset) {
  check(iovsBitsetFromAllowList(n, ids, nids, bitset));
}

}  // namespace iovs
