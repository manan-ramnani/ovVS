#pragma once

#include "ovvs.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ovvs {

inline void check(ovvsStatus s) {
  if (s != OVVS_STATUS_SUCCESS) {
    throw std::runtime_error(ovvsStatusString(s));
  }
}

class Resources {
 public:
  Resources() { check(ovvsResourcesCreate(&res_)); }
  ~Resources() {
    if (res_) ovvsResourcesDestroy(res_);
  }
  Resources(const Resources&) = delete;
  Resources& operator=(const Resources&) = delete;
  ovvsResources_t get() const { return res_; }
  void set_policy(ovvsPolicy p) { check(ovvsResourcesSetPolicy(res_, p)); }
  ovvsDevice last_device() const {
    ovvsDevice d = OVVS_DEVICE_CPU;
    check(ovvsResourcesLastDevice(res_, &d));
    return d;
  }
  ovvsDType last_compute_dtype() const {
    ovvsDType t = OVVS_DTYPE_F32;
    check(ovvsResourcesLastComputeDtype(res_, &t));
    return t;
  }
  ovvsCagraBuildStatsV1 cagra_build_stats() const {
    ovvsCagraBuildStatsV1 stats{};
    check(ovvsResourcesCagraBuildStatsV1(res_, &stats));
    return stats;
  }

 private:
  ovvsResources_t res_ = nullptr;
};

class BruteForceIndex {
 public:
  BruteForceIndex(Resources& res, const float* dataset, int64_t n, int64_t dim, ovvsMetric metric)
      : res_(&res) {
    check(ovvsBruteForceBuild(res.get(), dataset, n, dim, metric, &ix_));
  }
  ~BruteForceIndex() {
    if (ix_) ovvsBruteForceDestroy(ix_);
  }
  BruteForceIndex(const BruteForceIndex&) = delete;
  BruteForceIndex& operator=(const BruteForceIndex&) = delete;

  void search(const float* queries, int64_t nq, int64_t k, int64_t* neighbors, float* distances,
              const uint8_t* bitset = nullptr) {
    check(ovvsBruteForceSearch(res_->get(), ix_, queries, nq, k, bitset, neighbors, distances));
  }
  ovvsBruteForceIndex_t get() const { return ix_; }

 private:
  Resources* res_ = nullptr;
  ovvsBruteForceIndex_t ix_ = nullptr;
};

class IvfFlatIndex {
 public:
  IvfFlatIndex(Resources& res, const float* dataset, int64_t n, int64_t dim, ovvsMetric metric, int32_t nlist)
      : res_(&res) {
    check(ovvsIvfFlatBuild(res.get(), dataset, n, dim, metric, nlist, &ix_));
  }
  IvfFlatIndex(Resources& res, const char* path) : res_(&res) {
    check(ovvsIvfFlatDeserialize(res.get(), path, &ix_));
  }
  ~IvfFlatIndex() {
    if (ix_) ovvsIvfFlatDestroy(ix_);
  }
  IvfFlatIndex(const IvfFlatIndex&) = delete;
  IvfFlatIndex& operator=(const IvfFlatIndex&) = delete;
  void search(const float* queries, int64_t nq, int64_t k, int32_t nprobe, int64_t* neighbors,
              float* distances, const uint8_t* bitset = nullptr) {
    check(ovvsIvfFlatSearch(res_->get(), ix_, queries, nq, k, nprobe, bitset, neighbors, distances));
  }
  void serialize(const char* path) { check(ovvsIvfFlatSerialize(ix_, path)); }
  void extend(const float* extra, int64_t nextra) {
    check(ovvsIvfFlatExtend(res_->get(), ix_, extra, nextra));
  }

 private:
  Resources* res_ = nullptr;
  ovvsIvfFlatIndex_t ix_ = nullptr;
};

class IvfPqIndex {
 public:
  IvfPqIndex(Resources& res, const float* dataset, int64_t n, int64_t dim, ovvsMetric metric, int32_t nlist,
             int32_t pq_m, int32_t pq_nbits)
      : res_(&res) {
    check(ovvsIvfPqBuild(res.get(), dataset, n, dim, metric, nlist, pq_m, pq_nbits, &ix_));
  }
  IvfPqIndex(Resources& res, const char* path) : res_(&res) {
    check(ovvsIvfPqDeserialize(res.get(), path, &ix_));
  }
  ~IvfPqIndex() {
    if (ix_) ovvsIvfPqDestroy(ix_);
  }
  IvfPqIndex(const IvfPqIndex&) = delete;
  IvfPqIndex& operator=(const IvfPqIndex&) = delete;
  void search(const float* queries, int64_t nq, int64_t k, int32_t nprobe, int32_t krefine,
              int64_t* neighbors, float* distances, const uint8_t* bitset = nullptr) {
    check(ovvsIvfPqSearch(res_->get(), ix_, queries, nq, k, nprobe, krefine, bitset, neighbors,
                          distances));
  }
  void serialize(const char* path) { check(ovvsIvfPqSerialize(ix_, path)); }
  void extend(const float* extra, int64_t nextra) {
    check(ovvsIvfPqExtend(res_->get(), ix_, extra, nextra));
  }

 private:
  Resources* res_ = nullptr;
  ovvsIvfPqIndex_t ix_ = nullptr;
};

class IvfRabitqIndex {
 public:
  IvfRabitqIndex(Resources& res, const float* dataset, int64_t n, int64_t dim, ovvsMetric metric,
                 int32_t nlist)
      : res_(&res) {
    check(ovvsIvfRabitqBuild(res.get(), dataset, n, dim, metric, nlist, &ix_));
  }
  IvfRabitqIndex(Resources& res, const char* path) : res_(&res) {
    check(ovvsIvfRabitqDeserialize(res.get(), path, &ix_));
  }
  ~IvfRabitqIndex() {
    if (ix_) ovvsIvfRabitqDestroy(ix_);
  }
  IvfRabitqIndex(const IvfRabitqIndex&) = delete;
  IvfRabitqIndex& operator=(const IvfRabitqIndex&) = delete;
  void search(const float* queries, int64_t nq, int64_t k, int32_t nprobe, int32_t krefine,
              int64_t* neighbors, float* distances, const uint8_t* bitset = nullptr) {
    check(ovvsIvfRabitqSearch(res_->get(), ix_, queries, nq, k, nprobe, krefine, bitset, neighbors,
                              distances));
  }
  void serialize(const char* path) { check(ovvsIvfRabitqSerialize(ix_, path)); }
  void extend(const float* extra, int64_t nextra) {
    check(ovvsIvfRabitqExtend(res_->get(), ix_, extra, nextra));
  }

 private:
  Resources* res_ = nullptr;
  ovvsIvfRabitqIndex_t ix_ = nullptr;
};

class CagraIndex {
 public:
  CagraIndex(Resources& res, const float* dataset, int64_t n, int64_t dim, ovvsMetric metric,
             int32_t graph_degree, int32_t intermediate_degree)
      : res_(&res) {
    check(ovvsCagraBuild(res.get(), dataset, n, dim, metric, graph_degree, intermediate_degree, &ix_));
  }
  CagraIndex(Resources& res, const float* dataset, int64_t n, int64_t dim, ovvsMetric metric,
             int32_t graph_degree, int32_t intermediate_degree, ovvsCagraBuildAlgo algo)
      : res_(&res) {
    check(ovvsCagraBuildEx(res.get(), dataset, n, dim, metric, graph_degree,
                           intermediate_degree, algo, &ix_));
  }
  CagraIndex(Resources& res, const char* path) : res_(&res) {
    check(ovvsCagraDeserialize(res.get(), path, &ix_));
  }
  ~CagraIndex() {
    if (ix_) ovvsCagraDestroy(ix_);
  }
  CagraIndex(const CagraIndex&) = delete;
  CagraIndex& operator=(const CagraIndex&) = delete;
  void search(const float* queries, int64_t nq, int64_t k, int32_t itopk, int32_t search_width,
              int64_t* neighbors, float* distances, const uint8_t* bitset = nullptr) {
    check(ovvsCagraSearch(res_->get(), ix_, queries, nq, k, itopk, search_width, bitset, neighbors,
                          distances));
  }
  void serialize(const char* path) { check(ovvsCagraSerialize(ix_, path)); }
  void extend(const float* extra, int64_t nextra) {
    check(ovvsCagraExtend(res_->get(), ix_, extra, nextra));
  }
  ovvsCagraIndex_t get() const { return ix_; }

 private:
  Resources* res_ = nullptr;
  ovvsCagraIndex_t ix_ = nullptr;
};

class VamanaIndex {
 public:
  VamanaIndex(Resources& res, const float* dataset, int64_t n, int64_t dim, ovvsMetric metric,
              int32_t graph_degree, float alpha)
      : res_(&res) {
    check(ovvsVamanaBuild(res.get(), dataset, n, dim, metric, graph_degree, alpha, &ix_));
  }
  VamanaIndex(Resources& res, const char* path) : res_(&res) {
    check(ovvsVamanaDeserialize(res.get(), path, &ix_));
  }
  ~VamanaIndex() {
    if (ix_) ovvsVamanaDestroy(ix_);
  }
  VamanaIndex(const VamanaIndex&) = delete;
  VamanaIndex& operator=(const VamanaIndex&) = delete;
  void search(const float* queries, int64_t nq, int64_t k, int32_t beam, int64_t* neighbors,
              float* distances, const uint8_t* bitset = nullptr) {
    check(ovvsVamanaSearch(res_->get(), ix_, queries, nq, k, beam, bitset, neighbors, distances));
  }

 private:
  Resources* res_ = nullptr;
  ovvsVamanaIndex_t ix_ = nullptr;
};

class ScannIndex {
 public:
  ScannIndex(Resources& res, const float* dataset, int64_t n, int64_t dim, ovvsMetric metric, int32_t nlist,
             int32_t pq_m)
      : res_(&res) {
    check(ovvsScannBuild(res.get(), dataset, n, dim, metric, nlist, pq_m, &ix_));
  }
  ~ScannIndex() {
    if (ix_) ovvsScannDestroy(ix_);
  }
  ScannIndex(const ScannIndex&) = delete;
  ScannIndex& operator=(const ScannIndex&) = delete;
  void search(const float* queries, int64_t nq, int64_t k, int32_t nprobe, int32_t krefine,
              int64_t* neighbors, float* distances) {
    check(ovvsScannSearch(res_->get(), ix_, queries, nq, k, nprobe, krefine, neighbors, distances));
  }

 private:
  Resources* res_ = nullptr;
  ovvsScannIndex_t ix_ = nullptr;
};

class HnswIndex {
 public:
  HnswIndex(Resources& res, CagraIndex& cagra) : res_(&res) {
    check(ovvsHnswFromCagra(res.get(), cagra.get(), &ix_));
  }
  ~HnswIndex() {
    if (ix_) ovvsHnswDestroy(ix_);
  }
  HnswIndex(const HnswIndex&) = delete;
  HnswIndex& operator=(const HnswIndex&) = delete;
  void search(const float* queries, int64_t nq, int64_t k, int32_t ef, int64_t* neighbors, float* distances) {
    check(ovvsHnswSearch(res_->get(), ix_, queries, nq, k, ef, neighbors, distances));
  }
  void serialize(const char* path) { check(ovvsHnswSerialize(ix_, path)); }

 private:
  Resources* res_ = nullptr;
  ovvsHnswIndex_t ix_ = nullptr;
};

class KMeans {
 public:
  KMeans(Resources& res, const float* dataset, int64_t n, int64_t dim, int32_t nclusters, int32_t iters)
      : res_(&res) {
    check(ovvsKMeansFit(res.get(), dataset, n, dim, nclusters, iters, &m_));
  }
  ~KMeans() {
    if (m_) ovvsKMeansDestroy(m_);
  }
  KMeans(const KMeans&) = delete;
  KMeans& operator=(const KMeans&) = delete;
  void predict(const float* x, int64_t n, int64_t* labels, float* distances) {
    check(ovvsKMeansPredict(res_->get(), m_, x, n, labels, distances));
  }

 private:
  Resources* res_ = nullptr;
  ovvsKMeansModel_t m_ = nullptr;
};

class Batcher {
 public:
  Batcher(Resources& res, BruteForceIndex& index, int32_t max_batch, int32_t max_wait_ms) {
    check(ovvsBatcherCreate(res.get(), index.get(), max_batch, max_wait_ms, &b_));
  }
  ~Batcher() {
    if (b_) ovvsBatcherDestroy(b_);
  }
  Batcher(const Batcher&) = delete;
  Batcher& operator=(const Batcher&) = delete;
  void search(const float* queries, int64_t nq, int64_t k, int64_t* neighbors, float* distances) {
    check(ovvsBatcherSearch(b_, queries, nq, k, neighbors, distances));
  }

 private:
  ovvsBatcher_t b_ = nullptr;
};

inline void bitset_from_allow_list(int64_t n, const int64_t* ids, int64_t nids, uint8_t* bitset) {
  check(ovvsBitsetFromAllowList(n, ids, nids, bitset));
}

}  // namespace ovvs
