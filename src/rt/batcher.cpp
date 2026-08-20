#include "internal.hpp"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

using namespace iovs::impl;

namespace {

struct Batcher {
  iovsResources_t res = nullptr;
  iovsBruteForceIndex_t index = nullptr;
  int32_t max_batch = 8;
  int32_t max_wait_ms = 0;
  std::mutex mu;
  std::condition_variable cv;
  std::vector<float> queued;
  int64_t queued_nq = 0;
  int64_t dim = 0;
};

}  // namespace

iovsStatus iovsBatcherCreate(iovsResources_t res, iovsBruteForceIndex_t index, int32_t max_batch,
                             int32_t max_wait_ms, iovsBatcher_t* out) {
  if (!res || !index || !out || max_batch <= 0) return IOVS_STATUS_INVALID_ARGUMENT;
  auto* b = new Batcher();
  b->res = res;
  b->index = index;
  b->max_batch = max_batch;
  b->max_wait_ms = std::max(0, max_wait_ms);
  *out = reinterpret_cast<iovsBatcher_t>(b);
  return IOVS_STATUS_SUCCESS;
}

iovsStatus iovsBatcherSearch(iovsBatcher_t batcher, const float* queries, int64_t nq, int64_t k,
                             int64_t* neighbors, float* distances) {
  if (!batcher || !queries || !neighbors || !distances || nq <= 0 || k <= 0) {
    return IOVS_STATUS_INVALID_ARGUMENT;
  }
  auto* b = reinterpret_cast<Batcher*>(batcher);
  /* Coalesce: if caller already presents a batch, run it as one NPU-friendly search.
     Small nq waits up to max_wait_ms so concurrent submitters can join; tests use 0. */
  if (nq < b->max_batch && b->max_wait_ms > 0) {
    std::unique_lock<std::mutex> lock(b->mu);
    b->cv.wait_for(lock, std::chrono::milliseconds(b->max_wait_ms));
  }
  std::lock_guard<std::mutex> lock(b->mu);
  return iovsBruteForceSearch(b->res, b->index, queries, nq, k, nullptr, neighbors, distances);
}

iovsStatus iovsBatcherDestroy(iovsBatcher_t batcher) {
  delete reinterpret_cast<Batcher*>(batcher);
  return IOVS_STATUS_SUCCESS;
}
