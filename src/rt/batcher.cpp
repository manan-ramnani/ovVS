#include "internal.hpp"

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <vector>

using namespace ovvs::impl;

namespace {

struct Waiter {
  int64_t offset = 0;
  int64_t nq = 0;
  int64_t k = 0;
  int64_t* neighbors = nullptr;
  float* distances = nullptr;
  ovvsStatus status = OVVS_STATUS_SUCCESS;
  bool done = false;
};

struct Batcher {
  ovvsResources_t res = nullptr;
  ovvsBruteForceIndex_t index = nullptr;
  int32_t max_batch = 8;
  int32_t max_wait_ms = 0;
  int64_t dim = 0;
  int64_t batch_k = 0;
  int32_t last_batch_nq = 0;
  std::mutex mu;
  std::condition_variable cv;
  std::vector<float> queued;
  int64_t queued_nq = 0;
  std::vector<Waiter*> waiters;
  bool dead = false;

  void flush_locked() {
    if (queued_nq == 0) return;
    last_batch_nq = static_cast<int32_t>(queued_nq);
    std::vector<int64_t> nb(static_cast<size_t>(queued_nq * batch_k), -1);
    std::vector<float> ds(static_cast<size_t>(queued_nq * batch_k), 0.f);
    const ovvsStatus st = ovvsBruteForceSearch(res, index, queued.data(), queued_nq, batch_k, nullptr,
                                               nb.data(), ds.data());
    for (Waiter* w : waiters) {
      for (int64_t i = 0; i < w->nq; ++i) {
        std::memcpy(w->neighbors + i * w->k, nb.data() + static_cast<size_t>((w->offset + i) * batch_k),
                    static_cast<size_t>(w->k) * sizeof(int64_t));
        std::memcpy(w->distances + i * w->k, ds.data() + static_cast<size_t>((w->offset + i) * batch_k),
                    static_cast<size_t>(w->k) * sizeof(float));
      }
      w->status = st;
      w->done = true;
    }
    waiters.clear();
    queued.clear();
    queued_nq = 0;
    batch_k = 0;
    cv.notify_all();
  }
};

}  // namespace

ovvsStatus ovvsBatcherCreate(ovvsResources_t res, ovvsBruteForceIndex_t index, int32_t max_batch,
                             int32_t max_wait_ms, ovvsBatcher_t* out) {
  if (!res || !index || !out || max_batch <= 0) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* b = new Batcher();
  b->res = res;
  b->index = index;
  b->max_batch = max_batch;
  b->max_wait_ms = std::max(0, max_wait_ms);
  b->dim = brute_force_dim(index);
  if (b->dim <= 0) {
    delete b;
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  *out = reinterpret_cast<ovvsBatcher_t>(b);
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsBatcherSearch(ovvsBatcher_t batcher, const float* queries, int64_t nq, int64_t k,
                             int64_t* neighbors, float* distances) {
  if (!batcher || !queries || !neighbors || !distances || nq <= 0 || k <= 0) {
    return OVVS_STATUS_INVALID_ARGUMENT;
  }
  auto* b = reinterpret_cast<Batcher*>(batcher);
  Waiter w;
  w.nq = nq;
  w.k = k;
  w.neighbors = neighbors;
  w.distances = distances;
  {
    std::unique_lock<std::mutex> lock(b->mu);
    if (b->dead) return OVVS_STATUS_ERROR;
    if (b->queued_nq > 0 && b->batch_k != k) b->flush_locked();
    w.offset = b->queued_nq;
    b->queued.insert(b->queued.end(), queries, queries + nq * b->dim);
    b->queued_nq += nq;
    b->batch_k = k;
    b->waiters.push_back(&w);
    if (b->queued_nq >= b->max_batch) {
      b->flush_locked();
    } else {
      const auto pred = [&] { return w.done || b->dead; };
      if (b->max_wait_ms > 0) {
        b->cv.wait_for(lock, std::chrono::milliseconds(b->max_wait_ms), pred);
      }
      if (!w.done) b->flush_locked();
    }
  }
  return w.status;
}

ovvsStatus ovvsBatcherLastBatchSize(ovvsBatcher_t batcher, int32_t* nq) {
  if (!batcher || !nq) return OVVS_STATUS_INVALID_ARGUMENT;
  auto* b = reinterpret_cast<Batcher*>(batcher);
  std::lock_guard<std::mutex> lock(b->mu);
  *nq = b->last_batch_nq;
  return OVVS_STATUS_SUCCESS;
}

ovvsStatus ovvsBatcherDestroy(ovvsBatcher_t batcher) {
  auto* b = reinterpret_cast<Batcher*>(batcher);
  if (!b) return OVVS_STATUS_SUCCESS;
  {
    std::lock_guard<std::mutex> lock(b->mu);
    b->dead = true;
    for (Waiter* w : b->waiters) {
      w->status = OVVS_STATUS_ERROR;
      w->done = true;
    }
    b->waiters.clear();
    b->cv.notify_all();
  }
  delete b;
  return OVVS_STATUS_SUCCESS;
}
