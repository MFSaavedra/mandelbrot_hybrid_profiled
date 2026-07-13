/**
 * @file workqueue.h
 * @brief Thread-safe min-max task queue with GPU affinity.
 *
 * The single shared point of contention between all workers: every
 * @ref CalcThr pulls regions from one @ref WorkQueue, and @ref
 * MandelRegion::examine pushes split children back onto it. Ownership of a
 * region — which executor computes it — is decided entirely by @ref
 * WorkQueue::extract, making this the "task owner" load-balancing decision.
 */
#ifndef WORKQUEUE_H
#define WORKQUEUE_H

#include <QMutex>
#include <set>
#include "mandelregion.h"

using namespace std;

/**
 * @brief Mutex-protected region queue ordered by pixel count, with executor
 *        affinity on extraction.
 *
 * Regions are held in a @c std::multiset keyed by area (@ref
 * MandelRegion::Compare, ascending), so the smallest region is at @c begin()
 * and the largest at @c --end(). @ref extract routes by executor: the GPU
 * thread takes the largest region (where it is ~30&times; faster on big
 * coherent interior regions), CPU threads take the smallest. When no GPU is
 * present, CPU threads fall back to largest-first (LPT) scheduling.
 *
 * A @c multiset rather than a @c set because many regions share the same pixel
 * area. All three operations are serialized by the mutex @c l.
 */
class WorkQueue
{
private:
  QMutex l;                                            ///< Guards @c queue against concurrent worker access.
  /// Min-max region queue ordered ascending by pixel count (@ref MandelRegion::Compare):
  /// @c begin() is the smallest region, @c --end() the largest.
  multiset<MandelRegion *, MandelRegion::Compare> queue;
  /// When @c false (no GPU in the run), CPU threads pull the largest region too
  /// (largest-first / LPT) instead of the smallest. Set once via @ref setGpuPresent.
  bool gpuPresent = true;

public:
  /** @brief Insert a region into the queue (thread-safe). */
  void append(MandelRegion*);
  /**
   * @brief Remove and return one region, chosen by executor affinity.
   * @param isGPU @c true for the GPU worker (takes the largest region);
   *              @c false for a CPU worker (takes the smallest, unless no GPU
   *              is present, in which case it takes the largest — LPT).
   * @return The extracted region, or @c NULL when the queue is empty (the
   *         signal for @ref CalcThr::run to exit).
   */
  MandelRegion* extract(bool isGPU);
  /** @brief Record whether a GPU worker participates, selecting the CPU-only LPT fallback. */
  void setGpuPresent(bool g) { gpuPresent = g; }
  /** @brief Current number of pending regions (unsynchronized snapshot). */
  int size();
};
#endif
