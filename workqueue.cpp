/**
 * @file workqueue.cpp
 * @brief Implementation of the thread-safe min-max task queue (@ref WorkQueue).
 * @see workqueue.h
 */
#include "workqueue.h"

using namespace std;

void WorkQueue::append(MandelRegion* i)
{
  QMutexLocker ml(&l);
  queue.insert(i);
}


// GPU affinity: the GPU thread takes the largest pending region (--end()),
// CPU threads take the smallest (begin()). When only one size class remains
// both ends coincide, so neither executor idles while work exists.
//
// CPU-only guard: if there is no GPU in this run (gpuPresent == false), CPU
// threads also take the largest -- largest-first (LPT) -- so big regions start
// early instead of being deferred to the tail by smallest-first.
MandelRegion* WorkQueue::extract(bool isGPU)
{
  QMutexLocker ml(&l);
  if(queue.empty()) return NULL;

  multiset<MandelRegion *, MandelRegion::Compare>::iterator it;
  if(isGPU || !gpuPresent)
    {
      it = queue.end();
      --it;                 // largest (GPU affinity, or LPT when no GPU)
    }
  else
    it = queue.begin();     // smallest

  MandelRegion* temp = *it;
  queue.erase(it);
  return temp;
}


int WorkQueue::size()
{
  return queue.size();
}
