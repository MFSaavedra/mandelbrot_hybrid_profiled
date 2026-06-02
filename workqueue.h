#ifndef WORKQUEUE_H
#define WORKQUEUE_H

#include <QMutex>
#include <set>
#include "mandelregion.h"

using namespace std;

class WorkQueue
{
private:
  QMutex l;
  // Min-max work queue ordered by pixel count (MandelRegion::Compare, ascending).
  // extract(isGPU) gives the GPU thread the largest pending region and CPU
  // threads the smallest -- "GPU affinity": the big coherent interior regions
  // (~30x faster on the GPU, report 11 §3.4) run on the GPU, the small/divergent
  // ones on the CPU pool. multiset (not set) because many regions share a size.
  multiset<MandelRegion *, MandelRegion::Compare> queue;
  // When false (no GPU in the run), CPU threads pull the largest region too
  // (largest-first / LPT) instead of the smallest, so big regions are not
  // deferred to the tail. Set once from gpuEnable via setGpuPresent().
  bool gpuPresent = true;

public:
  void append(MandelRegion*);
  MandelRegion* extract(bool isGPU);
  void setGpuPresent(bool g) { gpuPresent = g; }
  int size();
};
#endif
