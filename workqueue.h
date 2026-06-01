#ifndef WORKQUEUE_H
#define WORKQUEUE_H

#include <QMutex>
#include <queue>
#include <vector>
#include "mandelregion.h"

using namespace std;

class WorkQueue
{
private:
  QMutex l;
  // Largest-first priority queue (was a FIFO deque): extract() returns the
  // region with the most pixels, so big regions are dispatched earliest and
  // the GPU thread tends to claim them. Ordering is MandelRegion::Compare.
  priority_queue<MandelRegion *, vector<MandelRegion *>, MandelRegion::Compare> queue;

public:
  void append(MandelRegion*);
  MandelRegion* extract();
  int size();
};
#endif
