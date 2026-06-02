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
MandelRegion* WorkQueue::extract(bool isGPU)
{
  QMutexLocker ml(&l);
  if(queue.empty()) return NULL;

  multiset<MandelRegion *, MandelRegion::Compare>::iterator it;
  if(isGPU)
    {
      it = queue.end();
      --it;                 // largest
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
