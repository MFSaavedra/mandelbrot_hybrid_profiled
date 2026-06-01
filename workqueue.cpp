#include "workqueue.h"

using namespace std;

void WorkQueue::append(MandelRegion* i)
{
  QMutexLocker ml(&l);
  queue.push(i);
}


MandelRegion* WorkQueue::extract()
{
  QMutexLocker ml(&l);
  if(queue.empty()) return NULL;

  MandelRegion* temp = queue.top();
  queue.pop();
  return temp;
}


int WorkQueue::size()
{
  return queue.size();
}
