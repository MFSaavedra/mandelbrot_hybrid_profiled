#include "mandelframe.h"

// Defined in main.cpp.  When 0, regionComplete skips img->save() so we can
// time the pure compute path without the PNG-encode tail.
extern "C" int profileSave;

//----------------------------------------------------------------------
MandelFrame::MandelFrame(double uX, double uY, double lX, double lY, int pX, int pY, char *c, int maxiter) {
    upperX = uX;
    upperY = uY;
    lowerX = lX;
    lowerY = lY;
    img = new QImage(pX, pY, QImage::Format_RGB32);
    memset(fname,0,MAXFNAME);
    strncpy(fname, c, MAXFNAME);
    MAXITER = maxiter;
    pixelsX = pX;
    pixelsY = pY;
    frameIndex = 0;        // overwritten by main.cpp after construction
    stepX = (lowerX - upperX) / pixelsX;
    stepY = (upperY - lowerY) / pixelsY;
    remainingRegions = 1;  // when this becomes 0, the frame has been calculated
}
//----------------------------------------------------------------------
// Thread-safe append of one subdivision node.  Called only on the vizMode
// path, so the lock never touches the timed compute path.
void MandelFrame::addVizRect(const VizRect &r)
{
    QMutexLocker ml(&vizLock);
    vizRects.append(r);
}
//----------------------------------------------------------------------
void MandelFrame::regionSplit()
{
  remainingRegions.fetchAndAddOrdered(3);
}
//----------------------------------------------------------------------
void MandelFrame::regionComplete()
{
  if(remainingRegions.fetchAndAddOrdered(-1) == 1) // if last was 1 now it is 0
  {
     if (profileSave)
       img->save(fname,"PNG");
  }
}
