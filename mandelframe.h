#ifndef MANDELFRAME_H
#define MANDELFRAME_H

#include <QImage>
#include <QAtomicInt>
#include <QVector>
#include <QMutex>

const int MAXFNAME=50;

// One subdivision node recorded during a vizMode run.  Every examined region
// (both internal split-nodes and computed leaves) registers its pixel
// rectangle, its recursion depth, whether it became a leaf, and — for leaves —
// whether it was computed on the GPU thread.  main.cpp replays these to emit
// the depth-by-depth subdivision animation.
struct VizRect
{
  int  x, y, w, h;   // pixel rectangle within the frame
  int  depth;        // recursion depth (root = 0)
  bool leaf;         // true if computed, false if split into four
  bool onGPU;        // executor (meaningful only when leaf)
};

// Used to represent an image frame
class MandelFrame
{
public:
    int MAXITER;
    int frameIndex;             // position in the interpolated sequence
    double upperX, upperY, lowerX, lowerY;
    double stepX, stepY;
    int pixelsX, pixelsY;
    QImage *img;
    char fname[MAXFNAME+1];
    QAtomicInt remainingRegions;

    // Populated only when vizMode is set; guarded by vizLock because every
    // worker thread appends concurrently from MandelRegion::examine().
    QVector<VizRect> vizRects;
    QMutex vizLock;

    MandelFrame(double, double, double, double, int, int, char *, int );
    void regionSplit();
    void regionComplete();
    void addVizRect(const VizRect &r);
};

#endif