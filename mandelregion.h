#ifndef MANDELREGION_H
#define MANDELREGION_H

#include <QImage>
#include <QRgb>
#include <stdlib.h>
#include <iostream>
#include <limits.h>
#include "mandelframe.h"

using namespace std;

const int UNKNOWN     = -1;
const int UPPER_RIGHT =  0;   // was UPPPER_RIGHT (triple-P typo) — mismatched with mandelregion.cpp
const int UPPER_LEFT  =  1;   // was UPPPER_LEFT  (triple-P typo) — mismatched with mandelregion.cpp
const int LOWER_RIGHT =  2;
const int LOWER_LEFT  =  3;

class WorkQueue;
//************************************************************

class MandelRegion
{
private:
  int diverge (double cx, double cy);

  double upperX, upperY, lowerX, lowerY;
  int imageX, imageY, pixelsX, pixelsY;
  int cornersIter[4];
  int depth;                    // recursion depth (root = 0)
  MandelFrame *ownerFrame;
  static QRgb *colormap;
  static double diffThresh;
  static int pixelSizeThresh;
  // Number of random interior points examine() samples (in addition to the 4
  // corners) to test region uniformity. Replaces the fixed 5-point stencil
  // (edge midpoints + centre). Set from $SAMPLE_N in initColorMapAndThrer
  // (default 5, matching the 9-point's five extra samples). N=0 reduces the
  // rule to the 4-corner test.
  static int sampleN;

public:
    MandelRegion (double, double, double, double, int, int, int, int, MandelFrame *, int depth = 0);
  void compute (bool onGPU);
  void examine (WorkQueue &, bool onGPU);
  void print ();
  bool operator< (const MandelRegion & a);
  static void initColorMapAndThrer (int, double, int);
  // Read-only accessor for the configured random-sample count (for the run's
  // [config] banner in main.cpp; the member itself is private).
  static int getSampleN () { return sampleN; }
  // True if (x,y) lies in the main cardioid or the period-2 bulb — the two
  // largest interior components of the Mandelbrot set.  Used as a region
  // metric to classify the slow all-interior outlier (informs whether an
  // interior certificate could skip its iteration entirely).
  static bool inMainCardioidOrBulb (double x, double y);
  // Prints per-thread CPU timing summary; call once per thread after the work
  // queue drains.  Safe to call from the GPU thread (prints nothing if that
  // thread never executed a CPU region).
  static void printCPUSummary();

  // Orders regions ascending by pixel count for WorkQueue's min-max multiset:
  // begin() is the smallest region, --end() the largest. The GPU thread pulls
  // the largest (where it is ~30x faster on big coherent interior regions),
  // CPU threads pull the smallest -- GPU affinity (report 11 §3.4).
  struct Compare
  {
    bool operator () (const MandelRegion * a, const MandelRegion * b) const
    {
      int Npixels4a = a->pixelsX * a->pixelsY;
      int Npixels4b = b->pixelsX * b->pixelsY;
        return Npixels4a < Npixels4b;
    }

  };
};
#endif
