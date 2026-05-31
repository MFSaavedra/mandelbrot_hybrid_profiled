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

public:
    MandelRegion (double, double, double, double, int, int, int, int, MandelFrame *, int depth = 0);
  void compute (bool onGPU);
  void examine (WorkQueue &, bool onGPU);
  void print ();
  bool operator< (const MandelRegion & a);
  static void initColorMapAndThrer (int, double, int);
  // True if (x,y) lies in the main cardioid or the period-2 bulb — the two
  // largest interior components of the Mandelbrot set.  Used as a region
  // metric to classify the slow all-interior outlier (informs whether an
  // interior certificate could skip its iteration entirely).
  static bool inMainCardioidOrBulb (double x, double y);
  // Prints per-thread CPU timing summary; call once per thread after the work
  // queue drains.  Safe to call from the GPU thread (prints nothing if that
  // thread never executed a CPU region).
  static void printCPUSummary();

  struct Compare
  {
    bool operator () (const MandelRegion * a, const MandelRegion * b)
    {
      int Npixels4a = a->pixelsX * a->pixelsY;
      int Npixels4b = b->pixelsX * b->pixelsY;
        return Npixels4a > Npixels4b;
    }

  };
};
#endif
