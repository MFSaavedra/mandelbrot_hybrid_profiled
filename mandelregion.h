/**
 * @file mandelregion.h
 * @brief The adaptive-subdivision unit of work: a rectangular sub-region of a
 *        frame that either computes its pixels or splits into four children.
 *
 * @ref MandelRegion is where the "task size" load-balancing decision lives (in
 * @ref MandelRegion::examine): sample the region, and if it looks uniform (or
 * hits the pixel-size floor) compute it, otherwise quarter it and push the
 * pieces back onto the @ref WorkQueue. It also holds the CPU pixel-iteration
 * kernel (@ref MandelRegion::diverge) and the static colour map / thresholds
 * shared by all regions.
 */
#ifndef MANDELREGION_H
#define MANDELREGION_H

#include <QImage>
#include <QRgb>
#include <stdlib.h>
#include <iostream>
#include <limits.h>
#include "mandelframe.h"

using namespace std;

const int UNKNOWN     = -1;   ///< Corner iteration count not yet evaluated.
const int UPPER_RIGHT =  0;   ///< Corner index: upper-right (was @c UPPPER_RIGHT, a triple-P typo).
const int UPPER_LEFT  =  1;   ///< Corner index: upper-left (was @c UPPPER_LEFT, a triple-P typo).
const int LOWER_RIGHT =  2;   ///< Corner index: lower-right.
const int LOWER_LEFT  =  3;   ///< Corner index: lower-left.

class WorkQueue;
//************************************************************

/**
 * @brief A rectangular region of the complex plane and its image footprint;
 *        the adaptive-subdivision task pulled from the @ref WorkQueue.
 *
 * Each region knows both its complex-plane rectangle
 * ([@c upperX,@c upperY]&ndash;[@c lowerX,@c lowerY]) and the pixel rectangle
 * it writes into its @ref ownerFrame (@c imageX,@c imageY origin, @c pixelsX
 * &times; @c pixelsY size). @ref examine decides whether to @ref compute the
 * region directly or split it; @ref compute runs the pixels on the CPU or hands
 * them to the GPU front-end. A child inherits its one shared outer corner from
 * the parent so that corner is not re-evaluated.
 */
class MandelRegion
{
private:
  /**
   * @brief Iteration count of the point @p (cx,cy) under @f$z \to z^2 + c@f$.
   *
   * Returns the escape iteration, or the frame's @c MAXITER if the point does
   * not escape. Includes an exact Brent periodicity check: an orbit that
   * exactly revisits a saved state can never escape, so it returns @c MAXITER
   * early without changing the result (byte-identical image).
   * @return Escape iteration in [0, MAXITER].
   */
  int diverge (double cx, double cy);

  double upperX, upperY, lowerX, lowerY;  ///< Complex-plane rectangle (upper-left / lower-right corner).
  int imageX, imageY, pixelsX, pixelsY;   ///< Pixel footprint: origin (@c imageX,@c imageY) and size.
  int cornersIter[4];           ///< Cached corner iteration counts, indexed by the @c UPPER_*/@c LOWER_* constants (@c UNKNOWN until evaluated).
  int depth;                    ///< Recursion depth (root = 0).
  MandelFrame *ownerFrame;      ///< The frame this region writes its pixels into.
  static QRgb *colormap;        ///< Shared iteration-count &rarr; colour lookup table.
  static double diffThresh;     ///< Uniformity threshold: split unless spread < @c diffThresh * maxIter.
  static int pixelSizeThresh;   ///< Pixel-count floor: regions below this are always computed, never split.

public:
  /**
   * @brief Construct a region over a complex rectangle and its pixel footprint.
   * @param uX,uY   Upper-left corner in the complex plane.
   * @param lX,lY   Lower-right corner in the complex plane.
   * @param iX,iY   Pixel origin within the owner frame.
   * @param pX,pY   Region size in pixels.
   * @param f       Owner frame (destination image + geometry + counter).
   * @param depth   Recursion depth (root = 0; children = parent + 1).
   */
  MandelRegion (double, double, double, double, int, int, int, int, MandelFrame *, int depth = 0);
  /**
   * @brief Compute every pixel of this region and write it to the owner frame.
   * @param onGPU @c true dispatches the region to the CUDA front-end (@ref
   *              hostFE); @c false runs the CPU iteration loop. Accumulates
   *              per-thread work metrics and dumps load-balancing outliers.
   */
  void compute (bool onGPU);
  /**
   * @brief Sample the region and either compute it or split it into four.
   *
   * Evaluates the four corners plus a 9-point stencil (edge midpoints +
   * centre). If the iteration spread is small relative to @ref diffThresh, or
   * the region is below @ref pixelSizeThresh, it is computed; otherwise it is
   * quartered and the four children are appended to @p q.
   * @param q     Work queue to push split children onto.
   * @param onGPU Whether the calling worker is the GPU thread (forwarded to @ref compute).
   */
  void examine (WorkQueue &, bool onGPU);
  /** @brief Print this region's pixel origin and size to @c stdout (debug aid). */
  void print ();
  /** @brief Orders regions by pixel count descending (largest is "less"); used by @c operator<. */
  bool operator< (const MandelRegion & a);
  /**
   * @brief Build the shared colour map and set the subdivision thresholds.
   * @param maxV  Largest @c MAXITER across all frames (colour-table size).
   * @param diffT Uniformity threshold (@ref diffThresh).
   * @param pixT  Pixel-size floor (@ref pixelSizeThresh).
   * @note Must be called once before any region is examined.
   */
  static void initColorMapAndThrer (int, double, int);
  /**
   * @brief Test whether @p (x,y) lies in the main cardioid or the period-2 bulb.
   *
   * These are the two largest interior components of the Mandelbrot set; a
   * point inside either is provably in the set. Used as a region metric to
   * classify the slow all-interior outliers (informs whether a cheap interior
   * certificate could skip their iteration entirely).
   * @return @c true if @p (x,y) is in the main cardioid or period-2 bulb.
   */
  static bool inMainCardioidOrBulb (double x, double y);
  /**
   * @brief Print this thread's accumulated CPU timing summary.
   *
   * Call once per thread after the work queue drains. Safe to call from the
   * GPU thread — it prints nothing if that thread never executed a CPU region.
   */
  static void printCPUSummary();

  /**
   * @brief Strict-weak ordering (ascending by pixel count) for the @ref
   *        WorkQueue min-max multiset.
   *
   * With this order, @c begin() is the smallest region and @c --end() the
   * largest. The GPU thread pulls the largest (where it is ~30&times; faster on
   * big coherent interior regions), CPU threads pull the smallest — GPU
   * affinity.
   */
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
