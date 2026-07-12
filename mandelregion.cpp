#include "mandelregion.h"
#include "workqueue.h"
#include "kernel.h"
#include <QColor>
#include <QAtomicInt>

// nvToolsExt — NVTX CPU-side annotation API (same header used in kernel.cu).
// Including it in a plain C++ translation unit is fine; it links against
// libnvToolsExt without requiring nvcc.
// nvToolsExt.h lives under nvtx3/ in CUDA 11+ installations.
// CPU-only builds (make GPU=0, for nodes with no usable CUDA stack) have no
// CUDA install to provide the header; -DNO_NVTX turns the annotations into
// no-ops with the real API's signatures.
#ifdef NO_NVTX
static inline int nvtxRangePush(const char *) { return 0; }
static inline int nvtxRangePop()              { return 0; }
#else
#include <nvtx3/nvToolsExt.h>
#endif

// ============================================================
// CPU-side timing via clock_gettime
//
// We track how many regions each thread computes on the CPU and the
// cumulative time spent in the iteration loop, then print a per-thread
// summary when the thread exits (called from examine() after the last region).
// thread_local gives each QThread its own copy with zero synchronization cost.
// ============================================================
#include <time.h>
static thread_local long   cpuRegionCount = 0;
static thread_local double cpuTotalMs     = 0.0;

// Defined in main.cpp.  When set, the per-region CPU timing line below is
// suppressed; the printCPUSummary call at thread exit is unaffected.
extern "C" int profileQuiet;

// Defined in main.cpp.  When set, every examined region registers its pixel
// rectangle + depth + executor with its owner frame for the subdivision
// animation.  Gated here so the recording cost never touches a timed run.
extern "C" int vizMode;

// A computed region whose wall time exceeds this is logged in full (location,
// corner spread, interior fraction, cardioid/bulb membership) regardless of
// the quiet flag.  These are the load-balancing outliers worth dissecting.
static const double OUTLIER_MS = 5000.0;

// Returns elapsed milliseconds between two timespec values.
static inline double elapsedMs(const struct timespec &t0, const struct timespec &t1)
{
  return (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6;
}

//--------------------------------------
QRgb *MandelRegion::colormap;
double MandelRegion::diffThresh;
int MandelRegion::pixelSizeThresh;
//--------------------------------------
void MandelRegion::initColorMapAndThrer (int maxV, double diffT=0.3, int pixT=2048)
{
  colormap = new QRgb[maxV];
  diffThresh = diffT;
  pixelSizeThresh = pixT;
  
  for (int i = 0; i < maxV; i++)
    {
      double ratio = i * 1.0l / maxV;
      ratio = (i % 256) / 255.0;
      int R, G, B;
      R = 255 * (1 - ratio);
      B = 255 * ratio;
      if (ratio >= 0.5)
        G = 255 - 255 * (ratio - 0.5);
      else
        G = 255 * ratio * 2;
      colormap[i] = qRgb (R, G, B);
    }
}

//--------------------------------------

void MandelRegion::print ()
{
  cout << "Coord(" << imageX << ", " << imageY << ")   Size : " << pixelsX << " x " << pixelsY << endl;
}

//--------------------------------------
int MandelRegion::diverge (double cx, double cy)
{
  int MAXITER = ownerFrame->MAXITER;
  int iter = 0;
  double vx = cx, vy = cy, tx, ty;
  // Brent-style periodicity check.  Interior orbits settle into an attracting
  // cycle; once the FP state exactly revisits a saved state the (deterministic)
  // iteration can never escape, so returning MAXITER now is *exact* -- the
  // plain loop would have ground to MAXITER and returned the same value.
  // Exact equality (no epsilon) is what guarantees a byte-identical image:
  // an escaping orbit can never exactly repeat, so exterior pixels are
  // untouched.  The saved point is refreshed at doubling intervals so a cycle
  // of period p entered after a transient of t iterations is caught within
  // O(max(p, t)) instead of running to MAXITER.
  double sx = vx, sy = vy;
  int nextSave = 8;
  while (iter < MAXITER && (vx * vx + vy * vy) < 4)
    {
      tx = vx * vx - vy * vy + cx;
      ty = 2 * vx * vy + cy;
      vx = tx;
      vy = ty;
      iter++;
      if (vx == sx && vy == sy)
        return MAXITER;
      if (iter == nextSave)
        {
          sx = vx;
          sy = vy;
          nextSave *= 2;
        }
    }
  return iter;
}

//--------------------------------------

MandelRegion::MandelRegion (double uX, double uY, double lX, double lY, int iX, int iY, int pX, int pY, MandelFrame * f, int d)
{
  upperX = uX;
  upperY = uY;
  lowerX = lX;
  lowerY = lY;
  imageX = iX;
  imageY = iY;
  pixelsX = pX;
  pixelsY = pY;
  depth = d;
  cornersIter[0] = cornersIter[1] = cornersIter[2] = cornersIter[3] = UNKNOWN;
  ownerFrame = f;
}

//--------------------------------------
// Interior-membership test for the two largest components of the Mandelbrot
// set.  A point inside either is provably in the set (iteration count would
// reach MAXITER), so a region whose corners all pass this test is a candidate
// for a constant-fill certificate.
bool MandelRegion::inMainCardioidOrBulb (double x, double y)
{
  // Main cardioid:  q = (x - 1/4)^2 + y^2 ;  q (q + (x - 1/4)) <= y^2 / 4
  double xm = x - 0.25;
  double q  = xm * xm + y * y;
  if (q * (q + xm) <= 0.25 * y * y)
    return true;
  // Period-2 bulb:  (x + 1)^2 + y^2 <= 1/16
  double xp = x + 1.0;
  if (xp * xp + y * y <= 0.0625)
    return true;
  return false;
}

//--------------------------------------

void MandelRegion::compute (bool onGPU)
{
  double stepX = ownerFrame->stepX;
  double stepY = ownerFrame->stepY;
  QImage *img = ownerFrame->img;
  int MAXGRAY = ownerFrame->MAXITER;

  if (onGPU)
    {
      // hostFE() has its own NVTX range and CUDA event timing; nothing extra needed here.
      unsigned int *h_res;
      int pitch;

      hostFE (upperX, upperY, lowerX, lowerY, pixelsX, pixelsY, &h_res, &pitch, MAXGRAY);
      pitch /= sizeof (int);

      // Copy GPU results into the Qt image buffer, accumulating work metrics
      // (total iterations and interior-pixel count) in the same pass.
      long iterSum    = 0;
      long insetCount = 0;
      for (int i = 0; i < pixelsX; i++)
        for (int j = 0; j < pixelsY; j++)
          {
            int color = h_res[j * pitch + i];
            iterSum += color;
            if (color == MAXGRAY)
              {
                insetCount++;
                img->setPixel (imageX + i, imageY + j, qRgb (0, 0, 0));
              }
            else
              img->setPixel (imageX + i, imageY + j, colormap[color]);
          }

      if (!profileQuiet)
        {
          long npix = (long) pixelsX * pixelsY;
          fprintf(stderr,
                  "[GPU region metrics] frame %3d depth %d size %4dx%4d  "
                  "meanIter %.1f  inset %.1f%%\n",
                  ownerFrame->frameIndex, depth, pixelsX, pixelsY,
                  (double) iterSum / npix, 100.0 * insetCount / npix);
        }
    }
  else                          // CPU execution
    {
      // NVTX range for the CPU compute path.
      // Nsight Systems will show this as a colored bar on the calling
      // thread's lane, letting you see CPU workers competing with the GPU
      // thread for work-queue items at a glance.
      nvtxRangePush("CPU region compute");

      struct timespec t0, t1;
      clock_gettime(CLOCK_MONOTONIC, &t0);

      long iterSum    = 0;
      long insetCount = 0;
      for (int i = 0; i < pixelsX; i++)
        for (int j = 0; j < pixelsY; j++)
          {
            double tempx, tempy;
            tempx = upperX + i * stepX;
            tempy = upperY - j * stepY;
            int color = diverge (tempx, tempy);
            iterSum += color;
            if (color == MAXGRAY)
              {
                insetCount++;
                img->setPixel (imageX + i, imageY + j, qRgb (0, 0, 0));
              }
            else
              img->setPixel (imageX + i, imageY + j, colormap[color]);
          }

      clock_gettime(CLOCK_MONOTONIC, &t1);
      double ms = elapsedMs(t0, t1);
      cpuTotalMs += ms;
      cpuRegionCount++;

      long   npix      = (long) pixelsX * pixelsY;
      double meanIter  = (double) iterSum / npix;
      double insetFrac = 100.0 * insetCount / npix;

      if (!profileQuiet)
        fprintf(stderr,
                "[CPU region %4ld] frame %3d depth %d size %4dx%4d  "
                "compute %.3f ms  meanIter %.1f  inset %.1f%%\n",
                cpuRegionCount, ownerFrame->frameIndex, depth,
                pixelsX, pixelsY, ms, meanIter, insetFrac);

      // Full dump for load-balancing outliers, independent of the quiet flag.
      if (ms > OUTLIER_MS)
        {
          int lo = cornersIter[0], hi = cornersIter[0];
          for (int k = 1; k < 4; k++)
            {
              if (cornersIter[k] < lo) lo = cornersIter[k];
              if (cornersIter[k] > hi) hi = cornersIter[k];
            }
          bool cornersInterior =
              inMainCardioidOrBulb (upperX, upperY) &&
              inMainCardioidOrBulb (lowerX, upperY) &&
              inMainCardioidOrBulb (upperX, lowerY) &&
              inMainCardioidOrBulb (lowerX, lowerY);
          fprintf(stderr,
                  "[OUTLIER] frame=%d depth=%d img=(%d,%d) px=%dx%d "
                  "c=[%.6f,%.6f]..[%.6f,%.6f] corners=%d/%d/%d/%d "
                  "spread=%d meanIter=%.1f inset=%.1f%% cardioidBulb=%d ms=%.1f\n",
                  ownerFrame->frameIndex, depth, imageX, imageY, pixelsX, pixelsY,
                  upperX, upperY, lowerX, lowerY,
                  cornersIter[0], cornersIter[1], cornersIter[2], cornersIter[3],
                  hi - lo, meanIter, insetFrac, (int) cornersInterior, ms);
        }

      nvtxRangePop(); // closes "CPU region compute"
    }
}

//--------------------------------------
// if the region is small enough, process it, or split it in 4 regions
void MandelRegion::examine (WorkQueue & q, bool onGPU = false)
{
  // NVTX range that wraps the full decision cycle for one region: corner
  // evaluation + either compute() or a four-way split back onto the queue.
  // In Nsight Systems this appears on the calling thread's row, so you can
  // visually count how many examine() calls each thread handles and compare
  // the depth/duration distribution between CPU and GPU workers.
  nvtxRangePush("examine region");

  int minIter = INT_MAX, maxIter = 0;

  // evaluate the corners first
  for (int i = 0; i < 4; i++)
    {
      if (cornersIter[i] == UNKNOWN)
        {
          switch (i)
            {
            case (UPPER_RIGHT):
              cornersIter[i] = diverge (lowerX, upperY);
              break;
            case (UPPER_LEFT):
              cornersIter[i] = diverge (upperX, upperY);
              break;
            case (LOWER_RIGHT):
              cornersIter[i] = diverge (lowerX, lowerY);
              break;
            default:           // LOWER_LEFT
              cornersIter[i] = diverge (upperX, lowerY);
            }
        }
      // Track min and max independently. The original code chained these
      // with `else if`, which made max-tracking unreachable whenever the
      // current sample also lowered the min — most notably on i==0, where
      // minIter starts at INT_MAX so the min branch always fires and
      // maxIter is never updated from the first corner.
      if (cornersIter[i] < minIter)
        minIter = cornersIter[i];
      if (cornersIter[i] > maxIter)
        maxIter = cornersIter[i];
    }

  // 9-point stencil: extend the four corners with the four edge midpoints and
  // the centre.  Sampling the interior catches high-iteration filaments that
  // pass between the corners -- the failure mode behind the all-interior
  // outliers diagnosed in report 08, whose four corners all sit at maxIter
  // (spread 0) so the 4-corner rule never splits them.  A child inherits only
  // its shared outer corner, never a midpoint, so these five are always fresh:
  // each examine() adds five diverge() calls over the 4-corner version.
  double midX = (upperX + lowerX) * 0.5;
  double midY = (upperY + lowerY) * 0.5;
  int extra[5] = {
    diverge (midX,   upperY),   // top edge midpoint
    diverge (midX,   lowerY),   // bottom edge midpoint
    diverge (upperX, midY),     // left edge midpoint
    diverge (lowerX, midY),     // right edge midpoint
    diverge (midX,   midY)      // centre
  };
  for (int i = 0; i < 5; i++)
    {
      if (extra[i] < minIter) minIter = extra[i];
      if (extra[i] > maxIter) maxIter = extra[i];
    }


  // either compute the pixels or break the region in 4 pieces
  if (maxIter - minIter < diffThresh * maxIter || pixelsX * pixelsY < pixelSizeThresh)
    {
      compute (onGPU);
      // Record this leaf for the subdivision animation, coloured by executor.
      if (vizMode)
        ownerFrame->addVizRect ({imageX, imageY, pixelsX, pixelsY,
                                 depth, true, onGPU});
      ownerFrame->regionComplete ();
    }
  else
    {
      // Record the internal (split) node — drawn as the subdivision skeleton.
      if (vizMode)
        ownerFrame->addVizRect ({imageX, imageY, pixelsX, pixelsY,
                                 depth, false, onGPU});

      double midDiagX1, midDiagY1;      // data for determining the new subregions
      double midDiagX2, midDiagY2;
      int subimageX, subimageY;
      subimageX = pixelsX / 2;  // concern the upper left quad.
      subimageY = pixelsY / 2;
      midDiagX1 = upperX + (subimageX - 1) * ownerFrame->stepX;
      midDiagY1 = upperY - (subimageY - 1) * ownerFrame->stepY;
      midDiagX2 = midDiagX1 + ownerFrame->stepX;
      midDiagY2 = midDiagY1 - ownerFrame->stepY;

      MandelRegion *sub[4];
      sub[UPPER_LEFT] = new MandelRegion (upperX, upperY, midDiagX1, midDiagY1, imageX, imageY, subimageX, subimageY, ownerFrame, depth + 1);
      sub[UPPER_LEFT]->cornersIter[UPPER_LEFT] = cornersIter[UPPER_LEFT];

      sub[UPPER_RIGHT] = new MandelRegion (midDiagX2, upperY, lowerX, midDiagY1, imageX + subimageX, imageY, pixelsX - subimageX, subimageY, ownerFrame, depth + 1);
      sub[UPPER_RIGHT]->cornersIter[UPPER_RIGHT] = cornersIter[UPPER_RIGHT];

      sub[LOWER_LEFT] = new MandelRegion (upperX, midDiagY2, midDiagX1, lowerY, imageX, imageY + subimageY, subimageX, pixelsY - subimageY, ownerFrame, depth + 1);
      sub[LOWER_LEFT]->cornersIter[LOWER_LEFT] = cornersIter[LOWER_LEFT];

      sub[LOWER_RIGHT] = new MandelRegion (midDiagX2, midDiagY2, lowerX, lowerY, imageX + subimageX, imageY + subimageY, pixelsX - subimageX, pixelsY - subimageY, ownerFrame, depth + 1);
      sub[LOWER_RIGHT]->cornersIter[LOWER_RIGHT] = cornersIter[LOWER_RIGHT];

      for (int i = 0; i < 4; i++)
        {
          q.append (sub[i]);
          //sub[i]->print();
        }
      ownerFrame->regionSplit ();
    }

  nvtxRangePop(); // closes "examine region"
}

// ============================================================
// Called by CalcThr::run() (via WorkQueue::extract returning NULL) once the
// work queue is empty and this thread is about to exit.  Prints the per-thread
// CPU timing summary accumulated by the thread_local counters above.
// ============================================================
void MandelRegion::printCPUSummary()
{
  if (cpuRegionCount == 0) return; // GPU-only thread: nothing to print
  fprintf(stderr,
          "[CPU profiling summary - this thread]\n"
          "  Regions computed on CPU : %ld\n"
          "  Total compute time      : %.3f ms  (avg %.3f ms/region)\n",
          cpuRegionCount,
          cpuTotalMs, cpuTotalMs / cpuRegionCount);
}

//--------------------------------------
bool MandelRegion::operator< (const MandelRegion & a)
{
  // cout << "Comparing " << this->pixelsX << " " << a.pixelsX<< endl; 
  int Npixels4a = a.pixelsX * a.pixelsY;
  int Npixels4b = this->pixelsX * this->pixelsY;
  return Npixels4a > Npixels4b;
}
