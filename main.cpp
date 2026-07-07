#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <limits.h>
#include <unistd.h>
#include <time.h>
#include <QThread>
#include <QImage>
#include <QPainter>
#include <QColor>
#include "workqueue.h"
#include "mandelframe.h"
#include "mandelregion.h"
#include "kernel.h"
#include "oclkernel.h"

// Profiling flags consulted by kernel.cu, mandelregion.cpp and mandelframe.cpp.
// Defined here so a single command-line switch controls them.  C linkage keeps
// the symbol name identical in nvcc and g++ object files.
extern "C" int profileQuiet = 0;   // 1 = suppress per-region stderr prints
extern "C" int profileSave  = 1;   // 0 = skip PNG saves (pure-compute timing)
extern "C" int vizMode      = 0;   // 1 = subdivision-animation mode (see below)

//************************************************************
// Draw the recorded subdivision outlines onto an image.  Internal (split)
// nodes are drawn as a grey skeleton first; computed leaves are drawn on top
// coloured by executor — cyan for the GPU thread, yellow for the CPU threads —
// so the load-balancing partition is directly visible.  maxDepth < 0 draws the
// complete partition; maxDepth >= 0 draws only nodes down to that depth.
static void drawVizOverlay(QImage &frame, const QVector<VizRect> &rects, int maxDepth)
{
  const QColor colSkeleton(90, 90, 90);     // internal / split nodes
  const QColor colGPU(0, 255, 255);         // leaves computed on the GPU thread
  const QColor colCPU(255, 230, 0);         // leaves computed on a CPU thread

  QPainter p(&frame);

  // Draw each border 2px thick, grown *inward* from the region's true boundary
  // (the outer rect marks the boundary; the inner rect, inset by 1px, thickens
  // it).  Growing inward keeps each colour entirely within its own region, so
  // a GPU (cyan) and a CPU (yellow) neighbour each show a solid 2px band rather
  // than two abutting 1px lines that blend to green when viewed at scale.
  auto strokeInward = [&](const VizRect &r) {
    p.drawRect(r.x, r.y, r.w - 1, r.h - 1);
    if (r.w > 3 && r.h > 3)
      p.drawRect(r.x + 1, r.y + 1, r.w - 3, r.h - 3);
  };

  p.setPen(QPen(colSkeleton, 1));
  for (const VizRect &r : rects)
    if (!r.leaf && (maxDepth < 0 || r.depth <= maxDepth))
      strokeInward(r);

  for (const VizRect &r : rects)
    if (r.leaf && (maxDepth < 0 || r.depth <= maxDepth))
      {
        p.setPen(QPen(r.onGPU ? colGPU : colCPU, 1));
        strokeInward(r);
      }
  p.end();
}

//************************************************************
// viz=1: render a single frozen frame (interpolation disabled) and replay the
// recorded subdivision tree as a depth-by-depth animation.  For each recursion
// depth k it writes "<prefix>_dNN.png": a copy of the finished image overlaid
// with the region outlines that exist down to depth k.  Runs after the
// wall-clock timer stops, so it never perturbs the measured compute phase.
static void generateDepthFrames(MandelFrame *f, const char *prefix)
{
  if (f->vizRects.isEmpty())
    {
      fprintf(stderr, "[viz] no recorded rectangles — nothing to draw\n");
      return;
    }

  int maxDepth = 0;
  for (const VizRect &r : f->vizRects)
    if (r.depth > maxDepth)
      maxDepth = r.depth;

  char outName[MAXFNAME + 16];
  for (int k = 0; k <= maxDepth; k++)
    {
      QImage frame = f->img->copy();        // detached deep copy per depth level
      drawVizOverlay(frame, f->vizRects, k);
      snprintf(outName, sizeof(outName), "%s_d%02d.png", prefix, k);
      frame.save(outName, "PNG");
      fprintf(stderr, "[viz] wrote %s (outlines for depth <= %d)\n", outName, k);
    }
  fprintf(stderr, "[viz] %d subdivision nodes, max depth %d\n",
          (int) f->vizRects.size(), maxDepth);
}

//************************************************************
// viz=2: render the full interpolated sequence (no frame frozen) and overlay
// each frame's complete subdivision partition, coloured by executor.  Writes
// the overlaid images under the normal "<prefix>NNNN.png" names so they can be
// assembled into a movie of the whole run.  Runs after the timer stops.
static void generateSequenceFrames(MandelFrame **fr, int n, const char *prefix)
{
  char outName[MAXFNAME + 16];
  long totalRects = 0;
  for (int i = 0; i < n; i++)
    {
      QImage frame = fr[i]->img->copy();
      drawVizOverlay(frame, fr[i]->vizRects, -1);   // -1 = full partition
      snprintf(outName, sizeof(outName), "%s%04d.png", prefix, i);
      frame.save(outName, "PNG");
      totalRects += fr[i]->vizRects.size();
    }
  fprintf(stderr, "[viz] wrote %d overlaid sequence frames (%ld subdivision nodes total)\n",
          n, totalRects);
}

//************************************************************
// viz=3: render the full interpolated sequence AND animate the splitting
// process itself.  For each run frame it emits one image per recursion depth
// (depth 0 = the whole frame, up to its terminal partition), then advances the
// camera to the next frame and starts over.  The result, played in order, shows
// "split, split, split, advance camera, split, ..." across the entire run.
// Because the work queue is processed concurrently, this is a depth-ordered
// reconstruction (deterministic), not a wall-clock replay.  Output PNGs are
// numbered with a single global counter so ffmpeg can assemble them directly.
static void generateProcessFrames(MandelFrame **fr, int n, const char *prefix)
{
  char outName[MAXFNAME + 24];
  int  seq = 0;
  long totalRects = 0;
  for (int i = 0; i < n; i++)
    {
      int maxDepth = 0;
      for (const VizRect &r : fr[i]->vizRects)
        if (r.depth > maxDepth)
          maxDepth = r.depth;

      for (int k = 0; k <= maxDepth; k++)
        {
          QImage frame = fr[i]->img->copy();
          drawVizOverlay(frame, fr[i]->vizRects, k);
          snprintf(outName, sizeof(outName), "%s%05d.png", prefix, seq++);
          frame.save(outName, "PNG");
        }
      totalRects += fr[i]->vizRects.size();
    }
  fprintf(stderr, "[viz] wrote %d process frames across %d run frames (%ld subdivision nodes total)\n",
          seq, n, totalRects);
}


//************************************************************
class CalcThr:public QThread
{
private:
  WorkQueue * que;
  ExecKind kind;
  int resX, resY;               // frame size; used only to size the iGPU buffers

public:
    CalcThr (WorkQueue * q, ExecKind k, int rx = 0, int ry = 0)
      : que (q), kind (k), resX (rx), resY (ry)
  {
  }
  void run ();
};

void CalcThr::run ()
{
  // The integrated-GPU worker owns its OpenCL context/queue/buffers for its
  // entire lifetime, so it builds them here on its own thread.  (The discrete-
  // GPU worker's CUDA setup is done by main() around thr[0].)
  if (kind == EXEC_IGPU)
    OCLmemSetup (resX, resY);

  // GPU-class workers (CUDA or iGPU) pull the largest pending region; CPU
  // workers pull the smallest -- the affinity rule in WorkQueue::extract.
  bool isGPU = (kind != EXEC_CPU);
  MandelRegion *t;
  while ((t = que->extract (isGPU)) != NULL)
    {
      t->examine (*que, kind);
      delete t;
    }
  // Work queue is empty: print the accumulated CPU timing for this thread.
  // GPU-class threads print nothing here -- CUDA regions are summarised in
  // CUDAmemCleanup(), iGPU regions in OCLmemCleanup().
  MandelRegion::printCPUSummary();

  if (kind == EXEC_IGPU)
    OCLmemCleanup ();
}

//************************************************************
// Expects an input file with the following data:
//  numframes resolutionX resolutionY imageFilePrefix.
//  upperCornerX upperCornerY lowerCornerX lowerCornerY maxIterations  ; for first frame
//  upperCornerX upperCornerY lowerCornerX lowerCornerY maxIterations  ; for last frame
// 
// Command-line parameters:  spec_file numThr GPUenable diffThreshold pixelThreshold quiet save
//              spec_file : the file holding the parameters mentioned above
//              numThr : number of threads (optional, defaults to the number of cores)
//              GPUenable : backend mode (optional, default 1):
//                          0 = CPU only, 1 = discrete GPU (CUDA),
//                          2 = integrated GPU (OpenCL), 3 = both GPUs.
//              diffThreshold pixelThreshold : optional thresholds for frame partitioning heuristics
//              quiet : 0/1, 1 suppresses per-region stderr prints (default 0)
//              save  : 0/1, 0 skips PNG image saves for pure-compute timing (default 1)
//              viz   : 0/1/2/3 (default 0).  1 = freeze one frame (vizFrame) and emit the
//                      depth-by-depth subdivision animation <prefix>_fNNNN_dKK.png.
//                      2 = full interpolated sequence, each frame overlaid with its
//                      complete subdivision partition (saved as <prefix>NNNN.png).
//                      3 = full sequence AND the splitting process: per run frame, one
//                      image per recursion depth (depth 0..terminal), then advance the
//                      camera (saved as <prefix>NNNNN.png, single global counter).
int main (int argc, char *argv[])
{
  int numframes, resolutionX, resolutionY;
  char imageFilePrefix[MAXFNAME - 8];
  double upperCornerX[2], upperCornerY[2], lowerCornerX[2], lowerCornerY[2];
  int maxIterations[2];
  double diffT = 0.5;
  int pixT = 32768;
  int vizFrame = 0;             // which interpolated frame to freeze on in vizMode

  if (argc < 2)
    {
      cerr << "Usage : " << argv[0]
           << " spec_file [numThr] [gpuMode] [diffT] [pixT] [quiet] [save] [viz] [vizFrame]\n"
           << "        gpuMode: 0=CPU only  1=discrete GPU (CUDA)  "
              "2=integrated GPU (OpenCL)  3=both GPUs\n";
      exit (1);
    }

  int numThreads = sysconf (_SC_NPROCESSORS_ONLN);
  if (argc > 2)
    numThreads = atoi (argv[2]);

  // arg 3 is a backend mode (bitmask): bit0 = discrete GPU (CUDA), bit1 =
  // integrated GPU (OpenCL).  0 = CPU only, 1 = dGPU (historical default),
  // 2 = iGPU only, 3 = both.  The old 0/1 meanings are preserved.
  int gpuMode = 1;
  if (argc > 3)
    gpuMode = atoi (argv[3]);
  // Only bits 0/1 are defined; anything else (e.g. a typo like "4") would
  // otherwise silently degrade to CPU-only.
  if (gpuMode & ~3)
    {
      cerr << "[config] gpuMode=" << gpuMode << " has undefined bits set "
              "(valid: 0=CPU 1=dGPU 2=iGPU 3=both); using gpuMode="
           << (gpuMode & 3) << "\n";
      gpuMode &= 3;
    }
  bool useCUDA = (gpuMode & 1) != 0;
  bool useIGPU = (gpuMode & 2) != 0;
  // In dual-GPU mode the iGPU runs on worker thread 1; without a spare thread
  // drop it rather than starve the CUDA path.
  bool igpuOnWorker = useCUDA && useIGPU;
  if (igpuOnWorker && numThreads < 2)
    {
      cerr << "[config] both GPUs requested but numThr<2; dropping iGPU "
              "(no spare worker thread)\n";
      igpuOnWorker = false;
      useIGPU = false;
    }
  bool anyGPU = useCUDA || useIGPU;

  if (argc > 4)
    diffT = atof (argv[4]);

  if (argc > 5)
    pixT = atoi (argv[5]);

  if (argc > 6)
    profileQuiet = atoi (argv[6]);

  if (argc > 7)
    profileSave = atoi (argv[7]);

  if (argc > 8)
    vizMode = atoi (argv[8]);

  if (argc > 9)
    vizFrame = atoi (argv[9]);

  // viz=2/3 overlay the full sequence; the overlaid PNGs are the deliverable,
  // so suppress the plain per-frame save done in regionComplete.
  if (vizMode >= 2)
    profileSave = 0;

  ifstream fin (argv[1]);
  fin >> numframes >> resolutionX >> resolutionY;
  fin >> imageFilePrefix;
  fin >> upperCornerX[0] >> upperCornerY[0] >> lowerCornerX[0] >> lowerCornerY[0] >> maxIterations[0];
  fin >> upperCornerX[1] >> upperCornerY[1] >> lowerCornerX[1] >> lowerCornerY[1] >> maxIterations[1];
  fin.close ();

  // generate the pseudocolor map to be used for all frames
  int MAXMAXITER = max (maxIterations[0], maxIterations[1]);
  MandelRegion::initColorMapAndThrer (MAXMAXITER, diffT, pixT);

  WorkQueue workQ;
  // CPU-only runs: CPU threads use largest-first (LPT) instead of smallest-first.
  workQ.setGpuPresent (anyGPU);

  // generate the needed frame objects and the corresponding regions
  double uX = upperCornerX[0], uY = upperCornerY[0];
  double lX = lowerCornerX[0], lY = lowerCornerY[0];
  int iter = maxIterations[0];
  double sx1, sx2, sy1, sy2;
  int iterInc;
  sx1 = (upperCornerX[1] - upperCornerX[0]) / numframes;        // steps are a little bit smaller to avoid round-off errors causing the
  sx2 = (lowerCornerX[1] - lowerCornerX[0]) / numframes;        // last image to not render
  sy1 = (upperCornerY[1] - upperCornerY[0]) / numframes;
  sy2 = (lowerCornerY[1] - lowerCornerY[0]) / numframes;
  iterInc = (maxIterations[1] - maxIterations[0]) * 1.0 / numframes;
  char fname[MAXFNAME];
  char vizPrefix[MAXFNAME];

  // viz=1 renders exactly one interpolated frame (selected by vizFrame),
  // freezing the camera; the depth animation replaces the usual sequence.
  // viz=0 and viz=2 build the full interpolated sequence.
  // The interpolation steps above are computed against the real frame count so
  // the chosen frame matches what a full run would have produced.
  int framesToBuild = (vizMode == 1) ? 1 : numframes;
  MandelFrame **fr = new MandelFrame *[framesToBuild];

  if (vizMode == 1)
    {
      if (vizFrame < 0)          vizFrame = 0;
      if (vizFrame >= numframes) vizFrame = numframes - 1;
      double fuX = upperCornerX[0] + sx1 * vizFrame;
      double fuY = upperCornerY[0] + sy1 * vizFrame;
      double flX = lowerCornerX[0] + sx2 * vizFrame;
      double flY = lowerCornerY[0] + sy2 * vizFrame;
      int    fiter = maxIterations[0] + iterInc * vizFrame;
      // Per-frame prefix so depth frames from different vizFrames don't collide.
      sprintf (vizPrefix, "%s_f%04d", imageFilePrefix, vizFrame);
      sprintf (fname, "%s.png", vizPrefix);
      fr[0] = new MandelFrame (fuX, fuY, flX, flY, resolutionX, resolutionY, fname, fiter);
      fr[0]->frameIndex = vizFrame;
      workQ.append (new MandelRegion (fuX, fuY, flX, flY, 0, 0, resolutionX, resolutionY, fr[0]));
    }
  else
    {
      for (int i = 0; i < numframes; i++)
        {
          sprintf (fname, "%s%04i.png", imageFilePrefix, i);
          fr[i] = new MandelFrame (uX, uY, lX, lY, resolutionX, resolutionY, fname, iter);
          fr[i]->frameIndex = i;
          workQ.append (new MandelRegion (uX, uY, lX, lY, 0, 0, resolutionX, resolutionY, fr[i]));
          uX += sx1;
          uY += sy1;
          lX += sx2;
          lY += sy2;
          iter += iterInc;
        }
    }


  // Self-describing banner so each run's log file identifies its configuration.
  fprintf(stderr,
          "[config] numThr=%d gpuMode=%d (CUDA=%d iGPU=%d) diffT=%.3f pixT=%d "
          "res=%dx%d frames=%d quiet=%d save=%d viz=%d vizFrame=%d\n",
          numThreads, gpuMode, (int)useCUDA, (int)useIGPU, diffT, pixT,
          resolutionX, resolutionY, framesToBuild, profileQuiet, profileSave,
          vizMode, vizMode ? vizFrame : -1);

  // generate the threads that will process the workload.  Thread 0 runs on the
  // main thread below; it drives the discrete GPU (CUDA) when requested, else
  // the integrated GPU, else CPU.  When both GPUs are active, thread 0 is CUDA
  // and the iGPU runs on worker thread 1; all remaining threads are CPU.
  ExecKind kind0 = useCUDA ? EXEC_CUDA : (useIGPU ? EXEC_IGPU : EXEC_CPU);
  CalcThr **thr = new CalcThr *[numThreads];
  thr[0] = new CalcThr (&workQ, kind0, resolutionX, resolutionY);
  for (int i = 1; i < numThreads; i++)
    {
      ExecKind k = (igpuOnWorker && i == 1) ? EXEC_IGPU : EXEC_CPU;
      thr[i] = new CalcThr (&workQ, k, resolutionX, resolutionY);
      thr[i]->start ();
    }

  // Wall-clock end-to-end timer wraps the entire compute phase (CUDAmemSetup +
  // worker threads + image saves done inside regionComplete + thread joins).
  // CUDAmemCleanup is included since its summary print is cheap and finishes
  // before t1; image-save time is naturally counted because the worker thread
  // that decrements remainingRegions to 0 performs img->save() before exiting.
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  // use the main thread to run one of the workers.  Only the CUDA path needs
  // its setup/cleanup wrapped here; an iGPU thread-0 (gpuMode 2) builds and
  // tears down its own OpenCL state inside run().
  if (kind0 == EXEC_CUDA)
    {
      CUDAmemSetup (resolutionX, resolutionY);
      thr[0]->run ();
      CUDAmemCleanup ();
    }
  else
    thr[0]->run ();

  for (int i = 1; i < numThreads; i++)
      thr[i]->wait ();

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double elapsed_s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

  // Machine-parseable line: sweep.sh greps for this exact prefix.
  fprintf(stderr, "[total_elapsed_s] %.6f\n", elapsed_s);

  // Emit the visualization outside the timed region so it never counts against
  // the measured compute phase.
  if (vizMode == 1)
    generateDepthFrames(fr[0], vizPrefix);       // one frame, depth animation
  else if (vizMode == 2)
    generateSequenceFrames(fr, numframes, imageFilePrefix);  // full run, final partition per frame
  else if (vizMode == 3)
    generateProcessFrames(fr, numframes, imageFilePrefix);   // full run, depth expansion per frame

  return 0;
}
