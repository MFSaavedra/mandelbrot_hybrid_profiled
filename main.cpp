#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <limits.h>
#include <unistd.h>
#include <time.h>
#include <QThread>
#include "workqueue.h"
#include "mandelframe.h"
#include "mandelregion.h"
#include "kernel.h"

// Profiling flags consulted by kernel.cu, mandelregion.cpp and mandelframe.cpp.
// Defined here so a single command-line switch controls them.  C linkage keeps
// the symbol name identical in nvcc and g++ object files.
extern "C" int profileQuiet = 0;   // 1 = suppress per-region stderr prints
extern "C" int profileSave  = 1;   // 0 = skip PNG saves (pure-compute timing)


//************************************************************
class CalcThr:public QThread
{
private:
  WorkQueue * que;
  bool isGPU;

public:
    CalcThr (WorkQueue * q, bool gpu):que (q), isGPU (gpu)
  {
  }
  void run ();
};

void CalcThr::run ()
{
  MandelRegion *t;
  while ((t = que->extract ()) != NULL)
    {
      t->examine (*que, isGPU);
      delete t;
    }
  // Work queue is empty: print the accumulated CPU timing for this thread.
  // The GPU thread prints nothing here because its regions are counted inside
  // hostFE() and summarised in CUDAmemCleanup().
  MandelRegion::printCPUSummary();
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
//              GPUenable : 0/1, 1(default) enables the GPU code (optional)
//              diffThreshold pixelThreshold : optional thresholds for frame partitioning heuristics
//              quiet : 0/1, 1 suppresses per-region stderr prints (default 0)
//              save  : 0/1, 0 skips PNG image saves for pure-compute timing (default 1)
int main (int argc, char *argv[])
{
  int numframes, resolutionX, resolutionY;
  char imageFilePrefix[MAXFNAME - 8];
  double upperCornerX[2], upperCornerY[2], lowerCornerX[2], lowerCornerY[2];
  int maxIterations[2];
  double diffT = 0.5;
  int pixT = 32768;

  if (argc < 2)
    {
      cerr << "Usage : " << argv[0]
           << " spec_file [numThr] [GPUenable] [diffT] [pixT] [quiet] [save]\n";
      exit (1);
    }

  int numThreads = sysconf (_SC_NPROCESSORS_ONLN);
  if (argc > 2)
    numThreads = atoi (argv[2]);

  bool enableGPU = true;
  if (argc > 3)
    enableGPU = (bool) atoi (argv[3]);

  if (argc > 4)
    diffT = atof (argv[4]);

  if (argc > 5)
    pixT = atoi (argv[5]);

  if (argc > 6)
    profileQuiet = atoi (argv[6]);

  if (argc > 7)
    profileSave = atoi (argv[7]);

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

  // generate the needed frame objects and the corresponding regions
  MandelFrame **fr = new MandelFrame *[numframes];
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
  for (int i = 0; i < numframes; i++)
    {
      sprintf (fname, "%s%04i.png", imageFilePrefix, i);
      fr[i] = new MandelFrame (uX, uY, lX, lY, resolutionX, resolutionY, fname, iter);
      workQ.append (new MandelRegion (uX, uY, lX, lY, 0, 0, resolutionX, resolutionY, fr[i]));
      uX += sx1;
      uY += sy1;
      lX += sx2;
      lY += sy2;
      iter += iterInc;
    }


  // Self-describing banner so each run's log file identifies its configuration.
  fprintf(stderr,
          "[config] numThr=%d gpuEnable=%d diffT=%.3f pixT=%d res=%dx%d "
          "frames=%d quiet=%d save=%d\n",
          numThreads, (int)enableGPU, diffT, pixT,
          resolutionX, resolutionY, numframes, profileQuiet, profileSave);

  // generate the threads that will process the workload
  CalcThr **thr = new CalcThr *[numThreads];
  thr[0] = new CalcThr (&workQ, enableGPU);
  for (int i = 1; i < numThreads; i++)
    {
      thr[i] = new CalcThr (&workQ, false);
      thr[i]->start ();
    }

  // Wall-clock end-to-end timer wraps the entire compute phase (CUDAmemSetup +
  // worker threads + image saves done inside regionComplete + thread joins).
  // CUDAmemCleanup is included since its summary print is cheap and finishes
  // before t1; image-save time is naturally counted because the worker thread
  // that decrements remainingRegions to 0 performs img->save() before exiting.
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  // use the main thread to run one of the workers
  if (enableGPU)
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

  return 0;
}
