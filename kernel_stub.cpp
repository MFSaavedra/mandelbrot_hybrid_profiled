// CPU-only link stub (make GPU=0): satisfies kernel.h's extern "C" interface
// on nodes without a usable CUDA stack — e.g. the GT 750M node (Kepler sm_30,
// dropped after CUDA 10.2).  main.cpp only calls CUDAmemSetup/hostFE when
// gpuEnable=1, so a CPU-only run (arg3 = 0) never reaches these; if someone
// forces gpuEnable=1 on a GPU-less binary, fail loudly instead of rendering
// garbage.
#include <stdio.h>
#include <stdlib.h>
#include "kernel.h"

static void noGPU()
{
  fprintf(stderr,
          "This binary was built without CUDA (make GPU=0); run with gpuEnable=0.\n");
  exit(1);
}

extern "C" unsigned int *CUDAmemSetup(int, int)
{
  noGPU();
  return NULL;                  // not reached
}

extern "C" void hostFE(double, double, double, double, int, int,
                       unsigned int **, int *, int)
{
  noGPU();
}

extern "C" void CUDAmemCleanup()
{
  // Reachable only after CUDAmemSetup, which never returns here.
}
