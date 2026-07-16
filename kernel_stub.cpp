/**
 * @file kernel_stub.cpp
 * @brief CPU-only link stub for @c make @c GPU=0 builds — a GPU-free
 *        implementation of the @ref kernel.h interface.
 *
 * Satisfies the linker on nodes without a usable CUDA stack (e.g. a Kepler
 * @c sm_30 GT 750M, dropped after CUDA 10.2). @ref main.cpp only calls
 * @ref CUDAmemSetup / @ref hostFE when @c gpuEnable=1, so a CPU-only run never
 * reaches these functions; if someone forces @c gpuEnable=1 on a GPU-less
 * binary, they abort loudly rather than render garbage.
 * @see kernel.h, kernel.cu
 */
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
