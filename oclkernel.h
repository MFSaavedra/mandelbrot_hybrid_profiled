#ifndef OCLKERNEL_H_
#define OCLKERNEL_H_

// OpenCL front-end for the integrated GPU (Intel UHD Graphics 630 on the
// i7-9750H).  Deliberately mirrors the CUDA interface in kernel.h so the rest
// of the program treats the discrete GPU (CUDA) and the integrated GPU (OpenCL)
// identically: MandelRegion::compute() picks a front-end by executor kind and
// then runs the same result-copy/metrics path for both.
//
// Unlike kernel.cu (compiled by nvcc), this translation unit is plain C++
// compiled by g++ against CUDA's bundled OpenCL headers (/opt/cuda/include/CL)
// and linked with the ICD loader (-lOpenCL).  All three functions are intended
// to be called from a single worker thread (the iGPU CalcThr), which owns the
// OpenCL context/queue/buffers for its entire lifetime — same single-thread
// assumption kernel.cu makes for the CUDA path.

// Initialise OpenCL on the integrated GPU and allocate the reusable device +
// host buffers sized for the largest frame.  Returns the host result buffer
// (row-major, tight stride = resX ints).  Aborts with an actionable message if
// no non-NVIDIA GPU OpenCL device is present (i.e. the Intel runtime is not
// installed).  Must be called from the thread that will issue oclFE().
unsigned int *OCLmemSetup(int maxResX, int maxResY);

// Compute one region on the integrated GPU.  Signature is identical to
// kernel.h's hostFE(): writes the result buffer pointer into *pixels and its
// byte pitch into *pitch.  Pitch is tight here (resX * sizeof(unsigned)).
void oclFE(double uX, double uY, double lX, double lY, int resX, int resY,
           unsigned int **pixels, int *pitch, int maxiter);

// Print the accumulated iGPU timing summary and release all OpenCL resources.
void OCLmemCleanup();

#endif /* OCLKERNEL_H_ */
