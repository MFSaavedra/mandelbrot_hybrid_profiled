// ===========================================================================
// oclkernel.cpp — OpenCL front-end for the integrated GPU (Intel UHD 630).
//
// This is the iGPU twin of kernel.cu.  It exposes the same three-call interface
// (OCLmemSetup / oclFE / OCLmemCleanup) so MandelRegion::compute() can dispatch
// a region to either GPU by executor kind and then run an identical
// result-copy/metrics path.  Compiled by g++ (not nvcc) against CUDA's bundled
// OpenCL headers; linked through the ICD loader (-lOpenCL), which dispatches at
// runtime to whichever vendor ICD drives the chosen device.
//
// Design parity with kernel.cu:
//   * device + host buffers allocated once in OCLmemSetup, reused per region;
//   * OpenCL profiling events time the kernel and the device->host read, just
//     as CUDA events do, and feed the same per-region / summary stderr lines;
//   * the FP64 Mandelbrot kernel below is a line-for-line port of mandelKernel.
//
// Why FP64 matters here (see reports 16/17): the kernel is double-precision and
// FP64-bound on the discrete GPU.  Intel iGPUs expose FP64 at a much reduced
// rate and some drivers omit cl_khr_fp64 entirely, in which case the program
// build below fails — we detect that early and print exactly what is missing.
// ===========================================================================

#ifndef _GNU_SOURCE
#define _GNU_SOURCE           // for strcasestr (GNU extension in <string.h>)
#endif
#define CL_TARGET_OPENCL_VERSION 120   // pin to the OpenCL 1.2 API subset

#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// NVTX CPU-side annotation, same as kernel.cu / mandelregion.cpp.  Lets the
// iGPU worker's compute spans show up on its own lane in the Nsight Systems
// timeline alongside the CUDA GPU thread and the CPU workers.
#include <nvtx3/nvToolsExt.h>

// Defined in main.cpp.  When set, the per-region stderr line is suppressed;
// the once-per-run summary in OCLmemCleanup() still prints.
extern "C" int profileQuiet;

// ---------------------------------------------------------------------------
#define CL_CHECK(value)                                                        \
  {                                                                            \
    cl_int _clStat = (value);                                                  \
    if (_clStat != CL_SUCCESS) {                                               \
      fprintf(stderr, "OpenCL error %d at line %d in file %s\n",               \
              (int)_clStat, __LINE__, __FILE__);                               \
      exit(1);                                                                 \
    }                                                                          \
  }

static const int BLOCK_SIDE = 16;   // 16x16 work-group, mirrors the CUDA block

// ---------------------------------------------------------------------------
// The Mandelbrot kernel, embedded as a string so there is no .cl file to locate
// at runtime.  diverge() and mandelKernel() are direct ports of kernel.cu: same
// escape test, same iteration recurrence, same pixel->complex mapping.  The one
// layout difference is the tight row stride (out[y*resX + x]) instead of CUDA's
// rounded pitch; compute()'s copy loop is told the matching pitch by oclFE().
static const char *KERNEL_SRC = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

int diverge(double cx, double cy, int MAXITER) {
  int iter = 0;
  double vx = cx, vy = cy, tx, ty;
  while (iter < MAXITER && (vx * vx + vy * vy) < 4.0) {
    tx = vx * vx - vy * vy + cx;
    ty = 2.0 * vx * vy + cy;
    vx = tx;
    vy = ty;
    iter++;
  }
  return iter;
}

__kernel void mandelKernel(__global unsigned int *out,
                           double upperX, double upperY,
                           double stepX, double stepY,
                           int resX, int resY, int MAXITER) {
  int myX = get_global_id(0);
  int myY = get_global_id(1);
  if (myX >= resX || myY >= resY)
    return;
  double tempx = upperX + myX * stepX;
  double tempy = upperY - myY * stepY;
  out[myY * resX + myX] = (unsigned int) diverge(tempx, tempy, MAXITER);
}
)CLC";

// ---------------------------------------------------------------------------
// OpenCL state, file-scope and reused across every region.  Touched only by the
// single iGPU worker thread (same single-thread assumption as kernel.cu's
// d_res / events), so no synchronisation is needed here.
static cl_context       ctx     = NULL;
static cl_command_queue queue   = NULL;
static cl_program       program = NULL;
static cl_kernel        kernel  = NULL;
static cl_device_id     device  = NULL;
static cl_mem           d_res   = NULL;   // device result buffer (max-frame sized)
static unsigned int    *h_res   = NULL;   // host result buffer returned to compute()
static int              bufResX = 0;       // max dimensions the buffers were sized for
static int              bufResY = 0;

// Cumulative timing, mirrors kernel.cu's totals.
static double totalKernelMs = 0.0;
static double totalReadMs   = 0.0;
static long   totalRegions  = 0;

// ---------------------------------------------------------------------------
// Pick the integrated-GPU OpenCL device.  Strategy, in order:
//   1. honour MANDEL_OCL_DEVICE (case-insensitive substring of the device name);
//   2. otherwise prefer a GPU whose vendor contains "Intel";
//   3. otherwise the first GPU that is NOT NVIDIA (we drive the discrete card
//      through CUDA, so running OpenCL on it too would be pointless contention);
//   4. if only NVIDIA / no GPU device exists, abort with install guidance.
// All discovered devices are printed so a failed run is self-explanatory.
static cl_device_id selectIGPU()
{
  const char *want = getenv("MANDEL_OCL_DEVICE");

  cl_uint nplat = 0;
  clGetPlatformIDs(0, NULL, &nplat);
  if (nplat == 0) {
    fprintf(stderr,
            "[iGPU] no OpenCL platforms found. Install the Intel compute "
            "runtime (Arch: 'sudo pacman -S intel-compute-runtime').\n");
    exit(1);
  }
  cl_platform_id *plats = (cl_platform_id *) malloc(nplat * sizeof(cl_platform_id));
  clGetPlatformIDs(nplat, plats, NULL);

  cl_device_id envMatch = NULL, intelDev = NULL, nonNvidiaDev = NULL;

  fprintf(stderr, "[iGPU] enumerating OpenCL GPU devices:\n");
  for (cl_uint p = 0; p < nplat; p++) {
    cl_uint ndev = 0;
    if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 0, NULL, &ndev) != CL_SUCCESS
        || ndev == 0)
      continue;
    cl_device_id *devs = (cl_device_id *) malloc(ndev * sizeof(cl_device_id));
    clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, ndev, devs, NULL);

    for (cl_uint d = 0; d < ndev; d++) {
      char name[256] = {0}, vendor[256] = {0};
      clGetDeviceInfo(devs[d], CL_DEVICE_NAME,   sizeof(name),   name,   NULL);
      clGetDeviceInfo(devs[d], CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
      fprintf(stderr, "         platform %u device %u: %s [%s]\n",
              p, d, name, vendor);

      if (want && strcasestr(name, want) && !envMatch)
        envMatch = devs[d];
      if (strcasestr(vendor, "Intel") && !intelDev)
        intelDev = devs[d];
      if (!strcasestr(vendor, "NVIDIA") && !nonNvidiaDev)
        nonNvidiaDev = devs[d];
    }
    free(devs);
  }
  free(plats);

  cl_device_id chosen = envMatch ? envMatch
                      : intelDev ? intelDev
                      : nonNvidiaDev;
  if (!chosen) {
    fprintf(stderr,
            "[iGPU] no integrated/non-NVIDIA GPU OpenCL device found.\n"
            "       The discrete GPU is driven via CUDA; the iGPU needs the\n"
            "       Intel compute runtime. On Arch:\n"
            "         sudo pacman -S intel-compute-runtime\n"
            "       (then re-run; verify with 'clinfo -l').\n");
    exit(1);
  }
  char name[256] = {0};
  clGetDeviceInfo(chosen, CL_DEVICE_NAME, sizeof(name), name, NULL);
  fprintf(stderr, "[iGPU] selected device: %s\n", name);
  return chosen;
}

// ---------------------------------------------------------------------------
unsigned int *OCLmemSetup(int maxResX, int maxResY)
{
  cl_int err;
  device  = selectIGPU();
  bufResX = maxResX;
  bufResY = maxResY;

  // FP64 capability check — the kernel is double precision.  Warn loudly before
  // attempting the build so a missing cl_khr_fp64 produces a clear diagnosis
  // rather than an opaque build failure.
  cl_device_fp_config fp64 = 0;
  clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp64), &fp64, NULL);
  if (fp64 == 0)
    fprintf(stderr,
            "[iGPU] WARNING: device reports no FP64 (CL_DEVICE_DOUBLE_FP_CONFIG=0).\n"
            "       The Mandelbrot kernel is double precision; the build will\n"
            "       likely fail. This iGPU may simply lack usable FP64.\n");

  ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CL_CHECK(err);

  // In-order queue with profiling enabled (OpenCL 1.2 entry point) so the same
  // event-timing scheme as the CUDA path is available.
  queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
  CL_CHECK(err);

  program = clCreateProgramWithSource(ctx, 1, &KERNEL_SRC, NULL, &err);
  CL_CHECK(err);
  err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t logSize = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    char *log = (char *) malloc(logSize + 1);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
    log[logSize] = '\0';
    fprintf(stderr, "[iGPU] kernel build failed:\n%s\n", log);
    free(log);
    exit(1);
  }

  kernel = clCreateKernel(program, "mandelKernel", &err);
  CL_CHECK(err);

  // Reusable buffers sized for the largest frame.  The device buffer is
  // write-only from the kernel's perspective; the host buffer is a plain array
  // that compute() reads with a tight stride (pitch = resX ints).
  size_t bytes = (size_t) maxResX * maxResY * sizeof(unsigned int);
  d_res = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
  CL_CHECK(err);
  h_res = (unsigned int *) malloc(bytes);
  if (!h_res) { fprintf(stderr, "[iGPU] host buffer malloc failed\n"); exit(1); }

  return h_res;
}

// ---------------------------------------------------------------------------
void oclFE(double upperX, double upperY, double lowerX, double lowerY,
           int resX, int resY, unsigned int **pixels, int *currpitch, int MAXITER)
{
  nvtxRangePush("iGPU region compute");

  double stepX = (lowerX - upperX) / resX;   // same mapping as hostFE()
  double stepY = (upperY - lowerY) / resY;

  // Kernel arguments, in declaration order.
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),  &d_res));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(double),  &upperX));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(double),  &upperY));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(double),  &stepX));
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(double),  &stepY));
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),     &resX));
  CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),     &resY));
  CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),     &MAXITER));

  // Round the global range up to whole 16x16 work-groups; the in-kernel bounds
  // check drops the overhang (exactly how the CUDA grid/block + guard work).
  auto roundUp = [](int v, int m) { return ((v + m - 1) / m) * m; };
  size_t local[2]  = { (size_t) BLOCK_SIDE, (size_t) BLOCK_SIDE };
  size_t global[2] = { (size_t) roundUp(resX, BLOCK_SIDE),
                       (size_t) roundUp(resY, BLOCK_SIDE) };

  cl_event evKernel, evRead;
  CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local,
                                  0, NULL, &evKernel));

  // Device->host read of just this region's tightly packed rows.
  size_t bytes = (size_t) resX * resY * sizeof(unsigned int);
  CL_CHECK(clEnqueueReadBuffer(queue, d_res, CL_TRUE, 0, bytes, h_res,
                               0, NULL, &evRead));

  // Resolve device-side durations from the profiling events (nanoseconds).
  cl_ulong ks = 0, ke = 0, rs = 0, re = 0;
  clGetEventProfilingInfo(evKernel, CL_PROFILING_COMMAND_START, sizeof(ks), &ks, NULL);
  clGetEventProfilingInfo(evKernel, CL_PROFILING_COMMAND_END,   sizeof(ke), &ke, NULL);
  clGetEventProfilingInfo(evRead,   CL_PROFILING_COMMAND_START, sizeof(rs), &rs, NULL);
  clGetEventProfilingInfo(evRead,   CL_PROFILING_COMMAND_END,   sizeof(re), &re, NULL);
  double kernelMs = (ke - ks) * 1e-6;
  double readMs   = (re - rs) * 1e-6;
  clReleaseEvent(evKernel);
  clReleaseEvent(evRead);

  totalKernelMs += kernelMs;
  totalReadMs   += readMs;
  totalRegions++;

  if (!profileQuiet)
    fprintf(stderr,
            "[iGPU region %4ld] size %4dx%4d  kernel %.3f ms  D->H %.3f ms\n",
            totalRegions, resX, resY, kernelMs, readMs);

  *pixels    = h_res;
  *currpitch = resX * sizeof(unsigned int);   // tight stride, in bytes

  nvtxRangePop();
}

// ---------------------------------------------------------------------------
void OCLmemCleanup()
{
  fprintf(stderr,
          "[iGPU profiling summary]\n"
          "  Regions computed on iGPU : %ld\n"
          "  Total kernel time        : %.3f ms  (avg %.3f ms/region)\n"
          "  Total D->H read time     : %.3f ms  (avg %.3f ms/region)\n",
          totalRegions,
          totalKernelMs, totalRegions ? totalKernelMs / totalRegions : 0.0,
          totalReadMs,   totalRegions ? totalReadMs   / totalRegions : 0.0);

  if (h_res)   { free(h_res); h_res = NULL; }
  if (d_res)   clReleaseMemObject(d_res);
  if (kernel)  clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue)   clReleaseCommandQueue(queue);
  if (ctx)     clReleaseContext(ctx);
}
