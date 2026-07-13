/**
 * @file kernel.cu
 * @brief CUDA implementation of the GPU executor: the Mandelbrot device kernel
 *        and the host-side front-end (@ref hostFE) with its reusable device /
 *        pinned-host buffers and CUDA-event timing.
 *
 * This translation unit provides the GPU half of the @ref kernel.h interface.
 * @ref CUDAmemSetup allocates once, @ref hostFE renders one region per call
 * (launch + synchronous copy-back + per-region timing), and @ref CUDAmemCleanup
 * prints the aggregate GPU summary and frees everything. NVTX ranges bracket
 * the host front-end so the GPU worker's activity is visible in Nsight Systems.
 * @see kernel.h
 */
#include <cstddef>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// nvToolsExt provides the NVTX (NVIDIA Tools Extension) API.
// NVTX lets you push/pop named "ranges" onto a per-thread stack; Nsight
// Systems records the open/close timestamps and displays them as colored
// spans on the CPU timeline.  Zero runtime overhead when not profiling.
// nvToolsExt.h lives under nvtx3/ in CUDA 11+ installations.
#include <nvtx3/nvToolsExt.h>

using namespace std;

#define CUDA_CHECK_RETURN(value)                                               \
  {                                                                            \
    cudaError_t _m_cudaStat = value;                                           \
    if (_m_cudaStat != cudaSuccess) {                                          \
      fprintf(stderr, "Error %s at line %d in file %s\n",                      \
              cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);            \
      exit(1);                                                                 \
    }                                                                          \
  }

static const int BLOCK_SIDE = 16; // size of 2D block of threads

// Defined in main.cpp.  Suppresses per-region stderr lines when set, leaving
// the once-per-run summary intact.
extern "C" int profileQuiet;

// ============================================================
// CUDA event timing helpers
//
// cudaEvent_t is an opaque GPU-side timestamp.  cudaEventRecord()
// inserts a "record this timestamp" command into the CUDA stream; it
// completes asynchronously alongside kernel/memcpy work.
// cudaEventSynchronize() blocks the host until the event has been
// stamped, then cudaEventElapsedTime() returns wall-clock milliseconds
// between two events as measured by the GPU's own hardware counter.
//
// This is more accurate than wrapping the launch with gettimeofday()
// because the GPU executes asynchronously; a host-side timer would
// include driver/queue overhead and could miss true execution time.
// ============================================================
static cudaEvent_t evKernelStart, evKernelStop;
static cudaEvent_t evMemcpyStart,  evMemcpyStop;

// Cumulative totals across all hostFE calls on this thread.
static double totalKernelMs = 0.0;
static double totalMemcpyMs = 0.0;
static long   totalRegions  = 0;

//************************************************************
/**
 * @brief Device-side escape-iteration count for one point (mirrors the CPU
 *        @ref MandelRegion::diverge).
 *
 * Iterates @f$z \to z^2 + c@f$ from @p (cx,cy) and returns the escape
 * iteration, or @p MAXITER if the point does not escape. Carries the same exact
 * Brent periodicity check as the CPU path: a lane that detects an exactly
 * repeating orbit returns @p MAXITER early and sits masked while the rest of
 * the warp finishes, so the warp's time is bounded by its slowest lane's
 * detection latency rather than @c 32&times;MAXITER. Exactness (no epsilon)
 * keeps the image byte-identical.
 * @param cx,cy   Complex coordinate of the point.
 * @param MAXITER Iteration ceiling.
 * @return Escape iteration in [0, MAXITER].
 */
__device__ int diverge(double cx, double cy, int MAXITER) {
  int iter = 0;
  double vx = cx, vy = cy, tx, ty;
  // Brent-style periodicity check, mirroring the CPU diverge() in
  // mandelregion.cpp.  Interior orbits settle into an attracting cycle; once
  // the FP state exactly revisits a saved state the (deterministic) iteration
  // can never escape, so returning MAXITER now is *exact* with respect to
  // this kernel's own (FMA-contracted) arithmetic -- the plain loop would
  // have ground to MAXITER and returned the same value.  Exact equality (no
  // epsilon) keeps exterior pixels untouched: an escaping orbit never exactly
  // repeats.  The saved point is refreshed at doubling intervals so a cycle
  // of period p entered after a transient of t is caught in O(max(p, t)).
  // A lane that detects a cycle exits and sits masked while the rest of the
  // warp finishes: the warp's time drops from MAXITER to the *slowest* lane's
  // detection latency.  Warp execution efficiency on interior regions falls
  // below the 100% of report 16, but wall time is what the check buys.
  double sx = vx, sy = vy;
  int nextSave = 8;
  while (iter < MAXITER && (vx * vx + vy * vy) < 4) {
    tx = vx * vx - vy * vy + cx;
    ty = 2 * vx * vy + cy;
    vx = tx;
    vy = ty;
    iter++;
    if (vx == sx && vy == sy)
      return MAXITER;
    if (iter == nextSave) {
      sx = vx;
      sy = vy;
      nextSave *= 2;
    }
  }
  return iter;
}

//************************************************************
/**
 * @brief Render one region: one CUDA thread computes one pixel.
 *
 * Launched over a 2D grid of 16&times;16 blocks covering the region. Each
 * thread maps its @c (blockIdx,threadIdx) to a pixel, converts it to a complex
 * coordinate via the per-pixel steps, calls @ref diverge, and writes the
 * iteration count into the pitched device buffer. Threads outside the
 * @c resX&times;resY region return immediately.
 * @param[out] d_res Pitched device result buffer (iteration counts).
 * @param upperX,upperY Complex coordinate of the region's upper-left pixel.
 * @param stepX,stepY   Per-pixel complex increment.
 * @param resX,resY     Region size in pixels.
 * @param pitch         Row stride of @p d_res in bytes.
 * @param MAXITER       Iteration ceiling.
 */
__global__ void mandelKernel(unsigned *d_res, double upperX, double upperY,
                             double stepX, double stepY, int resX, int resY,
                             int pitch, int MAXITER) {
  int myX, myY;
  myX = blockIdx.x * blockDim.x + threadIdx.x;
  myY = blockIdx.y * blockDim.y + threadIdx.y;
  if (myX >= resX || myY >= resY)
    return;

  double tempx, tempy;
  tempx = upperX + myX * stepX;
  tempy = upperY - myY * stepY;
  int color = diverge(tempx, tempy, MAXITER);
  d_res[myY * pitch / sizeof(int) + myX] = color;
}

//************************************************************
int maxResX = 0;
int maxResY = 0;
size_t pitch = 0;
unsigned int *h_res;
unsigned int *d_res;

//************************************************************
extern "C" void CUDAmemCleanup() {
  // Print the accumulated GPU-side timing summary gathered via CUDA events.
  // These numbers reflect only time the GPU spent executing the kernel and
  // performing the device→host copy — host-side overhead is excluded.
  fprintf(stderr,
          "[GPU profiling summary]\n"
          "  Regions computed on GPU : %ld\n"
          "  Total kernel time       : %.3f ms  (avg %.3f ms/region)\n"
          "  Total D→H memcpy time   : %.3f ms  (avg %.3f ms/region)\n",
          totalRegions,
          totalKernelMs, totalRegions ? totalKernelMs / totalRegions : 0.0,
          totalMemcpyMs, totalRegions ? totalMemcpyMs / totalRegions : 0.0);

  // Destroy the CUDA events we created in CUDAmemSetup.
  cudaEventDestroy(evKernelStart);
  cudaEventDestroy(evKernelStop);
  cudaEventDestroy(evMemcpyStart);
  cudaEventDestroy(evMemcpyStop);

  CUDA_CHECK_RETURN(cudaFreeHost(h_res));
  CUDA_CHECK_RETURN(cudaFree(d_res));
}

//************************************************************
extern "C" unsigned int *CUDAmemSetup(int maxResX, int maxResY) {
  CUDA_CHECK_RETURN(cudaMallocPitch((void **)&d_res, (size_t *)&pitch,
                                    maxResX * sizeof(unsigned), maxResY));
  CUDA_CHECK_RETURN(
      cudaHostAlloc(&h_res, maxResY * pitch, cudaHostAllocMapped));

  // Create CUDA events once; reuse them for every kernel call.
  // cudaEventCreate allocates GPU-side timer resources.
  CUDA_CHECK_RETURN(cudaEventCreate(&evKernelStart));
  CUDA_CHECK_RETURN(cudaEventCreate(&evKernelStop));
  CUDA_CHECK_RETURN(cudaEventCreate(&evMemcpyStart));
  CUDA_CHECK_RETURN(cudaEventCreate(&evMemcpyStop));

  return h_res;
}

//************************************************************
// Host front-end function that allocates the memory and launches the GPU kernel
extern "C" void hostFE(double upperX, double upperY, double lowerX,
                       double lowerY, int resX, int resY, unsigned int **pixels,
                       int *currpitch, int MAXITER) {

  // NVTX range: visible in Nsight Systems CPU timeline as "GPU region compute".
  // nvtxRangePush/Pop bracket a named interval on the calling thread's lane.
  // Nsight Systems captures these as user-defined ranges and color-codes them
  // alongside CUDA API calls, giving a unified view of CPU+GPU activity.
  nvtxRangePush("GPU region compute");

  int blocksX, blocksY;
  blocksX = (int)ceil(resX * 1.0 / BLOCK_SIDE);
  blocksY = (int)ceil(resY * 1.0 / BLOCK_SIDE);
  dim3 block(BLOCK_SIDE, BLOCK_SIDE);
  dim3 grid(blocksX, blocksY);

  int ptc = 32;
  while (ptc < resX * sizeof(unsigned))
    ptc += 32;

  double stepX = (lowerX - upperX) / resX;
  double stepY = (upperY - lowerY) / resY;

  // Record a GPU timestamp immediately before the kernel launch.
  // cudaEventRecord() enqueues the stamp into the default stream so it
  // executes after all previously-queued work finishes.
  CUDA_CHECK_RETURN(cudaEventRecord(evKernelStart));

  mandelKernel<<<grid, block>>>(d_res, upperX, upperY, stepX, stepY, resX, resY,
                                ptc, MAXITER);

  // Stamp the moment the kernel finishes (still async — we resolve it below).
  CUDA_CHECK_RETURN(cudaEventRecord(evKernelStop));

  // Stamp before the device→host copy.
  CUDA_CHECK_RETURN(cudaEventRecord(evMemcpyStart));

  CUDA_CHECK_RETURN(
      cudaMemcpy(h_res, d_res, resY * ptc, cudaMemcpyDeviceToHost));

  CUDA_CHECK_RETURN(cudaEventRecord(evMemcpyStop));

  // cudaEventSynchronize blocks until the GPU has written all timestamps above.
  // After this call, cudaEventElapsedTime gives accurate wall-clock durations.
  CUDA_CHECK_RETURN(cudaEventSynchronize(evMemcpyStop));

  float kernelMs = 0.0f, memcpyMs = 0.0f;
  CUDA_CHECK_RETURN(cudaEventElapsedTime(&kernelMs, evKernelStart, evKernelStop));
  CUDA_CHECK_RETURN(cudaEventElapsedTime(&memcpyMs, evMemcpyStart, evMemcpyStop));

  totalKernelMs += kernelMs;
  totalMemcpyMs += memcpyMs;
  totalRegions++;

  // Per-region timing printed to stderr so it doesn't mix with image output.
  if (!profileQuiet)
    fprintf(stderr,
            "[GPU region %4ld] size %4dx%4d  kernel %.3f ms  D→H %.3f ms\n",
            totalRegions, resX, resY, kernelMs, memcpyMs);

  *pixels = h_res;
  *currpitch = ptc;

  nvtxRangePop(); // closes "GPU region compute"
}
