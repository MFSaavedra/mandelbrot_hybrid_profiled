/**
 * @file kernel.h
 * @brief Host-side interface to the CUDA Mandelbrot kernel (the GPU executor).
 *
 * Three @c extern @c "C" entry points bridge the C++ worker threads to the
 * CUDA translation unit (@ref kernel.cu). C linkage keeps the symbol names
 * identical across the @c nvcc and @c g++ object files.
 *
 * @see kernel.cu
 */
#ifndef KERNEL_H_
#define KERNEL_H_

/**
 * @brief GPU front-end: render one rectangular region on the device.
 *
 * Launches @c mandelKernel over the region's pixel grid (16&times;16 blocks),
 * synchronously copies the iteration-count results back to pinned host memory,
 * and records per-region CUDA-event timing. The returned buffer and pitch are
 * reused across every region processed by the GPU thread.
 *
 * @param uX,uY     Complex coordinate of the region's upper-left corner.
 * @param lX,lY     Complex coordinate of the region's lower-right corner.
 * @param resX,resY Region size in pixels.
 * @param[out] pixels  Receives a pointer to the host result buffer (iteration
 *                     counts, row-major with @p pitch stride).
 * @param[out] pitch   Receives the row stride of @p pixels, in bytes.
 * @param maxiter   Iteration ceiling (the frame's @c MAXITER).
 */
extern "C"
void hostFE (double uX, double uY, double lX, double lY, int resX, int resY, unsigned  int **pixels, int *pitch, int maxiter);

/**
 * @brief Allocate the reusable device + pinned-host buffers and CUDA events.
 *
 * Called once at startup (before the GPU thread runs) for the largest frame
 * resolution; the pitched device memory and mapped host memory are then reused
 * by every @ref hostFE call, so per-region allocation cost is zero.
 *
 * @param maxResX,maxResY Maximum frame resolution the run will request.
 * @return Pointer to the pinned host result buffer.
 */
extern "C"
unsigned  int *CUDAmemSetup(int maxResX, int maxResY);

/**
 * @brief Print the aggregate GPU timing summary and free all device resources.
 *
 * Called once after the GPU thread drains the work queue. Emits total/average
 * kernel and device&rarr;host memcpy time, then destroys the CUDA events and
 * frees the buffers allocated by @ref CUDAmemSetup.
 */
extern "C"
void CUDAmemCleanup();
#endif /* KERNEL_H_ */
