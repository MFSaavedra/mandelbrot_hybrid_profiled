================================================================================
RUNNING THE PROGRAM
================================================================================

Setup the parameters in a text file (e.g. spec.in) and invoke:

    ./mandelHybrid spec.in <numThreads> <gpuEnable> [diffThreshold] [pixelThreshold]

    numThreads     : total CPU+GPU worker threads (defaults to CPU core count)
    gpuEnable      : 1 enables GPU (default), 0 for CPU-only
    diffThreshold  : fraction (default 0.5) — a region is considered uniform
                     enough to compute directly when:
                         (maxCornerIter - minCornerIter) < diffThresh * maxCornerIter
    pixelThreshold : pixel count (default 32768) — regions smaller than this
                     are always computed without further splitting

Example: ./mandelHybrid spec.in 7 1

Output images are PNGs written to the prefix given in spec.in (e.g. "img/")
as img/0000.png, img/0001.png, etc.  Create the output directory first.

spec.in format:
    numframes resolutionX resolutionY imageFilePrefix
    upperCornerX upperCornerY lowerCornerX lowerCornerY maxIterations  ; first frame
    upperCornerX upperCornerY lowerCornerX lowerCornerY maxIterations  ; last frame

Intermediate frames are interpolated linearly between the two corner specs.


================================================================================
PROFILING INSTRUMENTATION
================================================================================

Two complementary profiling mechanisms have been added to this program:
NVTX annotations (for Nsight Systems timeline views) and CUDA events
(for precise GPU-side kernel and memcpy timing).  Both operate at near-zero
overhead when no profiler is attached.

--------------------------------------------------------------------------------
1. NVTX Ranges  (kernel.cu, mandelregion.cpp)
--------------------------------------------------------------------------------

NVTX (NVIDIA Tools Extension) is a CPU-side annotation API.
nvtxRangePush("label") / nvtxRangePop() bracket a named interval on the
calling thread's stack.  Nsight Systems records the open/close timestamps
and renders them as colored bars in the CPU timeline, one lane per thread.

NVTX3 (CUDA 11+) is header-only: it dlopen()s the profiler's runtime
library only when Nsight Systems is active.  When the program runs standalone
there is no measurable overhead.

Three ranges were added:

  "examine region"      (mandelregion.cpp, MandelRegion::examine)
      Wraps the full decision cycle for one region: corner evaluation plus
      either a pixel compute or a four-way split back onto the work queue.
      Lets you visually count how many examine() calls each thread handles
      and compare depth/duration distributions between CPU and GPU workers.

  "CPU region compute"  (mandelregion.cpp, MandelRegion::compute — CPU branch)
      Wraps the pixel iteration loop for CPU-computed regions.
      Shows CPU workers competing with the GPU thread for work-queue items.

  "GPU region compute"  (kernel.cu, hostFE)
      Wraps the full GPU path: kernel launch + device-to-host memcpy + result
      copy into the QImage buffer.  Appears on thread 0's lane alongside the
      automatically recorded CUDA API calls.

--------------------------------------------------------------------------------
2. CUDA Event Timing  (kernel.cu, hostFE / CUDAmemSetup / CUDAmemCleanup)
--------------------------------------------------------------------------------

cudaEvent_t is an opaque GPU-side timestamp.  cudaEventRecord() inserts a
"record this timestamp" command into the CUDA stream; it completes
asynchronously alongside kernel/memcpy work.  cudaEventSynchronize() blocks
the host until the event has been stamped, then cudaEventElapsedTime()
returns wall-clock milliseconds between two events as measured by the GPU's
own hardware counter.

This is more accurate than wrapping the launch with gettimeofday() because
the GPU executes asynchronously; a host-side timer would include driver/queue
overhead and could miss the true execution time.

Four events are created once in CUDAmemSetup() and reused across all kernel
calls: evKernelStart, evKernelStop, evMemcpyStart, evMemcpyStop.

Each call to hostFE() prints a per-region line to stderr:
    [GPU region   42] size  960x 540  kernel 1.234 ms  D->H 0.456 ms

CUDAmemCleanup() prints an aggregate summary:
    [GPU profiling summary]
      Regions computed on GPU : 42
      Total kernel time       : 51.828 ms  (avg 1.234 ms/region)
      Total D->H memcpy time  : 19.152 ms  (avg 0.456 ms/region)

--------------------------------------------------------------------------------
3. CPU Thread Timing  (mandelregion.cpp, MandelRegion::compute / printCPUSummary)
--------------------------------------------------------------------------------

clock_gettime(CLOCK_MONOTONIC) brackets the pixel iteration loop for every
CPU-computed region.  thread_local counters accumulate region count and total
compute time per thread with no synchronization overhead.

Each CPU region prints a line to stderr:
    [CPU region    7] size  480x 270  compute 3.210 ms

When a thread's work queue drains, MandelRegion::printCPUSummary() prints:
    [CPU profiling summary - this thread]
      Regions computed on CPU : 7
      Total compute time      : 22.470 ms  (avg 3.210 ms/region)

The GPU thread (thread 0) prints nothing from printCPUSummary() because its
regions are counted inside hostFE() and summarised in CUDAmemCleanup().


================================================================================
HOW TO PROFILE WITH NSIGHT
================================================================================

--- Nsight Systems (timeline view, recommended first step) ---

    mkdir -p img
    nsys profile --trace=cuda,nvtx,osrt -o report ./mandelHybrid spec.in 7 1
    nsys-ui report.nsys-rep          # open in the Nsight Systems GUI

    # or print a text summary without the GUI:
    nsys stats report.nsys-rep

What to look for in the timeline:
  - The NVTX rows show "examine region", "CPU region compute", and
    "GPU region compute" bars side by side with CUDA API calls.
  - Compare how much wall time each CalcThr lane spends on GPU vs CPU work.
  - Watch when the work queue drains: GPU idle time at the end indicates
    CPU threads are finishing the last small regions while the GPU waits.
  - The CUDA row shows kernel execution and memcpy back-to-back; a large gap
    between them would indicate CPU-side stalls feeding the queue.

--- Nsight Compute (kernel deep-dive) ---

    ncu --set full -o kernel_report ./mandelHybrid spec.in 1 1
    ncu-ui kernel_report.ncu-rep

    Limit to 1 thread and 1 frame first: ncu replays each kernel launch
    multiple times to collect all hardware counter sets, so a full run with
    many regions is very slow.

What to look for:
  - Occupancy: are all SMs busy?  The 16x16 thread block gives 256 threads;
    check whether the GPU's warp scheduler is fully utilized.
  - Memory throughput: mandelKernel is compute-bound (no global reads after
    the kernel starts), but the D->H memcpy bandwidth is visible here.
  - Warp stall reasons: long-divergence regions near the Mandelbrot boundary
    cause warp divergence; ncu will flag stalls due to branch divergence.
