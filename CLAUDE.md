# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
make          # build mandelHybrid binary
make clean    # remove all object files and the binary
```

Dependencies: `nvcc` (CUDA), `g++`, Qt5 (`Qt5Core Qt5Gui` via pkg-config), CUDA installed at `/opt/cuda`.

CUDA architecture is set to `-arch=native` (requires CUDA 11.6+). For older `nvcc`, change `CUDA_ARCH` in the Makefile to a specific arch like `-arch=sm_61`.

## Run

```bash
./mandelHybrid spec.in <numThreads> <gpuEnable> [diffThreshold] [pixelThreshold]
```

- `numThreads`: total CPU+GPU worker threads (defaults to CPU core count)
- `gpuEnable`: `1` enables GPU (default), `0` CPU-only
- `diffThreshold`: fraction (default 0.5) — region is uniform enough to compute if `(maxCornerIter - minCornerIter) < diffThresh * maxCornerIter`
- `pixelThreshold`: pixel count (default 32768) — regions smaller than this are always computed without further splitting

Example from README: `./mandelHybrid spec.in 7 1`

Output images are PNGs written by concatenating the spec file prefix with the zero-padded frame number: prefix `img` → `img0000.png`, `img0001.png`, … in the current working directory. To write into a subdirectory, include the slash in the prefix (e.g. `img/`).

## spec.in format

```
numframes resolutionX resolutionY imageFilePrefix
upperCornerX upperCornerY lowerCornerX lowerCornerY maxIterations   # first frame
upperCornerX upperCornerY lowerCornerX lowerCornerY maxIterations   # last frame
```

Intermediate frames are interpolated linearly between the two corner specs.

## Architecture

This is a hybrid CPU+GPU Mandelbrot set renderer that uses adaptive region splitting as its load-balancing strategy.

**Execution model**: One `CalcThr` (QThread subclass) per logical worker. Thread 0 drives the GPU path; all others are CPU-only. All threads pull from a shared `WorkQueue` (mutex-protected `deque<MandelRegion*>`).

**Adaptive subdivision** (`MandelRegion::examine`): Each region evaluates its four corners first. If the corners are "uniform" (low iteration-count spread relative to `diffThresh`) or the region is smaller than `pixelSizeThresh`, it is computed immediately (`MandelRegion::compute`). Otherwise it is split into four equal quadrants and all four are pushed back onto the work queue. This keeps work granularity dynamic and concentrates GPU/CPU effort on complex boundary regions.

**GPU path** (`kernel.cu`): `CUDAmemSetup` pre-allocates pitched device memory and pinned host memory once at startup. `hostFE` launches `mandelKernel` (16×16 thread blocks) and synchronously copies results back. GPU memory is reused across all regions processed by thread 0.

**Frame completion** (`MandelFrame`): Uses a `QAtomicInt remainingRegions` counter. `regionSplit` increments it by 3 (one region becomes four, net +3); `regionComplete` decrements it; when it hits zero the frame saves to disk.

**Key files**:
- `main.cpp` — argument parsing, frame/region setup, thread launch
- `mandelregion.cpp/.h` — adaptive subdivision logic and pixel computation (CPU branch)
- `kernel.cu` / `kernel.h` — CUDA kernel and host-side GPU interface
- `mandelframe.cpp/.h` — per-frame image buffer and atomic region counter
- `workqueue.cpp/.h` — thread-safe task queue

## Profiling instrumentation (already added)

Two mechanisms are in place — no further instrumentation is needed.

**NVTX annotations** (`mandelregion.cpp`, `kernel.cu`): `nvtxRangePush/Pop` brackets around `examine()`, the CPU compute path in `compute()`, and the full `hostFE()` call. These appear as labeled, colored spans on each thread's lane in the Nsight Systems timeline. NVTX3 (CUDA 11+) is header-only (`nvtx3/nvToolsExt.h`); linked via `-ldl` in the Makefile (no `-lnvToolsExt` needed).

**CUDA event timing** (`kernel.cu`): four `cudaEvent_t` handles (`evKernelStart/Stop`, `evMemcpyStart/Stop`) are created once in `CUDAmemSetup` and reused across all 3,543 kernel calls. Each `hostFE` call records timestamps before/after the kernel and before/after `cudaMemcpy`, then resolves them with `cudaEventSynchronize` + `cudaEventElapsedTime`. Per-region lines and a final summary are printed to stderr.

**CPU thread timing** (`mandelregion.cpp`): `thread_local` counters (`cpuRegionCount`, `cpuTotalMs`) accumulate per-thread stats via `clock_gettime(CLOCK_MONOTONIC)`. `MandelRegion::printCPUSummary()` is called from `CalcThr::run()` when the work queue drains.

To profile:
```bash
mkdir -p img
nsys profile --trace=cuda,nvtx,osrt -o report ./mandelHybrid spec.in
nsys stats report.nsys-rep        # text summary
nsys-ui report.nsys-rep           # GUI timeline
ncu --set full -o kern ./mandelHybrid spec.in 1 1   # kernel deep-dive
```

A profiling report (`report.nsys-rep`, `report.sqlite`) from a 100-frame 1920×1080 run on a GTX 1660 Ti Max-Q with 12 threads already exists in this directory.

## Profiling results (100 frames, 1920×1080, GTX 1660 Ti Max-Q, 12 threads)

| Metric | Value |
|---|---|
| Total leaf regions | 4,603 (3,543 GPU + 1,060 CPU) |
| Total examine() calls | 6,104 (splits + leaf computes) |
| GPU avg time per region | 13.5 ms |
| CPU avg time per region | 535 ms (GPU is ~40× faster) |
| CPU thread wall-time spread | 50.5 s – 52.6 s (4% imbalance) |
| GPU thread wall time | 47.1 s (finishes ~3–5 s before last CPU thread) |
| Actual D→H transfer time | 15 µs avg (53 ms total for 600 MB) |
| `cudaMemcpy` reported time | 13.4 ms avg (99.9% is kernel wait, not transfer) |
| Worst CPU region | 15.3 s (corner heuristic misclassified a boundary region) |

**Key finding**: `cudaMemcpy` spends nearly all of its reported time waiting for the GPU kernel to finish (the call is synchronous). The actual PCIe transfer is 15 µs avg at ~11.3 GB/s — bandwidth is not a bottleneck.

**Suggested improvements** (not yet implemented):
1. Async CUDA streams — overlap kernel N with memcpy of kernel N-1 to recover the ~13 ms/region idle time on thread 0.
2. Lower `diffThreshold` (0.5 → ~0.2) to force more splits on boundary regions and eliminate the 15.3 s outlier.
3. Replace `WorkQueue`'s `deque` with a `priority_queue` using the existing `MandelRegion::Compare` functor (largest-first) so the GPU thread receives larger regions.

## Report

A full LaTeX report covering the instrumentation changes and analysis is in `profiling_report/main.tex`. Build with:
```bash
cd profiling_report && pdflatex main.tex && pdflatex main.tex
```
