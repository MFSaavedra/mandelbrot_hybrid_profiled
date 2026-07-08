# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
make          # build mandelHybrid binary
make clean    # remove all object files and the binary
```

Dependencies: `nvcc` (CUDA), `g++`, Qt5 (`Qt5Core Qt5Gui` via pkg-config), CUDA installed at `/opt/cuda`, and (since `feat/igpu-opencl`) the OpenCL ICD loader (`-lOpenCL`, package `ocl-icd`) — a hard *link* dependency even for CPU/CUDA-only use; the CL headers come from `/opt/cuda/include`. Running the iGPU additionally needs the Intel runtime (see Run). The Makefile pins `nvcc -ccbin g++-15` because `nvcc` 13.2 does not yet support gcc 16; the host compile uses the same `g++-15` so ABI is consistent across `nvcc`/`g++` translation units.

CUDA architecture is set to `-arch=native` (requires CUDA 11.6+). For older `nvcc`, change `CUDA_ARCH` in the Makefile to a specific arch like `-arch=sm_61`.

## Run

```bash
./mandelHybrid spec.in <numThreads> <gpuMode> [diffThreshold] [pixelThreshold] [quiet] [save] [viz] [vizFrame]
```

- `numThreads`: total CPU+GPU worker threads (defaults to CPU core count)
- `gpuMode`: backend bitmask (default `1`). `0` = CPU only, `1` = discrete GPU (CUDA), `2` = integrated GPU (OpenCL, `feat/igpu-opencl`), `3` = both GPUs. Bit0 = dGPU/CUDA, bit1 = iGPU/OpenCL. The historical `0`/`1` meanings are preserved. The iGPU runs on its own `CalcThr` (thread 1 in dual mode) and needs the Intel OpenCL runtime installed; without it the run aborts with install guidance.
- `diffThreshold`: fraction (default 0.5; **recommended new default: 0.1** — see Sensitivity studies). Region is uniform enough to compute if `(maxCornerIter - minCornerIter) < diffThresh * maxCornerIter`
- `pixelThreshold`: pixel count (default 32768). Regions smaller than this are always computed without further splitting
- `quiet`: `1` suppresses per-region stderr prints; per-thread and per-GPU summaries still emit
- `save`: `0` skips PNG writes (for pure-compute timing)
- `viz`: visualization mode, `0`/`1`/`2`/`3` (default `0`). All modes colour region outlines by executor (cyan = GPU thread, yellow = CPU thread, grey = split skeleton), stroke borders 2 px thick grown inward (so adjacent GPU/CPU cells don't blend to green), and render after the wall-clock timer stops so timing is never perturbed.
  - `1`: freeze one interpolated frame (`vizFrame`) and emit its depth-by-depth animation `<prefix>_fNNNN_dKK.png` (one PNG per recursion depth `K`).
  - `2`: full interpolated sequence, each frame overlaid with its complete partition (`<prefix>NNNN.png`) — a movie of the whole zoom.
  - `3`: full sequence AND the splitting process — per run frame, one image per depth (root→terminal) then advance the camera (`<prefix>NNNNN.png`, single global counter). Deterministic depth-ordered reconstruction, not a wall-clock replay.
- `vizFrame`: in `viz=1`, which interpolated frame index to freeze on (default `0`). Frame 0 is the wide full-set view (`maxIter=100`, cheap, no GPU participation); late frames are deep zooms with high `maxIter` where the GPU participates and the outliers live (e.g. `89`).
- `viz=2`/`viz=3` force `save=0` (the overlaid PNGs are the deliverable). Assemble to video with ffmpeg at native res, `-bf 0`, `-fps_mode passthrough` (no temporal interpolation); use `-pix_fmt yuv444p` for zero chroma blending. See `experiments/08-region-metrics-viz/` for the generated movies.

Example: `./mandelHybrid spec.in 7 1`

Output images are PNGs written by concatenating the spec file prefix with the zero-padded frame number: prefix `img` → `img0000.png`, `img0001.png`, … in the current working directory. To write into a subdirectory, include the slash in the prefix (e.g. `img/`).

## spec.in format

```
numframes resolutionX resolutionY imageFilePrefix
upperCornerX upperCornerY lowerCornerX lowerCornerY maxIterations   # first frame
upperCornerX upperCornerY lowerCornerX lowerCornerY maxIterations   # last frame
```

Intermediate frames are interpolated linearly between the two corner specs.

## Repository layout

```
mandelbrot_hybrid_profiled/
├── *.cpp, *.h, *.cu           source files (flat at root, matches Makefile)
├── Makefile, spec.in, README.txt, CLAUDE.md
├── reports/                   LaTeX reports + shared template/library
│   ├── template.tex, template_config.tex, library.bib, natnumurl.bst
│   ├── img/                   figures referenced by reports
│   ├── 01-initial-profile.tex
│   ├── 02-fig1115-replication.tex
│   ├── 03-bug-analysis.tex
│   ├── 04-postfix-profile.tex
│   ├── 05-fig1115-postfix.tex
│   ├── 06-difft-sweep.tex
│   ├── 07-difft-compare.tex
│   └── 08-region-metrics-viz.tex
├── experiments/               raw measurement data, numbered to match reports
│   ├── 01-initial-profile/    (was first_run/)
│   ├── 02-fig1115-prefix/     (was sweep_results/)
│   ├── 04-postfix-profile/    (was second_run/)
│   ├── 05-fig1115-postfix/    (was sweep_results_postfix/)
│   ├── 06-difft-postfix/      (was difft_sweep/)
│   ├── 07-difft-prefix/       (was difft_sweep_prefix/)
│   └── 08-region-metrics-viz/ (viz frames, metrics + outlier logs, perf A/B)
└── scripts/
    ├── sweep_fig1115.sh       (was sweep.sh; auto-resolves project root)
    ├── plot_fig1115.py
    └── build_report.sh        runs pdflatex twice for any report
```

Note: report `03` (bug analysis) has no corresponding `experiments/03-...` because it is pure code analysis with no associated measurement run.

## Branches and tags

- `main` — current best version. Holds all reports and all experiment data.
- `feat/viz-mode` — **merged into `main`** (fast-forward to commit `616c147`, tag `binary-v2-viz`). Added region work metrics, the 5 s outlier dump, and the three-mode `vizMode` subdivision visualizer (`viz=1/2/3`, report `08`). Instrumentation only; the subdivision decision is unchanged (verified: byte-identical 5,323-leaf decomposition vs `d5bf30c`).
- `examine/9-points-sampling` — **merged into `main`** (tag `binary-v3-9point`). Replaces the 4-corner uniformity test with a 9-point stencil (report `09`). Catches the frame-89 outlier but not frame 73; wall-neutral.
- `feat/priority-queue` — **NOT merged** (tag `binary-v4-pq`). Largest-first shared `priority_queue`; wall-neutral negative result (report `11`) — a shared queue can't route big regions to the GPU (11:1 CPU:GPU extraction). Code deliberately kept off `main` (FIFO retained until superseded by GPU affinity).
- `feat/gpu-affinity` — **merged into `main`** (tag `binary-v5-affinity`). Min-max work queue with GPU affinity + CPU-only LPT guard (report `12`). First wall win: −3.6%; routes the interior outlier to the GPU. Supersedes the FIFO queue.
- `examine/maxiter-split` — **NOT merged** (tag `binary-maxiter-split`, commit `d572c78`, atop `binary-v5-affinity`). Replaces the `diffThresh` relative-spread test with a parameter-free binary rule: compute iff all 9 stencil samples reached MAXITER, else split (report `18`). **Byte-identical decomposition** to `diffT=0.1` on the production zoom (5608 leaves) and on inside/Misiurewicz/seahorse; differs only on uniform-exterior `outside` (+22% leaves, flat wall). A characterization — shows `diffT=0.1` *is* an interior certificate — not an optimization; kept off `main`.
- `feat/igpu-opencl` — **NOT merged** (no tag yet). Adds a **third executor**: an OpenCL backend (`oclkernel.cpp/.h`) that runs the FP64 Mandelbrot kernel on the **integrated GPU** (Intel UHD 630). `arg3` becomes a backend bitmask (`gpuMode`: 0=CPU, 1=dGPU/CUDA, 2=iGPU/OpenCL, 3=both); the iGPU runs on its own `CalcThr` with the same GPU affinity as the dGPU (pulls largest). `compute()` dispatches by `ExecKind {EXEC_CPU, EXEC_CUDA, EXEC_IGPU}` and shares a `commitGPUResult()` copy/metrics helper between both GPU backends. Output is **equivalent to the CUDA path to within FP rounding** (full-run check `experiments/20-igpu-opencl/verify_output.{sh,csv}`: byte-identical only on the 2 shallow frames; GPU-vs-GPU 0.050% of all pixels differ, max 0.149%/frame, growing with depth — differing FMA contraction; both GPUs differ from the CPU render near-identically, ~8.07%). **Measured a wall win, not the predicted wash** (experiment 20, headline A/B, diffT=0.1, save=1, full spec, 3 reps): `dGPU+iGPU+10CPU` **44.32 s vs `dGPU+11CPU` 57.72 s = −23.2%** (and −72.6% vs CPU12's 161.6 s), winning *while giving up a CPU worker*. The earlier "marginal-to-negative, CPU pool is the binding wall" prediction was wrong: the co-binding constraint is the **serial tail of big coherent interior regions only a GPU runs fast** — a second accelerator pulling from the largest end parallelizes that tail (mode-3 split: dGPU 2915 / iGPU 1422 regions). iGPU-alone (mode 2) 81.7 s still ≈halves the CPU floor. **Needs the legacy Intel runtime** — modern `intel-compute-runtime` (NEO ≥24.x) dropped Gen9.5; UHD 630 (`8086:3e9b`) needs `intel-compute-runtime-legacy1` (AUR). Report `20` + `experiments/20-igpu-opencl/` document it. **Re-priced atop v6 (`main` merged in at `833a791`, experiment `22`): the iGPU’s value collapses to a wash** — mode 3 28.05±0.12 s vs mode 1 27.36±1.60 s (+2.5% on all-reps means, −0.7% excluding the cold first run; |Δ| ≪ mode-1 spread). Report 20’s −23.2% was priced against the CPU interior grind that v6 eliminates; post-periodicity the CPU pool still binds in both modes (busiest thread ≈ wall), and the ~21 s of kernel work the iGPU absorbs is exactly cancelled by giving up a CPU worker (pool 292→265 s over 11→10 threads = same 26.5 s/thread). **Recommendation: keep off `main`** — an OpenCL ICD + legacy-runtime dependency for zero wall; the branch stays as a characterization (report 20 = the pre-v6 win, report 22 + `experiments/22-igpu-atop-v6/` = its post-v6 repeal, including the 13-thread and BlockingSync probes and the device re-pricing: dGPU 23→2.7 CPU-worker equivalents, iGPU 12.7→≈1). Unmeasured residuals if revisited: cheapening the host-side commit (`scanLine` vs per-pixel `setPixel` — GPU throughput is *lane*-bound, see report 20 §Analysis) and strict dGPU-priority-on-largest (both GPUs currently race for the biggest region).
- `examine/random-sampling` — **NOT merged, WIP** (no report/experiment yet). Replaces the 9-point stencil's five fixed interior probes with `sampleN` uniform random probes per region (env `SAMPLE_N`, default 5; `0` = 4-corner rule). The decomposition becomes **nondeterministic by design** — the branch exists to measure that variance (smoke test, 4-frame 640×360 spec: 718 leaves every run at `SAMPLE_N=0` vs 721–730 varying at 5). Analytically it should catch the binding outliers *less* often than the stencil at equal budget: P(split) ≈ 1−(interior fraction)^5 → ~58% for the frame-89 region (stencil: always), ~14% for frame-73 (stencil: never), E[binding outliers] ≈ 1.28/run vs the stencil's 1.0 — muted on `main` because GPU affinity routes those regions to the GPU anyway. A characterization branch, not an optimization; random-vs-stencil A/Bs cannot use byte-identical decomposition checks.
- `feat/periodicity-check` — **merged into `main`** (fast-forward to `0f33b0f`, tag `binary-v6-periodicity` at code commit `fa6d8ac`). **Current best.** Exact Brent periodicity check in the CPU `diverge()`: a saved orbit state is refreshed at doubling intervals and the loop returns MAXITER as soon as the FP state *exactly* revisits it — an exactly-repeating orbit can never escape, so every pixel's return value (hence image and decomposition) is unchanged; only time changes. Verified: 100/100 frames byte-identical (mode-0 cmp), 5,608 leaves in all 12 A/B runs. **First work-reduction lever, biggest win so far: CPU12 −78.6% (164.4→35.2 s), hybrid dGPU+11CPU −46.3% (57.6→30.9 s, new best wall)** — vs −3.6% for affinity (report 12) and −23.2% for the iGPU (report 20); CPU12 pool compute drops 1,917→402 s (work eliminated, not moved). The hybrid self-rebalances: CPU threads pull 4,313/5,608 regions (was 2,148), the GPU keeps the 1,295 largest (12.4→19.6 ms/region). CPU branch only per report 16's caveat (GPU interior kernels are 100% warp-coherent + FP64-bound; A/B a GPU-side variant separately). Report `21` + `experiments/21-periodicity-check/`. Re-opens two measurements: the iGPU's marginal value (report 20's −23.2% was priced against the now-gone CPU interior grind — re-run experiment 20 atop v6 before merging `feat/igpu-opencl`) and the experiment-19 threshold optimum. Found along the way: `fin >> imageFilePrefix` into `char[MAXFNAME-8]` (42 bytes) overflows on spec prefixes >41 chars — stack smash, garbage `maxIterations[]`; production `img` prefix unaffected; fix separately.
- `examine_minmax_bugfix` — the commit that fixed the min/max reduction in `MandelRegion::examine()`. Same code state as `main` modulo subsequent reorganization commits.
- `examine_rewrite`, `master` — original code with the min/max tracking bug intact. Kept as historical references.

Tags pin the binary versions every report ultimately points at:

- `binary-v0-buggy` → commit `0f782fd` — original code with the min/max tracking bug
- `binary-v1-bugfix` → commit `d5bf30c` — same code with the one-line fix to `MandelRegion::examine()`'s reduction
- `binary-v2-viz` → commit `616c147` — region metrics + outlier dump + three-mode subdivision visualizer (report `08`); instrumentation only, decomposition-identical to `binary-v1-bugfix`
- `binary-v3-9point` → commit `ab47540` — 9-point stencil uniformity test (report `09`); on `main`
- `binary-v4-pq` → commit `37676d5` — largest-first shared priority queue (report `11`); **not on `main`** (negative result)
- `binary-v5-affinity` → commit `fc33e29` — min-max queue with GPU affinity + CPU-only guard (report `12`); on `main`.
- `binary-v6-periodicity` → commit `fa6d8ac` — exact Brent periodicity check in the CPU `diverge()` (report `21`); on `main`. Current best: hybrid 30.9 s, CPU12 35.2 s, byte-identical output.
- `binary-maxiter-split` → commit `d572c78` — parameter-free all-MAXITER split rule (report `18`); **not on `main`** (characterization, byte-identical decomposition to `diffT=0.1` on realistic content)

## Known bug: min/max tracking in `examine()`

`MandelRegion::examine()` in the original code uses a buggy reduction to compute the corner-iteration spread:

```cpp
if (minIter > cornersIter[i])
  minIter = cornersIter[i];
else if (maxIter < cornersIter[i])   // <-- bug: skipped whenever min updates
  maxIter = cornersIter[i];
```

The `else if` makes `maxIter` updates unreachable whenever the same sample also lowers `minIter`. On iteration 0, `minIter = INT_MAX` so the min branch always fires, meaning **the first corner's value can never set `maxIter`**. Roughly 25% of regions had their true maximum silently discarded; some of those then misclassified the region as "uniform" and skipped a split that should have happened.

Fix (lands at `binary-v1-bugfix`):

```cpp
if (cornersIter[i] < minIter) minIter = cornersIter[i];
if (cornersIter[i] > maxIter) maxIter = cornersIter[i];
```

Full analysis with the three failure-case walkthroughs is in `reports/03-bug-analysis.tex`.

## Architecture

Hybrid CPU+GPU Mandelbrot set renderer that uses adaptive region splitting as its load-balancing strategy.

**Execution model**: One `CalcThr` (QThread subclass) per logical worker. Thread 0 drives the GPU path; all others are CPU-only. All threads pull from a shared `WorkQueue` — a mutex-protected min-max `multiset<MandelRegion*>` ordered by pixel count. `extract(isGPU)` implements **GPU affinity** (report `12`): the GPU thread pulls the *largest* pending region (where it is ~30× faster on big coherent interior regions), CPU threads pull the *smallest*. A guard makes CPU threads pull largest (LPT) when no GPU is present. (Was a FIFO `deque` through `binary-v3-9point`.)

**Adaptive subdivision** (`MandelRegion::examine`): Each region evaluates its four corners first. If the corners are "uniform" (low iteration-count spread relative to `diffThresh`) or the region is smaller than `pixelSizeThresh`, it is computed immediately (`MandelRegion::compute`). Otherwise it is split into four equal quadrants and all four are pushed back onto the work queue.

**GPU path** (`kernel.cu`): `CUDAmemSetup` pre-allocates pitched device memory and pinned host memory once at startup. `hostFE` launches `mandelKernel` (16×16 thread blocks) and synchronously copies results back. GPU memory is reused across all regions processed by thread 0.

**Frame completion** (`MandelFrame`): Uses a `QAtomicInt remainingRegions` counter. `regionSplit` increments it by 3 (one region becomes four, net +3); `regionComplete` decrements it; when it hits zero the frame saves to disk.

**Key files**:
- `main.cpp` — argument parsing, frame/region setup, thread launch, end-to-end wall timer
- `mandelregion.cpp/.h` — adaptive subdivision logic and pixel computation (CPU branch)
- `kernel.cu` / `kernel.h` — CUDA kernel and host-side GPU interface
- `mandelframe.cpp/.h` — per-frame image buffer and atomic region counter
- `workqueue.cpp/.h` — thread-safe task queue

## Profiling instrumentation (already in the binary)

**NVTX annotations** (`mandelregion.cpp`, `kernel.cu`): `nvtxRangePush/Pop` brackets around `examine()`, the CPU compute path in `compute()`, and the full `hostFE()` call. These appear as labeled, colored spans on each thread's lane in the Nsight Systems timeline. NVTX3 (CUDA 11+) is header-only (`nvtx3/nvToolsExt.h`); linked via `-ldl` in the Makefile (no `-lnvToolsExt` needed).

**CUDA event timing** (`kernel.cu`): four `cudaEvent_t` handles (`evKernelStart/Stop`, `evMemcpyStart/Stop`) created once in `CUDAmemSetup`, reused across every kernel call. Each `hostFE` records the four timestamps and prints per-region lines plus a final aggregate summary to stderr.

**CPU thread timing** (`mandelregion.cpp`): `thread_local` counters (`cpuRegionCount`, `cpuTotalMs`) accumulate per-thread stats via `clock_gettime(CLOCK_MONOTONIC)`. `MandelRegion::printCPUSummary()` is called from `CalcThr::run()` when the work queue drains.

**End-to-end wall timer** (`main.cpp`): `clock_gettime` brackets the compute phase from `CUDAmemSetup` through every worker thread's `wait()`. Prints `[total_elapsed_s] X.XXXXXX` for machine-parseable extraction. `scripts/sweep_fig1115.sh` greps for this line.

**Quiet/save flags** (`main.cpp`): `profileQuiet` (positional arg 6) and `profileSave` (arg 7) gate per-region stderr prints and PNG writes respectively. Batch sweeps use `quiet=1, save=1`.

**Region work metrics + outlier dump** (`mandelregion.cpp`, added on `feat/viz-mode`): `compute()` accumulates total iterations and interior-pixel count per region on both the CPU and GPU branches (one add per pixel, reusing the existing `color==MAXGRAY` branch — negligible vs the `diverge()` loop), reporting mean iterations/pixel and interior fraction. Any CPU region exceeding `OUTLIER_MS` (5000 ms) emits an `[OUTLIER]` line — frame, depth, pixel + complex rectangles, corner iters, spread, work metrics, and `inMainCardioidOrBulb` membership — regardless of `quiet`. Regions also carry a `depth` field (root = 0, children = parent+1) and frames carry a `frameIndex`.

**Visualization mode** (`main.cpp`, `mandelframe.*`): the `viz` flag (arg 8) has three modes — `1` = single-frame depth animation (`generateDepthFrames`, frame selected by `vizFrame` arg 9), `2` = full-run partition overlay (`generateSequenceFrames`), `3` = full-run split-by-split process animation (`generateProcessFrames`). Every examined region registers a `VizRect` (rect, depth, leaf/split, executor) with its owner frame under a mutex; all three modes render after the timer stops via a shared `drawVizOverlay()` that strokes outlines 2 px inward (grey split skeleton, cyan GPU leaves, yellow CPU leaves). Verified performance-neutral and decomposition-identical vs `d5bf30c` (report `08`).

To profile (example):
```bash
mkdir -p experiments/initial-profile
cd experiments/initial-profile
nsys profile --trace=cuda,nvtx,osrt -o report ../../mandelHybrid ../../spec.in
nsys stats report.nsys-rep                # text summary
nsys-ui report.nsys-rep                   # GUI timeline
```

## Profiling results (100 frames, 1920×1080, i7-9750H + GTX 1660 Ti Max-Q, 12 threads)

### Pre-fix vs.\ post-fix headline numbers (default `diffT=0.5`)

| Metric | Pre-fix (`binary-v0-buggy`) | Post-fix (`binary-v1-bugfix`) | Δ |
|---|---|---|---|
| Total leaf regions | 4,603 | 5,152 | +12% |
| `examine()` calls | 6,104 | 6,836 | +12% |
| CPU avg per region | 535 ms | 398 ms | −26% |
| GPU avg per region | 13.5 ms | 12.8 ms | −5% |
| Worst CPU region | 15.3 s | 14.2 s | −7% |
| Worst GPU region | 466 ms | 139 ms | −70% |
| CPU thread wall spread | 50.5–52.6 s | 50.2–52.2 s | similar |
| GPU thread wall | 47.1 s | 47.8 s | +1% |
| End-to-end wall | ≈53 s | 53.2 s | flat |

**`cudaMemcpy` is dominated by kernel-wait time, not transfer**: the actual PCIe D→H transfer averages 15–24 µs, while `cudaMemcpy` reports 12–13 ms per call — 99% is the synchronous wait for the kernel. PCIe bandwidth is *not* a bottleneck; async streams would recover this latency.

**The worst-case region is `960×540` and takes ~15 s on a single CPU thread.** It survives the bug fix and the diffT sweep. Its four corners all sit inside the Mandelbrot set (iter = the frame's `maxIter`), so the corner-spread is identically zero and the uniformity rule passes at every `diffThresh`. The four-corner heuristic cannot subdivide this region without sampling additional interior points. Report `08` later pinned the two concrete instances (frames 73 and 89 at `diffT=0.1`) and showed via a cardioid/bulb membership test that they are **minibrot interiors, not main-cardioid/bulb regions** — so the cheap cardioid certificate cannot help; edge-midpoint sampling is required.

## Sensitivity studies

### Fig 11.15-style thread-configuration sweep

Seven configs (`CPU12, GPU, GPU+1CPU, …, GPU+11CPU`) × 3 reps at default `diffT=0.5, pixT=32768`, with PNG saves enabled. Runs about 25 minutes on this laptop.

```bash
# Auto-resolves project root; OUT defaults to experiments/sweep_results/
scripts/sweep_fig1115.sh
# Override:
REPS=3 OUT=experiments/05-fig1115-postfix SAVE=1 scripts/sweep_fig1115.sh
scripts/plot_fig1115.py experiments/05-fig1115-postfix/results.csv experiments/05-fig1115-postfix/fig1115.png
```

Results at `diffT=0.5` (means over 3 reps):

| Config | Pre-fix (s) | Post-fix (s) | Δ |
|---|---|---|---|
| CPU12 | 163.47 | 162.76 | −0.4% |
| GPU | 76.54 | 78.03 | +1.9% |
| GPU+11CPU (best hybrid) | 53.62 | 54.61 | +1.8% |

Speed-ups are preserved across the fix: best-hybrid/GPU-alone = 1.43× (both branches); best-hybrid/CPU12 ≈ 3× (both branches).

### `diffThreshold` sensitivity

Sweep across `diffT ∈ {0.05, 0.10, 0.20, 0.30, 0.50}` × 3 configs (`CPU12, GPU, GPU+11CPU`) × 3 reps. About 50 minutes per binary version.

Hybrid wall time at each `diffT` (means over 3 reps):

| `diffT` | Pre-fix (s) | Post-fix (s) | Δ |
|---|---|---|---|
| 0.05 | 53.24 | 53.45 | +0.4% |
| 0.10 | 52.85 | **52.80** | −0.1% |
| 0.20 | 53.19 | 52.90 | −0.5% |
| 0.30 | 53.30 | 53.21 | −0.2% |
| 0.50 | 53.62 | 54.61 | +1.8% |

**Both binaries agree the optimal `diffT` is 0.1**, with a hybrid wall of ~52.8 s — a 3.3% improvement over the legacy 0.5 default. Below 0.05 the pixel-size floor saturates the splitting; the leaf count tops out at 5,323 by `diffT = 0.1`. Run-to-run variance climbs at `diffT = 0.05`, so 0.1 is the predictable sweet spot.

**Why 0.1 is the sweet spot (report `18`):** at `diffT=0.1` the relative-spread test collapses to a parameter-free *interior certificate*. A parameter-free rule — "compute iff all 9 stencil samples reached MAXITER, else split" — produces a **byte-identical** decomposition to `diffT=0.1` on the production zoom (5608 leaves) and on inside/Misiurewicz/seahorse; it differs only on uniform-*exterior* content (`outside` +22% leaves, flat wall). So 0.1 is the operating point where the heuristic coincides with "compute the interior, split everything else." Higher `diffT` (0.5) admits near-uniform non-interior regions as leaves (the +1–2% penalty of reports 05/06); lower (0.05) saturates against the pixel floor.

### What the diffT sweep revealed about the bug fix

The fix's wall-time delta is strongly *config-dependent*: CPU-only is 0.4–1.2% faster post-fix (the 12 CPU workers absorb the extra leaves), GPU-only is 0.4–1.9% slower (no parallelism to absorb overhead), and the hybrid is within ±0.5% at every `diffT` except 0.5 (where the +549 extra leaves of the fix cost about 1 s).

### What no parameter sweep could fix

The 960×540 worst-case CPU region remains 14–15 s at every `diffT` on every binary version. It is the binding constraint on end-to-end wall time. Tuning cannot help; the heuristic itself must change. Report `08` characterised it fully: the binding instances (frames 73, 89) are minibrot interiors with zero corner spread, so the fix is the 9-point edge-midpoint stencil (next-step 2b), not the cardioid certificate (2a).

## Reports

Eighteen LaTeX reports live on `main` (numbered 01–09, 11–18, 21; there are no reports 10/19, and reports 20 and 22 live on the unmerged `feat/igpu-opencl` branch). Each report's title page pins the binary commit(s) that produced its measurements. Reports 01–13 are measurement reports; report 14 is a code/architecture guide and report 15 is an analysis/reconsideration (neither with `experiments/` data); reports 16–17 are kernel-level Nsight Compute measurements; report 18 is a split-rule decomposition characterization (16–18 each with `experiments/` data); report 21 is the periodicity-check measurement (experiment `21`).

| File | Binary commit(s) | Subject |
|---|---|---|
| `reports/01-initial-profile.tex` | `0f782fd` (v0) | Original NVTX/CUDA-event instrumentation pass and pre-fix profile |
| `reports/02-fig1115-replication.tex` | `0f782fd` (v0) | Replication of Barlas Fig 11.15 (thread-configuration sweep) on this laptop |
| `reports/03-bug-analysis.tex` | `0f782fd` (the bug) + `d5bf30c` (the fix) | The min/max tracking bug, three failure walkthroughs, the one-line fix |
| `reports/04-postfix-profile.tex` | `d5bf30c` (v1) | Re-profile of the post-fix binary, vs.\ pre-fix headline numbers |
| `reports/05-fig1115-postfix.tex` | `d5bf30c` (v1) | Repeated Fig 11.15 sweep on the post-fix binary |
| `reports/06-difft-sweep.tex` | `d5bf30c` (v1) | `diffThreshold` sensitivity on the post-fix binary, sweet spot identified |
| `reports/07-difft-compare.tex` | `0f782fd` + `d5bf30c` | Pre-fix vs.\ post-fix diffT sweep comparison (sanity check + load-balance analysis) |
| `reports/08-region-metrics-viz.tex` | `feat/viz-mode` atop `d5bf30c` | Region work metrics + subdivision visualizer; locates the two binding outliers (frames 73, 89) and shows via the new cardioid/bulb test that they are minibrot interiors (not main cardioid/bulb); performance-neutral A/B vs `d5bf30c` |
| `reports/09-9point-sampling.tex` | `examine/9-points-sampling` atop `binary-v2-viz` | 9-point stencil vs 4-corner sampler; catches the frame-89 outlier (84% interior) but not frame 73 (97% interior); +5.4% leaves; wall flat (+0.6%, throughput-bound) |
| `reports/11-priority-queue.tex` | `feat/priority-queue` / `binary-v4-pq` atop `binary-v3-9point` | Largest-first priority queue vs FIFO; **wall-neutral negative result** (−0.55%); outlier not routed to GPU (shared queue, 11:1 CPU:GPU extraction). §3.4 shows the outlier is GPU-friendly (coherent, ~30×) → the real lever is GPU **affinity**, not a shared queue. Code NOT merged to main (FIFO kept). |
| `reports/12-gpu-affinity.tex` | `feat/gpu-affinity` / `binary-v5-affinity` atop `binary-v3-9point` | Min-max work queue, GPU thread pops largest / CPU smallest. **First wall win: −3.6%** (51.73→49.86 s, disjoint A/B); routes the 960×540 outlier to the GPU (460 ms vs 13.3 s on CPU), CPU outliers 1→0. Bounded by GPU near-saturation (~91%). CPU-only guard (LPT) measured a wash. Merged to main. |
| `reports/13-zoom-points.tex` | `binary-v5-affinity` (`fc33e29`) | Load-balance characterization across **four zoom regimes** (outside / inside / Misiurewicz / seahorse), same binary, only the deep target differs. Cost spans **32×** (2.37–76.3 s) yet CPU thread spread stays **<1.4% in every regime** — dynamic balance is not the bottleneck; the binding resource shifts CPU-pool→GPU (**95–96% saturation** on heavy frames, same ceiling as report 12). Seahorse reproduces the interior outlier **8×**, all `cardioidBulb=1` main-cardioid interiors overflowing the saturated GPU — the **first measured beneficiary of the cardioid certificate** (canonical minibrot is `cardioidBulb=0`). Sets the baseline for the static-LB study. |
| `reports/14-architecture-guide.tex` | `binary-v5-affinity` (`fc33e29`) | **Code/architecture guide** (not a measurement). File-by-file walkthrough of the whole source with real code excerpts from every translation unit (`main.cpp`, `mandelframe`, `mandelregion`, `workqueue`, `kernel.cu`) plus the Makefile/instrumentation/viz layers — an in-depth reference for reading and extending the code. Explains how the three local decisions (task size in `examine`, task owner in `WorkQueue::extract`, completion in the atomic counter) compose into the measured behaviour. Companion to `INSTRUMENTATION.md`; attributes all numbers to reports 01–13. |
| `reports/15-async-streams-analysis.tex` | `binary-v5-affinity` (`fc33e29`) | **Async-streams reconsideration** (analysis, no new measurements). Re-derives the cost of async CUDA streams from reports 01/09/12 + one code fact (no `cudaSetDeviceFlags`). The `cudaMemcpy` "99% of API time" is **99.8% kernel-wait**, only ~22 µs real D→H transfer — so streams have ~0.18% of wall in transfer to hide. The only other prize, the GPU-driver thread's stall, is a **busy-spin** (`ScheduleAuto`, 1 ctx < 12 cores), worth ≤1 median CPU region (8.7 ms) → **~0.4 of a physical core** after 2-way SMT, past the knee on the 6-core part, while the GPU is ~91% saturated (gap <9%). **Three ceilings bound the lever to sub-1% of wall**; retires the "largest unexplored optimisation" framing of reports 01/04/05/06. Includes a one-line `cudaDeviceScheduleBlockingSync` A/B to confirm cheaply. |
| `reports/16-ncu-divergence.tex` | `binary-v5-affinity` (`fc33e29`) vs Nsight Compute 2026.2 | **First kernel-level `ncu` measurement** — converts report 11's *structural* divergence argument to *measurement*. GPU-only runs across four content regimes (interior/exterior/boundary/canonical) so the GPU sees every region type. **Coherent regions = exactly 100.0% warp execution efficiency + 100.0% branch efficiency** (zero divergence, not "small"), and **FP64-bound at 85.6% of peak** (~110/129 GFLOPS @1.34 GHz, 1:32 TU116), 98.9% occupancy, ~0% memory; the 15% gap is the dependent iteration recurrence (`long_scoreboard` ~92–95% of issue stalls, **uniform across regimes** → intrinsic, not divergence). Divergence is real but **confined to 240×135 pixel-floor boundary leaves** (down to 33.7%; 480×270 leaves 99.9%) — the uniformity test doubles as a coherence filter and affinity routes those cheap floor regions to the CPU. **Corrects report 11's ~40% FP64 estimate → 85.6%.** Includes the nsys-vs-ncu altitude table. ncu counters are admin-gated (`ERR_NVGPUCTRPERM`); capture via `sudo experiments/16-ncu-divergence/capture.sh`, parse via `ncu --import` (no sudo). |
| `reports/17-ncu-zoom-points.tex` | `binary-v5-affinity` (`fc33e29`) vs ncu 2026.2 | **Verifies report 16 on the four report-13 zoom points** (outside/inside/Misiurewicz/seahorse) instead of synthetic specs; adds **Misiurewicz**. Same `ncu` GPU-only method (24-frame specs — warp/FP64 are rates, frame-count-independent). Reproduces exactly: **≥480×270 leaves 99.7–100% warp efficiency, 0% below 70%** in every regime; **FP64-bound at 85.6%** (~110/129 GFLOPS) on the inside point; divergence (down to **38%**) **confined to the 240×135 floor leaves**, near-identically across four geometries (mean ~89%, 8–14% <70%). Divergence tracks region *size*, not content — structural. Reconciles report 16's heavier aggregate (its boundary spec was engineered boundary-heavy; here floor leaves are a minority). |
| `reports/18-maxiter-split.tex` | `examine/maxiter-split` atop `binary-v5-affinity` (`fc33e29`); **NOT merged** | **Split-rule characterization.** Replaces the `diffThresh` relative-spread test with a parameter-free binary rule — compute iff all 9 stencil samples reached MAXITER (`minIter == frame MAXITER`), else split. **Byte-identical decomposition** to `diffT=0.1` on the production zoom (**5608 leaves both**) and on inside/Misiurewicz/seahorse; differs only on **outside** (+22% leaves — uniform exterior the relative test keeps whole), at **flat wall**. So `diffT=0.1` *is* a parameter-free interior certificate on realistic content — explains why 0.1 is the sweet spot; the relative test's only distinctive job is large uniform exterior (cost-free here). Branch kept off main (characterization, like report 11). |
| `reports/21-periodicity-check.tex` | `feat/periodicity-check` / `binary-v6-periodicity` (`fa6d8ac`) atop `main` `3061307` | **Exact Brent periodicity check in the CPU `diverge()`** — the first work-reduction lever. Motivated by the leaf-purity census (99.5% of the 495 G iterations are interior pixels grinding to MAXITER). **CPU12 −78.6% (164.4→35.2 s), hybrid −46.3% (57.6→30.9 s, new best wall)**; output byte-identical (100/100 mode-0 frames), decomposition unchanged (5,608 leaves). Hybrid self-rebalances (GPU 3,448→1,329 regions, keeps the largest). Detection latency ~900 iters-equivalent in the deep minibrot vs 67 in a period-3 bulb. Merged; tag `binary-v6-periodicity`. |

Build any report with:
```bash
scripts/build_report.sh 03-bug-analysis      # one report
scripts/build_report.sh all                  # every report
```

## Adding a new report

1. `cp reports/01-initial-profile.tex reports/NN-name.tex` (use the next available number).
2. Set `\documenttitle`, `\documentsubtitle`, `\documentsubject` in the preamble.
3. Update the centered binary-version stamp right after `\inserttitle` to point at the commit(s) whose binary produced your measurements. Reference commits by their short hash and the matching tag (`binary-v0-buggy` / `binary-v1-bugfix`); add new tags if the report describes a new optimization branch.
4. If the report has associated raw data, create `experiments/NN-name/` with a `README.md` recording the commit, command, and date used to generate the data.
5. Build with `scripts/build_report.sh NN-name`.

## Profiling artefacts

Raw measurement data, by experiment number. Per-rep `.stderr` and `.stdout` files inside each `logs/` directory are tracked; large binaries (`*.nsys-rep`, `*.sqlite`) and figures (`*.png`) are gitignored and regenerated on demand.

| Experiment | Binary | Contents |
|---|---|---|
| `experiments/01-initial-profile/` | `binary-v0-buggy` | `report.nsys-rep`, `report.sqlite` |
| `experiments/02-fig1115-prefix/` | `binary-v0-buggy` | `results.csv` + 21 stderr logs |
| `experiments/04-postfix-profile/` | `binary-v1-bugfix` | `report.nsys-rep`, `report.sqlite`, raw stderr/stdout |
| `experiments/05-fig1115-postfix/` | `binary-v1-bugfix` | `results.csv` + 21 stderr logs + `fig1115.png` |
| `experiments/06-difft-postfix/` | `binary-v1-bugfix` | `results.csv` + 36 stderr logs + `quiet0/` max-region logs + `difft_sweep.png` |
| `experiments/07-difft-prefix/` | `binary-v0-buggy` | `results.csv` + 36 stderr logs + `quiet0/` max-region logs + `compare.png` |
| `experiments/08-region-metrics-viz/` | `feat/viz-mode` atop `d5bf30c` | `viz/` depth frames + montage/GIF, `logs/` (metrics + outlier dumps), `perf/results.csv` (A/B vs `d5bf30c`) |
| `experiments/09-9point-sampling/` | `examine/9-points-sampling` atop `binary-v2-viz` | `logs/` (9pt + 4pt metrics + quiet0 distributions), `nsys/` (NVTX/CUDA stats), `perf/results.csv` (A/B), `viz_process/` (9pt viz=3 anim), `viz/` (frame-89 figure) |
| `experiments/10-difft-9point/` | `binary-v3-9point` | `results.csv` + 15 stderr logs — `diffThreshold` sweep confirming 0.1 is still optimal for the 9-point binary (report `09`, §Re-Tuning Check) |
| `experiments/11-priority-queue/` | `binary-v4-pq` vs `binary-v3-9point` | `logs/` (metrics + quiet0 dist), `nsys/` (NVTX/CUDA stats), `perf/results.csv` (A/B) — FIFO vs largest-first priority queue (report `11`); includes the AC-power contamination note |
| `experiments/12-gpu-affinity/` | `binary-v5-affinity` vs `binary-v3-9point` | `logs/`, `nsys/`, `perf/results.csv` (hybrid A/B, −3.6%), `cpu_only/` (CPU-only 3-way: FIFO vs smallest-first vs guarded LPT) — report `12` |
| `experiments/13-zoom-points/` | `binary-v5-affinity` (`fc33e29`) | `spec_{outside,inside,misiurewicz,seahorse}.in`, `run.sh`, `README.md`, `logs/` (`.timing`/`.viz` `.stderr` per point) — four-regime characterization (report `13`). `viz_<pt>/` PNG frames and `anim_<pt>.mp4` animations gitignored, regenerated via `run.sh viz` + ffmpeg |
| `experiments/16-ncu-divergence/` | `binary-v5-affinity` (`fc33e29`) vs ncu 2026.2 | `capture.sh` (the **sudo** capture script — counters are admin-gated), `specs/{interior,exterior,boundary,canonical}.in` (GPU-only, 1920×1080), `analysis/summary.csv`, `ncu/gpu_state.csv`, `README.md` — report `16`. `ncu/*.ncu-rep` (large) + `analysis/*_full.csv`/`*_warp.csv` (wide raw dumps) gitignored, regenerated via `sudo capture.sh` + `ncu --import`. Capture is GPU-only so the GPU thread sees every region type (in production, affinity routes boundary regions to the CPU). |
| `experiments/17-ncu-zoom-points/` | `binary-v5-affinity` (`fc33e29`) vs ncu 2026.2 | `capture.sh` (sudo), `specs/{outside,inside,misiurewicz,seahorse}.in` (the four report-13 points, 24-frame), `analysis/summary.csv`, `ncu/gpu_state.csv`, `README.md` — report `17` (verifies report 16 on the report-13 geometries). `ncu/*.ncu-rep` + wide `analysis/*.csv` gitignored. |
| `experiments/18-maxiter-split/` | `examine/maxiter-split` atop `binary-v5-affinity` (`fc33e29`) | `results.csv` (GPU-only leaf counts: all-MAXITER vs diffT=0.1, 5 specs), `README.md` — report `18`. Decomposition is deterministic; no large data. |
| `experiments/21-periodicity-check/` | `binary-v6-periodicity` (`fa6d8ac`) vs `main` (`3061307`) | `ab.sh`, `results.csv` (2 configs × 2 binaries × 3 reps, save=1), `logs/` (per-run summaries + `dist_pc` quiet=0 per-region capture), `verify_identity.sh` (100/100 byte-identical), `plot_periodicity.py` (→ `reports/img/periodicity_*.png`, gitignored) — report `21` |

## Conclusions

1. The bug fix is a correctness improvement. It exposes additional splits (~12% more leaves at `diffT=0.5`, ~9% at `diffT=0.1`) that the buggy reduction was silently skipping.
2. The bug fix does not improve end-to-end wall time at the legacy default. It adds 1–2% across hybrid configs because the new splits' overhead exceeds the load-balance gain — *at `diffT=0.5`*.
3. Lowering `diffThreshold` from 0.5 to 0.1 recovers the difference and yields the best hybrid time on either binary version (~52.8 s, −3.3% vs.\ the 0.5 baseline). At this tighter threshold the bug fix becomes essentially performance-neutral.
4. The binding constraint on the wall time is a single ~15 s CPU region whose four corners all lie inside the Mandelbrot set, so corner-spread is zero and no `diffThreshold` value can subdivide it. The four-corner heuristic has reached its ceiling on this workload.
5. Work-reduction beats scheduling by an order of magnitude (report `21`, `binary-v6-periodicity`): 99.5% of all iterations were interior pixels grinding to MAXITER; an exact Brent periodicity check in the CPU `diverge()` eliminates most of that — hybrid wall 57.6→30.9 s (−46.3%), CPU12 164.4→35.2 s (−78.6%), byte-identical output. The CPU pool remains the binding resource (29.5 s/thread ≈ wall) but at a quarter of its former cost; what remains on it is the truly-escaping boundary band plus detection latency (~900 iters-equivalent in the deep minibrot).

## Recommended next steps

In priority order:

1. **Locate the worst region.** ✅ **Done — see `reports/08-region-metrics-viz.tex`.** The `[OUTLIER]` dump in `MandelRegion::compute()` (fires above 5000 ms) found exactly two binding outliers at `diffT=0.1`: frames **73** (13.6 s) and **89** (14.3 s), both `960×540` depth-1 upper-left quadrants whose four corners all sit at the frame's `maxIter` (spread 0). The new cardioid/bulb membership test reports **`cardioidBulb=0`** for both, and the interior-pixel fraction is 84–97%.
2a. ~~**If the outlier is inside the main cardioid or the period-2 bulb**~~ **Ruled out by report 08.** The membership test (`q = (x-1/4)^2 + y^2; q*(q + (x-1/4)) <= y^2/4` for the cardioid; `(x+1)^2 + y^2 <= 1/16` for the bulb) returns 0 for both outliers — they are interior to a **minibrot**, not the main cardioid/bulb. A cardioid/bulb certificate would not fire on them, so it cannot relieve the 14 s tail. (It remains a cheap win for the early wide frames whose pixels genuinely fall in the main cardioid, but that is not the binding constraint.)
2b. ✅ **Implemented and deep-profiled (nsys + quiet=0) — see `reports/09-9point-sampling.tex` (branch `examine/9-points-sampling`).** The 9-point stencil (4 corners + 4 edge midpoints + centre) is a **partial** fix: it subdivides the frame-89 outlier (84% interior — an interior sample lands off the minibrot) but **not** frame 73 (97% interior — all 9 points still hit `maxIter`, spread 0). It exposes +5.4% leaves and shifts the CPU per-region distribution down (median 126→99 ms, max 15.5→13.4 s). **Wall is flat** (+0.6%, noise). The nsys breakdown corrects an earlier guess: the sampling overhead is **negligible** (NVTX `examine − compute` = +0.40 s CPU total, +29 µs/`examine`), and splitting **conserves total compute** (~595 s in both binaries) — it redistributes work but removes none. With load already balanced <2%, finer splitting can't help at 12 threads. **The lever is work-reduction, not subdivision:** periodicity checking in `diverge()` (exact) or a heuristic interior certificate (risks filling over thin filaments). The exact cardioid/bulb certificate doesn't apply (minibrot interior).
3. ✅ **GPU affinity (min-max work queue) — DONE and merged, see `reports/12-gpu-affinity.tex` (`binary-v5-affinity`).** The GPU thread pops the largest region, CPU threads the smallest, so the big coherent interior regions (~30× faster on the GPU) run on the GPU. **First lever to move the wall: −3.6%** (51.73→49.86 s, disjoint A/B), routes the 960×540 outlier to the GPU (460 ms vs 13.3 s on CPU), CPU outliers 1→0. CPU-only LPT guard kept (measured a wash). This **supersedes** the shared priority queue (step 4 below). Bounded at −3.6% because the GPU is now ~91% utilized.
4. ~~**Largest-first priority queue**~~ ❌ **Tried — wall-neutral (report `11`, `binary-v4-pq`, NOT merged).** A *shared* largest-first `priority_queue` can't route big regions to the GPU (11:1 CPU:GPU extraction → outlier stays on CPU). Superseded by GPU affinity (step 3), which makes extraction executor-aware.
5. ✅ **Work-reduction (periodicity checking) — DONE and merged, see `reports/21-periodicity-check.tex` (`binary-v6-periodicity`).** Exact Brent cycle detection in the CPU `diverge()`: **CPU12 −78.6% (164.4→35.2 s), hybrid −46.3% (57.6→30.9 s, new best)**, byte-identical output, decomposition unchanged. Landed CPU-branch-only per report 16's caveat; a GPU-side variant is a separate A/B (expected loss). It re-opens: re-run experiment 20 atop v6 (the iGPU's −23.2% was priced against the now-gone interior grind) and re-price the experiment-19 threshold sweep. The heuristic interior certificate (constant-fill) is retired — the leaf census showed the certificate lies on 43/260 certified regions, and the exact detector already captures the prize.
6. **Async CUDA streams** in `hostFE()` (lower priority): overlap the kernel with the previous region's host work via a cross-region pipeline (deferred completion / double buffering). ❌ **Priced out — see `reports/15-async-streams-analysis.tex`.** Sub-1% of wall: transfer to hide is ~0.18% (the `cudaMemcpy` "99%" is 99.8% kernel-wait, ~22 µs real D→H), the GPU-driver stall is a busy-spun lane worth ~0.4 of a physical core past the SMT knee, and the GPU is ~91% saturated (gap <9%). Needs a non-trivial restructure of the GPU thread's region lifetime for that return; ranked below work-reduction. Cheap confirmation first: one-line `cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)` A/B.
7. **`pixelSizeThresh` sweep at `diffT=0.1`** (low priority): the decision rule is `diff_uniform OR below_pixT`, so `pixT` cannot force-split the 15 s outlier (which passes the diff test). It can tighten load balance among floor-classified regions. Quick to test, but report 12 showed CPU-only queue ordering is a wash, so expect little.

Each new optimization should land on its own feature branch (`feat/cardioid-certificate`, `feat/edge-midpoint`, `feat/async-streams`, `feat/priority-queue`) with a new tag pinning the binary version, a new `reports/NN-…tex` describing the measurements, and `experiments/NN-…/` holding the raw data. Merge to `main` when validated.

The default `diffThreshold` should change from 0.5 to 0.1 regardless of which optimization is pursued next.
