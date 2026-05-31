# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
make          # build mandelHybrid binary
make clean    # remove all object files and the binary
```

Dependencies: `nvcc` (CUDA), `g++`, Qt5 (`Qt5Core Qt5Gui` via pkg-config), CUDA installed at `/opt/cuda`. The Makefile pins `nvcc -ccbin g++-15` because `nvcc` 13.2 does not yet support gcc 16; the host compile uses the same `g++-15` so ABI is consistent across `nvcc`/`g++` translation units.

CUDA architecture is set to `-arch=native` (requires CUDA 11.6+). For older `nvcc`, change `CUDA_ARCH` in the Makefile to a specific arch like `-arch=sm_61`.

## Run

```bash
./mandelHybrid spec.in <numThreads> <gpuEnable> [diffThreshold] [pixelThreshold] [quiet] [save] [viz] [vizFrame]
```

- `numThreads`: total CPU+GPU worker threads (defaults to CPU core count)
- `gpuEnable`: `1` enables GPU (default), `0` CPU-only
- `diffThreshold`: fraction (default 0.5; **recommended new default: 0.1** — see Sensitivity studies). Region is uniform enough to compute if `(maxCornerIter - minCornerIter) < diffThresh * maxCornerIter`
- `pixelThreshold`: pixel count (default 32768). Regions smaller than this are always computed without further splitting
- `quiet`: `1` suppresses per-region stderr prints; per-thread and per-GPU summaries still emit
- `save`: `0` skips PNG writes (for pure-compute timing)
- `viz`: `1` enables visualization mode — renders one frozen interpolated frame (interpolation disabled) and emits the depth-by-depth subdivision animation `<prefix>_fNNNN_dKK.png`, one PNG per recursion depth `K`, with region outlines coloured by executor (cyan = GPU thread, yellow = CPU thread, grey = split skeleton). Default `0`. The animation is rendered after the wall-clock timer stops, so it never perturbs timing.
- `vizFrame`: in `viz` mode, which interpolated frame index to freeze on (default `0`). Frame 0 is the wide full-set view (`maxIter=100`, cheap, no GPU participation); late frames are deep zooms with high `maxIter` where the GPU participates and the outliers live (e.g. `89`).

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
- `feat/viz-mode` — adds region work metrics, the 5 s outlier dump, and the `vizMode` subdivision animation (report `08`). Instrumentation only; the subdivision decision is unchanged (verified: byte-identical 5,323-leaf decomposition vs `d5bf30c`). Not yet merged/tagged; intended tag on merge is `binary-v2-viz`.
- `examine_minmax_bugfix` — the commit that fixed the min/max reduction in `MandelRegion::examine()`. Same code state as `main` modulo subsequent reorganization commits.
- `examine_rewrite`, `master` — original code with the min/max tracking bug intact. Kept as historical references.

Tags pin the two binary versions every report ultimately points at:

- `binary-v0-buggy` → commit `0f782fd` — original code with the min/max tracking bug
- `binary-v1-bugfix` → commit `d5bf30c` — same code with the one-line fix to `MandelRegion::examine()`'s reduction

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

**Execution model**: One `CalcThr` (QThread subclass) per logical worker. Thread 0 drives the GPU path; all others are CPU-only. All threads pull from a shared `WorkQueue` (mutex-protected `deque<MandelRegion*>`).

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

**Visualization mode** (`main.cpp` `generateDepthFrames`, `mandelframe.*`): `viz=1` (arg 8) freezes one interpolated frame (`vizFrame`, arg 9) and emits a depth-by-depth subdivision animation `<prefix>_fNNNN_dKK.png`. Every examined region registers a `VizRect` (rect, depth, leaf/split, executor) with its owner frame under a mutex; after the timer stops, each depth `K` is rendered as the finished image overlaid with all outlines of depth `≤ K` — grey for split nodes, cyan for GPU leaves, yellow for CPU leaves. Verified performance-neutral and decomposition-identical vs `d5bf30c` (report `08`).

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

### What the diffT sweep revealed about the bug fix

The fix's wall-time delta is strongly *config-dependent*: CPU-only is 0.4–1.2% faster post-fix (the 12 CPU workers absorb the extra leaves), GPU-only is 0.4–1.9% slower (no parallelism to absorb overhead), and the hybrid is within ±0.5% at every `diffT` except 0.5 (where the +549 extra leaves of the fix cost about 1 s).

### What no parameter sweep could fix

The 960×540 worst-case CPU region remains 14–15 s at every `diffT` on every binary version. It is the binding constraint on end-to-end wall time. Tuning cannot help; the heuristic itself must change. Report `08` characterised it fully: the binding instances (frames 73, 89) are minibrot interiors with zero corner spread, so the fix is the 9-point edge-midpoint stencil (next-step 2b), not the cardioid certificate (2a).

## Reports

Seven LaTeX reports document the work chronologically. Each report's title page pins the binary commit(s) that produced its measurements.

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

Build any report with:
```bash
scripts/build_report.sh 03-bug-analysis      # one report
scripts/build_report.sh all                  # all seven
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

## Conclusions

1. The bug fix is a correctness improvement. It exposes additional splits (~12% more leaves at `diffT=0.5`, ~9% at `diffT=0.1`) that the buggy reduction was silently skipping.
2. The bug fix does not improve end-to-end wall time at the legacy default. It adds 1–2% across hybrid configs because the new splits' overhead exceeds the load-balance gain — *at `diffT=0.5`*.
3. Lowering `diffThreshold` from 0.5 to 0.1 recovers the difference and yields the best hybrid time on either binary version (~52.8 s, −3.3% vs.\ the 0.5 baseline). At this tighter threshold the bug fix becomes essentially performance-neutral.
4. The binding constraint on the wall time is a single ~15 s CPU region whose four corners all lie inside the Mandelbrot set, so corner-spread is zero and no `diffThreshold` value can subdivide it. The four-corner heuristic has reached its ceiling on this workload.

## Recommended next steps

In priority order:

1. **Locate the worst region.** ✅ **Done — see `reports/08-region-metrics-viz.tex`.** The `[OUTLIER]` dump in `MandelRegion::compute()` (fires above 5000 ms) found exactly two binding outliers at `diffT=0.1`: frames **73** (13.6 s) and **89** (14.3 s), both `960×540` depth-1 upper-left quadrants whose four corners all sit at the frame's `maxIter` (spread 0). The new cardioid/bulb membership test reports **`cardioidBulb=0`** for both, and the interior-pixel fraction is 84–97%.
2a. ~~**If the outlier is inside the main cardioid or the period-2 bulb**~~ **Ruled out by report 08.** The membership test (`q = (x-1/4)^2 + y^2; q*(q + (x-1/4)) <= y^2/4` for the cardioid; `(x+1)^2 + y^2 <= 1/16` for the bulb) returns 0 for both outliers — they are interior to a **minibrot**, not the main cardioid/bulb. A cardioid/bulb certificate would not fire on them, so it cannot relieve the 14 s tail. (It remains a cheap win for the early wide frames whose pixels genuinely fall in the main cardioid, but that is not the binding constraint.)
2b. ✅ **This is the path.** Add edge-midpoint sampling (9-point stencil), inheriting the new samples as children's corners on split. Report 08 shows the outliers' mean iteration count (7,131 / 7,551) is measurably below their corner value (7,327 / 8,911), proving sub-`maxIter` structure exists between the corners that the four-point sampler misses — so a 9-point stencil would expose a non-zero spread and trigger the split the corners suppress.
3. **Async CUDA streams** in `hostFE()`: launch the kernel into a non-default stream and use `cudaStreamSynchronize` only when reading results. Recovers up to ~13 ms/region of GPU-thread idle time spent in `cudaMemcpy`. Independent of any heuristic change.
4. **Largest-first priority queue**: replace `WorkQueue`'s `deque` with `priority_queue` using the existing `MandelRegion::Compare` functor. Routes larger regions to the GPU thread first. Modest expected upside (~1 s).
5. **`pixelSizeThresh` sweep at `diffT=0.1`**: the decision rule is `diff_uniform OR below_pixT`, so `pixT` cannot force-split the 15 s outlier (which passes the diff test). It can tighten load balance among floor-classified regions. Quick to test.

Each new optimization should land on its own feature branch (`feat/cardioid-certificate`, `feat/edge-midpoint`, `feat/async-streams`, `feat/priority-queue`) with a new tag pinning the binary version, a new `reports/NN-…tex` describing the measurements, and `experiments/NN-…/` holding the raw data. Merge to `main` when validated.

The default `diffThreshold` should change from 0.5 to 0.1 regardless of which optimization is pursued next.
