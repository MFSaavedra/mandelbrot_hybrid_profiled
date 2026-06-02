# mandelHybrid — Hybrid CPU+GPU Mandelbrot Renderer

A 100-frame Mandelbrot zoom renderer that splits work between CPU threads and
a single GPU thread, using adaptive region subdivision as its load-balancing
strategy. Built on the example from Section 11.5.1 of Barlas,
*Multicore and GPU Programming* (2nd ed., 2022).

This repository extends the original code with NVTX + CUDA-event
instrumentation and end-to-end wall timing, a one-line bug fix to
`MandelRegion::examine()`'s min/max reduction, a **9-point sampling stencil**,
a **GPU-affinity work queue**, a three-mode **subdivision visualizer**, batch
sweep drivers, and twelve LaTeX reports covering everything that has been
measured.

**Current best:** `main` = tag `binary-v5-affinity` (9-point stencil + GPU
affinity). Hybrid wall ≈ **49.9 s** (100 frames, 1920×1080, i7-9750H +
GTX 1660 Ti Max-Q, 12 threads, `diffT=0.1`), down from the FIFO baseline's
51.7 s and the legacy ~53 s.

## Quick start

```bash
make                                       # build mandelHybrid
./mandelHybrid spec.in                     # render with default settings
./mandelHybrid spec.in 12 1 0.1            # 12 threads, GPU on, diffThreshold 0.1
./mandelHybrid spec.in 12 1 0.1 32768 1 0 2  # viz=2: overlay each frame's partition
scripts/sweep_fig1115.sh                   # thread-configuration sweep (Fig 11.15)
scripts/build_report.sh all                # build every report PDF
```

Run signature: `./mandelHybrid spec.in [numThr] [gpuEnable] [diffT] [pixT] [quiet] [save] [viz] [vizFrame]`.
`viz` is `0`/`1`/`2`/`3` (single-frame depth animation / full-run partition
overlay / full-run split-by-split process), coloured by executor
(cyan = GPU leaf, yellow = CPU leaf, grey = split skeleton). See `CLAUDE.md`
for the full argument reference.

Dependencies: `nvcc` (CUDA 11.6+), `g++-15`, Qt5 (`Qt5Core` + `Qt5Gui` via
`pkg-config`). The Makefile pins `nvcc -ccbin g++-15` because `nvcc` 13.2
does not yet support gcc 16.

## Repository layout

```
.
├── *.cpp, *.h, *.cu, Makefile, spec.in   source and build
├── reports/                              numbered LaTeX reports + shared template
│   └── NN-name.tex
├── experiments/                          raw measurement data per report
│   └── NN-name/
├── scripts/
│   ├── sweep_fig1115.sh                  Fig 11.15 thread-configuration sweep
│   ├── plot_fig1115.py                   plot the sweep CSV
│   └── build_report.sh                   pdflatex×2 for any (or all) reports
├── INSTRUMENTATION.md                    deep-dive on the NVTX / CUDA hooks
└── CLAUDE.md                             full project reference for Claude Code agents
```

## Reports

Each report's title page lists the commit hash and tag of the binary that
produced its measurements. (There is no report 10; the `diffThreshold` re-tuning
for the 9-point binary is documented in report 09 §Re-Tuning Check.)

| # | Name | Binary | Subject |
|---|---|---|---|
| 01 | initial-profile | `binary-v0-buggy` | Original NVTX / CUDA-event profile |
| 02 | fig1115-replication | `binary-v0-buggy` | Fig 11.15 thread-configuration sweep |
| 03 | bug-analysis | `v0` + `v1-bugfix` | The min/max tracking bug and the one-line fix |
| 04 | postfix-profile | `binary-v1-bugfix` | Re-profile after the bug fix |
| 05 | fig1115-postfix | `binary-v1-bugfix` | Fig 11.15 sweep on the post-fix binary |
| 06 | difft-sweep | `binary-v1-bugfix` | `diffThreshold` sensitivity sweep (0.1 is best) |
| 07 | difft-compare | `v0` + `v1` | Pre-fix vs. post-fix `diffT` comparison |
| 08 | region-metrics-viz | `binary-v2-viz` | Region work metrics + subdivision visualizer; locates the outliers |
| 09 | 9point-sampling | `binary-v3-9point` | 9-point stencil vs 4-corner; catches one of two outliers; wall-neutral |
| 11 | priority-queue | `binary-v4-pq` | Shared largest-first queue — **negative result** (not merged) |
| 12 | gpu-affinity | `binary-v5-affinity` | Min-max queue, GPU pops largest — **first wall win, −3.6%** |
| 13 | zoom-points | `binary-v5-affinity` | Load-balance characterization across four zoom regimes |

PDFs build under `reports/NN-name.pdf` via `scripts/build_report.sh`.

## Key findings

| Finding | Numbers |
|---|---|
| **`diffThreshold` should be 0.1, not 0.5** | Flat optimum over 0.05–0.20; leaf count saturates by 0.1. Re-verified for the 9-point binary (report 09). |
| **The binding outlier is a minibrot interior** | 960×540 depth-1 regions (frames 73/89) whose four corners all reach `maxIter` (spread 0) → never split; ~13–15 s on one CPU thread. `cardioidBulb=0`, so the cardioid certificate cannot help. |
| **9-point stencil is a partial fix** | Subdivides frame 89 (84 % interior) but not frame 73 (97 %); +5.4 % leaves; wall-neutral. Sampling overhead measured negligible (+0.40 s); splitting **conserves** total compute (~595 s). |
| **GPU affinity is the first lever to move the wall** | Min-max queue (GPU pops largest, CPU smallest): −3.6 % (51.73 → 49.86 s, disjoint A/B). Routes the outlier to the GPU (460 ms vs 13.3 s on CPU); CPU outliers 1 → 0. Bounded by GPU ~91 % saturation. |
| **A *shared* largest-first queue does not help** | Wall-neutral (−0.55 %): 11 CPU threads vs 1 GPU win the big regions 11:1, so the outlier stays on CPU. Negative result, not merged. |
| **`cudaMemcpy` time is mostly kernel-wait, not transfer** | ~99 % kernel-wait; the actual D → H transfer is only ~95 ms total. |
| **Cost varies 32×, but dynamic balance never binds** | Across four zoom regimes (report 13) the CPU thread spread stays <1.4 %; the binding resource is GPU saturation (95–96 %), not load imbalance. |

## Branches and tags

- `main` — current best (9-point stencil + GPU affinity); holds all reports and data.
- `feat/gpu-affinity` — **merged** (`binary-v5-affinity`): min-max GPU-affinity queue + CPU-only guard (report 12).
- `feat/priority-queue` — **not merged** (`binary-v4-pq`): shared priority queue, negative result (report 11).
- `examine/9-points-sampling`, `feat/viz-mode` — merged (`binary-v3-9point`, `binary-v2-viz`).
- `examine_minmax_bugfix`, `examine_rewrite`, `master` — historical (bug-fix / original buggy code).

Tags pin the binary versions every report points at:

| Tag | Commit | What |
|---|---|---|
| `binary-v0-buggy` | `0f782fd` | original code with the min/max tracking bug |
| `binary-v1-bugfix` | `d5bf30c` | one-line reduction fix |
| `binary-v2-viz` | `616c147` | region metrics + outlier dump + visualizer |
| `binary-v3-9point` | `ab47540` | 9-point sampling stencil |
| `binary-v4-pq` | `37676d5` | shared priority queue (off `main`, negative) |
| `binary-v5-affinity` | `fc33e29` | GPU-affinity min-max queue (current best) |

## What's next

Scheduling is now exhausted: with GPU affinity the GPU runs ~91 % utilised and
the CPU pool is the wall, while total compute (~595 s) is **conserved** by any
partitioning. The only remaining lever is **work-reduction** — skipping
iterations on in-set pixels:

1. **Periodicity checking** in `diverge()` — detect the cycle interior orbits
   settle into and bail early. Exact, no image change; directly cuts the
   in-set pixel cost that bounds the wall. *(Recommended next: `feat/periodicity-check`.)*
2. **Heuristic interior certificate** — constant-fill a region whose samples all
   reach `maxIter`. Faster but risks filling over thin filaments. (The exact
   cardioid/bulb certificate applies only where corners are in the main
   cardioid/bulb — true for the seahorse regime of report 13, not the canonical
   minibrot outlier.)

Lower-priority: async CUDA streams (small recoverable idle, needs a cross-region
pipeline) and a `pixelSizeThresh` sweep. Full rationale: [`CLAUDE.md`](CLAUDE.md),
*Recommended next steps*.

## More

- [`CLAUDE.md`](CLAUDE.md) — full project reference (architecture, results, recommended next steps)
- [`INSTRUMENTATION.md`](INSTRUMENTATION.md) — instrumentation deep-dive
- [`reports/`](reports/) — the LaTeX reports
- [`experiments/`](experiments/) — raw measurement data
