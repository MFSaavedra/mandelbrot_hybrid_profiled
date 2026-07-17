# mandelHybrid — Hybrid CPU+GPU Mandelbrot Renderer

A 100-frame Mandelbrot zoom renderer that splits work between CPU threads and
a GPU thread, using adaptive region subdivision as its load-balancing
strategy. Built on the example from Section 11.5.1 of Barlas,
*Multicore and GPU Programming* (2nd ed., 2022).

This repository extends the original code with NVTX + CUDA-event
instrumentation and end-to-end wall timing, a one-line bug fix to
`MandelRegion::examine()`'s min/max reduction, a **9-point sampling
stencil**, a **GPU-affinity work queue**, an exact **Brent periodicity
check** in both the CPU and CUDA `diverge()` (the two big wall wins), a
three-mode **subdivision visualizer**, an OpenCL **iGPU third executor**
(branch-only), **multi-node frame distribution** with weighted-random static
shares and a work-stealing coordinator, a CPU-only `make GPU=0` build,
Doxygen API docs, batch sweep drivers, and twenty-six LaTeX reports
(measurements, a code/architecture guide, kernel-level Nsight Compute
studies, and a divisible-load-theory analysis).

**Current best (single machine):** `main` carries tag
`binary-v7-gpu-periodicity` — hybrid wall **23.72 s** (100 frames,
1920×1080, i7-9750H + GTX 1660 Ti Max-Q, 12 threads, `diffT=0.1`), down
from the legacy ~53 s: −3.6 % from GPU affinity, −46 % from the CPU
periodicity check, −20.5 % more from the GPU-side check — with
byte-identical output at every step.

**Current best (distributed):** laptop + a 4× faster WAN node (RTX 4090
behind an SSH jump), weighted static shares + tar-stream collection:
**6.45 ± 0.26 s** end-to-end delivered — parity with fetching the fast
node's solo run, the pair's analytic ceiling (reports 25–28).

## Quick start

```bash
make                                       # build mandelHybrid (autodetects CUDA + host g++)
make GPU=0                                 # CPU-only build (no CUDA needed; kernel stub)
./mandelHybrid spec.in                     # render with default settings
./mandelHybrid spec.in 12 1 0.1            # 12 threads, GPU on, diffThreshold 0.1
./mandelHybrid spec.in 12 1 0.1 32768 1 0 2  # viz=2: overlay each frame's partition
scripts/sweep_fig1115.sh                   # thread-configuration sweep (Fig 11.15)
scripts/dist_frames.sh hosts.txt           # static multi-node frame distribution over SSH
WEIGHTS=1,4 scripts/dist_dynamic.sh hosts.txt  # weighted shares + work stealing
scripts/build_report.sh all                # build every report PDF
doxygen Doxyfile                           # API docs -> docs/html/
```

Run signature: `./mandelHybrid spec.in [numThr] [gpuEnable] [diffT] [pixT] [quiet] [save] [viz] [vizFrame]`.
`viz` is `0`/`1`/`2`/`3` (single-frame depth animation / full-run partition
overlay / full-run split-by-split process), coloured by executor
(cyan = GPU leaf, yellow = CPU leaf, grey = split skeleton).

Multi-node frame ownership is controlled by environment variables —
`DIST_NODES`/`DIST_RANK`/`DIST_BLOCK` (block-cyclic),
`DIST_WEIGHTS`+`DIST_SEED` (seeded weighted-random shares, communication-free)
and `DIST_FRAMES` (explicit list, the work-stealing dispenser interface) —
frames are distributed *across* nodes, the region queue stays intra-node,
and output keeps global frame indices so collection is a plain union. See
`CLAUDE.md` for the full reference.

Dependencies: `nvcc` (CUDA 11.6+ for `-arch=native`), `g++`, Qt5
(`Qt5Core` + `Qt5Gui` via `pkg-config`). The Makefile autodetects the
toolchain (`/opt/cuda` or `/usr/local/cuda`; `g++-15` if present, else
system `g++`; override with `make CUDAINST=… HOSTCC=…`) and always invokes
the toolkit's own `nvcc`, never `PATH`'s. `make GPU=0` needs only `g++`
and Qt5.

## Repository layout

```
.
├── *.cpp, *.h, *.cu, Makefile, spec.in   source and build
├── reports/                              numbered LaTeX reports + shared template
│   └── NN-name.tex (+ tracked .pdf)
├── experiments/                          raw measurement data per report
│   └── NN-name/
├── scripts/
│   ├── sweep_fig1115.sh                  Fig 11.15 thread-configuration sweep
│   ├── plot_fig1115.py                   plot the sweep CSV
│   ├── dist_frames.sh                    static multi-node distribution (GNU parallel/SSH)
│   ├── dist_dynamic.sh                   weighted shares + work-stealing coordinator
│   └── build_report.sh                   pdflatex×2 for any (or all) reports
├── docs/                                 Doxygen mainpage (html output gitignored)
├── INSTRUMENTATION.md                    deep-dive on the NVTX / CUDA hooks
└── CLAUDE.md                             full project reference for Claude Code agents
```

## Reports

Each report's title page lists the commit hash and tag of the binary that
produced its measurements. (There are no reports 10/19; the `diffT`
re-tuning lives in report 09 §Re-Tuning Check and the threshold-optimum
re-price in report 21's follow-ups.)

| # | Name | Binary | Subject |
|---|---|---|---|
| 01 | initial-profile | `binary-v0-buggy` | Original NVTX / CUDA-event profile |
| 02 | fig1115-replication | `binary-v0-buggy` | Fig 11.15 thread-configuration sweep |
| 03 | bug-analysis | `v0` + `v1-bugfix` | The min/max tracking bug and the one-line fix |
| 04 | postfix-profile | `binary-v1-bugfix` | Re-profile after the bug fix |
| 05 | fig1115-postfix | `binary-v1-bugfix` | Fig 11.15 sweep on the post-fix binary |
| 06 | difft-sweep | `binary-v1-bugfix` | `diffThreshold` sensitivity (0.1 is best) |
| 07 | difft-compare | `v0` + `v1` | Pre- vs. post-fix `diffT` comparison |
| 08 | region-metrics-viz | `binary-v2-viz` | Region metrics + visualizer; locates the outliers |
| 09 | 9point-sampling | `binary-v3-9point` | 9-point stencil; catches one of two outliers; wall-neutral |
| 11 | priority-queue | `binary-v4-pq` | Shared largest-first queue — **negative result** (not merged) |
| 12 | gpu-affinity | `binary-v5-affinity` | Min-max queue, GPU pops largest — **first wall win, −3.6 %** |
| 13 | zoom-points | `binary-v5-affinity` | Load balance across four zoom regimes (cost spans 32×) |
| 14 | architecture-guide | `binary-v5-affinity` | File-by-file code guide (not a measurement) |
| 15 | async-streams-analysis | `binary-v5-affinity` | Async streams priced out at sub-1 % of wall |
| 16 | ncu-divergence | `binary-v5-affinity` | Coherent regions 100 % warp-efficient, FP64-bound at 85.6 % of peak |
| 17 | ncu-zoom-points | `binary-v5-affinity` | Report 16 verified on the four zoom regimes |
| 18 | maxiter-split | `binary-maxiter-split` | `diffT=0.1` ≡ a parameter-free interior certificate (not merged) |
| 20 | igpu-opencl | `feat/igpu-opencl` | iGPU third executor: −23.2 % atop v5 (code not merged) |
| 21 | periodicity-check | `binary-v6-periodicity` | **CPU Brent periodicity: hybrid −46.3 %, CPU12 −78.6 %**, byte-identical |
| 22 | igpu-atop-v6 | `feat/igpu-opencl` @ v6 | iGPU re-priced after periodicity: a wash — keep off main |
| 23 | gpu-periodicity | `binary-v7-gpu-periodicity` | **GPU Brent periodicity: kernel 2.34×, hybrid 23.72 s** (best) |
| 24 | ncu-gpu-periodicity | `binary-v7` vs ncu | ncu re-baseline: “degraded” warp metrics = eliminated waste |
| 25 | frame-distribution | `binary-frame-dist` | Multi-node frames: identity 100/100; slow nodes buy capacity, not latency |
| 26 | dynamic-balancing | branch @ `8c21bf4` | Weighted shares + work stealing; `KCAP·w_r` chunk-cap fix |
| 27 | collection-batching | branch @ `f310656` | Fast WAN node: collection becomes the wall; one tar stream −51 % |
| 28 | dlt-analysis | analysis (DLTlib) | DLT retrodicts 25–27: weights rule is share ∝ 1/(pᵢ+lᵢ) |

PDFs are tracked at `reports/NN-name.pdf` and rebuilt via `scripts/build_report.sh`.

## Key findings

| Finding | Numbers |
|---|---|
| **`diffThreshold` should be 0.1, not 0.5** | Flat optimum over 0.05–0.20; leaf count saturates by 0.1. At 0.1 the rule coincides with a parameter-free interior certificate (report 18). |
| **Work-reduction beats scheduling by an order of magnitude** | 99.5 % of all iterations were interior pixels grinding to `maxIter`. Exact Brent periodicity in the CPU `diverge()`: hybrid 57.6 → 30.9 s, CPU12 164.4 → 35.2 s, output byte-identical (report 21). |
| **The “GPU periodicity will lose” prediction was wrong** | Same check in the CUDA kernel: mix-controlled kernel 2.34×, GPU-only −48.8 %, hybrid 29.83 → **23.72 s** (report 23). A warp pays its slowest lane's detection latency, not 32×`maxIter`. |
| **GPU affinity is the scheduling lever that worked** | Min-max queue (GPU pops largest): −3.6 %; routes the 960×540 minibrot-interior outlier to the GPU (460 ms vs 13.3 s). A *shared* priority queue does nothing (report 11, 11:1 extraction). |
| **100 % warp efficiency can measure waste, not health** | Reports 16/24: v5's interior kernels were 100 % warp-efficient *because every lane ground to `maxIter`*; v7 drops to 96.3 % while running 9.9× faster. FP64 pipe ~85 % of peak in both. |
| **The iGPU third executor stopped paying after periodicity** | −23.2 % atop v5 → a wash atop v6 (its 27 CPU-s of relief ≈ the displaced CPU worker's 26.5 s). Kept off main (reports 20/22). |
| **Slow nodes buy capacity, not latency** | Frame distribution: every equal-share 2-node config loses to the fast node alone (laptop+ivy report 25, laptop+yeco report 27); the weighted ideal is only 1.22–1.24× over the fast node solo — realized three times. |
| **Balanced compute exposes the transport** | With calibrated 1:4 shares both nodes hit the 4.14 s harmonic ideal yet e2e was 13.11 s: per-file scp costs ~68 ms/file (2–3 RTTs). One tar stream per rank: **6.45 ± 0.26 s** (−51 %, σ ×7 tighter) — report 27. |
| **The theory retrodicts all of it** | Collection-aware DLT (report 28): our guessed weights are the model's `l=0` solution; the work-stealer's 42/58 scp equilibrium is 3 frames from the closed-form optimum; weights rule going forward: **share ∝ 1/(pᵢ + lᵢ)**. |

## Branches and tags

- `main` — current best; holds all reports, all experiment data, and every merged optimization (bug fix, 9-point stencil, GPU affinity, CPU+GPU periodicity, frame distribution + weighted/dynamic balancing + collection batching, `GPU=0` build, Doxygen docs).
- `feat/priority-queue` — **not merged** (`binary-v4-pq`): shared priority queue, negative result (report 11).
- `examine/maxiter-split` — **not merged** (`binary-maxiter-split`): parameter-free split rule, characterization (report 18).
- `feat/igpu-opencl` — **code not merged** (reports 20/22 and their data are on `main`): OpenCL iGPU executor, re-priced to a wash atop v6.
- `examine/random-sampling` — **not merged, WIP**: random interior probes, variance characterization.
- `feat/frame-distribution` — **merged** (reports 25–28, tag `binary-frame-dist`).
- `examine_minmax_bugfix`, `examine_rewrite`, `master` — historical (bug-fix / original buggy code).

Tags pin the binary versions every report points at:

| Tag | Commit | What |
|---|---|---|
| `binary-v0-buggy` | `0f782fd` | original code with the min/max tracking bug |
| `binary-v1-bugfix` | `d5bf30c` | one-line reduction fix |
| `binary-v2-viz` | `616c147` | region metrics + outlier dump + visualizer |
| `binary-v3-9point` | `ab47540` | 9-point sampling stencil |
| `binary-v4-pq` | `37676d5` | shared priority queue (off `main`, negative) |
| `binary-v5-affinity` | `fc33e29` | GPU-affinity min-max queue |
| `binary-v6-periodicity` | `fa6d8ac` | CPU Brent periodicity check |
| `binary-v7-gpu-periodicity` | `2abc909` | GPU Brent periodicity check (**current best**) |
| `binary-maxiter-split` | `d572c78` | all-`maxIter` split rule (off `main`, characterization) |
| `binary-frame-dist` | `aae49fb` | `DIST_*` static frame distribution PoC |

## What's next

With v7 the CPU pool still binds (busiest worker ≈ wall) at −20 % cost, and
the GPU thread's kernel duty fell 85 → 45 % of its wall, so the open levers
are:

1. **Host-side commit cheapening** — the GPU thread now spends most of its
   time in `scanLine` row copies, not kernels; this lever finally transfers
   to wall (report 23).
2. **Rate-adaptive chunk caps** in the dispenser — derive each node's cap
   and steal size from its first measured chunk instead of operator weights
   (`KCAP·w_r` inherits weight miscalibration; reports 26/27). This is a
   runtime fit of the DLT rate 1/(pᵢ+lᵢ) (report 28).
3. **Homogeneous interleave study** (report 13's earmark) — block vs cyclic
   vs block-cyclic on comparable nodes; both measured pairs were too
   asymmetric for interleave effects to be visible.
4. **~1.2 s fixed orchestration** on the distributed path (parallel
   spin-up, per-rank mkdir/cleanup) — amortizable by keeping rank workdirs
   warm.

Full rationale: [`CLAUDE.md`](CLAUDE.md), *Recommended next steps*.

## More

- [`CLAUDE.md`](CLAUDE.md) — full project reference (architecture, results, recommended next steps)
- [`INSTRUMENTATION.md`](INSTRUMENTATION.md) — instrumentation deep-dive
- `doxygen Doxyfile` → `docs/html/` — API documentation
- [`reports/`](reports/) — the LaTeX reports (tracked PDFs)
- [`experiments/`](experiments/) — raw measurement data
