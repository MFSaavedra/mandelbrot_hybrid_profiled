# mandelHybrid — Hybrid CPU+GPU Mandelbrot Renderer

A 100-frame Mandelbrot zoom renderer that splits work between CPU threads and
a single GPU thread, using adaptive region subdivision as its load-balancing
strategy. Built on the example from Section 11.5.1 of Barlas,
*Multicore and GPU Programming* (2nd ed., 2022).

This repository extends the original code with NVTX + CUDA event
instrumentation, end-to-end wall timing, batch sweep drivers, a one-line bug
fix to `MandelRegion::examine()`'s min/max reduction, and seven LaTeX reports
covering everything that has been measured.

## Quick start

```bash
make                                       # build mandelHybrid
./mandelHybrid spec.in                     # render with default settings
scripts/sweep_fig1115.sh                   # full thread-configuration sweep
scripts/build_report.sh all                # build all seven report PDFs
```

Dependencies: `nvcc` (CUDA 11.6+), `g++-15`, Qt5 (`Qt5Core` + `Qt5Gui` via
`pkg-config`). The Makefile pins `nvcc -ccbin g++-15` because `nvcc` 13.2
does not yet support gcc 16.

## Repository layout

```
.
├── *.cpp, *.h, *.cu, Makefile, spec.in   source and build
├── reports/                              7 numbered LaTeX reports + shared template
│   └── NN-name.tex
├── experiments/                          raw measurement data per report
│   └── NN-name/
├── scripts/
│   ├── sweep_fig1115.sh                  Fig 11.15 thread-configuration sweep
│   ├── plot_fig1115.py                   plot the sweep CSV
│   └── build_report.sh                   pdflatex×2 for any (or all) reports
├── INSTRUMENTATION.md                    deep-dive on the NVTX / CUDA hooks
└── CLAUDE.md                             reference for Claude Code agents
```

## Reports

Each report's title page lists the commit hash and tag of the binary that
produced its measurements.

| # | Name | Binary | Subject |
|---|---|---|---|
| 01 | initial-profile | `0f782fd` | Original NVTX / CUDA-event profile |
| 02 | fig1115-replication | `0f782fd` | Fig 11.15 thread-configuration sweep |
| 03 | bug-analysis | `0f782fd` + `d5bf30c` | The min/max tracking bug and the one-line fix |
| 04 | postfix-profile | `d5bf30c` | Re-profile after the bug fix |
| 05 | fig1115-postfix | `d5bf30c` | Fig 11.15 sweep on the post-fix binary |
| 06 | difft-sweep | `d5bf30c` | `diffThreshold` sensitivity sweep |
| 07 | difft-compare | `0f782fd` + `d5bf30c` | Pre-fix vs. post-fix `diffT` comparison |

PDFs build under `reports/NN-name.pdf` via `scripts/build_report.sh`.

## Key findings

| Finding | Numbers |
|---|---|
| **`diffThreshold` should be 0.1, not 0.5** | Best hybrid wall ≈ 52.80 s vs. 54.61 s at the legacy default (−3.3 %) |
| **The four-corner heuristic has a hard binding outlier** | A 960 × 540 region with all four corners inside the Mandelbrot set takes ~15 s on a single CPU thread at *every* `diffT` value |
| **`cudaMemcpy` time is mostly kernel-wait, not transfer** | Reported 13 ms / call vs. 24 µs of actual D → H transfer |
| **The bug fix is correctness-positive, performance config-dependent** | CPU-only ~1 % faster, GPU-only ~1 % slower, hybrid neutral at every `diffT` except 0.5 |

## Branches and tags

- `main` — current best version with all reports and data.
- `examine_minmax_bugfix` — the bug-fix commit, kept as a historical reference.
- `examine_rewrite`, `master` — original buggy code.

Tags pin the two binary versions every report ultimately points at:

- `binary-v0-buggy` → `0f782fd` — original code with the min/max tracking bug
- `binary-v1-bugfix` → `d5bf30c` — same code with the one-line reduction fix

## What's next

The four-corner heuristic has reached its ceiling on this workload. Further
wall-time gains require attacking the 15 s outlier region directly. In
priority order:

1. Locate the worst region (one short instrumentation patch).
2. If interior to the main cardioid / period-2 bulb: ship an analytic
   interior certificate. Otherwise: add edge-midpoint sampling.
3. Async CUDA streams (independent; recovers ~13 ms / region of GPU thread
   idle time spent in `cudaMemcpy`).
4. Largest-first priority queue for the work scheduler.

Full discussion and rationale: [`CLAUDE.md`](CLAUDE.md), the *Recommended next
steps* section.

## More

- [`CLAUDE.md`](CLAUDE.md) — full project reference (architecture, results, recommended next steps)
- [`INSTRUMENTATION.md`](INSTRUMENTATION.md) — instrumentation deep-dive
- [`reports/`](reports/) — the seven LaTeX reports
- [`experiments/`](experiments/) — raw measurement data
