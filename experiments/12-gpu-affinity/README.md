# 12 — GPU affinity (min-max work queue) vs FIFO

Binary: `feat/gpu-affinity` (intended tag `binary-v5-affinity`), built on
`binary-v3-9point` (`ab47540`). Baseline: `binary-v3-9point` (FIFO `deque`),
worktree `/home/lynx/box/cpp/mandel-base-v3`.

Change: `WorkQueue` `deque` → `std::multiset` ordered by pixel count;
`extract(isGPU)` gives the **GPU thread the largest** region and **CPU threads
the smallest** (GPU affinity). `MandelRegion::Compare` made `const` + ascending.
Decomposition unchanged (5,608 leaves). On AC (machine healthy post-reboot;
base gate = 51.2 s, GPU 4,235).

## Results (healthy machine, diffT=0.1, hybrid)

| Metric | FIFO base | GPU affinity | Δ |
|---|---:|---:|---:|
| Wall (A/B mean, 3 reps) | 51.73 s | **49.86 s** | **−3.6%** (non-overlapping) |
| Leaves | 5,608 | 5,608 | 0 |
| GPU / CPU leaves | 4,235 / 1,373 | 3,679 / 1,929 | bigger→GPU |
| GPU kernel avg / max | 2.96 / 139 ms | 5.80 / **460 ms** | GPU took the outlier |
| GPU took 960×540 / 480×270 | 0 / 182 | **1 / 259** | |
| CPU worst region | 13.3 s | **1.34 s** | outlier off CPU |
| CPU median / p99 | 88 / 4004 ms | 8.7 / 1205 ms | tail collapsed |
| CPU outliers >5 s | 1 | **0** | |
| CPU thread wall | 50.4–51.2 s | 49.0–49.6 s | relieved |

**First lever to move the wall.** GPU affinity routes the big coherent interior
regions (incl. the 960×540 / 14 s outlier → 460 ms on GPU) to the ~30×-faster
GPU, relieving the CPU pool. Unlike the shared largest-first priority queue
(report 11, 11:1 CPU:GPU extraction kept the outlier on CPU), the executor-aware
min-max extraction reserves the big regions for the GPU. Bounded at −3.6%
because the GPU is now ~91% utilized (near-saturation) and the CPU pool (~49.5 s)
is the new wall — further gains need work-reduction (periodicity checking).
A CPU-only guard makes CPU threads pull the largest (LPT) when no GPU is present
(see below; measured a wash at this scale). Full analysis: `reports/12-gpu-affinity.tex`.

## Contents
- `logs/` — `metrics_{base,aff}.stderr`, `dist_{base,aff}.stderr`.
- `perf/` — `results.csv` + per-rep stderr.
- `nsys/` — `{base,aff}_stats.txt`. `.nsys-rep`/`.sqlite` gitignored.

## CPU-only guard (cpu_only/)

The guard makes CPU threads pull the largest region (LPT) when no GPU is present
(instead of smallest-first). Measured head-to-head (gpu=0, 12 threads, 2 reps):

| Ordering | wall mean | worst region |
|---|---:|---:|
| FIFO (baseline) | 156.84 s | 14.1 s |
| smallest-first (no guard) | 157.41 s | 14.0 s |
| largest-first / LPT (guard) | 156.88 s | 14.3 s |

All within ~0.4% (noise): at 5,608 fine-grained leaves / 12 threads the tail is
self-filling, so ordering is a wash. The feared smallest-first regression doesn't
materialize; LPT doesn't beat FIFO. Guard kept as the principled, zero-cost
choice (protects coarser/smaller workloads), but earns nothing measurable here.
