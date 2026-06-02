# 11 — Largest-first priority queue vs FIFO

Binary: `feat/priority-queue` (intended tag `binary-v4-pq`), built on
`binary-v3-9point` (`ab47540`). Baseline: `binary-v3-9point` (FIFO `deque`),
built in worktree `/home/lynx/box/cpp/mandel-base-v3`.

Change: `WorkQueue` `deque` → `std::priority_queue` with `MandelRegion::Compare`
corrected to largest-first (was `>` = smallest-first; now `<`). Decomposition
unchanged (5,608 leaves) — reorders processing, not which regions exist.

Machine: i7-9750H + GTX 1660 Ti Max-Q, **on AC** (see note). Config: 100 frames,
1920×1080, `diffT=0.1`, `pixT=32768`, 12 threads, `quiet=1`, `save=0`.

> **AC-power gate.** All numbers here are post-gate: one FIFO base run must
> return ≈52 s with GPU ≥ 4,000 leaves before the pass proceeds. See the
> contamination note below and memory `ac-power-for-timing`.

## Results (healthy machine)

| Metric | FIFO base | Priority (largest-first) | Δ |
|---|---:|---:|---:|
| Wall (A/B mean, 3 reps) | 51.37 s | 51.09 s | −0.55% (noise) |
| Leaves | 5,608 | 5,608 | 0 |
| GPU / CPU leaves | 4,194 / 1,414 | 4,014 / 1,594 | +180 to CPU |
| GPU-thread busy (NVTX) | 46.9 s | 46.5 s | ~equal (~91% util) |
| Kernel max (CUPTI) | 139 ms | 139 ms | **unchanged** |
| Worst CPU region | 13.3 s | 14.4 s | outlier stays on CPU |

**Conclusion: wall-neutral.** The priority queue does NOT route the ~14 s
outlier to the GPU (kernel max stays 139 ms → the big region ran on a CPU thread
both ways). With 11 CPU extractors vs 1 GPU, a *shared* largest-first queue hands
big regions to a CPU thread ~11:1; it reorders *when*, not *which processor*.
Load was already balanced <2% and total compute is conserved (~610 s), so
reordering can't help. To offload the outlier to the GPU you need GPU *affinity*
(separate lane / GPU pulls largest), not a shared queue. Full analysis:
`reports/11-priority-queue.tex`.

## ⚠ Power-state contamination (resolved)

The first attempt was run on **battery**: the Max-Q GPU was power-capped to
**30 W** (default 60 W, `SW Power Cap` active → 14 W / 1140 MHz under load), so
the GPU did ~1,670 leaves instead of ~4,190 and the run took ~147 s — a false
2× "regression". A reboot restored the 60 W limit; the AC gate then passed
(50.5 s, GPU 4,167, 35.8 W / 1845 MHz mid-run) and the data above was collected.

## Directory contents

- `logs/` — `metrics_{base,prio}.stderr`, `dist_{base,prio}.stderr`, `gate3_base.*`.
- `perf/` — `results.csv` + per-rep stderr.
- `nsys/` — `{base,prio}_stats.txt` (NVTX/CUDA/kernel). `.nsys-rep`/`.sqlite` gitignored.
