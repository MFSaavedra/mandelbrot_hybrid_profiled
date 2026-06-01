# 09 — 9-point vs 4-point sampling

Binary: `examine/9-points-sampling` branch (9-point stencil), built on top of
`binary-v2-viz` (`616c147`). Baseline: `binary-v2-viz` (4-corner sampler), built
in worktree `/home/lynx/box/cpp/mandel-4point`.

Machine: i7-9750H (6c/12t) + GTX 1660 Ti Max-Q. Date: 2026-05-31.
Config for all runs: 100 frames, 1920×1080, `diffT=0.1`, `pixT=32768`,
12 threads, GPU on, `quiet=1`, `save=0`.

> Note: pass run args as separate literal tokens. This shell is **zsh**, which
> does NOT word-split an unquoted `$VAR`, so `./mandelHybrid spec $CFG` would
> pass the whole string as one argv token (silently falling back to defaults:
> `diffT=0.5`, `save=1`). Always: `./mandelHybrid spec 12 1 0.1 32768 1 0`.

## Commands

```bash
# metrics runs (rich per-thread + outlier data)
./mandelHybrid spec.in 12 1 0.1 32768 1 0   # 9-point (this branch)
/home/lynx/box/cpp/mandel-4point/mandelHybrid spec.in 12 1 0.1 32768 1 0   # 4-point baseline

# A/B: 3 reps each, alternating -> perf/results.csv
# viz=3 split-by-split animation (9-point) -> viz_process/
./mandelHybrid spec.in 12 1 0.1 32768 1 0 3
```

## Directory contents

- `logs/` — `metrics_{9point,4point}.stderr` (config, outliers, per-thread + GPU summaries).
- `perf/` — `results.csv` + per-rep stderr for the wall-time A/B.
- `viz_process/` — 9-point `viz=3` animation (400 frames) + `.mp4`s.
- `viz/` — 9-point `viz=1` frame-89 partition (figure source).

## Headline results (`diffT=0.1`, 100 frames)

| Metric | 4-point | 9-point | Δ |
|---|---:|---:|---:|
| Leaf regions | 5,323 | 5,608 | +5.4% |
| `examine()` calls | 7,064 | 7,444 | +5.4% |
| Outliers >5 s | 2 (f73, f89) | **1 (f73)** | −1 |
| Worst region | 15.4 s (f89) | 13.3 s (f73) | −2.1 s |
| GPU avg ms/region | 11.56 | 11.01 | −0.55 |
| Wall (A/B mean, 3 reps) | 52.07 s | 52.39 s | +0.6% (noise) |

Key finding: the 9-point stencil **subdivides the frame-89 outlier** (84.4%
interior — an interior sample hit a divergent pixel) but **not frame 73**
(97.3% interior — all 9 points still land at `maxIter`, spread 0).

Deep-profile (nsys + quiet=0) findings — see `nsys/` and `reports/09-…`:
- **Sampling overhead is negligible**, measured via NVTX (`examine − compute`):
  +0.40 s CPU total across 11 threads (+29 µs/`examine`, ~37 ms wall-equiv).
  My first draft wrongly said it "offsets the gain" — it does not.
- **Total compute is conserved** under any partitioning: CPU+GPU pixel-loop time
  is ~595 s in both binaries (+0.9%, noise). Splitting moves work between
  threads but removes none.
- **CPU per-region distribution shifts down** (median 126→99 ms, p99 4247→3833,
  max 15536→13426 ms); GPU distribution and kernel max (~137 ms) unchanged;
  `cudaMemcpy` is 99% kernel-wait (report-04 finding holds).
- **Conclusion:** the wall is bound by total interior-pixel work; the load is
  already balanced <2%, so finer splitting can't help here. The lever is
  work-*reduction* (periodicity checking in `diverge()`, or an interior
  certificate), not subdivision. Full analysis in `reports/09-9point-sampling.tex`.
