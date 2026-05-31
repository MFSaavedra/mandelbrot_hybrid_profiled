# 08 — Region metrics + subdivision visualization

Binary: `feat/viz-mode` branch, built on top of `d5bf30c` (`binary-v1-bugfix`).
New code adds: per-region work metrics (mean iterations/pixel, interior-pixel
fraction), an `[OUTLIER]` dump for any CPU region exceeding 5 s (location,
corner spread, cardioid/bulb membership), and a `vizMode` that freezes one
interpolated frame and emits a depth-by-depth subdivision animation coloured by
executor (cyan = GPU thread, yellow = CPU threads, grey = split skeleton).

Machine: i7-9750H (6c/12t) + GTX 1660 Ti Max-Q. Date: 2026-05-31.

## Commands

All from the project root, `spec.in` = 100 frames, 1920×1080, diffT=0.1, pixT=32768.

```bash
# Visualization run — freeze frame N, emit <prefix>_f00NN_dKK.png per depth K.
#   args: spec numThr gpu diffT pixT quiet save viz vizFrame
./mandelHybrid spec.in 12 1 0.1 32768 1 0 1 89    # frame 89 (outlier frame)
./mandelHybrid spec.in 12 1 0.1 32768 1 0 1 90    # frame 90 (GPU+CPU mix)

# Metrics run — full 100-frame hybrid, quiet=1 save=0; [OUTLIER] lines fire regardless of quiet.
./mandelHybrid spec.in 12 1 0.1 32768 1 0 > logs/metrics_hybrid.stdout 2> logs/metrics_hybrid.stderr

# Performance A/B — new binary vs d5bf30c baseline, 3 reps each, alternating.
#   (baseline built in worktree /home/lynx/box/cpp/mandel-baseline-d5bf30c)
```

## Directory contents

- `viz/` — depth-animation PNGs (`*_dNN.png`) and the contact sheet / GIF.
- `logs/` — stderr/stdout from the viz and metrics runs.
- `perf/` — `results.csv` and per-rep logs for the new-vs-baseline timing A/B.

## Headline results (metrics run, hybrid, diffT=0.1, save=0)

- End-to-end wall: 50.27 s. Total leaves 5,323 (1,537 CPU + 3,786 GPU).
- GPU avg 11.99 ms/region; CPU threads 49.5–50.3 s wall (≈1.5 % spread).
- Two binding outliers, both 960×540 depth-1, corners all at `maxIter`
  (spread 0 → classified uniform → never split):

  | Frame | corners (=maxIter) | meanIter | inset % | cardioid/bulb | CPU time |
  |------:|-------------------:|---------:|--------:|:-------------:|---------:|
  | 73    | 7327               | 7131.2   | 97.3 %  | **no**        | 13.64 s  |
  | 89    | 8911               | 7551.2   | 84.4 %  | **no**        | 14.29 s  |

  Key finding: `cardioidBulb=0` for both. The outliers are interior to the set
  but NOT in the main cardioid or period-2 bulb — they are inside a *minibrot*.
  The cheap cardioid/bulb interior certificate (CLAUDE.md next-step 2a) would
  therefore NOT catch them; a general interior test or edge-midpoint sampling
  (2b) is required.

## Performance A/B (see perf/results.csv)

New instrumentation is flag-gated (vizMode off, quiet on) plus a negligible
always-on per-pixel accumulation. The A/B confirms no measurable regression
versus the `d5bf30c` baseline at the same config — numbers in `perf/results.csv`.
