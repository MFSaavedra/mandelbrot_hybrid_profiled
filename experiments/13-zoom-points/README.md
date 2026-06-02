# Experiment 13 — Zoom-point load-balance characterization

Profiles the **current best binary** (`binary-v5-affinity`, GPU affinity min-max queue)
zooming into four qualitatively different targets, to see how the dynamic work
queue + GPU affinity behaves across regimes. All runs share the same wide
full-set first frame; only the deep-zoom target (spec line 3) differs.

- **Binary**: commit `ff1f2f9` (tag `binary-v5-affinity-1-gff1f2f9`, i.e. `binary-v5-affinity` + CLAUDE.md registration commit)
- **Machine**: i7-9750H + GTX 1660 Ti Max-Q, on AC power
- **Date**: 2026-06-01
- **Params**: `numThreads=12 gpuEnable=1 diffT=0.1 pixT=32768 quiet=1 save=0`
- **Frames**: 100 @ 1920×1080, `maxIter` ramp 100→10000

## Targets

| Point | spec | Target c | Regime |
|---|---|---|---|
| outside | `spec_outside.in` | (1.0, 1.0) | exterior — deep frames vanish |
| misiurewicz | `spec_misiurewicz.in` | (−0.77568377, 0.13646737) | boundary, thin exterior spiral |
| seahorse | `spec_seahorse.in` | (−0.745428, 0.113009) | boundary, main-cardioid crossing |
| inside | `spec_inside.in` | (−0.5, 0.0) | interior — all maxIter |

Canonical point used by all prior reports (for reference): minibrot at
c ≈ (0.0016437, −0.8224676), `cardioidBulb=0`.

## Run

```bash
experiments/13-zoom-points/run.sh timing   # viz=0 save=0  -> logs/<pt>.timing.stderr
experiments/13-zoom-points/run.sh viz      # viz=3 save=0  -> logs/<pt>.viz.stderr, viz_<pt>/*.png
```

Timing and viz run the four points sequentially (no concurrent run) so wall
timing is never perturbed. `viz=3` forces `save=0`; the overlaid PNGs are the
deliverable (cyan = GPU leaf, yellow = CPU leaf, grey = split skeleton).

## Results (timing phase, 1 rep each)

| Point | Wall (s) | CPU thread spread | GPU util | GPU regions | CPU regions | GPU avg/region | CPU outliers (>5 s) |
|---|---|---|---|---|---|---|---|
| outside     | 16.95 | 1.4% | 79% | 2190 | 2758 | 6.1 ms | 0 |
| misiurewicz |  2.37 | 0.2% | 1.4%| 3312 | 3088 | 0.010 ms | 0 |
| seahorse    | 69.82 | 0.9% | 95% | 1128 | 2908 | 58.7 ms | **8** |
| inside      | 76.28 | 0.8% | 96% |  573 | 3346 | 127 ms | 0 |

### Key observations

1. **32× cost range** (2.37→76.3 s) from the same binary/params — cost is set
   entirely by how much in-set / slow-escape area the zoom path sweeps.
2. **Dynamic balance is excellent in every regime** (0.2–1.4% thread spread).
   The work queue + affinity is not the bottleneck anywhere.
3. **The binding resource shifts**: light regimes (misiurewicz, outside) are
   CPU-pool bound but trivially; heavy regimes (seahorse 95%, inside 96%) are
   **GPU-saturated**.
4. **GPU affinity in action**: as work coarsens toward interior, the GPU takes
   *fewer but bigger* regions (inside: 15% of leaf count, 127 ms each, 96% util;
   misiurewicz: 52% of count, 0.01 ms each, 1.4% util).
5. **Outliers only in the mixed-boundary regime** (seahorse, 8×). All 8 are
   `cardioidBulb=1` 960×540 depth-1 quadrants at frames 85–98 (17–18.7 s each) —
   **main-cardioid interior**, the case the cheap cardioid certificate (report 08)
   *would* fix, unlike the canonical minibrot (`cardioidBulb=0`). They overflow
   to CPU because the GPU is already saturated; balance stays tight because the
   run is throughput-bound (the outliers overlap hundreds of other regions).
6. **Uniform interior (inside) produces ZERO outliers** despite max total work:
   subdivision to the pixel floor + GPU affinity grabbing the largest tiles keeps
   every CPU region under 5 s.

## Visualization (viz=3) — animations + figures

`run.sh viz` generated the split-by-split process animation for each point
(`viz_<pt>/NNNNN.png`, single global counter; cyan = GPU leaf, yellow = CPU leaf,
grey = split skeleton, 2 px inward stroke). Frame counts: outside 370,
misiurewicz 397, seahorse 373, inside 352.

Assembled animations (ffmpeg, native res, `-bf 0 -fps_mode passthrough
-pix_fmt yuv444p`): `anim_<pt>.mp4`.

Curated report figures (copied to `reports/img/`):

| File | Source | Shows |
|---|---|---|
| `13-zoom-baseline-frame0.png`    | outside/00003     | Frame 0 full set, all-yellow CPU grid — **no GPU participation at maxIter=100** |
| `13-zoom-outside-deep.png`       | outside/00369     | Deep exterior: solid red, single **cyan GPU** tile — the trivial floor |
| `13-zoom-misiurewicz-deep.png`   | misiurewicz/00396 | Misiurewicz spiral filaments in red fast-escape exterior — thin = cheap |
| `13-zoom-seahorse-deep.png`      | seahorse/00372    | Mixed: main-cardioid interior (orange) tiled in **yellow CPU** + spiral + black exterior |
| `13-zoom-inside-deep.png`        | inside/00351      | Uniform interior: all black, single **cyan GPU** tile — GPU-routed, no outlier |

## Tooling notes / caveats

- **Filename-buffer bug** (`main.cpp:201`, `char imageFilePrefix[MAXFNAME-8]`,
  ~42 bytes; read by `fin >> imageFilePrefix`): image prefixes ≥42 chars truncate
  silently (the original `experiments/13-zoom-points/viz_misiurewicz/` → flat
  `viz_misiurewic00000.png`). Worked around by `cd`-ing into the experiment dir and
  using short `viz_<pt>/` prefixes — no recompile, binary unchanged. (Latent bug
  worth a one-line fix later: `char imageFilePrefix[256]` or `snprintf` bounds.)
- **viz=3 is NOT performance-neutral for trivial-region workloads.** misiurewicz
  compute was 2.37 s at viz=0 but **66.5 s at viz=3** — the per-region VizRect
  registration is a fixed cost that is negligible when regions are expensive
  (outside 16.95→16.89 s; inside ~76 s both ways) but dominates when regions are
  ~0.01 ms (misiurewicz has 3312 tiny GPU regions). CLAUDE.md's "perf-neutral"
  claim was validated only on expensive-region frames. **All timing numbers in
  this README are the clean viz=0 measurements.**
- seahorse outlier count varied 8 (viz=0) vs 7 (viz=3): one region sat right at
  the 5 s threshold — run-to-run noise, not a regime change.
