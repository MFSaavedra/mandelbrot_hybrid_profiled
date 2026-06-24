# experiment 19 — 2-D diffThreshold × pixelSizeThresh sweep

Binary: `binary-v5-affinity` (`fc33e29`), on `main` (no code change — both
thresholds are CLI args). Machine: i7-9750H + GTX 1660 Ti Max-Q, AC power.

## Goal

Find the wall-optimal **(diffThreshold, pixelSizeThresh)** pair on the production
zoom. Prior work swept each knob alone (reports 06/07/10 for diffT; pixT never
swept); this sweeps the full grid.

## Grid

| knob | values | what it controls |
|---|---|---|
| `diffThreshold` | 0.05, 0.1, 0.2, 0.3, 0.5 | corner-spread tolerance (split if `maxIter−minIter ≥ diffT·maxIter`) |
| `pixelSizeThresh` | 2048, 8192, 32768, 131072, 524288 | the floor — regions below this many pixels are force-computed |

The pixT values move the quadtree floor across its natural depths: 60×34 (2048),
120×68 (8192), **240×135 (32768, default)**, 480×270 (131072), 960×540 (524288).

5×5 = 25 combinations × 3 reps = 75 runs. Each run is the canonical 100-frame
`spec.in`, hybrid (12 threads + GPU), `save=0` (pure compute — thresholds never
change the output image). ~50 s/run → ~60 min total.

## Run + analyze

```bash
experiments/19-threshold-sweep/sweep.sh                 # -> results.csv
experiments/19-threshold-sweep/analyze.py               # mean grid + optimum + heatmap.png
# override: DIFFTS="0.08 0.1 0.12" PIXTS="16384 32768 65536" REPS=5 sweep.sh
```

`sweep.sh` sweeps the whole grid once per rep (rep outer, grid inner) so thermal
drift spreads across cells rather than piling on one. AC power is checked.

## Files

- `sweep.sh` — the 2-D sweep driver.
- `analyze.py` — mean-wall grid, optimal pair, `heatmap.png`.
- `results.csv` — raw per-run wall times (`diffT,pixT,rep,wall_s`).
- `heatmap.png` — gitignored (regenerate via `analyze.py`).
