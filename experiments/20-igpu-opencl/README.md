# experiment 20 — integrated-GPU (OpenCL) backend, headline A/B

Branch `feat/igpu-opencl` (atop `binary-v5-affinity`, `fc33e29` / parent commit
`c44411c`); iGPU backend `oclkernel.cpp/.h`, working tree (uncommitted at capture).
Hardware: i7-9750H + GTX 1660 Ti Max-Q (dGPU/CUDA) + Intel UHD 630 (iGPU/OpenCL,
legacy `intel-compute-runtime`). Date: 2026-06-15. **On AC power.**

## Question

Does adding the integrated GPU as a third executor move the end-to-end wall?
Best-config disjoint A/B on one binary in one thermal state, 12 threads fixed,
only the backend mode (arg3 = `gpuMode`) varies:

| label | numThreads | gpuMode | composition |
|---|---|---|---|
| CPU12 | 12 | 0 | 12 CPU workers, no accelerator (floor) |
| dGPU+11CPU | 12 | 1 | dGPU + 11 CPU — current best (report 12) |
| dGPU+iGPU+10CPU | 12 | 3 | dGPU + iGPU + 10 CPU — iGPU added (displaces 1 CPU worker) |
| iGPU+11CPU | 12 | 2 | iGPU + 11 CPU — iGPU instead of the dGPU |

The mode 1 vs 3 contrast isolates *is the iGPU worth more than the CPU worker it
replaces?*; mode 1 vs 2 contrasts the two accelerators head-to-head.

## Method

Mirrors `scripts/sweep_fig1115.sh`: full `spec.in` (100 frames, 1920×1080, deep
zoom to maxIter 10000), `quiet=1`, `save=1`, **`diffT=0.1`** (the sweet spot,
reports 06/12), `pixT=32768`, 3 reps/config, scratch PNGs cleaned between runs.
The discrete-GPU baseline is re-measured on this binary (not reused from
experiments/05/12) so the comparison is thermally disjoint.

## Run

```bash
experiments/20-igpu-opencl/sweep_igpu.sh        # ~15 min on AC
```

Writes `results.csv` (`label,numThreads,gpuMode,rep,elapsed_s`) and per-run
`logs/<label>.r<n>.log.{stdout,stderr}` (the stderr holds `[total_elapsed_s]`
plus the per-backend region/kernel summaries).

Figures (`reports/img/igpu_{wall,work}.png`, gitignored — regenerated on demand)
and the report:

```bash
python3 experiments/20-igpu-opencl/plot_igpu.py   # -> reports/img/igpu_*.png
scripts/build_report.sh 20-igpu-opencl            # -> reports/20-igpu-opencl.pdf
```

## Results (3 reps each, means)

| config | mode | wall (s) | vs CPU12 | vs dGPU best |
|---|---|---|---|---|
| CPU12 | 0 | 161.64 | — | — |
| iGPU+11CPU | 2 | 81.72 | −49.4% | +41.6% |
| dGPU+11CPU | 1 | 57.72 | −64.3% | baseline |
| **dGPU+iGPU+10CPU** | **3** | **44.32** | **−72.6%** | **−23.2%** |

**Adding the iGPU is a −23.2% wall win** at the best config (57.72→44.32 s), and it
wins while *giving up a CPU worker* (10 vs 11) — so the iGPU is worth far more than
the CPU thread it displaces. Variance is tiny (44.14/44.62/44.19), well outside noise.

**This refutes the pre-measurement prediction** (CLAUDE branch note: "marginal-to-
negative; FP64-weak iGPU thermally coupled to the CPU pool, the binding wall"). The
binding constraint under these conditions (diffT=0.1, save=1, deep zoom to maxIter
10000) is not the CPU pool alone but the **serial tail of big coherent interior
regions only a GPU runs fast**. With one GPU they queue on the dGPU (~91% saturated,
report 12); the iGPU pulls from the same largest end and runs them in **parallel on a
second accelerator**. Mode-3 region split (rep 1): dGPU 2915 regions / 34.4 s kernel
(11.8 ms/region), iGPU 1422 regions / 30.4 s kernel (21.4 ms/region) — the iGPU took
33% of the GPU regions at ~1.8× the dGPU's per-region cost.

Secondary findings: the iGPU *alone* as the accelerator (mode 2) still nearly halves
the CPU-only floor (161.6→81.7 s), absorbing the interior outliers (0 CPU outliers in
every accelerated mode vs 4 in CPU12). The dGPU is the stronger single accelerator
(57.7 vs 81.7 s). Output equivalence to the CUDA path is verified below (§Output
verification): equivalent to within ±1-iteration FP rounding, **not** byte-identical
on deep frames.

Open question for a fuller sweep: both GPUs pull from the largest end, so the weak
iGPU can grab the single biggest outlier the fast dGPU would clear sooner — a strict
dGPU-priority-on-largest refinement might widen the win.

## save=0 cross-check (`ab_save0.csv`)

The headline table above uses `save=1` (PNG writes, Fig-11.15 methodology), which
raises absolute times ~5 s vs the `save=0` numbers in report 12. To confirm the win
is not a `save`/thermal artifact and to reconcile with report 12's 49.86 s, a clean
`save=0` A/B (alternating mode 1 / mode 3, 3 reps, diffT=0.1, 12 threads, on AC):

| config (save=0) | mean (s) | range |
|---|---|---|
| dGPU+11CPU (mode 1) | 49.24 | 48.09–50.11 |
| dGPU+iGPU+10CPU (mode 3) | 38.19 | 37.94–38.54 |
| Δ | **−11.05 (−22.4%)** | disjoint |

Mode 1 at `save=0` (49.24 s) reproduces report 12's 49.86 s (same binary's CUDA
path) — so report 20's higher 57.72 s baseline is `save=1` + thermal, not a code
regression. The iGPU win holds at `save=0` (−22.4%, vs −23.2% at `save=1`), and
mode 3's 38.19 s is a new best, −23.4% below report 12's prior best.

## Output verification (`verify_output.sh` / `verify_output.csv`)

Full-run output-identity check, 2026-07-07 (correctness only — power state is
irrelevant, run on battery): the whole 100-frame spec rendered three times, every
pixel computed by exactly one backend per render (`mode 1`/1 thread = all-CUDA,
`mode 2`/1 thread = all-OpenCL, `mode 0` = all-CPU), then compared per frame
(`cmp` for byte identity, ImageMagick `AE` for differing-pixel counts). This
**corrects the earlier single-frame check**, which had been overgeneralised to
"byte-identical":

| comparison | result |
|---|---|
| dGPU vs iGPU, byte-identical frames | **2**/100 (frames 0–1 only) |
| dGPU vs iGPU, differing px | 104,355 = **0.050%** of 207.36M; grows 2 px (f2) → **3,089 px = 0.149%** (f88) |
| dGPU vs CPU, differing px | 16,731,040 = **8.07%** (up to 19.7% on f98) |
| iGPU vs CPU, differing px | 16,730,971 — tracks dGPU-vs-CPU within a few px/frame |

The GPU–GPU disagreement grows monotonically with zoom depth (`maxIter`) — the
signature of the two kernel compilers making different FMA-contraction choices,
flipping `diverge()` ±1 iteration for orbits near the escape threshold. Both GPUs
sit on the same side of the much larger (~8%) GPU-vs-CPU contraction gap (host
`g++ -O2` emits no FMA). So: **the iGPU path is exactly as correct as the CUDA
path** (same ±1-iteration rounding class, no iGPU-specific artifact), but bytewise
GPU–GPU identity holds only on shallow frames. Re-run: `verify_output.sh [workdir]`
(renders ~20 min; `RENDER=0` re-runs just the comparison).
