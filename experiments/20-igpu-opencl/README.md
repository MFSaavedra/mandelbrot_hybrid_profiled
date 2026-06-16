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
(57.7 vs 81.7 s). Output is byte-identical to the CUDA path (verified separately).

Open question for a fuller sweep: both GPUs pull from the largest end, so the weak
iGPU can grab the single biggest outlier the fast dGPU would clear sooner — a strict
dGPU-priority-on-largest refinement might widen the win.
