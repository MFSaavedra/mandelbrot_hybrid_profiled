# experiment 17 — Nsight Compute on the report-13 zoom points

Binary: `binary-v5-affinity` (`fc33e29`). Machine: i7-9750H + GTX 1660 Ti Max-Q
(Turing TU116), AC power. Tool: Nsight Compute. Report: `reports/17-ncu-zoom-points.tex`.

## Purpose

Verify the report-16 kernel result — coherent regions run divergence-free and
FP64-bound — on the **four report-13 zoom points** instead of report 16's
synthetic specs, and pick up **Misiurewicz**, which report 16 never covered.

## Same points as report 13, fewer frames

`specs/{outside,inside,misiurewicz,seahorse}.in` use the **exact** report-13
deep targets (same first frame, same deep-zoom rect and `maxIter`=10000), but
with the frame count reduced **100 → 24**. Warp execution efficiency, branch
efficiency, and FP64 pipe utilisation are *rates* — independent of how many
frames are rendered — so the reduction makes the `ncu` kernel-replay tractable
(~24 s GPU-only for the heaviest, `inside`, vs 103 s at 100 frames) without
changing the measured quantities. The four target centres are:

| regime | target c | character |
|---|---|---|
| outside | (1.0, 1.0) | escapes at iter 2 — coherent fast-escape exterior |
| inside | (−0.5, 0.0) | period-1 interior — every deep pixel hits `maxIter` |
| misiurewicz | (−0.77568377, 0.13646737) | self-similar spirals, exterior-dominated |
| seahorse | (−0.745428, 0.113009) | filaments + minibrots (the interior-outlier regime) |

## Capture (sudo) and parse (no sudo)

GPU counters are admin-restricted (`ERR_NVGPUCTRPERM`), so capture runs as root;
parsing a saved `.ncu-rep` does not.

```bash
sudo experiments/17-ncu-zoom-points/capture.sh         # ~8-12 min; chowns reports back
ncu --import experiments/17-ncu-zoom-points/ncu/inside_full.ncu-rep --page raw --csv
```

`capture.sh` runs GPU-only so the GPU sees every region type. Each regime is
captured twice: `--set full` (roofline / occupancy / stalls / FP64, with warp
efficiency included) and a cheap two-metric pass for the warp + branch
distribution on more kernels.

## Files

- `capture.sh` — the sudo capture script.
- `specs/*.in` — the four 24-frame report-13-point specs.
- `ncu/*.ncu-rep` (gitignored) + `ncu/gpu_state.csv` — captured reports.
- `analysis/` — extracted CSVs + `summary.csv`, produced after capture.
