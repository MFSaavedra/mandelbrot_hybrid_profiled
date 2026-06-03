# experiment 16 — Nsight Compute warp-divergence / GPU-metrics study

Binary: `binary-v5-affinity` (`fc33e29`). Machine: i7-9750H + GTX 1660 Ti Max-Q
(Turing TU116), AC power. Companion report: `reports/16-ncu-divergence.tex`.

## Why ncu (and why sudo)

`nsys` (used in reports 01/04/09) traces the timeline — CUDA API calls, NVTX
ranges, kernel/transfer durations — but reports **nothing about what happens
inside a kernel**. Warp execution efficiency, branch divergence, FP64 pipe
utilisation, occupancy, and the memory/compute roofline are *hardware
counters*, read only by **Nsight Compute (`ncu`)**.

Those counters are admin-restricted on the NVIDIA Linux driver
(`ERR_NVGPUCTRPERM`), so the **capture** step needs root. Parsing a saved
report does not, so the split is:

```bash
sudo experiments/16-ncu-divergence/capture.sh     # capture only (root)
# then, unprivileged:
ncu --import experiments/16-ncu-divergence/ncu/<name>_full.ncu-rep --page raw --csv
```

`capture.sh` chowns the `ncu/` reports back to `$SUDO_USER` when done.

## Design

GPU-only runs (`numThreads=1`, GPU on) so the single GPU thread processes
**every** region type. In production the affinity queue routes boundary regions
to the CPU pool, so a hybrid run never shows the GPU the divergent regions; to
*measure* divergence we must force them onto the GPU.

Four content regimes isolate the variables (`specs/`):

| regime | target | what the GPU sees | tests |
|---|---|---|---|
| `interior` | inside the main cardioid (c≈−0.5) | few giant coherent kernels, all pixels → MAXITER | zero divergence + FP64-bound |
| `exterior` | outside the set (c≈1+1i) | coherent, fast-escape | ~100 % efficiency at ~0 FP64 work |
| `boundary` | seahorse valley (c≈−0.7454+0.1130i) | many small split regions, mixed escape | the divergence case |
| `canonical` | the real `spec.in` minibrot zoom | production mix | tie-in to reports 12–13 |

Each regime is captured twice: `--set full` (all sections: SpeedOfLight /
Occupancy / WarpStateStats / InstructionStats / Memory & Compute workload) on a
bounded kernel count, and a cheap `--metrics` pass for warp + branch efficiency
on more kernels (better distribution). `_full` and `_warp` `.ncu-rep` per regime.

## Files

- `capture.sh` — the sudo capture script (idempotent, `-f` overwrites).
- `specs/{interior,exterior,boundary,canonical}.in` — 6-frame, 1920×1080 specs.
- `ncu/` — captured `*.ncu-rep` + `gpu_state.csv` (gitignored; regenerate via `capture.sh`).
- `analysis/` — extracted CSVs + summary produced unprivileged after capture.
