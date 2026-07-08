# experiment 23 — exact Brent periodicity check in the GPU kernel `diverge()`

Branch `feat/gpu-periodicity` (commit `2abc909`, atop `main` @ `c4744c0`,
the binary-v6-periodicity lineage — the CPU-side check is already in both
binaries) vs baseline `main` (`c4744c0`). Hardware: i7-9750H + GTX 1660 Ti
Max-Q. Date: 2026-07-08. **On AC power**, battery *charging* during the
sweep (see calibration note below).

## The change

The device `diverge()` in `kernel.cu` gets the exact mirror of the CPU
check merged at `binary-v6-periodicity`: a saved orbit state refreshed at
doubling intervals (Brent), early `return MAXITER` on *exact* FP revisit.
Exactness holds with respect to the kernel's own (FMA-contracted)
arithmetic — the plain loop would have ground to MAXITER and returned the
same value — so the GPU's output is unchanged pixel-for-pixel *provided
nvcc contracts the arithmetic identically in both builds* (it does; see
identity gate below). A lane that detects a cycle exits and sits masked
while the rest of the warp finishes: the warp's cost drops from MAXITER to
the *slowest lane's* detection latency. This deliberately violates the
100% warp-execution-efficiency of reports 16/17 — that metric measured the
absence of early exits, which is exactly what the check introduces.

## Method

`ab.sh`: production `spec.in` (100 frames, 1920×1080, deep zoom to maxIter
10000), diffT=0.1, pixT=32768, quiet=1, **save=1** (Fig-11.15 methodology,
same as the experiment-21 headline). Three configs × 2 binaries × 3 reps,
binaries alternated back-to-back inside each config so thermal drift hits
both sides equally:

- `gpuonly` (1 thread, mode 1) — isolates the kernel change
- `hybrid` (12 threads, mode 1) — the production config
- `cpu12` (12 threads, mode 0) — control (CPU code byte-identical between
  binaries) and same-batch anchor for the worker-equivalence `k`

Baseline built from `main` (`c4744c0`) in a clean worktree.

## Calibration: rep 1 is a cold-start transient; the control proves it

The battery was charging throughout the sweep (extra thermal load), and
the sweep started on cold silicon. The CPU12 control — which *must* be a
wash, its pool compute is 408.6 s in both binaries to within 0.1 s — reads
**+6.4% → +1.2% → +0.07%** across reps 1→3: the batch reaches thermal
steady state during rep 2 and the control converges to zero. Because
`base` always runs first in each pair, rep 1's cold bias favours `base`,
i.e. the 3-rep means *understate* the improvement. Headline numbers below
are steady-state (reps 2–3); 3-rep means in parentheses. An earlier
same-day non-charging batch (`logs/smoke_hybrid.out`,
`logs/verify_identity.out`) ran ~1.5 s faster absolute at the same
relative delta (hybrid 27.93 → 22.44 s, −19.7%).

## Results (steady state = reps 2–3; 3-rep means in parentheses)

| config | baseline (s) | gpu-periodicity (s) | Δ wall | speed-up |
|---|---|---|---|---|
| GPU-only (1 thread) | 78.21 (77.83) | **40.07** (41.69) | **−48.8%** (−46.4%) | 1.95× |
| dGPU+11CPU | 29.83 (29.58) | **23.72** (24.34) | **−20.5%** (−17.7%) | 1.26× |
| CPU12 (control) | 35.78 (35.54) | 36.00 (36.43) | +0.6% (+2.5%) | — |

- **23.72 s is the best wall recorded for this project** (v6 hybrid best:
  29.8 s same-batch; 30.9 s in experiment 21's thermal era). The two gp
  hybrid steady-state reps agree to 1.4 ms.
- **Kernel-level, mix-controlled** (gpuonly, same 5,608 regions through the
  kernel in both binaries): total kernel 66.17 → 28.25 s = **2.34×**;
  real D→H transfer unchanged (~0.19 s total, ~35 µs/region).
- **Hybrid rebalance**: GPU regions 1,008 → 3,006 (18.0% → 53.6% of
  5,608), kernel avg 25.10 → 3.56 ms/region; CPU-pool compute
  311.6 → 249.6 s (−19.9%); busiest CPU worker 29.0 → 23.4 s ≈ wall in
  both binaries — the pool binds before and after, at 20% less cost.
- **GPU-thread cost structure inverts**: kernel occupancy 85% → 45% of the
  wall; the host side (commit + examine + queue, ~4.3 ms/region in both
  binaries) is now the lane's majority cost.
- Report 21's "expected loss" prediction for the GPU-side variant is
  **falsified**: warp cost is the slowest lane's detection latency, not
  32×MAXITER; lanes in a 16×16 block are spatially correlated, so the max
  is a small multiple of the ~900-iteration mean latency.

## Identity gate

`verify_identity.sh` renders the full production spec **GPU-only**
(1 thread — every region of every class through the modified kernel) with
both binaries and byte-compares all frames: **100/100 byte-identical**
(`logs/verify_identity.out`). This simultaneously confirms the exactness
argument and that nvcc's FMA contraction did not change between builds.

## Files

- `verify_identity.sh` — the identity gate
- `ab.sh` — the 3-config × 3-rep A/B; refuses to run off AC
- `results.csv` — the 18 sweep walls
- `logs/` — full per-run stderr/stdout (GPU summaries incl. memcpy totals,
  per-thread CPU summaries) + the earlier non-charging single-rep captures
- `plot_gpu_periodicity.py` — figures → `reports/img/gpuperiod_*.png` (gitignored)
