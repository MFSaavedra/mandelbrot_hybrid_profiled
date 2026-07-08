# experiment 24 — ncu re-baseline after the GPU-side periodicity check

Binary: `binary-v7-gpu-periodicity` (`2abc909`, on `main` since `41b6c2e`).
Machine: i7-9750H + GTX 1660 Ti Max-Q (Turing TU116), AC power.
Companion report: `reports/24-ncu-gpu-periodicity.tex`.

## Why

Reports 16/17 pinned the pre-check kernel at **100.0% warp execution
efficiency** and **85.6% FP64 pipe** on interior content, and report 21
cited exactly those numbers to defer the GPU-side check ("expect a loss").
Report 23 measured the check anyway: kernel 2.34×, hybrid −20.5%,
byte-identical — the efficiency metric was measuring the waste (every lane
grinding provably-useless iterations to MAXITER in perfect lockstep). This
experiment puts the post-check counters on record so the stale 100% is
replaced by measured numbers, with the mechanism visible: early-exit
divergence is the *cost side* of a trade that report 23 already showed wins
on time.

## Method

Identical to experiment 16 — same four content-regime specs (reused
verbatim from `experiments/16-ncu-divergence/specs/`), same GPU-only runs
(`1 1 0.1 32768 1 0`; the single GPU thread sees every region class), same
captures per regime (`--set full` on 6/6/60/60 kernels; cheap warp+branch
`--metrics` pass on 6/6/300/300), same ncu. **Baseline = experiment 16's
tracked `analysis/summary.csv`**: the device `diverge()` is unchanged
between `binary-v5-affinity` (which exp 16 captured) and v6, so that
capture *is* the immediately-pre-v7 kernel. Ratio/percentage metrics
(warp/branch efficiency, pipe %, SOL %, occupancy) are clock-independent
and comparable across sessions; kernel *durations* carry the usual
thermal/clock caveat and are quoted as indicative only.

```bash
sudo experiments/24-ncu-gpu-periodicity/capture.sh   # counters are admin-gated
python3 experiments/24-ncu-gpu-periodicity/analyze.py  # unprivileged
```

`analyze.py` reproduces experiment 16's aggregation exactly (validated
in-session against its tracked summary on all four regimes: simple mean
over captured kernels of the same raw-page columns). One recording slip in
the reference: exp-16's `exterior` `dur_ms` value is in µs (614.869);
this experiment's summary emits true ms everywhere.

## Files

- `capture.sh` — sudo capture (idempotent, `-f` overwrites); records
  `ncu/gpu_state.csv`
- `analyze.py` — export via `ncu --import` + aggregate to
  `analysis/summary.csv` (full-set means) and `analysis/warp_dist.csv`
  (warp-efficiency distribution over the larger `_warp` captures), and
  print the v6-vs-v7 side-by-side
- `ncu/*.ncu-rep` + `analysis/*_full.csv`/`*_warp.csv` — gitignored
  (large); regenerate via the two commands above
- `analysis/summary.csv`, `analysis/warp_dist.csv` — tracked results
