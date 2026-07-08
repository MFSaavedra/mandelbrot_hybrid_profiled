# experiment 21 — exact Brent periodicity check in the CPU `diverge()`

Branch `feat/periodicity-check` (commit `fa6d8ac`, atop `main` @ `3061307`,
binary-v5-affinity lineage) vs baseline `main` (`3061307`). Hardware:
i7-9750H + GTX 1660 Ti Max-Q. Date: 2026-07-07. **On AC power** (calibration:
baseline CPU12 reproduces experiment 20's 161.64 s within 0.7–3%; baseline
hybrid reproduces 57.72 s within 1%).

## The change

`MandelRegion::diverge()` keeps a saved orbit state refreshed at doubling
intervals (Brent's cycle detection) and returns MAXITER as soon as the FP
state *exactly* revisits the saved state: the iteration map is deterministic,
so an exactly-repeating orbit can never escape — the plain loop would have
ground to MAXITER and returned the same value. Exact equality (no epsilon)
is what makes the transform *exact*: escaping orbits never exactly repeat,
so every pixel's return value — and therefore both the image and the
decomposition — is unchanged. CPU branch only, per report 16's caveat (the
GPU's interior kernels are 100% warp-coherent and FP64-bound; an early-out
would introduce divergence there).

Motivation (from the leaf-purity analysis of `dist_aff.stderr`, experiment
12): 99.5% of the 495 G iterations of the production run are interior pixels
grinding to the frame's MAXITER; the CPU pool — the binding wall — spends
99.0% of its 545 s on them.

## Method

`ab.sh`: production `spec.in` (100 frames, 1920×1080, deep zoom to maxIter
10000), diffT=0.1, pixT=32768, quiet=1, **save=1** (Fig-11.15 methodology,
same as the experiment-20 headline). Configs: CPU12 (mode 0) and dGPU+11CPU
(mode 1). 3 reps, binaries alternated inside each rep so thermal drift hits
both sides equally. Baseline binary built from `main` in a clean worktree.

## Results (3 reps, means; per-rep spread < 2.5%)

| config | baseline (s) | periodicity (s) | Δ wall | speed-up |
|---|---|---|---|---|
| CPU12 (mode 0) | 164.43 | **35.21** | **−78.6%** | 4.67× |
| dGPU+11CPU (mode 1) | 57.64 | **30.94** | **−46.3%** | 1.86× |

- **First work-reduction lever, and it dwarfs every scheduling lever**: the
  hybrid wall drops 46%, vs −3.6% for GPU affinity (report 12) and −23% for
  adding the iGPU (report 20). 30.94 s is the new best hybrid time, −38% below
  experiment 20's two-GPU mode 3 (44.32 s) *using one GPU*.
- **Work eliminated, not moved**: CPU12 pool compute 1,917 → 402 s (−79%).
  Single pure-interior 1920×1080 region (zoom target at t=1.0, one thread):
  67.27 → 5.98 s (11.2×), i.e. mean detection latency ~900 iterations-
  equivalent (incl. check overhead) vs 10,000 for the grind.
- **The hybrid rebalances itself** (rep-1 logs): CPU threads, now ~5× faster
  on interior content, pull 4,313 of the 5,608 regions (baseline: 2,148); the
  GPU keeps the 1,295 largest (avg 12.4 → 19.6 ms/region, kernel total
  42.8 → 25.4 s). Busiest CPU thread 29.7 s ≈ wall — the CPU pool is still
  (mildly) the binding resource; what remains on it is detection latency plus
  the truly-escaping high-iteration boundary band, which no exact early-out
  can skip.
- **Decomposition unchanged**: all 12 runs produce exactly 5,608 leaves
  (the known diffT=0.1 decomposition; the check returns identical values at
  the stencil sample points).

## Output verification (`verify_identity.sh`)

Full production spec rendered CPU-only (mode 0: every pixel through the
modified `diverge()`) with both binaries, all 100 frames byte-compared with
`cmp`: **100/100 byte-identical** (mode-0 output is deterministic — pixel
values never depend on thread count or decomposition). The GPU path is
untouched by the branch, so this covers the entire change.

## Files

- `ab.sh` — the A/B sweep (BASE_BIN=<main binary> required)
- `results.csv` — `binary,config,numThreads,gpuMode,rep,elapsed_s`
- `logs/{base,pc}.{cpu12,hybrid}.r{1..3}.stderr` — per-run logs (thread/GPU
  summaries; `[total_elapsed_s]`)
- `verify_identity.sh` — byte-identity check (renders ~3.5 min, then cmp)

## Note: spec-prefix buffer overflow found during smoke testing

`main.cpp` reads the image prefix with `fin >> imageFilePrefix` into
`char imageFilePrefix[MAXFNAME-8]` (= 42 bytes, `mandelframe.h`). A spec
whose prefix exceeds 41 chars silently smashes the stack (observed:
`maxIterations[]` corrupted to garbage, runs computing nonsense). Production
specs use short prefixes (`img`), so no historical measurement is affected.
Work around with short prefixes + `cd`; a bounds fix should land separately.
