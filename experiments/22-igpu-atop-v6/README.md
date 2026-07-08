# experiment 22 — re-pricing the iGPU atop the periodicity check (v6)

Branch `feat/igpu-opencl` after merging `main` (`binary-v6-periodicity`
lineage): merge commit `833a791` = OpenCL third executor (`47c4372`..`2991dd9`)
+ exact Brent periodicity check in the CPU `diverge()` (`fa6d8ac`). One binary,
both changes. Hardware: i7-9750H + GTX 1660 Ti Max-Q + UHD 630
(`intel-compute-runtime-legacy1`). Date: 2026-07-08. **On AC power.**

## Question

Report 20 measured `dGPU+iGPU+10CPU` (mode 3) at **−23.2%** vs `dGPU+11CPU`
(mode 1) — but that was priced against a CPU pool spending 99% of its time
grinding interior pixels to MAXITER. Report 21's periodicity check eliminated
that grind (hybrid 57.6 → 30.9 s), which re-opens the question (report 21,
recommendation 2): does a second GPU lane still pay when the CPU pool it
relieves is ~5× faster?

## Method

`ab.sh`: production `spec.in` (100 frames, 1920×1080, deep zoom to maxIter
10000), diffT=0.1, pixT=32768, quiet=1, **save=1** (same methodology as the
exp-20 and exp-21 headlines). Two configs, one binary, modes alternated inside
each of 3 reps. Mode 1 doubles as the calibration arm.

## Result: the −23.2% collapses to a wash

| config | mean (s) | σ | per-rep |
|---|---|---|---|
| mode 1 `dGPU+11CPU` | 27.36 | 1.60 | 25.54, 28.55, 27.99 |
| mode 3 `dGPU+iGPU+10CPU` | 28.05 | 0.12 | 27.98, 28.19, 27.98 |

Δ = **+0.69 s (+2.5%)** on all-reps means — *slower* with the iGPU, but the
delta is 0.43σ of mode 1's spread. Rep 1 of mode 1 ran first (cold package,
25.54 s); excluding it, mode 1 = 28.27 s and the delta is **−0.19 s (−0.7%)**.
Either way: **no measurable wall benefit from the iGPU atop v6** (pre-v6 it
was −13.4 s).

## Mechanism (rep logs; all 6 runs = 5,608 leaves, decomposition unchanged)

- **The CPU pool binds in both modes.** Busiest CPU thread ≈ wall everywhere:
  24.6–28.0 s (mode 1), 27.2–27.7 s (mode 3).
- **What the iGPU strips, the lost worker gives back.** Mode 3's two GPUs
  absorb ~2,700 regions (dGPU 1,671–1,784 @ ~12.3 ms; iGPU 936–1,036 @
  ~21 ms; ~21 s kernel each), shrinking the pool's work from ~292 s (mode 1,
  r2/r3) to ~265 s — but over 10 threads instead of 11 that is the *same*
  26.4–26.5 s/thread. Post-periodicity there is no interior grind left to
  relieve: the pool's remaining work is the boundary band + detection latency,
  which the pool itself handles at GPU-competitive speed.
- Contrast with pre-v6 (report 20): the second lane stripped 187 s of
  MAXITER-grind off a 620 s pool — work the CPUs were pathologically slow at.
  That work class no longer exists.

## Calibration note

Mode 1 today (27.4–28.3 s steady-state) runs ~9% faster than exp-21's pc
hybrid (30.94 ± 0.56). Different thermal history: exp-21 interleaved 163 s
CPU12 baseline grinds before each hybrid run; today's sweep is six ~28 s runs
(and rep 1, coldest, is the fastest run of the day at 25.54 s). The internal
mode-1-vs-mode-3 comparison is alternated within reps, so it is unaffected.

## Conclusion

Do **not** merge `feat/igpu-opencl`: atop `binary-v6-periodicity` the iGPU
adds an OpenCL ICD + legacy-runtime dependency and a third executor for zero
wall time. The branch stays as a characterization (like reports 11/18), with
report 20 documenting the pre-v6 win and this experiment its post-v6 repeal.
Unmeasured residuals if the iGPU is ever revisited: the scanLine commit
cheapening and strict dGPU-priority-on-largest (report 20 §Analysis), and
mode 2 (`iGPU+11CPU`) as a budget config on machines without the dGPU.

## Files

- `ab.sh` — the sweep (single binary, modes 1/3 alternated, REPS=3)
- `results.csv` — `config,numThreads,gpuMode,rep,elapsed_s`
- `logs/m{1,3}.r{1..3}.{stdout,stderr}` — per-run thread/GPU summaries +
  `[total_elapsed_s]`

## Probe: recover the worker by oversubscribing (13 threads = 11 CPU + 2 lanes)

`probe_t13.sh`, 2 alternated reps vs the 12-thread mode-1 control, same day:

| config | reps (s) | mean |
|---|---|---|
| mode 1, 12 thr (control) | 26.28, 28.43 | 27.36 |
| mode 3, 13 thr | 29.58, 29.18 | **29.38 (+7.4%)** |

Worse than *both* 12-thread configs. The GPU lanes do not sleep while the
kernels run — the CUDA wait busy-spins (report 15) and both lanes do the
host-side commit — so 13 runnable threads on 12 logical CPUs just timeshare:
the pool's busiest thread rises 26.5–27.9 → 28.2–29.0 s and pool busy time
inflates 265 → ~305 s (descheduled time counts against the wall-clock
per-thread timers). The 11th worker's nominal ~27 s of capacity is consumed
by the contention it creates. Implication: any "one thread drives both GPUs"
or extra-worker scheme first requires the lanes to actually yield
(`cudaDeviceScheduleBlockingSync` one-liner, report 15) — and even then the
ceiling is one recovered worker ≈ −7–9%, on a pool that exp-22 shows binding
in every config.

## Probe 2: BlockingSync (`cudaDeviceScheduleBlockingSync`) + re-run

One-liner added at the top of `CUDAmemSetup()` (kernel.cu): the driver now
sleeps the dGPU lane on synchronous waits instead of busy-spinning — the
cheap confirmation report 15 called for. `probe_blocking.sh`, 3 configs
alternated, 2 reps:

| config | reps (s) | mean | pre-BlockingSync same-day |
|---|---|---|---|
| mode 1, 12 thr (control) | 24.33, 27.04 | 25.69 | 27.36 |
| mode 1, 13 thr (12 CPU + sleeping lane) | 26.30, 26.52 | 26.41 | — |
| mode 3, 13 thr (11 CPU + both lanes) | 28.63, 26.87 | 27.75 | 29.38 |

- **BlockingSync is safe and ~neutral-to-mildly-positive** (means shifted
  ~−1.6 s vs the pre-BS batch, but first-of-batch cold runs of 24.3–25.5 s
  put that inside the thermal confound). Confirms report 15's ceiling
  analysis; no reason to revert it.
- **The recovered worker is a wash** (m1 t13 26.41 vs m1 t12 25.69,
  overlapping spreads). Mechanism: per-region CPU cost rises 53→57 ms at 13
  threads — the 12th worker shares a physical core (SMT), adding only the
  ~0.3–0.4-core past-the-knee capacity report 15 priced, while the spinning
  lane it replaced was already SMT-friendly (PAUSE loop). The naive "free a
  logical CPU → −8%" does not materialize.
- **Mode 3 still loses everywhere** (27.75 vs 25.69/26.41): the iGPU fails
  to pay under all three schemes tried — worker swap (12 thr), spin
  oversubscription (13 thr), blocking oversubscription (13 thr + BS). The
  OpenCL lane's wait behaviour is unchanged and both lanes still carry the
  host-side commit; the iGPU also shares package power with the pool.
- **Implication for the single-thread dual-GPU multiplexer**: its entire
  prize is the recovered worker just measured at ≈ noise. Not worth the
  restructure on this machine; closes the idea alongside report 15.

Report: `reports/22-igpu-atop-v6.tex` (figures regenerated by
`plot_igpu22.py` into `reports/img/igpu22_*.png`, gitignored).
