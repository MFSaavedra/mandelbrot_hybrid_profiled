# Experiment 25 — distributed frame rendering: PoC identity validation

**Branch / binary:** `feat/frame-distribution` commit `aae49fb` (tag `binary-frame-dist`), atop `main` `8acde54`
(= `binary-v7-gpu-periodicity` code state).
**Date:** 2026-07-12.
**Machine:** i7-9750H + GTX 1660 Ti Max-Q (the usual laptop); identity does not
depend on timing, so no AC-power constraint applies.

## What this is

Proof-of-concept validation for the two-level distribution hierarchy: frames
are distributed **across** nodes (static cyclic / block-cyclic, selected by the
`DIST_NODES` / `DIST_RANK` / `DIST_BLOCK` environment variables in `main.cpp`),
while the region-level work queue stays **intra-node** and untouched. Frame `i`
is rendered iff `(i / DIST_BLOCK) % DIST_NODES == DIST_RANK`. Every node reads
the same `spec.in`; the corner interpolation still accumulates over *all* frame
indices (skipping the accumulation would change the FP corners), and output
PNGs keep the global frame index so collection is a plain union.

Orchestration: `scripts/dist_frames.sh` (GNU parallel over SSH; host `":"` =
this machine without SSH). The manager-worker frame dispenser is deliberately
not built; the seam is the ownership predicate in `main.cpp`'s frame loop.

## Method

`verify_dist.sh 3` — mirrors `experiments/21`/`23`'s `verify_identity.sh`:
3 simulated nodes on this machine (hosts `":"`, `JOBS=1` to serialize), union
of their PNGs `cmp`'d frame-by-frame against a single-process reference run of
the same binary, spec (production 100-frame 1920×1080 zoom), and config.
Executor is pinned in every check — CPU-only (`12 0`) and GPU-only (`1 1`) —
because CPU and CUDA FP64 differ in last-ulp rounding on deep-frame boundary
pixels (report 20: ~8% of pixels GPU-vs-CPU), which makes *hybrid* output
timing-dependent; that nondeterminism is pre-existing single-node behaviour,
not introduced by the distribution.

Checks:

1. CPU-only, pure cyclic (`DIST_BLOCK=1`) — ranks own 34/33/33 frames
2. GPU-only, pure cyclic (`DIST_BLOCK=1`)
3. CPU-only, block-cyclic (`DIST_BLOCK=8`)

A 4-frame smoke test additionally covered contiguous-block assignment
(`DIST_BLOCK >= ceil(frames/N)`), rank/env validation (bad `DIST_RANK`
rejected, exit 1), and hybrid-config file coverage.

## Results — `results_identity.csv`

All checks **PASS**: every collected union covers 100/100 frames and every
frame is byte-identical (`cmp`) to the single-process reference.

| Check | Collected | Byte-identical |
|---|---|---|
| CPU-only cyclic (block=1) | 100/100 | 100/100 |
| GPU-only cyclic (block=1) | 100/100 | 100/100 |
| CPU-only block-cyclic (block=8) | 100/100 | 100/100 |

Side observation (not a timed A/B): the three CPU-only cyclic ranks reported
11.9 / 11.2 / 10.9 s of wall each, and the GPU-only cyclic ranks
14.2 / 13.5 / 13.5 s — the cyclic interleave splits the 100-frame workload
(~35 s CPU12 / ~40 s GPU-only single-node, reports 21/23) into three
near-equal thirds, exactly the cost-spread absorption the cyclic assignment
is for (contiguous blocks would strand the deep-zoom third).

## Cross-machine check (laptop + ivy, 2026-07-12)

First run on real hardware: 2-node cyclic over SSH, production spec, CPU-only
(pinned executor, as above), compared against the same single-process
reference. Node 1 is **ivy** (i7-4702MQ, 4C/8T Haswell, GT 750M = Kepler
sm_30 with no modern-CUDA support), running the `make GPU=0` CPU-only build
(g++ 16, `kernel_stub.cpp`, `-DNO_NVTX`) at commit `52c40c4`:

```
# hosts.txt
: - 12 0 0.1 32768 1 1
ivy /home/lynx/box/mandelbrot_hybrid_profiled 8 0 0.1 32768 1 1
```

Result: **100/100 frames byte-identical** (laptop's 50 even frames 13.2 s,
ivy's 50 odd frames 41.9 s). Cross-compiler and cross-microarchitecture
identity (g++-15/Coffee Lake vs g++ 16/Haswell) holds because the build uses
plain `-O2` with no `-march`: both compilers emit baseline x86-64 scalar SSE2
FP64 with no FMA contraction, so `diverge()` computes bit-equal orbits.
(Adding `-march=native` would break this — FMA contraction changes last-ulp
rounding, the same mechanism as the CPU-vs-CUDA differences.)

The 3.2× per-share wall gap between the two nodes is the heterogeneous-node
imbalance the static assignment cannot absorb — the concrete motivation for
the manager-worker frame dispenser (seam in `main.cpp`) and for weighted
static shares in the report-25 study.

## 2-node benchmark (laptop + ivy, 2026-07-12) — `results_bench.csv`

`bench_2node.sh` (both machines on AC, verified by the script): production
spec, 3 reps, save=1, laptop hybrid `12 1` + ivy CPU-only `8 0`. `wall_e2e_s`
times the whole `dist_frames.sh` invocation (ship + ranks + collect);
`laptop_s`/`ivy_s` are each node's own `[total_elapsed_s]`.

| Config (means) | e2e wall | laptop rank | ivy rank | vs baseline |
|---|---|---|---|---|
| `laptop_hybrid` baseline | **18.75 s** | 18.75 | — | — |
| `2node_cyclic` (block=1) | 46.90 s | 10.33 | 42.67 | +150% |
| `2node_bc8` (block=8) | 45.00 s | 9.91 | 41.54 | +140% |
| `2node_block` (ivy gets deep half) | 65.12 s | 5.70 | 61.55 | +247% |
| `2node_block_rev` (ivy gets cheap half) | **28.24 s** | 12.52 | 24.86 | +51% |
| `ivy_solo` (1 rep) | 85.48 s | — | 85.48 | — |

**Every 2-node static config loses to the laptop alone.** The mechanism is
pure throughput asymmetry: per-frame cost is 0.188 s on the laptop vs 0.855 s
on ivy (4.56×), so any equal-share assignment strands the wall on ivy's rank
(cyclic: ivy 42.7 s vs laptop 10.3 s — the laptop idles 77% of the run).
Block orientation acts as a crude 70/30 weighting — the zoom's cost trend
puts ~70% of total work in the deep half (laptop halves: 12.5 vs 5.7 s; ivy
halves: 61.5 vs 24.9 s, both ≈70:30) — which is why `block_rev` (fast node
takes the deep half) is the best 2-node config and `block` is the worst.

The ceiling is analytic and already visible: combined throughput 1/18.75 +
1/85.5 s⁻¹ gives an ideal (perfectly weighted, zero-overhead) 2-node wall of
**15.4 s = 1.22×** — ivy can contribute at most 18% of the work. Measured
orchestration overhead (e2e − max rank: ship + SSH + PNG collection) is
3.3–4.2 s, which would consume most of that ideal 3.4 s gain. Conclusions:
(1) static equal-share distribution requires near-homogeneous nodes; (2) for
this pair, the manager-worker dispenser or weighted shares are *necessary
but barely sufficient* — the honest use of ivy is capacity (freeing the
laptop) rather than latency; (3) cyclic vs block-cyclic(8) is a wash when a
slow node binds (46.9 vs 45.0 s) — the interleave only matters between
comparable nodes, which is what the report-25 study should use.

Side note: today's laptop baseline (18.75±0.3 s) is 21% faster than report
23's 23.72 s steady state on the same commit lineage — same-day internal
comparisons above are unaffected, but absolute walls are not comparable
across reports without a same-day baseline (thermals/driver drift).

## Weighted static + dynamic work stealing (post-report-25, 2026-07-12) — `results_dynamic.csv`

Two balancing modes added after report 25's equal-share benchmark
(`ffe61aa`+): **weighted-random static** (`DIST_WEIGHTS`/`DIST_SEED` in the
binary: owner(i) = rank whose cumulative-weight interval contains
`splitmix64(seed+i) mod W` — communication-free, cross-machine
deterministic) and **coordinator-driven work stealing**
(`scripts/dist_dynamic.sh`: seeded shuffle dealt proportionally into
per-rank bags; per-node drivers dispatch guided chunks — half the remaining
bag, capped at `KCAP·w_r` frames, min `KMIN` — via the new `DIST_FRAMES`
explicit-list mode; a node whose bag empties steals half the richest
undispatched tail under one flock).

**Identity (production spec, laptop+ivy, CPU-pinned):** weighted static
5:1 → 85/15 split, 100/100 byte-identical; dynamic 5:1 → 100/100
byte-identical *with a live corrective steal*: in the CPU-pinned config the
laptop is only ~2.4× ivy (no GPU), so 5:1 overloaded it and ivy stole 5
frames back — miscalibration correction observed in the wild.

**Benchmark** (`bench_dynamic.sh`, 3 reps, AC, hybrid laptop + CPU-only ivy):

| Config (means) | e2e wall | laptop | ivy | shares (L/I) |
|---|---|---|---|---|
| `laptop_hybrid` baseline | 21.06 ± 2.73 s | — | — | 100/— |
| `weighted_static` 5:1 | 22.93 ± 0.11 | 17.93 | 19.85 | 85/15 |
| `dyn_51` (right weights) | 22.01 ± 0.56 | 18.96 busy | 16.31 busy | 83/17, 0 stolen |
| `dyn_11` (wrong weights) | 28.18 ± 0.37 | 18.93 busy | 22.15 busy | 79/21, 29 stolen |

Findings:

1. **Right weights reach baseline parity, as report 25's ceiling said they
   must — and no more.** Weighted static (22.93 s) and dynamic (22.01 s)
   both land within this batch's noisy baseline (18.87/24.12/20.18 s —
   sustained benching thermals; earlier same-day batches measured
   18.75±0.34 and 19.20±0.55). The 1.22× ideal minus 3.4–4.2 s orchestration
   overhead is a wash, and that is what was measured. The win vs *equal*
   shares is large: 46.90 → ~22 s.
2. **Stealing recovers 72% of a miscalibration autonomously.** Wrong (1:1)
   weights: static equal-share 46.90 s → uncapped stealing 31.27 s →
   weight-capped stealing **28.18 s**. The first run exposed the binding
   defect: a slow node's half-bag chunk (25 frames = 27.4 s) is in flight
   and unstealable. The fix caps chunks at `KCAP·w_r` frames — in-flight
   exposure is k/rate with rate ∝ weight, so the weight-proportional cap
   bounds every node's unstealable exposure at the same wall-time budget
   (ivy busy fell 28.0 → 22.2 s; the laptop's stolen haul rose 25 → 29
   frames). The residual +28% vs baseline is ivy's still-inflated share
   plus per-chunk dispatch overhead.
3. **With calibrated weights, stealing never fires** (dyn_51: 0 stolen,
   both nodes finish within ~2 s) — it is pure insurance, costing only the
   chunked dispatch (+0–1 s vs weighted static, within noise).
4. **Random assignment has real variance at small shares:** ivy's fixed
   (seed-1234) 15-frame draw costs 19.9 s = 1.32 s/frame vs its 0.855
   overall mean — one draw from a 32×-spread population. A deterministic
   weighted *interleave* would cut that variance; with stealing on top it
   is moot (the dynamic tail absorbs it).

## Files

- `verify_dist.sh` — the identity harness (run it from anywhere; takes NNODES)
- `results_identity.csv` — one row per check
- `bench_2node.sh` — the laptop+ivy equal-share benchmark driver (AC-gated; REPS env)
- `results_bench.csv` — one row per run, columns as above
- `bench_dynamic.sh` / `results_dynamic.csv` — weighted-static vs dynamic-stealing A/B
- `collect/`, `verify_work/`, `bench/` — gitignored scratch (PNG trees, logs)

## Regenerate

```bash
make
experiments/25-frame-distribution/verify_dist.sh 3
```

## Pending

The actual multi-node measurement — block vs cyclic vs block-cyclic across
real machines (report 13's static-LB earmark) — lands as report `25` when a
second machine with the CUDA/Qt stack is available. `scripts/dist_frames.sh`
already takes a real hosts file; the binary must be pre-built at `RDIR` on
every host.
