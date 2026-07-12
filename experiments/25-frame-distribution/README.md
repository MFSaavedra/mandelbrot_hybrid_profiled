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

## Files

- `verify_dist.sh` — the identity harness (run it from anywhere; takes NNODES)
- `results_identity.csv` — one row per check
- `bench_2node.sh` — the laptop+ivy benchmark driver (AC-gated; REPS env)
- `results_bench.csv` — one row per run, columns as above
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
