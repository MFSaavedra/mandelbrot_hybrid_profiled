# Experiment 25 — distributed frame rendering: PoC identity validation

**Branch / binary:** `feat/frame-distribution` (tag `binary-frame-dist`), atop `main` `8acde54`
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

## Files

- `verify_dist.sh` — the identity harness (run it from anywhere; takes NNODES)
- `results_identity.csv` — one row per check
- `collect/`, `verify_work/` — gitignored scratch (PNG trees, per-rank logs)

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
