# 10 — diffThreshold sweep on the 9-point binary

Binary: `examine/9-points-sampling` / tag `binary-v3-9point` (`ab47540`).
Purpose: confirm the optimal `diffThreshold` did not move when the uniformity
test changed from 4 corners to a 9-point stencil.

Machine: i7-9750H (6c/12t) + GTX 1660 Ti Max-Q. Date: 2026-05-31.
Config: hybrid (12 threads, GPU on), 100 frames 1920×1080, `pixT=32768`,
`quiet=1`, `save=0`. Sweep `diffT ∈ {0.05,0.10,0.20,0.30,0.50}` × 3 reps.

```bash
for d in 0.05 0.10 0.20 0.30 0.50; do for r in 1 2 3; do
  ./mandelHybrid spec.in 12 1 $d 32768 1 0 ; done ; done   # literal $d (single token; zsh-safe)
```

## Results (mean of 3 reps)

| diffT | wall mean (s) | min–max | leaves |
|------:|--------------:|:--------|-------:|
| 0.05  | 51.08 | 49.93–52.22 | 5,608 |
| 0.10  | 52.09 | 51.55–52.52 | 5,608 |
| 0.20  | 51.66 | 51.45–51.78 | 5,605 |
| 0.30  | 52.58 | 52.31–52.99 | 5,581 |
| 0.50  | 52.69 | 52.51–52.96 | 5,491 |

## Conclusion

**The optimal diffThreshold did not move — 0.1 remains a sound default.**

- Leaf count **saturates at ~5,608 by `diffT=0.10`** (0.05 gives the identical
  5,608 — the `pixT=32768` floor is reached, so lower `diffT` adds no splits).
- `diffT ∈ {0.05, 0.10, 0.20}` are **statistically indistinguishable** (~51–52 s,
  ranges overlap within the ~2–3% run-to-run noise). `0.30` (+~1 s) and `0.50`
  (+~1.6 s) are measurably worse as splitting drops off (5,581 / 5,491 leaves).
- `0.05` has the lowest mean but the widest spread (2.3 s) — noisy, not reliably
  faster; `0.20` is the least noisy. `0.10` sits squarely in the flat optimum.

This mirrors the 4-point finding (report 06): the optimum is a **plateau**
[0.05–0.20], with 0.1 the predictable sweet spot (0.05 too noisy, ≥0.3 loses
splits). The 9-point stencil does not shift it, consistent with report 09:
wall is throughput-bound and total compute is conserved, so `diffThreshold`
only affects partition granularity, which saturates at the pixel floor by ~0.1
for both samplers. **No change to the recommended default.**

Note: hybrid config only (the operating point the default targets). CPU-only /
GPU-only were not swept.
