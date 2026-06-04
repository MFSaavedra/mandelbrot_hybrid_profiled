# experiment 18 — the all-MAXITER split rule vs diffThreshold

Branch: `examine/maxiter-split` atop `binary-v5-affinity` (`fc33e29`).
Report: `reports/18-maxiter-split.tex`.

## What changed (one line in `mandelregion.cpp::examine`)

```cpp
// before (diffThreshold relative-spread test):
if (maxIter - minIter < diffThresh * maxIter || pixelsX * pixelsY < pixelSizeThresh)
// after (binary "all 9 samples in-set"):
if (minIter == ownerFrame->MAXITER       || pixelsX * pixelsY < pixelSizeThresh)
```

Tests the hypothesis that `diffThreshold = 0.1` is operationally just "split unless
the region is interior (all stencil points reach MAXITER)". The pixel floor is
unchanged so boundary regions still terminate.

## Decomposition comparison (`results.csv`)

GPU-only leaf counts (deterministic — decomposition is thread-independent):

| spec | diffT=0.1 | all-MAXITER | Δ |
|---|---|---|---|
| canonical `spec.in` | 5608 | 5608 | **identical** |
| inside | 969 | 969 | identical |
| misiurewicz | 1050 | 1050 | identical |
| seahorse | 942 | 942 | identical |
| outside | 1206 | 1470 | +264 (+22%) |

The two rules are **byte-identical** on the production zoom and on three of the
four report-13 regimes. They diverge only on `outside` (uniform deep exterior),
where diffThreshold computes a big uniform-exterior region as one leaf while the
all-MAXITER rule splits it to the floor. `outside` wall is flat (~5.8 s both,
GPU-only) — the extra leaves are trivial iter-2 exterior.

## How to reproduce

```bash
git checkout examine/maxiter-split && make          # the new rule
git checkout main && make                            # diffThreshold (baseline)
./mandelHybrid <spec> 1 1 0.1 32768 1 0 | grep 'Regions computed on GPU'
```
Specs: `experiments/17-ncu-zoom-points/specs/*.in` and `spec.in`.
