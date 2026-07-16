# 28 — DLT retrodiction of the yeco frame-distribution campaign

Analysis experiment for `reports/28-dlt-analysis.tex`: feeds the measured
report-27 parameters into DLTlib's collection-aware divisible-load solver
(`Network::SolveImage`, G. Barlas, GPL-3.0) and compares the closed-form
optima against the empirical weights, the measured walls, and the
work-stealer's converged splits. **No new renderer measurements** — every
input number comes from `experiments/25-frame-distribution/`
(`results_bench_yeco.csv`, `results_dynamic_yeco{,_scp}.csv`, and the
report-27 transfer micro-A/B), produced at code state `c5bd3ed` / data
commit `f310656`.

## Provenance

- **DLTlib**: sibling directory of the project root
  (`../../../DLTlib`, i.e. `Chapter_11_Loadbalancing/DLTlib`) — the
  Chapter-11 library from the course text (Barlas, *Multicore and GPU
  Programming*, 2nd ed., §11.4). Not part of this repo; not modified.
  `run.sh` copies it into `build/` (gitignored) and patches one line there
  (`random.c`'s hard-coded `/papers/cpp_lib/random.h` include → local
  `random.h`) before compiling.
- **Model mapping**: distribution factor `a = 0` (every node reads the same
  tiny `spec.in`; shipping it is per-run constant), collection factor
  `c = 1` with link cost `l` = measured per-frame PNG collection time.
  Parameters (s/frame): `p_laptop = 0.2102`, `p_yeco = 0.0515`,
  `l_scp = 0.0835` (6.60 s / 79 files), `l_tar = 0.0122` (0.96 s / 79
  files); `L = 100` frames; 1.2 s fixed orchestration added outside the
  model.

## Files

- `mandel_dlt.cpp` — the driver (unity-includes `dltlib.cpp`, DLTdemo
  style). Default mode prints the analysis table; `sweep` mode emits the
  optimum-vs-`l` CSV. The 3-node outlook row uses a hand closed form
  because `SolveImage` assumes one link speed per parent
  (`ImageAggregate` reads `temp->link[0]` for every child) — a mixed
  WAN+LAN star is outside the library's scope.
- `run.sh` — copy DLTlib → `build/`, patch, compile (`g++ ... -lglpk`),
  regenerate `results.txt` + `sweep.csv`. Requires GLPK (`pacman -S glpk`).
- `results.txt` — solver output: DLT optima at `l ∈ {0, tar, scp}`, our
  1:4 partition evaluated under both transports, the measured-vs-optimal
  steal endpoints, the 3-node outlook.
- `sweep.csv` — optimum yeco share and compute wall vs `l` (0–100 ms/frame,
  1 ms steps), from the library solver at each point.
- `plot_dlt28.py` — → `reports/img/dlt28_model.png` (gitignored): panel (a)
  optimal share vs `l`; panel (b) predicted e2e for optimal vs our-1:4
  shares, with the measured weighted-static and dynamic points overlaid.

## Headline numbers (from `results.txt`)

| Scenario | laptop/yeco frames | predicted e2e | measured |
|---|---|---|---|
| DLT optimum, `l=0` | 19.7 / 80.3 | 5.34 s | — |
| DLT optimum, scp | 39.1 / 60.9 | 9.42 s | — |
| DLT optimum, tar | 23.3 / 76.7 | 6.09 s | — |
| our 1:4 under scp | 21 / 79 | 11.87 s | 13.11 ± 1.81 |
| our 1:4 under tar | 21 / 79 | 6.23 s | 6.45 ± 0.26 |
| dyn steal under scp | 42 / 58 | 10.03 s | 12.95 ± 3.45 |
| + ivy (3-node, tar) | 22.1 / 73.0 / 4.8 | 5.85 s | — |

Date: 2026-07-16.
