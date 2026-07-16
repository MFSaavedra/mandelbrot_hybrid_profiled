#!/usr/bin/env python3
"""Plot the weighted-static vs dynamic-stealing benchmark for report 26.

Reads results_dynamic.csv (weight-capped chunks, the shipped configuration)
and results_dynamic_uncapped.csv (the first run, before the KCAP*w_r chunk
cap; preserved because it motivates the fix).  Output:
reports/img/framedist_dynamic.png (gitignored, regenerated on demand).
"""
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "reports" / "img" / "framedist_dynamic.png"

def load(name):
    rows = defaultdict(list)
    with open(HERE / name) as f:
        for r in csv.DictReader(f):
            rows[r["config"]].append(float(r["wall_e2e_s"]))
    return rows

cap = load("results_dynamic.csv")
unc = load("results_dynamic_uncapped.csv")

# report-25 references (same pair, same spec, earlier same-day batch)
EQUAL_SHARE = 46.90   # 2node_cyclic mean, results_bench.csv
IDEAL = 15.38         # harmonic ideal: 1/(1/18.75 + 1/85.48)

bars = [
    ("laptop\nalone",              cap["laptop_hybrid"],   "#666666"),
    ("weighted\nstatic 5:1",       cap["weighted_static"], "#2E8B57"),
    ("dynamic 5:1\n(0 stolen)",    cap["dyn_51"],          "#2E8B57"),
    ("dynamic 1:1\nuncapped",      unc["dyn_11"],          "#C00000"),
    ("dynamic 1:1\ncapped KCAP*w", cap["dyn_11"],          "#E8853D"),
]

fig, ax = plt.subplots(figsize=(8.6, 4.6))
for x, (label, reps, color) in enumerate(bars):
    m, s = st.mean(reps), st.stdev(reps)
    ax.bar(x, m, 0.62, yerr=s, capsize=4, color=color, zorder=2)
    ax.plot([x] * len(reps), reps, "o", ms=4, mfc="white", mec="black", mew=0.7, zorder=3)
    ax.text(x, m + st.stdev(reps) + 0.8, f"{m:.1f}", ha="center", fontsize=10)

ax.axhline(EQUAL_SHARE, color="#C00000", ls="--", lw=1.4, zorder=1)
ax.text(1.62, EQUAL_SHARE - 2.4, "equal-share static (report 25): 46.9 s", color="#C00000", fontsize=9)
ax.axhline(IDEAL, color="#1f3a73", ls=":", lw=1.4, zorder=1)
ax.text(-0.45, IDEAL - 2.4, "ideal weighted, zero overhead: 15.4 s", color="#1f3a73", fontsize=9)

ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=9)
ax.set_ylabel("end-to-end wall (s)")
ax.set_ylim(0, 52)
ax.set_title("Weighted shares + work stealing, laptop (hybrid) + ivy (CPU-only)\n"
             "production spec, 3 reps each — right weights reach parity; stealing rescues wrong ones")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=160)
print(f"wrote {OUT}", file=sys.stderr)
