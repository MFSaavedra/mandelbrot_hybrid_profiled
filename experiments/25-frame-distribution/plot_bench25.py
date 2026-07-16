#!/usr/bin/env python3
"""Plot the 2-node heterogeneous benchmark (results_bench.csv) for report 25.

Output: reports/img/framedist_bench.png (gitignored, regenerated on demand).
Grouped bars per distributed config: end-to-end wall, laptop rank wall, ivy
rank wall (3-rep means, error bars = sample sigma; dots = individual reps).
Horizontal lines: single-node laptop baseline and the analytic ideal
weighted 2-node wall (harmonic sum of the two solo throughputs).
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
CSV = HERE / "results_bench.csv"
OUT = HERE.parents[1] / "reports" / "img" / "framedist_bench.png"

rows = defaultdict(lambda: defaultdict(list))  # config -> col -> [reps]
with open(CSV) as f:
    for r in csv.DictReader(f):
        for col in ("wall_e2e_s", "laptop_s", "ivy_s"):
            if r[col]:
                rows[r["config"]][col].append(float(r[col]))

base = st.mean(rows["laptop_hybrid"]["wall_e2e_s"])
ivy_solo = rows["ivy_solo"]["wall_e2e_s"][0]
ideal = 1.0 / (1.0 / base + 1.0 / ivy_solo)

configs = ["2node_cyclic", "2node_bc8", "2node_block", "2node_block_rev"]
labels = ["cyclic\n(block=1)", "block-cyclic\n(block=8)",
          "block\n(ivy deep half)", "block reversed\n(ivy cheap half)"]
series = [("wall_e2e_s", "end-to-end", "#444444"),
          ("laptop_s", "laptop rank", "#2E8B57"),
          ("ivy_s", "ivy rank", "#C00000")]

fig, ax = plt.subplots(figsize=(8.6, 4.6))
w = 0.26
for k, (col, lab, color) in enumerate(series):
    xs = [i + (k - 1) * w for i in range(len(configs))]
    means = [st.mean(rows[c][col]) for c in configs]
    sigs = [st.stdev(rows[c][col]) for c in configs]
    ax.bar(xs, means, w, yerr=sigs, capsize=4, color=color, label=lab, zorder=2)
    for x, c in zip(xs, configs):
        ax.plot([x] * len(rows[c][col]), rows[c][col], "o", ms=3.5,
                mfc="white", mec="black", mew=0.6, zorder=3)

ax.axhline(base, color="#1f3a73", ls="--", lw=1.4, zorder=1)
ax.text(-0.55, base + 0.8, f"laptop alone {base:.1f} s", color="#1f3a73", fontsize=9)
ax.axhline(ideal, color="#1f3a73", ls=":", lw=1.4, zorder=1)
ax.text(-0.55, ideal - 2.6, f"ideal weighted 2-node {ideal:.1f} s", color="#1f3a73", fontsize=9)

ax.set_xticks(range(len(configs)), labels, fontsize=9)
ax.set_ylabel("wall time (s)")
ax.set_title("2-node static frame distribution, laptop (hybrid 12 1) + ivy (CPU-only 8 0)\n"
             "production spec, 3 reps, save=1 — every config above the laptop-alone line")
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=160)
print(f"wrote {OUT}", file=sys.stderr)
