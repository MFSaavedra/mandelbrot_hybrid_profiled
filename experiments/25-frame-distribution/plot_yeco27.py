#!/usr/bin/env python3
"""Plot the laptop+yeco benchmarks for report 27.

Reads results_bench_yeco.csv (equal-share static sweep, scp-era transport),
results_dynamic_yeco_scp.csv (weighted/dynamic A/B before collection
batching; preserved because it motivates the fix) and
results_dynamic_yeco.csv (same protocol after the tar-stream batching).
Output: reports/img/framedist_yeco.png (gitignored, regenerated on demand).
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
OUT = HERE.parents[1] / "reports" / "img" / "framedist_yeco.png"

def load(name):
    rows = defaultdict(list)
    with open(HERE / name) as f:
        for r in csv.DictReader(f):
            rows[r["config"]].append(float(r["wall_e2e_s"]))
    return rows

bench = load("results_bench_yeco.csv")
scp = load("results_dynamic_yeco_scp.csv")
tar = load("results_dynamic_yeco.csv")

YECO_SOLO = st.mean(bench["yeco_solo"])       # 5.15 s, images stay on yeco
IDEAL = 1 / (1 / st.mean(bench["laptop_hybrid"]) + 1 / YECO_SOLO)  # 4.14 s
SOLO_FETCH = YECO_SOLO + 1.4                  # + tar fetch of 100 PNGs / 46 MB

fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                               gridspec_kw={"width_ratios": [6, 7]})

# ---- panel (a): equal-share static sweep (scp-era, like report 25's) ----
bars_a = [
    ("laptop\nalone",        bench["laptop_hybrid"],   "#666666"),
    ("yeco\nsolo",           bench["yeco_solo"],       "#1f3a73"),
    ("cyclic",               bench["2node_cyclic"],    "#E8853D"),
    ("block-\ncyclic(8)",    bench["2node_bc8"],       "#E8853D"),
    ("block", bench["2node_block"],     "#E8853D"),
    ("block\nreversed", bench["2node_block_rev"], "#E8853D"),
]
for x, (label, reps, color) in enumerate(bars_a):
    m, s = st.mean(reps), st.stdev(reps)
    axa.bar(x, m, 0.62, yerr=s, capsize=4, color=color, zorder=2)
    axa.plot([x] * len(reps), reps, "o", ms=4, mfc="white", mec="black", mew=0.7, zorder=3)
    axa.text(x, m + s + 0.5, f"{m:.1f}", ha="center", fontsize=10)
axa.axhline(IDEAL, color="#1f3a73", ls=":", lw=1.4, zorder=1)
axa.text(1.6, IDEAL - 1.3, f"ideal weighted, zero overhead: {IDEAL:.1f} s",
         color="#1f3a73", fontsize=9)
axa.set_xticks(range(len(bars_a)), [b[0] for b in bars_a], fontsize=9)
axa.set_ylabel("end-to-end wall (s)")
axa.set_ylim(0, 25)
axa.set_title("(a) equal-share static, per-file scp collection\n"
              "every 2-node config beats the laptop, loses to yeco solo")
axa.grid(axis="y", alpha=0.3)

# ---- panel (b): weighted/dynamic A/B, scp vs tar-batched collection ----
groups = [("weighted\nstatic 1:4", "weighted_static"),
          ("dynamic 1:4", "dyn_14"),
          ("dynamic 1:1\n(wrong)", "dyn_11")]
w = 0.36
for x, (label, key) in enumerate(groups):
    for dx, rows, color, tag in ((-w / 2, scp, "#C00000", "scp"),
                                 (+w / 2, tar, "#2E8B57", "tar")):
        m, s = st.mean(rows[key]), st.stdev(rows[key])
        axb.bar(x + dx, m, w, yerr=s, capsize=4, color=color, zorder=2)
        axb.plot([x + dx] * len(rows[key]), rows[key], "o", ms=4,
                 mfc="white", mec="black", mew=0.7, zorder=3)
        axb.text(x + dx, m + s + 0.4, f"{m:.1f}", ha="center", fontsize=9)
axb.axhline(SOLO_FETCH, color="#1f3a73", ls="--", lw=1.4, zorder=1)
axb.text(2.42, SOLO_FETCH + 0.25, f"yeco solo + fetch: ~{SOLO_FETCH:.1f} s",
         color="#1f3a73", fontsize=9, ha="right")
axb.axhline(IDEAL, color="#1f3a73", ls=":", lw=1.4, zorder=1)
axb.text(-0.42, IDEAL + 0.25, f"ideal: {IDEAL:.1f} s", color="#1f3a73", fontsize=9)
axb.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=9)
axb.set_ylim(0, 19)
axb.set_title("(b) weighted + dynamic, collection transport A/B\n"
              "red = per-file scp, green = single tar stream")
axb.grid(axis="y", alpha=0.3)

fig.suptitle("laptop (12T hybrid) + yeco (20T + RTX 4090, via SSH jump), "
             "production spec, save=1, 3 reps each", fontsize=10, y=1.0)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"wrote {OUT}", file=sys.stderr)
