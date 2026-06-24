#!/usr/bin/env python3
"""Analyze the diffThreshold x pixelSizeThresh sweep: mean-wall grid, the
optimal pair, and a heatmap.  Usage: analyze.py [results.csv]"""
import csv, sys, statistics as st
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np

HERE = "experiments/19-threshold-sweep"
path = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/results.csv"
vals = defaultdict(list)
for r in csv.DictReader(open(path)):
    try:
        vals[(float(r["diffT"]), int(r["pixT"]))].append(float(r["wall_s"]))
    except ValueError:
        pass

diffts = sorted({k[0] for k in vals})
pixts  = sorted({k[1] for k in vals})
grid = np.full((len(diffts), len(pixts)), np.nan)
for i, d in enumerate(diffts):
    for j, p in enumerate(pixts):
        if (d, p) in vals:
            grid[i, j] = st.mean(vals[(d, p)])

print(f"mean wall (s)   reps={max(len(v) for v in vals.values())}")
print("diffT \\ pixT  " + "".join(f"{p:>9}" for p in pixts))
for i, d in enumerate(diffts):
    print(f"{d:<12}" + "".join(f"{grid[i,j]:9.2f}" for j in range(len(pixts))))

best = min(vals, key=lambda k: st.mean(vals[k]))
bm = st.mean(vals[best]); bsd = st.pstdev(vals[best]) if len(vals[best]) > 1 else 0.0
worst = max(vals, key=lambda k: st.mean(vals[k]))
wm = st.mean(vals[worst])
print(f"\nOPTIMAL : diffT={best[0]}, pixT={best[1]}  ->  {bm:.2f}s  (+/-{bsd:.2f}, n={len(vals[best])})")
print(f"worst   : diffT={worst[0]}, pixT={worst[1]}  ->  {wm:.2f}s   (spread across grid: {100*(wm-bm)/bm:.1f}%)")
print(f"current default (0.1, 32768): {st.mean(vals.get((0.1,32768),[float('nan')])):.2f}s")

fig, ax = plt.subplots(figsize=(7.5, 4.6))
im = ax.imshow(grid, aspect="auto", cmap="viridis_r", origin="lower")
ax.set_xticks(range(len(pixts))); ax.set_xticklabels(pixts, rotation=25)
ax.set_yticks(range(len(diffts))); ax.set_yticklabels(diffts)
ax.set_xlabel("pixelSizeThresh (pixel floor)"); ax.set_ylabel("diffThreshold")
for i in range(len(diffts)):
    for j in range(len(pixts)):
        if not np.isnan(grid[i, j]):
            ax.text(j, i, f"{grid[i,j]:.1f}", ha="center", va="center",
                    color="white", fontsize=8)
bi, bj = diffts.index(best[0]), pixts.index(best[1])
ax.add_patch(plt.Rectangle((bj-0.5, bi-0.5), 1, 1, fill=False, edgecolor="red", lw=2.5))
plt.colorbar(im, label="wall (s)")
ax.set_title("Wall time by (diffThreshold, pixelSizeThresh) -- red = optimum")
plt.tight_layout(); plt.savefig(f"{HERE}/heatmap.png", dpi=150, bbox_inches="tight")
print(f"saved {HERE}/heatmap.png")
