#!/usr/bin/env python3
"""Figures for report 23 (GPU-side periodicity check).

Reads results.csv (bars are 3-rep means, per-rep dots overlaid; the high
gp dots in rep 1 are the cold-start transient the CPU12 control exposes).
Lane/pool numbers are steady-state (reps 2-3) means extracted from
logs/{base,gp}.{hybrid,gpuonly}.r{2,3}.stderr. Outputs to
reports/img/gpuperiod_{wall,lane,share,pool}.png (gitignored).
"""
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
IMG = os.path.join(ROOT, "reports", "img")
os.makedirs(IMG, exist_ok=True)

RED, GREEN, BLUE = "#C00000", "#2E8B57", "#1f3a73"

# ---- results.csv -> walls[config][binary] = [reps...] -----------------
walls = defaultdict(lambda: defaultdict(list))
with open(os.path.join(HERE, "results.csv")) as f:
    for row in csv.DictReader(f):
        walls[row["config"]][row["binary"]].append(float(row["elapsed_s"]))

# ---- steady-state (reps 2-3) means from the logs ----------------------
LANE = {  # hybrid runs, GPU thread
    "base": dict(wall=29.829, regions=1008, kernel_s=25.312, memcpy_s=0.096),
    "gp": dict(wall=23.715, regions=3006, kernel_s=10.692, memcpy_s=0.258),
}
POOL = {  # hybrid runs, 11 CPU workers
    "base": dict(regions=4600, compute_s=311.6, busiest_s=29.04),
    "gp": dict(regions=2602, compute_s=249.6, busiest_s=23.40),
}
TOTAL_LEAVES = 5608

# ================= fig 1: wall time =====================================
fig, ax = plt.subplots(figsize=(8.2, 4.4))
configs = [("gpuonly", "GPU-only (1 thread)"), ("hybrid", "dGPU+11CPU"),
           ("cpu12", "CPU12 (control)")]
width = 0.34
for i, (cfg, label) in enumerate(configs):
    for j, (b, color, name) in enumerate(
        [("base", RED, "main (v6)"), ("gp", GREEN, "gpu-periodicity")]
    ):
        reps = walls[cfg][b]
        mean = sum(reps) / len(reps)
        x = i + (j - 0.5) * width
        ax.bar(x, mean, width * 0.92, color=color,
               label=name if i == 0 else None)
        ax.scatter([x] * len(reps), reps, color="black", zorder=3, s=14)
        ax.text(x, mean + 0.8, f"{mean:.2f}", ha="center", fontsize=10)
    b0 = sum(walls[cfg]["base"]) / len(walls[cfg]["base"])
    g0 = sum(walls[cfg]["gp"]) / len(walls[cfg]["gp"])
    d = (g0 / b0 - 1) * 100
    ax.text(i, max(b0, g0) + 6.5, f"{d:+.1f}%".replace("-", "$-$"),
            ha="center", fontsize=12, fontweight="bold", color=BLUE)
ax.set_xticks(range(len(configs)))
ax.set_xticklabels([c[1] for c in configs])
ax.set_ylabel("wall time (s)")
ax.set_ylim(0, 92)
ax.set_title("Production spec, save=1, 3 reps (dots; the high dots are rep 1, pre-steady-state)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(IMG, "gpuperiod_wall.png"), dpi=150)

# ================= fig 2: GPU-thread cost structure (hybrid) ============
fig, ax = plt.subplots(figsize=(7.2, 4.2))
xs = [0, 1]
for x, b in zip(xs, ["base", "gp"]):
    d = LANE[b]
    host = d["wall"] - d["kernel_s"] - d["memcpy_s"]
    ax.bar(x, d["kernel_s"], 0.55, color=BLUE,
           label="kernel (CUDA events)" if x == 0 else None)
    ax.bar(x, host, 0.55, bottom=d["kernel_s"] + d["memcpy_s"], color="#c9a227",
           label="host side (commit + examine + queue)" if x == 0 else None)
    duty = d["kernel_s"] / d["wall"] * 100
    ax.text(x, d["kernel_s"] / 2, f"{d['kernel_s']:.1f} s\n({duty:.0f}% of wall)",
            ha="center", va="center", color="white", fontsize=10)
    ax.text(x, d["kernel_s"] + d["memcpy_s"] + host / 2, f"{host:.1f} s",
            ha="center", va="center", fontsize=10)
    ax.text(x, d["wall"] + 1.3, f"wall {d['wall']:.1f} s", ha="center", fontsize=10)
    ax.hlines(POOL[b]["busiest_s"], x - 0.38, x + 0.38, color=RED,
              linestyle="--", linewidth=1.6,
              label="busiest CPU worker" if x == 0 else None)
ax.set_xticks(xs)
ax.set_xticklabels(["main (v6)\n1,008 GPU regions", "gpu-periodicity\n3,006 GPU regions"])
ax.set_ylabel("GPU-thread time (s)")
ax.set_ylim(0, 35)
ax.set_title("Hybrid, steady state: what the GPU thread's wall is made of")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "gpuperiod_lane.png"), dpi=150)

# ================= fig 3: region share ==================================
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for y, b in [(1, "base"), (0, "gp")]:
    g = LANE[b]["regions"]
    c = TOTAL_LEAVES - g
    ax.barh(y, g, 0.5, color=GREEN if b == "gp" else RED)
    ax.barh(y, c, 0.5, left=g, color="#bbbbbb")
    ax.text(g / 2, y, f"GPU {g:,}\n({g / TOTAL_LEAVES * 100:.1f}%)",
            ha="center", va="center", fontsize=10,
            color="white" if g > 1500 else "black")
    ax.text(g + c / 2, y, f"CPU pool {c:,}", ha="center", va="center", fontsize=10)
    avg = LANE[b]["kernel_s"] * 1000 / g
    ax.text(TOTAL_LEAVES + 60, y, f"{avg:.2f} ms/region\n(kernel avg)",
            va="center", fontsize=9, color=BLUE)
ax.set_yticks([1, 0])
ax.set_yticklabels(["main (v6)", "gpu-periodicity"])
ax.set_xlabel(f"leaf regions (of {TOTAL_LEAVES:,}; GPU takes the largest first)")
ax.set_xlim(0, TOTAL_LEAVES + 1500)
ax.set_title("Hybrid, steady state: who computes the queue")
fig.tight_layout()
fig.savefig(os.path.join(IMG, "gpuperiod_share.png"), dpi=150)

# ================= fig 4: pool relief -> wall ===========================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.8))
for i, b in enumerate(["base", "gp"]):
    color = RED if b == "base" else GREEN
    a1.bar(i, POOL[b]["compute_s"], 0.55, color=color)
    a1.text(i, POOL[b]["compute_s"] + 6, f"{POOL[b]['compute_s']:.0f} s",
            ha="center", fontsize=10)
    a2.bar(i, POOL[b]["busiest_s"], 0.55, color=color)
    wall = LANE[b]["wall"]
    a2.hlines(wall, i - 0.38, i + 0.38, color=BLUE, linewidth=1.6)
    a2.text(i, POOL[b]["busiest_s"] - 2.4, f"{POOL[b]['busiest_s']:.1f}",
            ha="center", fontsize=10, color="white")
    a2.text(i + 0.02, wall + 0.5, f"wall {wall:.1f}", fontsize=9, color=BLUE)
a1.set_title("CPU-pool compute (11 workers)")
a1.set_ylabel("seconds")
a1.text(0.5, 285, "$-$19.9%", ha="center", fontsize=12, fontweight="bold", color=BLUE)
a2.set_title("busiest CPU worker vs wall")
a2.set_ylim(0, 34)
for a in (a1, a2):
    a.set_xticks([0, 1])
    a.set_xticklabels(["main (v6)", "gpu-periodicity"])
fig.suptitle("Hybrid, steady state: the pool stays the binding resource, at 20% less cost", y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "gpuperiod_pool.png"), dpi=150)

print("wrote", ", ".join(f"gpuperiod_{n}.png" for n in ["wall", "lane", "share", "pool"]))
