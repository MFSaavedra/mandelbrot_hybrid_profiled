#!/usr/bin/env python3
"""Figures for report 24 (ncu re-baseline after the GPU-side periodicity check).

Reads this experiment's analysis/summary.csv (v7) and experiment 16's
tracked analysis/summary.csv (the v6 baseline; kernel unchanged v5->v6),
plus both experiments' *_warp.csv wide dumps for the per-kernel warp
efficiency distributions. Outputs reports/img/ncu24_{warpeff,duration,dist}.png
(gitignored). Exp-16's exterior dur_ms is recorded in µs (slip); corrected
here to 0.615 ms.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
IMG = os.path.join(ROOT, "reports", "img")
E16 = os.path.join(ROOT, "experiments", "16-ncu-divergence", "analysis")

RED, GREEN, BLUE = "#C00000", "#2E8B57", "#1f3a73"
REGIMES = ["interior", "exterior", "boundary", "canonical"]


def summary(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["regime"]] = {k: float(v) for k, v in r.items() if k != "regime"}
    return out


v6 = summary(os.path.join(E16, "summary.csv"))
v7 = summary(os.path.join(HERE, "analysis", "summary.csv"))
v6["exterior"]["dur_ms"] = 0.615  # reference slip: recorded in µs


def wdist(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    hdr, data = rows[0], rows[2:]
    i = hdr.index("smsp__thread_inst_executed_per_inst_executed.ratio")
    return [float(r[i].replace(",", "")) / 32 * 100 for r in data if r[i] != ""]


# ============ fig 1: full-set warp efficiency, v6 vs v7 =================
fig, ax = plt.subplots(figsize=(7.6, 4.0))
w = 0.34
for j, (tag, d, color) in enumerate([("main (v6)", v6, RED), ("gpu-periodicity (v7)", v7, GREEN)]):
    xs = [i + (j - 0.5) * w for i in range(len(REGIMES))]
    vals = [d[r]["warpEff"] for r in REGIMES]
    ax.bar(xs, vals, w * 0.92, color=color, label=tag)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.6, f"{v:.1f}", ha="center", fontsize=9)
ax.set_xticks(range(len(REGIMES)))
ax.set_xticklabels(REGIMES)
ax.set_ylabel("warp execution efficiency (%)")
ax.set_ylim(60, 106)
ax.set_title("Full-set captures: the check costs 3.7 points on interior, zero elsewhere")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(IMG, "ncu24_warpeff.png"), dpi=150)

# ============ fig 2: mean kernel duration (log), v6 vs v7 ===============
fig, ax = plt.subplots(figsize=(7.6, 4.2))
for j, (tag, d, color) in enumerate([("main (v6)", v6, RED), ("gpu-periodicity (v7)", v7, GREEN)]):
    xs = [i + (j - 0.5) * w for i in range(len(REGIMES))]
    vals = [d[r]["dur_ms"] for r in REGIMES]
    ax.bar(xs, vals, w * 0.92, color=color, label=tag)
    for x, v in zip(xs, vals):
        ax.text(x, v * 1.15, f"{v:.3g}", ha="center", fontsize=9)
for i, r in enumerate(REGIMES):
    ratio = v6[r]["dur_ms"] / v7[r]["dur_ms"]
    lab = f"{ratio:.1f}$\\times$ faster" if ratio > 1 else f"+{(1/ratio-1)*100:.0f}%"
    ax.text(i, max(v6[r]["dur_ms"], v7[r]["dur_ms"]) * 2.6, lab,
            ha="center", fontsize=11, fontweight="bold", color=BLUE)
ax.set_yscale("log")
ax.set_xticks(range(len(REGIMES)))
ax.set_xticklabels(REGIMES)
ax.set_ylabel("mean kernel duration (ms, log)")
ax.set_ylim(0.3, 4000)
ax.set_title("Where the check wins and what it costs (clocks match within 1%)")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(IMG, "ncu24_duration.png"), dpi=150)

# ============ fig 3: warp-efficiency distribution (300-kernel captures) =
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.9))
for a, reg in [(a1, "boundary"), (a2, "canonical")]:
    d6 = wdist(os.path.join(E16, f"{reg}_warp.csv"))
    d7 = wdist(os.path.join(HERE, "analysis", f"{reg}_warp.csv"))
    bins = list(range(15, 105, 5))
    a.hist([d6, d7], bins=bins, color=[RED, GREEN],
           label=["main (v6)", "gpu-periodicity (v7)"], density=True)
    a.set_title(f"{reg} ({len(d7)} kernels)")
    a.set_xlabel("per-kernel warp efficiency (%)")
    m6, m7 = sum(d6) / len(d6), sum(d7) / len(d7)
    a.axvline(m6, color=RED, linestyle="--", linewidth=1.4)
    a.axvline(m7, color=GREEN, linestyle="--", linewidth=1.4)
    a.text(0.03, 0.95, f"mean {m6:.1f} $\\to$ {m7:.1f}", transform=a.transAxes,
           fontsize=10, va="top", color=BLUE)
a1.set_ylabel("kernel density")
a1.legend(loc="center left", fontsize=9)
fig.suptitle("Divergence structure: boundary unchanged; canonical's interior kernels absorb the early exits", y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "ncu24_dist.png"), dpi=150)

print("wrote ncu24_warpeff.png, ncu24_duration.png, ncu24_dist.png")
