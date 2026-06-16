#!/usr/bin/env python3
"""Plots for report 20 (integrated-GPU OpenCL backend, headline A/B).

Reads experiments/20-igpu-opencl/results.csv (wall times) and the per-run
stderr logs (per-backend region counts) and writes two figures into
reports/img/:

  igpu_wall.png  -- mean wall time per config, range error bars, % of the
                    dGPU+11CPU baseline overlaid. The headline.
  igpu_work.png  -- the 5608 leaves split by executor class (CPU pool / dGPU /
                    iGPU). Same total height in every bar (decomposition is
                    identical); only the assignment shifts.

Usage: python3 plot_igpu.py            # run from anywhere; paths are resolved
"""
import os
import re
import csv
from collections import defaultdict
from statistics import mean, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV = os.path.join(HERE, "results.csv")
IMG = os.path.join(ROOT, "reports", "img")

# Display order: descending wall, so the figure reads as a ladder down to the win.
ORDER = ["CPU12", "iGPU+11CPU", "dGPU+11CPU", "dGPU+iGPU+10CPU"]
BASELINE = "dGPU+11CPU"          # current best -> 100% reference for the overlay

GREEN, BLUE, GREY = "#2E8B57", "#4472C4", "#9aa0a6"
ORANGE = "#ED7D31"


def load_walls():
    runs = defaultdict(list)
    with open(CSV) as f:
        for row in csv.DictReader(f):
            runs[row["label"]].append(float(row["elapsed_s"]))
    return runs


def region_split(label):
    """Parse dGPU / iGPU / CPU-pool region counts from the rep-1 stderr log."""
    f = os.path.join(HERE, "logs", f"{label}.r1.log.stderr")
    txt = open(f).read()
    def grab(pat):
        m = re.search(pat, txt)
        return int(m.group(1)) if m else 0
    dgpu = grab(r"Regions computed on GPU\s*:\s*(\d+)")
    igpu = grab(r"Regions computed on iGPU\s*:\s*(\d+)")
    cpu = sum(int(n) for n in re.findall(r"Regions computed on CPU\s*:\s*(\d+)", txt))
    return cpu, dgpu, igpu


def plot_wall(runs):
    means = [mean(runs[L]) for L in ORDER]
    # range as asymmetric error bars (min/max around the mean)
    lo = [mean(runs[L]) - min(runs[L]) for L in ORDER]
    hi = [max(runs[L]) - mean(runs[L]) for L in ORDER]
    base = mean(runs[BASELINE])
    pct = [100.0 * m / base for m in means]
    colors = [GREEN if L == "dGPU+iGPU+10CPU" else BLUE if L == BASELINE else GREY
              for L in ORDER]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x = list(range(len(ORDER)))
    bars = ax1.bar(x, means, yerr=[lo, hi], capsize=5, color=colors,
                   edgecolor="#1f3a73")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ORDER, rotation=12, ha="right")
    ax1.set_ylabel("End-to-end wall time (s)")
    ax1.set_ylim(0, max(means) * 1.22)
    # Bar value labels inside each bar (white) so they don't collide with the
    # red percent-of-baseline markers that ride along the bar tops.
    for rect, m in zip(bars, means):
        ax1.text(rect.get_x() + rect.get_width() / 2, rect.get_height() * 0.5,
                 f"{m:.2f} s", ha="center", va="center", fontsize=11,
                 color="white", fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(x, pct, "o-", color="#C00000", label=f"% of {BASELINE}")
    ax2.set_ylabel(f"Wall as % of {BASELINE}")
    ax2.set_ylim(0, max(pct) * 1.22)
    for xi, p in zip(x, pct):
        ax2.text(xi + 0.06, p + max(pct) * 0.035, f"{p:.1f}%", ha="left",
                 va="bottom", fontsize=10, color="#C00000", fontweight="bold")

    plt.title("Hybrid Mandelbrot wall time by backend mode "
              "(diffT=0.1, full spec.in, 3 reps; error bars = min/max)")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    out = os.path.join(IMG, "igpu_wall.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def plot_work(runs):
    cpu, dgpu, igpu = [], [], []
    for L in ORDER:
        c, d, i = region_split(L)
        cpu.append(c); dgpu.append(d); igpu.append(i)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = list(range(len(ORDER)))
    b1 = ax.bar(x, cpu, color=ORANGE, edgecolor="white", label="CPU pool")
    b2 = ax.bar(x, dgpu, bottom=cpu, color=BLUE, edgecolor="white",
                label="dGPU (CUDA)")
    b3 = ax.bar(x, igpu, bottom=[c + d for c, d in zip(cpu, dgpu)],
                color=GREEN, edgecolor="white", label="iGPU (OpenCL)")
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, rotation=12, ha="right")
    ax.set_ylabel("Leaf regions computed (of 5608 total)")
    ax.set_ylim(0, 6300)

    # annotate each non-zero segment with its count
    for xi, (c, d, i) in enumerate(zip(cpu, dgpu, igpu)):
        if c: ax.text(xi, c / 2, f"{c}", ha="center", va="center",
                      color="white", fontsize=9, fontweight="bold")
        if d: ax.text(xi, c + d / 2, f"{d}", ha="center", va="center",
                      color="white", fontsize=9, fontweight="bold")
        if i: ax.text(xi, c + d + i / 2, f"{i}", ha="center", va="center",
                      color="white", fontsize=9, fontweight="bold")
    ax.axhline(5608, ls="--", lw=1, color="#444",
               label="total leaves (5608, identical)")
    ax.set_title("Where the 5608 leaves ran: decomposition is identical, "
                 "only the assignment shifts")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    out = os.path.join(IMG, "igpu_work.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(IMG, exist_ok=True)
    r = load_walls()
    plot_wall(r)
    plot_work(r)
