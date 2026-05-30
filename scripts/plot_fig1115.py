#!/usr/bin/env python3
"""Plot a Fig 11.15-style chart from sweep.sh output.

Bars show mean elapsed time per configuration (with std-dev error bars when
multiple reps are present).  An overlaid line shows each config's time as a
percentage of the GPU-alone time.

Usage:
    python3 plot_fig1115.py [results.csv] [out.png]
Defaults: sweep_results/results.csv -> sweep_results/fig1115.png
"""
import sys
import csv
from collections import defaultdict
from statistics import mean, pstdev

import matplotlib.pyplot as plt


def main(csv_path: str, out_path: str) -> None:
    runs = defaultdict(list)            # preserves insertion order of first sight
    order = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"]
            if label not in runs:
                order.append(label)
            runs[label].append(float(row["elapsed_s"]))

    means = [mean(runs[L]) for L in order]
    stds = [pstdev(runs[L]) if len(runs[L]) > 1 else 0.0 for L in order]

    # GPU-alone baseline for the percentage overlay.  Fall back to the first
    # config if there is no row literally labeled "GPU".
    if "GPU" in runs:
        baseline = mean(runs["GPU"])
        baseline_label = "GPU"
    else:
        baseline = means[0]
        baseline_label = order[0]
    pct = [100.0 * m / baseline for m in means]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    x = list(range(len(order)))
    bars = ax1.bar(x, means, yerr=stds, capsize=4,
                   color="#4472C4", edgecolor="#1f3a73",
                   label="Mean execution time (s)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(order, rotation=20, ha="right")
    ax1.set_ylabel("Execution time (s)")
    ax1.set_ylim(0, max(means) * 1.28)
    # Bar value labels placed inside each bar (white) to avoid colliding with
    # the percent-of-baseline labels overlaid via the right axis.
    for rect, m in zip(bars, means):
        ax1.text(rect.get_x() + rect.get_width() / 2,
                 rect.get_height() * 0.5,
                 f"{m:.2f} s",
                 ha="center", va="center", fontsize=9,
                 color="white", fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(x, pct, "o-", color="#C00000",
             label=f"% of {baseline_label}-alone")
    ax2.set_ylabel(f"Time as % of {baseline_label}-alone")
    ax2.set_ylim(0, max(pct) * 1.28)
    for xi, p in zip(x, pct):
        ax2.text(xi, p + max(pct) * 0.03, f"{p:.1f}%",
                 ha="center", va="bottom", fontsize=9, color="#C00000")

    n_reps = len(next(iter(runs.values())))
    title = (f"Hybrid Mandelbrot: average execution time "
             f"({n_reps} rep{'s' if n_reps != 1 else ''} per config)")
    plt.title(title)
    fig.tight_layout()

    # Combined legend on the upper-left of the bar axis.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "sweep_results/results.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "sweep_results/fig1115.png"
    main(csv_path, out_path)
