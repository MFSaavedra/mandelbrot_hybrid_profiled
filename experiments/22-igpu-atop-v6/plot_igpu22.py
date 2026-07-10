#!/usr/bin/env python3
"""Figures for report 22 (iGPU re-priced atop binary-v6-periodicity).

Reads experiments/20-igpu-opencl/results.csv, experiments/22-igpu-atop-v6/
results.csv and the probe logs; writes reports/img/igpu22_*.png (gitignored,
regenerated on demand).
"""
import csv
import glob
import os
import re
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
IMG = os.path.join(ROOT, "reports", "img")
os.makedirs(IMG, exist_ok=True)

RED, GREEN, BLUE = "#C00000", "#2E8B57", "#1f3a73"


def read_csv(path, key_fields, val_field):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            k = tuple(row[k] for k in key_fields)
            out.setdefault(k, []).append(float(row[val_field]))
    return out


def wall(path):
    txt = open(path).read()
    return float(re.search(r"\[total_elapsed_s\] ([\d.]+)", txt).group(1))


def pool_stats(path):
    txt = open(path).read()
    cpu = [int(x) for x in re.findall(r"Regions computed on CPU : (\d+)", txt)]
    dgpu = sum(int(x) for x in re.findall(r"Regions computed on GPU : (\d+)", txt))
    igpu = sum(int(x) for x in re.findall(r"Regions computed on iGPU : (\d+)", txt))
    busy = [float(x) / 1e3 for x in re.findall(r"Total compute time\s*:\s*([\d.]+) ms", txt)]
    return dict(nthr=len(cpu), pool_regions=sum(cpu), dgpu=dgpu, igpu=igpu,
                pool_busy=sum(busy))


def bars_with_dots(ax, xs, groups, colors, labels, width=0.35):
    for i, (vals, c, lab) in enumerate(zip(groups, colors, labels)):
        pos = [x + (i - (len(groups) - 1) / 2) * width for x in xs]
        means = [st.mean(v) for v in vals]
        ax.bar(pos, means, width * 0.92, color=c, label=lab, zorder=2)
        for p, v in zip(pos, vals):
            ax.plot([p] * len(v), v, "o", ms=4, mfc="white", mec="black",
                    mew=0.8, zorder=3)
        for p, m in zip(pos, means):
            ax.annotate(f"{m:.2f}", (p, m), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9)


# ---------------------------------------------------------------- data
e20 = read_csv(os.path.join(ROOT, "experiments/20-igpu-opencl/results.csv"),
               ("gpuMode",), "elapsed_s")
e22 = read_csv(os.path.join(HERE, "results.csv"), ("gpuMode",), "elapsed_s")
pre_m1, pre_m3 = e20[("1",)], e20[("3",)]
post_m1, post_m3 = e22[("1",)], e22[("3",)]

# ------------------------------------------------- fig 1: the collapse
fig, ax = plt.subplots(figsize=(7.2, 4.4))
bars_with_dots(ax, [0, 1], [[pre_m1, post_m1], [pre_m3, post_m3]],
               [RED, GREEN],
               ["dGPU+11CPU (mode 1)", "dGPU+iGPU+10CPU (mode 3)"])
for x, a, b in ((0, pre_m1, pre_m3), (1, post_m1, post_m3)):
    d = (st.mean(b) - st.mean(a)) / st.mean(a) * 100
    y = max(st.mean(a), st.mean(b)) + 8
    ax.annotate(f"iGPU: {d:+.1f}%", (x, y), ha="center", fontsize=11,
                fontweight="bold")
ax.set_xticks([0, 1])
ax.set_xticklabels(["pre-v6 (exp 20, binary-v5 lineage)",
                    "post-v6 (exp 22, periodicity merged)"])
ax.set_ylabel("wall time (s)")
ax.set_ylim(0, 72)
ax.set_title("The iGPU's $-23.2\\%$ collapses to a wash once the CPU grind is gone")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.3, zorder=0)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "igpu22_collapse.png"), dpi=160)

# -------------------------------- fig 2: pool accounting (r2/r3 means)
m1 = [pool_stats(os.path.join(HERE, f"logs/m1.r{r}.stderr")) for r in (2, 3)]
m3 = [pool_stats(os.path.join(HERE, f"logs/m3.r{r}.stderr")) for r in (2, 3)]
fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 4.2))
pb1 = st.mean([s["pool_busy"] for s in m1])
pb3 = st.mean([s["pool_busy"] for s in m3])
axa.bar([0, 1], [pb1, pb3], 0.55, color=[RED, GREEN], zorder=2)
for x, pb, n in ((0, pb1, 11), (1, pb3, 10)):
    axa.annotate(f"{pb:.0f} s / {n} thr\n= {pb / n:.1f} s/thr", (x, pb),
                 textcoords="offset points", xytext=(0, 8), ha="center",
                 fontsize=10)
axa.set_xticks([0, 1])
axa.set_xticklabels(["mode 1\n(11 CPU workers)", "mode 3\n(10 CPU workers)"])
axa.set_ylabel("CPU pool busy time (s)")
axa.set_ylim(0, 360)
axa.set_title("Relief = forfeit: same s/thread")
axa.grid(axis="y", alpha=0.3, zorder=0)

cats = ("CPU pool", "dGPU", "iGPU")
r1 = [st.mean([s["pool_regions"] for s in m1]),
      st.mean([s["dgpu"] for s in m1]), 0]
r3 = [st.mean([s["pool_regions"] for s in m3]),
      st.mean([s["dgpu"] for s in m3]),
      st.mean([s["igpu"] for s in m3])]
xs = range(len(cats))
for i, (vals, c, lab) in enumerate(
        (( r1, RED, "mode 1"), (r3, GREEN, "mode 3"))):
    pos = [x + (i - 0.5) * 0.36 for x in xs]
    axb.bar(pos, vals, 0.33, color=c, label=lab, zorder=2)
    for p, v in zip(pos, vals):
        if v:
            axb.annotate(f"{v:.0f}", (p, v), textcoords="offset points",
                         xytext=(0, 4), ha="center", fontsize=9)
axb.set_xticks(list(xs))
axb.set_xticklabels(cats)
axb.set_ylabel("regions (of 5,608)")
axb.set_title("Where the regions go")
axb.legend(frameon=False)
axb.grid(axis="y", alpha=0.3, zorder=0)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "igpu22_accounting.png"), dpi=160)

# --------------------------------------------- fig 3: the probe ladder
def probe(pat):
    return [wall(p) for p in sorted(glob.glob(os.path.join(HERE, "logs", pat)))]

arms = [
    ("mode 1, 12 thr\n(spin)", probe("probe_m1.t12.r*.stderr"), RED),
    ("mode 3, 13 thr\n(spin)", probe("probe_m3.t13.r*.stderr"), GREEN),
    ("mode 1, 12 thr\n(BlockingSync)", probe("bs_m1.t12.r*.stderr"), RED),
    ("mode 1, 13 thr\n(BlockingSync)", probe("bs_m1.t13.r*.stderr"), BLUE),
    ("mode 3, 13 thr\n(BlockingSync)", probe("bs_m3.t13.r*.stderr"), GREEN),
]
fig, ax = plt.subplots(figsize=(8.6, 4.4))
for x, (lab, vals, c) in enumerate(arms):
    ax.bar(x, st.mean(vals), 0.6, color=c, zorder=2)
    ax.plot([x] * len(vals), vals, "o", ms=4, mfc="white", mec="black",
            mew=0.8, zorder=3)
    ax.annotate(f"{st.mean(vals):.2f}", (x, st.mean(vals)),
                textcoords="offset points", xytext=(0, 12), ha="center",
                fontsize=9)
ax.axvline(1.5, color="grey", lw=0.8, ls="--")
ax.text(0.75, 33.4, "probe 1", ha="center", fontsize=9, color="grey")
ax.text(3.0, 33.4, "probe 2 (kernel.cu one-liner)", ha="center", fontsize=9,
        color="grey")
ax.set_xticks(range(len(arms)))
ax.set_xticklabels([a[0] for a in arms], fontsize=9)
ax.set_ylabel("wall time (s)")
ax.set_ylim(0, 35)
ax.set_title("Oversubscription never beats the 12-thread control")
ax.grid(axis="y", alpha=0.3, zorder=0)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "igpu22_probes.png"), dpi=160)

# --------------------------------------- fig 4: repricing the devices
cpu12_pre, hyb_pre = 164.43, 57.64          # exp 21, matched same-day pair
cpu12_e20, m2_e20 = st.mean(e20[("0",)]), st.mean(e20[("2",)])
cpu12_post, hyb_post = 35.21, 30.94          # exp 21
k_d_pre = 12 * cpu12_pre / hyb_pre - 11
k_i_pre = 12 * cpu12_e20 / m2_e20 - 11
k_d_post = 12 * cpu12_post / hyb_post - 11
k_i_post = 1.0                               # exp-22 wash (range 0.9-1.4)
fig, ax = plt.subplots(figsize=(7.2, 4.4))
xs = [0, 1]
for i, (vals, c, lab) in enumerate(
        (([k_d_pre, k_i_pre], RED, "pre-v6"),
         ([k_d_post, k_i_post], GREEN, "post-v6 (periodicity)"))):
    pos = [x + (i - 0.5) * 0.36 for x in xs]
    ax.bar(pos, vals, 0.33, color=c, label=lab, zorder=2)
    for p, v in zip(pos, vals):
        ax.annotate(f"{v:.1f}", (p, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=10)
ax.axhline(1.0, color="black", lw=0.9, ls=":")
ax.text(-0.44, 1.35, "one CPU worker", fontsize=9, ha="left")
for x, a, b in zip(xs, (k_d_pre, k_i_pre), (k_d_post, k_i_post)):
    ax.annotate(f"$\\div${a / b:.0f}", (x, max(a, b) + 1.2), ha="center",
                fontsize=11, fontweight="bold")
ax.set_ylim(0, 27)
ax.set_xticks(xs)
ax.set_xticklabels(["dGPU (GTX 1660 Ti Max-Q)", "iGPU (UHD 630)"])
ax.set_ylabel("marginal value (CPU-worker equivalents)")
ax.set_title("Periodicity re-priced every accelerator in the system")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.3, zorder=0)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "igpu22_repricing.png"), dpi=160)

print("k_d_pre=%.1f k_i_pre=%.1f k_d_post=%.2f" % (k_d_pre, k_i_pre, k_d_post))
print("wrote 4 figures to", IMG)
