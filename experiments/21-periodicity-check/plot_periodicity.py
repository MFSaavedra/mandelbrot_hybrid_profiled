#!/usr/bin/env python3
"""Figures for report 21 (exact Brent periodicity check).

Reads:
  experiments/21-periodicity-check/results.csv          (A/B walls)
  experiments/21-periodicity-check/logs/*.stderr        (summaries, rep logs)
  experiments/21-periodicity-check/logs/dist_pc.stderr  (per-region, periodicity)
  experiments/12-gpu-affinity/logs/dist_aff.stderr      (per-region, baseline)

Writes reports/img/periodicity_{wall,dist,split,orbit}.png
(gitignored, regenerated on demand like the other report figures).
"""
import csv, re, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP  = ROOT / "experiments/21-periodicity-check"
IMG  = ROOT / "reports/img"
IMG.mkdir(exist_ok=True)

RED, GREEN, BLUE = "#C00000", "#2E8B57", "#1f3a73"

# ---------------------------------------------------------------- wall A/B --
walls = defaultdict(list)
for r in csv.DictReader(open(EXP / "results.csv")):
    walls[(r["binary"], r["config"])].append(float(r["elapsed_s"]))

fig, ax = plt.subplots(figsize=(7, 4.2))
configs = [("cpu12", "CPU12\n(mode 0)"), ("hybrid", "dGPU+11CPU\n(mode 1)")]
x = np.arange(len(configs)); w = 0.34
for off, (bin_, col, lbl) in ((-w/2, ("base", RED, "main (binary-v5 lineage)")),
                              (+w/2, ("pc", GREEN, "periodicity check"))):
    means = [np.mean(walls[(bin_, c)]) for c, _ in configs]
    ax.bar(x + off, means, w, color=col, label=lbl, zorder=2)
    for i, (c, _) in enumerate(configs):
        ax.scatter([x[i] + off] * len(walls[(bin_, c)]), walls[(bin_, c)],
                   color="black", s=12, zorder=3)
        ax.annotate(f"{means[i]:.1f}", (x[i] + off, means[i]),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
for i, (c, _) in enumerate(configs):
    b, p = np.mean(walls[("base", c)]), np.mean(walls[("pc", c)])
    ax.annotate(f"$-${100*(b-p)/b:.1f}%  ({b/p:.2f}$\\times$)",
                (x[i], max(b, p) + 12), ha="center", fontsize=11, color=BLUE)
ax.set_ylim(0, 195)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in configs])
ax.set_ylabel("wall time (s)  — 100 frames, save=1")
ax.set_title("Periodicity check: end-to-end wall, 3 reps (dots) and means")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(IMG / "periodicity_wall.png", dpi=150)

# ------------------------------------------------ per-region distributions --
re_cpu = re.compile(r"\[CPU region\s+\d+\] frame\s+(\d+) depth \d+ size\s+\d+x\s*\d+"
                    r"\s+compute ([\d.]+) ms")
def cpu_ms(path):
    return np.array([float(m[2]) for line in open(path, errors="replace")
                     if (m := re_cpu.search(line))])

base_ms = cpu_ms(ROOT / "experiments/12-gpu-affinity/logs/dist_aff.stderr")
pc_ms   = cpu_ms(EXP / "logs/dist_pc.stderr")

fig, ax = plt.subplots(figsize=(7, 4.2))
for ms, col, lbl in ((base_ms, RED,  f"main: {len(base_ms)} CPU regions"),
                     (pc_ms, GREEN, f"periodicity: {len(pc_ms)} CPU regions")):
    xs = np.sort(ms); ys = 1.0 - np.arange(1, len(xs) + 1) / len(xs)
    ax.step(xs, ys, where="post", color=col, lw=2, label=lbl)
    ax.axvline(np.median(ms), color=col, ls=":", alpha=0.7)
ax.set_xscale("log")
ax.set_xlabel("per-region CPU compute time (ms, log scale)")
ax.set_ylabel("fraction of regions costlier than x")
ax.set_title("CPU per-region cost, hybrid mode 1 (dotted lines: medians)")
ax.legend(); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(IMG / "periodicity_dist.png", dpi=150)
print(f"[dist] base: median {np.median(base_ms):.1f} ms  max {base_ms.max():.0f} ms  "
      f"sum {base_ms.sum()/1000:.1f} s over {len(base_ms)} regions")
print(f"[dist] pc  : median {np.median(pc_ms):.1f} ms  max {pc_ms.max():.0f} ms  "
      f"sum {pc_ms.sum()/1000:.1f} s over {len(pc_ms)} regions")

# ------------------------------------------------------- executor rebalance --
def summar(path):
    txt = open(path, errors="replace").read()
    gpu_n  = int(re.search(r"Regions computed on GPU : (\d+)", txt)[1])
    kern   = float(re.search(r"Total kernel time\s+: ([\d.]+) ms", txt)[1]) / 1000
    cpu_n  = sum(int(m) for m in re.findall(r"Regions computed on CPU : (\d+)", txt))
    pool   = sum(float(m) for m in
                 re.findall(r"Total compute time\s+: ([\d.]+) ms", txt)) / 1000
    return gpu_n, cpu_n, kern, pool

reps = range(1, 4)
B = [summar(EXP / f"logs/base.hybrid.r{r}.stderr") for r in reps]
P = [summar(EXP / f"logs/pc.hybrid.r{r}.stderr") for r in reps]
bg, bc = np.mean([s[0] for s in B]), np.mean([s[1] for s in B])
pg, pc_ = np.mean([s[0] for s in P]), np.mean([s[1] for s in P])
bk, bp = np.mean([s[2] for s in B]), np.mean([s[3] for s in B])
pk, pp = np.mean([s[2] for s in P]), np.mean([s[3] for s in P])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.0))
x = [0, 1]
a1.bar(x, [bg, pg], 0.5, color=BLUE, label="GPU (largest regions)")
a1.bar(x, [bc, pc_], 0.5, bottom=[bg, pg], color="#e0a800", label="CPU pool")
for xi, (g, c) in zip(x, ((bg, bc), (pg, pc_))):
    a1.annotate(f"{g:.0f}", (xi, g / 2), ha="center", color="white", fontweight="bold")
    a1.annotate(f"{c:.0f}", (xi, g + c / 2), ha="center", fontweight="bold")
a1.set_xticks(x); a1.set_xticklabels(["main", "periodicity"])
a1.set_ylabel("regions (of 5,608)"); a1.set_title("who computes the regions")
a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
a2.bar([xi - 0.19 for xi in x], [bk, pk], 0.36, color=BLUE, label="GPU kernel total")
a2.bar([xi + 0.19 for xi in x], [bp / 11, pp / 11], 0.36, color="#e0a800",
       label="CPU pool per thread")
for xi, v in zip([xi - 0.19 for xi in x], [bk, pk]):
    a2.annotate(f"{v:.1f}", (xi, v), ha="center", va="bottom")
for xi, v in zip([xi + 0.19 for xi in x], [bp / 11, pp / 11]):
    a2.annotate(f"{v:.1f}", (xi, v), ha="center", va="bottom")
a2.set_xticks(x); a2.set_xticklabels(["main", "periodicity"])
a2.set_ylabel("busy seconds"); a2.set_title("how long each executor grinds")
a2.legend();
for a in (a1, a2): a.grid(axis="y", alpha=0.3)
fig.suptitle("Hybrid mode 1 self-rebalances (means over 3 reps)")
fig.tight_layout(); fig.savefig(IMG / "periodicity_split.png", dpi=150)
print(f"[split] regions GPU {bg:.0f}->{pg:.0f}, CPU {bc:.0f}->{pc_:.0f}; "
      f"kernel {bk:.1f}->{pk:.1f} s; pool {bp:.1f}->{pp:.1f} s")

# ------------------------------------------------------------ orbit diagram --
def orbit_with_brent(cx, cy, maxiter=100000):
    """Replicates diverge(): returns trajectory, checkpoints, detection iter."""
    vx, vy = cx, cy
    sx, sy = vx, vy
    nxt, traj, saves = 8, [(vx, vy)], [(0, vx, vy)]
    dists = []
    for it in range(1, maxiter + 1):
        vx, vy = vx * vx - vy * vy + cx, 2 * vx * vy + cy
        traj.append((vx, vy))
        dists.append(np.hypot(vx - sx, vy - sy))
        if vx == sx and vy == sy:
            return np.array(traj), saves, it, np.array(dists)
        if it == nxt:
            sx, sy = vx, vy
            saves.append((it, vx, vy))
            nxt *= 2
    return np.array(traj), saves, None, np.array(dists)

c = (-0.12, 0.75)                       # inside the period-3 bulb
traj, saves, det, dists = orbit_with_brent(*c)
print(f"[orbit] c = {c}, exact FP cycle detected at iteration {det}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.2))
n = det + 1
a1.plot(traj[:n, 0], traj[:n, 1], "-", color="0.75", lw=0.6, zorder=1)
a1.scatter(traj[:n, 0], traj[:n, 1], c=np.arange(n), cmap="viridis", s=8, zorder=2)
sv = np.array([(x, y) for _, x, y in saves])
a1.scatter(sv[:, 0], sv[:, 1], marker="s", s=70, facecolors="none",
           edgecolors=RED, lw=1.8, zorder=3, label="saved state (iter 8, 16, 32, ...)")
a1.scatter([traj[det, 0]], [traj[det, 1]], marker="*", s=220, color=GREEN,
           zorder=4, label=f"exact revisit at iter {det}")
a1.set_title(f"orbit of interior c = {c[0]}+{c[1]}i settling into its 3-cycle")
a1.set_xlabel("Re(z)"); a1.set_ylabel("Im(z)"); a1.legend(fontsize=8, loc="lower left")
a1.set_aspect("equal")
it = np.arange(1, len(dists[:n]) + 1)
pos = dists[:n] > 0
a2.semilogy(it[pos], dists[:n][pos], ".-", color=BLUE, ms=3, lw=0.7)
for k, _, _ in saves[1:]:
    a2.axvline(k, color=RED, ls=":", alpha=0.6)
a2.scatter([det], [1e-19], marker="*", s=220, color=GREEN, zorder=4, clip_on=False)
a2.set_ylim(1e-19, 10)
a2.set_title("distance to saved state (red: checkpoint refresh)\nstar: exact FP equality -> return MAXITER")
a2.set_xlabel("iteration"); a2.set_ylabel(r"$|z_n - z_{saved}|$")
a2.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(IMG / "periodicity_orbit.png", dpi=150)
print("[ok] wrote", IMG / "periodicity_{wall,dist,split,orbit}.png")
