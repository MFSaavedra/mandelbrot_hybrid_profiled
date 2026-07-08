#!/usr/bin/env python3
"""Unprivileged analysis step for experiment 24 (run after sudo capture.sh).

1. Exports each captured ncu/<regime>_{full,warp}.ncu-rep to a wide raw CSV
   in analysis/ via `ncu --import --page raw --csv` (skipped if present).
2. Aggregates the full-set captures into analysis/summary.csv with exactly
   the experiment-16 reduction (simple mean over captured kernels of the
   same metric columns), validated in-session against exp-16's tracked
   summary: reproduces all four regimes to the printed precision. One known
   slip in the reference: exp-16's exterior dur_ms is recorded in µs
   (614.869); this script emits true ms everywhere.
3. Aggregates the cheap _warp captures (more kernels, better distribution)
   into analysis/warp_dist.csv: mean warp efficiency plus the share of
   kernels below 70%, per regime.
4. Prints a side-by-side vs the exp-16 baseline (the kernel is unchanged
   v5->v6, so report 16's capture is the v6 baseline).
"""
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NCU = os.environ.get("NCU", "/usr/bin/ncu")
ANA = os.path.join(HERE, "analysis")
os.makedirs(ANA, exist_ok=True)

REGIMES = ["interior", "exterior", "boundary", "canonical"]

COLS = {
    "warpEff": "smsp__thread_inst_executed_per_inst_executed.ratio",
    "branchEff": "smsp__sass_average_branch_targets_threads_uniform.pct",
    "fp64_pipe_pct": "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active",
    "compute_SOL_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "mem_SOL_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "dur_ms": "gpu__time_duration.avg",
    "regs": "launch__registers_per_thread",
    "L1_hit_pct": "l1tex__t_sector_hit_rate.pct",
    "L2_hit_pct": "lts__t_sector_hit_rate.pct",
    "clk_GHz": "gpc__cycles_elapsed.avg.per_second",
}


def export(rep, out_csv):
    if os.path.exists(out_csv):
        return
    rep_path = os.path.join(HERE, "ncu", rep)
    if not os.path.exists(rep_path):
        sys.exit(f"missing capture {rep_path} -- run: sudo {HERE}/capture.sh")
    with open(out_csv, "w") as f:
        subprocess.run([NCU, "--import", rep_path, "--page", "raw", "--csv"],
                       stdout=f, check=True)
    print(f"exported {rep} -> {os.path.relpath(out_csv, ROOT)}")


def load(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1], rows[2:]  # header, units, data


def col(hdr, unit, data, name):
    i = hdr.index(name)
    vals = [float(r[i].replace(",", "")) for r in data if r[i] != ""]
    return vals, unit[i]


def mean(v):
    return sum(v) / len(v)


# ---- summary.csv from the _full captures --------------------------------
summary = []
for regime in REGIMES:
    out_csv = os.path.join(ANA, f"{regime}_full.csv")
    export(f"{regime}_full.ncu-rep", out_csv)
    hdr, unit, data = load(out_csv)
    row = {"regime": regime, "kernels": len(data)}
    for k, metric in COLS.items():
        vals, u = col(hdr, unit, data, metric)
        m = mean(vals)
        if k == "warpEff":
            m = m / 32 * 100
        elif k == "dur_ms":
            m = m / 1e6 if u == "ns" else (m / 1e3 if u == "us" else m)
        elif k == "clk_GHz" and u == "Hz":
            m = m / 1e9
        row[k] = round(m, 3)
    summary.append(row)

with open(os.path.join(ANA, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)

# ---- warp_dist.csv from the _warp captures ------------------------------
dist = []
for regime in REGIMES:
    out_csv = os.path.join(ANA, f"{regime}_warp.csv")
    export(f"{regime}_warp.ncu-rep", out_csv)
    hdr, unit, data = load(out_csv)
    vals, _ = col(hdr, unit, data, COLS["warpEff"])
    eff = [v / 32 * 100 for v in vals]
    dist.append({
        "regime": regime,
        "kernels": len(eff),
        "warpEff_mean": round(mean(eff), 3),
        "warpEff_min": round(min(eff), 3),
        "pct_below_70": round(100 * sum(1 for e in eff if e < 70) / len(eff), 1),
        "pct_below_90": round(100 * sum(1 for e in eff if e < 90) / len(eff), 1),
    })

with open(os.path.join(ANA, "warp_dist.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(dist[0].keys()))
    w.writeheader()
    w.writerows(dist)

# ---- side-by-side vs the exp-16 baseline --------------------------------
base = {}
with open(os.path.join(ROOT, "experiments/16-ncu-divergence/analysis/summary.csv")) as f:
    for r in csv.DictReader(f):
        base[r["regime"]] = r

print("\n== v7 (this capture) vs v6 baseline (experiment 16) ==")
hdrline = f"{'regime':<10} {'metric':<16} {'v6 (exp16)':>12} {'v7':>12}"
print(hdrline)
print("-" * len(hdrline))
for row in summary:
    b = base[row["regime"]]
    for k in ["warpEff", "branchEff", "fp64_pipe_pct", "compute_SOL_pct",
              "occupancy_pct", "dur_ms", "regs"]:
        print(f"{row['regime']:<10} {k:<16} {float(b[k]):>12.3f} {row[k]:>12.3f}")
    print("-" * len(hdrline))
print("(exp-16 exterior dur_ms is in µs -- recording slip in the reference)")
print("\nwrote analysis/summary.csv, analysis/warp_dist.csv")
