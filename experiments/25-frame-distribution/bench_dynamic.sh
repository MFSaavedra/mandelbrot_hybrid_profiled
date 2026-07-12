#!/usr/bin/env bash
# Weighted-static vs dynamic-stealing benchmark on the laptop+ivy pair --
# the follow-up to bench_2node.sh's equal-share rows (report 25 measured the
# 4.56x throughput gap and predicted a ~15.4 s ideal weighted wall).
#   laptop_hybrid    single-process baseline (same-day anchor)
#   weighted_static  DIST_WEIGHTS=5,1 in the binary via dist_frames.sh:
#                    one invocation per node, seeded weighted-random shares
#   dyn_51           dist_dynamic.sh WEIGHTS=5,1 -- right weights + stealing
#   dyn_11           dist_dynamic.sh WEIGHTS=1,1 -- WRONG (equal) weights +
#                    stealing: measures how much of static equal-share's
#                    +150% (46.9 s) the stealing recovers on its own
# Production spec, save=1, 3 reps, both machines on AC (gated).
# Rows: config,rep,wall_e2e_s,laptop_s,ivy_s,laptop_frames,ivy_frames
#   laptop_s/ivy_s: static = the rank's [total_elapsed_s]; dynamic = the
#   rank's busy time (sum of its chunks' [total_elapsed_s]).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
BIN=$PROJECT_ROOT/mandelHybrid
SPEC=$PROJECT_ROOT/spec.in
IVY=ivy
IVY_RDIR=/home/lynx/box/mandelbrot_hybrid_profiled
LAPTOP_ARGS="12 1 0.1 32768 1 1"
IVY_ARGS="8 0 0.1 32768 1 1"
REPS=${REPS:-3}

WORK=$SCRIPT_DIR/bench          # gitignored scratch (shared with bench_2node)
CSV=$SCRIPT_DIR/results_dynamic.csv
mkdir -p "$WORK/scratch"

on_ac() { local h=$1; if [[ $h == local ]]; then cat /sys/class/power_supply/AC*/online 2>/dev/null | grep -q 1;
          else ssh "$h" 'cat /sys/class/power_supply/AC*/online 2>/dev/null' | grep -q 1; fi }
on_ac local || { echo "laptop not on AC — aborting" >&2; exit 1; }
on_ac $IVY  || { echo "ivy not on AC — aborting" >&2; exit 1; }

printf ': - %s\nivy %s %s\n' "$LAPTOP_ARGS" "$IVY_RDIR" "$IVY_ARGS" > "$WORK/hosts_fwd.txt"
echo "config,rep,wall_e2e_s,laptop_s,ivy_s,laptop_frames,ivy_frames" > "$CSV"

elapsed_of() { grep '^\[total_elapsed_s\]' "$1" | awk '{print $2}'; }
owned_of()   { grep -oP 'ownedFrames=\K[0-9]+' "$1"; }

for rep in $(seq 1 "$REPS"); do
  # -- baseline --
  rm -f "$WORK/scratch"/img*.png
  (cd "$WORK/scratch" && "$BIN" "$SPEC" $LAPTOP_ARGS 2> "$WORK/dbase.r${rep}.stderr")
  e=$(elapsed_of "$WORK/dbase.r${rep}.stderr")
  echo "laptop_hybrid,$rep,$e,$e,,100," >> "$CSV"; tail -1 "$CSV"

  # -- weighted static 5:1 (binary DIST_WEIGHTS, one shot per node) --
  out=$WORK/collect_ws
  t0=$(date +%s.%N)
  OUT=$out WEIGHTS=5,1 SPEC=$SPEC "$PROJECT_ROOT/scripts/dist_frames.sh" \
      "$WORK/hosts_fwd.txt" > "$WORK/ws.r${rep}.log" 2>&1
  t1=$(date +%s.%N)
  [[ $(ls "$out"/img*.png | wc -l) -eq 100 ]] || { echo "ws rep$rep incomplete" >&2; exit 1; }
  echo "weighted_static,$rep,$(echo "$t1 $t0" | awk '{printf "%.3f", $1-$2}'),$(elapsed_of "$out/logs/rank0.stderr"),$(elapsed_of "$out/logs/rank1.stderr"),$(owned_of "$out/logs/rank0.stderr"),$(owned_of "$out/logs/rank1.stderr")" >> "$CSV"
  tail -1 "$CSV"

  # -- dynamic stealing, right (5,1) and wrong (1,1) weights --
  for wcfg in 5,1 1,1; do
    label=dyn_${wcfg/,/}
    out=$WORK/collect_$label
    OUT=$out WEIGHTS=$wcfg SPEC=$SPEC "$PROJECT_ROOT/scripts/dist_dynamic.sh" \
        "$WORK/hosts_fwd.txt" > "$WORK/$label.r${rep}.log" 2>&1
    e=$(grep -oP '^\[dyn\] wall_e2e_s \K[0-9.]+' "$WORK/$label.r${rep}.log")
    # bag/stats.csv rows: rank,rendered,stolen,busy
    l=$(grep '^0,' "$out/bag/stats.csv"); i=$(grep '^1,' "$out/bag/stats.csv")
    echo "$label,$rep,$e,$(cut -d, -f4 <<<"$l"),$(cut -d, -f4 <<<"$i"),$(cut -d, -f2 <<<"$l"),$(cut -d, -f2 <<<"$i")" >> "$CSV"
    tail -1 "$CSV"
  done
done
echo "[bench] done -> $CSV"
