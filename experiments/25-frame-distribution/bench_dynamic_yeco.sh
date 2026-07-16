#!/usr/bin/env bash
# Weighted-static vs dynamic-stealing benchmark on the laptop+yeco pair --
# the report-26 protocol (bench_dynamic.sh) rerun with a remote node ~4.1x
# FASTER than the laptop (bench_2node_yeco.sh measured laptop 21.02 s vs
# yeco solo 5.15 s -> right weights 1:4, the mirror of ivy's 5:1; analytic
# weighted ideal 1/(1/21.02+1/5.15) = 4.14 s + overhead).
#   laptop_hybrid    single-process baseline (same-day anchor)
#   weighted_static  DIST_WEIGHTS=1,4 in the binary via dist_frames.sh:
#                    one invocation per node, seeded weighted-random shares
#   dyn_14           dist_dynamic.sh WEIGHTS=1,4 -- right weights + stealing
#   dyn_11           dist_dynamic.sh WEIGHTS=1,1 -- WRONG (equal) weights +
#                    stealing: how much of equal-share's stranding the
#                    stealing recovers when the SLOW node is overloaded
# New exposure vs the ivy pair: every dynamic chunk is a fresh mandelHybrid
# invocation, so yeco pays CUDA-context init + a WAN ssh/scp roundtrip per
# chunk against only ~4 s of total compute -- per-chunk overhead is
# proportionally far dearer than on CPU-only LAN ivy.
# Production spec, save=1, 3 reps, laptop on AC (yeco is a desktop).
# Rows: config,rep,wall_e2e_s,laptop_s,yeco_s,laptop_frames,yeco_frames
#   laptop_s/yeco_s: static = the rank's [total_elapsed_s]; dynamic = the
#   rank's busy time (sum of its chunks' [total_elapsed_s]).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
BIN=$PROJECT_ROOT/mandelHybrid
SPEC=$PROJECT_ROOT/spec.in
NODE=yeco
NODE_RDIR=/home/mfsaaved/box/mandelbrot_hybrid_profiled
LAPTOP_ARGS="12 1 0.1 32768 1 1"
NODE_ARGS="20 1 0.1 32768 1 1"
REPS=${REPS:-3}

WORK=$SCRIPT_DIR/bench_yeco     # gitignored scratch (shared with bench_2node_yeco)
CSV=$SCRIPT_DIR/results_dynamic_yeco.csv
mkdir -p "$WORK/scratch"

cat /sys/class/power_supply/AC*/online 2>/dev/null | grep -q 1 \
    || { echo "laptop not on AC — aborting" >&2; exit 1; }

printf ': - %s\n%s %s %s\n' "$LAPTOP_ARGS" "$NODE" "$NODE_RDIR" "$NODE_ARGS" > "$WORK/hosts_fwd.txt"
echo "config,rep,wall_e2e_s,laptop_s,yeco_s,laptop_frames,yeco_frames" > "$CSV"

elapsed_of() { grep '^\[total_elapsed_s\]' "$1" | awk '{print $2}'; }
owned_of()   { grep -oP 'ownedFrames=\K[0-9]+' "$1"; }

for rep in $(seq 1 "$REPS"); do
  # -- baseline --
  rm -f "$WORK/scratch"/img*.png
  (cd "$WORK/scratch" && "$BIN" "$SPEC" $LAPTOP_ARGS 2> "$WORK/dbase.r${rep}.stderr")
  e=$(elapsed_of "$WORK/dbase.r${rep}.stderr")
  echo "laptop_hybrid,$rep,$e,$e,,100," >> "$CSV"; tail -1 "$CSV"

  # -- weighted static 1:4 (binary DIST_WEIGHTS, one shot per node) --
  out=$WORK/collect_ws
  t0=$(date +%s.%N)
  OUT=$out WEIGHTS=1,4 SPEC=$SPEC "$PROJECT_ROOT/scripts/dist_frames.sh" \
      "$WORK/hosts_fwd.txt" > "$WORK/ws.r${rep}.log" 2>&1
  t1=$(date +%s.%N)
  [[ $(ls "$out"/img*.png | wc -l) -eq 100 ]] || { echo "ws rep$rep incomplete" >&2; exit 1; }
  echo "weighted_static,$rep,$(echo "$t1 $t0" | awk '{printf "%.3f", $1-$2}'),$(elapsed_of "$out/logs/rank0.stderr"),$(elapsed_of "$out/logs/rank1.stderr"),$(owned_of "$out/logs/rank0.stderr"),$(owned_of "$out/logs/rank1.stderr")" >> "$CSV"
  tail -1 "$CSV"

  # -- dynamic stealing, right (1,4) and wrong (1,1) weights --
  for wcfg in 1,4 1,1; do
    label=dyn_${wcfg/,/}
    out=$WORK/collect_$label
    OUT=$out WEIGHTS=$wcfg SPEC=$SPEC "$PROJECT_ROOT/scripts/dist_dynamic.sh" \
        "$WORK/hosts_fwd.txt" > "$WORK/$label.r${rep}.log" 2>&1
    e=$(grep -oP '^\[dyn\] wall_e2e_s \K[0-9.]+' "$WORK/$label.r${rep}.log")
    # bag/stats.csv rows: rank,rendered,stolen,busy
    l=$(grep '^0,' "$out/bag/stats.csv"); y=$(grep '^1,' "$out/bag/stats.csv")
    echo "$label,$rep,$e,$(cut -d, -f4 <<<"$l"),$(cut -d, -f4 <<<"$y"),$(cut -d, -f2 <<<"$l"),$(cut -d, -f2 <<<"$y")" >> "$CSV"
    tail -1 "$CSV"
  done
done
echo "[bench] done -> $CSV"
