#!/usr/bin/env bash
# BlockingSync probe: with cudaDeviceScheduleBlockingSync in CUDAmemSetup,
# does an extra worker now pay?  Three configs, alternated, 2 reps:
#   m1 t12 -- control (also measures BlockingSync's effect on mode 1 itself)
#   m1 t13 -- 12 CPU + sleeping dGPU lane (BlockingSync value without iGPU)
#   m3 t13 -- 11 CPU + both lanes (the exp-22 probe, re-run)
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
for rep in 1 2; do
  for cfg in "1 12" "1 13" "3 13"; do
    set -- $cfg; mode=$1; thr=$2
    rm -rf "$SCRIPT_DIR/scratch"; mkdir -p "$SCRIPT_DIR/scratch"
    log="$SCRIPT_DIR/logs/bs_m${mode}.t${thr}.r${rep}"
    ( cd "$SCRIPT_DIR/scratch" && "$ROOT/mandelHybrid" "$ROOT/spec.in" "$thr" "$mode" 0.1 32768 1 1 \
        > "$log.stdout" 2> "$log.stderr" )
    head -1 "$log.stderr"
    awk '/\[total_elapsed_s\]/{print "  -> " $2 " s"}' "$log.stderr"
  done
done
rm -rf "$SCRIPT_DIR/scratch"
