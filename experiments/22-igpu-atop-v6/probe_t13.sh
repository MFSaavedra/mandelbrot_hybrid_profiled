#!/usr/bin/env bash
# Probe: does mode 3 win once it doesn't cost a CPU worker?
# 13 threads = 11 CPU + dGPU lane + iGPU lane, oversubscribing 12 logical CPUs,
# vs the 12-thread mode-1 control, alternated. Appends to logs/.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
for rep in 1 2; do
  for cfg in "1 12" "3 13"; do
    set -- $cfg; mode=$1; thr=$2
    rm -rf "$SCRIPT_DIR/scratch"; mkdir -p "$SCRIPT_DIR/scratch"
    log="$SCRIPT_DIR/logs/probe_m${mode}.t${thr}.r${rep}"
    ( cd "$SCRIPT_DIR/scratch" && "$ROOT/mandelHybrid" "$ROOT/spec.in" "$thr" "$mode" 0.1 32768 1 1 \
        > "$log.stdout" 2> "$log.stderr" )
    head -1 "$log.stderr"
    awk '/\[total_elapsed_s\]/{print "  -> " $2 " s"}' "$log.stderr"
  done
done
rm -rf "$SCRIPT_DIR/scratch"
