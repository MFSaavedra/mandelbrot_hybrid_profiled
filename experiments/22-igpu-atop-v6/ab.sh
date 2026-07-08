#!/usr/bin/env bash
#
# Experiment 22: re-price the iGPU atop the periodicity check (binary-v6).
#
# Report 20 measured dGPU+iGPU+10CPU (mode 3) at -23.2% vs dGPU+11CPU
# (mode 1) -- but that was priced against a CPU pool grinding interior
# pixels to MAXITER. binary-v6-periodicity (report 21) eliminated that
# grind (hybrid 57.6 -> 30.9 s), so the iGPU's marginal value must be
# re-measured before merging feat/igpu-opencl (report 21, rec. 2).
#
# Single binary (main merged into feat/igpu-opencl: periodicity + OpenCL
# backend), two configs, production spec.in, diffT=0.1, pixT=32768,
# quiet=1, save=1 (Fig-11.15 methodology, same as the exp-20 and exp-21
# headlines). Modes alternate inside each rep so thermal drift hits both
# sides equally. Mode 1 doubles as calibration: it must reproduce
# experiment 21's pc hybrid (30.94 s +/- 0.56).
#
# Usage:  experiments/22-igpu-atop-v6/ab.sh      (REPS=3, WORK overridable)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
SPEC="$ROOT/spec.in"
BIN="$ROOT/mandelHybrid"
WORK=${WORK:-$SCRIPT_DIR/scratch}
CSV="$SCRIPT_DIR/results.csv"
LOGS="$SCRIPT_DIR/logs"
REPS=${REPS:-3}

[[ -x "$BIN" ]] || { echo "missing binary $BIN" >&2; exit 1; }
mkdir -p "$LOGS"
echo "config,numThreads,gpuMode,rep,elapsed_s" > "$CSV"

for rep in $(seq 1 "$REPS"); do
  for cfg in "dGPU+11CPU 12 1" "dGPU+iGPU+10CPU 12 3"; do
    set -- $cfg; name=$1; thr=$2; mode=$3
    rm -rf "$WORK"; mkdir -p "$WORK"
    log="$LOGS/m${mode}.r${rep}"
    ( cd "$WORK" && "$BIN" "$SPEC" "$thr" "$mode" 0.1 32768 1 1 \
        > "$log.stdout" 2> "$log.stderr" )
    t=$(awk '/\[total_elapsed_s\]/{print $2}' "$log.stderr")
    echo "$name,$thr,$mode,$rep,$t" | tee -a "$CSV"
  done
done
rm -rf "$WORK"
echo "[ab] done -> $CSV"
