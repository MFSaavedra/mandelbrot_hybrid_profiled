#!/usr/bin/env bash
# Headline best-config A/B for the integrated-GPU (OpenCL) backend.
#
# Mirrors scripts/sweep_fig1115.sh (same spec, same timing extraction, same
# quiet=1/save=1 batch settings) but fixes the thread count at 12 and varies
# only the backend mode (arg3 = gpuMode), so the four runs are a clean disjoint
# A/B on one binary in one thermal state:
#
#   CPU12            12 0  -- pure CPU floor (no accelerator)
#   dGPU+11CPU       12 1  -- current best (report 12, binary-v5-affinity)
#   dGPU+iGPU+10CPU  12 3  -- iGPU added: does it beat the 11th CPU worker it displaces?
#   iGPU+11CPU       12 2  -- iGPU instead of the dGPU
#
# All at diffT=0.1 (the established sweet spot, report 06/12), the operating
# point feat/igpu-opencl sits on. Run on AC power (battery inverts the split).
#
# Usage:  experiments/20-igpu-opencl/sweep_igpu.sh
#   REPS=3 DIFFT=0.1 SAVE=1 override as needed.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

REPS=${REPS:-3}
OUT=${OUT:-$SCRIPT_DIR}
SPEC=${SPEC:-$PROJECT_ROOT/spec.in}
BIN=${BIN:-$PROJECT_ROOT/mandelHybrid}
SAVE=${SAVE:-1}
DIFFT=${DIFFT:-0.1}
PIXT=${PIXT:-32768}

[[ -x "$BIN" ]]  || { echo "Binary $BIN not found — run make first." >&2; exit 1; }
[[ -f "$SPEC" ]] || { echo "Spec file $SPEC not found." >&2; exit 1; }

OUT_ABS=$(readlink -f "$OUT"); mkdir -p "$OUT_ABS/logs"
SCRATCH="$OUT_ABS/scratch_img"
SPEC_ABS=$(readlink -f "$SPEC"); BIN_ABS=$(readlink -f "$BIN")

# label, numThreads, gpuMode
CONFIGS=(
  "CPU12            12  0"
  "dGPU+11CPU       12  1"
  "dGPU+iGPU+10CPU  12  3"
  "iGPU+11CPU       12  2"
)

CSV="$OUT_ABS/results.csv"
echo "label,numThreads,gpuMode,rep,elapsed_s" > "$CSV"
echo "[sweep] reps=$REPS save=$SAVE diffT=$DIFFT pixT=$PIXT spec=$SPEC_ABS"
echo "[sweep] $(date -Iseconds) starting ${#CONFIGS[@]} configs"

for entry in "${CONFIGS[@]}"; do
  # shellcheck disable=SC2086
  set -- $entry
  label=$1; nthr=$2; mode=$3
  for r in $(seq 1 "$REPS"); do
    rm -rf "$SCRATCH" && mkdir -p "$SCRATCH"
    log="$OUT_ABS/logs/${label}.r${r}.log"
    printf "[sweep] %-16s rep %d/%d ... " "$label" "$r" "$REPS"
    pushd "$SCRATCH" >/dev/null
    "$BIN_ABS" "$SPEC_ABS" "$nthr" "$mode" "$DIFFT" "$PIXT" 1 "$SAVE" \
        > "$log.stdout" 2> "$log.stderr"
    popd >/dev/null
    elapsed=$(grep '^\[total_elapsed_s\]' "$log.stderr" | awk '{print $2}')
    echo "$label,$nthr,$mode,$r,$elapsed" >> "$CSV"
    echo "${elapsed}s"
  done
done

rm -rf "$SCRATCH"
echo "[sweep] done. Results: $CSV"
