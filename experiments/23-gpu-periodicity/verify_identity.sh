#!/usr/bin/env bash
#
# Byte-identity check for the GPU-side periodicity check: the exact-equality
# cycle detection must not change a single pixel *of the GPU's own output*.
# Renders the full production spec GPU-only (numThreads=1, gpuEnable=1: every
# region of every class through the modified kernel) with the baseline and
# the gpu-periodicity binaries, then byte-compares all frames.
# Identity is expected from the exactness argument (an exactly-repeating
# orbit can never escape, so the early return equals the ground-out value)
# PROVIDED nvcc contracts the iteration arithmetic identically in both
# builds; a mismatch here means codegen changed the FP stream, not that the
# check is wrong -- fall back to a report-20-style magnitude analysis.
#
# Usage: BASE_BIN=/path/to/main/mandelHybrid experiments/23-gpu-periodicity/verify_identity.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
SPEC="$ROOT/spec.in"
BASE_BIN=${BASE_BIN:?set BASE_BIN to the baseline (main) mandelHybrid}
GP_BIN="$ROOT/mandelHybrid"
WORK=${WORK:-$SCRIPT_DIR/identity_scratch}

for pair in "base $BASE_BIN" "gp $GP_BIN"; do
  set -- $pair
  rm -rf "$WORK/$1"; mkdir -p "$WORK/$1"
  echo "[identity] rendering $1 (GPU-only, 1 thread) ..."
  ( cd "$WORK/$1" && "$2" "$SPEC" 1 1 0.1 32768 1 1 > run.stdout 2> run.stderr )
  awk '/\[total_elapsed_s\]/{print "[identity] '"$1"' wall: " $2 " s"}' "$WORK/$1/run.stderr"
done

bad=0; n=0
for p in "$WORK"/base/img*.png; do
  f=$(basename "$p")
  n=$((n+1))
  cmp -s "$p" "$WORK/gp/$f" || { bad=$((bad+1)); echo "[identity] MISMATCH $f"; }
done
echo "[identity] $((n-bad)) / $n frames byte-identical"
[[ $bad -eq 0 ]] && echo "[identity] PASS" || { echo "[identity] FAIL"; exit 1; }
