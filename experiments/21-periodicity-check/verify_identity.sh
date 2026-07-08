#!/usr/bin/env bash
#
# Byte-identity check for the periodicity check: the exact-equality cycle
# detection must not change a single pixel. Renders the full production spec
# CPU-only (mode 0, so every pixel goes through the modified diverge()) with
# the baseline and the periodicity binaries, then byte-compares all frames.
# Mode 0 output is deterministic (pixel values never depend on the
# decomposition or on which thread computes them), so cmp is exact.
# The GPU path is untouched by the branch (kernel.cu unchanged), so mode-0
# identity covers the whole change.
#
# Usage: BASE_BIN=/path/to/main/mandelHybrid experiments/21-periodicity-check/verify_identity.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
SPEC="$ROOT/spec.in"
BASE_BIN=${BASE_BIN:?set BASE_BIN to the baseline (main) mandelHybrid}
PC_BIN="$ROOT/mandelHybrid"
WORK=${WORK:-$SCRIPT_DIR/identity_scratch}

for pair in "base $BASE_BIN" "pc $PC_BIN"; do
  set -- $pair
  rm -rf "$WORK/$1"; mkdir -p "$WORK/$1"
  echo "[identity] rendering $1 (mode 0, 12 threads) ..."
  ( cd "$WORK/$1" && "$2" "$SPEC" 12 0 0.1 32768 1 1 > run.stdout 2> run.stderr )
done

bad=0; n=0
for p in "$WORK"/base/img*.png; do
  f=$(basename "$p")
  n=$((n+1))
  cmp -s "$p" "$WORK/pc/$f" || { bad=$((bad+1)); echo "[identity] MISMATCH $f"; }
done
echo "[identity] $((n-bad)) / $n frames byte-identical"
[[ $bad -eq 0 ]] && echo "[identity] PASS" || { echo "[identity] FAIL"; exit 1; }
