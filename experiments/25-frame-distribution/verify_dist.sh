#!/usr/bin/env bash
# Byte-identity check for the distributed frame renderer: N simulated nodes on
# THIS machine (hosts file of ":" entries, no SSH) vs a single-process run of
# the same binary, spec, and config — mirroring experiments/21 and /23's
# verify_identity.sh methodology.
#
# Identity is only well-defined with the executor pinned: CPU and CUDA FP64
# differ in last-ulp rounding on deep-frame boundary pixels (report 20:
# GPU-vs-CPU ~8% of pixels), and the hybrid's region->executor assignment is
# timing-dependent — so, like experiments 21 (mode-0 cmp) and 23 (GPU-only),
# the A/B runs CPU-only (12 0) and GPU-only (1 1).  JOBS=1 serializes the
# simulated nodes, which share this machine's cores and GPU (identity does not
# depend on timing, so contention hygiene is all that matters).
#
# Checks (production spec, 100 frames, 1920x1080, diffT=0.1):
#   1. CPU-only, pure cyclic  (block=1, N nodes)
#   2. GPU-only, pure cyclic  (block=1, N nodes)
#   3. CPU-only, block-cyclic (block=8, N nodes)
# Each check asserts: union covers all frames, and every PNG is byte-identical
# (cmp) to the single-process reference.
#
# Usage:  experiments/25-frame-distribution/verify_dist.sh [NNODES]   # default 3
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
BIN=$PROJECT_ROOT/mandelHybrid
SPEC=$PROJECT_ROOT/spec.in
NNODES=${1:-3}
WORK=$SCRIPT_DIR/verify_work          # gitignored scratch (PNG trees)
CSV=$SCRIPT_DIR/results_identity.csv  # tracked result

[[ -x "$BIN" ]] || { echo "binary $BIN not found — run make first" >&2; exit 1; }
numframes=$(awk 'NR==1{print $1}' "$SPEC")
prefix=$(awk 'NR==1{print $4}' "$SPEC")

rm -rf "$WORK"; mkdir -p "$WORK"
hosts=$WORK/hosts.txt
for i in $(seq 1 "$NNODES"); do echo ":"; done > "$hosts"

echo "mode,block,nodes,frames_total,frames_collected,frames_identical" > "$CSV"
fail=0

# check_mode LABEL "RUN_ARGS" BLOCK — one reference (cached per LABEL) vs one
# distributed collection; appends a CSV row.
check_mode() {
  local label=$1 args=$2 block=$3
  local ref=$WORK/ref_$label collect=$WORK/collect_${label}_b${block}

  if [[ ! -d "$ref" ]]; then
    echo "[verify] reference run: $label ($args), single process, $numframes frames"
    mkdir -p "$ref"
    (cd "$ref" && "$BIN" "$SPEC" $args 2> ref.stderr)
  fi

  echo "[verify] distributed run: $label block=$block over $NNODES local nodes"
  mkdir -p "$collect"
  OUT=$collect RDIR=$PROJECT_ROOT ARGS=$args BLOCK=$block JOBS=1 \
    "$PROJECT_ROOT/scripts/dist_frames.sh" "$hosts"

  local got same
  got=$(ls "$collect/$prefix"*.png 2>/dev/null | wc -l)
  same=0
  for f in "$ref/$prefix"*.png; do
    if cmp -s "$f" "$collect/$(basename "$f")" 2>/dev/null; then
      same=$((same + 1))
    fi
  done
  echo "$label,$block,$NNODES,$numframes,$got,$same" >> "$CSV"
  echo "[verify] $label block=$block: collected $got/$numframes, identical $same/$numframes"
  [[ "$got" -eq "$numframes" && "$same" -eq "$numframes" ]] || fail=1
}

check_mode cpu "12 0 0.1 32768 1 1" 1
check_mode gpu "1 1 0.1 32768 1 1"  1
check_mode cpu "12 0 0.1 32768 1 1" 8

echo "[verify] results: $CSV"
if [[ "$fail" -ne 0 ]]; then
  echo "[verify] FAIL — at least one mode is not byte-identical" >&2
  exit 1
fi
echo "[verify] PASS — all modes byte-identical to the single-process reference"
