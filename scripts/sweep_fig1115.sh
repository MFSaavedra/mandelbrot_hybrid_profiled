#!/usr/bin/env bash
# Sweep driver that replicates Fig 11.15 of Barlas, Multicore & GPU Programming (2e).
#
# Runs the hybrid Mandelbrot renderer across a set of (numThreads, gpuEnable)
# configurations N times each and writes one CSV row per run.  The CSV is
# consumed by plot_fig1115.py.
#
# The script auto-locates the project root from its own path, so it works
# from any cwd.  Run it as scripts/sweep_fig1115.sh (or just give the full path).
#
# Usage:
#   scripts/sweep_fig1115.sh                       # default: 3 reps, results in experiments/sweep_results/
#   REPS=5 OUT=experiments/05-fig1115-postfix scripts/sweep_fig1115.sh
#   SAVE=0 scripts/sweep_fig1115.sh                # skip PNG writes for pure-compute timing
#
# Environment variables (all optional):
#   REPS  = number of repetitions per config        (default 3)
#   OUT   = output directory (relative to cwd if not absolute; default experiments/sweep_results)
#   SPEC  = spec.in file                            (default $PROJECT_ROOT/spec.in)
#   BIN   = mandelHybrid binary                     (default $PROJECT_ROOT/mandelHybrid)
#   SAVE  = 1 to write PNGs, 0 to skip              (default 1, matching Fig 11.15)
#   DIFFT = diffThreshold                           (default 0.5)
#   PIXT  = pixelSizeThreshold                      (default 32768)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

REPS=${REPS:-3}
OUT=${OUT:-$PROJECT_ROOT/experiments/sweep_results}
SPEC=${SPEC:-$PROJECT_ROOT/spec.in}
BIN=${BIN:-$PROJECT_ROOT/mandelHybrid}
SAVE=${SAVE:-1}
DIFFT=${DIFFT:-0.5}
PIXT=${PIXT:-32768}

[[ -x "$BIN" ]] || { echo "Binary $BIN not found — run make first." >&2; exit 1; }
[[ -f "$SPEC" ]] || { echo "Spec file $SPEC not found." >&2; exit 1; }

mkdir -p "$OUT/logs"
# Resolve to absolute paths so redirections work regardless of cwd changes below.
OUT_ABS=$(readlink -f "$OUT")
# Per-run image scratch dir; cleaned between runs so disk usage stays bounded.
SCRATCH="$OUT_ABS/scratch_img"

# Configurations: label, numThreads, gpuEnable.
# Total threads (numThreads) on this 12-thread laptop never exceeds 12.
# In main.cpp, thread index 0 is the GPU driver when gpuEnable=1, so the number
# of pure CPU workers in a hybrid run = numThreads - 1.
CONFIGS=(
  "CPU12         12  0"
  "GPU            1  1"
  "GPU+1CPU       2  1"
  "GPU+2CPU       3  1"
  "GPU+4CPU       5  1"
  "GPU+8CPU       9  1"
  "GPU+11CPU     12  1"
)

CSV="$OUT_ABS/results.csv"
echo "label,numThreads,gpuEnable,rep,elapsed_s" > "$CSV"

echo "[sweep] reps=$REPS save=$SAVE diffT=$DIFFT pixT=$PIXT spec=$SPEC out=$OUT_ABS"
echo "[sweep] $(date -Iseconds) starting ${#CONFIGS[@]} configs"

# The spec's imageFilePrefix is read from the spec file itself; we redirect
# image writes into SCRATCH by running each invocation with that as cwd.
SPEC_ABS=$(readlink -f "$SPEC")
BIN_ABS=$(readlink -f "$BIN")

for entry in "${CONFIGS[@]}"; do
  # shellcheck disable=SC2086
  set -- $entry
  label=$1; nthr=$2; gpu=$3
  for r in $(seq 1 "$REPS"); do
    rm -rf "$SCRATCH" && mkdir -p "$SCRATCH"
    log="$OUT_ABS/logs/${label}.r${r}.log"
    printf "[sweep] %-12s rep %d/%d ... " "$label" "$r" "$REPS"
    # quiet=1 to suppress 4000+ per-region prints; summary lines still emitted.
    pushd "$SCRATCH" >/dev/null
    "$BIN_ABS" "$SPEC_ABS" "$nthr" "$gpu" "$DIFFT" "$PIXT" 1 "$SAVE" \
        > "$log.stdout" 2> "$log.stderr"
    popd >/dev/null
    # The total_elapsed_s line is the last informative stderr line.
    elapsed=$(grep '^\[total_elapsed_s\]' "$log.stderr" | awk '{print $2}')
    echo "$label,$nthr,$gpu,$r,$elapsed" >> "$CSV"
    echo "${elapsed}s"
  done
done

rm -rf "$SCRATCH"
echo "[sweep] done. Results: $CSV"
