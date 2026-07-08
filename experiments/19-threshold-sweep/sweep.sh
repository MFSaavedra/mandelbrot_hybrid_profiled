#!/usr/bin/env bash
#
# 2-D sweep of diffThreshold x pixelSizeThresh on binary-v5-affinity.
# Both are CLI args -- no code change -- so this just runs the existing binary
# across the grid and records the wall time, to find the optimal (diffT, pixT).
#
#     experiments/19-threshold-sweep/sweep.sh
#
# Override via env: DIFFTS, PIXTS, REPS, SPEC, THREADS, GPU.
# Hybrid (12 threads, GPU), quiet, save=0 (pure compute -- the thresholds change
# only HOW the image is computed, never the output, so PNG encode is just noise).

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BIN="$ROOT/mandelHybrid"
SPEC="${SPEC:-$ROOT/spec.in}"
CSV="$HERE/results.csv"

# diffT: relative corner-spread tolerance.  pixT: pixel floor (region sizes in the
# quadtree are 1920x1080, 960x540, 480x270, 240x135, 120x68, 60x34 -> these pixT
# values move the floor across those depths; 32768 is the current default).
DIFFTS="${DIFFTS:-0.05 0.1 0.2 0.3 0.5}"
PIXTS="${PIXTS:-2048 8192 32768 131072 524288}"
REPS="${REPS:-3}"
THREADS="${THREADS:-12}"
GPU="${GPU:-1}"

[ -x "$BIN" ] || { echo "ERROR: binary missing ($BIN) -- run make" >&2; exit 1; }
ac=$(cat /sys/class/power_supply/ACAD/online 2>/dev/null || echo "?")
[ "$ac" = "1" ] || echo "WARNING: AC power not detected (online=$ac). Timing needs AC -- battery throttles CPU ~3x and caps the Max-Q GPU." >&2

nd=$(echo $DIFFTS | wc -w); np=$(echo $PIXTS | wc -w); total=$(( nd * np * REPS ))
echo "spec=$SPEC  threads=$THREADS gpu=$GPU  grid=${nd}x${np}  reps=$REPS  ($total runs)"
echo "diffT,pixT,rep,wall_s" > "$CSV"

n=0
# reps OUTER, grid INNER: each rep sweeps the whole grid once, so thermal drift
# is spread across the grid rather than piled on one cell.
for r in $(seq 1 "$REPS"); do
  for d in $DIFFTS; do
    for p in $PIXTS; do
      n=$((n+1))
      w=$("$BIN" "$SPEC" "$THREADS" "$GPU" "$d" "$p" 1 0 2>&1 \
            | grep -oE 'total_elapsed_s\] [0-9.]+' | grep -oE '[0-9.]+$')
      echo "$d,$p,$r,${w:-NA}" >> "$CSV"
      printf "[%3d/%3d] rep %d  diffT=%-5s pixT=%-7s -> %ss\n" "$n" "$total" "$r" "$d" "$p" "${w:-NA}"
    done
  done
done
echo "DONE -> $CSV   (analyze: experiments/19-threshold-sweep/analyze.py)"
