#!/usr/bin/env bash
# Driver for experiment 13 (zoom-point characterization).
#   ./run.sh timing                 -> wall/balance/outlier profile (viz=0, save=0)
#   ./run.sh viz                    -> viz=3 split-process animation (save=0 forced)
#   ./run.sh viz "seahorse inside"  -> only the listed points
# Runs points SEQUENTIALLY so wall timing is never perturbed by a concurrent run.
# NOTE: cd's into the experiment dir and uses SHORT prefixes (viz_<pt>/) because the
# binary reads imageFilePrefix into a fixed ~42-byte buffer (main.cpp:201); long
# absolute prefixes truncate (the viz_misiurewicz/ overflow bug).
set -u
DIR=/home/lynx/box/cpp/Multicore_and_GPU_2e_code/Chapter_11_Loadbalancing/mandelbrot_hybrid_profiled/experiments/13-zoom-points
cd "$DIR" || exit 1
BIN=../../mandelHybrid
PHASE="${1:-timing}"
POINTS="${2:-outside misiurewicz seahorse inside}"
THREADS=12; GPU=1; DIFFT=0.1; PIXT=32768; QUIET=1
if [ "$PHASE" = "timing" ]; then VIZ=0; SAVE=0; SUF=timing; else VIZ=3; SAVE=0; SUF=viz; fi

for p in $POINTS; do
  LOG="logs/$p.$SUF.stderr"   # .stderr not .log: *.log is gitignored (LaTeX intermediates)
  echo "=== $p ($PHASE) start $(date +%H:%M:%S) ==="
  "$BIN" "spec_$p.in" $THREADS $GPU $DIFFT $PIXT $QUIET $SAVE $VIZ > "$LOG" 2>&1
  rc=$?
  W=$(grep -h total_elapsed "$LOG" | tail -1)
  O=$(grep -hc '\[OUTLIER\]' "$LOG")
  echo "=== $p done rc=$rc  $W  outliers=$O ==="
done
echo "ALL $PHASE DONE $(date +%H:%M:%S)"
