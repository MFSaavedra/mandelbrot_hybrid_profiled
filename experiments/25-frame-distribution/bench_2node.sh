#!/usr/bin/env bash
# 2-node heterogeneous benchmark — first measured pass of the static-LB study
# report 13 earmarked (block vs cyclic vs block-cyclic), on real hardware:
#   node A (laptop): i7-9750H + GTX 1660 Ti Max-Q, hybrid 12 1   (this machine)
#   node B (ivy):    i7-4702MQ, CPU-only build (make GPU=0), 8 0
# Production spec (100 frames 1920x1080), diffT=0.1, quiet=1, save=1 — same
# operating point as reports 21/23.  BOTH MACHINES MUST BE ON AC.
#
# Configs:
#   laptop_hybrid   single-process baseline on the laptop (the 23.7 s of report 23)
#   ivy_solo        single-process CPU-only on ivy (its solo capability; 1 rep)
#   2node_cyclic    laptop+ivy, DIST_BLOCK=1  (equal-share interleave)
#   2node_bc8       laptop+ivy, DIST_BLOCK=8  (block-cyclic interleave)
#   2node_block     laptop+ivy, DIST_BLOCK=50 (laptop frames 0-49, ivy 50-99:
#                   the slow node gets the deep/expensive half)
#   2node_block_rev hosts reversed, DIST_BLOCK=50 (ivy 0-49 cheap, laptop
#                   50-99 deep: crude weighted assignment via block orientation)
#
# Output rows: config,block,rep,wall_e2e_s,laptop_s,ivy_s
#   wall_e2e_s = end-to-end dist_frames.sh time (ship + max(rank) + collect),
#                or the single process's [total_elapsed_s] for the baselines
#   laptop_s / ivy_s = each node's own [total_elapsed_s] (compute wall)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
BIN=$PROJECT_ROOT/mandelHybrid
SPEC=$PROJECT_ROOT/spec.in
IVY=ivy
IVY_RDIR=/home/lynx/box/mandelbrot_hybrid_profiled
LAPTOP_ARGS="12 1 0.1 32768 1 1"
IVY_ARGS="8 0 0.1 32768 1 1"
REPS=${REPS:-3}

WORK=$SCRIPT_DIR/bench          # gitignored scratch
CSV=$SCRIPT_DIR/results_bench.csv
mkdir -p "$WORK/scratch"

# Refuse to time on battery (repo methodology: battery throttles ~3x).
on_ac() { local h=$1; if [[ $h == local ]]; then cat /sys/class/power_supply/AC*/online 2>/dev/null | grep -q 1;
          else ssh "$h" 'cat /sys/class/power_supply/AC*/online 2>/dev/null' | grep -q 1; fi }
on_ac local || { echo "laptop not on AC — aborting" >&2; exit 1; }
on_ac $IVY  || { echo "ivy not on AC — aborting" >&2; exit 1; }

printf ': - %s\nivy %s %s\n' "$LAPTOP_ARGS" "$IVY_RDIR" "$IVY_ARGS" > "$WORK/hosts_fwd.txt"
printf 'ivy %s %s\n: - %s\n' "$IVY_RDIR" "$IVY_ARGS" "$LAPTOP_ARGS" > "$WORK/hosts_rev.txt"

echo "config,block,rep,wall_e2e_s,laptop_s,ivy_s" > "$CSV"

elapsed_of() { grep '^\[total_elapsed_s\]' "$1" | awk '{print $2}'; }

# dist_rep CONFIG HOSTSFILE BLOCK LAPTOP_RANK IVY_RANK REP
dist_rep() {
  local config=$1 hosts=$2 block=$3 lrank=$4 irank=$5 rep=$6
  local out=$WORK/collect_$config
  local t0 t1
  t0=$(date +%s.%N)
  OUT=$out BLOCK=$block SPEC=$SPEC "$PROJECT_ROOT/scripts/dist_frames.sh" "$hosts" \
      > "$WORK/${config}.r${rep}.log" 2>&1
  t1=$(date +%s.%N)
  local n
  n=$(ls "$out"/img*.png | wc -l)
  [[ "$n" -eq 100 ]] || { echo "$config rep $rep: incomplete union ($n/100)" >&2; exit 1; }
  echo "$config,$block,$rep,$(echo "$t1 $t0" | awk '{printf "%.3f", $1-$2}'),$(elapsed_of "$out/logs/rank$lrank.stderr"),$(elapsed_of "$out/logs/rank$irank.stderr")" >> "$CSV"
  tail -1 "$CSV"
}

for rep in $(seq 1 "$REPS"); do
  # -- laptop hybrid baseline --
  rm -f "$WORK/scratch"/img*.png
  (cd "$WORK/scratch" && "$BIN" "$SPEC" $LAPTOP_ARGS 2> "$WORK/laptop_hybrid.r${rep}.stderr")
  echo "laptop_hybrid,,${rep},$(elapsed_of "$WORK/laptop_hybrid.r${rep}.stderr"),$(elapsed_of "$WORK/laptop_hybrid.r${rep}.stderr")," >> "$CSV"
  tail -1 "$CSV"

  # -- distributed configs --
  dist_rep 2node_cyclic    "$WORK/hosts_fwd.txt" 1  0 1 "$rep"
  dist_rep 2node_bc8       "$WORK/hosts_fwd.txt" 8  0 1 "$rep"
  dist_rep 2node_block     "$WORK/hosts_fwd.txt" 50 0 1 "$rep"
  dist_rep 2node_block_rev "$WORK/hosts_rev.txt" 50 1 0 "$rep"
done

# -- ivy solo capability (1 rep; ~85 s, a capability datapoint not an A/B) --
ssh $IVY "mkdir -p $IVY_RDIR/dist_work/solo && cd $IVY_RDIR/dist_work/solo && rm -f img*.png && $IVY_RDIR/mandelHybrid $IVY_RDIR/spec.in $IVY_ARGS" \
    2> "$WORK/ivy_solo.r1.stderr" >/dev/null
echo "ivy_solo,,1,$(elapsed_of "$WORK/ivy_solo.r1.stderr"),,$(elapsed_of "$WORK/ivy_solo.r1.stderr")" >> "$CSV"
tail -1 "$CSV"

echo "[bench] done -> $CSV"
