#!/usr/bin/env bash
# 2-node heterogeneous benchmark, yeco edition — the bench_2node.sh protocol
# rerun with a remote node that is FASTER than the laptop (the ivy study's
# heterogeneity, mirrored):
#   node A (laptop): i7-9750H + GTX 1660 Ti Max-Q, hybrid 12 1   (this machine)
#   node B (yeco):   Core Ultra 7 265K + RTX 4090, hybrid 20 1
#                    (Ubuntu 24.04, reached via the dichato ProxyJump — the
#                    `yeco` ssh alias; ~4.2 s solo save=0, ~5x the laptop)
# Production spec (100 frames 1920x1080), diffT=0.1, quiet=1, save=1 — same
# operating point as bench_2node.sh.  LAPTOP MUST BE ON AC (yeco is a desktop,
# no battery to check).
#
# Configs (hosts order mirrors the ivy sweep; the fast/slow roles are swapped,
# so the block orientations invert their meaning):
#   laptop_hybrid   single-process baseline on the laptop
#   yeco_solo       single-process hybrid on yeco (capability; cheap -> 3 reps)
#   2node_cyclic    laptop+yeco, DIST_BLOCK=1  (equal-share interleave)
#   2node_bc8       laptop+yeco, DIST_BLOCK=8  (block-cyclic interleave)
#   2node_block     laptop+yeco, DIST_BLOCK=50 (laptop frames 0-49 cheap,
#                   yeco 50-99: the deep/expensive half on the FAST node)
#   2node_block_rev hosts reversed, DIST_BLOCK=50 (yeco 0-49 cheap, laptop
#                   50-99 deep: the losing orientation, deep half on slow node)
#
# Output rows: config,block,rep,wall_e2e_s,laptop_s,yeco_s
#   wall_e2e_s = end-to-end dist_frames.sh time (ship + max(rank) + collect),
#                or the single process's [total_elapsed_s] for the baselines
#   laptop_s / yeco_s = each node's own [total_elapsed_s] (compute wall)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
BIN=$PROJECT_ROOT/mandelHybrid
SPEC=$PROJECT_ROOT/spec.in
NODE=yeco
NODE_RDIR=/home/mfsaaved/box/mandelbrot_hybrid_profiled
LAPTOP_ARGS="12 1 0.1 32768 1 1"
NODE_ARGS="20 1 0.1 32768 1 1"
REPS=${REPS:-3}

WORK=$SCRIPT_DIR/bench_yeco     # gitignored scratch
CSV=$SCRIPT_DIR/results_bench_yeco.csv
mkdir -p "$WORK/scratch"

# Refuse to time on battery (repo methodology: battery throttles ~3x).
# yeco is a desktop: no battery, nothing to check remotely.
cat /sys/class/power_supply/AC*/online 2>/dev/null | grep -q 1 \
    || { echo "laptop not on AC — aborting" >&2; exit 1; }

printf ': - %s\n%s %s %s\n' "$LAPTOP_ARGS" "$NODE" "$NODE_RDIR" "$NODE_ARGS" > "$WORK/hosts_fwd.txt"
printf '%s %s %s\n: - %s\n' "$NODE" "$NODE_RDIR" "$NODE_ARGS" "$LAPTOP_ARGS" > "$WORK/hosts_rev.txt"

echo "config,block,rep,wall_e2e_s,laptop_s,yeco_s" > "$CSV"

elapsed_of() { grep '^\[total_elapsed_s\]' "$1" | awk '{print $2}'; }

# dist_rep CONFIG HOSTSFILE BLOCK LAPTOP_RANK NODE_RANK REP
dist_rep() {
  local config=$1 hosts=$2 block=$3 lrank=$4 nrank=$5 rep=$6
  local out=$WORK/collect_$config
  local t0 t1
  t0=$(date +%s.%N)
  OUT=$out BLOCK=$block SPEC=$SPEC "$PROJECT_ROOT/scripts/dist_frames.sh" "$hosts" \
      > "$WORK/${config}.r${rep}.log" 2>&1
  t1=$(date +%s.%N)
  local n
  n=$(ls "$out"/img*.png | wc -l)
  [[ "$n" -eq 100 ]] || { echo "$config rep $rep: incomplete union ($n/100)" >&2; exit 1; }
  echo "$config,$block,$rep,$(echo "$t1 $t0" | awk '{printf "%.3f", $1-$2}'),$(elapsed_of "$out/logs/rank$lrank.stderr"),$(elapsed_of "$out/logs/rank$nrank.stderr")" >> "$CSV"
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

  # -- yeco solo capability (cheap enough to rep alongside) --
  ssh $NODE "mkdir -p $NODE_RDIR/dist_work/solo && cd $NODE_RDIR/dist_work/solo && rm -f img*.png && $NODE_RDIR/mandelHybrid $NODE_RDIR/spec.in $NODE_ARGS" \
      2> "$WORK/yeco_solo.r${rep}.stderr" >/dev/null
  echo "yeco_solo,,${rep},$(elapsed_of "$WORK/yeco_solo.r${rep}.stderr"),,$(elapsed_of "$WORK/yeco_solo.r${rep}.stderr")" >> "$CSV"
  tail -1 "$CSV"
done

echo "[bench] done -> $CSV"
