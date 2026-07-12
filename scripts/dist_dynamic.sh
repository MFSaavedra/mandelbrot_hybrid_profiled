#!/usr/bin/env bash
# Dynamic frame distribution: weighted-random initial assignment + work
# stealing, driven entirely from this coordinator over the same SSH plumbing
# as dist_frames.sh -- no node-to-node connections, no resident daemon, and
# the intra-node region queue stays untouched.
#
#   1. STATIC HALF: the frame list 0..F-1 is shuffled with a seeded
#      deterministic shuffle and dealt to nodes proportionally to WEIGHTS
#      ("5,1" = rank 0 starts with ~5/6 of the frames) -- each node begins
#      with a random, speed-proportional share (the weighted analogue of
#      cyclic: the zoom's cost trend is sampled uniformly by every share).
#   2. DYNAMIC HALF: each node's driver dispatches its bag in guided chunks
#      (half the remaining bag, min KMIN) as repeated mandelHybrid
#      invocations with DIST_FRAMES=<explicit list>.  When a node's own bag
#      empties -- it is about to finish -- it STEALS half the richest
#      node's undispatched tail and keeps rendering.  All bag mutations are
#      serialized under one flock, and in-flight chunks are already out of
#      the bags, so every frame is rendered exactly once.
#
# Guided halving keeps per-invocation overhead off the critical path (a
# node's own share costs ~log2(share/KMIN)+1 process startups) while leaving
# a fine-grained tail that stealing can rebalance.  Chunks are additionally
# capped at KCAP*weight_r frames per rank: an in-flight chunk cannot be
# stolen, and its exposure time is k/rate_r with rate proportional to the
# weight, so the weight-proportional cap bounds every node's unstealable
# exposure at the same wall-time budget -- without it, a slow node's first
# half-bag chunk binds the whole run when the weights are miscalibrated
# (measured: 25 frames = 27.4 s in flight on ivy at WEIGHTS=1,1).  Output
# PNGs keep the global frame index, so collection stays a plain union.
#
# Usage:
#   scripts/dist_dynamic.sh hosts.txt
#   WEIGHTS="5,1" scripts/dist_dynamic.sh hosts.txt
#
# hosts.txt lines: host [rdir [args...]]  -- same format as dist_frames.sh
# (":" = this machine, no SSH).
#
# Environment variables (all optional):
#   SPEC / OUT / RDIR / ARGS   as in dist_frames.sh
#   WEIGHTS = comma ints, one per rank (default: all 1 = equal shares)
#   SEED    = shuffle seed (default 1234)
#   KMIN    = minimum chunk size in frames (default 4)
#   KCAP    = max chunk per unit weight (default 8): rank r's chunks are
#             capped at max(KMIN, KCAP*w_r) frames
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

HOSTS=${1:?usage: dist_dynamic.sh hosts.txt   (one node per line: host [rdir [args...]])}
[[ -f "$HOSTS" ]] || { echo "hosts file $HOSTS not found" >&2; exit 1; }

SPEC=${SPEC:-$PROJECT_ROOT/spec.in}
OUT=${OUT:-$PROJECT_ROOT/experiments/25-frame-distribution/collect}
RDIR=${RDIR:-$PROJECT_ROOT}
ARGS=${ARGS:-"12 1 0.1 32768 1 1"}
SEED=${SEED:-1234}
KMIN=${KMIN:-4}
KCAP=${KCAP:-8}

N=$(grep -c . "$HOSTS")
[[ "$N" -ge 1 ]] || { echo "hosts file $HOSTS has no hosts" >&2; exit 1; }
[[ -f "$SPEC" ]] || { echo "spec file $SPEC not found" >&2; exit 1; }
WEIGHTS=${WEIGHTS:-$(yes 1 | head -n "$N" | paste -sd,)}

mkdir -p "$OUT/logs"
SPEC_ABS=$(readlink -f "$SPEC")
OUT_ABS=$(readlink -f "$OUT")
F=$(awk 'NR==1{print $1}' "$SPEC_ABS")
PREFIX=$(awk 'NR==1{print $4}' "$SPEC_ABS")
BAG="$OUT_ABS/bag"
rm -rf "$BAG" && mkdir -p "$BAG"
rm -f "$OUT_ABS/${PREFIX}"*.png
: > "$BAG/.lock"

# ---- static half: seeded shuffle, dealt proportionally to WEIGHTS ----
IFS=, read -ra W <<< "$WEIGHTS"
[[ ${#W[@]} -eq $N ]] || { echo "WEIGHTS has ${#W[@]} entries for $N hosts" >&2; exit 1; }
WSUM=0; for w in "${W[@]}"; do WSUM=$((WSUM + w)); done
shuf --random-source=<(yes "$SEED") -i 0-$((F - 1)) > "$BAG/all.txt"
start=1
for r in $(seq 0 $((N - 1))); do
  if [[ "$r" -lt $((N - 1)) ]]; then cnt=$((F * W[r] / WSUM)); else cnt=$((F - start + 1)); fi
  sed -n "${start},$((start + cnt - 1))p" "$BAG/all.txt" > "$BAG/rank$r.list"
  start=$((start + cnt))
  cap=$((KCAP * W[r])); [[ "$cap" -lt "$KMIN" ]] && cap=$KMIN
  echo "$cap" > "$BAG/rank$r.cap"
  echo "[dyn] rank $r initial share: $cnt frames (weight ${W[r]}/$WSUM, chunk cap $cap)"
done

# ---- take_chunk RANK: print "own N i,j,k" or "steal<victim> N i,j,k";
# print nothing when every bag is empty.  Serialized by flock so bag reads
# and edits are atomic; guided halving; steal = half the richest tail. ----
take_chunk() {
  local r=$1
  (
    flock 9
    local own="$BAG/rank$r.list" cap n k
    cap=$(cat "$BAG/rank$r.cap")
    n=$(wc -l < "$own")
    if [[ "$n" -gt 0 ]]; then
      k=$(( (n + 1) / 2 ))
      [[ "$k" -gt "$cap" ]] && k=$cap
      [[ "$k" -lt "$KMIN" ]] && k=$n
      echo "own $k $(head -n "$k" "$own" | paste -sd,)"
      tail -n +$((k + 1)) "$own" > "$own.t" && mv "$own.t" "$own"
      return 0
    fi
    local best="" bn=0 m f
    for f in "$BAG"/rank*.list; do
      m=$(wc -l < "$f")
      if [[ "$m" -gt "$bn" ]]; then bn=$m; best=$f; fi
    done
    [[ "$bn" -eq 0 ]] && return 0
    k=$(( (bn + 1) / 2 ))
    [[ "$k" -gt "$cap" ]] && k=$cap
    [[ "$k" -lt "$KMIN" ]] && k=$bn
    local victim; victim=$(basename "$best" .list); victim=${victim#rank}
    echo "steal$victim $k $(tail -n "$k" "$best" | paste -sd,)"
    head -n -"$k" "$best" > "$best.t" && mv "$best.t" "$best"
  ) 9>>"$BAG/.lock"
}

# ---- driver RANK: prep the node once, then dispatch chunks until the
# whole cluster's bags are empty ----
driver() {
  local rank=$1 host rdir args
  read -r host rdir args <<< "$(grep . "$HOSTS_ABS" | sed -n "$((rank + 1))p")"
  [[ -z "${rdir:-}" || "$rdir" == "-" ]] && rdir=$RDIR
  [[ -z "${args:-}" ]] && args=$ARGS
  local wd="$rdir/dist_work/dynrank$rank"

  rsh() { if [[ "$host" == ":" ]]; then bash -c "$1"; else ssh "$host" "$1"; fi }

  rsh "mkdir -p '$wd' && rm -f '$wd'/${PREFIX}*.png"
  if [[ "$host" == ":" ]]; then cp "$SPEC_ABS" "$wd/spec.in"
  else scp -q "$SPEC_ABS" "$host:$wd/spec.in"; fi

  local rendered=0 stolen=0 busy=0 c=0 desc src k list log el
  while :; do
    desc=$(take_chunk "$rank")
    [[ -z "$desc" ]] && break
    read -r src k list <<< "$desc"
    log="$OUT_ABS/logs/dynrank$rank.chunk$c"
    rsh "cd '$wd' && rm -f ${PREFIX}*.png && DIST_FRAMES=$list \
         '$rdir/mandelHybrid' spec.in $args" > "$log.stdout" 2> "$log.stderr"
    if [[ "$host" == ":" ]]; then cp "$wd/"${PREFIX}*.png "$OUT_ABS/"
    else scp -q "$host:$wd/${PREFIX}*.png" "$OUT_ABS/"; fi
    el=$(grep '^\[total_elapsed_s\]' "$log.stderr" | awk '{print $2}')
    busy=$(echo "$busy $el" | awk '{printf "%.3f", $1 + $2}')
    rendered=$((rendered + k))
    [[ "$src" != own ]] && stolen=$((stolen + k))
    echo "[dyn] rank $rank chunk $c ($src): $k frames in ${el}s  [$list]"
    c=$((c + 1))
  done
  echo "[dyn] rank $rank on $host: done -- $rendered frames ($stolen stolen), busy ${busy}s over $c chunks"
  echo "$rank,$rendered,$stolen,$busy" >> "$BAG/stats.csv"
}

HOSTS_ABS=$(readlink -f "$HOSTS")
echo "[dyn] $N nodes, weights=$WEIGHTS, seed=$SEED, kmin=$KMIN, $F frames"
t0=$(date +%s.%N)
pids=()
for r in $(seq 0 $((N - 1))); do driver "$r" & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done
t1=$(date +%s.%N)

total=$(ls "$OUT_ABS/${PREFIX}"*.png 2>/dev/null | wc -l)
echo "[dyn] wall_e2e_s $(echo "$t1 $t0" | awk '{printf "%.3f", $1 - $2}')"
echo "[dyn] done: $total/$F PNGs collected in $OUT_ABS"
[[ "$total" -eq "$F" ]] || { echo "[dyn] INCOMPLETE UNION" >&2; exit 1; }
