#!/usr/bin/env bash
# Distributed frame rendering: static cyclic / block-cyclic frame assignment
# across machines, orchestrated with GNU parallel over SSH.  Proof of concept
# for the two-level hierarchy — frames ACROSS nodes, region queue INTRA-node —
# before committing to MPI (the block vs cyclic vs block-cyclic static study
# report 13 earmarked).
#
# Every node reads the SAME spec file (shipped by this script); the binary
# selects its frame subset from DIST_NODES/DIST_RANK/DIST_BLOCK (environment
# variables, read in main.cpp).  Output PNGs carry the GLOBAL frame index, so
# collection is a plain union into OUT with no renumbering and no collisions.
#
# Usage:
#   scripts/dist_frames.sh hosts.txt
#   BLOCK=4 ARGS="12 1 0.1 32768 1 1" OUT=/tmp/collect scripts/dist_frames.sh hosts.txt
#
# hosts.txt: one node per line, rank r = line r+1:
#
#   host [rdir [args...]]
#
# host = SSH destination (user@host), or ":" for "this machine, no SSH" (GNU
# parallel's convention — the identity check in experiments/25-... uses it to
# simulate N nodes locally).  rdir = that host's project root ("-" or omitted
# = the RDIR default below).  args = that host's mandelHybrid arguments
# (omitted = the ARGS default) — this is how heterogeneous nodes join: a
# GPU-less node (built with `make GPU=0`, e.g. the GT 750M machine, Kepler
# sm_30, no modern CUDA) runs with its own thread count and gpuEnable=0:
#
#   :                                                        # this laptop, hybrid
#   ivy /home/lynx/box/mandelbrot_hybrid_profiled 8 0 0.1 32768 1 1
#
# The mandelHybrid binary must already be built at each node's rdir; this
# script ships only the spec and collects only PNGs.
#
# Environment variables (all optional):
#   SPEC  = spec file to ship                      (default $PROJECT_ROOT/spec.in)
#   OUT   = local collection dir for the PNG union (default experiments/25-frame-distribution/collect)
#   RDIR  = project root on each host              (default: this checkout's absolute path)
#   ARGS  = mandelHybrid args after the spec       (default "12 1 0.1 32768 1 1"
#           = numThr gpuEnable diffT pixT quiet save)
#   BLOCK = DIST_BLOCK: 1 = pure cyclic (default); k = block-cyclic;
#           >= ceil(frames/N) = contiguous block
#   WEIGHTS = DIST_WEIGHTS, comma ints one per rank (e.g. "5,1"): switches
#           the binary to seeded weighted-random assignment -- rank r owns
#           ~frames*w_r/W frames.  SEED = DIST_SEED (default: binary's 1234).
#           When set, BLOCK is ignored.
#   JOBS  = concurrent node jobs (default: all hosts at once; the local
#           identity check sets JOBS=1 because simulated nodes share one GPU)
#
# Each rank runs in $RDIR/dist_work/rank<r> with the spec copied there, so the
# spec's own short image prefix is used as-is.  Do NOT lengthen the prefix to
# encode per-node paths instead: `fin >> imageFilePrefix` overflows char[42]
# on prefixes > 41 chars (known bug, CLAUDE.md / report 21 note).
#
# Manager-worker seam (deliberately not built): for heterogeneous nodes or a
# mispredicted monster frame, the static rank predicate in main.cpp becomes a
# per-frame index dispenser (one message per frame).  The per-rank plumbing
# here — ship spec, run, collect by global index — is unchanged by that.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
SELF="$SCRIPT_DIR/dist_frames.sh"

#--------------- per-node job (invoked by GNU parallel as: SELF _node RANK) ---------------
if [[ "${1:-}" == "_node" ]]; then
  rank=$2
  # Node line: host [rdir [args...]]; "-"/empty fall back to the exported
  # defaults.  read -r keeps everything after the second field in args.
  read -r host rdir args <<< "$(grep . "$DIST_HOSTS" | sed -n "$((rank + 1))p")"
  [[ -z "${rdir:-}" || "$rdir" == "-" ]] && rdir=$DIST_RDIR
  [[ -z "${args:-}" ]] && args=$DIST_ARGS
  wd="$rdir/dist_work/rank$rank"

  # rsh CMD: run CMD on this rank's host (":" = locally, no SSH).
  rsh() { if [[ "$host" == ":" ]]; then bash -c "$1"; else ssh "$host" "$1"; fi }

  rsh "mkdir -p '$wd' && rm -f '$wd'/${DIST_PREFIX}*.png"
  if [[ "$host" == ":" ]]; then
    cp "$DIST_SPEC" "$wd/spec.in"
  else
    scp -q "$DIST_SPEC" "$host:$wd/spec.in"
  fi

  denv="DIST_NODES=$DIST_N DIST_RANK=$rank DIST_BLOCK=$DIST_B"
  [[ -n "${DIST_W:-}" ]] && denv+=" DIST_WEIGHTS=$DIST_W"
  [[ -n "${DIST_SEEDV:-}" ]] && denv+=" DIST_SEED=$DIST_SEEDV"
  log="$DIST_OUT/logs/rank$rank"
  rsh "cd '$wd' && $denv '$rdir/mandelHybrid' spec.in $args" \
      > "$log.stdout" 2> "$log.stderr"

  owned=$(grep -oP 'ownedFrames=\K[0-9]+' "$log.stderr" || echo 0)
  elapsed=$(grep '^\[total_elapsed_s\]' "$log.stderr" | awk '{print $2}')
  if [[ "$owned" -gt 0 ]]; then
    if [[ "$host" == ":" ]]; then
      cp "$wd/"${DIST_PREFIX}*.png "$DIST_OUT/"
    else
      scp -q "$host:$wd/${DIST_PREFIX}*.png" "$DIST_OUT/"
    fi
  fi
  echo "[dist] rank $rank on $host ($args): $owned frames in ${elapsed}s"
  exit 0
fi
#-------------------------------------------------------------------------------------------

HOSTS=${1:?usage: dist_frames.sh hosts.txt   (one SSH destination per line, \":\" = local)}
[[ -f "$HOSTS" ]] || { echo "hosts file $HOSTS not found" >&2; exit 1; }

SPEC=${SPEC:-$PROJECT_ROOT/spec.in}
OUT=${OUT:-$PROJECT_ROOT/experiments/25-frame-distribution/collect}
RDIR=${RDIR:-$PROJECT_ROOT}
ARGS=${ARGS:-"12 1 0.1 32768 1 1"}
BLOCK=${BLOCK:-1}

N=$(grep -c . "$HOSTS")
JOBS=${JOBS:-$N}
[[ "$N" -ge 1 ]] || { echo "hosts file $HOSTS has no hosts" >&2; exit 1; }
[[ -f "$SPEC" ]] || { echo "spec file $SPEC not found" >&2; exit 1; }

mkdir -p "$OUT/logs"
export DIST_HOSTS=$(readlink -f "$HOSTS")
export DIST_SPEC=$(readlink -f "$SPEC")
export DIST_OUT=$(readlink -f "$OUT")
export DIST_RDIR="$RDIR"
export DIST_ARGS="$ARGS"
export DIST_B="$BLOCK"
export DIST_N="$N"
export DIST_W="${WEIGHTS:-}"
export DIST_SEEDV="${SEED:-}"
# Image prefix = 4th field of the spec's first line; the union is collected
# by this glob, keyed by the global frame index the binary already emits.
export DIST_PREFIX=$(awk 'NR==1{print $4}' "$DIST_SPEC")

# Clean stale collected frames so the final count reflects this run only.
rm -f "$DIST_OUT/${DIST_PREFIX}"*.png

echo "[dist] $N nodes, block=$BLOCK, jobs=$JOBS, args='$ARGS'"
echo "[dist] spec=$DIST_SPEC -> collect=$DIST_OUT"
seq 0 $((N - 1)) | parallel -j "$JOBS" --halt soon,fail=1 --line-buffer "$SELF" _node {}

total=$(ls "$DIST_OUT/${DIST_PREFIX}"*.png 2>/dev/null | wc -l)
echo "[dist] done: $total PNGs collected in $DIST_OUT"
