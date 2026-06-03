#!/usr/bin/env bash
#
# Nsight Compute (ncu) capture for the warp-divergence / GPU-metrics study
# (report 16).  GPU performance counters are admin-restricted on the NVIDIA
# Linux driver (ERR_NVGPUCTRPERM), so this CAPTURE step must run as root:
#
#     sudo experiments/16-ncu-divergence/capture.sh
#
# It only *captures* (saves .ncu-rep reports) and then chowns them back to the
# invoking user.  All parsing/analysis is done afterward unprivileged via
# `ncu --import`, which needs no GPU access.
#
# Each run is GPU-only (numThreads=1, GPU on) so the single GPU thread processes
# EVERY region -- interior, exterior, and boundary -- exposing the full warp
# spectrum.  In production the affinity queue routes boundary regions to the CPU
# pool, so a normal hybrid run would never show the GPU the divergent regions.

set -u

NCU="${NCU:-/usr/bin/ncu}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BIN="$ROOT/mandelHybrid"
SPECS="$HERE/specs"
OUT="$HERE/ncu"
RUNUSER="${SUDO_USER:-$(id -un)}"

# Make CUDA/Qt libs resolvable even under sudo's reset environment.
export LD_LIBRARY_PATH="/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"

# GPU-only, diffT=0.1 (project default), quiet, no PNG writes.
APP="1 1 0.1 32768 1 0"

# Divergence metrics, isolated in their own (cheap, 1-2 pass) invocation so an
# unexpected metric-name rejection cannot abort the full-set captures:
#   warp execution efficiency = threads active per executed inst (out of 32)
#   branch efficiency         = % of branches that are warp-uniform
WARP="smsp__thread_inst_executed_per_inst_executed.ratio,smsp__sass_average_branch_targets_threads_uniform.pct"

if [ ! -x "$BIN" ]; then echo "ERROR: binary not found/executable: $BIN" >&2; exit 1; fi
if [ ! -x "$NCU" ]; then echo "ERROR: ncu not found at $NCU" >&2; exit 1; fi
mkdir -p "$OUT"

echo "== environment =="
"$NCU" --version | head -3
echo "ROOT=$ROOT"; echo "RUNUSER=$RUNUSER"
echo "== GPU state (recorded for the report) =="
nvidia-smi --query-gpu=name,persistence_mode,power.draw,power.limit,clocks.sm,clocks.max.sm,temperature.gpu \
    --format=csv 2>&1 | tee "$OUT/gpu_state.csv"

# run <name> <spec> <full_kernel_cap> <warp_kernel_cap>
run () {
  local name="$1" spec="$2" fullcap="$3" warpcap="$4"
  echo
  echo "############################ $name ############################"
  echo "-- full metric set  (--set full, up to $fullcap kernels) --"
  timeout 1200 "$NCU" -f -o "$OUT/${name}_full" --set full \
      --launch-count "$fullcap" --target-processes all \
      "$BIN" "$spec" $APP 2>&1 | tail -5
  echo "-- warp/branch efficiency  (up to $warpcap kernels) --"
  timeout 1200 "$NCU" -f -o "$OUT/${name}_warp" --metrics "$WARP" \
      --launch-count "$warpcap" --target-processes all \
      "$BIN" "$spec" $APP 2>&1 | tail -5
}

#    name        spec                       full  warp
run interior   "$SPECS/interior.in"          6     6
run exterior   "$SPECS/exterior.in"          6     6
run boundary   "$SPECS/boundary.in"         60   300
run canonical  "$SPECS/canonical.in"        60   300

echo
echo "== handing reports back to $RUNUSER =="
chown -R "$RUNUSER":"$RUNUSER" "$OUT" 2>/dev/null || chown -R "$RUNUSER" "$OUT"
chmod -R u+rw "$OUT" 2>/dev/null || true

echo
echo "== DONE.  Captured reports: =="
ls -la "$OUT"
echo
echo "Next: the analysis step runs unprivileged (ncu --import), no sudo needed."
