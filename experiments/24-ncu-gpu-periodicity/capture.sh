#!/usr/bin/env bash
#
# Nsight Compute (ncu) re-baseline of the GPU warp/pipe metrics after the
# GPU-side periodicity check (binary-v7-gpu-periodicity, report 23).
# Report 16 pinned the pre-check kernel at 100.0% warp execution efficiency
# and 85.6% FP64 pipe on interior content; the check spends that metric
# deliberately (early-exit divergence), and this capture puts the new
# numbers on record. Baseline = experiments/16-ncu-divergence/analysis/
# summary.csv (the kernel is unchanged v5->v6, so report 16's capture is
# the v6 baseline).
#
# GPU performance counters are admin-restricted on the NVIDIA Linux driver
# (ERR_NVGPUCTRPERM), so this CAPTURE step must run as root:
#
#     sudo experiments/24-ncu-gpu-periodicity/capture.sh
#
# It only *captures* (saves .ncu-rep) and chowns the output back to the
# invoking user. Parsing runs unprivileged afterward: analyze.py.
#
# Same design as experiment 16 (same specs, caps, metrics, GPU-only runs so
# the single GPU thread sees every region class); see that README for the
# regime rationale.

set -u

NCU="${NCU:-/usr/bin/ncu}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BIN="$ROOT/mandelHybrid"
SPECS="$ROOT/experiments/16-ncu-divergence/specs"   # reuse exp-16 specs verbatim
OUT="$HERE/ncu"
RUNUSER="${SUDO_USER:-$(id -un)}"

export LD_LIBRARY_PATH="/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"

# GPU-only, diffT=0.1 (project default), quiet, no PNG writes.
APP="1 1 0.1 32768 1 0"

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
echo "Next: python3 $HERE/analyze.py   (unprivileged)"
