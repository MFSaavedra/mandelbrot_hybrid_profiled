#!/usr/bin/env bash
#
# Nsight Compute capture for report 17 -- verifying the warp-divergence / FP64
# result (report 16) on the FOUR report-13 zoom points (outside / inside /
# misiurewicz / seahorse), instead of report 16's synthetic specs.
#
# Same admin gate as report 16: GPU performance counters are root-only
# (ERR_NVGPUCTRPERM), so the CAPTURE runs as root; PARSING a saved report does
# not (ncu --import, no sudo). Run:
#
#     sudo experiments/17-ncu-zoom-points/capture.sh
#
# It chowns the reports back to the invoking user when done.
#
# GPU-only (numThreads=1) so the single GPU thread processes EVERY region type
# (in production, affinity routes the divergent boundary regions to the CPU).

set -u

NCU="${NCU:-/usr/bin/ncu}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BIN="$ROOT/mandelHybrid"
SPECS="$HERE/specs"
OUT="$HERE/ncu"
RUNUSER="${SUDO_USER:-$(id -un)}"

export LD_LIBRARY_PATH="/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"

# GPU-only, diffT=0.1 (project default), quiet, no PNG writes.
APP="1 1 0.1 32768 1 0"

WARP="smsp__thread_inst_executed_per_inst_executed.ratio,smsp__sass_average_branch_targets_threads_uniform.pct"

[ -x "$BIN" ] || { echo "ERROR: binary not found/executable: $BIN" >&2; exit 1; }
[ -x "$NCU" ] || { echo "ERROR: ncu not found at $NCU" >&2; exit 1; }
mkdir -p "$OUT"

echo "== ncu / GPU state =="
"$NCU" --version | head -3
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu \
    --format=csv 2>&1 | tee "$OUT/gpu_state.csv"

# run <regime> <full_kernel_cap> <warp_kernel_cap>
run () {
  local name="$1" full="$2" warp="$3"
  echo
  echo "######################## $name ########################"
  echo "-- full metric set (--set full, up to $full kernels) --"
  timeout 1500 "$NCU" -f -o "$OUT/${name}_full" --set full \
      --launch-count "$full" --target-processes all \
      "$BIN" "$SPECS/$name.in" $APP 2>&1 | tail -5
  echo "-- warp/branch efficiency (up to $warp kernels) --"
  timeout 1500 "$NCU" -f -o "$OUT/${name}_warp" --metrics "$WARP" \
      --launch-count "$warp" --target-processes all \
      "$BIN" "$SPECS/$name.in" $APP 2>&1 | tail -5
}

# cheapest regimes first so progress is visible early; inside is the slowest.
run outside     40 300
run misiurewicz 40 300
run seahorse    40 300
run inside      40 300

echo
echo "== handing reports back to $RUNUSER =="
chown -R "$RUNUSER":"$RUNUSER" "$OUT" 2>/dev/null || chown -R "$RUNUSER" "$OUT"
chmod -R u+rw "$OUT" 2>/dev/null || true

echo
echo "== DONE.  Captured reports: =="
ls -la "$OUT"
echo
echo "Next: parsing runs unprivileged (ncu --import), no sudo needed."
