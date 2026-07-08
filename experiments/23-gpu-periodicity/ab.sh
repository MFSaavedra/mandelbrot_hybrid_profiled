#!/usr/bin/env bash
#
# A/B for the exact Brent periodicity check in the GPU kernel diverge()
# (feat/gpu-periodicity, 2abc909) against main (c4744c0, the
# binary-v6-periodicity lineage: CPU-side check already merged).
#
# Production spec.in (100 frames, 1920x1080, deep zoom to maxIter 10000),
# diffT=0.1, pixT=32768, quiet=1, save=1 (Fig-11.15 methodology, same as the
# experiment-21 headline). Three configs:
#   gpuonly (1 thread, mode 1)  -- isolates the kernel change: every region
#                                  of every class through the modified kernel
#   hybrid  (12 threads, mode 1) -- the production config
#   cpu12   (12 threads, mode 0) -- control (CPU code is byte-identical
#                                  between the binaries; any delta is noise)
#                                  and same-batch anchor for the
#                                  worker-equivalence k of report 22
# Binaries alternate inside each rep so thermal drift hits both sides
# equally. ~12 min for REPS=3, ON AC POWER ONLY.
#
# Usage:  BASE_BIN=/path/to/main/mandelHybrid experiments/23-gpu-periodicity/ab.sh
#         REPS=3 (default) WORK=<scratch dir>
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
SPEC="$ROOT/spec.in"
BASE_BIN=${BASE_BIN:?set BASE_BIN to the baseline (main) mandelHybrid}
GP_BIN="$ROOT/mandelHybrid"
WORK=${WORK:-$SCRIPT_DIR/scratch}
CSV="$SCRIPT_DIR/results.csv"
LOGS="$SCRIPT_DIR/logs"
REPS=${REPS:-3}

[[ -x "$BASE_BIN" && -x "$GP_BIN" ]] || { echo "missing binary" >&2; exit 1; }
for ps in /sys/class/power_supply/*/online; do
  [[ $(cat "$ps") == 1 ]] || { echo "NOT ON AC ($ps) -- timing would be garbage" >&2; exit 1; }
done
mkdir -p "$LOGS"
echo "binary,config,numThreads,gpuMode,rep,elapsed_s" > "$CSV"

for rep in $(seq 1 "$REPS"); do
  for cfg in "gpuonly 1 1" "hybrid 12 1" "cpu12 12 0"; do
    set -- $cfg; name=$1; thr=$2; mode=$3
    for bin in base gp; do
      B=$BASE_BIN; [[ $bin == gp ]] && B=$GP_BIN
      rm -rf "$WORK"; mkdir -p "$WORK"
      log="$LOGS/${bin}.${name}.r${rep}"
      ( cd "$WORK" && "$B" "$SPEC" "$thr" "$mode" 0.1 32768 1 1 \
          > "$log.stdout" 2> "$log.stderr" )
      t=$(awk '/\[total_elapsed_s\]/{print $2}' "$log.stderr")
      echo "$bin,$name,$thr,$mode,$rep,$t" | tee -a "$CSV"
    done
  done
done
rm -rf "$WORK"
echo "[ab] done -> $CSV"
