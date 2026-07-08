#!/usr/bin/env bash
#
# A/B for the exact Brent periodicity check in the CPU diverge()
# (feat/periodicity-check) against main (binary-v5-affinity lineage).
#
# Production spec.in (100 frames, 1920x1080, deep zoom to maxIter 10000),
# diffT=0.1, pixT=32768, quiet=1, save=1 (Fig-11.15 methodology, same as the
# experiment-20 headline). Two configs: CPU12 (mode 0, where the check does
# all the work) and dGPU+11CPU (mode 1, the production config, where the CPU
# pool is the wall). Binaries alternate inside each rep so thermal drift hits
# both sides equally.
#
# Usage:  BASE_BIN=/path/to/main/mandelHybrid experiments/21-periodicity-check/ab.sh
#         REPS=3 (default) WORK=<scratch dir>
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
SPEC="$ROOT/spec.in"
BASE_BIN=${BASE_BIN:?set BASE_BIN to the baseline (main) mandelHybrid}
PC_BIN="$ROOT/mandelHybrid"
WORK=${WORK:-$SCRIPT_DIR/scratch}
CSV="$SCRIPT_DIR/results.csv"
LOGS="$SCRIPT_DIR/logs"
REPS=${REPS:-3}

[[ -x "$BASE_BIN" && -x "$PC_BIN" ]] || { echo "missing binary" >&2; exit 1; }
mkdir -p "$LOGS"
echo "binary,config,numThreads,gpuMode,rep,elapsed_s" > "$CSV"

for rep in $(seq 1 "$REPS"); do
  for cfg in "cpu12 12 0" "hybrid 12 1"; do
    set -- $cfg; name=$1; thr=$2; mode=$3
    for bin in base pc; do
      B=$BASE_BIN; [[ $bin == pc ]] && B=$PC_BIN
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
