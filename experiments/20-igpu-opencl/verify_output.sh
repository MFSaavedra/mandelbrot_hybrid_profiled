#!/usr/bin/env bash
#
# Full-run output-identity verification for the iGPU (OpenCL) backend.
#
# Renders the entire production spec (100 frames, 1920x1080) three times:
#   dgpu : mode 1, 1 thread  -> every pixel computed by the CUDA path
#   igpu : mode 2, 1 thread  -> every pixel computed by the OpenCL path
#   cpu  : mode 0, 12 threads -> every pixel computed by the host path
#     (thread count is irrelevant for output: pixel values depend only on
#      which *backend* computes them, never on the decomposition)
#
# Then compares, per frame:
#   dgpu vs igpu : byte compare (cmp) -- the "byte-identical" claim
#   dgpu vs cpu  : differing-pixel count (ImageMagick AE) -- the FMA rounding gap
#   igpu vs cpu  : same, to confirm both GPUs differ from the CPU identically
#
# Writes verify_output.csv (frame,dgpu_vs_igpu,dgpu_vs_cpu_AE,igpu_vs_cpu_AE)
# and a summary block to stdout. Timing is irrelevant here (correctness only),
# so AC power is not required.
#
# Usage: experiments/20-igpu-opencl/verify_output.sh [workdir]
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
BIN="$ROOT/mandelHybrid"
SPEC="$ROOT/spec.in"
WORK=${1:-$SCRIPT_DIR/verify_scratch}
CSV="$SCRIPT_DIR/verify_output.csv"

[[ -x "$BIN" ]] || { echo "binary missing: $BIN (run make)" >&2; exit 1; }
command -v magick >/dev/null && IM="magick compare" || IM="compare"

# IM 7 prints AE as "count (normalized)"; keep the integer count only.
ae() {
  local out; out=$($IM -metric AE "$1" "$2" null: 2>&1) || true
  out=${out%% *}
  [[ $out =~ ^[0-9]+$ ]] && echo "$out" || echo "-1"
}

# RENDER=0 skips the (long) render stage and re-runs only the comparison,
# e.g. after fixing this script with the renders already on disk.
if [[ ${RENDER:-1} -eq 1 ]]; then
  declare -A CFG=( [dgpu]="1 1" [igpu]="1 2" [cpu]="12 0" )
  for name in dgpu igpu cpu; do
    set -- ${CFG[$name]}
    rm -rf "$WORK/$name"; mkdir -p "$WORK/$name"
    echo "[verify] rendering $name (numThr=$1 gpuMode=$2) ..."
    ( cd "$WORK/$name" && "$BIN" "$SPEC" "$1" "$2" 0.1 32768 1 1 \
        > run.stdout 2> run.stderr )
  done
fi

nframes=$(ls "$WORK/dgpu"/img*.png | wc -l)
echo "frame,dgpu_vs_igpu_AE,dgpu_vs_cpu_AE,igpu_vs_cpu_AE" > "$CSV"
identical=0; total_gg=0; total_dc=0; total_ic=0; mismatch_frames=""
for f in $(seq -f "%04g" 0 $((nframes - 1))); do
  p="img$f.png"
  if cmp -s "$WORK/dgpu/$p" "$WORK/igpu/$p"; then gg=0; identical=$((identical+1));
  else gg=$(ae "$WORK/dgpu/$p" "$WORK/igpu/$p"); mismatch_frames="$mismatch_frames $f:$gg"; fi
  dc=$(ae "$WORK/dgpu/$p" "$WORK/cpu/$p")
  ic=$(ae "$WORK/igpu/$p" "$WORK/cpu/$p")
  total_gg=$((total_gg + gg)); total_dc=$((total_dc + dc)); total_ic=$((total_ic + ic))
  echo "$f,$gg,$dc,$ic" >> "$CSV"
done

echo "[verify] frames rendered            : $nframes  ($((nframes * 1920 * 1080)) px total)"
echo "[verify] dGPU vs iGPU byte-identical: $identical / $nframes frames; differing px total: $total_gg"
echo "[verify] frames differing (frame:px):${mismatch_frames:- none}"
echo "[verify] dGPU vs CPU differing px   : $total_dc"
echo "[verify] iGPU vs CPU differing px   : $total_ic"
echo "[verify] per-frame detail -> $CSV"
