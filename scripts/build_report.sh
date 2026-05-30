#!/usr/bin/env bash
# Build a report PDF from reports/NN-name.tex.
#
# Each report's preamble already pins the commit hash(es) that produced its
# measurements (see the "\begin{center}\small\textsf{Binary: ...}" line right
# after \inserttitle). This script just runs pdflatex twice from inside
# reports/ so cross-references resolve.
#
# Usage:
#   scripts/build_report.sh 01-initial-profile
#   scripts/build_report.sh 03-bug-analysis
#   scripts/build_report.sh all                # build every NN-*.tex
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPORTS_DIR=$(cd "$SCRIPT_DIR/../reports" && pwd)

usage() {
  echo "Usage: $0 <report-name|all>" >&2
  echo "Available reports:" >&2
  ls "$REPORTS_DIR" | grep -E '^[0-9]{2}-.*\.tex$' | sed 's/\.tex$//' | sed 's/^/  /' >&2
  exit 1
}

[[ $# -eq 1 ]] || usage

build_one() {
  local name=$1
  local src="$REPORTS_DIR/$name.tex"
  [[ -f "$src" ]] || { echo "Report $name.tex not found in $REPORTS_DIR" >&2; return 1; }
  echo "[build_report] building $name.pdf"
  (
    cd "$REPORTS_DIR"
    pdflatex -interaction=nonstopmode "$name.tex" > /dev/null
    pdflatex -interaction=nonstopmode "$name.tex" > /dev/null
  )
  echo "[build_report] -> $REPORTS_DIR/$name.pdf"
}

if [[ "$1" == "all" ]]; then
  for src in "$REPORTS_DIR"/[0-9][0-9]-*.tex; do
    build_one "$(basename "$src" .tex)"
  done
else
  build_one "$1"
fi
