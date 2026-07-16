#!/usr/bin/env bash
# Build and run the DLT retrodiction driver (report 28).
#
# DLTlib (G. Barlas, GPL-3.0) lives OUTSIDE this repo, as a sibling of the
# project root: ../../../DLTlib (override with DLTDIR=...). It is copied into
# ./build/ and patched there -- the pristine library tree is never modified.
# The patch fixes DLTlib's random.c line 21, which hard-codes an absolute
# include path ("/papers/cpp_lib/random.h") that exists only on the author's
# machine; the local random.h declares the same symbols.
#
# Requires: g++, GLPK (pacman -S glpk / apt install libglpk-dev).
#
# Regenerates: results.txt (the analysis table) and sweep.csv (optimum
# share/wall vs per-frame collection cost l, for plot_dlt28.py).
set -euo pipefail
cd "$(dirname "$0")"

DLTDIR=${DLTDIR:-$(cd ../../../DLTlib && pwd)}
[ -f "$DLTDIR/dltlib.cpp" ] || { echo "DLTlib not found at $DLTDIR" >&2; exit 1; }

rm -rf build && mkdir build
cp "$DLTDIR"/{dltlib.cpp,dltlib.h,node_que.cpp,random.c,random.h} build/
sed -i 's|#include "/papers/cpp_lib/random.h"|#include "random.h"|' build/random.c
cp mandel_dlt.cpp build/

g++ build/mandel_dlt.cpp -o build/mandel_dlt -lglpk

./build/mandel_dlt       | tee results.txt
./build/mandel_dlt sweep > sweep.csv
echo "wrote results.txt and sweep.csv"
