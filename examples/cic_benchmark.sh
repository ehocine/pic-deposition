#!/usr/bin/env bash
# Minimal reproduction of one CIC microbenchmark row (N_p=1e5, 128^2, SoA sorted).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT}/build"
cmake -S "${ROOT}" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD}" -j
"${BUILD}/pic_test"
cd "${BUILD}"
./pic_benchmark
echo "Compare latest results/benchmark_*.csv CIC SoA sorted 128x128 100000 row to data/benchmarks/benchmark.csv"
