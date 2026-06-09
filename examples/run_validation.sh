#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT}/build"
cmake -S "${ROOT}" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD}" -j
"${BUILD}/pic_validation"
"${BUILD}/pic_benchmark" --validate
echo "Validation CSVs written under results/; archive to data/benchmarks/ for figure regeneration."
