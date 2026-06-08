#!/usr/bin/env bash
# Run on Kaggle with GPU enabled (Settings -> Accelerator -> GPU).
# Paste into notebook cells or run after attaching this repo as a dataset.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT}/build"
ARCHIVE="${ROOT}/data/benchmarks"

echo "=== GPU info ==="
nvidia-smi || { echo "No GPU detected. Enable GPU in notebook Settings." >&2; exit 1; }
nvcc --version

echo "=== Installing dependencies ==="
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq cmake build-essential libfftw3-dev
fi

echo "=== Building with CUDA ==="
cmake -S "${ROOT}" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build "${BUILD}" -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"

echo "=== Running tests ==="
"${BUILD}/pic_test"

echo "=== Running GPU benchmark matrix ==="
mkdir -p "${ROOT}/results" "${ARCHIVE}"
cd "${BUILD}"
./pic_benchmark --gpu

LATEST="$(ls -t "${ROOT}"/results/benchmark_*.csv 2>/dev/null | grep -v benchmark_sim | head -1)"
if [[ -n "${LATEST}" ]]; then
  cp "${LATEST}" "${ARCHIVE}/benchmark_gpu.csv"
  echo "Archived GPU results to ${ARCHIVE}/benchmark_gpu.csv"
fi

echo "=== Done ==="
ls -la "${ARCHIVE}/benchmark_gpu.csv" 2>/dev/null || true
