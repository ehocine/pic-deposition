#!/usr/bin/env bash
# Run on Kaggle with GPU enabled (Settings -> Accelerator -> GPU T4).
# Paste into notebook cells or run after attaching this repo as a dataset.
#
# Re-run after CUDA kernel changes (e.g. multi-block GPU_Priv tiles):
#   1. Upload updated project to Kaggle input
#   2. Run this script in a GPU notebook
#   3. Download /kaggle/working/benchmark_gpu.csv
#   4. Replace data/benchmarks/benchmark_gpu.csv locally
#   5. python3 scripts/plot_results.py
set -euo pipefail

SOURCE="$(cd "$(dirname "$0")/.." && pwd)"

# /kaggle/input is read-only; copy to /kaggle/working before building.
if [[ "${SOURCE}" == /kaggle/input/* ]]; then
  WORK="/kaggle/working/pic"
  echo "=== Copying project to writable directory: ${WORK} ==="
  rm -rf "${WORK}"
  cp -r "${SOURCE}" "${WORK}"
  ROOT="${WORK}"
else
  ROOT="${SOURCE}"
fi

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

echo "=== Building with CUDA in ${ROOT} ==="
cmake -S "${ROOT}" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build "${BUILD}" -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"

echo "=== Running tests (includes multi-tile GPU_Priv at 128^2) ==="
"${BUILD}/pic_test"

echo "=== Running GPU benchmark matrix (CPU, GPU_Atomics, GPU_Priv for CIC) ==="
mkdir -p "${ROOT}/results" "${ARCHIVE}"
cd "${BUILD}"
./pic_benchmark --gpu

LATEST="$(ls -t "${ROOT}"/results/benchmark_*.csv 2>/dev/null | grep -v benchmark_sim | head -1)"
if [[ -n "${LATEST}" ]]; then
  cp "${LATEST}" "${ARCHIVE}/benchmark_gpu.csv"
  cp "${LATEST}" /kaggle/working/benchmark_gpu.csv 2>/dev/null || true
  echo "Archived GPU results to ${ARCHIVE}/benchmark_gpu.csv"
  echo ""
  echo "=== GPU_Priv CIC sanity check (128^2, 1e5 particles) ==="
  awk -F, '$2=="128x128" && $3=="CIC" && $6=="GPU_Priv" && $1==100000 {printf "GPU_Priv deposit_ms=%s (expect <5 ms, not ~300 ms)\n", $8}' "${ARCHIVE}/benchmark_gpu.csv" || true
fi

echo "=== Done ==="
echo "Download benchmark_gpu.csv from the Kaggle Output tab or /kaggle/working/benchmark_gpu.csv"
ls -la "${ARCHIVE}/benchmark_gpu.csv" 2>/dev/null || true
ls -la /kaggle/working/benchmark_gpu.csv 2>/dev/null || true
