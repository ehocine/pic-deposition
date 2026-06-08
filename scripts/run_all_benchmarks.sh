#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT}/build"
RESULTS="${ROOT}/results"
ARCHIVE="${ROOT}/data/benchmarks"

mkdir -p "${BUILD}" "${RESULTS}" "${ARCHIVE}"

if [[ ! -x "${BUILD}/pic_benchmark" ]]; then
  cmake -S "${ROOT}" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release
  cmake --build "${BUILD}" -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
fi

MODE="${1:---quick}"
cd "${BUILD}"
./pic_benchmark "${MODE}"
if [[ "${MODE}" != "--sim" && "${MODE}" != "--validate" ]]; then
  ./pic_benchmark --sim
  ./pic_benchmark --validate
fi

LATEST="$(ls -t "${RESULTS}"/benchmark_*.csv 2>/dev/null | grep -v benchmark_sim | head -1 || true)"
SIM="$(ls -t "${RESULTS}"/benchmark_sim_*.csv 2>/dev/null | head -1 || true)"
VAL="$(ls -t "${RESULTS}"/validation_*.csv 2>/dev/null | head -1 || true)"

if [[ -n "${LATEST}" ]]; then
  cp "${LATEST}" "${ARCHIVE}/benchmark.csv"
  echo "Archived ${LATEST} -> ${ARCHIVE}/benchmark.csv"
fi
if [[ -n "${SIM}" ]]; then
  cp "${SIM}" "${ARCHIVE}/benchmark_sim.csv"
  echo "Archived ${SIM} -> ${ARCHIVE}/benchmark_sim.csv"
fi
if [[ -n "${VAL}" ]]; then
  cp "${VAL}" "${ARCHIVE}/validation.csv"
  echo "Archived ${VAL} -> ${ARCHIVE}/validation.csv"
fi

if [[ -n "${LATEST}" ]]; then
  cd "${ROOT}"
  python3 "${ROOT}/scripts/plot_results.py"
fi
