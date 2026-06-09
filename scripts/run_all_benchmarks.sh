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
  ./pic_benchmark --timestep
  ./pic_benchmark --amortized
fi

LATEST="$(ls -t "${RESULTS}"/benchmark_*.csv 2>/dev/null | grep -v benchmark_sim | head -1 || true)"
SIM="$(ls -t "${RESULTS}"/benchmark_sim_*.csv 2>/dev/null | head -1 || true)"
VAL="$(ls -t "${RESULTS}"/validation_*.csv 2>/dev/null | head -1 || true)"
CONS="$(ls -t "${RESULTS}"/conservation_study_*.csv 2>/dev/null | head -1 || true)"
TS="$(ls -t "${RESULTS}"/timestep_profile_*.csv 2>/dev/null | head -1 || true)"
TW="$(ls -t "${RESULTS}"/two_stream_validation_*.csv 2>/dev/null | head -1 || true)"
AMORT="$(ls -t "${RESULTS}"/amortized_timestep_*.csv 2>/dev/null | head -1 || true)"
NOISE="$(ls -t "${RESULTS}"/noise_vs_grid_*.csv 2>/dev/null | head -1 || true)"

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
if [[ -n "${CONS}" ]]; then
  cp "${CONS}" "${ARCHIVE}/conservation_study.csv"
  echo "Archived ${CONS} -> ${ARCHIVE}/conservation_study.csv"
fi
if [[ -n "${TS}" ]]; then
  cp "${TS}" "${ARCHIVE}/timestep_profile.csv"
  echo "Archived ${TS} -> ${ARCHIVE}/timestep_profile.csv"
fi
if [[ -n "${TW}" ]]; then
  cp "${TW}" "${ARCHIVE}/two_stream_validation.csv"
  echo "Archived ${TW} -> ${ARCHIVE}/two_stream_validation.csv"
fi
if [[ -n "${AMORT}" ]]; then
  cp "${AMORT}" "${ARCHIVE}/amortized_timestep.csv"
  echo "Archived ${AMORT} -> ${ARCHIVE}/amortized_timestep.csv"
fi
if [[ -n "${NOISE}" ]]; then
  cp "${NOISE}" "${ARCHIVE}/noise_vs_grid.csv"
  echo "Archived ${NOISE} -> ${ARCHIVE}/noise_vs_grid.csv"
fi

if [[ -n "${LATEST}" ]]; then
  cd "${ROOT}"
  python3 "${ROOT}/scripts/plot_results.py"
fi
