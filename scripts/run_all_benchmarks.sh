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
if [[ "${MODE}" != "--sim" && "${MODE}" != "--validate" && "${MODE}" != "--physics" && "${MODE}" != "--timeseries" ]]; then
  ./pic_benchmark --sim
  ./pic_benchmark --validate
  ./pic_benchmark --physics
  ./pic_benchmark --timestep
  ./pic_benchmark --amortized
fi

LATEST="$(ls -t "${RESULTS}"/benchmark_*.csv 2>/dev/null | grep -v benchmark_sim | head -1 || true)"
SIM="$(ls -t "${RESULTS}"/benchmark_sim_*.csv 2>/dev/null | head -1 || true)"
VAL="$(ls -t "${RESULTS}"/validation_[0-9]*.csv 2>/dev/null | head -1 || true)"
CONS="$(ls -t "${RESULTS}"/conservation_study_*.csv 2>/dev/null | head -1 || true)"
TS="$(ls -t "${RESULTS}"/timestep_profile_*.csv 2>/dev/null | head -1 || true)"
TW="$(ls -t "${RESULTS}"/two_stream_validation_*.csv 2>/dev/null | head -1 || true)"
AMORT="$(ls -t "${RESULTS}"/amortized_timestep_*.csv 2>/dev/null | head -1 || true)"
NOISE="$(ls -t "${RESULTS}"/noise_vs_grid_*.csv 2>/dev/null | head -1 || true)"
PTS="$(ls -t "${RESULTS}"/physics_timeseries_*.csv 2>/dev/null | head -1 || true)"
LC="$(ls -t "${RESULTS}"/langmuir_convergence_*.csv 2>/dev/null | head -1 || true)"
Q1D="$(ls -t "${RESULTS}"/two_stream_quasi1d_*.csv 2>/dev/null | head -1 || true)"
LD="$(ls -t "${RESULTS}"/landau_damping_*.csv 2>/dev/null | head -1 || true)"
MS="$(ls -t "${RESULTS}"/validation_multi_seed_*.csv 2>/dev/null | head -1 || true)"
PROD="$(ls -t "${RESULTS}"/production_sim_*.csv 2>/dev/null | head -1 || true)"

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
if [[ -n "${PTS}" ]]; then
  cp "${PTS}" "${ARCHIVE}/physics_timeseries.csv"
  echo "Archived ${PTS} -> ${ARCHIVE}/physics_timeseries.csv"
fi
if [[ -n "${LC}" ]]; then
  cp "${LC}" "${ARCHIVE}/langmuir_convergence.csv"
  echo "Archived ${LC} -> ${ARCHIVE}/langmuir_convergence.csv"
fi
if [[ -n "${Q1D}" ]]; then
  cp "${Q1D}" "${ARCHIVE}/two_stream_quasi1d.csv"
  echo "Archived ${Q1D} -> ${ARCHIVE}/two_stream_quasi1d.csv"
fi
if [[ -n "${LD}" ]]; then
  cp "${LD}" "${ARCHIVE}/landau_damping.csv"
  echo "Archived ${LD} -> ${ARCHIVE}/landau_damping.csv"
fi
if [[ -n "${MS}" ]]; then
  cp "${MS}" "${ARCHIVE}/validation_multi_seed.csv"
  echo "Archived ${MS} -> ${ARCHIVE}/validation_multi_seed.csv"
fi
if [[ -n "${PROD}" ]]; then
  cp "${PROD}" "${ARCHIVE}/production_sim.csv"
  echo "Archived ${PROD} -> ${ARCHIVE}/production_sim.csv"
fi

if [[ -n "${LATEST}" ]]; then
  cd "${ROOT}"
  python3 "${ROOT}/scripts/plot_results.py"
fi
