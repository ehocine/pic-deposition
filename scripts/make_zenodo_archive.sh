#!/usr/bin/env bash
# Create a source archive suitable for manual Zenodo upload.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${1:-v1.0.0}"
OUT_DIR="${ROOT}/dist"
ARCHIVE="${OUT_DIR}/pic-deposition-${VERSION}-source.zip"

mkdir -p "${OUT_DIR}"
rm -f "${ARCHIVE}"

cd "${ROOT}"
zip -r "${ARCHIVE}" . \
  -x './build/*' \
  -x './results/*' \
  -x './figures/*' \
  -x './paper/*' \
  -x './dist/*' \
  -x './.git/*' \
  -x '*/__pycache__/*' \
  -x '*.pyc' \
  -x '*/.DS_Store' \
  -x './benchmark_gpu.csv' \
  -x '*/agent-tools/*'

echo "Created ${ARCHIVE}"
ls -lh "${ARCHIVE}"
