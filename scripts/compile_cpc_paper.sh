#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAPER="${ROOT}/paper/cpc"

if command -v tectonic >/dev/null 2>&1; then
  cd "${PAPER}"
  tectonic main.tex
  echo "Built ${PAPER}/main.pdf"
  exit 0
fi

if command -v pdflatex >/dev/null 2>&1; then
  cd "${PAPER}"
  pdflatex -interaction=nonstopmode main.tex
  bibtex main || true
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
  echo "Built ${PAPER}/main.pdf"
  exit 0
fi

echo "No LaTeX engine found. Install tectonic or MacTeX, then rerun." >&2
exit 1
