# Cache-Aware Charge Deposition for Electrostatic PIC Simulations

A reproducible two-dimensional electrostatic Particle-In-Cell (PIC) framework for evaluating charge deposition schemes (NGP, CIC, TSC, Esirkepov) with cache-aware CPU optimizations, OpenMP parallelization, and physics validation benchmarks.

## Prerequisites

- CMake 3.18+
- C++17 compiler (Apple clang or GCC)
- [FFTW3](https://www.fftw.org/) (`brew install fftw` on macOS)
- [libomp](https://github.com/llvm/llvm-project) (`brew install libomp` on macOS)
- Python 3.10+ with matplotlib and pandas (for figure generation)

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Optional CUDA deposition kernels:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA=ON
cmake --build build -j
```

## Run Tests

```bash
./build/pic_test
```

## Run Benchmarks

Full benchmark suite (microbenchmarks, energy-drift simulations, and Langmuir validation):

```bash
bash scripts/run_all_benchmarks.sh
```

Results are written to `results/` and archived to `data/benchmarks/`.

## Regenerate Paper Figures

```bash
pip install -r requirements.txt
python3 scripts/plot_results.py
```

## Compile Paper

Requires [Tectonic](https://tectonic-typesetting.github.io/) or a LaTeX distribution with `pdflatex` and `bibtex`:

```bash
bash scripts/compile_paper.sh
```

Output: `paper/main.pdf`

## Project Layout

| Path | Description |
|------|-------------|
| `src/` | C++ PIC simulation and deposition kernels |
| `scripts/` | Benchmark runner, figure plotting, paper compilation |
| `data/benchmarks/` | Archived CSV results for figure regeneration |
| `paper/` | LaTeX source and generated figures |
| `results/` | Ephemeral benchmark output (timestamped CSVs) |

## License

MIT License. See [LICENSE](LICENSE).
