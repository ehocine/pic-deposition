# PIC Deposition Benchmark Suite

A reproducible two-dimensional electrostatic Particle-In-Cell (PIC) reference program for evaluating charge deposition schemes (NGP, CIC, TSC, Esirkepov) with cache-aware CPU optimizations, optional CUDA backends, OpenMP parallelization, and bundled validation benchmarks.

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

GPU deposition benchmarks (CPU vs `GPU_Atomics` vs `GPU_Priv`):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA=ON
cmake --build build -j
./build/pic_benchmark --gpu
```

On Kaggle or other CUDA hosts:

```bash
bash scripts/kaggle_gpu_benchmark.sh
```

### Other benchmark modes

```bash
./build/pic_benchmark --validate      # Langmuir + conservation + two-stream + noise grid
./build/pic_benchmark --timestep      # full PIC timestep breakdown
./build/pic_benchmark --amortized       # sort-interval sweep
./build/pic_benchmark --conservation    # 2000-step conservation study
```

## Regenerate Figures

From archived CSVs in `data/benchmarks/`:

```bash
pip install -r requirements.txt
python3 scripts/plot_results.py
```

Output PDFs are written to `figures/` (not tracked in git by default).

## Project Layout

| Path | Description |
|------|-------------|
| `src/` | C++ PIC simulation and deposition kernels |
| `scripts/` | Benchmark runner and figure plotting |
| `data/benchmarks/` | Archived CSV results for reproducibility |
| `examples/` | Example benchmark shell scripts |
| `docs/` | Usage notes |
| `results/` | Ephemeral benchmark output (timestamped CSVs) |

## Citation

If you use this code or benchmark data in published work, please cite:

```bibtex
@software{hocine2026picdeposition,
  author = {Hocine, Elhadj},
  title = {PIC Deposition Benchmark Suite},
  year = {2026},
  url = {https://github.com/ehocine/pic-deposition},
  version = {1.0.0},
  doi = {10.5281/zenodo.20607455}
}
```

Metadata is in [`CITATION.cff`](CITATION.cff) (used by GitHub’s “Cite this repository” and by Zenodo when `.zenodo.json` is absent).

### Zenodo DOI

Permanent archive: [10.5281/zenodo.20607455](https://doi.org/10.5281/zenodo.20607455)

## Full Reproducibility Workflow

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/pic_test
bash scripts/run_all_benchmarks.sh
pip install -r requirements.txt
python3 scripts/plot_results.py
```

GPU benchmarks require CUDA (`-DBUILD_CUDA=ON`) or Kaggle (`scripts/kaggle_gpu_benchmark.sh`).

## License

MIT License. See [LICENSE](LICENSE).
