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

GPU deposition benchmarks (CPU vs `GPU_Atomics` vs `GPU_Priv`):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA=ON
cmake --build build -j
./build/pic_benchmark --gpu
```

## Conservation Study

Long-run (2000-step) charge and energy conservation at $N_p=10^4$, $64^2$ grid:

```bash
./build/pic_benchmark --conservation
```

Output is archived to `data/benchmarks/conservation_study.csv` by `run_all_benchmarks.sh`.

## Kaggle (free GPU)

1. Create a Kaggle notebook and enable **GPU** (Settings → Accelerator → GPU T4).
2. Attach this repo as a [Dataset](https://www.kaggle.com/docs/datasets) or push to GitHub and upload.
3. Run:

```bash
bash scripts/kaggle_gpu_benchmark.sh
```

The script auto-copies from read-only `/kaggle/input/` to `/kaggle/working/pic` before building.
Results are written to `data/benchmarks/benchmark_gpu.csv` inside the working copy and copied to `/kaggle/working/benchmark_gpu.csv` for download.

### Re-run after GPU kernel updates

After updating `src/deposition/cuda/deposition_cuda.cu` (e.g. multi-block privatized tiles), re-run on Kaggle to refresh GPU benchmarks:

1. Upload the updated project as a Kaggle dataset or sync from GitHub.
2. Run `bash scripts/kaggle_gpu_benchmark.sh` in a GPU-enabled notebook.
3. Download `/kaggle/working/benchmark_gpu.csv` and replace `data/benchmarks/benchmark_gpu.csv` locally.
4. Regenerate figures: `python3 scripts/plot_results.py`.

Verify `GPU_Priv` CIC rows at $128^2$ are competitive with `GPU_Atomics` (not hundreds of ms as with the old single-block kernel).

## Regenerate Paper Figures

```bash
pip install -r requirements.txt
python3 scripts/plot_results.py
```

## Compile Paper

IEEE conference draft:

```bash
bash scripts/compile_paper.sh
```

Output: `paper/main.pdf`

CPC journal draft (elsarticle + Program Summary):

```bash
bash scripts/compile_cpc_paper.sh
```

Output: `paper/cpc/main.pdf`

## Project Layout

| Path | Description |
|------|-------------|
| `src/` | C++ PIC simulation and deposition kernels |
| `scripts/` | Benchmark runner, figure plotting, paper compilation |
| `data/benchmarks/` | Archived CSV results for figure regeneration |
| `paper/` | LaTeX source and generated figures |
| `results/` | Ephemeral benchmark output (timestamped CSVs) |

## Citation

If you use this code or benchmark data in published work, please cite:

```bibtex
@software{hocine2026picdeposition,
  author = {Hocine, Elhadj},
  title = {Cache-Aware and GPU-Accelerated Charge Deposition for Electrostatic PIC Simulations},
  year = {2026},
  url = {https://github.com/ehocine/pic-deposition},
  version = {1.0.0},
  doi = {10.5281/zenodo.TODO-DOI}
}
```

Metadata is also in [`CITATION.cff`](CITATION.cff) for GitHub and Zenodo integration.

### Zenodo DOI

1. Push this repository to GitHub (`https://github.com/ehocine/pic-deposition`).
2. Enable the [Zenodo–GitHub integration](https://zenodo.org/account/settings/github/).
3. Create a GitHub release tagged `v1.0.0`.
4. Replace `TODO-DOI` in `CITATION.cff`, `paper/main.tex`, and this README with the assigned Zenodo DOI.

## Full Reproducibility Workflow

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/pic_test
bash scripts/run_all_benchmarks.sh          # CPU microbenchmarks + sim + validation
./build/pic_benchmark --conservation        # long-run conservation study
pip install -r requirements.txt
python3 scripts/plot_results.py               # figures from data/benchmarks/
bash scripts/compile_paper.sh               # paper/main.pdf
```

GPU benchmarks require CUDA (`-DBUILD_CUDA=ON`) or Kaggle (`scripts/kaggle_gpu_benchmark.sh`).

## License

MIT License. See [LICENSE](LICENSE).
