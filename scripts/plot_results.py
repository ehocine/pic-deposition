#!/usr/bin/env python3
"""Generate paper figures from benchmark CSV results."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ARCHIVED_NAMES = {
    "benchmark_": "benchmark.csv",
    "benchmark_sim_": "benchmark_sim.csv",
    "validation_": "validation.csv",
}


def resolve_csv(results_dir: Path, prefix: str = "benchmark_") -> Path:
    archived = results_dir / ARCHIVED_NAMES[prefix]
    if archived.exists():
        return archived
    files = sorted(results_dir.glob(f"{prefix}*.csv"), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(f"No {prefix}*.csv files in {results_dir}")
    return files[-1]


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["grid_n"] = df["grid"].str.extract(r"(\d+)").astype(int)
    if "deposit_ms" not in df.columns:
        df["deposit_ms"] = df["deposition_ms"]
    if "sort_ms" not in df.columns:
        df["sort_ms"] = 0.0
    return df


def filter_deposition(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["backend"] == "CPU") & (df["threads"] == 1)].copy()


def require_sim_data(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["energy_drift"] != 0.0].copy()
    if sub.empty:
        raise ValueError("No simulation rows with energy_drift; run ./pic_benchmark --sim first.")
    return sub


def fig_runtime_vs_particles(df: pd.DataFrame, out: Path) -> None:
    sub = filter_deposition(df)
    sub = sub[(sub["layout"] == "SoA") & (sub["sorted"] == "on") & (sub["grid_n"] == 128)]
    plt.figure(figsize=(7, 4.5))
    for scheme, group in sub.groupby("scheme"):
        g = group.sort_values("num_particles")
        plt.loglog(g["num_particles"], g["deposit_ms"], marker="o", label=scheme)
    plt.xlabel("Number of particles")
    plt.ylabel("Deposition kernel time (ms)")
    plt.title("Deposition runtime vs particle count (128x128, SoA, sorted)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def fig_bandwidth_vs_grid(df: pd.DataFrame, out: Path) -> None:
    sub = filter_deposition(df)
    sub = sub[(sub["layout"] == "SoA") & (sub["sorted"] == "on") & (sub["num_particles"] == 100000)]
    plt.figure(figsize=(7, 4.5))
    for scheme, group in sub.groupby("scheme"):
        g = group.sort_values("grid_n")
        plt.plot(g["grid_n"], g["bandwidth_gbs"], marker="s", label=scheme)
    plt.xlabel("Grid size")
    plt.ylabel("Effective bandwidth (GB/s)")
    plt.title("Memory bandwidth vs grid resolution (1e5 particles)")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def fig_charge_error(df: pd.DataFrame, out: Path) -> None:
    sub = filter_deposition(df)
    sub = sub[(sub["layout"] == "SoA") & (sub["sorted"] == "on") & (sub["grid_n"] == 128) & (sub["num_particles"] == 100000)]
    order = ["NGP", "CIC", "TSC", "Esirkepov"]
    means = sub.groupby("scheme")["charge_error"].mean().reindex(order)
    plt.figure(figsize=(6, 4.5))
    plt.bar(means.index.astype(str), means.values)
    plt.yscale("log")
    plt.ylabel("Charge conservation error")
    plt.title("Charge conservation by deposition scheme")
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def fig_energy_drift(df: pd.DataFrame, out: Path) -> None:
    sub = require_sim_data(df)
    sub = sub[(sub["grid_n"] == 128) & (sub["num_particles"] == 100000)]
    order = ["NGP", "CIC", "TSC", "Esirkepov"]
    grouped = sub.groupby("scheme")["energy_drift"].mean().reindex(order)
    plt.figure(figsize=(6, 4.5))
    plt.bar(grouped.index.astype(str), grouped.values)
    plt.ylabel("Relative energy drift")
    plt.title("Energy drift from 500-step simulations")
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def fig_cpu_scaling(df: pd.DataFrame, out: Path) -> None:
    sub = df[(df["backend"] == "CPU") & (df["layout"] == "SoA") & (df["sorted"] == "off") &
             (df["scheme"] == "CIC") & (df["grid_n"] == 128) & (df["num_particles"] == 100000)]
    if sub.empty:
        return
    g = sub.groupby("threads")["deposit_ms"].mean().reset_index().sort_values("threads")
    baseline = g.loc[g["threads"] == g["threads"].min(), "deposit_ms"].iloc[0]
    speedup = baseline / g["deposit_ms"]
    plt.figure(figsize=(6, 4.5))
    plt.plot(g["threads"], speedup, marker="o", label="Measured")
    plt.plot(g["threads"], g["threads"] / g["threads"].min(), "--", label="Ideal")
    plt.xlabel("OpenMP threads")
    plt.ylabel("Speedup")
    plt.title("CPU strong scaling (CIC, SoA, unsorted, privatized)")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def fig_spectral_noise(df: pd.DataFrame, out: Path) -> None:
    sub = filter_deposition(df)
    sub = sub[(sub["layout"] == "SoA") & (sub["grid_n"] == 128) & (sub["num_particles"] == 100000)]
    if "spectral_noise" not in sub.columns or sub["spectral_noise"].max() <= 0:
        val = resolve_csv(Path("data/benchmarks"), "validation_")
        vdf = pd.read_csv(val)
        order = ["NGP", "CIC", "TSC", "Esirkepov"]
        noise = vdf.groupby("scheme")["spectral_noise"].mean().reindex(order)
    else:
        order = ["NGP", "CIC", "TSC", "Esirkepov"]
        noise = sub.groupby("scheme")["spectral_noise"].mean().reindex(order)
    plt.figure(figsize=(6, 4.5))
    plt.bar(noise.index.astype(str), noise.values)
    plt.ylabel("High-k spectral noise ratio")
    plt.title("Grid noise vs deposition scheme")
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def fig_sort_breakdown(df: pd.DataFrame, out: Path) -> None:
    sub = filter_deposition(df)
    sub = sub[(sub["scheme"] == "CIC") & (sub["grid_n"] == 128) & (sub["num_particles"] == 100000) &
              (sub["layout"] == "SoA") & (sub["sorted"] == "on")]
    if sub.empty:
        return
    row = sub.iloc[0]
    labels = ["sort", "deposit"]
    values = [row["sort_ms"], row["deposit_ms"]]
    plt.figure(figsize=(5, 4.5))
    plt.bar(labels, values)
    plt.ylabel("Time (ms)")
    plt.title("Sort vs deposit cost (CIC, SoA, sorted)")
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def write_summary(df: pd.DataFrame, out: Path, validation_path: Path | None) -> None:
    sub = filter_deposition(df)
    sub = sub[(sub["layout"] == "SoA") & (sub["sorted"] == "on") & (sub["grid_n"] == 128) & (sub["num_particles"] == 100000)]
    lines = ["Benchmark summary (128 grid, 1e5 particles, SoA sorted):", ""]
    table_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Representative CPU deposition results ($N_p=10^5$, $128^2$ grid, SoA, sorted).}",
        "\\label{tab:summary}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Scheme & Sort & Deposit & Thrpt. & $\\epsilon_Q$ \\\\",
        " & (ms) & (ms) & ($10^8$ p/s) & \\\\",
        "\\midrule",
    ]
    for scheme in ["NGP", "CIC", "TSC", "Esirkepov"]:
        row = sub[sub["scheme"] == scheme]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(
            f"- {scheme}: sort={r['sort_ms']:.3f} ms, deposit={r['deposit_ms']:.3f} ms, "
            f"throughput={r['throughput']:.2e}, charge_error={r['charge_error']:.2e}"
        )
        table_lines.append(
            f"{scheme} & {r['sort_ms']:.3f} & {r['deposit_ms']:.3f} & {r['throughput'] / 1e8:.2f} & {r['charge_error']:.2e} \\\\"
        )
    table_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_name("benchmark_summary.tex").write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    if validation_path and validation_path.exists():
        vdf = pd.read_csv(validation_path)
        val_lines = ["", "Validation summary:", ""]
        for _, r in vdf.iterrows():
            val_lines.append(
                f"- {r['scheme']}: spectral_noise={r['spectral_noise']:.4f}, "
                f"omega_meas={r['omega_measured']:.4f}, passed={int(r['passed'])}"
            )
        with out.open("a", encoding="utf-8") as f:
            f.write("\n".join(val_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--validation-csv", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--figures-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    results_dir = args.results_dir
    csv_path = args.csv or resolve_csv(results_dir)
    validation_path = args.validation_csv
    if validation_path is None:
        try:
            validation_path = resolve_csv(results_dir, "validation_")
        except FileNotFoundError:
            validation_path = None

    args.figures_dir.mkdir(parents=True, exist_ok=True)

    bench_df = load_data(csv_path)
    sim_path = None
    try:
        sim_path = resolve_csv(results_dir, "benchmark_sim_")
        sim_df = load_data(sim_path)
        bench_df = pd.concat([bench_df, sim_df], ignore_index=True)
    except FileNotFoundError:
        pass

    fig_runtime_vs_particles(bench_df, args.figures_dir / "fig1_runtime_vs_particles.pdf")
    fig_bandwidth_vs_grid(bench_df, args.figures_dir / "fig2_bandwidth_vs_grid.pdf")
    fig_charge_error(bench_df, args.figures_dir / "fig3_charge_error.pdf")
    fig_energy_drift(bench_df, args.figures_dir / "fig4_energy_drift.pdf")
    fig_cpu_scaling(bench_df, args.figures_dir / "fig5_cpu_scaling.pdf")
    fig_spectral_noise(bench_df, args.figures_dir / "fig6_spectral_noise.pdf")
    fig_sort_breakdown(bench_df, args.figures_dir / "fig7_layout_sort.pdf")
    write_summary(bench_df, args.figures_dir / "benchmark_summary.txt", validation_path)
    print(f"Generated figures from {csv_path}" + (f" and {sim_path}" if sim_path else ""))


if __name__ == "__main__":
    main()
