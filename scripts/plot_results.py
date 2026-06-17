#!/usr/bin/env python3
"""Generate paper figures from benchmark CSV results."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARCHIVED_NAMES = {
    "benchmark_": "benchmark.csv",
    "benchmark_sim_": "benchmark_sim.csv",
    "validation_": "validation.csv",
    "conservation_study_": "conservation_study.csv",
    "timestep_profile_": "timestep_profile.csv",
    "two_stream_validation_": "two_stream_validation.csv",
    "amortized_timestep_": "amortized_timestep.csv",
    "noise_vs_grid_": "noise_vs_grid.csv",
    "physics_timeseries_": "physics_timeseries.csv",
    "langmuir_convergence_": "langmuir_convergence.csv",
    "two_stream_quasi1d_": "two_stream_quasi1d.csv",
    "landau_damping_": "landau_damping.csv",
    "validation_multi_seed_": "validation_multi_seed.csv",
    "production_sim_": "production_sim.csv",
}

FLOPS_PER_PARTICLE = {"NGP": 10.0, "CIC": 40.0, "TSC": 90.0, "Esirkepov": 50.0}


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


def estimate_bytes(row: pd.Series) -> float:
    layout = str(row["layout"])
    record = 48.0 if layout == "AoS" else 40.0
    scheme = str(row["scheme"])
    writes = {"NGP": 1, "CIC": 4, "TSC": 9, "Esirkepov": 4}.get(scheme, 4)
    grid_n = int(row["grid_n"])
    return float(row["num_particles"]) * record + float(row["num_particles"]) * writes * 8.0 + grid_n * grid_n * 8.0


def estimate_flops(row: pd.Series) -> float:
    scheme = str(row["scheme"])
    return float(row["num_particles"]) * FLOPS_PER_PARTICLE.get(scheme, 40.0)


def fig_amortized_sort(ts_path: Path, out: Path) -> None:
    if not ts_path.exists():
        return
    df = pd.read_csv(ts_path)
    if "sort_interval" not in df.columns:
        return
    labels = ["push", "sort", "deposit", "poisson", "gather"]
    x = range(len(df))
    width = 0.6
    plt.figure(figsize=(7, 4.5))
    bottom = [0.0] * len(df)
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"]
    for label, color in zip(labels, colors):
        values = df[f"{label}_ms"].tolist()
        plt.bar(list(x), values, width=width, bottom=bottom, label=label, color=color)
        bottom = [b + v for b, v in zip(bottom, values)]
    tick_labels = [f"K={int(k)}" for k in df["sort_interval"]]
    plt.xticks(list(x), tick_labels)
    plt.xlabel("Sort interval (every K steps)")
    plt.ylabel("Time per timestep (ms)")
    plt.title("Amortized sort: CIC full timestep ($N_p=10^5$, $128^2$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def fig_baseline_comparison(cpu_df: pd.DataFrame, out: Path) -> None:
    sub = filter_deposition(cpu_df)
    sub = sub[(sub["scheme"] == "CIC") & (sub["grid_n"] == 128) & (sub["num_particles"] == 100000) &
              (sub["threads"] == 1)]
    naive = sub[(sub["layout"] == "AoS") & (sub["sorted"] == "off")]
    optimized = sub[(sub["layout"] == "SoA") & (sub["sorted"] == "on")]
    if naive.empty or optimized.empty:
        return
    labels = ["AoS unsorted", "SoA sorted"]
    values = [naive.iloc[0]["deposit_ms"], optimized.iloc[0]["deposit_ms"]]
    plt.figure(figsize=(5, 4.5))
    plt.bar(labels, values)
    plt.ylabel("Deposition kernel time (ms)")
    plt.title("Naive vs optimized CIC layout (M1 Pro)")
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def fig_conservation_study(cons_path: Path, out: Path) -> None:
    if not cons_path.exists():
        return
    df = pd.read_csv(cons_path)
    order = ["NGP", "CIC", "TSC", "Esirkepov"]
    charge = df.set_index("scheme")["max_charge_error"].reindex(order)
    drift = df.set_index("scheme")["final_energy_drift"].reindex(order)
    x = range(len(order))
    width = 0.35
    plt.figure(figsize=(7, 4.5))
    plt.bar([i - width / 2 for i in x], charge.values, width=width, label="Max charge error")
    plt.bar([i + width / 2 for i in x], drift.values, width=width, label="Energy drift")
    plt.xticks(list(x), order)
    plt.yscale("symlog", linthresh=1e-12)
    plt.ylabel("Magnitude")
    plt.title("2000-step conservation study")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def fig_gpu_runtime(gpu_df: pd.DataFrame, out: Path) -> None:
    sub = gpu_df[(gpu_df["layout"] == "SoA") & (gpu_df["sorted"] == "off") & (gpu_df["grid_n"] == 128) &
                 (gpu_df["backend"].isin(["CPU", "GPU_Atomics"]))]
    plt.figure(figsize=(7, 4.5))
    for scheme in ["NGP", "CIC", "TSC"]:
        for backend, style in [("CPU", "-o"), ("GPU_Atomics", "--s")]:
            g = sub[(sub["scheme"] == scheme) & (sub["backend"] == backend)].sort_values("num_particles")
            if g.empty:
                continue
            label = f"{scheme} {backend.replace('_', ' ')}"
            plt.loglog(g["num_particles"], g["deposit_ms"], style, label=label)
    plt.xlabel("Number of particles")
    plt.ylabel("Deposition kernel time (ms)")
    plt.title("CPU vs GPU atomic deposition (128x128, SoA)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def fig_gpu_speedup(gpu_df: pd.DataFrame, out: Path) -> None:
    sub = gpu_df[(gpu_df["layout"] == "SoA") & (gpu_df["sorted"] == "off") & (gpu_df["grid_n"] == 128) &
                 (gpu_df["backend"].isin(["CPU", "GPU_Atomics"]))]
    rows = []
    for scheme in ["NGP", "CIC", "TSC"]:
        for np in sorted(sub["num_particles"].unique()):
            cpu = sub[(sub["scheme"] == scheme) & (sub["num_particles"] == np) & (sub["backend"] == "CPU")]
            gpu = sub[(sub["scheme"] == scheme) & (sub["num_particles"] == np) & (sub["backend"] == "GPU_Atomics")]
            if cpu.empty or gpu.empty:
                continue
            rows.append({
                "scheme": scheme,
                "num_particles": np,
                "speedup": cpu.iloc[0]["deposit_ms"] / gpu.iloc[0]["deposit_ms"],
            })
    if not rows:
        return
    sdf = pd.DataFrame(rows)
    labels = [f"{int(r['num_particles']):.0e}" for r in rows[:3]]
    x = range(len(["NGP", "CIC", "TSC"]))
    width = 0.25
    plt.figure(figsize=(7, 4.5))
    for i, np in enumerate(sorted(sdf["num_particles"].unique())):
        vals = sdf[sdf["num_particles"] == np].set_index("scheme").reindex(["NGP", "CIC", "TSC"])["speedup"]
        offset = (i - 1) * width
        plt.bar([xi + offset for xi in x], vals.values, width=width, label=f"{int(np):.0e}")
    plt.xticks(list(x), ["NGP", "CIC", "TSC"])
    plt.ylabel("Speedup (CPU / GPU atomics)")
    plt.title("GPU speedup on Tesla T4 (128x128 grid)")
    plt.axhline(1.0, color="gray", ls="--", lw=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def fig_noise_vs_grid(noise_path: Path, out: Path) -> None:
    if not noise_path.exists():
        return
    df = pd.read_csv(noise_path)
    df["grid_n"] = df["grid"].str.extract(r"(\d+)").astype(int)
    order = ["NGP", "CIC", "TSC", "Esirkepov"]
    plt.figure(figsize=(7, 4.5))
    for scheme in order:
        sub = df[df["scheme"] == scheme].sort_values("grid_n")
        if sub.empty:
            continue
        plt.plot(sub["grid_n"], sub["spectral_noise"], marker="o", label=scheme)
    plt.xlabel("Grid size")
    plt.ylabel("Spectral noise ratio")
    plt.title("Spectral noise vs grid ($N_p=10^5$, SoA sorted)")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def fig_roofline(cpu_df: pd.DataFrame, gpu_df: pd.DataFrame | None, out: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    cpu_sub = filter_deposition(cpu_df)
    cpu_sub = cpu_sub[(cpu_sub["layout"] == "SoA") & (cpu_sub["sorted"] == "off") & (cpu_sub["grid_n"] == 128) &
                      (cpu_sub["num_particles"] == 100000)]
    for scheme, group in cpu_sub.groupby("scheme"):
        if scheme == "Esirkepov":
            continue
        row = group.iloc[0]
        bytes_total = estimate_bytes(row)
        ai = estimate_flops(row) / max(bytes_total, 1.0)
        gflops = estimate_flops(row) / max(row["deposit_ms"] / 1000.0, 1e-12) / 1e9
        plt.scatter(ai, gflops, marker="o", label=f"M1 {scheme}")
    if gpu_df is not None:
        gpu_sub = gpu_df[(gpu_df["layout"] == "SoA") & (gpu_df["sorted"] == "off") & (gpu_df["grid_n"] == 128) &
                         (gpu_df["num_particles"] == 100000) & (gpu_df["backend"] == "GPU_Atomics")]
        for scheme, group in gpu_sub.groupby("scheme"):
            row = group.iloc[0]
            bytes_total = estimate_bytes(row)
            ai = estimate_flops(row) / max(bytes_total, 1.0)
            gflops = estimate_flops(row) / max(row["deposit_ms"] / 1000.0, 1e-12) / 1e9
            plt.scatter(ai, gflops, marker="s", label=f"T4 GPU {scheme}")
    ai_line = [0.01, 1.0]
    plt.plot(ai_line, [0.05, 5.0], "--", color="gray", label="M1 ~5 GFLOP/s ref.")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Arithmetic intensity (FLOP/byte)")
    plt.ylabel("Achieved throughput (GFLOP/s)")
    plt.title("Roofline-style deposition performance (1e5, 128x128)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def fig_timestep_breakdown(ts_path: Path, out: Path) -> None:
    if not ts_path.exists():
        return
    df = pd.read_csv(ts_path)
    if "sort_interval" in df.columns:
        df = df[df["sort_interval"] == 1]
    labels = ["push", "sort", "deposit", "poisson", "gather"]
    x = range(len(df))
    width = 0.6
    plt.figure(figsize=(7, 4.5))
    bottom = [0.0] * len(df)
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"]
    for label, color in zip(labels, colors):
        values = df[f"{label}_ms"].tolist()
        plt.bar(list(x), values, width=width, bottom=bottom, label=label, color=color)
        bottom = [b + v for b, v in zip(bottom, values)]
    plt.xticks(list(x), [f"{int(n):.0e}" for n in df["num_particles"]])
    plt.xlabel("Number of particles")
    plt.ylabel("Time per timestep (ms)")
    plt.title("Full PIC timestep breakdown (CIC, SoA, sorted, 128x128)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def fig_two_stream(ts_path: Path, out: Path) -> None:
    if not ts_path.exists():
        return
    df = pd.read_csv(ts_path)
    order = ["NGP", "CIC", "TSC", "Esirkepov"]
    measured = df.set_index("scheme")["growth_rate_measured"].reindex(order)
    theory = df.set_index("scheme")["growth_rate_theory"].reindex(order)
    x = range(len(order))
    width = 0.35
    plt.figure(figsize=(7, 4.5))
    plt.bar([i - width / 2 for i in x], measured.values, width=width, label="Measured")
    plt.bar([i + width / 2 for i in x], theory.values, width=width, label="Theory")
    plt.xticks(list(x), order)
    plt.ylabel("Growth rate")
    plt.title("Two-stream instability growth rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def _load_optional_csv(results_dir: Path, basename: str, prefix: str) -> Path | None:
    archived = results_dir / basename
    if archived.exists():
        return archived
    try:
        return resolve_csv(results_dir, prefix)
    except FileNotFoundError:
        return None


def fig_langmuir_timeseries(pts_path: Path, out: Path) -> None:
    if pts_path is None or not pts_path.exists():
        return
    df = pd.read_csv(pts_path)
    sub = df[df["test"] == "langmuir"]
    if sub.empty:
        return
    plt.figure(figsize=(7, 4.5))
    for scheme in ["NGP", "CIC", "TSC", "Esirkepov"]:
        s = sub[sub["scheme"] == scheme]
        if s.empty:
            continue
        step = s["step"].values
        stride = max(1, len(step) // 2000)
        plt.plot(s["time"].values[::stride], s["field_energy"].values[::stride], label=scheme)
    plt.xlabel("Time")
    plt.ylabel("Field energy")
    plt.title("Langmuir wave field energy (all schemes)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def fig_twostream_logE(pts_path: Path, out: Path) -> None:
    if pts_path is None or not pts_path.exists():
        return
    df = pd.read_csv(pts_path)
    sub = df[df["test"] == "twostream"]
    if sub.empty:
        return
    plt.figure(figsize=(7, 4.5))
    for scheme in ["NGP", "CIC", "TSC", "Esirkepov"]:
        s = sub[sub["scheme"] == scheme]
        if s.empty:
            continue
        e = s["field_energy"].values
        t = s["time"].values
        e = np.maximum(e, 1e-30)
        plt.plot(t, np.log(e), label=scheme)
    plt.xlabel("Time")
    plt.ylabel("ln(field energy)")
    plt.title("Two-stream growth (log field energy)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def fig_energy_budget(pts_path: Path, out: Path) -> None:
    if pts_path is None or not pts_path.exists():
        return
    df = pd.read_csv(pts_path)
    sub = df[(df["test"] == "conservation") & (df["scheme"].isin(["CIC", "Esirkepov"]))]
    if sub.empty:
        return
    plt.figure(figsize=(7, 4.5))
    for scheme in ["CIC", "Esirkepov"]:
        s = sub[sub["scheme"] == scheme]
        if s.empty:
            continue
        stride = max(1, len(s) // 500)
        t = s["time"].values[::stride]
        plt.plot(t, s["kinetic_energy"].values[::stride], "--", label=f"{scheme} kin")
        plt.plot(t, s["field_energy"].values[::stride], ":", label=f"{scheme} field")
        plt.plot(t, s["total_energy"].values[::stride], "-", label=f"{scheme} total")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title("Energy budget (2000-step conservation run)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def fig_conservation_timeseries(pts_path: Path, out: Path) -> None:
    if pts_path is None or not pts_path.exists():
        return
    df = pd.read_csv(pts_path)
    sub = df[df["test"] == "conservation"]
    if sub.empty:
        return
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    scheme = "CIC"
    s = sub[sub["scheme"] == scheme]
    if s.empty:
        scheme = sub["scheme"].iloc[0]
        s = sub[sub["scheme"] == scheme]
    stride = max(1, len(s) // 500)
    t = s["time"].values[::stride]
    ax1.plot(t, s["charge_error"].values[::stride], "b-", label="Charge error")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Charge error", color="b")
    ax2 = ax1.twinx()
    ax2.plot(t, s["total_energy"].values[::stride], "r-", label="Total energy")
    ax2.set_ylabel("Total energy", color="r")
    plt.title(f"Conservation time series ({scheme})")
    fig.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def fig_langmuir_convergence(lc_path: Path, out: Path) -> None:
    if lc_path is None or not lc_path.exists():
        return
    df = pd.read_csv(lc_path)
    cic = df[df["scheme"] == "CIC"]
    if cic.empty:
        cic = df
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for g, grp in cic.groupby("grid_n"):
        axes[0].plot(grp["num_particles"], grp["omega_measured"], marker="o", label=f"{g}^2")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("N_p")
    axes[0].set_ylabel("omega_measured")
    axes[0].set_title("Langmuir frequency vs N_p")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, ls="--", alpha=0.4)
    np_ref = cic["num_particles"].median()
    gsub = cic[cic["num_particles"] == np_ref] if np_ref in cic["num_particles"].values else cic
    if gsub.empty:
        gsub = cic
    for np, grp in gsub.groupby("num_particles"):
        axes[1].plot(grp["grid_n"], grp["omega_measured"], marker="o", label=f"{int(np):.0e}")
    axes[1].set_xlabel("Grid N")
    axes[1].set_ylabel("omega_measured")
    axes[1].set_title("Langmuir frequency vs grid")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def fig_twostream_geometry(ts2d_path: Path, q1d_path: Path, out: Path) -> None:
    if ts2d_path is None or not ts2d_path.exists():
        return
    df2 = pd.read_csv(ts2d_path)
    rows = []
    for _, r in df2.iterrows():
        rows.append({"geometry": "2D 64^2", "scheme": r["scheme"],
                     "ratio": r["growth_rate_over_omega_p"]})
    if q1d_path is not None and q1d_path.exists():
        df1 = pd.read_csv(q1d_path)
        for _, r in df1.iterrows():
            rows.append({"geometry": "quasi-1D", "scheme": r["scheme"],
                         "ratio": r["growth_rate_over_omega_p"]})
    if not rows:
        return
    sdf = pd.DataFrame(rows)
    order = ["NGP", "CIC", "TSC", "Esirkepov"]
    geoms = ["2D 64^2", "quasi-1D"]
    x = range(len(order))
    width = 0.35
    plt.figure(figsize=(7, 4.5))
    for i, geom in enumerate(geoms):
        g = sdf[sdf["geometry"] == geom].set_index("scheme").reindex(order)
        if g["ratio"].isna().all():
            continue
        offset = (i - 0.5) * width
        plt.bar([xi + offset for xi in x], g["ratio"].values, width=width, label=geom)
    plt.xticks(list(x), order)
    plt.ylabel("gamma / omega_p")
    plt.title("Two-stream growth: 2D vs quasi-1D")
    plt.axhline(0.71, color="gray", ls="--", lw=0.8, label="1D theory ~0.71")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def fig_landau_damping(ld_path: Path, pts_path: Path, out: Path) -> None:
    if ld_path is None or not ld_path.exists():
        return
    df = pd.read_csv(ld_path)
    cic = df[df["scheme"] == "CIC"]
    if cic.empty:
        cic = df.iloc[[0]]
    row = cic.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    if pts_path is not None and pts_path.exists():
        pts = pd.read_csv(pts_path)
        sub = pts[(pts["test"] == "landau") & (pts["scheme"] == "CIC")]
        if sub.empty:
            sub = pts[(pts["test"] == "langmuir") & (pts["scheme"] == "CIC")]
        if not sub.empty:
            stride = max(1, len(sub) // 1000)
            axes[0].plot(sub["time"].values[::stride], sub["field_energy"].values[::stride])
            axes[0].set_xlabel("Time")
            axes[0].set_ylabel("Field energy")
            axes[0].set_title("Warm Langmuir field energy (CIC)")
    schemes = df["scheme"].tolist()
    meas = df["damping_rate_measured"].tolist()
    theory = df["damping_rate_theory"].tolist()
    x = range(len(schemes))
    w = 0.35
    axes[1].bar([i - w / 2 for i in x], meas, width=w, label="Measured")
    axes[1].bar([i + w / 2 for i in x], theory, width=w, label="Theory")
    axes[1].set_xticks(list(x), schemes, rotation=15)
    axes[1].set_ylabel("Damping rate")
    axes[1].set_title("Landau damping (CIC ratio={:.2f})".format(row.get("damping_ratio", 0)))
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def write_langmuir_convergence_summary(lc_path: Path, out: Path) -> None:
    if lc_path is None or not lc_path.exists():
        return
    df = pd.read_csv(lc_path)
    cic = df[(df["scheme"] == "CIC") & (df["grid_n"] == 128) & (df["num_particles"] == 200)]
    if cic.empty:
        cic = df[df["scheme"] == "CIC"].head(1)
    if cic.empty:
        return
    r = cic.iloc[0]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Langmuir convergence at $128^2$, $N_p=200$ (CIC).}",
        r"\label{tab:langmuir_conv}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Quantity & Value & Note \\",
        r"\midrule",
        rf"$\omega_{{\mathrm{{meas}}}}$ & {r['omega_measured']:.4f} & CIC \\",
        rf"$\omega_{{\mathrm{{meas}}}}/\omega_{{p,\mathrm{{macro}}}}$ & {r['omega_ratio']:.2f} & converged value \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_twostream_geometry_summary(ts2d_path: Path, q1d_path: Path, out: Path) -> None:
    if ts2d_path is None or q1d_path is None:
        return
    if not ts2d_path.exists() or not q1d_path.exists():
        return
    df2 = pd.read_csv(ts2d_path)
    df1 = pd.read_csv(q1d_path)
    c2 = df2[df2["scheme"] == "CIC"].iloc[0]
    c1 = df1[df1["scheme"] == "CIC"].iloc[0]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Two-stream $\gamma/\omega_p$: 2D vs quasi-1D (CIC).}",
        r"\label{tab:twostream_geom}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Geometry & $\gamma/\omega_p$ & Pass \\",
        r"\midrule",
        rf"2D $64^2$ & {c2['growth_rate_over_omega_p']:.1f} & {'yes' if int(c2.get('passed', 0)) else 'no'} \\",
        rf"Quasi-1D $256\times4$ & {c1['growth_rate_over_omega_p']:.1f} & {'yes' if int(c1.get('passed', 0)) else 'no'} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_landau_summary(ld_path: Path, out: Path) -> None:
    if ld_path is None or not ld_path.exists():
        return
    df = pd.read_csv(ld_path)
    cic = df[df["scheme"] == "CIC"]
    if cic.empty:
        return
    r = cic.iloc[0]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Landau damping rates (warm Langmuir, CIC).}",
        r"\label{tab:landau}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Quantity & Measured & Theory \\",
        r"\midrule",
        rf"Damping rate & {r['damping_rate_measured']:.4f} & {r['damping_rate_theory']:.4f} \\",
        rf"$k\lambda_D$ & {r['k_lambda_d']:.3f} & -- \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_gpu_data(results_dir: Path) -> pd.DataFrame | None:
    gpu_path = results_dir / "benchmark_gpu.csv"
    if not gpu_path.exists():
        return None
    return load_data(gpu_path)


def resolve_csv(results_dir: Path, prefix: str = "benchmark_") -> Path:
    archived = results_dir / ARCHIVED_NAMES[prefix]
    if archived.exists():
        return archived
    files = sorted(results_dir.glob(f"{prefix}*.csv"), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(f"No {prefix}*.csv files in {results_dir}")
    return files[-1]

def write_cross_platform_summary(cpu_df: pd.DataFrame, gpu_df: pd.DataFrame | None, out: Path) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Cross-platform CIC deposition ($128^2$ grid, SoA).}",
        "\\label{tab:cross_platform}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "$N_p$ & M1 CPU (ms) & M1 8T (ms) & T4 CPU (ms) & T4 GPU (ms) \\\\",
        "\\midrule",
    ]
    if gpu_df is None:
        out.write_text("\n".join(lines + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]) + "\n", encoding="utf-8")
        return
    # Use the full CPU frame (all thread counts) so the 8-thread column is not
    # pre-filtered away. filter_deposition() restricts to threads==1, which would
    # make the 8T lookup always empty.
    m1_cpu = cpu_df[cpu_df["backend"] == "CPU"].copy()
    for np in [100000, 1000000]:
        m1_base = m1_cpu[(m1_cpu["scheme"] == "CIC") & (m1_cpu["layout"] == "SoA") &
                         (m1_cpu["sorted"] == "off") & (m1_cpu["grid_n"] == 128) &
                         (m1_cpu["num_particles"] == np)]
        m1_1 = m1_base[m1_base["threads"] == 1]
        m1_8 = m1_base[m1_base["threads"] == 8]
        t4_cpu = gpu_df[(gpu_df["scheme"] == "CIC") & (gpu_df["backend"] == "CPU") & (gpu_df["grid_n"] == 128) &
                        (gpu_df["num_particles"] == np)]
        t4_gpu = gpu_df[(gpu_df["scheme"] == "CIC") & (gpu_df["backend"] == "GPU_Atomics") & (gpu_df["grid_n"] == 128) &
                        (gpu_df["num_particles"] == np)]
        if m1_1.empty or t4_cpu.empty or t4_gpu.empty:
            continue
        m1_8_cell = f"{m1_8.iloc[0]['deposit_ms']:.2f}" if not m1_8.empty else "--"
        lines.append(
            f"{int(np):.0e} & {m1_1.iloc[0]['deposit_ms']:.2f} & {m1_8_cell} & "
            f"{t4_cpu.iloc[0]['deposit_ms']:.2f} & {t4_gpu.iloc[0]['deposit_ms']:.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gpu_priv_summary(gpu_df: pd.DataFrame, out: Path) -> None:
    sub = gpu_df[(gpu_df["layout"] == "SoA") & (gpu_df["sorted"] == "off") & (gpu_df["grid_n"] == 128) &
                 (gpu_df["scheme"] == "CIC") & (gpu_df["backend"].isin(["GPU_Atomics", "GPU_Priv"]))]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{GPU atomic vs privatized CIC on Tesla T4 ($128^2$ grid, SoA).}",
        "\\label{tab:gpu_priv}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "$N_p$ & Atomics (ms) & Priv (ms) & Atomics speedup & Priv speedup \\\\",
        "\\midrule",
    ]
    for np in sorted(sub["num_particles"].unique()):
        cpu = gpu_df[(gpu_df["num_particles"] == np) & (gpu_df["grid_n"] == 128) & (gpu_df["scheme"] == "CIC") &
                     (gpu_df["backend"] == "CPU") & (gpu_df["layout"] == "SoA")]
        atom = sub[(sub["num_particles"] == np) & (sub["backend"] == "GPU_Atomics")]
        priv = sub[(sub["num_particles"] == np) & (sub["backend"] == "GPU_Priv")]
        if cpu.empty or atom.empty or priv.empty:
            continue
        c_ms = cpu.iloc[0]["deposit_ms"]
        a_ms = atom.iloc[0]["deposit_ms"]
        p_ms = priv.iloc[0]["deposit_ms"]
        lines.append(
            f"{int(np):.0e} & {a_ms:.2f} & {p_ms:.2f} & {c_ms / a_ms:.1f}$\\times$ & {c_ms / p_ms:.1f}$\\times$ \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gpu_summary(gpu_df: pd.DataFrame, out: Path) -> None:
    sub = gpu_df[(gpu_df["layout"] == "SoA") & (gpu_df["sorted"] == "off") & (gpu_df["grid_n"] == 128) &
                 (gpu_df["num_particles"] == 100000) & (gpu_df["backend"].isin(["CPU", "GPU_Atomics"]))]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{CPU vs GPU atomic deposition on Tesla T4 ($N_p=10^5$, $128^2$ grid, SoA).}",
        "\\label{tab:gpu_summary}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Scheme & CPU (ms) & GPU (ms) & Speedup \\\\",
        "\\midrule",
    ]
    for scheme in ["NGP", "CIC", "TSC"]:
        cpu = sub[(sub["scheme"] == scheme) & (sub["backend"] == "CPU")]
        gpu = sub[(sub["scheme"] == scheme) & (sub["backend"] == "GPU_Atomics")]
        if cpu.empty or gpu.empty:
            continue
        c_ms = cpu.iloc[0]["deposit_ms"]
        g_ms = gpu.iloc[0]["deposit_ms"]
        speedup = c_ms / g_ms
        lines.append(f"{scheme} & {c_ms:.2f} & {g_ms:.2f} & {speedup:.1f}$\\times$ \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(df: pd.DataFrame, out: Path, validation_path: Path | None) -> None:
    sub = filter_deposition(df)
    sub = sub[(sub["layout"] == "SoA") & (sub["sorted"] == "on") & (sub["grid_n"] == 128) & (sub["num_particles"] == 100000)]
    lines = ["Benchmark summary (128 grid, 1e5 particles, SoA sorted):", ""]
    table_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Representative CPU deposition results ($N_p=10^5$, $128^2$ grid, SoA, sorted).}",
        r"\label{tab:summary}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Scheme & Sort & Deposit & Thrpt. & $\epsilon_Q$ \\",
        r" & (ms) & (ms) & ($10^8$ p/s) & \\",
        r"\midrule",
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
        std_ms = r["deposit_std_ms"] if "deposit_std_ms" in r and pd.notna(r["deposit_std_ms"]) else 0.0
        deposit_cell = rf"{r['deposit_ms']:.3f} $\pm$ {std_ms:.3f}" if std_ms > 0 else f"{r['deposit_ms']:.3f}"
        table_lines.append(
            rf"{scheme} & {r['sort_ms']:.3f} & {deposit_cell} & {r['throughput'] / 1e8:.2f} & {r['charge_error']:.2e} \\"
        )
    table_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
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


def write_two_stream_summary(ts_path: Path, out: Path) -> None:
    if not ts_path.exists():
        return
    df = pd.read_csv(ts_path)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Two-stream growth rates from \texttt{--validate} (inter-scheme benchmark).}",
        r"\label{tab:twostream_sim}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Scheme & $\gamma_{\mathrm{meas}}$ & $\gamma/\omega_{p,\mathrm{macro}}$ & Pass \\",
        r"\midrule",
    ]
    for scheme in ["NGP", "CIC", "TSC", "Esirkepov"]:
        row = df[df["scheme"] == scheme]
        if row.empty:
            continue
        r = row.iloc[0]
        gamma = r["growth_rate_measured"]
        if "growth_rate_over_omega_p" in r and pd.notna(r["growth_rate_over_omega_p"]):
            ratio = r["growth_rate_over_omega_p"]
        elif "omega_p_macro" in r and pd.notna(r.get("omega_p_macro", float("nan"))) and r["omega_p_macro"] > 0:
            ratio = gamma / r["omega_p_macro"]
        else:
            ratio = float("nan")
        passed = "yes" if int(r.get("passed", 0)) == 1 else "no"
        ratio_cell = f"{ratio:.1f}" if pd.notna(ratio) else "---"
        lines.append(rf"{scheme} & {gamma:.3f} & {ratio_cell} & {passed} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--validation-csv", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
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

    gpu_df = load_gpu_data(results_dir)
    fig_roofline(bench_df, gpu_df, args.figures_dir / "fig10_roofline.pdf")
    fig_baseline_comparison(bench_df, args.figures_dir / "fig11_baseline_comparison.pdf")
    write_cross_platform_summary(bench_df, gpu_df, args.figures_dir / "cross_platform_summary.tex")

    if gpu_df is not None:
        fig_gpu_runtime(gpu_df, args.figures_dir / "fig8_gpu_runtime.pdf")
        fig_gpu_speedup(gpu_df, args.figures_dir / "fig9_gpu_speedup.pdf")
        write_gpu_summary(gpu_df, args.figures_dir / "gpu_summary.tex")
        write_gpu_priv_summary(gpu_df, args.figures_dir / "gpu_priv_summary.tex")
        print(f"Generated GPU figures from {results_dir / 'benchmark_gpu.csv'}")

    cons_path = results_dir / "conservation_study.csv"
    try:
        cons_path = resolve_csv(results_dir, "conservation_study_")
    except FileNotFoundError:
        pass
    fig_conservation_study(cons_path, args.figures_dir / "fig12_conservation_study.pdf")

    ts_path = results_dir / "timestep_profile.csv"
    try:
        ts_path = resolve_csv(results_dir, "timestep_profile_")
    except FileNotFoundError:
        pass
    fig_timestep_breakdown(ts_path, args.figures_dir / "fig13_timestep_breakdown.pdf")

    amort_path = results_dir / "amortized_timestep.csv"
    try:
        amort_path = resolve_csv(results_dir, "amortized_timestep_")
    except FileNotFoundError:
        pass
    fig_amortized_sort(amort_path, args.figures_dir / "fig15_amortized_sort.pdf")

    noise_path = results_dir / "noise_vs_grid.csv"
    try:
        noise_path = resolve_csv(results_dir, "noise_vs_grid_")
    except FileNotFoundError:
        pass
    fig_noise_vs_grid(noise_path, args.figures_dir / "fig16_noise_vs_grid.pdf")

    tw_path = results_dir / "two_stream_validation.csv"
    try:
        tw_path = resolve_csv(results_dir, "two_stream_validation_")
    except FileNotFoundError:
        pass
    fig_two_stream(tw_path, args.figures_dir / "fig14_two_stream.pdf")
    write_two_stream_summary(tw_path, args.figures_dir / "two_stream_summary.tex")

    pts_path = _load_optional_csv(results_dir, "physics_timeseries.csv", "physics_timeseries_")
    lc_path = _load_optional_csv(results_dir, "langmuir_convergence.csv", "langmuir_convergence_")
    q1d_path = _load_optional_csv(results_dir, "two_stream_quasi1d.csv", "two_stream_quasi1d_")
    ld_path = _load_optional_csv(results_dir, "landau_damping.csv", "landau_damping_")

    fig_langmuir_timeseries(pts_path, args.figures_dir / "fig17_langmuir_timeseries.pdf")
    fig_twostream_logE(pts_path, args.figures_dir / "fig18_twostream_logE.pdf")
    fig_energy_budget(pts_path, args.figures_dir / "fig19_energy_budget.pdf")
    fig_conservation_timeseries(pts_path, args.figures_dir / "fig20_conservation_timeseries.pdf")
    fig_langmuir_convergence(lc_path, args.figures_dir / "fig21_langmuir_convergence.pdf")
    fig_twostream_geometry(tw_path, q1d_path, args.figures_dir / "fig22_twostream_geometry.pdf")
    fig_landau_damping(ld_path, pts_path, args.figures_dir / "fig23_landau_damping.pdf")
    write_langmuir_convergence_summary(lc_path, args.figures_dir / "langmuir_convergence_summary.tex")
    write_twostream_geometry_summary(tw_path, q1d_path, args.figures_dir / "twostream_geometry_summary.tex")
    write_landau_summary(ld_path, args.figures_dir / "landau_summary.tex")

    print(f"Generated figures from {csv_path}" + (f" and {sim_path}" if sim_path else ""))


if __name__ == "__main__":
    main()
