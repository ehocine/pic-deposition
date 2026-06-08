#include "benchmark/benchmark_runner.hpp"

#include "pic/diagnostics.hpp"
#include "pic/simulation.hpp"
#include "pic/validation.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>

namespace pic {

BenchmarkRunner::BenchmarkRunner(std::vector<BenchmarkCase> cases) : cases_(std::move(cases)) {}

std::vector<BenchmarkCase> BenchmarkRunner::quickMatrix() {
    std::vector<BenchmarkCase> cases;
    const std::vector<std::size_t> particle_counts = {10000, 100000, 1000000};
    const std::vector<int> grids = {64, 128, 256};
    const std::vector<DepositionScheme> schemes = {DepositionScheme::NGP, DepositionScheme::CIC, DepositionScheme::TSC,
                                                   DepositionScheme::Esirkepov};
    const std::vector<ParticleLayout> layouts = {ParticleLayout::AoS, ParticleLayout::SoA};
    const std::vector<bool> sorted_opts = {false, true};
    const std::vector<int> thread_counts = {1, 2, 4, 8};

    for (std::size_t np : particle_counts) {
        for (int g : grids) {
            for (auto scheme : schemes) {
                for (auto layout : layouts) {
                    for (bool sorted : sorted_opts) {
                        BenchmarkCase base;
                        base.num_particles = np;
                        base.grid_n = g;
                        base.scheme = scheme;
                        base.layout = layout;
                        base.sorted = sorted;
                        base.backend = DepositionBackend::CPU;
                        base.num_threads = 1;
                        base.repeats = 5;
                        base.measure_spectral_noise = (np == 100000 && g == 128 && layout == ParticleLayout::SoA);
                        cases.push_back(base);

                        if (np == 100000 && g == 128 && layout == ParticleLayout::SoA && !sorted) {
                            for (int t : thread_counts) {
                                if (t == 1) continue;
                                BenchmarkCase threaded = base;
                                threaded.num_threads = t;
                                threaded.measure_spectral_noise = false;
                                cases.push_back(threaded);
                            }
                        }
                    }
                }
            }
        }
    }
    return cases;
}

std::vector<BenchmarkCase> BenchmarkRunner::simMatrix() {
    std::vector<BenchmarkCase> cases;
    const std::vector<DepositionScheme> schemes = {DepositionScheme::NGP, DepositionScheme::CIC, DepositionScheme::TSC,
                                                   DepositionScheme::Esirkepov};
    for (auto scheme : schemes) {
        BenchmarkCase c;
        c.num_particles = 100000;
        c.grid_n = 128;
        c.scheme = scheme;
        c.layout = ParticleLayout::SoA;
        c.sorted = true;
        c.num_threads = 1;
        c.repeats = 3;
        c.run_simulation = true;
        c.sim_steps = 500;
        cases.push_back(c);
    }
    return cases;
}

std::vector<BenchmarkCase> BenchmarkRunner::defaultMatrix(bool include_gpu) {
    auto cases = quickMatrix();
    if (include_gpu) {
#ifdef PIC_BUILD_CUDA
        (void)cases;
#endif
    }
    return cases;
}

std::vector<BenchmarkRow> BenchmarkRunner::runAll() const {
    std::vector<BenchmarkRow> rows;
    rows.reserve(cases_.size());

    double baseline_ms = 0.0;
    for (const auto& c : cases_) {
        Domain domain;
        domain.Nx = c.grid_n;
        domain.Ny = c.grid_n;
        domain.updateDerived();

        FieldGrid grid(domain);
        DepositionConfig dep;
        dep.scheme = c.scheme;
        dep.layout = c.layout;
        dep.sorted = c.sorted;
        dep.backend = c.backend;
        dep.num_threads = c.num_threads;
        dep.esirkepov_dt = domain.dt;

        DepositionStats stats;
        if (c.layout == ParticleLayout::AoS) {
            ParticlesAoS particles(c.num_particles);
            particles.initializeUniformMaxwellian(domain, 1.0, 42);
            stats = depositChargeTimedAoS(particles, grid, dep, c.repeats);
        } else {
            ParticlesSoA particles(c.num_particles);
            particles.initializeUniformMaxwellian(domain, 1.0, 42);
            stats = depositChargeTimedSoA(particles, grid, dep, c.repeats);
        }

        SimulationConfig sim_cfg;
        sim_cfg.domain = domain;
        sim_cfg.domain.steps = c.sim_steps;
        sim_cfg.num_particles = c.num_particles;
        sim_cfg.scheme = c.scheme;
        sim_cfg.layout = c.layout;
        sim_cfg.sorted = c.sorted;
        sim_cfg.deposition = dep;

        double full_ms = 0.0;
        double energy_drift = 0.0;
        double spectral_noise = 0.0;
        if (c.run_simulation) {
            const auto sim_start = std::chrono::steady_clock::now();
            Simulation sim(sim_cfg);
            const auto sim_result = sim.run();
            const auto sim_end = std::chrono::steady_clock::now();
            full_ms = std::chrono::duration<double, std::milli>(sim_end - sim_start).count() / sim_cfg.domain.steps;
            energy_drift = sim_result.energy_drift;
            spectral_noise = sim_result.spectral_noise;
        } else if (c.measure_spectral_noise) {
            ParticlesSoA particles(c.num_particles);
            particles.initializeUniformMaxwellian(domain, 1.0, 42);
            depositChargeSoA(particles, grid, dep);
            spectral_noise = spectralNoiseLevel(grid);
        }

        BenchmarkRow row;
        row.case_config = c;
        row.sort_ms = stats.sort_ms;
        row.deposit_ms = stats.deposit_ms;
        row.deposition_ms = stats.time_ms;
        row.full_step_ms = full_ms;
        row.throughput = stats.throughput_particles_per_s;
        row.bandwidth_gbs = stats.effective_bandwidth_gbs;
        row.charge_error = stats.charge_error;
        row.energy_drift = energy_drift;
        row.spectral_noise = spectral_noise;

        if (baseline_ms <= 0.0 && c.num_threads == 1 && c.backend == DepositionBackend::CPU && !c.sorted &&
            c.layout == ParticleLayout::SoA && c.scheme == DepositionScheme::CIC && c.grid_n == 64 &&
            c.num_particles == 10000) {
            baseline_ms = stats.deposit_ms;
        }
        row.speedup = baseline_ms > 0.0 ? baseline_ms / std::max(stats.deposit_ms, 1e-12) : 1.0;
        rows.push_back(row);
    }
    return rows;
}

void BenchmarkRunner::writeCsv(const std::string& path, const std::vector<BenchmarkRow>& rows) const {
    std::ofstream out(path);
    out << "num_particles,grid,scheme,layout,sorted,backend,threads,sort_ms,deposit_ms,deposition_ms,"
           "full_step_ms,throughput,bandwidth_gbs,charge_error,energy_drift,spectral_noise,speedup\n";
    out << std::setprecision(10);
    for (const auto& row : rows) {
        const auto& c = row.case_config;
        out << c.num_particles << ',' << c.grid_n << 'x' << c.grid_n << ',' << schemeName(c.scheme) << ','
            << layoutName(c.layout) << ',' << (c.sorted ? "on" : "off") << ',' << backendName(c.backend) << ','
            << c.num_threads << ',' << row.sort_ms << ',' << row.deposit_ms << ',' << row.deposition_ms << ','
            << row.full_step_ms << ',' << row.throughput << ',' << row.bandwidth_gbs << ',' << row.charge_error << ','
            << row.energy_drift << ',' << row.spectral_noise << ',' << row.speedup << '\n';
    }
}

void BenchmarkRunner::writeValidationCsv(const std::string& path, const std::vector<ValidationSummary>& rows) {
    std::ofstream out(path);
    out << "scheme,spectral_noise,charge_error,omega_theory,omega_measured,relative_error,omega_ratio,passed\n";
    out << std::setprecision(10);
    for (const auto& row : rows) {
        out << schemeName(row.scheme) << ',' << row.spectral_noise << ',' << row.charge_error << ','
            << row.langmuir.omega_theory << ',' << row.langmuir.omega_measured << ',' << row.langmuir.relative_error
            << ',' << row.langmuir.omega_ratio << ',' << (row.langmuir.passed ? 1 : 0) << '\n';
    }
}

}  // namespace pic
