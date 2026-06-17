#include "pic/physics_studies.hpp"

#include "deposition/deposition.hpp"
#include "pic/diagnostics.hpp"
#include "pic/poisson_fft.hpp"
#include "pic/validation.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace pic {

namespace {

const std::vector<DepositionScheme> kAllSchemes = {DepositionScheme::NGP, DepositionScheme::CIC,
                                                   DepositionScheme::TSC, DepositionScheme::Esirkepov};

double estimateDominantFrequency(const std::vector<double>& signal, double dt) {
    if (signal.size() < 8) {
        return 0.0;
    }
    double mean = 0.0;
    for (double v : signal) {
        mean += v;
    }
    mean /= static_cast<double>(signal.size());
    std::vector<double> centered(signal.size());
    for (std::size_t i = 0; i < signal.size(); ++i) {
        centered[i] = signal[i] - mean;
    }
    const int n = static_cast<int>(centered.size());
    double best_power = 0.0;
    int best_k = 1;
    for (int k = 1; k < n / 2; ++k) {
        double re = 0.0;
        double im = 0.0;
        for (int t = 0; t < n; ++t) {
            const double phase = 2.0 * M_PI * static_cast<double>(k * t) / static_cast<double>(n);
            re += centered[static_cast<std::size_t>(t)] * std::cos(phase);
            im -= centered[static_cast<std::size_t>(t)] * std::sin(phase);
        }
        const double power = re * re + im * im;
        if (power > best_power) {
            best_power = power;
            best_k = k;
        }
    }
    return static_cast<double>(best_k) / (static_cast<double>(n) * dt);
}

double estimateGrowthRate(const std::vector<double>& field_energy, double dt, int record_start_step) {
    if (field_energy.size() < 8) {
        return 0.0;
    }
    const std::size_t n = field_energy.size();
    const std::size_t start = n / 4;
    const std::size_t end = (3 * n) / 4;
    if (end <= start + 2) {
        return 0.0;
    }
    double sum_t = 0.0;
    double sum_l = 0.0;
    double sum_tt = 0.0;
    double sum_tl = 0.0;
    int count = 0;
    for (std::size_t i = start; i < end; ++i) {
        const double e = std::max(field_energy[i], 1e-30);
        const double t = static_cast<double>(record_start_step + static_cast<int>(i)) * dt;
        const double l = std::log(e);
        sum_t += t;
        sum_l += l;
        sum_tt += t * t;
        sum_tl += t * l;
        ++count;
    }
    const double denom = static_cast<double>(count) * sum_tt - sum_t * sum_t;
    if (std::abs(denom) < 1e-30) {
        return 0.0;
    }
    const double slope = (static_cast<double>(count) * sum_tl - sum_t * sum_l) / denom;
    return 0.5 * slope;
}

double estimateDampingRate(const std::vector<double>& field_energy, double dt, int record_start_step) {
    if (field_energy.size() < 8) {
        return 0.0;
    }
    const std::size_t n = field_energy.size();
    const std::size_t start = n / 4;
    const std::size_t end = (3 * n) / 4;
    if (end <= start + 2) {
        return 0.0;
    }
    double sum_t = 0.0;
    double sum_l = 0.0;
    double sum_tt = 0.0;
    double sum_tl = 0.0;
    int count = 0;
    for (std::size_t i = start; i < end; ++i) {
        const double e = std::max(field_energy[i], 1e-30);
        const double t = static_cast<double>(record_start_step + static_cast<int>(i)) * dt;
        const double l = std::log(e);
        sum_t += t;
        sum_l += l;
        sum_tt += t * t;
        sum_tl += t * l;
        ++count;
    }
    const double denom = static_cast<double>(count) * sum_tt - sum_t * sum_t;
    if (std::abs(denom) < 1e-30) {
        return 0.0;
    }
    const double slope = (static_cast<double>(count) * sum_tl - sum_t * sum_l) / denom;
    return -0.5 * slope;
}

double theoreticalPlasmaFrequency(const Domain& domain, std::size_t num_particles) {
    const double volume = domain.domainVolume();
    const double n0 = static_cast<double>(num_particles) / volume;
    const double q_macro = 1.0 / static_cast<double>(num_particles);
    return std::sqrt(n0 * q_macro * q_macro);
}

double coldTwoStreamGrowthRate(const Domain& domain, std::size_t num_particles) {
    return theoreticalPlasmaFrequency(domain, num_particles) / std::sqrt(2.0);
}

double landauDampingTheory(const Domain& domain, std::size_t num_particles, int mode, double temperature) {
    const double omega_p = theoreticalPlasmaFrequency(domain, num_particles);
    const double k = 2.0 * M_PI * static_cast<double>(mode) / domain.Lx;
    const double lambda_d = std::sqrt(temperature) / std::max(omega_p, 1e-30);
    const double kld = k * lambda_d;
    if (kld < 1e-6) {
        return 0.0;
    }
    const double kld2 = kld * kld;
    return omega_p * std::sqrt(M_PI / 8.0) * kld2 * std::exp(-1.0 / (2.0 * kld2));
}

void applyNeutralizingBackground(FieldGrid& grid) {
    const double bg = 1.0 / grid.domain().domainVolume();
    for (double& v : grid.rho()) {
        v += bg;
    }
}

std::vector<PhysicsTimeseriesRow> runLangmuirTimeseries(DepositionScheme scheme) {
    SimulationConfig cfg;
    cfg.domain.Nx = 64;
    cfg.domain.Ny = 64;
    cfg.domain.Lx = 1.0;
    cfg.domain.Ly = 1.0;
    cfg.domain.dt = 0.002;
    cfg.domain.steps = 16384;
    cfg.domain.updateDerived();
    cfg.num_particles = 200;
    cfg.scheme = scheme;
    cfg.langmuir_amplitude = 0.05;
    cfg.langmuir_mode_number = 1;
    cfg.layout = ParticleLayout::SoA;

    Domain domain = cfg.domain;
    FieldGrid grid(domain);
    ParticlesSoA particles(cfg.num_particles);
    particles.initializeLangmuirWave(domain, cfg.langmuir_amplitude, cfg.langmuir_mode_number, cfg.seed);

    PoissonSolverFFT fft_solver(domain);
    DepositionConfig dep = cfg.deposition;
    dep.scheme = scheme;
    dep.layout = ParticleLayout::SoA;
    dep.sorted = false;
    dep.esirkepov_dt = domain.dt;

    const bool esirkepov = scheme == DepositionScheme::Esirkepov;
    std::vector<PhysicsTimeseriesRow> rows;
    rows.reserve(static_cast<std::size_t>(domain.steps));

    for (int step = 0; step < domain.steps; ++step) {
        if (esirkepov) {
            depositChargeSoA(particles, grid, dep);
        } else {
            pushParticlesSoA(particles, grid, domain.dt);
            depositChargeSoA(particles, grid, dep);
        }
        applyNeutralizingBackground(grid);
        fft_solver.solve(grid);
        if (esirkepov) {
            pushParticlesSoA(particles, grid, domain.dt);
            gatherFieldsSoA(particles, grid, domain.dt);
        } else {
            gatherFieldsSoA(particles, grid, domain.dt);
        }
        const auto diag = computeDiagnostics(grid, particles);
        PhysicsTimeseriesRow row;
        row.test = "langmuir";
        row.scheme = scheme;
        row.step = step;
        row.time = static_cast<double>(step) * domain.dt;
        row.field_energy = diag.field_energy;
        row.kinetic_energy = diag.kinetic_energy;
        row.total_energy = diag.total_energy;
        row.charge_error = diag.charge_error;
        rows.push_back(row);
    }
    return rows;
}

std::vector<PhysicsTimeseriesRow> runTwoStreamTimeseries(DepositionScheme scheme) {
    SimulationConfig cfg;
    cfg.domain.Nx = 64;
    cfg.domain.Ny = 64;
    cfg.domain.Lx = 1.0;
    cfg.domain.Ly = 1.0;
    cfg.domain.dt = 0.002;
    cfg.domain.steps = 2000;
    cfg.domain.updateDerived();
    cfg.num_particles = 5000;
    cfg.scheme = scheme;
    cfg.two_stream_beam_velocity = 0.3;
    cfg.two_stream_perturbation = 0.01;
    cfg.layout = ParticleLayout::SoA;

    Domain domain = cfg.domain;
    FieldGrid grid(domain);
    ParticlesSoA particles(cfg.num_particles);
    particles.initializeTwoStream(domain, cfg.two_stream_beam_velocity, cfg.two_stream_perturbation, cfg.seed);

    PoissonSolverFFT fft_solver(domain);
    DepositionConfig dep = cfg.deposition;
    dep.scheme = scheme;
    dep.layout = ParticleLayout::SoA;
    dep.sorted = false;
    dep.esirkepov_dt = domain.dt;

    const bool esirkepov = scheme == DepositionScheme::Esirkepov;
    std::vector<PhysicsTimeseriesRow> rows;
    rows.reserve(static_cast<std::size_t>(domain.steps));

    for (int step = 0; step < domain.steps; ++step) {
        if (esirkepov) {
            depositChargeSoA(particles, grid, dep);
            pushParticlesSoA(particles, grid, domain.dt);
        } else {
            pushParticlesSoA(particles, grid, domain.dt);
            depositChargeSoA(particles, grid, dep);
        }
        applyNeutralizingBackground(grid);
        fft_solver.solve(grid);
        gatherFieldsSoA(particles, grid, domain.dt);
        const auto diag = computeDiagnostics(grid, particles);
        PhysicsTimeseriesRow row;
        row.test = "twostream";
        row.scheme = scheme;
        row.step = step;
        row.time = static_cast<double>(step) * domain.dt;
        row.field_energy = diag.field_energy;
        row.kinetic_energy = diag.kinetic_energy;
        row.total_energy = diag.total_energy;
        row.charge_error = diag.charge_error;
        rows.push_back(row);
    }
    return rows;
}

std::vector<PhysicsTimeseriesRow> runConservationTimeseries(DepositionScheme scheme) {
    SimulationConfig cfg;
    cfg.domain.Nx = 64;
    cfg.domain.Ny = 64;
    cfg.domain.Lx = 1.0;
    cfg.domain.Ly = 1.0;
    cfg.domain.dt = 0.005;
    cfg.domain.steps = 2000;
    cfg.domain.updateDerived();
    cfg.num_particles = 10000;
    cfg.scheme = scheme;
    cfg.layout = ParticleLayout::SoA;
    cfg.sorted = true;

    Simulation sim(cfg);
    const auto result = sim.run();

    std::vector<PhysicsTimeseriesRow> rows;
    const std::size_t n = result.energy_history.size();
    rows.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        PhysicsTimeseriesRow row;
        row.test = "conservation";
        row.scheme = scheme;
        row.step = static_cast<int>(i);
        row.time = static_cast<double>(i) * cfg.domain.dt;
        row.field_energy = i < result.field_energy_history.size() ? result.field_energy_history[i] : 0.0;
        row.kinetic_energy = i < result.kinetic_energy_history.size() ? result.kinetic_energy_history[i] : 0.0;
        row.total_energy = result.energy_history[i];
        row.charge_error = result.charge_error_history[i];
        rows.push_back(row);
    }
    return rows;
}

TwoStreamValidationResult runTwoStreamForDomain(const SimulationConfig& base, int nx, int ny) {
    SimulationConfig cfg = base;
    cfg.two_stream_mode = true;
    cfg.langmuir_mode = false;
    cfg.landau_mode = false;
    cfg.sorted = false;
    cfg.domain.Nx = nx;
    cfg.domain.Ny = ny;
    cfg.domain.updateDerived();

    Domain domain = cfg.domain;
    FieldGrid grid(domain);
    ParticlesSoA particles(cfg.num_particles);
    particles.initializeTwoStream(domain, cfg.two_stream_beam_velocity, cfg.two_stream_perturbation, cfg.seed);

    PoissonSolverFFT fft_solver(domain);
    DepositionConfig dep = cfg.deposition;
    dep.scheme = cfg.scheme;
    dep.layout = ParticleLayout::SoA;
    dep.sorted = false;
    dep.esirkepov_dt = domain.dt;

    const int record_start = domain.steps / 4;
    std::vector<double> field_energy;
    field_energy.reserve(static_cast<std::size_t>(domain.steps - record_start));

    const bool esirkepov = dep.scheme == DepositionScheme::Esirkepov;
    for (int step = 0; step < domain.steps; ++step) {
        if (esirkepov) {
            depositChargeSoA(particles, grid, dep);
            pushParticlesSoA(particles, grid, domain.dt);
        } else {
            pushParticlesSoA(particles, grid, domain.dt);
            depositChargeSoA(particles, grid, dep);
        }
        applyNeutralizingBackground(grid);
        fft_solver.solve(grid);
        gatherFieldsSoA(particles, grid, domain.dt);
        if (step >= record_start) {
            field_energy.push_back(grid.fieldEnergy());
        }
    }

    TwoStreamValidationResult result;
    result.scheme = cfg.scheme;
    const double omega_p = theoreticalPlasmaFrequency(domain, cfg.num_particles);
    result.omega_p_macro = omega_p;
    result.growth_rate_theory = coldTwoStreamGrowthRate(domain, cfg.num_particles);
    result.growth_rate_measured = estimateGrowthRate(field_energy, domain.dt, record_start);
    if (result.growth_rate_theory > 0.0) {
        result.growth_rate_ratio = result.growth_rate_measured / result.growth_rate_theory;
    }
    if (result.omega_p_macro > 0.0) {
        result.growth_rate_over_omega_p = result.growth_rate_measured / result.omega_p_macro;
    }
    result.passed = result.growth_rate_measured > 0.0;
    return result;
}

std::vector<PhysicsTimeseriesRow> runLandauTimeseries(DepositionScheme scheme) {
    const double temperature = 0.1;
    const int mode = 1;
    const double amplitude = 0.05;

    SimulationConfig cfg;
    cfg.domain.Nx = 64;
    cfg.domain.Ny = 64;
    cfg.domain.Lx = 1.0;
    cfg.domain.Ly = 1.0;
    cfg.domain.dt = 0.002;
    cfg.domain.steps = 8192;
    cfg.domain.updateDerived();
    cfg.num_particles = 2000;
    cfg.scheme = scheme;

    Domain domain = cfg.domain;
    FieldGrid grid(domain);
    ParticlesSoA particles(cfg.num_particles);
    particles.initializeWarmLangmuirWave(domain, amplitude, mode, temperature, cfg.seed);

    PoissonSolverFFT fft_solver(domain);
    DepositionConfig dep;
    dep.scheme = scheme;
    dep.layout = ParticleLayout::SoA;
    dep.sorted = false;
    dep.esirkepov_dt = domain.dt;

    const bool esirkepov = scheme == DepositionScheme::Esirkepov;
    std::vector<PhysicsTimeseriesRow> rows;
    rows.reserve(static_cast<std::size_t>(domain.steps));

    for (int step = 0; step < domain.steps; ++step) {
        if (esirkepov) {
            depositChargeSoA(particles, grid, dep);
        } else {
            pushParticlesSoA(particles, grid, domain.dt);
            depositChargeSoA(particles, grid, dep);
        }
        applyNeutralizingBackground(grid);
        fft_solver.solve(grid);
        if (esirkepov) {
            pushParticlesSoA(particles, grid, domain.dt);
            gatherFieldsSoA(particles, grid, domain.dt);
        } else {
            gatherFieldsSoA(particles, grid, domain.dt);
        }
        const auto diag = computeDiagnostics(grid, particles);
        PhysicsTimeseriesRow row;
        row.test = "landau";
        row.scheme = scheme;
        row.step = step;
        row.time = static_cast<double>(step) * domain.dt;
        row.field_energy = diag.field_energy;
        row.kinetic_energy = diag.kinetic_energy;
        row.total_energy = diag.total_energy;
        row.charge_error = diag.charge_error;
        rows.push_back(row);
    }
    return rows;
}

}  // namespace

std::vector<PhysicsTimeseriesRow> runPhysicsTimeseries() {
    std::vector<PhysicsTimeseriesRow> rows;
    for (auto scheme : kAllSchemes) {
        auto lang = runLangmuirTimeseries(scheme);
        rows.insert(rows.end(), lang.begin(), lang.end());
        auto ts = runTwoStreamTimeseries(scheme);
        rows.insert(rows.end(), ts.begin(), ts.end());
        auto cons = runConservationTimeseries(scheme);
        rows.insert(rows.end(), cons.begin(), cons.end());
        if (scheme == DepositionScheme::CIC) {
            auto landau = runLandauTimeseries(scheme);
            rows.insert(rows.end(), landau.begin(), landau.end());
        }
    }
    return rows;
}

std::vector<TwoStreamValidationResult> runQuasi1DTwoStreamValidation() {
    SimulationConfig cfg;
    cfg.domain.Lx = 1.0;
    cfg.domain.Ly = 1.0;
    cfg.domain.dt = 0.002;
    cfg.domain.steps = 2000;
    cfg.num_particles = 5000;
    cfg.two_stream_beam_velocity = 0.3;
    cfg.two_stream_perturbation = 0.01;

    std::vector<TwoStreamValidationResult> results;
    for (auto scheme : kAllSchemes) {
        cfg.scheme = scheme;
        cfg.deposition.scheme = scheme;
        results.push_back(runTwoStreamForDomain(cfg, 256, 4));
    }

    double gamma_ref = 0.0;
    int agree_count = 0;
    for (const auto& r : results) {
        if (r.growth_rate_measured > 0.0) {
            gamma_ref += r.growth_rate_measured;
            ++agree_count;
        }
    }
    if (agree_count > 0) {
        gamma_ref /= static_cast<double>(agree_count);
        for (auto& r : results) {
            if (r.growth_rate_measured <= 0.0) {
                r.passed = false;
                continue;
            }
            const double spread = std::abs(r.growth_rate_measured - gamma_ref) / gamma_ref;
            r.passed = spread <= 0.10;
        }
    }
    return results;
}

std::vector<LangmuirConvergenceRow> runLangmuirConvergenceSweep() {
    std::vector<LangmuirConvergenceRow> rows;
    const std::vector<int> grids = {64, 128, 256};
    const std::vector<std::size_t> particle_counts = {200, 2000, 20000};

    for (int g : grids) {
        for (std::size_t np : particle_counts) {
            for (auto scheme : kAllSchemes) {
                SimulationConfig cfg;
                cfg.domain.Nx = g;
                cfg.domain.Ny = g;
                cfg.domain.Lx = 1.0;
                cfg.domain.Ly = 1.0;
                cfg.domain.dt = 0.002;
                cfg.domain.steps = 8192;
                cfg.domain.updateDerived();
                cfg.num_particles = np;
                cfg.scheme = scheme;
                cfg.deposition.scheme = scheme;
                cfg.langmuir_amplitude = 0.05;
                cfg.langmuir_mode_number = 1;

                const auto lang = runLangmuirValidation(cfg);
                LangmuirConvergenceRow row;
                row.grid_n = g;
                row.num_particles = np;
                row.scheme = scheme;
                row.omega_measured = lang.omega_measured;
                row.omega_theory = lang.omega_theory;
                row.omega_ratio = lang.omega_ratio;
                rows.push_back(row);
            }
        }
    }
    return rows;
}

std::vector<LandauDampingRow> runLandauDampingValidation() {
    std::vector<LandauDampingRow> rows;
    const double temperature = 0.1;
    const int mode = 1;
    const double amplitude = 0.05;

    for (auto scheme : kAllSchemes) {
        SimulationConfig cfg;
        cfg.domain.Nx = 64;
        cfg.domain.Ny = 64;
        cfg.domain.Lx = 1.0;
        cfg.domain.Ly = 1.0;
        cfg.domain.dt = 0.002;
        cfg.domain.steps = 8192;
        cfg.domain.updateDerived();
        cfg.num_particles = 2000;
        cfg.scheme = scheme;
        cfg.landau_mode = true;
        cfg.landau_temperature = temperature;
        cfg.langmuir_amplitude = amplitude;
        cfg.langmuir_mode_number = mode;
        cfg.layout = ParticleLayout::SoA;

        Domain domain = cfg.domain;
        FieldGrid grid(domain);
        ParticlesSoA particles(cfg.num_particles);
        particles.initializeWarmLangmuirWave(domain, amplitude, mode, temperature, cfg.seed);

        PoissonSolverFFT fft_solver(domain);
        DepositionConfig dep = cfg.deposition;
        dep.scheme = scheme;
        dep.layout = ParticleLayout::SoA;
        dep.sorted = false;
        dep.esirkepov_dt = domain.dt;

        const bool esirkepov = scheme == DepositionScheme::Esirkepov;
        const int record_start = domain.steps / 4;
        std::vector<double> field_energy;
        field_energy.reserve(static_cast<std::size_t>(domain.steps - record_start));

        for (int step = 0; step < domain.steps; ++step) {
            if (esirkepov) {
                depositChargeSoA(particles, grid, dep);
            } else {
                pushParticlesSoA(particles, grid, domain.dt);
                depositChargeSoA(particles, grid, dep);
            }
            applyNeutralizingBackground(grid);
            fft_solver.solve(grid);
            if (esirkepov) {
                pushParticlesSoA(particles, grid, domain.dt);
                gatherFieldsSoA(particles, grid, domain.dt);
            } else {
                gatherFieldsSoA(particles, grid, domain.dt);
            }
            if (step >= record_start) {
                field_energy.push_back(grid.fieldEnergy());
            }
        }

        const double omega_p = theoreticalPlasmaFrequency(domain, cfg.num_particles);
        const double k = 2.0 * M_PI * static_cast<double>(mode) / domain.Lx;
        const double kld = k * std::sqrt(temperature) / std::max(omega_p, 1e-30);
        const double gamma_theory = landauDampingTheory(domain, cfg.num_particles, mode, temperature);
        const double gamma_meas = estimateDampingRate(field_energy, domain.dt, record_start);

        LandauDampingRow row;
        row.scheme = scheme;
        row.damping_rate_measured = gamma_meas;
        row.damping_rate_theory = gamma_theory;
        row.k_lambda_d = kld;
        if (gamma_theory > 0.0) {
            row.damping_ratio = gamma_meas / gamma_theory;
        }
        rows.push_back(row);
    }
    return rows;
}

std::vector<MultiSeedRow> runMultiSeedValidation() {
    std::vector<MultiSeedRow> rows;
    const std::vector<unsigned> seeds = {42, 43, 44, 45, 46};

    for (unsigned seed : seeds) {
        SimulationConfig lang_cfg;
        lang_cfg.domain.Nx = 64;
        lang_cfg.domain.Ny = 64;
        lang_cfg.domain.Lx = 1.0;
        lang_cfg.domain.Ly = 1.0;
        lang_cfg.domain.dt = 0.002;
        lang_cfg.domain.steps = 16384;
        lang_cfg.domain.updateDerived();
        lang_cfg.num_particles = 200;
        lang_cfg.langmuir_amplitude = 0.05;
        lang_cfg.langmuir_mode_number = 1;
        lang_cfg.seed = seed;

        std::vector<double> omegas;
        for (auto scheme : kAllSchemes) {
            lang_cfg.scheme = scheme;
            lang_cfg.deposition.scheme = scheme;
            const auto lang = runLangmuirValidation(lang_cfg);
            MultiSeedRow row;
            row.seed = seed;
            row.test = "langmuir";
            row.scheme = scheme;
            row.metric = "omega_measured";
            row.value = lang.omega_measured;
            row.passed = false;
            rows.push_back(row);
            omegas.push_back(lang.omega_measured);
        }
        double omega_ref = 0.0;
        for (double o : omegas) {
            omega_ref += o;
        }
        omega_ref /= static_cast<double>(omegas.size());
        for (auto& row : rows) {
            if (row.seed == seed && row.test == "langmuir" && omega_ref > 0.0) {
                const double spread = std::abs(row.value - omega_ref) / omega_ref;
                row.passed = spread <= 0.05;
            }
        }

        SimulationConfig ts_cfg;
        ts_cfg.domain.Nx = 64;
        ts_cfg.domain.Ny = 64;
        ts_cfg.domain.Lx = 1.0;
        ts_cfg.domain.Ly = 1.0;
        ts_cfg.domain.dt = 0.002;
        ts_cfg.domain.steps = 2000;
        ts_cfg.domain.updateDerived();
        ts_cfg.num_particles = 5000;
        ts_cfg.two_stream_beam_velocity = 0.3;
        ts_cfg.two_stream_perturbation = 0.01;
        ts_cfg.seed = seed;

        std::vector<double> gammas;
        for (auto scheme : kAllSchemes) {
            ts_cfg.scheme = scheme;
            ts_cfg.deposition.scheme = scheme;
            const auto ts = runTwoStreamValidation(ts_cfg);
            MultiSeedRow row;
            row.seed = seed;
            row.test = "twostream";
            row.scheme = scheme;
            row.metric = "growth_rate_measured";
            row.value = ts.growth_rate_measured;
            row.passed = ts.passed;
            rows.push_back(row);
            if (ts.growth_rate_measured > 0.0) {
                gammas.push_back(ts.growth_rate_measured);
            }
        }
        if (!gammas.empty()) {
            double gamma_ref = 0.0;
            for (double g : gammas) {
                gamma_ref += g;
            }
            gamma_ref /= static_cast<double>(gammas.size());
            for (auto& row : rows) {
                if (row.seed == seed && row.test == "twostream" && row.value > 0.0 && gamma_ref > 0.0) {
                    const double spread = std::abs(row.value - gamma_ref) / gamma_ref;
                    row.passed = spread <= 0.10;
                }
            }
        }
    }
    return rows;
}

std::vector<ProductionSimRow> runProductionSimulation() {
    SimulationConfig cfg;
    cfg.domain.Nx = 128;
    cfg.domain.Ny = 128;
    cfg.domain.Lx = 1.0;
    cfg.domain.Ly = 1.0;
    cfg.domain.dt = 0.01;
    cfg.domain.steps = 500;
    cfg.domain.updateDerived();
    cfg.num_particles = 100000;
    cfg.scheme = DepositionScheme::CIC;
    cfg.layout = ParticleLayout::SoA;
    cfg.sorted = true;
    cfg.sort_interval = 10;

    Simulation sim(cfg);
    const auto [result, profile] = sim.runProfiled(3, 500);

    ProductionSimRow row;
    row.num_particles = cfg.num_particles;
    row.grid_n = 128;
    row.scheme = DepositionScheme::CIC;
    row.sort_interval = cfg.sort_interval;
    row.steps = 500;
    row.push_ms = profile.push_ms;
    row.sort_ms = profile.sort_ms;
    row.deposit_ms = profile.deposit_ms;
    row.poisson_ms = profile.poisson_ms;
    row.gather_ms = profile.gather_ms;
    row.total_ms = profile.total_ms;
    row.energy_drift = result.energy_drift;
    row.max_charge_error = result.charge_error_history.empty()
                               ? result.final_charge_error
                               : *std::max_element(result.charge_error_history.begin(),
                                                   result.charge_error_history.end());
    return {row};
}

void writePhysicsTimeseriesCsv(const std::string& path, const std::vector<PhysicsTimeseriesRow>& rows) {
    std::ofstream out(path);
    out << "test,scheme,step,time,field_energy,kinetic_energy,total_energy,charge_error\n";
    out << std::setprecision(10);
    for (const auto& row : rows) {
        out << row.test << ',' << schemeName(row.scheme) << ',' << row.step << ',' << row.time << ','
            << row.field_energy << ',' << row.kinetic_energy << ',' << row.total_energy << ',' << row.charge_error
            << '\n';
    }
}

void writeLangmuirConvergenceCsv(const std::string& path, const std::vector<LangmuirConvergenceRow>& rows) {
    std::ofstream out(path);
    out << "grid_n,num_particles,scheme,omega_measured,omega_theory,omega_ratio\n";
    out << std::setprecision(10);
    for (const auto& row : rows) {
        out << row.grid_n << ',' << row.num_particles << ',' << schemeName(row.scheme) << ',' << row.omega_measured
            << ',' << row.omega_theory << ',' << row.omega_ratio << '\n';
    }
}

void writeLandauDampingCsv(const std::string& path, const std::vector<LandauDampingRow>& rows) {
    std::ofstream out(path);
    out << "scheme,damping_rate_measured,damping_rate_theory,damping_ratio,k_lambda_d\n";
    out << std::setprecision(10);
    for (const auto& row : rows) {
        out << schemeName(row.scheme) << ',' << row.damping_rate_measured << ',' << row.damping_rate_theory << ','
            << row.damping_ratio << ',' << row.k_lambda_d << '\n';
    }
}

void writeMultiSeedCsv(const std::string& path, const std::vector<MultiSeedRow>& rows) {
    std::ofstream out(path);
    out << "seed,test,scheme,metric,value,passed\n";
    out << std::setprecision(10);
    for (const auto& row : rows) {
        out << row.seed << ',' << row.test << ',' << schemeName(row.scheme) << ',' << row.metric << ',' << row.value
            << ',' << (row.passed ? 1 : 0) << '\n';
    }
}

void writeProductionSimCsv(const std::string& path, const std::vector<ProductionSimRow>& rows) {
    std::ofstream out(path);
    out << "num_particles,grid,scheme,sort_interval,steps,push_ms,sort_ms,deposit_ms,poisson_ms,gather_ms,total_ms,"
           "energy_drift,max_charge_error\n";
    out << std::setprecision(10);
    for (const auto& row : rows) {
        out << row.num_particles << ',' << row.grid_n << 'x' << row.grid_n << ',' << schemeName(row.scheme) << ','
            << row.sort_interval << ',' << row.steps << ',' << row.push_ms << ',' << row.sort_ms << ','
            << row.deposit_ms << ',' << row.poisson_ms << ',' << row.gather_ms << ',' << row.total_ms << ','
            << row.energy_drift << ',' << row.max_charge_error << '\n';
    }
}

void writeQuasi1DTwoStreamCsv(const std::string& path, const std::vector<TwoStreamValidationResult>& rows) {
    std::ofstream out(path);
    out << "geometry,scheme,growth_rate_measured,growth_rate_theory,growth_rate_ratio,omega_p_macro,"
           "growth_rate_over_omega_p,passed\n";
    out << std::setprecision(10);
    for (const auto& row : rows) {
        out << "quasi1d," << schemeName(row.scheme) << ',' << row.growth_rate_measured << ','
            << row.growth_rate_theory << ',' << row.growth_rate_ratio << ',' << row.omega_p_macro << ','
            << row.growth_rate_over_omega_p << ',' << (row.passed ? 1 : 0) << '\n';
    }
}

void runAllPhysicsStudies(const std::string& results_dir) {
    const std::string stamp = std::to_string(static_cast<long long>(std::time(nullptr)));

    const auto timeseries = runPhysicsTimeseries();
    writePhysicsTimeseriesCsv(results_dir + "/physics_timeseries_" + stamp + ".csv", timeseries);

    const auto quasi1d = runQuasi1DTwoStreamValidation();
    writeQuasi1DTwoStreamCsv(results_dir + "/two_stream_quasi1d_" + stamp + ".csv", quasi1d);

    const auto lang_conv = runLangmuirConvergenceSweep();
    writeLangmuirConvergenceCsv(results_dir + "/langmuir_convergence_" + stamp + ".csv", lang_conv);

    const auto landau = runLandauDampingValidation();
    writeLandauDampingCsv(results_dir + "/landau_damping_" + stamp + ".csv", landau);

    const auto multi_seed = runMultiSeedValidation();
    writeMultiSeedCsv(results_dir + "/validation_multi_seed_" + stamp + ".csv", multi_seed);

    const auto production = runProductionSimulation();
    writeProductionSimCsv(results_dir + "/production_sim_" + stamp + ".csv", production);
}

}  // namespace pic
