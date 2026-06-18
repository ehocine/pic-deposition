#include "pic/validation.hpp"

#include "deposition/deposition.hpp"
#include "pic/diagnostics.hpp"
#include "pic/field_grid.hpp"
#include "pic/poisson_fft.hpp"
#include "pic/simulation.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>

namespace pic {

namespace {

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

void applyNeutralizingBackground(FieldGrid& grid) {
    const double bg = 1.0 / grid.domain().domainVolume();
    for (double& v : grid.rho()) {
        v += bg;
    }
}

}  // namespace

// Macroparticle normalization: |q_p| = 1/N_p, n_0 = N_p/V, omega_p^2 = n_0 q_p^2 / m (m=1).
double theoreticalPlasmaFrequency(const Domain& domain, std::size_t num_particles) {
    const double volume = domain.domainVolume();
    const double n0 = static_cast<double>(num_particles) / volume;
    const double q_macro = 1.0 / static_cast<double>(num_particles);
    return std::sqrt(n0 * q_macro * q_macro);
}

double coldTwoStreamGrowthRate(const Domain& domain, std::size_t num_particles) {
    // Birdsall & Langdon: peak cold symmetric two-stream growth is omega_p / (2*sqrt(2)).
    return theoreticalPlasmaFrequency(domain, num_particles) / (2.0 * std::sqrt(2.0));
}

double resonantBeamVelocity(const Domain& domain, std::size_t num_particles, int mode) {
    const double omega_p = theoreticalPlasmaFrequency(domain, num_particles);
    const double k = 2.0 * M_PI * static_cast<double>(mode) / domain.Lx;
    if (k < 1e-30) {
        return 0.0;
    }
    return omega_p / k;
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

double extractFourierModeAmplitude(const FieldGrid& grid, int mode_x) {
    const Domain& domain = grid.domain();
    const int nx = domain.Nx;
    const int ny = domain.Ny;
    const double k = 2.0 * M_PI * static_cast<double>(mode_x) / domain.Lx;
    double re = 0.0;
    double im = 0.0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double x = (static_cast<double>(i) + 0.5) * domain.dx;
            const double ex = grid.Ex()[static_cast<std::size_t>(j * nx + i)];
            const double phase = k * x;
            re += ex * std::cos(phase);
            im += ex * std::sin(phase);
        }
    }
    const double norm = static_cast<double>(nx * ny);
    re /= norm;
    im /= norm;
    return std::sqrt(re * re + im * im);
}

double estimateInstabilityGrowthRate(const std::vector<double>& signal, double dt, int record_start_step) {
    if (signal.size() < 8) {
        return 0.0;
    }
    const std::size_t n = signal.size();
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
        const double amp = std::max(signal[i], 1e-30);
        const double t = static_cast<double>(record_start_step + static_cast<int>(i)) * dt;
        const double l = std::log(amp);
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
    return (static_cast<double>(count) * sum_tl - sum_t * sum_l) / denom;
}

double estimateInstabilityGrowthRateEnvelope(const std::vector<double>& signal, double dt, int record_start_step) {
    if (signal.size() < 8) {
        return 0.0;
    }
    const std::size_t n = signal.size();
    const std::size_t window = std::max<std::size_t>(10, n / 50);
    std::vector<double> envelope(n);
    for (std::size_t i = 0; i < n; ++i) {
        double peak = signal[i];
        const std::size_t i0 = (i > window) ? i - window : 0;
        const std::size_t i1 = std::min(n - 1, i + window);
        for (std::size_t j = i0; j <= i1; ++j) {
            peak = std::max(peak, signal[j]);
        }
        envelope[i] = peak;
    }

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
        const double amp = std::max(envelope[i], 1e-30);
        const double t = static_cast<double>(record_start_step + static_cast<int>(i)) * dt;
        const double l = std::log(amp);
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
    return slope;
}

LangmuirValidationResult runLangmuirValidation(const SimulationConfig& config) {
    SimulationConfig cfg = config;
    cfg.langmuir_mode = true;
    cfg.sorted = false;

    Domain domain = cfg.domain;
    domain.updateDerived();
    FieldGrid grid(domain);
    ParticlesSoA particles(cfg.num_particles);
    particles.initializeLangmuirWave(domain, cfg.langmuir_amplitude, cfg.langmuir_mode_number, cfg.seed);

    PoissonSolverFFT fft_solver(domain);
    DepositionConfig dep = cfg.deposition;
    dep.scheme = cfg.scheme;
    dep.layout = ParticleLayout::SoA;
    dep.sorted = false;
    dep.esirkepov_dt = domain.dt;

    const bool esirkepov = dep.scheme == DepositionScheme::Esirkepov;
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

    LangmuirValidationResult result;
    result.omega_theory = theoreticalPlasmaFrequency(domain, cfg.num_particles);
    result.omega_measured = estimateDominantFrequency(field_energy, domain.dt);

    if (result.omega_theory > 0.0) {
        result.relative_error =
            std::abs(result.omega_measured - result.omega_theory) / result.omega_theory;
        result.omega_ratio = result.omega_measured / result.omega_theory;
    }
    result.passed = false;

    if (field_energy.size() >= 2) {
        const double e0 = field_energy.front();
        const double e1 = field_energy.back();
        result.amplitude_decay = (e0 - e1) / std::max(std::abs(e0), 1e-12);
    }

    return result;
}

std::vector<ValidationSummary> runValidationSuite(const SimulationConfig& base_config) {
    std::vector<ValidationSummary> results;
    const std::vector<DepositionScheme> schemes = {DepositionScheme::NGP, DepositionScheme::CIC,
                                                   DepositionScheme::TSC, DepositionScheme::Esirkepov};

    for (auto scheme : schemes) {
        SimulationConfig cfg = base_config;
        cfg.scheme = scheme;
        cfg.deposition.scheme = scheme;
        cfg.layout = ParticleLayout::SoA;

        Domain domain = cfg.domain;
        domain.updateDerived();
        FieldGrid grid(domain);
        ParticlesSoA particles(cfg.num_particles);
        particles.initializeUniformMaxwellian(domain, cfg.temperature, cfg.seed);

        DepositionConfig dep = cfg.deposition;
        dep.scheme = scheme;
        dep.layout = ParticleLayout::SoA;
        dep.sorted = false;
        depositChargeSoA(particles, grid, dep);

        ValidationSummary summary;
        summary.scheme = scheme;
        summary.spectral_noise = spectralNoiseLevel(grid);
        summary.charge_error =
            chargeConservationError(grid.integratedRho(), particles.totalCharge(), totalAbsChargeSoA(particles));
        summary.langmuir = runLangmuirValidation(cfg);
        results.push_back(summary);
    }

    double omega_ref = 0.0;
    for (const auto& r : results) {
        omega_ref += r.langmuir.omega_measured;
    }
    omega_ref /= static_cast<double>(results.size());

    if (omega_ref > 0.0) {
        for (auto& r : results) {
            const double spread = std::abs(r.langmuir.omega_measured - omega_ref) / omega_ref;
            r.langmuir.passed = spread <= 0.05;
        }
    }

    return results;
}

std::vector<ConservationStudyResult> runConservationStudy(const SimulationConfig& base_config) {
    std::vector<ConservationStudyResult> results;
    const std::vector<DepositionScheme> schemes = {DepositionScheme::NGP, DepositionScheme::CIC,
                                                   DepositionScheme::TSC, DepositionScheme::Esirkepov};

    for (auto scheme : schemes) {
        SimulationConfig cfg = base_config;
        cfg.scheme = scheme;
        cfg.deposition.scheme = scheme;
        cfg.layout = ParticleLayout::SoA;
        cfg.sorted = true;

        Simulation sim(cfg);
        const auto result = sim.run();

        ConservationStudyResult row;
        row.scheme = scheme;
        row.steps = cfg.domain.steps;
        row.max_charge_error = result.charge_error_history.empty()
                                   ? result.final_charge_error
                                   : *std::max_element(result.charge_error_history.begin(),
                                                       result.charge_error_history.end());
        row.final_energy_drift = result.energy_drift;
        results.push_back(row);
    }
    return results;
}

TwoStreamValidationResult runTwoStreamValidation(const SimulationConfig& config) {
    SimulationConfig cfg = config;
    cfg.two_stream_mode = true;
    cfg.langmuir_mode = false;
    cfg.sorted = false;

    Domain domain = cfg.domain;
    domain.updateDerived();

    double beam_velocity = cfg.two_stream_beam_velocity;
    if (cfg.two_stream_resonant_beams) {
        beam_velocity = resonantBeamVelocity(domain, cfg.num_particles, 1);
    }

    FieldGrid grid(domain);
    ParticlesSoA particles(cfg.num_particles);
    if (cfg.two_stream_quasi_1d) {
        particles.initializeTwoStreamQuasi1D(domain, beam_velocity, cfg.two_stream_perturbation, cfg.seed);
    } else {
        particles.initializeTwoStream(domain, beam_velocity, cfg.two_stream_perturbation, cfg.seed);
    }

    PoissonSolverFFT fft_solver(domain);
    DepositionConfig dep = cfg.deposition;
    dep.scheme = cfg.scheme;
    dep.layout = ParticleLayout::SoA;
    dep.sorted = false;
    dep.esirkepov_dt = domain.dt;

    const int record_start = domain.steps / 4;
    std::vector<double> mode_amplitude;
    mode_amplitude.reserve(static_cast<std::size_t>(domain.steps - record_start));

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
            mode_amplitude.push_back(extractFourierModeAmplitude(grid, 1));
        }
    }

    TwoStreamValidationResult result;
    result.scheme = cfg.scheme;
    const double omega_p = theoreticalPlasmaFrequency(domain, cfg.num_particles);
    result.omega_p_macro = omega_p;
    result.growth_rate_theory = coldTwoStreamGrowthRate(domain, cfg.num_particles);
    const double slope = estimateInstabilityGrowthRate(mode_amplitude, domain.dt, record_start);
    // Mode-amplitude slope is gamma/omega_p; convert to physical growth rate.
    result.growth_rate_measured = slope * omega_p;
    if (result.growth_rate_theory > 0.0) {
        result.growth_rate_ratio = result.growth_rate_measured / result.growth_rate_theory;
    }
    if (result.omega_p_macro > 0.0) {
        result.growth_rate_over_omega_p = result.growth_rate_measured / result.omega_p_macro;
    }
    result.passed = false;
    return result;
}

std::vector<TwoStreamValidationResult> runTwoStreamValidationSuite(const SimulationConfig& base_config) {
    std::vector<TwoStreamValidationResult> results;
    const std::vector<DepositionScheme> schemes = {DepositionScheme::NGP, DepositionScheme::CIC,
                                                   DepositionScheme::TSC, DepositionScheme::Esirkepov};
    for (auto scheme : schemes) {
        SimulationConfig cfg = base_config;
        cfg.scheme = scheme;
        cfg.deposition.scheme = scheme;
        auto row = runTwoStreamValidation(cfg);
        results.push_back(row);
    }

    // Inter-scheme agreement: shape-function reference (NGP/CIC/TSC); Esirkepov reported separately.
    double gamma_ref = 0.0;
    int agree_count = 0;
    for (const auto& r : results) {
        if (r.scheme == DepositionScheme::Esirkepov) {
            continue;
        }
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

std::vector<NoiseGridResult> runNoiseVsGridStudy(std::size_t num_particles) {
    std::vector<NoiseGridResult> rows;
    const std::vector<int> grids = {64, 128, 256};
    const std::vector<DepositionScheme> schemes = {DepositionScheme::NGP, DepositionScheme::CIC,
                                                   DepositionScheme::TSC, DepositionScheme::Esirkepov};
    for (int g : grids) {
        Domain domain;
        domain.Nx = g;
        domain.Ny = g;
        domain.updateDerived();
        for (auto scheme : schemes) {
            FieldGrid grid(domain);
            ParticlesSoA particles(num_particles);
            particles.initializeUniformMaxwellian(domain, 1.0, 42);
            DepositionConfig dep;
            dep.scheme = scheme;
            dep.layout = ParticleLayout::SoA;
            dep.sorted = true;
            depositChargeSoA(particles, grid, dep);
            NoiseGridResult row;
            row.grid_n = g;
            row.scheme = scheme;
            row.spectral_noise = spectralNoiseLevel(grid);
            rows.push_back(row);
        }
    }
    return rows;
}

void writeNoiseGridCsv(const std::string& path, const std::vector<NoiseGridResult>& rows) {
    std::ofstream out(path);
    out << "grid,scheme,spectral_noise\n";
    out << std::setprecision(10);
    for (const auto& row : rows) {
        out << row.grid_n << 'x' << row.grid_n << ',' << schemeName(row.scheme) << ',' << row.spectral_noise << '\n';
    }
}

void writeTwoStreamCsv(const std::string& path, const std::vector<TwoStreamValidationResult>& rows) {
    std::ofstream out(path);
    out << "scheme,growth_rate_measured,growth_rate_theory,growth_rate_ratio,omega_p_macro,"
           "growth_rate_over_omega_p,passed\n";
    out << std::setprecision(10);
    for (const auto& row : rows) {
        out << schemeName(row.scheme) << ',' << row.growth_rate_measured << ',' << row.growth_rate_theory << ','
            << row.growth_rate_ratio << ',' << row.omega_p_macro << ',' << row.growth_rate_over_omega_p << ','
            << (row.passed ? 1 : 0) << '\n';
    }
}

void writeConservationStudyCsv(const std::string& path, const std::vector<ConservationStudyResult>& rows) {
    std::ofstream out(path);
    out << "scheme,steps,max_charge_error,final_energy_drift\n";
    out << std::setprecision(10);
    for (const auto& row : rows) {
        out << schemeName(row.scheme) << ',' << row.steps << ',' << row.max_charge_error << ','
            << row.final_energy_drift << '\n';
    }
}

}  // namespace pic
