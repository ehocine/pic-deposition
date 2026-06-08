#include "pic/validation.hpp"

#include "deposition/deposition.hpp"
#include "pic/diagnostics.hpp"
#include "pic/poisson_fft.hpp"

#include <algorithm>
#include <cmath>
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

// Macroparticle normalization: |q_p| = 1/N_p, n_0 = N_p/V, omega_p^2 = n_0 q_p^2 / m (m=1).
double theoreticalPlasmaFrequency(const Domain& domain, std::size_t num_particles) {
    const double volume = domain.domainVolume();
    const double n0 = static_cast<double>(num_particles) / volume;
    const double q_macro = 1.0 / static_cast<double>(num_particles);
    return std::sqrt(n0 * q_macro * q_macro);
}

void applyNeutralizingBackground(FieldGrid& grid) {
    const double bg = 1.0 / grid.domain().domainVolume();
    for (double& v : grid.rho()) {
        v += bg;
    }
}

}  // namespace

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

}  // namespace pic
