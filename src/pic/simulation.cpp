#include "pic/simulation.hpp"

#include "pic/poisson_fft.hpp"

namespace pic {

Simulation::Simulation(SimulationConfig config)
    : config_(std::move(config)), grid_(config_.domain) {
    config_.domain.updateDerived();
    config_.deposition.scheme = config_.scheme;
    config_.deposition.layout = config_.layout;
    config_.deposition.sorted = config_.sorted;

    if (config_.layout == ParticleLayout::AoS) {
        particles_aos_.resize(config_.num_particles);
        if (config_.langmuir_mode) {
            particles_aos_.initializeLangmuirWave(config_.domain, config_.langmuir_amplitude,
                                                  config_.langmuir_mode_number, config_.seed);
        } else {
            particles_aos_.initializeUniformMaxwellian(config_.domain, config_.temperature, config_.seed);
        }
    } else {
        particles_soa_.resize(config_.num_particles);
        if (config_.langmuir_mode) {
            particles_soa_.initializeLangmuirWave(config_.domain, config_.langmuir_amplitude,
                                                  config_.langmuir_mode_number, config_.seed);
        } else {
            particles_soa_.initializeUniformMaxwellian(config_.domain, config_.temperature, config_.seed);
        }
    }
}

SimulationResult Simulation::run() {
    SimulationResult result;
    PoissonSolverFFT fft_solver(config_.domain);
    PoissonSolverGS gs_solver(config_.domain);

    DepositionConfig dep = config_.deposition;
    dep.scheme = config_.scheme;
    dep.layout = config_.layout;
    dep.sorted = config_.sorted;
    dep.esirkepov_dt = config_.domain.dt;

    const bool esirkepov = config_.scheme == DepositionScheme::Esirkepov;

    for (int step = 0; step < config_.domain.steps; ++step) {
        if (config_.layout == ParticleLayout::AoS) {
            if (esirkepov) {
                depositChargeAoS(particles_aos_, grid_, dep);
                pushParticlesAoS(particles_aos_, grid_, config_.domain.dt);
            } else {
                pushParticlesAoS(particles_aos_, grid_, config_.domain.dt);
                depositChargeAoS(particles_aos_, grid_, dep);
            }
            if (config_.langmuir_mode) {
                const double bg = 1.0 / config_.domain.domainVolume();
                for (double& v : grid_.rho()) {
                    v += bg;
                }
            }
        } else {
            if (esirkepov) {
                depositChargeSoA(particles_soa_, grid_, dep);
                pushParticlesSoA(particles_soa_, grid_, config_.domain.dt);
            } else {
                pushParticlesSoA(particles_soa_, grid_, config_.domain.dt);
                depositChargeSoA(particles_soa_, grid_, dep);
            }
            if (config_.langmuir_mode) {
                const double bg = 1.0 / config_.domain.domainVolume();
                for (double& v : grid_.rho()) {
                    v += bg;
                }
            }
        }

        if (config_.poisson == PoissonSolverType::FFT) {
            fft_solver.solve(grid_);
        } else {
            gs_solver.solve(grid_);
        }

        if (config_.layout == ParticleLayout::AoS) {
            gatherFieldsAoS(particles_aos_, grid_, config_.domain.dt);
            const auto diag = computeDiagnostics(grid_, particles_aos_);
            result.energy_history.push_back(diag.total_energy);
            result.charge_error_history.push_back(diag.charge_error);
        } else {
            gatherFieldsSoA(particles_soa_, grid_, config_.domain.dt);
            const auto diag = computeDiagnostics(grid_, particles_soa_);
            result.energy_history.push_back(diag.total_energy);
            result.charge_error_history.push_back(diag.charge_error);
        }
    }

    if (!result.energy_history.empty()) {
        result.initial_energy = result.energy_history.front();
        result.final_energy = result.energy_history.back();
        result.energy_drift =
            (result.final_energy - result.initial_energy) / std::max(std::abs(result.initial_energy), 1e-12);
    }
    if (!result.charge_error_history.empty()) {
        result.final_charge_error = result.charge_error_history.back();
    }
    result.spectral_noise = spectralNoiseLevel(grid_);
    return result;
}

}  // namespace pic
