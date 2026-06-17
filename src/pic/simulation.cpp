#include "pic/simulation.hpp"

#include "pic/poisson_fft.hpp"
#include "sorting/cell_sort.hpp"

#include <chrono>

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
        } else if (config_.landau_mode) {
            particles_aos_.initializeWarmLangmuirWave(config_.domain, config_.langmuir_amplitude,
                                                      config_.langmuir_mode_number, config_.landau_temperature,
                                                      config_.seed);
        } else if (config_.two_stream_mode) {
            particles_aos_.initializeTwoStream(config_.domain, config_.two_stream_beam_velocity,
                                               config_.two_stream_perturbation, config_.seed);
        } else {
            particles_aos_.initializeUniformMaxwellian(config_.domain, config_.temperature, config_.seed);
        }
    } else {
        particles_soa_.resize(config_.num_particles);
        if (config_.langmuir_mode) {
            particles_soa_.initializeLangmuirWave(config_.domain, config_.langmuir_amplitude,
                                                  config_.langmuir_mode_number, config_.seed);
        } else if (config_.landau_mode) {
            particles_soa_.initializeWarmLangmuirWave(config_.domain, config_.langmuir_amplitude,
                                                      config_.langmuir_mode_number, config_.landau_temperature,
                                                      config_.seed);
        } else if (config_.two_stream_mode) {
            particles_soa_.initializeTwoStream(config_.domain, config_.two_stream_beam_velocity,
                                               config_.two_stream_perturbation, config_.seed);
        } else {
            particles_soa_.initializeUniformMaxwellian(config_.domain, config_.temperature, config_.seed);
        }
    }
}

namespace {

void applyBackgroundIfNeeded(const SimulationConfig& config, FieldGrid& grid) {
    if (config.langmuir_mode || config.landau_mode || config.two_stream_mode) {
        const double bg = 1.0 / config.domain.domainVolume();
        for (double& v : grid.rho()) {
            v += bg;
        }
    }
}

using Clock = std::chrono::steady_clock;

double elapsedMs(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

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
            } else {
                pushParticlesAoS(particles_aos_, grid_, config_.domain.dt);
                depositChargeAoS(particles_aos_, grid_, dep);
            }
            applyBackgroundIfNeeded(config_, grid_);
        } else {
            if (esirkepov) {
                depositChargeSoA(particles_soa_, grid_, dep);
            } else {
                pushParticlesSoA(particles_soa_, grid_, config_.domain.dt);
                depositChargeSoA(particles_soa_, grid_, dep);
            }
            applyBackgroundIfNeeded(config_, grid_);
        }

        if (config_.poisson == PoissonSolverType::FFT) {
            fft_solver.solve(grid_);
        } else {
            gs_solver.solve(grid_);
        }

        if (config_.layout == ParticleLayout::AoS) {
            if (esirkepov) {
                pushParticlesAoS(particles_aos_, grid_, config_.domain.dt);
                gatherFieldsAoS(particles_aos_, grid_, config_.domain.dt);
            } else {
                gatherFieldsAoS(particles_aos_, grid_, config_.domain.dt);
            }
            const auto diag = computeDiagnostics(grid_, particles_aos_);
            result.energy_history.push_back(diag.total_energy);
            result.kinetic_energy_history.push_back(diag.kinetic_energy);
            result.field_energy_history.push_back(diag.field_energy);
            result.charge_error_history.push_back(diag.charge_error);
        } else {
            if (esirkepov) {
                pushParticlesSoA(particles_soa_, grid_, config_.domain.dt);
                gatherFieldsSoA(particles_soa_, grid_, config_.domain.dt);
            } else {
                gatherFieldsSoA(particles_soa_, grid_, config_.domain.dt);
            }
            const auto diag = computeDiagnostics(grid_, particles_soa_);
            result.energy_history.push_back(diag.total_energy);
            result.kinetic_energy_history.push_back(diag.kinetic_energy);
            result.field_energy_history.push_back(diag.field_energy);
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

std::pair<SimulationResult, TimestepProfile> Simulation::runProfiled(int warmup_steps, int timed_steps) {
    SimulationResult result;
    TimestepProfile profile;
    PoissonSolverFFT fft_solver(config_.domain);
    PoissonSolverGS gs_solver(config_.domain);

    DepositionConfig dep = config_.deposition;
    dep.scheme = config_.scheme;
    dep.layout = config_.layout;
    dep.esirkepov_dt = config_.domain.dt;
    dep.sorted = false;

    const bool esirkepov = config_.scheme == DepositionScheme::Esirkepov;
    const int total_steps = warmup_steps + timed_steps;

    double acc_push = 0.0;
    double acc_sort = 0.0;
    double acc_deposit = 0.0;
    double acc_poisson = 0.0;
    double acc_gather = 0.0;

    for (int step = 0; step < total_steps; ++step) {
        const bool accumulate = step >= warmup_steps;
        double step_sort = 0.0;
        double step_push = 0.0;
        double step_deposit = 0.0;
        double step_poisson = 0.0;
        double step_gather = 0.0;

        if (config_.layout == ParticleLayout::AoS) {
            if (esirkepov) {
                auto td0 = Clock::now();
                depositChargeAoS(particles_aos_, grid_, dep);
                step_deposit = elapsedMs(td0, Clock::now());
            } else {
                auto tp0 = Clock::now();
                pushParticlesAoS(particles_aos_, grid_, config_.domain.dt);
                step_push = elapsedMs(tp0, Clock::now());
                auto td0 = Clock::now();
                depositChargeAoS(particles_aos_, grid_, dep);
                step_deposit = elapsedMs(td0, Clock::now());
            }
            applyBackgroundIfNeeded(config_, grid_);
        } else {
            ParticlesSoA deposit_particles = particles_soa_;
            const int interval = std::max(1, config_.sort_interval);
            if (config_.sorted && (step % interval == 0)) {
                auto ts = Clock::now();
                sortParticlesByCell(deposit_particles, config_.domain);
                step_sort = elapsedMs(ts, Clock::now());
            }
            if (esirkepov) {
                auto td0 = Clock::now();
                depositChargeSoA(deposit_particles, grid_, dep);
                step_deposit = elapsedMs(td0, Clock::now());
            } else {
                auto tp0 = Clock::now();
                pushParticlesSoA(particles_soa_, grid_, config_.domain.dt);
                step_push = elapsedMs(tp0, Clock::now());
                auto td0 = Clock::now();
                depositChargeSoA(deposit_particles, grid_, dep);
                step_deposit = elapsedMs(td0, Clock::now());
            }
            applyBackgroundIfNeeded(config_, grid_);
        }

        auto tp0 = Clock::now();
        if (config_.poisson == PoissonSolverType::FFT) {
            fft_solver.solve(grid_);
        } else {
            gs_solver.solve(grid_);
        }
        step_poisson = elapsedMs(tp0, Clock::now());

        if (config_.layout == ParticleLayout::AoS) {
            if (esirkepov) {
                auto tp1 = Clock::now();
                pushParticlesAoS(particles_aos_, grid_, config_.domain.dt);
                step_push = elapsedMs(tp1, Clock::now());
                auto tg0 = Clock::now();
                gatherFieldsAoS(particles_aos_, grid_, config_.domain.dt);
                step_gather = elapsedMs(tg0, Clock::now());
            } else {
                auto tg0 = Clock::now();
                gatherFieldsAoS(particles_aos_, grid_, config_.domain.dt);
                step_gather = elapsedMs(tg0, Clock::now());
            }
            const auto diag = computeDiagnostics(grid_, particles_aos_);
            if (step >= warmup_steps) {
                result.energy_history.push_back(diag.total_energy);
                result.kinetic_energy_history.push_back(diag.kinetic_energy);
                result.field_energy_history.push_back(diag.field_energy);
                result.charge_error_history.push_back(diag.charge_error);
            }
        } else {
            if (esirkepov) {
                auto tp1 = Clock::now();
                pushParticlesSoA(particles_soa_, grid_, config_.domain.dt);
                step_push = elapsedMs(tp1, Clock::now());
                auto tg0 = Clock::now();
                gatherFieldsSoA(particles_soa_, grid_, config_.domain.dt);
                step_gather = elapsedMs(tg0, Clock::now());
            } else {
                auto tg0 = Clock::now();
                gatherFieldsSoA(particles_soa_, grid_, config_.domain.dt);
                step_gather = elapsedMs(tg0, Clock::now());
            }
            const auto diag = computeDiagnostics(grid_, particles_soa_);
            if (step >= warmup_steps) {
                result.energy_history.push_back(diag.total_energy);
                result.kinetic_energy_history.push_back(diag.kinetic_energy);
                result.field_energy_history.push_back(diag.field_energy);
                result.charge_error_history.push_back(diag.charge_error);
            }
        }

        if (accumulate) {
            acc_push += step_push;
            acc_sort += step_sort;
            acc_deposit += step_deposit;
            acc_poisson += step_poisson;
            acc_gather += step_gather;
        }
    }

    const double n = static_cast<double>(timed_steps);
    profile.push_ms = acc_push / n;
    profile.sort_ms = acc_sort / n;
    profile.deposit_ms = acc_deposit / n;
    profile.poisson_ms = acc_poisson / n;
    profile.gather_ms = acc_gather / n;
    profile.total_ms = profile.push_ms + profile.sort_ms + profile.deposit_ms + profile.poisson_ms + profile.gather_ms;
    profile.sort_interval = std::max(1, config_.sort_interval);

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
    return {result, profile};
}

}  // namespace pic
