#pragma once

#include "deposition/deposition.hpp"
#include "pic/diagnostics.hpp"
#include "pic/domain.hpp"
#include "pic/field_grid.hpp"
#include "pic/particles.hpp"
#include "pic/poisson_fft.hpp"

#include <string>
#include <vector>

namespace pic {

enum class PoissonSolverType { FFT, GaussSeidel };

struct SimulationConfig {
    Domain domain;
    ParticleLayout layout = ParticleLayout::SoA;
    DepositionScheme scheme = DepositionScheme::CIC;
    DepositionConfig deposition;
    PoissonSolverType poisson = PoissonSolverType::FFT;
    std::size_t num_particles = 10000;
    unsigned seed = 42;
    double temperature = 1.0;
    bool sorted = false;
    int sort_interval = 1;
    bool langmuir_mode = false;
    double langmuir_amplitude = 0.01;
    int langmuir_mode_number = 1;
    bool landau_mode = false;
    double landau_temperature = 0.1;
    bool two_stream_mode = false;
    bool two_stream_resonant_beams = true;
    bool two_stream_quasi_1d = false;
    double two_stream_beam_velocity = 0.3;
    double two_stream_perturbation = 0.01;
};

struct TimestepProfile {
    double push_ms = 0.0;
    double sort_ms = 0.0;
    double deposit_ms = 0.0;
    double poisson_ms = 0.0;
    double gather_ms = 0.0;
    double total_ms = 0.0;
    int sort_interval = 1;
};

struct SimulationResult {
    std::vector<double> energy_history;
    std::vector<double> kinetic_energy_history;
    std::vector<double> field_energy_history;
    std::vector<double> charge_error_history;
    double initial_energy = 0.0;
    double final_energy = 0.0;
    double energy_drift = 0.0;
    double final_charge_error = 0.0;
    double spectral_noise = 0.0;
};

class Simulation {
public:
    explicit Simulation(SimulationConfig config);

    SimulationResult run();
    std::pair<SimulationResult, TimestepProfile> runProfiled(int warmup_steps = 3, int timed_steps = 20);

private:
    SimulationConfig config_;
    FieldGrid grid_;
    ParticlesAoS particles_aos_;
    ParticlesSoA particles_soa_;
};

}  // namespace pic
