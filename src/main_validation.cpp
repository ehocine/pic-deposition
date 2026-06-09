#include "pic/validation.hpp"

#include <iostream>

int main() {
    pic::SimulationConfig cfg;
    cfg.domain.Nx = 64;
    cfg.domain.Ny = 64;
    cfg.domain.Lx = 1.0;
    cfg.domain.Ly = 1.0;
    cfg.domain.dt = 0.002;
    cfg.domain.steps = 16384;
    cfg.domain.updateDerived();
    cfg.num_particles = 200;
    cfg.layout = pic::ParticleLayout::SoA;
    cfg.langmuir_amplitude = 0.05;
    cfg.langmuir_mode_number = 1;

    const auto summaries = pic::runValidationSuite(cfg);
    for (const auto& s : summaries) {
        std::cout << pic::schemeName(s.scheme) << ": spectral_noise=" << s.spectral_noise
                  << " charge_error=" << s.charge_error << " omega_theory=" << s.langmuir.omega_theory
                  << " omega_measured=" << s.langmuir.omega_measured
                  << " rel_error=" << s.langmuir.relative_error
                  << " omega_ratio=" << s.langmuir.omega_ratio << " passed=" << s.langmuir.passed << '\n';
    }

    pic::SimulationConfig cons_cfg;
    cons_cfg.domain.Nx = 64;
    cons_cfg.domain.Ny = 64;
    cons_cfg.domain.Lx = 1.0;
    cons_cfg.domain.Ly = 1.0;
    cons_cfg.domain.dt = 0.005;
    cons_cfg.domain.steps = 2000;
    cons_cfg.domain.updateDerived();
    cons_cfg.num_particles = 10000;
    cons_cfg.layout = pic::ParticleLayout::SoA;
    cons_cfg.sorted = true;

    const auto conservation = pic::runConservationStudy(cons_cfg);
    for (const auto& c : conservation) {
        std::cout << pic::schemeName(c.scheme) << ": max_charge_error=" << c.max_charge_error
                  << " energy_drift=" << c.final_energy_drift << " steps=" << c.steps << '\n';
    }

    pic::SimulationConfig ts_cfg;
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

    const auto two_stream = pic::runTwoStreamValidationSuite(ts_cfg);
    for (const auto& t : two_stream) {
        std::cout << pic::schemeName(t.scheme) << ": gamma_meas=" << t.growth_rate_measured
                  << " gamma_theory=" << t.growth_rate_theory << " passed=" << t.passed << '\n';
    }
    return 0;
}
