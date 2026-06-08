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
    return 0;
}
