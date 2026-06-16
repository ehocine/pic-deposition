#include "deposition/deposition.hpp"
#include "pic/diagnostics.hpp"
#include "pic/domain.hpp"
#include "pic/field_grid.hpp"
#include "pic/particles.hpp"

#include <cmath>
#include <iostream>
#include <numeric>
#include <string>

namespace {

bool expectNear(double value, double expected, double tol, const std::string& name) {
    if (std::abs(value - expected) > tol) {
        std::cerr << "FAIL " << name << ": got " << value << " expected " << expected << "\n";
        return false;
    }
    std::cout << "PASS " << name << "\n";
    return true;
}

bool testSchemeChargeConservation(pic::DepositionScheme scheme) {
    pic::Domain domain;
    domain.Nx = 32;
    domain.Ny = 32;
    domain.updateDerived();
    pic::FieldGrid grid(domain);
    pic::ParticlesSoA particles(1000);
    particles.initializeUniformMaxwellian(domain, 1.0, 123);

    pic::DepositionConfig cfg;
    cfg.scheme = scheme;
    cfg.layout = pic::ParticleLayout::SoA;
    cfg.esirkepov_dt = 0.0;
    pic::depositChargeSoA(particles, grid, cfg);

    const double charge_error =
        pic::chargeConservationError(grid.integratedRho(), particles.totalCharge(),
                                     std::accumulate(particles.q().begin(), particles.q().end(), 0.0,
                                                     [](double s, double q) { return s + std::abs(q); }));
    return expectNear(charge_error, 0.0, scheme == pic::DepositionScheme::NGP ? 0.05 : 1e-10,
                      pic::schemeName(scheme) + " charge conservation");
}

// Regression test: Esirkepov must conserve charge with moving particles
// (non-zero dt). The static dt=0 path masked a bug where trajectory weights
// summed to displacement/dx instead of unity, breaking charge conservation
// whenever particles moved.
bool testEsirkepovMovingChargeConservation() {
    pic::Domain domain;
    domain.Nx = 32;
    domain.Ny = 32;
    domain.updateDerived();
    pic::FieldGrid grid(domain);
    pic::ParticlesSoA particles(1000);
    particles.initializeUniformMaxwellian(domain, 1.0, 123);

    pic::DepositionConfig cfg;
    cfg.scheme = pic::DepositionScheme::Esirkepov;
    cfg.layout = pic::ParticleLayout::SoA;
    cfg.esirkepov_dt = 0.01;
    pic::depositChargeSoA(particles, grid, cfg);

    const double charge_error =
        pic::chargeConservationError(grid.integratedRho(), particles.totalCharge(),
                                     std::accumulate(particles.q().begin(), particles.q().end(), 0.0,
                                                     [](double s, double q) { return s + std::abs(q); }));
    return expectNear(charge_error, 0.0, 1e-10, "Esirkepov charge conservation (moving, dt=0.01)");
}

}  // namespace

#ifdef PIC_BUILD_CUDA
bool testGpuDeposit(pic::DepositionBackend backend, const std::string& name) {
    pic::Domain domain;
    domain.Nx = 128;
    domain.Ny = 128;
    domain.updateDerived();
    pic::FieldGrid grid(domain);
    pic::ParticlesSoA particles(5000);
    particles.initializeUniformMaxwellian(domain, 1.0, 123);

    pic::DepositionConfig cfg;
    cfg.scheme = pic::DepositionScheme::CIC;
    cfg.layout = pic::ParticleLayout::SoA;
    cfg.backend = backend;
    pic::depositChargeSoA(particles, grid, cfg);

    const double charge_error =
        pic::chargeConservationError(grid.integratedRho(), particles.totalCharge(),
                                     std::accumulate(particles.q().begin(), particles.q().end(), 0.0,
                                                     [](double s, double q) { return s + std::abs(q); }));
    return expectNear(charge_error, 0.0, 1e-9, name);
}
#endif

int main() {
    bool ok = true;
    ok &= testSchemeChargeConservation(pic::DepositionScheme::CIC);
    ok &= testSchemeChargeConservation(pic::DepositionScheme::TSC);
    ok &= testSchemeChargeConservation(pic::DepositionScheme::Esirkepov);
    ok &= testEsirkepovMovingChargeConservation();
    ok &= testSchemeChargeConservation(pic::DepositionScheme::NGP);
#ifdef PIC_BUILD_CUDA
    ok &= testGpuDeposit(pic::DepositionBackend::GPUAtomics, "GPU CIC atomics charge conservation");
    ok &= testGpuDeposit(pic::DepositionBackend::GPUPrivatized, "GPU CIC privatized charge conservation");
#endif
    return ok ? 0 : 1;
}
