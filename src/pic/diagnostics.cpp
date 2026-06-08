#include "pic/diagnostics.hpp"

#include <fftw3.h>

#include <cmath>
#include <vector>

namespace pic {

double totalAbsChargeAoS(const ParticlesAoS& particles) {
    double sum = 0.0;
    for (const auto& p : particles.data()) {
        sum += std::abs(p.q);
    }
    return sum;
}

double totalAbsChargeSoA(const ParticlesSoA& particles) {
    double sum = 0.0;
    for (double q : particles.q()) {
        sum += std::abs(q);
    }
    return sum;
}

double chargeConservationError(double integrated_rho, double total_charge, double abs_charge_sum) {
    const double denom = std::max(abs_charge_sum, 1e-12);
    return std::abs(integrated_rho - total_charge) / denom;
}

Diagnostics computeDiagnostics(const FieldGrid& grid, const ParticlesAoS& particles) {
    Diagnostics d;
    d.total_charge = particles.totalCharge();
    d.integrated_rho = grid.integratedRho();
    d.charge_error = chargeConservationError(d.integrated_rho, d.total_charge, totalAbsChargeAoS(particles));
    d.kinetic_energy = particles.kineticEnergy();
    d.field_energy = grid.fieldEnergy();
    d.total_energy = d.kinetic_energy + d.field_energy;
    return d;
}

Diagnostics computeDiagnostics(const FieldGrid& grid, const ParticlesSoA& particles) {
    Diagnostics d;
    d.total_charge = particles.totalCharge();
    d.integrated_rho = grid.integratedRho();
    d.charge_error = chargeConservationError(d.integrated_rho, d.total_charge, totalAbsChargeSoA(particles));
    d.kinetic_energy = particles.kineticEnergy();
    d.field_energy = grid.fieldEnergy();
    d.total_energy = d.kinetic_energy + d.field_energy;
    return d;
}

double spectralNoiseLevel(const FieldGrid& grid) {
    const auto& domain = grid.domain();
    const int nx = domain.Nx;
    const int ny = domain.Ny;
    const int n = nx * ny;

    double* in = fftw_alloc_real(static_cast<std::size_t>(n));
    double* out = fftw_alloc_real(static_cast<std::size_t>(n));
    if (!in || !out) {
        fftw_free(in);
        fftw_free(out);
        return 0.0;
    }

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            in[j * nx + i] = grid.rho()[static_cast<std::size_t>(j * nx + i)];
        }
    }

    fftw_plan plan = fftw_plan_r2r_2d(ny, nx, in, out, FFTW_R2HC, FFTW_R2HC, FFTW_ESTIMATE);
    fftw_execute(plan);

    const int k_cut_x = std::max(1, nx / 3);
    const int k_cut_y = std::max(1, ny / 3);

    double total_power = 0.0;
    double high_k_power = 0.0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int idx = j * nx + i;
            const double power = out[idx] * out[idx];
            total_power += power;
            const int kx = (i <= nx / 2) ? i : nx - i;
            const int ky = (j <= ny / 2) ? j : ny - j;
            if (kx >= k_cut_x || ky >= k_cut_y) {
                high_k_power += power;
            }
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    if (total_power <= 1e-30) {
        return 0.0;
    }
    return high_k_power / total_power;
}

}  // namespace pic
