#pragma once

#include "pic/field_grid.hpp"
#include "pic/particles.hpp"

namespace pic {

struct Diagnostics {
    double charge_error = 0.0;
    double total_charge = 0.0;
    double integrated_rho = 0.0;
    double kinetic_energy = 0.0;
    double field_energy = 0.0;
    double total_energy = 0.0;
};

Diagnostics computeDiagnostics(const FieldGrid& grid, const ParticlesAoS& particles);
Diagnostics computeDiagnostics(const FieldGrid& grid, const ParticlesSoA& particles);

double chargeConservationError(double integrated_rho, double total_charge, double abs_charge_sum);
double totalAbsChargeAoS(const ParticlesAoS& particles);
double totalAbsChargeSoA(const ParticlesSoA& particles);

double spectralNoiseLevel(const FieldGrid& grid);

}  // namespace pic
