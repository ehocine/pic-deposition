#pragma once

#include "deposition/deposition_impl.hpp"
#include "pic/field_grid.hpp"
#include "pic/particles.hpp"

namespace pic {

void depositNGP_AoS_Parallel(const ParticlesAoS& particles, FieldGrid& grid, int num_threads);
void depositNGP_SoA_Parallel(const ParticlesSoA& particles, FieldGrid& grid, int num_threads);
void depositCIC_AoS_Parallel(const ParticlesAoS& particles, FieldGrid& grid, int num_threads);
void depositCIC_SoA_Parallel(const ParticlesSoA& particles, FieldGrid& grid, int num_threads);
void depositTSC_AoS_Parallel(const ParticlesAoS& particles, FieldGrid& grid, int num_threads);
void depositTSC_SoA_Parallel(const ParticlesSoA& particles, FieldGrid& grid, int num_threads);
void depositEsirkepov_AoS_Parallel(const ParticlesAoS& particles, FieldGrid& grid, double dt, int num_threads);
void depositEsirkepov_SoA_Parallel(const ParticlesSoA& particles, FieldGrid& grid, double dt, int num_threads);

}  // namespace pic
