#pragma once

#include "deposition/deposition.hpp"
#include "pic/field_grid.hpp"
#include "pic/particles.hpp"

namespace pic {

constexpr int kGpuTileSize = 16;

void depositChargeCudaAoS(const ParticlesAoS& particles, FieldGrid& grid, const DepositionConfig& config);
void depositChargeCudaSoA(const ParticlesSoA& particles, FieldGrid& grid, const DepositionConfig& config);

}  // namespace pic
