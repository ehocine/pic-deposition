#pragma once

#include "pic/domain.hpp"
#include "pic/particles.hpp"

namespace pic {

void sortParticlesByCell(ParticlesAoS& particles, const Domain& domain);
void sortParticlesByCell(ParticlesSoA& particles, const Domain& domain);

int particleCellIndex(double x, double y, const Domain& domain);

}  // namespace pic
