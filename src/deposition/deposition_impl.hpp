#pragma once

#include "deposition/deposition.hpp"

#include <vector>

namespace pic {

struct EsirkepovWeight {
    int index = 0;
    double weight = 0.0;
};

void tscWeights(double xi, double w[3], int& i0);
void esirkepovWeights1D(double x0, double x1, double dx, int n, std::vector<double>& weights);
void esirkepovWeights1DSparse(double x0, double x1, double dx, int n, std::vector<EsirkepovWeight>& weights);

void depositNGP_AoS(const ParticlesAoS& particles, FieldGrid& grid);
void depositNGP_SoA(const ParticlesSoA& particles, FieldGrid& grid);

void depositCIC_AoS(const ParticlesAoS& particles, FieldGrid& grid);
void depositCIC_SoA(const ParticlesSoA& particles, FieldGrid& grid);

void depositTSC_AoS(const ParticlesAoS& particles, FieldGrid& grid);
void depositTSC_SoA(const ParticlesSoA& particles, FieldGrid& grid);

void depositEsirkepov_AoS(const ParticlesAoS& particles, FieldGrid& grid, double dt);
void depositEsirkepov_SoA(const ParticlesSoA& particles, FieldGrid& grid, double dt);

}  // namespace pic
