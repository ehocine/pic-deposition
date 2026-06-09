#pragma once

#include "pic/domain.hpp"
#include "pic/field_grid.hpp"
#include "pic/particles.hpp"

#include <string>

namespace pic {

enum class DepositionScheme { NGP, CIC, TSC, Esirkepov };

enum class DepositionBackend { CPU, GPUAtomics, GPUPrivatized };

struct DepositionConfig {
    DepositionScheme scheme = DepositionScheme::CIC;
    ParticleLayout layout = ParticleLayout::SoA;
    DepositionBackend backend = DepositionBackend::CPU;
    bool sorted = false;
    int num_threads = 1;
    double dx_esirkepov = 0.0;
    double dy_esirkepov = 0.0;
    double esirkepov_dt = 0.0;
};

struct DepositionStats {
    double sort_ms = 0.0;
    double deposit_ms = 0.0;
    double deposit_std_ms = 0.0;
    double time_ms = 0.0;
    double throughput_particles_per_s = 0.0;
    double effective_bandwidth_gbs = 0.0;
    double charge_error = 0.0;
    std::size_t bytes_read = 0;
    std::size_t bytes_written = 0;
};

DepositionStats estimateBytes(const DepositionConfig& config, std::size_t num_particles, int nx, int ny);

void depositChargeAoS(const ParticlesAoS& particles, FieldGrid& grid, const DepositionConfig& config);
void depositChargeSoA(const ParticlesSoA& particles, FieldGrid& grid, const DepositionConfig& config);

DepositionStats depositChargeTimedAoS(const ParticlesAoS& particles, FieldGrid& grid,
                                      const DepositionConfig& config, int repeats = 1);
DepositionStats depositChargeTimedSoA(const ParticlesSoA& particles, FieldGrid& grid,
                                      const DepositionConfig& config, int repeats = 1);

void gatherFieldsAoS(ParticlesAoS& particles, FieldGrid& grid, double dt);
void gatherFieldsSoA(ParticlesSoA& particles, FieldGrid& grid, double dt);
void pushParticlesAoS(ParticlesAoS& particles, const FieldGrid& grid, double dt);
void pushParticlesSoA(ParticlesSoA& particles, const FieldGrid& grid, double dt);

std::string schemeName(DepositionScheme scheme);
std::string backendName(DepositionBackend backend);
DepositionScheme schemeFromString(const std::string& name);
DepositionBackend backendFromString(const std::string& name);

}  // namespace pic
