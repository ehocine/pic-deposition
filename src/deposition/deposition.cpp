#include "deposition/deposition.hpp"
#include "deposition/deposition_impl.hpp"
#include "deposition/deposition_parallel.hpp"
#include "sorting/cell_sort.hpp"

#include "pic/diagnostics.hpp"

#include <chrono>
#include <cmath>
#include <omp.h>
#include <stdexcept>

#ifdef PIC_BUILD_CUDA
#include "deposition/cuda/deposition_cuda.hpp"
#endif

namespace pic {

namespace {

void setThreadCount(int num_threads) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
}

void depositCpuAoS(const ParticlesAoS& particles, FieldGrid& grid, const DepositionConfig& config) {
    if (config.num_threads > 1) {
        switch (config.scheme) {
            case DepositionScheme::NGP:
                depositNGP_AoS_Parallel(particles, grid, config.num_threads);
                return;
            case DepositionScheme::CIC:
                depositCIC_AoS_Parallel(particles, grid, config.num_threads);
                return;
            case DepositionScheme::TSC:
                depositTSC_AoS_Parallel(particles, grid, config.num_threads);
                return;
            case DepositionScheme::Esirkepov:
                depositEsirkepov_AoS_Parallel(particles, grid, config.esirkepov_dt, config.num_threads);
                return;
        }
    }
    switch (config.scheme) {
        case DepositionScheme::NGP:
            depositNGP_AoS(particles, grid);
            break;
        case DepositionScheme::CIC:
            depositCIC_AoS(particles, grid);
            break;
        case DepositionScheme::TSC:
            depositTSC_AoS(particles, grid);
            break;
        case DepositionScheme::Esirkepov:
            depositEsirkepov_AoS(particles, grid, config.esirkepov_dt);
            break;
    }
}

void depositCpuSoA(const ParticlesSoA& particles, FieldGrid& grid, const DepositionConfig& config) {
    if (config.num_threads > 1) {
        switch (config.scheme) {
            case DepositionScheme::NGP:
                depositNGP_SoA_Parallel(particles, grid, config.num_threads);
                return;
            case DepositionScheme::CIC:
                depositCIC_SoA_Parallel(particles, grid, config.num_threads);
                return;
            case DepositionScheme::TSC:
                depositTSC_SoA_Parallel(particles, grid, config.num_threads);
                return;
            case DepositionScheme::Esirkepov:
                depositEsirkepov_SoA_Parallel(particles, grid, config.esirkepov_dt, config.num_threads);
                return;
        }
    }
    switch (config.scheme) {
        case DepositionScheme::NGP:
            depositNGP_SoA(particles, grid);
            break;
        case DepositionScheme::CIC:
            depositCIC_SoA(particles, grid);
            break;
        case DepositionScheme::TSC:
            depositTSC_SoA(particles, grid);
            break;
        case DepositionScheme::Esirkepov:
            depositEsirkepov_SoA(particles, grid, config.esirkepov_dt);
            break;
    }
}

DepositionStats finalizeStats(const DepositionConfig& config, std::size_t num_particles, int nx, int ny,
                              double sort_seconds, double deposit_seconds, double charge_error) {
    DepositionStats stats = estimateBytes(config, num_particles, nx, ny);
    stats.sort_ms = sort_seconds * 1000.0;
    stats.deposit_ms = deposit_seconds * 1000.0;
    stats.time_ms = stats.sort_ms + stats.deposit_ms;
    stats.throughput_particles_per_s = num_particles / std::max(deposit_seconds, 1e-12);
    stats.effective_bandwidth_gbs = (static_cast<double>(stats.bytes_read + stats.bytes_written) / 1e9) /
                                    std::max(deposit_seconds, 1e-12);
    stats.charge_error = charge_error;
    return stats;
}

void depositKernelAoS(const ParticlesAoS& particles, FieldGrid& grid, const DepositionConfig& config) {
    grid.clearRho();
    setThreadCount(config.num_threads);
    depositCpuAoS(particles, grid, config);
}

void depositKernelSoA(const ParticlesSoA& particles, FieldGrid& grid, const DepositionConfig& config) {
    grid.clearRho();
    setThreadCount(config.num_threads);
    depositCpuSoA(particles, grid, config);
}

}  // namespace

DepositionStats estimateBytes(const DepositionConfig& config, std::size_t num_particles, int nx, int ny) {
    DepositionStats stats;
    const std::size_t particle_record = config.layout == ParticleLayout::AoS ? 48 : 40;
    stats.bytes_read = num_particles * particle_record;

    std::size_t writes_per_particle = 1;
    switch (config.scheme) {
        case DepositionScheme::NGP:
            writes_per_particle = 1;
            break;
        case DepositionScheme::CIC:
            writes_per_particle = 4;
            break;
        case DepositionScheme::TSC:
            writes_per_particle = 9;
            break;
        case DepositionScheme::Esirkepov:
            writes_per_particle = 4;
            break;
    }
    stats.bytes_written = num_particles * writes_per_particle * sizeof(double) + static_cast<std::size_t>(nx * ny) * sizeof(double);
    return stats;
}

void depositChargeAoS(const ParticlesAoS& particles, FieldGrid& grid, const DepositionConfig& config) {
    grid.clearRho();
    setThreadCount(config.num_threads);

    ParticlesAoS ordered = particles;
    if (config.sorted) {
        sortParticlesByCell(ordered, grid.domain());
    }

    if (config.backend == DepositionBackend::CPU) {
        depositCpuAoS(ordered, grid, config);
        return;
    }

#ifdef PIC_BUILD_CUDA
    depositChargeCudaAoS(ordered, grid, config);
#else
    throw std::runtime_error("CUDA backend requested but project was built without BUILD_CUDA=ON");
#endif
}

void depositChargeSoA(const ParticlesSoA& particles, FieldGrid& grid, const DepositionConfig& config) {
    grid.clearRho();
    setThreadCount(config.num_threads);

    ParticlesSoA ordered = particles;
    if (config.sorted) {
        sortParticlesByCell(ordered, grid.domain());
    }

    if (config.backend == DepositionBackend::CPU) {
        depositCpuSoA(ordered, grid, config);
        return;
    }

#ifdef PIC_BUILD_CUDA
    depositChargeCudaSoA(ordered, grid, config);
#else
    throw std::runtime_error("CUDA backend requested but project was built without BUILD_CUDA=ON");
#endif
}

DepositionStats depositChargeTimedAoS(const ParticlesAoS& particles, FieldGrid& grid,
                                      const DepositionConfig& config, int repeats) {
    ParticlesAoS ordered = particles;
    DepositionConfig dep = config;
    dep.sorted = false;

    for (int i = 0; i < 3; ++i) {
        if (config.sorted) {
            sortParticlesByCell(ordered, grid.domain());
        }
        depositKernelAoS(ordered, grid, dep);
    }

    double sort_seconds = 0.0;
    if (config.sorted) {
        const auto sort_start = std::chrono::steady_clock::now();
        sortParticlesByCell(ordered, grid.domain());
        const auto sort_end = std::chrono::steady_clock::now();
        sort_seconds = std::chrono::duration<double>(sort_end - sort_start).count();
    }

    const auto dep_start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        depositKernelAoS(ordered, grid, dep);
    }
    const auto dep_end = std::chrono::steady_clock::now();
    const double deposit_seconds = std::chrono::duration<double>(dep_end - dep_start).count() / repeats;

    const double charge_error = chargeConservationError(grid.integratedRho(), particles.totalCharge(),
                                                        totalAbsChargeAoS(particles));
    return finalizeStats(config, particles.size(), grid.domain().Nx, grid.domain().Ny, sort_seconds, deposit_seconds,
                         charge_error);
}

DepositionStats depositChargeTimedSoA(const ParticlesSoA& particles, FieldGrid& grid,
                                      const DepositionConfig& config, int repeats) {
    ParticlesSoA ordered = particles;
    DepositionConfig dep = config;
    dep.sorted = false;

    for (int i = 0; i < 3; ++i) {
        if (config.sorted) {
            sortParticlesByCell(ordered, grid.domain());
        }
        depositKernelSoA(ordered, grid, dep);
    }

    double sort_seconds = 0.0;
    if (config.sorted) {
        const auto sort_start = std::chrono::steady_clock::now();
        sortParticlesByCell(ordered, grid.domain());
        const auto sort_end = std::chrono::steady_clock::now();
        sort_seconds = std::chrono::duration<double>(sort_end - sort_start).count();
    }

    const auto dep_start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        depositKernelSoA(ordered, grid, dep);
    }
    const auto dep_end = std::chrono::steady_clock::now();
    const double deposit_seconds = std::chrono::duration<double>(dep_end - dep_start).count() / repeats;

    const double charge_error = chargeConservationError(grid.integratedRho(), particles.totalCharge(),
                                                        totalAbsChargeSoA(particles));
    return finalizeStats(config, particles.size(), grid.domain().Nx, grid.domain().Ny, sort_seconds, deposit_seconds,
                         charge_error);
}

void gatherFieldsAoS(ParticlesAoS& particles, FieldGrid& grid, double dt) {
    const auto& domain = grid.domain();
    const auto& Ex = grid.Ex();
    const auto& Ey = grid.Ey();
    for (auto& p : particles.data()) {
        int ix, iy;
        double wx, wy;
        domain.cellCoords(p.x, p.y, ix, iy, wx, wy);
        const int ixp = (ix + 1) % domain.Nx;
        const int iyp = (iy + 1) % domain.Ny;
        const double ex = Ex[static_cast<std::size_t>(domain.cellIndex(ix, iy))] * (1.0 - wx) * (1.0 - wy) +
                          Ex[static_cast<std::size_t>(domain.cellIndex(ixp, iy))] * wx * (1.0 - wy) +
                          Ex[static_cast<std::size_t>(domain.cellIndex(ix, iyp))] * (1.0 - wx) * wy +
                          Ex[static_cast<std::size_t>(domain.cellIndex(ixp, iyp))] * wx * wy;
        const double ey = Ey[static_cast<std::size_t>(domain.cellIndex(ix, iy))] * (1.0 - wx) * (1.0 - wy) +
                          Ey[static_cast<std::size_t>(domain.cellIndex(ixp, iy))] * wx * (1.0 - wy) +
                          Ey[static_cast<std::size_t>(domain.cellIndex(ix, iyp))] * (1.0 - wx) * wy +
                          Ey[static_cast<std::size_t>(domain.cellIndex(ixp, iyp))] * wx * wy;
        p.vx += -p.q / p.m * ex * dt;
        p.vy += -p.q / p.m * ey * dt;
    }
}

void gatherFieldsSoA(ParticlesSoA& particles, FieldGrid& grid, double dt) {
    const auto& domain = grid.domain();
    const auto& Ex = grid.Ex();
    const auto& Ey = grid.Ey();
    for (std::size_t i = 0; i < particles.size(); ++i) {
        int ix, iy;
        double wx, wy;
        domain.cellCoords(particles.x()[i], particles.y()[i], ix, iy, wx, wy);
        const int ixp = (ix + 1) % domain.Nx;
        const int iyp = (iy + 1) % domain.Ny;
        const double ex = Ex[static_cast<std::size_t>(domain.cellIndex(ix, iy))] * (1.0 - wx) * (1.0 - wy) +
                          Ex[static_cast<std::size_t>(domain.cellIndex(ixp, iy))] * wx * (1.0 - wy) +
                          Ex[static_cast<std::size_t>(domain.cellIndex(ix, iyp))] * (1.0 - wx) * wy +
                          Ex[static_cast<std::size_t>(domain.cellIndex(ixp, iyp))] * wx * wy;
        const double ey = Ey[static_cast<std::size_t>(domain.cellIndex(ix, iy))] * (1.0 - wx) * (1.0 - wy) +
                          Ey[static_cast<std::size_t>(domain.cellIndex(ixp, iy))] * wx * (1.0 - wy) +
                          Ey[static_cast<std::size_t>(domain.cellIndex(ix, iyp))] * (1.0 - wx) * wy +
                          Ey[static_cast<std::size_t>(domain.cellIndex(ixp, iyp))] * wx * wy;
        particles.vx()[i] += -particles.q()[i] / particles.m()[i] * ex * dt;
        particles.vy()[i] += -particles.q()[i] / particles.m()[i] * ey * dt;
    }
}

void pushParticlesAoS(ParticlesAoS& particles, const FieldGrid& grid, double dt) {
    const auto& domain = grid.domain();
    for (auto& p : particles.data()) {
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        domain.wrapPosition(p.x, p.y);
    }
}

void pushParticlesSoA(ParticlesSoA& particles, const FieldGrid& grid, double dt) {
    const auto& domain = grid.domain();
    for (std::size_t i = 0; i < particles.size(); ++i) {
        particles.x()[i] += particles.vx()[i] * dt;
        particles.y()[i] += particles.vy()[i] * dt;
        domain.wrapPosition(particles.x()[i], particles.y()[i]);
    }
}

std::string schemeName(DepositionScheme scheme) {
    switch (scheme) {
        case DepositionScheme::NGP:
            return "NGP";
        case DepositionScheme::CIC:
            return "CIC";
        case DepositionScheme::TSC:
            return "TSC";
        case DepositionScheme::Esirkepov:
            return "Esirkepov";
    }
    return "Unknown";
}

std::string backendName(DepositionBackend backend) {
    switch (backend) {
        case DepositionBackend::CPU:
            return "CPU";
        case DepositionBackend::GPUAtomics:
            return "GPU_Atomics";
        case DepositionBackend::GPUPrivatized:
            return "GPU_Priv";
    }
    return "Unknown";
}

DepositionScheme schemeFromString(const std::string& name) {
    if (name == "NGP") return DepositionScheme::NGP;
    if (name == "CIC") return DepositionScheme::CIC;
    if (name == "TSC") return DepositionScheme::TSC;
    if (name == "Esirkepov") return DepositionScheme::Esirkepov;
    throw std::runtime_error("Unknown deposition scheme: " + name);
}

DepositionBackend backendFromString(const std::string& name) {
    if (name == "CPU") return DepositionBackend::CPU;
    if (name == "GPU_Atomics") return DepositionBackend::GPUAtomics;
    if (name == "GPU_Priv") return DepositionBackend::GPUPrivatized;
    throw std::runtime_error("Unknown backend: " + name);
}

}  // namespace pic
