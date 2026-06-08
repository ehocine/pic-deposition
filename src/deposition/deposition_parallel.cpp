#include "deposition/deposition_parallel.hpp"
#include "pic/particles.hpp"

#include <algorithm>
#include <omp.h>
#include <vector>

namespace pic {

namespace {

void mergeRho(std::vector<double>& global, const std::vector<double>& local) {
    for (std::size_t i = 0; i < global.size(); ++i) {
        global[i] += local[i];
    }
}

template <typename DepositLocalFn>
void parallelDepositPrivatized(const ParticlesSoA& particles, FieldGrid& grid, int num_threads,
                               DepositLocalFn deposit_local) {
    auto& rho = grid.rho();
    const std::size_t ncells = rho.size();

    if (num_threads <= 1) {
        deposit_local(particles, rho, 0, particles.size());
        return;
    }

    omp_set_num_threads(num_threads);
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        const std::size_t chunk =
            (particles.size() + static_cast<std::size_t>(nthreads) - 1) / static_cast<std::size_t>(nthreads);
        const std::size_t begin = static_cast<std::size_t>(tid) * chunk;
        const std::size_t end = std::min(begin + chunk, particles.size());

        std::vector<double> rho_local(ncells, 0.0);
        if (begin < end) {
            deposit_local(particles, rho_local, begin, end);
        }

#pragma omp critical
        { mergeRho(rho, rho_local); }
    }
}

}  // namespace

void depositNGP_SoA_Parallel(const ParticlesSoA& particles, FieldGrid& grid, int num_threads) {
    const auto& domain = grid.domain();
    const double inv_vol = 1.0 / domain.cellVolume();
    parallelDepositPrivatized(
        particles, grid, num_threads,
        [&](const ParticlesSoA& p, std::vector<double>& rho_local, std::size_t begin, std::size_t end) {
            for (std::size_t idx = begin; idx < end; ++idx) {
                int ix, iy;
                double wx, wy;
                domain.cellCoords(p.x()[idx], p.y()[idx], ix, iy, wx, wy);
                const int jx = wx >= 0.5 ? (ix + 1) % domain.Nx : ix;
                const int jy = wy >= 0.5 ? (iy + 1) % domain.Ny : iy;
                rho_local[static_cast<std::size_t>(domain.cellIndex(jx, jy))] += p.q()[idx] * inv_vol;
            }
        });
}

void depositCIC_SoA_Parallel(const ParticlesSoA& particles, FieldGrid& grid, int num_threads) {
    const auto& domain = grid.domain();
    const double inv_vol = 1.0 / domain.cellVolume();
    parallelDepositPrivatized(
        particles, grid, num_threads,
        [&](const ParticlesSoA& p, std::vector<double>& rho_local, std::size_t begin, std::size_t end) {
            for (std::size_t idx = begin; idx < end; ++idx) {
                int ix, iy;
                double wx, wy;
                domain.cellCoords(p.x()[idx], p.y()[idx], ix, iy, wx, wy);
                const double q = p.q()[idx] * inv_vol;
                const int ixp = (ix + 1) % domain.Nx;
                const int iyp = (iy + 1) % domain.Ny;
                rho_local[static_cast<std::size_t>(domain.cellIndex(ix, iy))] += q * (1.0 - wx) * (1.0 - wy);
                rho_local[static_cast<std::size_t>(domain.cellIndex(ixp, iy))] += q * wx * (1.0 - wy);
                rho_local[static_cast<std::size_t>(domain.cellIndex(ix, iyp))] += q * (1.0 - wx) * wy;
                rho_local[static_cast<std::size_t>(domain.cellIndex(ixp, iyp))] += q * wx * wy;
            }
        });
}

void depositTSC_SoA_Parallel(const ParticlesSoA& particles, FieldGrid& grid, int num_threads) {
    if (num_threads <= 1) {
        depositTSC_SoA(particles, grid);
        return;
    }
    const auto& domain = grid.domain();
    const double inv_vol = 1.0 / domain.cellVolume();
    parallelDepositPrivatized(
        particles, grid, num_threads,
        [&](const ParticlesSoA& p, std::vector<double>& rho_local, std::size_t begin, std::size_t end) {
            for (std::size_t idx = begin; idx < end; ++idx) {
                const double fx = p.x()[idx] / domain.dx;
                const double fy = p.y()[idx] / domain.dy;
                int ix0, iy0;
                double wx[3], wy[3];
                tscWeights(fx, wx, ix0);
                tscWeights(fy, wy, iy0);
                const double q = p.q()[idx] * inv_vol;
                for (int dy = 0; dy < 3; ++dy) {
                    const int jy = (iy0 + dy) % domain.Ny;
                    for (int dx = 0; dx < 3; ++dx) {
                        const int jx = (ix0 + dx) % domain.Nx;
                        rho_local[static_cast<std::size_t>(domain.cellIndex(jx, jy))] += q * wx[dx] * wy[dy];
                    }
                }
            }
        });
}

void depositEsirkepov_SoA_Parallel(const ParticlesSoA& particles, FieldGrid& grid, double dt, int num_threads) {
    if (num_threads <= 1) {
        depositEsirkepov_SoA(particles, grid, dt);
        return;
    }
    const auto& domain = grid.domain();
    const double inv_vol = 1.0 / domain.cellVolume();
    parallelDepositPrivatized(
        particles, grid, num_threads,
        [&](const ParticlesSoA& p, std::vector<double>& rho_local, std::size_t begin, std::size_t end) {
            std::vector<EsirkepovWeight> x_support;
            std::vector<EsirkepovWeight> y_support;
            for (std::size_t idx = begin; idx < end; ++idx) {
                esirkepovWeights1DSparse(p.x()[idx], p.x()[idx] + p.vx()[idx] * dt, domain.dx, domain.Nx,
                                         x_support);
                esirkepovWeights1DSparse(p.y()[idx], p.y()[idx] + p.vy()[idx] * dt, domain.dy, domain.Ny,
                                         y_support);
                const double q = p.q()[idx];
                for (const auto& sx : x_support) {
                    for (const auto& sy : y_support) {
                        rho_local[static_cast<std::size_t>(domain.cellIndex(sx.index, sy.index))] +=
                            q * sx.weight * sy.weight * inv_vol;
                    }
                }
            }
        });
}

void depositNGP_AoS_Parallel(const ParticlesAoS& particles, FieldGrid& grid, int num_threads) {
    depositNGP_SoA_Parallel(toSoA(particles), grid, num_threads);
}

void depositCIC_AoS_Parallel(const ParticlesAoS& particles, FieldGrid& grid, int num_threads) {
    depositCIC_SoA_Parallel(toSoA(particles), grid, num_threads);
}

void depositTSC_AoS_Parallel(const ParticlesAoS& particles, FieldGrid& grid, int num_threads) {
    depositTSC_SoA_Parallel(toSoA(particles), grid, num_threads);
}

void depositEsirkepov_AoS_Parallel(const ParticlesAoS& particles, FieldGrid& grid, double dt, int num_threads) {
    depositEsirkepov_SoA_Parallel(toSoA(particles), grid, dt, num_threads);
}

}  // namespace pic
