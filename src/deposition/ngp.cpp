#include "deposition/deposition_impl.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace pic {

inline int wrapIndex(int i, int n) { return (i % n + n) % n; }

namespace {

void depositEsirkepovParticle(double q, double x, double y, double vx, double vy, double dt, const Domain& domain,
                              double inv_vol, std::vector<double>& rho, std::vector<EsirkepovWeight>& x_support,
                              std::vector<EsirkepovWeight>& y_support) {
    esirkepovWeights1DSparse(x, x + vx * dt, domain.dx, domain.Nx, x_support);
    esirkepovWeights1DSparse(y, y + vy * dt, domain.dy, domain.Ny, y_support);
    for (const auto& sx : x_support) {
        for (const auto& sy : y_support) {
            rho[static_cast<std::size_t>(domain.cellIndex(sx.index, sy.index))] +=
                q * sx.weight * sy.weight * inv_vol;
        }
    }
}

}  // namespace

void tscWeights(double xi, double w[3], int& i0) {
    i0 = static_cast<int>(std::floor(xi - 0.5));
    const double x = xi - static_cast<double>(i0);
    w[0] = 0.5 * (1.5 - x) * (1.5 - x);
    w[1] = 0.75 - (x - 1.0) * (x - 1.0);
    w[2] = 0.5 * (x - 0.5) * (x - 0.5);
}

void esirkepovWeights1DSparse(double x0, double x1, double dx, int n, std::vector<EsirkepovWeight>& weights) {
    weights.clear();
    weights.reserve(8);
    if (std::abs(x1 - x0) < 1e-15) {
        int ix = static_cast<int>(std::floor(x0 / dx));
        const double xi = x0 / dx - static_cast<double>(ix);
        weights.push_back({wrapIndex(ix, n), 1.0 - xi});
        weights.push_back({wrapIndex(ix + 1, n), xi});
        return;
    }

    const double inv_dx = 1.0 / dx;
    double x_curr = x0;
    const double x_end = x1;
    int cell = static_cast<int>(std::floor(x_curr * inv_dx));

    if (x_end > x0) {
        while (x_curr < x_end - 1e-15) {
            const double x_next = std::min(x_end, (cell + 1) * dx);
            const double w = (x_next - x_curr) * inv_dx;
            weights.push_back({wrapIndex(cell, n), w});
            x_curr = x_next;
            cell++;
        }
    } else {
        while (x_curr > x_end + 1e-15) {
            const double x_next = std::max(x_end, cell * dx);
            const double w = (x_curr - x_next) * inv_dx;
            weights.push_back({wrapIndex(cell, n), w});
            x_curr = x_next;
            cell--;
        }
    }
}

void esirkepovWeights1D(double x0, double x1, double dx, int n, std::vector<double>& weights) {
    weights.assign(static_cast<std::size_t>(n), 0.0);
    std::vector<EsirkepovWeight> sparse;
    esirkepovWeights1DSparse(x0, x1, dx, n, sparse);
    for (const auto& entry : sparse) {
        weights[static_cast<std::size_t>(entry.index)] += entry.weight;
    }
}

void depositNGP_AoS(const ParticlesAoS& particles, FieldGrid& grid) {
    const auto& domain = grid.domain();
    auto& rho = grid.rho();
    for (const auto& p : particles.data()) {
        int ix, iy;
        double wx, wy;
        domain.cellCoords(p.x, p.y, ix, iy, wx, wy);
        const int jx = wx >= 0.5 ? wrapIndex(ix + 1, domain.Nx) : ix;
        const int jy = wy >= 0.5 ? wrapIndex(iy + 1, domain.Ny) : iy;
        rho[static_cast<std::size_t>(domain.cellIndex(jx, jy))] += p.q / domain.cellVolume();
    }
}

void depositNGP_SoA(const ParticlesSoA& particles, FieldGrid& grid) {
    const auto& domain = grid.domain();
    auto& rho = grid.rho();
    const std::size_t n = particles.size();
    for (std::size_t idx = 0; idx < n; ++idx) {
        int ix, iy;
        double wx, wy;
        domain.cellCoords(particles.x()[idx], particles.y()[idx], ix, iy, wx, wy);
        const int jx = wx >= 0.5 ? wrapIndex(ix + 1, domain.Nx) : ix;
        const int jy = wy >= 0.5 ? wrapIndex(iy + 1, domain.Ny) : iy;
        rho[static_cast<std::size_t>(domain.cellIndex(jx, jy))] += particles.q()[idx] / domain.cellVolume();
    }
}

void depositCIC_AoS(const ParticlesAoS& particles, FieldGrid& grid) {
    const auto& domain = grid.domain();
    auto& rho = grid.rho();
    const double inv_vol = 1.0 / domain.cellVolume();
    for (const auto& p : particles.data()) {
        int ix, iy;
        double wx, wy;
        domain.cellCoords(p.x, p.y, ix, iy, wx, wy);
        const double q = p.q * inv_vol;
        const int ixp = wrapIndex(ix + 1, domain.Nx);
        const int iyp = wrapIndex(iy + 1, domain.Ny);
        rho[static_cast<std::size_t>(domain.cellIndex(ix, iy))] += q * (1.0 - wx) * (1.0 - wy);
        rho[static_cast<std::size_t>(domain.cellIndex(ixp, iy))] += q * wx * (1.0 - wy);
        rho[static_cast<std::size_t>(domain.cellIndex(ix, iyp))] += q * (1.0 - wx) * wy;
        rho[static_cast<std::size_t>(domain.cellIndex(ixp, iyp))] += q * wx * wy;
    }
}

void depositCIC_SoA(const ParticlesSoA& particles, FieldGrid& grid) {
    const auto& domain = grid.domain();
    auto& rho = grid.rho();
    const double inv_vol = 1.0 / domain.cellVolume();
    const std::size_t n = particles.size();
    for (std::size_t idx = 0; idx < n; ++idx) {
        int ix, iy;
        double wx, wy;
        domain.cellCoords(particles.x()[idx], particles.y()[idx], ix, iy, wx, wy);
        const double q = particles.q()[idx] * inv_vol;
        const int ixp = wrapIndex(ix + 1, domain.Nx);
        const int iyp = wrapIndex(iy + 1, domain.Ny);
        rho[static_cast<std::size_t>(domain.cellIndex(ix, iy))] += q * (1.0 - wx) * (1.0 - wy);
        rho[static_cast<std::size_t>(domain.cellIndex(ixp, iy))] += q * wx * (1.0 - wy);
        rho[static_cast<std::size_t>(domain.cellIndex(ix, iyp))] += q * (1.0 - wx) * wy;
        rho[static_cast<std::size_t>(domain.cellIndex(ixp, iyp))] += q * wx * wy;
    }
}

void depositTSC_AoS(const ParticlesAoS& particles, FieldGrid& grid) {
    const auto& domain = grid.domain();
    auto& rho = grid.rho();
    const double inv_vol = 1.0 / domain.cellVolume();
    for (const auto& p : particles.data()) {
        const double fx = p.x / domain.dx;
        const double fy = p.y / domain.dy;
        int ix0, iy0;
        double wx[3], wy[3];
        tscWeights(fx, wx, ix0);
        tscWeights(fy, wy, iy0);
        const double q = p.q * inv_vol;
        for (int dy = 0; dy < 3; ++dy) {
            const int jy = wrapIndex(iy0 + dy, domain.Ny);
            for (int dx = 0; dx < 3; ++dx) {
                const int jx = wrapIndex(ix0 + dx, domain.Nx);
                rho[static_cast<std::size_t>(domain.cellIndex(jx, jy))] += q * wx[dx] * wy[dy];
            }
        }
    }
}

void depositTSC_SoA(const ParticlesSoA& particles, FieldGrid& grid) {
    const auto& domain = grid.domain();
    auto& rho = grid.rho();
    const double inv_vol = 1.0 / domain.cellVolume();
    const std::size_t n = particles.size();
    for (std::size_t idx = 0; idx < n; ++idx) {
        const double fx = particles.x()[idx] / domain.dx;
        const double fy = particles.y()[idx] / domain.dy;
        int ix0, iy0;
        double wx[3], wy[3];
        tscWeights(fx, wx, ix0);
        tscWeights(fy, wy, iy0);
        const double q = particles.q()[idx] * inv_vol;
        for (int dy = 0; dy < 3; ++dy) {
            const int jy = wrapIndex(iy0 + dy, domain.Ny);
            for (int dx = 0; dx < 3; ++dx) {
                const int jx = wrapIndex(ix0 + dx, domain.Nx);
                rho[static_cast<std::size_t>(domain.cellIndex(jx, jy))] += q * wx[dx] * wy[dy];
            }
        }
    }
}

void depositEsirkepov_AoS(const ParticlesAoS& particles, FieldGrid& grid, double dt) {
    const auto& domain = grid.domain();
    auto& rho = grid.rho();
    const double inv_vol = 1.0 / domain.cellVolume();
    std::vector<EsirkepovWeight> x_support;
    std::vector<EsirkepovWeight> y_support;

    for (const auto& p : particles.data()) {
        depositEsirkepovParticle(p.q, p.x, p.y, p.vx, p.vy, dt, domain, inv_vol, rho, x_support, y_support);
    }
}

void depositEsirkepov_SoA(const ParticlesSoA& particles, FieldGrid& grid, double dt) {
    const auto& domain = grid.domain();
    auto& rho = grid.rho();
    const double inv_vol = 1.0 / domain.cellVolume();
    std::vector<EsirkepovWeight> x_support;
    std::vector<EsirkepovWeight> y_support;
    const std::size_t n = particles.size();

    for (std::size_t idx = 0; idx < n; ++idx) {
        depositEsirkepovParticle(particles.q()[idx], particles.x()[idx], particles.y()[idx], particles.vx()[idx],
                                 particles.vy()[idx], dt, domain, inv_vol, rho, x_support, y_support);
    }
}

}  // namespace pic
