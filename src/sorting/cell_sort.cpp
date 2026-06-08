#include "sorting/cell_sort.hpp"

#include <algorithm>
#include <vector>

namespace pic {

int particleCellIndex(double x, double y, const Domain& domain) {
    int ix, iy;
    double wx, wy;
    domain.cellCoords(x, y, ix, iy, wx, wy);
    return domain.cellIndex(ix, iy);
}

void sortParticlesByCell(ParticlesAoS& particles, const Domain& domain) {
    auto& data = particles.data();
    std::sort(data.begin(), data.end(), [&](const ParticleAoS& a, const ParticleAoS& b) {
        return particleCellIndex(a.x, a.y, domain) < particleCellIndex(b.x, b.y, domain);
    });
}

void sortParticlesByCell(ParticlesSoA& particles, const Domain& domain) {
    const std::size_t n = particles.size();
    std::vector<int> order(n);
    for (std::size_t i = 0; i < n; ++i) {
        order[i] = static_cast<int>(i);
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return particleCellIndex(particles.x()[static_cast<std::size_t>(a)], particles.y()[static_cast<std::size_t>(a)],
                                 domain) <
               particleCellIndex(particles.x()[static_cast<std::size_t>(b)], particles.y()[static_cast<std::size_t>(b)],
                                 domain);
    });

    ParticlesSoA sorted(n);
    for (std::size_t i = 0; i < n; ++i) {
        const int src = order[i];
        sorted.x()[i] = particles.x()[static_cast<std::size_t>(src)];
        sorted.y()[i] = particles.y()[static_cast<std::size_t>(src)];
        sorted.vx()[i] = particles.vx()[static_cast<std::size_t>(src)];
        sorted.vy()[i] = particles.vy()[static_cast<std::size_t>(src)];
        sorted.q()[i] = particles.q()[static_cast<std::size_t>(src)];
        sorted.m()[i] = particles.m()[static_cast<std::size_t>(src)];
    }
    particles = std::move(sorted);
}

}  // namespace pic
