#include "pic/poisson_fft.hpp"

#include <algorithm>
#include <cmath>

namespace pic {

PoissonSolverGS::PoissonSolverGS(const Domain& domain, int max_iters, double tol)
    : domain_(domain), max_iters_(max_iters), tol_(tol) {}

void PoissonSolverGS::solve(FieldGrid& grid) {
    const int nx = domain_.Nx;
    const int ny = domain_.Ny;
    const double dx2 = domain_.dx * domain_.dx;
    const double dy2 = domain_.dy * domain_.dy;
    const double denom = 2.0 * (1.0 / dx2 + 1.0 / dy2);
    auto& phi = grid.phi();
    auto& rho = grid.rho();
    const double scale = domain_.cellVolume();

    std::fill(phi.begin(), phi.end(), 0.0);

    for (int iter = 0; iter < max_iters_; ++iter) {
        double max_delta = 0.0;
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int im = (i - 1 + nx) % nx;
                const int ip = (i + 1) % nx;
                const int jm = (j - 1 + ny) % ny;
                const int jp = (j + 1) % ny;
                const std::size_t id = static_cast<std::size_t>(j * nx + i);
                const double new_phi =
                    ((phi[static_cast<std::size_t>(j * nx + im)] + phi[static_cast<std::size_t>(j * nx + ip)]) /
                         dx2 +
                     (phi[static_cast<std::size_t>(jm * nx + i)] + phi[static_cast<std::size_t>(jp * nx + i)]) /
                         dy2 +
                     rho[id] * scale) /
                    denom;
                max_delta = std::max(max_delta, std::abs(new_phi - phi[id]));
                phi[id] = new_phi;
            }
        }
        if (max_delta < tol_) {
            break;
        }
    }

    const double idx_dx = 1.0 / domain_.dx;
    const double idx_dy = 1.0 / domain_.dy;
    auto& Ex = grid.Ex();
    auto& Ey = grid.Ey();
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int ip = (i + 1) % nx;
            const int jp = (j + 1) % ny;
            const std::size_t id = static_cast<std::size_t>(j * nx + i);
            Ex[id] = -(phi[static_cast<std::size_t>(j * nx + ip)] - phi[id]) * idx_dx;
            Ey[id] = -(phi[static_cast<std::size_t>(jp * nx + i)] - phi[id]) * idx_dy;
        }
    }
}

}  // namespace pic
