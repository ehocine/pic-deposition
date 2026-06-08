#include "pic/poisson_fft.hpp"

#include <fftw3.h>

#include <cmath>
#include <stdexcept>

namespace pic {

PoissonSolverFFT::PoissonSolverFFT(const Domain& domain) : domain_(domain) {
    const int n = domain_.Nx * domain_.Ny;
    fft_in_ = fftw_alloc_real(static_cast<std::size_t>(n));
    fft_out_ = fftw_alloc_real(static_cast<std::size_t>(n));
    if (!fft_in_ || !fft_out_) {
        throw std::runtime_error("FFTW allocation failed");
    }

    plan_forward_ = fftw_plan_r2r_2d(domain_.Ny, domain_.Nx, fft_in_, fft_out_, FFTW_R2HC, FFTW_R2HC,
                                     FFTW_MEASURE);
    plan_backward_ = fftw_plan_r2r_2d(domain_.Ny, domain_.Nx, fft_out_, fft_in_, FFTW_HC2R, FFTW_HC2R,
                                      FFTW_MEASURE);

    k2_.assign(static_cast<std::size_t>(n), 0.0);
    const double lx = domain_.Lx;
    const double ly = domain_.Ly;
    for (int j = 0; j < domain_.Ny; ++j) {
        for (int i = 0; i < domain_.Nx; ++i) {
            double kx = 2.0 * M_PI * ((i <= domain_.Nx / 2) ? i : i - domain_.Nx) / lx;
            double ky = 2.0 * M_PI * ((j <= domain_.Ny / 2) ? j : j - domain_.Ny) / ly;
            k2_[static_cast<std::size_t>(j * domain_.Nx + i)] = kx * kx + ky * ky;
        }
    }
    if (!k2_.empty()) {
        k2_[0] = 1.0;
    }
}

PoissonSolverFFT::~PoissonSolverFFT() {
    fftw_destroy_plan(static_cast<fftw_plan>(plan_forward_));
    fftw_destroy_plan(static_cast<fftw_plan>(plan_backward_));
    fftw_free(fft_in_);
    fftw_free(fft_out_);
}

void PoissonSolverFFT::solve(FieldGrid& grid) {
    const int nx = domain_.Nx;
    const int ny = domain_.Ny;
    const int n = nx * ny;
    const double cell_vol = domain_.cellVolume();

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            fft_in_[j * nx + i] = grid.rho()[static_cast<std::size_t>(j * nx + i)] * cell_vol;
        }
    }

    fftw_execute(static_cast<fftw_plan>(plan_forward_));

    fft_out_[0] = 0.0;
    for (int idx = 1; idx < n; ++idx) {
        fft_out_[idx] /= k2_[static_cast<std::size_t>(idx)];
    }

    fftw_execute(static_cast<fftw_plan>(plan_backward_));

    const double norm = 1.0 / static_cast<double>(n);
    auto& phi = grid.phi();
    auto& Ex = grid.Ex();
    auto& Ey = grid.Ey();

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            phi[static_cast<std::size_t>(j * nx + i)] = fft_in_[j * nx + i] * norm;
        }
    }

    const double idx_dx = 1.0 / domain_.dx;
    const double idx_dy = 1.0 / domain_.dy;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int ip = (i + 1) % nx;
            const int jp = (j + 1) % ny;
            const double dphidx = (phi[static_cast<std::size_t>(j * nx + ip)] -
                                   phi[static_cast<std::size_t>(j * nx + i)]) *
                                  idx_dx;
            const double dphidy = (phi[static_cast<std::size_t>(jp * nx + i)] -
                                   phi[static_cast<std::size_t>(j * nx + i)]) *
                                  idx_dy;
            const std::size_t id = static_cast<std::size_t>(j * nx + i);
            Ex[id] = -dphidx;
            Ey[id] = -dphidy;
        }
    }
}

}  // namespace pic
