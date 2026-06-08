#include "deposition/cuda/deposition_cuda.hpp"
#include "deposition/deposition_impl.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

namespace pic {

namespace {

#define CUDA_CHECK(call)                                                                             \
    do {                                                                                             \
        cudaError_t err = (call);                                                                    \
        if (err != cudaSuccess) {                                                                    \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));         \
        }                                                                                            \
    } while (0)

__device__ int wrapDev(int i, int n) { return (i % n + n) % n; }

__device__ void depositNGP(double q, double x, double y, double dx, double dy, double inv_vol, int nx, int ny,
                           double* rho, bool use_atomics) {
    const double fx = x / dx;
    const double fy = y / dy;
    int ix = static_cast<int>(floor(fx));
    int iy = static_cast<int>(floor(fy));
    const double wx = fx - ix;
    const double wy = fy - iy;
    ix = wrapDev(ix, nx);
    iy = wrapDev(iy, ny);
    const int jx = wx >= 0.5 ? wrapDev(ix + 1, nx) : ix;
    const int jy = wy >= 0.5 ? wrapDev(iy + 1, ny) : iy;
    const double val = q * inv_vol;
    const int idx = jy * nx + jx;
    if (use_atomics) {
        atomicAdd(rho + idx, val);
    } else {
        rho[idx] += val;
    }
}

__device__ void depositCIC(double q, double x, double y, double dx, double dy, double inv_vol, int nx, int ny,
                           double* rho, bool use_atomics) {
    const double fx = x / dx;
    const double fy = y / dy;
    int ix = static_cast<int>(floor(fx));
    int iy = static_cast<int>(floor(fy));
    const double wx = fx - ix;
    const double wy = fy - iy;
    ix = wrapDev(ix, nx);
    iy = wrapDev(iy, ny);
    const int ixp = wrapDev(ix + 1, nx);
    const int iyp = wrapDev(iy + 1, ny);
    const double v = q * inv_vol;
    const double w00 = (1.0 - wx) * (1.0 - wy);
    const double w10 = wx * (1.0 - wy);
    const double w01 = (1.0 - wx) * wy;
    const double w11 = wx * wy;
    const int i00 = iy * nx + ix;
    const int i10 = iy * nx + ixp;
    const int i01 = iyp * nx + ix;
    const int i11 = iyp * nx + ixp;
    if (use_atomics) {
        atomicAdd(rho + i00, v * w00);
        atomicAdd(rho + i10, v * w10);
        atomicAdd(rho + i01, v * w01);
        atomicAdd(rho + i11, v * w11);
    } else {
        rho[i00] += v * w00;
        rho[i10] += v * w10;
        rho[i01] += v * w01;
        rho[i11] += v * w11;
    }
}

__device__ void tscWeightsDev(double xi, double w[3], int& i0) {
    i0 = static_cast<int>(floor(xi - 0.5));
    const double x = xi - static_cast<double>(i0);
    w[0] = 0.5 * (1.5 - x) * (1.5 - x);
    w[1] = 0.75 - (x - 1.0) * (x - 1.0);
    w[2] = 0.5 * (x - 0.5) * (x - 0.5);
}

__device__ void depositTSC(double q, double x, double y, double dx, double dy, double inv_vol, int nx, int ny,
                           double* rho, bool use_atomics) {
    const double fx = x / dx;
    const double fy = y / dy;
    int ix0, iy0;
    double wx[3], wy[3];
    tscWeightsDev(fx, wx, ix0);
    tscWeightsDev(fy, wy, iy0);
    const double v = q * inv_vol;
    for (int dyi = 0; dyi < 3; ++dyi) {
        const int jy = wrapDev(iy0 + dyi, ny);
        for (int dxi = 0; dxi < 3; ++dxi) {
            const int jx = wrapDev(ix0 + dxi, nx);
            const double amount = v * wx[dxi] * wy[dyi];
            const int idx = jy * nx + jx;
            if (use_atomics) {
                atomicAdd(rho + idx, amount);
            } else {
                rho[idx] += amount;
            }
        }
    }
}

__device__ void depositEsirkepovStatic(double q, double x, double y, double dx, double dy, double inv_vol, int nx,
                                       int ny, double* rho, bool use_atomics) {
    depositCIC(q, x, y, dx, dy, inv_vol, nx, ny, rho, use_atomics);
}

__global__ void depositKernelSoA(const double* x, const double* y, const double* q, int num_particles, int nx, int ny,
                                 double dx, double dy, double inv_vol, int scheme, bool use_atomics, double* rho) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    const double px = x[idx];
    const double py = y[idx];
    const double pq = q[idx];
    switch (scheme) {
        case 0:
            depositNGP(pq, px, py, dx, dy, inv_vol, nx, ny, rho, use_atomics);
            break;
        case 1:
            depositCIC(pq, px, py, dx, dy, inv_vol, nx, ny, rho, use_atomics);
            break;
        case 2:
            depositTSC(pq, px, py, dx, dy, inv_vol, nx, ny, rho, use_atomics);
            break;
        default:
            depositEsirkepovStatic(pq, px, py, dx, dy, inv_vol, nx, ny, rho, use_atomics);
            break;
    }
}

__global__ void depositKernelPrivatized(const double* x, const double* y, const double* q, int num_particles, int nx,
                                        int ny, double dx, double dy, double inv_vol, int scheme, int tile_x,
                                        int tile_y, double* rho_global) {
    extern __shared__ double tile_rho[];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int tile_cells = tile_x * tile_y;
    if (tid < tile_cells) {
        tile_rho[tid] = 0.0;
    }
    __syncthreads();

    for (int p = blockIdx.x; p < num_particles; p += gridDim.x) {
        const double px = x[p];
        const double py = y[p];
        const double pq = q[p];
        depositCIC(pq, px, py, dx, dy, inv_vol, nx, ny, tile_rho, false);
    }
    __syncthreads();

    if (tid < tile_cells) {
        atomicAdd(rho_global + tid, tile_rho[tid]);
    }
}

int schemeToInt(DepositionScheme scheme) {
    switch (scheme) {
        case DepositionScheme::NGP:
            return 0;
        case DepositionScheme::CIC:
            return 1;
        case DepositionScheme::TSC:
            return 2;
        default:
            return 3;
    }
}

void launchDeposition(const ParticlesSoA& particles, FieldGrid& grid, const DepositionConfig& config) {
    const auto& domain = grid.domain();
    const int n = domain.Nx * domain.Ny;
    const int num_particles = static_cast<int>(particles.size());
    const double inv_vol = 1.0 / domain.cellVolume();

    double *d_x, *d_y, *d_q, *d_rho;
    CUDA_CHECK(cudaMalloc(&d_x, static_cast<std::size_t>(num_particles) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, static_cast<std::size_t>(num_particles) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_q, static_cast<std::size_t>(num_particles) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rho, static_cast<std::size_t>(n) * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_x, particles.x().data(), static_cast<std::size_t>(num_particles) * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, particles.y().data(), static_cast<std::size_t>(num_particles) * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, particles.q().data(), static_cast<std::size_t>(num_particles) * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_rho, 0, static_cast<std::size_t>(n) * sizeof(double)));

    const int threads = 256;
    const int blocks = (num_particles + threads - 1) / threads;
    const int scheme = schemeToInt(config.scheme);
    const bool use_atomics = config.backend == DepositionBackend::GPUAtomics;

    if (config.backend == DepositionBackend::GPUPrivatized && domain.Nx == domain.Ny && domain.Nx <= 256) {
        dim3 block(16, 16);
        const int tile_cells = domain.Nx * domain.Ny;
        depositKernelPrivatized<<<1, block, static_cast<std::size_t>(tile_cells) * sizeof(double)>>>(
            d_x, d_y, d_q, num_particles, domain.Nx, domain.Ny, domain.dx, domain.dy, inv_vol, scheme, domain.Nx,
            domain.Ny, d_rho);
    } else {
        depositKernelSoA<<<blocks, threads>>>(d_x, d_y, d_q, num_particles, domain.Nx, domain.Ny, domain.dx,
                                              domain.dy, inv_vol, scheme, use_atomics, d_rho);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(grid.rho().data(), d_rho, static_cast<std::size_t>(n) * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_q);
    cudaFree(d_rho);
}

}  // namespace

void depositChargeCudaAoS(const ParticlesAoS& particles, FieldGrid& grid, const DepositionConfig& config) {
    ParticlesSoA soa = toAoS(particles);
    launchDeposition(soa, grid, config);
}

void depositChargeCudaSoA(const ParticlesSoA& particles, FieldGrid& grid, const DepositionConfig& config) {
    launchDeposition(particles, grid, config);
}

}  // namespace pic
