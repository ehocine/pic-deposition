#include "deposition/cuda/deposition_cuda.hpp"
#include "deposition/deposition_impl.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

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

__device__ void depositNGPGlobal(double q, double x, double y, double dx, double dy, double inv_vol, int nx, int ny,
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

__device__ void depositCICGlobal(double q, double x, double y, double dx, double dy, double inv_vol, int nx, int ny,
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

__device__ void depositTSCGlobal(double q, double x, double y, double dx, double dy, double inv_vol, int nx, int ny,
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

__device__ void depositBySchemeGlobal(int scheme, double q, double x, double y, double dx, double dy, double inv_vol,
                                      int nx, int ny, double* rho, bool use_atomics) {
    switch (scheme) {
        case 0:
            depositNGPGlobal(q, x, y, dx, dy, inv_vol, nx, ny, rho, use_atomics);
            break;
        case 1:
            depositCICGlobal(q, x, y, dx, dy, inv_vol, nx, ny, rho, use_atomics);
            break;
        case 2:
            depositTSCGlobal(q, x, y, dx, dy, inv_vol, nx, ny, rho, use_atomics);
            break;
        default:
            depositCICGlobal(q, x, y, dx, dy, inv_vol, nx, ny, rho, use_atomics);
            break;
    }
}

__device__ void addToTile(int jx, int jy, double amount, int tile_ox, int tile_oy, int tile_size, double* tile_rho) {
    if (jx >= tile_ox && jx < tile_ox + tile_size && jy >= tile_oy && jy < tile_oy + tile_size) {
        const int lx = jx - tile_ox;
        const int ly = jy - tile_oy;
        atomicAdd(&tile_rho[ly * tile_size + lx], amount);
    }
}

__device__ void depositNGPIntoTile(double q, double x, double y, double dx, double dy, double inv_vol, int nx, int ny,
                                   int tile_ox, int tile_oy, int tile_size, double* tile_rho) {
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
    addToTile(jx, jy, q * inv_vol, tile_ox, tile_oy, tile_size, tile_rho);
}

__device__ void depositCICIntoTile(double q, double x, double y, double dx, double dy, double inv_vol, int nx, int ny,
                                 int tile_ox, int tile_oy, int tile_size, double* tile_rho) {
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
    addToTile(ix, iy, v * (1.0 - wx) * (1.0 - wy), tile_ox, tile_oy, tile_size, tile_rho);
    addToTile(ixp, iy, v * wx * (1.0 - wy), tile_ox, tile_oy, tile_size, tile_rho);
    addToTile(ix, iyp, v * (1.0 - wx) * wy, tile_ox, tile_oy, tile_size, tile_rho);
    addToTile(ixp, iyp, v * wx * wy, tile_ox, tile_oy, tile_size, tile_rho);
}

__device__ void depositTSCIntoTile(double q, double x, double y, double dx, double dy, double inv_vol, int nx, int ny,
                                 int tile_ox, int tile_oy, int tile_size, double* tile_rho) {
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
            addToTile(jx, jy, v * wx[dxi] * wy[dyi], tile_ox, tile_oy, tile_size, tile_rho);
        }
    }
}

__device__ void depositBySchemeIntoTile(int scheme, double q, double x, double y, double dx, double dy, double inv_vol,
                                        int nx, int ny, int tile_ox, int tile_oy, int tile_size, double* tile_rho) {
    switch (scheme) {
        case 0:
            depositNGPIntoTile(q, x, y, dx, dy, inv_vol, nx, ny, tile_ox, tile_oy, tile_size, tile_rho);
            break;
        case 1:
            depositCICIntoTile(q, x, y, dx, dy, inv_vol, nx, ny, tile_ox, tile_oy, tile_size, tile_rho);
            break;
        case 2:
            depositTSCIntoTile(q, x, y, dx, dy, inv_vol, nx, ny, tile_ox, tile_oy, tile_size, tile_rho);
            break;
        default:
            depositCICIntoTile(q, x, y, dx, dy, inv_vol, nx, ny, tile_ox, tile_oy, tile_size, tile_rho);
            break;
    }
}

__global__ void depositKernelSoA(const double* x, const double* y, const double* q, int num_particles, int nx, int ny,
                                 double dx, double dy, double inv_vol, int scheme, bool use_atomics, double* rho) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    depositBySchemeGlobal(scheme, q[idx], x[idx], y[idx], dx, dy, inv_vol, nx, ny, rho, use_atomics);
}

__global__ void depositKernelSpatialTiles(const double* x, const double* y, const double* q, int num_particles, int nx,
                                        int ny, double dx, double dy, double inv_vol, int scheme, int tile_size,
                                        double* rho_global) {
    extern __shared__ double tile_rho[];
    const int tile_ix = blockIdx.x;
    const int tile_iy = blockIdx.y;
    const int tile_ox = tile_ix * tile_size;
    const int tile_oy = tile_iy * tile_size;
    const int tile_cells = tile_size * tile_size;

    for (int i = threadIdx.x; i < tile_cells; i += blockDim.x) {
        tile_rho[i] = 0.0;
    }
    __syncthreads();

    for (int p = threadIdx.x; p < num_particles; p += blockDim.x) {
        depositBySchemeIntoTile(scheme, q[p], x[p], y[p], dx, dy, inv_vol, nx, ny, tile_ox, tile_oy, tile_size, tile_rho);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < tile_cells; i += blockDim.x) {
        const int lx = i % tile_size;
        const int ly = i / tile_size;
        const int gx = tile_ox + lx;
        const int gy = tile_oy + ly;
        if (gx < nx && gy < ny && tile_rho[i] != 0.0) {
            atomicAdd(rho_global + gy * nx + gx, tile_rho[i]);
        }
    }
}

struct GpuBufferPool {
    double* d_x = nullptr;
    double* d_y = nullptr;
    double* d_q = nullptr;
    double* d_rho = nullptr;
    std::size_t particle_cap = 0;
    std::size_t grid_cap = 0;

    void ensureParticles(std::size_t num_particles) {
        if (num_particles <= particle_cap) return;
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
        if (d_q) cudaFree(d_q);
        CUDA_CHECK(cudaMalloc(&d_x, num_particles * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_y, num_particles * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_q, num_particles * sizeof(double)));
        particle_cap = num_particles;
    }

    void ensureGrid(std::size_t n) {
        if (n <= grid_cap) return;
        if (d_rho) cudaFree(d_rho);
        CUDA_CHECK(cudaMalloc(&d_rho, n * sizeof(double)));
        grid_cap = n;
    }
};

GpuBufferPool& bufferPool() {
    static GpuBufferPool pool;
    return pool;
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
    const std::size_t np = static_cast<std::size_t>(num_particles);
    const std::size_t grid_cells = static_cast<std::size_t>(n);

    auto& pool = bufferPool();
    pool.ensureParticles(np);
    pool.ensureGrid(grid_cells);

    CUDA_CHECK(cudaMemcpy(pool.d_x, particles.x().data(), np * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pool.d_y, particles.y().data(), np * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pool.d_q, particles.q().data(), np * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(pool.d_rho, 0, grid_cells * sizeof(double)));

    const int threads = 256;
    const int blocks = (num_particles + threads - 1) / threads;
    const int scheme = schemeToInt(config.scheme);
    const bool use_privatized = config.backend == DepositionBackend::GPUPrivatized;
    const bool use_atomics = config.backend == DepositionBackend::GPUAtomics;

    if (use_privatized) {
        const int tile = kGpuTileSize;
        const dim3 grid_blocks((domain.Nx + tile - 1) / tile, (domain.Ny + tile - 1) / tile);
        const std::size_t shared_bytes = static_cast<std::size_t>(tile * tile) * sizeof(double);
        depositKernelSpatialTiles<<<grid_blocks, threads, shared_bytes>>>(
            pool.d_x, pool.d_y, pool.d_q, num_particles, domain.Nx, domain.Ny, domain.dx, domain.dy, inv_vol, scheme,
            tile, pool.d_rho);
    } else {
        depositKernelSoA<<<blocks, threads>>>(pool.d_x, pool.d_y, pool.d_q, num_particles, domain.Nx, domain.Ny,
                                              domain.dx, domain.dy, inv_vol, scheme, use_atomics, pool.d_rho);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(grid.rho().data(), pool.d_rho, grid_cells * sizeof(double), cudaMemcpyDeviceToHost));
}

}  // namespace

void depositChargeCudaAoS(const ParticlesAoS& particles, FieldGrid& grid, const DepositionConfig& config) {
    ParticlesSoA soa = toSoA(particles);
    launchDeposition(soa, grid, config);
}

void depositChargeCudaSoA(const ParticlesSoA& particles, FieldGrid& grid, const DepositionConfig& config) {
    launchDeposition(particles, grid, config);
}

}  // namespace pic
