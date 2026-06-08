#pragma once

#include <cmath>

namespace pic {

struct Domain {
    double Lx = 1.0;
    double Ly = 1.0;
    int Nx = 64;
    int Ny = 64;
    double dx = Lx / Nx;
    double dy = Ly / Ny;
    double dt = 0.01;
    int steps = 100;

    void updateDerived() {
        dx = Lx / Nx;
        dy = Ly / Ny;
    }

    double cellVolume() const { return dx * dy; }
    double domainVolume() const { return Lx * Ly; }

    int cellIndex(int ix, int iy) const { return iy * Nx + ix; }

    void wrapPosition(double& x, double& y) const {
        x = std::fmod(x, Lx);
        if (x < 0.0) x += Lx;
        y = std::fmod(y, Ly);
        if (y < 0.0) y += Ly;
    }

    void cellCoords(double x, double y, int& ix, int& iy, double& wx, double& wy) const {
        wx = x / dx;
        wy = y / dy;
        ix = static_cast<int>(std::floor(wx));
        iy = static_cast<int>(std::floor(wy));
        wx -= ix;
        wy -= iy;
        ix = (ix % Nx + Nx) % Nx;
        iy = (iy % Ny + Ny) % Ny;
    }
};

}  // namespace pic
