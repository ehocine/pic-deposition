#include "pic/field_grid.hpp"

#include <cmath>

namespace pic {

FieldGrid::FieldGrid(const Domain& domain) : domain_(domain) {
    const std::size_t n = static_cast<std::size_t>(domain_.Nx * domain_.Ny);
    rho_.assign(n, 0.0);
    phi_.assign(n, 0.0);
    Ex_.assign(n, 0.0);
    Ey_.assign(n, 0.0);
}

void FieldGrid::clearFields() {
    std::fill(rho_.begin(), rho_.end(), 0.0);
    std::fill(phi_.begin(), phi_.end(), 0.0);
    std::fill(Ex_.begin(), Ex_.end(), 0.0);
    std::fill(Ey_.begin(), Ey_.end(), 0.0);
}

void FieldGrid::clearRho() { std::fill(rho_.begin(), rho_.end(), 0.0); }

double FieldGrid::integratedRho() const {
    double sum = 0.0;
    for (double v : rho_) {
        sum += v;
    }
    return sum * domain_.cellVolume();
}

double FieldGrid::fieldEnergy() const {
    double sum = 0.0;
    for (std::size_t i = 0; i < Ex_.size(); ++i) {
        sum += Ex_[i] * Ex_[i] + Ey_[i] * Ey_[i];
    }
    return 0.5 * sum * domain_.cellVolume();
}

}  // namespace pic
