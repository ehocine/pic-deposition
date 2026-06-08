#pragma once

#include "pic/domain.hpp"

#include <vector>

namespace pic {

class FieldGrid {
public:
    explicit FieldGrid(const Domain& domain);

    void clearFields();
    void clearRho();

    const Domain& domain() const { return domain_; }

    std::vector<double>& rho() { return rho_; }
    const std::vector<double>& rho() const { return rho_; }

    std::vector<double>& phi() { return phi_; }
    const std::vector<double>& phi() const { return phi_; }

    std::vector<double>& Ex() { return Ex_; }
    const std::vector<double>& Ex() const { return Ex_; }

    std::vector<double>& Ey() { return Ey_; }
    const std::vector<double>& Ey() const { return Ey_; }

    double integratedRho() const;
    double fieldEnergy() const;

private:
    Domain domain_;
    std::vector<double> rho_;
    std::vector<double> phi_;
    std::vector<double> Ex_;
    std::vector<double> Ey_;
};

}  // namespace pic
