#pragma once

#include "pic/domain.hpp"
#include "pic/field_grid.hpp"

namespace pic {

class PoissonSolverFFT {
public:
    explicit PoissonSolverFFT(const Domain& domain);
    ~PoissonSolverFFT();

    PoissonSolverFFT(const PoissonSolverFFT&) = delete;
    PoissonSolverFFT& operator=(const PoissonSolverFFT&) = delete;

    void solve(FieldGrid& grid);

private:
    Domain domain_;
    void* plan_forward_;
    void* plan_backward_;
    double* fft_in_;
    double* fft_out_;
    std::vector<double> k2_;
};

class PoissonSolverGS {
public:
    explicit PoissonSolverGS(const Domain& domain, int max_iters = 500, double tol = 1e-6);

    void solve(FieldGrid& grid);

private:
    Domain domain_;
    int max_iters_;
    double tol_;
};

}  // namespace pic
