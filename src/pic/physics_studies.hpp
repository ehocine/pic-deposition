#pragma once

#include "deposition/deposition.hpp"
#include "pic/simulation.hpp"
#include "pic/validation.hpp"

#include <string>
#include <vector>

namespace pic {

struct PhysicsTimeseriesRow {
    std::string test;
    DepositionScheme scheme = DepositionScheme::CIC;
    int step = 0;
    double time = 0.0;
    double field_energy = 0.0;
    double kinetic_energy = 0.0;
    double total_energy = 0.0;
    double charge_error = 0.0;
};

struct LangmuirConvergenceRow {
    int grid_n = 64;
    std::size_t num_particles = 0;
    DepositionScheme scheme = DepositionScheme::CIC;
    double omega_measured = 0.0;
    double omega_theory = 0.0;
    double omega_ratio = 0.0;
};

struct LandauDampingRow {
    DepositionScheme scheme = DepositionScheme::CIC;
    double damping_rate_measured = 0.0;
    double damping_rate_theory = 0.0;
    double damping_ratio = 0.0;
    double k_lambda_d = 0.0;
};

struct MultiSeedRow {
    unsigned seed = 0;
    std::string test;
    DepositionScheme scheme = DepositionScheme::CIC;
    std::string metric;
    double value = 0.0;
    bool passed = false;
};

struct ProductionSimRow {
    std::size_t num_particles = 0;
    int grid_n = 128;
    DepositionScheme scheme = DepositionScheme::CIC;
    int sort_interval = 10;
    int steps = 0;
    double push_ms = 0.0;
    double sort_ms = 0.0;
    double deposit_ms = 0.0;
    double poisson_ms = 0.0;
    double gather_ms = 0.0;
    double total_ms = 0.0;
    double energy_drift = 0.0;
    double max_charge_error = 0.0;
};

std::vector<PhysicsTimeseriesRow> runPhysicsTimeseries();
std::vector<TwoStreamValidationResult> runQuasi1DTwoStreamValidation();
std::vector<LangmuirConvergenceRow> runLangmuirConvergenceSweep();
std::vector<LandauDampingRow> runLandauDampingValidation();
std::vector<MultiSeedRow> runMultiSeedValidation();
std::vector<ProductionSimRow> runProductionSimulation();

void writePhysicsTimeseriesCsv(const std::string& path, const std::vector<PhysicsTimeseriesRow>& rows);
void writeLangmuirConvergenceCsv(const std::string& path, const std::vector<LangmuirConvergenceRow>& rows);
void writeLandauDampingCsv(const std::string& path, const std::vector<LandauDampingRow>& rows);
void writeMultiSeedCsv(const std::string& path, const std::vector<MultiSeedRow>& rows);
void writeProductionSimCsv(const std::string& path, const std::vector<ProductionSimRow>& rows);
void writeQuasi1DTwoStreamCsv(const std::string& path, const std::vector<TwoStreamValidationResult>& rows);

void runAllPhysicsStudies(const std::string& results_dir);

}  // namespace pic
