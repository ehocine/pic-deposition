#pragma once

#include "deposition/deposition.hpp"
#include "pic/simulation.hpp"

namespace pic {

struct LangmuirValidationResult {
    double omega_theory = 0.0;
    double omega_measured = 0.0;
    double relative_error = 0.0;
    double omega_ratio = 0.0;
    double amplitude_decay = 0.0;
    bool passed = false;
};

struct ValidationSummary {
    DepositionScheme scheme = DepositionScheme::CIC;
    LangmuirValidationResult langmuir;
    double spectral_noise = 0.0;
    double charge_error = 0.0;
};

struct ConservationStudyResult {
    DepositionScheme scheme = DepositionScheme::CIC;
    int steps = 0;
    double max_charge_error = 0.0;
    double final_energy_drift = 0.0;
};

LangmuirValidationResult runLangmuirValidation(const SimulationConfig& config);
std::vector<ValidationSummary> runValidationSuite(const SimulationConfig& base_config);
std::vector<ConservationStudyResult> runConservationStudy(const SimulationConfig& base_config);
void writeConservationStudyCsv(const std::string& path, const std::vector<ConservationStudyResult>& rows);

struct TwoStreamValidationResult {
    DepositionScheme scheme = DepositionScheme::CIC;
    double growth_rate_measured = 0.0;
    double growth_rate_theory = 0.0;
    double growth_rate_ratio = 0.0;
    double omega_p_macro = 0.0;
    double growth_rate_over_omega_p = 0.0;
    bool passed = false;
};

struct NoiseGridResult {
    int grid_n = 64;
    DepositionScheme scheme = DepositionScheme::CIC;
    double spectral_noise = 0.0;
};

std::vector<NoiseGridResult> runNoiseVsGridStudy(std::size_t num_particles);
void writeNoiseGridCsv(const std::string& path, const std::vector<NoiseGridResult>& rows);

std::vector<TwoStreamValidationResult> runTwoStreamValidationSuite(const SimulationConfig& base_config);
TwoStreamValidationResult runTwoStreamValidation(const SimulationConfig& config);
void writeTwoStreamCsv(const std::string& path, const std::vector<TwoStreamValidationResult>& rows);

double theoreticalPlasmaFrequency(const Domain& domain, std::size_t num_particles);
double coldTwoStreamGrowthRate(const Domain& domain, std::size_t num_particles);
double resonantBeamVelocity(const Domain& domain, std::size_t num_particles, int mode = 1);
double extractFourierModeAmplitude(const FieldGrid& grid, int mode_x);
double estimateInstabilityGrowthRate(const std::vector<double>& signal, double dt, int record_start_step);
double landauDampingTheory(const Domain& domain, std::size_t num_particles, int mode, double temperature);

}  // namespace pic
