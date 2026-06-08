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

LangmuirValidationResult runLangmuirValidation(const SimulationConfig& config);
std::vector<ValidationSummary> runValidationSuite(const SimulationConfig& base_config);

}  // namespace pic
