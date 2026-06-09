#pragma once

#include "deposition/deposition.hpp"
#include "pic/simulation.hpp"
#include "pic/validation.hpp"

#include <fstream>
#include <string>
#include <vector>

namespace pic {

struct BenchmarkCase {
    std::size_t num_particles = 10000;
    int grid_n = 64;
    DepositionScheme scheme = DepositionScheme::CIC;
    ParticleLayout layout = ParticleLayout::SoA;
    bool sorted = false;
    DepositionBackend backend = DepositionBackend::CPU;
    int num_threads = 1;
    int repeats = 10;
    int sim_steps = 200;
    bool run_simulation = false;
    bool measure_spectral_noise = false;
};

struct TimestepProfileRow {
    std::size_t num_particles = 0;
    int grid_n = 128;
    DepositionScheme scheme = DepositionScheme::CIC;
    int sort_interval = 1;
    TimestepProfile profile;
};

struct BenchmarkRow {
    BenchmarkCase case_config;
    double sort_ms = 0.0;
    double deposit_ms = 0.0;
    double deposit_std_ms = 0.0;
    double deposition_ms = 0.0;
    double full_step_ms = 0.0;
    double throughput = 0.0;
    double bandwidth_gbs = 0.0;
    double charge_error = 0.0;
    double energy_drift = 0.0;
    double spectral_noise = 0.0;
    double speedup = 1.0;
};

class BenchmarkRunner {
public:
    explicit BenchmarkRunner(std::vector<BenchmarkCase> cases);

    std::vector<BenchmarkRow> runAll() const;
    void writeCsv(const std::string& path, const std::vector<BenchmarkRow>& rows) const;
    static void writeValidationCsv(const std::string& path, const std::vector<ValidationSummary>& rows);

    static std::vector<BenchmarkCase> quickMatrix();
    static std::vector<BenchmarkCase> simMatrix();
    static std::vector<BenchmarkCase> gpuMatrix();
    static std::vector<BenchmarkCase> defaultMatrix(bool include_gpu);

    static std::vector<TimestepProfileRow> runTimestepProfiles();
    static std::vector<TimestepProfileRow> runAmortizedTimestepProfiles();
    static void writeTimestepProfileCsv(const std::string& path, const std::vector<TimestepProfileRow>& rows);

private:
    std::vector<BenchmarkCase> cases_;
};

}  // namespace pic
