#include "benchmark/benchmark_runner.hpp"

#include <ctime>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

namespace {

bool flag(int argc, char** argv, const std::string& name) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == name) return true;
    }
    return false;
}

std::filesystem::path resultsDir() {
    std::filesystem::path results_dir = std::filesystem::current_path() / "results";
    if (std::filesystem::exists("../results")) {
        results_dir = std::filesystem::current_path().parent_path() / "results";
    }
    std::filesystem::create_directories(results_dir);
    return results_dir;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool sim = flag(argc, argv, "--sim");
        const bool validate = flag(argc, argv, "--validate");
        const bool full = flag(argc, argv, "--full");
        const bool include_gpu = flag(argc, argv, "--gpu");

        const auto results_dir = resultsDir();

        if (validate) {
            pic::SimulationConfig cfg;
            cfg.domain.Nx = 64;
            cfg.domain.Ny = 64;
            cfg.domain.Lx = 1.0;
            cfg.domain.Ly = 1.0;
            cfg.domain.dt = 0.002;
            cfg.domain.steps = 16384;
            cfg.domain.updateDerived();
            cfg.num_particles = 200;
            cfg.layout = pic::ParticleLayout::SoA;
            cfg.langmuir_amplitude = 0.05;
            cfg.langmuir_mode_number = 1;

            const auto summaries = pic::runValidationSuite(cfg);
            std::ostringstream path;
            path << results_dir.string() << "/validation_" << std::time(nullptr) << ".csv";
            pic::BenchmarkRunner::writeValidationCsv(path.str(), summaries);
            std::cout << "Wrote " << path.str() << " (" << summaries.size() << " rows)\n";
            return 0;
        }

        std::vector<pic::BenchmarkCase> cases;
        if (sim) {
            cases = pic::BenchmarkRunner::simMatrix();
        } else if (full) {
            cases = pic::BenchmarkRunner::defaultMatrix(include_gpu);
        } else {
            cases = pic::BenchmarkRunner::quickMatrix();
        }

        pic::BenchmarkRunner runner(std::move(cases));
        const auto rows = runner.runAll();

        std::ostringstream path;
        if (sim) {
            path << results_dir.string() << "/benchmark_sim_" << std::time(nullptr) << ".csv";
        } else {
            path << results_dir.string() << "/benchmark_" << std::time(nullptr) << ".csv";
        }
        runner.writeCsv(path.str(), rows);
        std::cout << "Wrote " << path.str() << " (" << rows.size() << " rows)\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
