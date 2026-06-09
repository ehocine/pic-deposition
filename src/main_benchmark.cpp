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
        const bool conservation = flag(argc, argv, "--conservation");
        const bool timestep = flag(argc, argv, "--timestep");
        const bool amortized = flag(argc, argv, "--amortized");
        const bool full = flag(argc, argv, "--full");
        const bool include_gpu = flag(argc, argv, "--gpu");

        const auto results_dir = resultsDir();

        if (timestep) {
            const auto rows = pic::BenchmarkRunner::runTimestepProfiles();
            std::ostringstream path;
            path << results_dir.string() << "/timestep_profile_" << std::time(nullptr) << ".csv";
            pic::BenchmarkRunner::writeTimestepProfileCsv(path.str(), rows);
            std::cout << "Wrote " << path.str() << " (" << rows.size() << " rows)\n";
            return 0;
        }

        if (amortized) {
            const auto rows = pic::BenchmarkRunner::runAmortizedTimestepProfiles();
            std::ostringstream path;
            path << results_dir.string() << "/amortized_timestep_" << std::time(nullptr) << ".csv";
            pic::BenchmarkRunner::writeTimestepProfileCsv(path.str(), rows);
            std::cout << "Wrote " << path.str() << " (" << rows.size() << " rows)\n";
            return 0;
        }

        if (conservation) {
            pic::SimulationConfig cons_cfg;
            cons_cfg.domain.Nx = 64;
            cons_cfg.domain.Ny = 64;
            cons_cfg.domain.Lx = 1.0;
            cons_cfg.domain.Ly = 1.0;
            cons_cfg.domain.dt = 0.005;
            cons_cfg.domain.steps = 2000;
            cons_cfg.domain.updateDerived();
            cons_cfg.num_particles = 10000;
            cons_cfg.layout = pic::ParticleLayout::SoA;
            cons_cfg.sorted = true;

            const auto rows = pic::runConservationStudy(cons_cfg);
            std::ostringstream path;
            path << results_dir.string() << "/conservation_study_" << std::time(nullptr) << ".csv";
            pic::writeConservationStudyCsv(path.str(), rows);
            std::cout << "Wrote " << path.str() << " (" << rows.size() << " rows)\n";
            return 0;
        }

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
            const std::string stamp = std::to_string(std::time(nullptr));
            std::ostringstream val_path;
            val_path << results_dir.string() << "/validation_" << stamp << ".csv";
            pic::BenchmarkRunner::writeValidationCsv(val_path.str(), summaries);
            std::cout << "Wrote " << val_path.str() << " (" << summaries.size() << " rows)\n";

            pic::SimulationConfig cons_cfg;
            cons_cfg.domain.Nx = 64;
            cons_cfg.domain.Ny = 64;
            cons_cfg.domain.Lx = 1.0;
            cons_cfg.domain.Ly = 1.0;
            cons_cfg.domain.dt = 0.005;
            cons_cfg.domain.steps = 2000;
            cons_cfg.domain.updateDerived();
            cons_cfg.num_particles = 10000;
            cons_cfg.layout = pic::ParticleLayout::SoA;
            cons_cfg.sorted = true;

            const auto conservation = pic::runConservationStudy(cons_cfg);
            std::ostringstream cons_path;
            cons_path << results_dir.string() << "/conservation_study_" << stamp << ".csv";
            pic::writeConservationStudyCsv(cons_path.str(), conservation);
            std::cout << "Wrote " << cons_path.str() << " (" << conservation.size() << " rows)\n";

            pic::SimulationConfig ts_cfg;
            ts_cfg.domain.Nx = 64;
            ts_cfg.domain.Ny = 64;
            ts_cfg.domain.Lx = 1.0;
            ts_cfg.domain.Ly = 1.0;
            ts_cfg.domain.dt = 0.002;
            ts_cfg.domain.steps = 2000;
            ts_cfg.domain.updateDerived();
            ts_cfg.num_particles = 5000;
            ts_cfg.two_stream_beam_velocity = 0.3;
            ts_cfg.two_stream_perturbation = 0.01;

            const auto two_stream = pic::runTwoStreamValidationSuite(ts_cfg);
            std::ostringstream ts_path;
            ts_path << results_dir.string() << "/two_stream_validation_" << stamp << ".csv";
            pic::writeTwoStreamCsv(ts_path.str(), two_stream);
            std::cout << "Wrote " << ts_path.str() << " (" << two_stream.size() << " rows)\n";

            const auto noise_grid = pic::runNoiseVsGridStudy(100000);
            std::ostringstream noise_path;
            noise_path << results_dir.string() << "/noise_vs_grid_" << stamp << ".csv";
            pic::writeNoiseGridCsv(noise_path.str(), noise_grid);
            std::cout << "Wrote " << noise_path.str() << " (" << noise_grid.size() << " rows)\n";
            return 0;
        }

        std::vector<pic::BenchmarkCase> cases;
        if (sim) {
            cases = pic::BenchmarkRunner::simMatrix();
        } else if (include_gpu) {
#ifndef PIC_BUILD_CUDA
            std::cerr << "Error: rebuild with -DBUILD_CUDA=ON to run GPU benchmarks.\n";
            return 1;
#else
            cases = pic::BenchmarkRunner::gpuMatrix();
#endif
        } else if (full) {
            cases = pic::BenchmarkRunner::defaultMatrix(false);
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
