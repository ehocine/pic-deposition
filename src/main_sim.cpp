#include "pic/simulation.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

pic::SimulationConfig parseArgs(int argc, char** argv) {
    pic::SimulationConfig cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + arg);
            }
            return argv[++i];
        };
        if (arg == "--particles") {
            cfg.num_particles = static_cast<std::size_t>(std::stoull(next()));
        } else if (arg == "--grid") {
            cfg.domain.Nx = std::stoi(next());
            cfg.domain.Ny = cfg.domain.Nx;
        } else if (arg == "--steps") {
            cfg.domain.steps = std::stoi(next());
        } else if (arg == "--scheme") {
            cfg.scheme = pic::schemeFromString(next());
        } else if (arg == "--layout") {
            cfg.layout = next() == "AoS" ? pic::ParticleLayout::AoS : pic::ParticleLayout::SoA;
        } else if (arg == "--sort") {
            cfg.sorted = next() == "on";
        } else if (arg == "--threads") {
            cfg.deposition.num_threads = std::stoi(next());
        }
    }
    cfg.deposition.scheme = cfg.scheme;
    cfg.deposition.layout = cfg.layout;
    cfg.deposition.sorted = cfg.sorted;
    cfg.deposition.esirkepov_dt = cfg.domain.dt;
    cfg.domain.updateDerived();
    return cfg;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        pic::SimulationConfig cfg = parseArgs(argc, argv);
        pic::Simulation sim(cfg);
        const auto result = sim.run();
        std::cout << "initial_energy=" << result.initial_energy << "\n";
        std::cout << "final_energy=" << result.final_energy << "\n";
        std::cout << "energy_drift=" << result.energy_drift << "\n";
        std::cout << "charge_error=" << result.final_charge_error << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
