#pragma once

#include "pic/domain.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace pic {

enum class ParticleLayout { AoS, SoA };

struct ParticleAoS {
    double x = 0.0;
    double y = 0.0;
    double vx = 0.0;
    double vy = 0.0;
    double q = 0.0;
    double m = 1.0;
};

class ParticlesAoS {
public:
    explicit ParticlesAoS(std::size_t count = 0);

    std::size_t size() const { return data_.size(); }
    std::vector<ParticleAoS>& data() { return data_; }
    const std::vector<ParticleAoS>& data() const { return data_; }

    void resize(std::size_t count);
    void initializeUniformMaxwellian(const Domain& domain, double T, unsigned seed);
    void initializeLangmuirWave(const Domain& domain, double amplitude, int mode, unsigned seed);
    double totalCharge() const;
    double kineticEnergy() const;

private:
    std::vector<ParticleAoS> data_;
};

class ParticlesSoA {
public:
    explicit ParticlesSoA(std::size_t count = 0);

    std::size_t size() const { return x_.size(); }

    void resize(std::size_t count);
    void initializeUniformMaxwellian(const Domain& domain, double T, unsigned seed);
    void initializeLangmuirWave(const Domain& domain, double amplitude, int mode, unsigned seed);
    double totalCharge() const;
    double kineticEnergy() const;

    std::vector<double>& x() { return x_; }
    std::vector<double>& y() { return y_; }
    std::vector<double>& vx() { return vx_; }
    std::vector<double>& vy() { return vy_; }
    std::vector<double>& q() { return q_; }
    std::vector<double>& m() { return m_; }

    const std::vector<double>& x() const { return x_; }
    const std::vector<double>& y() const { return y_; }
    const std::vector<double>& vx() const { return vx_; }
    const std::vector<double>& vy() const { return vy_; }
    const std::vector<double>& q() const { return q_; }
    const std::vector<double>& m() const { return m_; }

private:
    std::vector<double> x_;
    std::vector<double> y_;
    std::vector<double> vx_;
    std::vector<double> vy_;
    std::vector<double> q_;
    std::vector<double> m_;
};

ParticlesAoS toAoS(const ParticlesSoA& soa);
ParticlesSoA toSoA(const ParticlesAoS& aos);

std::string layoutName(ParticleLayout layout);

}  // namespace pic
