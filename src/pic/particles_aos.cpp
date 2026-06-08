#include "pic/particles.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace pic {

namespace {

double sampleNormal(std::mt19937& rng, double stddev) {
    std::normal_distribution<double> dist(0.0, stddev);
    return dist(rng);
}

}  // namespace

ParticlesAoS::ParticlesAoS(std::size_t count) { resize(count); }

void ParticlesAoS::resize(std::size_t count) { data_.resize(count); }

void ParticlesAoS::initializeUniformMaxwellian(const Domain& domain, double T, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ux(0.0, domain.Lx);
    std::uniform_real_distribution<double> uy(0.0, domain.Ly);

    const double stdv = std::sqrt(T);
    const double qp = 1.0;
    for (std::size_t i = 0; i < data_.size(); ++i) {
        auto& p = data_[i];
        p.x = ux(rng);
        p.y = uy(rng);
        p.vx = sampleNormal(rng, stdv);
        p.vy = sampleNormal(rng, stdv);
        p.m = 1.0;
        p.q = (i % 2 == 0) ? qp : -qp;
    }
}

double ParticlesAoS::totalCharge() const {
    double sum = 0.0;
    for (const auto& p : data_) {
        sum += p.q;
    }
    return sum;
}

double ParticlesAoS::kineticEnergy() const {
    double sum = 0.0;
    for (const auto& p : data_) {
        sum += 0.5 * p.m * (p.vx * p.vx + p.vy * p.vy);
    }
    return sum;
}

ParticlesSoA::ParticlesSoA(std::size_t count) { resize(count); }

void ParticlesSoA::resize(std::size_t count) {
    x_.resize(count);
    y_.resize(count);
    vx_.resize(count);
    vy_.resize(count);
    q_.resize(count);
    m_.resize(count);
}

void ParticlesSoA::initializeUniformMaxwellian(const Domain& domain, double T, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ux(0.0, domain.Lx);
    std::uniform_real_distribution<double> uy(0.0, domain.Ly);
    const double stdv = std::sqrt(T);
    const double qp = 1.0;
    for (std::size_t i = 0; i < x_.size(); ++i) {
        x_[i] = ux(rng);
        y_[i] = uy(rng);
        vx_[i] = sampleNormal(rng, stdv);
        vy_[i] = sampleNormal(rng, stdv);
        m_[i] = 1.0;
        q_[i] = (i % 2 == 0) ? qp : -qp;
    }
}

void ParticlesSoA::initializeLangmuirWave(const Domain& domain, double amplitude, int mode, unsigned seed) {
    (void)seed;
    const std::size_t n = x_.size();
    const double k = 2.0 * M_PI * static_cast<double>(mode) / domain.Lx;
    const double qp = -1.0 / static_cast<double>(n);
    for (std::size_t i = 0; i < n; ++i) {
        x_[i] = domain.Lx * static_cast<double>(i) / static_cast<double>(n);
        y_[i] = 0.5 * domain.Ly;
        vx_[i] = amplitude * std::sin(k * x_[i]);
        vy_[i] = 0.0;
        m_[i] = 1.0;
        q_[i] = qp;
    }
}

void ParticlesAoS::initializeLangmuirWave(const Domain& domain, double amplitude, int mode, unsigned seed) {
    ParticlesSoA soa(data_.size());
    soa.initializeLangmuirWave(domain, amplitude, mode, seed);
    data_ = toAoS(soa).data();
}

double ParticlesSoA::totalCharge() const {
    double sum = 0.0;
    for (double v : q_) {
        sum += v;
    }
    return sum;
}

double ParticlesSoA::kineticEnergy() const {
    double sum = 0.0;
    for (std::size_t i = 0; i < x_.size(); ++i) {
        sum += 0.5 * m_[i] * (vx_[i] * vx_[i] + vy_[i] * vy_[i]);
    }
    return sum;
}

ParticlesAoS toAoS(const ParticlesSoA& soa) {
    ParticlesAoS aos(soa.size());
    for (std::size_t i = 0; i < soa.size(); ++i) {
        aos.data()[i] = {soa.x()[i], soa.y()[i], soa.vx()[i], soa.vy()[i], soa.q()[i], soa.m()[i]};
    }
    return aos;
}

ParticlesSoA toSoA(const ParticlesAoS& aos) {
    ParticlesSoA soa(aos.size());
    for (std::size_t i = 0; i < aos.size(); ++i) {
        const auto& p = aos.data()[i];
        soa.x()[i] = p.x;
        soa.y()[i] = p.y;
        soa.vx()[i] = p.vx;
        soa.vy()[i] = p.vy;
        soa.q()[i] = p.q;
        soa.m()[i] = p.m;
    }
    return soa;
}

std::string layoutName(ParticleLayout layout) {
    return layout == ParticleLayout::AoS ? "AoS" : "SoA";
}

}  // namespace pic
