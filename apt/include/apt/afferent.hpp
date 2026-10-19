// apt/afferent.hpp - Module E (1, 3) and Module O (5): the afferent channel, the latency-to-
// learning-rate law, and the sensing specification the transducer must meet.
#pragma once

#include <string>
#include <vector>

#include "apt/evidence.hpp"

namespace apt {

// Gaussian-channel description of the afferent return path (manuscript Section 3.6).
// Latency does not reduce Shannon rate; it removes the fraction of the movement bandwidth for
// which feedback still arrives within one cycle, 1 / (2 tau f_max).
struct AfferentChannel {
    double bandwidth_hz;     // proprioceptive bandwidth of voluntary hand movement
    double snr_biological;   // 1 / Weber_force^2
    double snr_device;       // (span / quantum)^2

    static AfferentChannel from_evidence(const EvidenceBase& eb);

    static double shannon_rate(double snr, double bandwidth_hz);
    static double usable_fraction(double tau_ms, double bandwidth_hz);

    double rate_biological() const { return shannon_rate(snr_biological, bandwidth_hz); }
    double rate_device() const { return shannon_rate(snr_device, bandwidth_hz); }
    double effective_rate(double tau_ms) const {
        return rate_device() * usable_fraction(tau_ms, bandwidth_hz);
    }
};

// a(tau) = a0 / (1 + tau / tau_half), fixed *exactly* by the two verified visuomotor adaptation
// rates (60 ms and 259.2 ms). Two points, two parameters, nothing left over.
class LatencyLearningLaw {
public:
    static LatencyLearningLaw from_evidence(const EvidenceBase& eb);
    LatencyLearningLaw(double d0_ms, double a0_obs, double d1_ms, double a1_obs);

    double rate(double tau_ms) const { return a0_ / (1.0 + tau_ms / tau_half_); }
    // learning rate relative to a reference latency - the only way latency enters Module D
    double relative(double tau_ms, double ref_ms) const { return rate(tau_ms) / rate(ref_ms); }
    double a0() const { return a0_; }
    double tau_half_ms() const { return tau_half_; }

private:
    double a0_, tau_half_;
};

// Module O5: resolution a transducer must reach before the afferent channel is limited by
// biology rather than hardware, one row per psychophysical channel.
struct SensingChannel {
    std::string channel;
    double biological_resolution;
    std::string unit;
    std::string basis;
    double required_bits;
    bool independent;  // false for channels that restate another channel's mechanical variable
};

struct SensingSpecification {
    std::vector<SensingChannel> channels;
    double bits_all() const;          // naive sum over six channels
    double bits_independent() const;  // dropping the two channels that are not independent
    double gating_limit_hz;           // 1 / verified somatosensory gating ISI
    double rate_all_bit_s() const { return bits_all() * gating_limit_hz; }
    double rate_independent_bit_s() const { return bits_independent() * gating_limit_hz; }
};

SensingSpecification sensing_specification(const EvidenceBase& eb);

}  // namespace apt
