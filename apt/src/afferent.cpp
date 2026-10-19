#include "apt/afferent.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace apt {

AfferentChannel AfferentChannel::from_evidence(const EvidenceBase& eb) {
    const double w_force = eb.get("weber_fraction_force");
    const double quantum = STRUCTURAL("sensor_quantum_N", 0.01, "N",
                                      "FSR-class force transducer quantum (Module E)");
    const double span = STRUCTURAL("sensor_span_N", 20.0, "N",
                                   "force transducer span specified for the interface (Module E)");
    const double bw = STRUCTURAL("proprioceptive_bandwidth_Hz", 30.0, "Hz",
                                 "bandwidth of voluntary hand movement (Module E)");
    return {bw, 1.0 / (w_force * w_force), (span / quantum) * (span / quantum)};
}

double AfferentChannel::shannon_rate(double snr, double bw) { return bw * std::log2(1.0 + snr); }

double AfferentChannel::usable_fraction(double tau_ms, double bw) {
    if (tau_ms <= 0) return 1.0;
    return std::clamp(1.0 / (2.0 * (tau_ms / 1000.0) * bw), 0.0, 1.0);
}

LatencyLearningLaw::LatencyLearningLaw(double d0, double a0_obs, double d1, double a1_obs) {
    const double r = a0_obs / a1_obs;
    if (!(r > 1.0) || !(d1 > d0))
        throw std::invalid_argument("learning law needs a slower rate at the longer delay");
    tau_half_ = (d1 - d0 * r) / (r - 1.0);
    a0_ = a0_obs * (1.0 + d0 / tau_half_);
}

LatencyLearningLaw LatencyLearningLaw::from_evidence(const EvidenceBase& eb) {
    // 259.2 ms is the verified *measured total* delay of the slow condition (Module E)
    return {60.0, eb.get("adaptation_rate_60ms"), 259.2, eb.get("adaptation_rate_259ms")};
}

double SensingSpecification::bits_all() const {
    double s = 0;
    for (const auto& c : channels) s += c.required_bits;
    return s;
}

double SensingSpecification::bits_independent() const {
    double s = 0;
    for (const auto& c : channels)
        if (c.independent) s += c.required_bits;
    return s;
}

SensingSpecification sensing_specification(const EvidenceBase& eb) {
    const double w_force = eb.get("weber_fraction_force");
    const double grip_norm = eb.get("grip_norm_men_60_69");
    const double jnd_static = eb.get("jnd_elbow_torque_static");
    const double jnd_dynamic = eb.get("jnd_elbow_torque_dynamic");
    const double mcp = eb.get("position_threshold_mcp");
    const double pip = eb.get("position_threshold_pip");
    const double w_stiff = eb.get("weber_fraction_stiffness");
    const double torque_fs = STRUCTURAL("elbow_torque_full_scale_Nm", 1.0, "N m",
                                        "declared full-scale range (Section 4.6) - unsourced");
    const double angle_fs = STRUCTURAL("finger_angle_full_scale_deg", 90.0, "deg",
                                       "declared full-scale excursion (Section 4.6) - unsourced");
    SensingSpecification s;
    s.gating_limit_hz = 1000.0 / eb.get("somatosensory_gating_isi");
    s.channels = {
        {"grip force", w_force * grip_norm, "kg at a 40 kg reference",
         "verified force Weber fraction x verified norm", std::log2(1.0 / w_force), true},
        {"elbow torque, static", jnd_static, "N m", "verified just-noticeable difference",
         std::log2(torque_fs / jnd_static), true},
        // the same mechanical variable as the static channel, measured during movement
        {"elbow torque, dynamic", jnd_dynamic, "N m", "verified just-noticeable difference",
         std::log2(torque_fs / jnd_dynamic), false},
        {"finger joint angle (MCP)", mcp, "degrees", "verified position threshold",
         std::log2(angle_fs / mcp), true},
        {"finger joint angle (PIP)", pip, "degrees", "verified position threshold",
         std::log2(angle_fs / pip), true},
        // a force-to-displacement ratio the force and angle channels already carry
        {"stiffness", w_stiff, "fraction", "verified stiffness Weber fraction",
         std::log2(1.0 / w_stiff), false},
    };
    return s;
}

}  // namespace apt
