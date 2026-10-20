// apt/stability.hpp - Module E (2): the delay-dependent stability envelope of the admittance law,
// simulated rather than asserted, and the Colgate-Schenkel sampled-data passivity condition that
// the manuscript reports separately as a requirement on the hardware.
#pragma once

#include <vector>

namespace apt {

struct AdmittanceDesign {
    double mass_kg = 0.25;      // rendered virtual mass   (structural, swept)
    double damping_Ns_m = 6.0;  // rendered virtual damping (structural, swept)
    double dt_s = 1e-3;         // 1 kHz interface loop
};

// Simulate the delayed mass-spring-damper loop exactly as Module E does, judging stability by
// whether the oscillation envelope grows (last third vs middle third), not by settling time.
bool delayed_loop_stable(double stiffness_N_m, double tau_ms, const AdmittanceDesign& d,
                         double tmax_s = 8.0, double kick = 1.0);

// Largest stiffness on geomspace(0.5, 2e4, 80) below the first unstable grid point; 0 if none.
double max_stable_stiffness(double tau_ms, const AdmittanceDesign& d);

struct EnvelopeRow {
    double latency_ms;
    double max_stable_stiffness_N_m;
    double usable_bandwidth_fraction;
};

std::vector<EnvelopeRow> stability_envelope(const std::vector<double>& latencies_ms,
                                            const AdmittanceDesign& d, double bandwidth_hz);

// Latency at which the envelope closes (max stable stiffness reaches zero) for a given design,
// found by bisection on the grid search; returns +inf if still open at `tau_hi_ms`.
double closure_latency_ms(const AdmittanceDesign& d, double tau_hi_ms = 400.0);

// Colgate & Schenkel (1997): a sampled virtual wall K, B rendered at period T is passive iff the
// device's physical damping b exceeds K T / 2 + |B|.
struct PassivityCondition {
    static double required_physical_damping(double K_N_m, double B_Ns_m, double T_s) {
        return K_N_m * T_s / 2.0 + (B_Ns_m < 0 ? -B_Ns_m : B_Ns_m);
    }
    static double max_passive_stiffness(double b_phys, double B_Ns_m, double T_s) {
        const double head = b_phys - (B_Ns_m < 0 ? -B_Ns_m : B_Ns_m);
        return head > 0 ? 2.0 * head / T_s : 0.0;
    }
};

}  // namespace apt
