// apt/dsp.hpp - causal front-end filtering and the latency budget it has to fit inside.
//
// Section 3.16 of the manuscript states the specification on offer as a *budget*, not a filter
// topology: the 30 ms discrimination threshold, less whatever group delay a physically
// realisable causal filter imposes. This header provides the filters (Butterworth low-pass as
// cascaded biquads, a notch for mains), their exact group delay, the bit-exact fixed-point
// biquad the FPGA implements, and a budget allocator that walks the order/cutoff trade-off and
// returns the Pareto front of rejection against delay.
#pragma once

#include <complex>
#include <string>
#include <vector>

#include "apt/fixed.hpp"

namespace apt::dsp {

struct Biquad {
    double b0, b1, b2, a1, a2;  // y = b0 x + b1 x1 + b2 x2 - a1 y1 - a2 y2

    std::complex<double> response(double f_hz, double fs_hz) const;
    double group_delay_samples(double f_hz, double fs_hz) const;
    bool stable() const;
};

struct Cascade {
    std::vector<Biquad> sections;
    double fs_hz;

    std::complex<double> response(double f_hz) const;
    double magnitude_db(double f_hz) const;
    double group_delay_ms(double f_hz) const;
    // worst-case group delay over [0, f_hi] - the latency the passband actually experiences
    double max_group_delay_ms(double f_hi_hz, int n = 512) const;
};

// N-th order Butterworth low-pass by bilinear transform with frequency pre-warping, factored into
// second-order sections (plus one first-order section, written as a biquad, for odd N).
Cascade butterworth_lowpass(int order, double fc_hz, double fs_hz);
// Second-order IIR notch (RBJ cookbook form) with quality factor Q.
Biquad notch(double f0_hz, double q, double fs_hz);

// Direct-form-I biquad in the shared fixed-point contract: Q12.20 signals, Q8.24 coefficients,
// one exact 64-bit accumulation, one rounding. Bit-identical to rtl/apt_core.v.
struct FixedBiquadCoefs {
    fx::coef_t b0, b1, b2, a1, a2;
    static FixedBiquadCoefs quantise(const Biquad& b);
};

class FixedBiquad {
public:
    explicit FixedBiquad(FixedBiquadCoefs c) : c_(c) {}
    fx::sig_t step(fx::sig_t x);
    void reset() { x1_ = x2_ = y1_ = y2_ = 0; }

private:
    FixedBiquadCoefs c_;
    fx::sig_t x1_ = 0, x2_ = 0, y1_ = 0, y2_ = 0;
};

// ----------------------------------------------------------------------- latency budget
struct BudgetStage {
    std::string name;
    double ms;
    std::string basis;
};

struct LatencyBudget {
    std::vector<BudgetStage> stages;
    double total_ms() const;
    double headroom_ms(double budget_ms) const { return budget_ms - total_ms(); }
};

struct ElectronicsTiming {
    double clk_hz = 50e6;
    double ctrl_hz = 1000.0;
    double adc_conversion_s = 4e-6;
    double adc_sclk_hz = 12.5e6;
    int adc_channels = 8;
    int adc_bits = 16;
    int compute_cycles = 260;   // measured from the RTL simulation (latency monitor)
    double pwm_hz = 20e3;
};

LatencyBudget afferent_budget(const Cascade& front_end, double band_hz, const ElectronicsTiming& t,
                              double stimulator_onset_ms);
LatencyBudget motor_budget(const Cascade& front_end, double band_hz, const ElectronicsTiming& t,
                           double motor_electrical_tau_ms);

struct FilterCandidate {
    int order;
    double fc_hz;
    double group_delay_ms;      // worst case over the movement band
    double reject_artefact_db;  // attenuation at the artefact frequency
    bool pareto;
};

// Enumerate Butterworth orders and cut-offs whose passband reaches `band_hz`, score each by
// rejection at `artefact_hz` against worst-case in-band delay, and mark the Pareto front.
std::vector<FilterCandidate> filter_tradeoff(double fs_hz, double band_hz, double artefact_hz,
                                             int max_order = 8);

}  // namespace apt::dsp
