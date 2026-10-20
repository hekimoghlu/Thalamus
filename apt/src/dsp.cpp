#include "apt/dsp.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace apt::dsp {

namespace {
constexpr double kPi = std::numbers::pi;
using cd = std::complex<double>;
}  // namespace

cd Biquad::response(double f, double fs) const {
    const cd z1 = std::polar(1.0, -2.0 * kPi * f / fs);
    const cd z2 = z1 * z1;
    return (b0 + b1 * z1 + b2 * z2) / (1.0 + a1 * z1 + a2 * z2);
}

double Biquad::group_delay_samples(double f, double fs) const {
    // tau(w) = Re{ z dH/dz / H } evaluated analytically for numerator and denominator:
    // for P(z^-1) = sum c_k z^-k, the group delay contribution is Re{ sum k c_k z^-k / P }.
    const cd z1 = std::polar(1.0, -2.0 * kPi * f / fs);
    const cd z2 = z1 * z1;
    const cd num = b0 + b1 * z1 + b2 * z2, dnum = b1 * z1 + 2.0 * b2 * z2;
    const cd den = 1.0 + a1 * z1 + a2 * z2, dden = a1 * z1 + 2.0 * a2 * z2;
    return std::real(dnum / num) - std::real(dden / den);
}

bool Biquad::stable() const {
    // Jury conditions for 1 + a1 z^-1 + a2 z^-2
    return std::abs(a2) < 1.0 && std::abs(a1) < 1.0 + a2;
}

cd Cascade::response(double f) const {
    cd h = 1.0;
    for (const auto& s : sections) h *= s.response(f, fs_hz);
    return h;
}

double Cascade::magnitude_db(double f) const { return 20.0 * std::log10(std::abs(response(f))); }

double Cascade::group_delay_ms(double f) const {
    double g = 0;
    for (const auto& s : sections) g += s.group_delay_samples(f, fs_hz);
    return 1000.0 * g / fs_hz;
}

double Cascade::max_group_delay_ms(double f_hi, int n) const {
    double worst = 0;
    for (int i = 0; i <= n; ++i) worst = std::max(worst, group_delay_ms(f_hi * double(i) / n));
    return worst;
}

Cascade butterworth_lowpass(int order, double fc, double fs) {
    if (order < 1 || fc <= 0 || fc >= fs / 2) throw std::invalid_argument("bad Butterworth spec");
    const double k = std::tan(kPi * fc / fs);  // pre-warped analogue cut-off (normalised)
    Cascade c{{}, fs};
    for (int i = 0; i < order / 2; ++i) {
        // analogue pole pair s^2 + 2 sin(theta) s + 1, theta = pi (2i+1) / (2N)
        const double q2 = 2.0 * std::sin(kPi * (2.0 * i + 1.0) / (2.0 * order));
        const double norm = 1.0 + q2 * k + k * k;
        Biquad b;
        b.b0 = k * k / norm;
        b.b1 = 2.0 * b.b0;
        b.b2 = b.b0;
        b.a1 = 2.0 * (k * k - 1.0) / norm;
        b.a2 = (1.0 - q2 * k + k * k) / norm;
        c.sections.push_back(b);
    }
    if (order % 2) {
        const double norm = 1.0 + k;
        c.sections.push_back({k / norm, k / norm, 0.0, (k - 1.0) / norm, 0.0});
    }
    return c;
}

Biquad notch(double f0, double q, double fs) {
    const double w0 = 2.0 * kPi * f0 / fs;
    const double alpha = std::sin(w0) / (2.0 * q);
    const double a0 = 1.0 + alpha;
    return {1.0 / a0, -2.0 * std::cos(w0) / a0, 1.0 / a0, -2.0 * std::cos(w0) / a0,
            (1.0 - alpha) / a0};
}

FixedBiquadCoefs FixedBiquadCoefs::quantise(const Biquad& b) {
    for (double v : {b.b0, b.b1, b.b2, b.a1, b.a2})
        if (std::abs(v) >= 2.0)
            throw std::range_error("biquad coefficient outside the accumulator headroom (|c| < 2)");
    return {fx::to_coef(b.b0), fx::to_coef(b.b1), fx::to_coef(b.b2), fx::to_coef(b.a1),
            fx::to_coef(b.a2)};
}

fx::sig_t FixedBiquad::step(fx::sig_t x) {
    const fx::acc_t acc = fx::acc_t{c_.b0} * x + fx::acc_t{c_.b1} * x1_ + fx::acc_t{c_.b2} * x2_ -
                          fx::acc_t{c_.a1} * y1_ - fx::acc_t{c_.a2} * y2_;
    const fx::sig_t y = fx::rnd_sc(acc);
    x2_ = x1_;
    x1_ = x;
    y2_ = y1_;
    y1_ = y;
    return y;
}

double LatencyBudget::total_ms() const {
    double t = 0;
    for (const auto& s : stages) t += s.ms;
    return t;
}

namespace {
std::vector<BudgetStage> electronics(const ElectronicsTiming& t) {
    const double adc_ms = 1000.0 * (t.adc_conversion_s + t.adc_channels * t.adc_bits / t.adc_sclk_hz);
    return {
        {"sample-and-hold phase (mean wait for the 1 kHz sampling instant)", 500.0 / t.ctrl_hz,
         "half a control period"},
        {"ADC conversion + SPI transfer", adc_ms, "8 x 16 bit at the configured SCLK"},
        {"datapath compute (FPGA)", 1000.0 * t.compute_cycles / t.clk_hz, "RTL latency monitor"},
    };
}
}  // namespace

LatencyBudget afferent_budget(const Cascade& fe, double band, const ElectronicsTiming& t,
                              double stim_onset_ms) {
    LatencyBudget b{electronics(t)};
    b.stages.push_back({"causal front-end filter (worst in-band group delay)",
                        fe.max_group_delay_ms(band), "exact, from the realised coefficients"});
    b.stages.push_back({"stimulator onset (first phase of the biphasic pulse)", stim_onset_ms,
                        "pulse generator configuration"});
    return b;
}

LatencyBudget motor_budget(const Cascade& fe, double band, const ElectronicsTiming& t,
                           double motor_tau_ms) {
    LatencyBudget b{electronics(t)};
    b.stages.push_back({"causal front-end filter (worst in-band group delay)",
                        fe.max_group_delay_ms(band), "exact, from the realised coefficients"});
    b.stages.push_back({"PWM shadow-register update (mean)", 500.0 / t.pwm_hz,
                        "half a PWM period, centre-aligned"});
    b.stages.push_back({"motor electrical time constant", motor_tau_ms, "L/R of the actuator"});
    return b;
}

std::vector<FilterCandidate> filter_tradeoff(double fs, double band, double artefact, int max_order) {
    std::vector<FilterCandidate> out;
    for (int n = 1; n <= max_order; ++n) {
        for (double fc = band; fc < 0.45 * fs; fc *= 1.15) {
            const auto c = butterworth_lowpass(n, fc, fs);
            out.push_back({n, fc, c.max_group_delay_ms(band), -c.magnitude_db(artefact), false});
        }
    }
    for (auto& a : out) {
        a.pareto = std::none_of(out.begin(), out.end(), [&](const FilterCandidate& b) {
            return b.group_delay_ms <= a.group_delay_ms && b.reject_artefact_db >= a.reject_artefact_db &&
                   (b.group_delay_ms < a.group_delay_ms || b.reject_artefact_db > a.reject_artefact_db);
        });
    }
    return out;
}

}  // namespace apt::dsp
