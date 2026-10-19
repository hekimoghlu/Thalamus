// apt/fixed.hpp - the fixed-point contract shared bit-for-bit by the C++ golden model, the RTL
// (rtl/apt_core.v) and the firmware (fw/include/apt_fixed.h).
//
//   sig_t   Q12.20 signed 32-bit. Physical signals in SI-like device units:
//           force [N], position [mm], velocity [mm/s], energy [mJ], power [mW], duty [-1, 1].
//   coef_t  Q8.24  signed 32-bit. Dimensionless gains and discrete-time coefficients.
//   code    16-bit two's-complement ADC sample, read as a Q1.15 fraction of full scale.
//
// Every product is formed exactly in 64 bits, rounded once (round-half-up: add half an LSB,
// then arithmetic shift) and saturated to 32 bits. Multi-term expressions are accumulated
// exactly and rounded once at the end. Those three rules are the whole contract; the RTL
// implements them with the same shift amounts and the same rounding constant.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace apt::fx {

using sig_t = std::int32_t;
using coef_t = std::int32_t;
using acc_t = std::int64_t;

inline constexpr int kSigFrac = 20;
inline constexpr int kCoefFrac = 24;
inline constexpr int kCodeFrac = 15;
inline constexpr sig_t kSigOne = sig_t{1} << kSigFrac;
inline constexpr coef_t kCoefOne = coef_t{1} << kCoefFrac;
inline constexpr sig_t kSigMax = std::numeric_limits<sig_t>::max();
inline constexpr sig_t kSigMin = std::numeric_limits<sig_t>::min();

constexpr sig_t sat32(acc_t v) {
    return v > kSigMax ? kSigMax : v < kSigMin ? kSigMin : static_cast<sig_t>(v);
}

// Round-half-up then arithmetic shift; C++20 guarantees >> on negative values is arithmetic.
constexpr acc_t round_shift(acc_t acc, int shift) {
    return (acc + (acc_t{1} << (shift - 1))) >> shift;
}

constexpr sig_t rnd_sc(acc_t acc) { return sat32(round_shift(acc, kCoefFrac)); }
constexpr sig_t rnd_ss(acc_t acc) { return sat32(round_shift(acc, kSigFrac)); }

constexpr sig_t add(sig_t a, sig_t b) { return sat32(acc_t{a} + b); }
constexpr sig_t sub(sig_t a, sig_t b) { return sat32(acc_t{a} - b); }
constexpr sig_t neg(sig_t a) { return sat32(-acc_t{a}); }
constexpr sig_t abs(sig_t a) { return sat32(a < 0 ? -acc_t{a} : acc_t{a}); }
constexpr sig_t clamp(sig_t v, sig_t lo, sig_t hi) { return v < lo ? lo : v > hi ? hi : v; }

// signal x coefficient -> signal
constexpr sig_t mul_sc(sig_t s, coef_t c) { return rnd_sc(acc_t{s} * c); }
// signal x signal -> signal
constexpr sig_t mul_ss(sig_t a, sig_t b) { return rnd_ss(acc_t{a} * b); }
// ADC code x full-scale (signal) -> signal
constexpr sig_t mul_code(std::int16_t code, sig_t full_scale) {
    return sat32(round_shift(acc_t{code} * full_scale, kCodeFrac));
}

inline sig_t to_sig(double v) {
    const double q = std::nearbyint(v * kSigOne);
    if (!(q >= kSigMin && q <= kSigMax))
        throw std::range_error("value " + std::to_string(v) + " does not fit Q12.20");
    return static_cast<sig_t>(q);
}

inline coef_t to_coef(double v) {
    const double q = std::nearbyint(v * kCoefOne);
    if (!(q >= std::numeric_limits<coef_t>::min() && q <= std::numeric_limits<coef_t>::max()))
        throw std::range_error("coefficient " + std::to_string(v) + " does not fit Q8.24");
    return static_cast<coef_t>(q);
}

constexpr double from_sig(sig_t v) { return static_cast<double>(v) / kSigOne; }
constexpr double from_coef(coef_t v) { return static_cast<double>(v) / kCoefOne; }

}  // namespace apt::fx
