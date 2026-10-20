// apt/neuromorphic.hpp - the afferent return path as a spiking relay.
//
// The device returns the sensed state to the user as timed stimulus events, not as an analogue
// level. Each afferent channel is a leaky integrate-and-fire (LIF) unit with subtractive reset,
// followed by paired-pulse sensory gating: a second event arriving within the verified gating
// interval (30 ms) is delivered at the verified second-response ratio (0.4) - the suppression
// a healthy thalamocortical relay applies and that stroke degrades. The fixed-point step here is
// the one the golden model and the RTL execute.
#pragma once

#include <cstdint>

#include "apt/fixed.hpp"

namespace apt::neuro {

struct LifFixed {
    fx::coef_t leak;   // fraction of membrane lost per tick
    fx::coef_t gain;   // membrane increment per unit input per tick
    fx::sig_t theta;   // threshold (subtracted on a spike)

    // returns true on a spike; v is updated in place
    constexpr bool step(fx::sig_t& v, fx::sig_t input_abs) const {
        v = fx::add(fx::sub(v, fx::mul_sc(v, leak)), fx::mul_sc(input_abs, gain));
        if (theta > 0 && v >= theta) {
            v = fx::sub(v, theta);
            return true;
        }
        return false;
    }
};

struct GateFixed {
    std::uint32_t isi_ticks;   // gating interval
    std::uint32_t ratio_q8;    // second-response ratio, Q0.8
    static constexpr std::uint32_t kSinceMax = 0xFFFF;

    // returns delivered amplitude; `gated` reports whether the event fell inside the interval
    constexpr std::uint8_t on_spike(std::uint32_t& since, std::uint8_t amp, bool& gated) const {
        gated = since < isi_ticks;
        since = 0;
        return gated ? static_cast<std::uint8_t>((std::uint32_t{amp} * ratio_q8) >> 8) : amp;
    }
    static constexpr void idle(std::uint32_t& since) {
        if (since < kSinceMax) ++since;
    }
};

// Continuous-time approximation of the LIF rate code for a constant input u, in events per tick:
// the membrane relaxes toward v* = g u / leak; with reset to ~0 the inter-event interval is
// T = -ln(1 - theta / v*) / leak (and gain * u / theta when leak = 0). Zero below rheobase.
double lif_rate(double leak, double gain, double theta, double u);

// Fraction of events delivered at full amplitude for a regular train at `rate_hz` through a gate
// with interval `isi_ms`: 1 below the gating limit (1 / isi), 0 above it.
double ungated_fraction(double rate_hz, double isi_ms);

}  // namespace apt::neuro
