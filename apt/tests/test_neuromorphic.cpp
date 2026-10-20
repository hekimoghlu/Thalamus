// LIF afferent encoder and paired-pulse sensory gate.
#include "apt/neuromorphic.hpp"
#include "check.hpp"

using namespace apt;
using namespace apt::neuro;

TEST("no leak: rate is exactly gain * u / theta") {
    const LifFixed lif{0, fx::to_coef(0.01), fx::to_sig(1.0)};
    fx::sig_t v = 0;
    int spikes = 0;
    for (int i = 0; i < 100000; ++i) spikes += lif.step(v, fx::to_sig(5.0));  // 0.05 / tick
    CHECK_NEAR(spikes, 5000, 1);
}

TEST("leaky rate matches the analytic transfer function; silent below rheobase") {
    const double leak = 0.01, gain = 0.005, theta = 1.0;
    const LifFixed lif{fx::to_coef(leak), fx::to_coef(gain), fx::to_sig(theta)};
    for (double u : {3.0, 10.0, 40.0}) {
        fx::sig_t v = 0;
        int spikes = 0;
        for (int i = 0; i < 200000; ++i) spikes += lif.step(v, fx::to_sig(u));
        const double analytic = lif_rate(leak, gain, theta, u);
        if (analytic == 0) CHECK(spikes == 0);
        else CHECK_REL(spikes / 200000.0, analytic, 0.08);
    }
    CHECK(lif_rate(leak, gain, theta, 1.9) == 0.0);  // v* = 0.95 < theta
}

TEST("gate: second event inside the verified 30 ms interval is delivered at 0.4") {
    const GateFixed g{30, 102};  // round(0.4 * 256)
    std::uint32_t since = GateFixed::kSinceMax;
    bool gated = false;
    CHECK(g.on_spike(since, 200, gated) == 200);
    CHECK(!gated);
    for (int i = 0; i < 10; ++i) GateFixed::idle(since);
    CHECK(g.on_spike(since, 200, gated) == (200 * 102) >> 8);
    CHECK(gated);
    for (int i = 0; i < 30; ++i) GateFixed::idle(since);
    CHECK(g.on_spike(since, 200, gated) == 200);
    CHECK(!gated);
}

TEST("gating limit: trains faster than 1 / 30 ms lose their full-amplitude events") {
    CHECK(ungated_fraction(20, 30) == 1.0);
    CHECK(ungated_fraction(33, 30) == 1.0);
    CHECK(ungated_fraction(34, 30) == 0.0);
}

TEST_MAIN()
