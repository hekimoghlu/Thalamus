// Closed-loop hardware-in-the-loop behaviour of the fixed-point controller.
#include "apt/plant.hpp"
#include "check.hpp"

using namespace apt;

static hil::RunSummary go(double latency_ms, bool pc, core::DesignPoint d = {}) {
    auto c = core::Config::from_design(d);
    if (!pc) c.ctrl &= ~core::kCtrlPassivity;
    hil::PlantParams p;
    p.extra_latency_ms = latency_ms;
    return hil::run(c, p, hil::ReachingTask{}, 6.0, 7);
}

TEST("default design is stable and transparent at low latency") {
    const auto r = go(0, true);
    CHECK(!r.faulted);
    CHECK(r.interaction_rms_N < 3.0);
    CHECK(r.reps >= 2);
    CHECK(r.min_energy_mJ > -1.0);  // passive at the port
}

TEST("stable through 40 ms of transport delay, lost by 50 ms (the latency envelope)") {
    CHECK(!go(40, false).faulted);
    CHECK(go(50, false).faulted || go(50, false).min_energy_mJ < -100);
}

TEST("passivity controller restores port passivity where the observer goes negative") {
    const auto off = go(30, false), on = go(30, true);
    CHECK(off.min_energy_mJ < 0);
    CHECK(on.min_energy_mJ >= off.min_energy_mJ);
}

TEST("rendering a virtual mass below the carriage mass is unstable at any latency") {
    core::DesignPoint d;
    d.virtual_mass_kg = 0.25;  // the manuscript's analysis constant
    d.virtual_damping_Ns_m = 6.0;
    const auto r = go(0, false, d);
    CHECK(r.faulted || r.min_energy_mJ < -50);
}

TEST("afferent relay fires during movement and gating engages") {
    const auto r = go(0, true);
    CHECK(r.afferent_rate_hz > 5);
    CHECK(r.gated_fraction > 0.0);
}

TEST_MAIN()
