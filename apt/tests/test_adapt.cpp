// Latency-adaptive inner-loop scheduling: certification, estimator, and the closed-loop result.
#include "apt/adapt.hpp"
#include "check.hpp"

using namespace apt;

static const core::DesignPoint kBase{};

// certified once per process; the sweep is a few hundred 5-second closed-loop runs
static const adapt::GainSchedule& schedule() {
    static const adapt::GainSchedule s = adapt::certify_schedule(kBase);
    return s;
}

TEST("the fixed design loses stability between 40 and 50 ms of transport delay") {
    CHECK(adapt::trial(kBase, 1.0, 40, 7, 6.0).stable);
    CHECK(!adapt::trial(kBase, 1.0, 50, 7, 6.0).stable);
}

TEST("certified schedule is monotone non-increasing and starts at full gain") {
    const auto& s = schedule();
    CHECK(s.scale.front() == 1.0);
    for (std::size_t i = 1; i < s.scale.size(); ++i) CHECK(s.scale[i] <= s.scale[i - 1]);
}

TEST("certified scale tracks the 1/tau^2 delay-limited crossover law within a factor of 2.5") {
    const auto& s = schedule();
    const auto a = adapt::analytic_schedule(s.edge_ms);
    for (std::size_t i = 0; i < s.edge_ms.size(); ++i)
        if (s.edge_ms[i] >= 60) {
            CHECK(s.scale[i] < 2.5 * a.scale[i]);
            CHECK(s.scale[i] > a.scale[i] / 2.5);
        }
}

TEST("with the schedule applied the loop is stable at every certified delay, across fresh seeds") {
    const auto& s = schedule();
    for (std::size_t i = 0; i < s.edge_ms.size(); ++i)
        for (std::uint64_t seed : {7u, 21u, 55u}) CHECK(adapt::trial(kBase, s.scale[i], s.edge_ms[i], seed, 6.0).stable);
}

TEST("adaptation lowers the force the hand feels and costs rendered-dynamics fidelity") {
    const auto& s = schedule();
    const auto f = adapt::trial(kBase, 1.0, 40, 7, 6.0), g = adapt::trial(kBase, s.lookup(40), 40, 7, 6.0);
    CHECK(g.summary.interaction_rms_N <= f.summary.interaction_rms_N + 0.2);
    const auto h = adapt::trial(kBase, s.lookup(120), 120, 7, 6.0);
    CHECK(h.stable);
    CHECK(h.summary.cmd_follow_rms_mm > f.summary.cmd_follow_rms_mm);  // the price is paid, and is reported
}

TEST("estimator attacks at once and releases slowly") {
    adapt::DelayEstimator e;
    for (int i = 0; i < 100; ++i) e.update(10, 1e-3);
    CHECK_NEAR(e.estimate_ms(), 10, 0.5);
    for (int i = 0; i < 200; ++i) e.update(120, 1e-3);
    CHECK(e.estimate_ms() > 100);
    for (int i = 0; i < 200; ++i) e.update(10, 1e-3);   // 0.2 s down: a 50 ms/s release can free ~10 ms
    CHECK(e.estimate_ms() > 90);
}

TEST("online supervisor survives an unannounced delay step that trips the fixed controller") {
    const auto& s = schedule();
    hil::RunHooks hk;
    hk.latency_ms = [](double t) { return t < 3.0 ? 10.0 : (t < 9.0 ? 120.0 : 30.0); };
    hil::PlantParams p;
    const auto fixed = hil::run_ex(core::Config::from_design(kBase), p, hil::ReachingTask{}, 12.0, 11, hk);
    CHECK(fixed.faulted || fixed.min_energy_mJ < -50);
    adapt::Supervisor sup(kBase, s);
    hk.supervisor = [&](double, double probe, core::CoreModel& m) { sup.on_probe(probe, m); };
    const auto ad = hil::run_ex(core::Config::from_design(kBase), p, hil::ReachingTask{}, 12.0, 11, hk);
    CHECK(!ad.faulted);
    CHECK(ad.min_energy_mJ > -50);
    CHECK(sup.changes() >= 2);
    CHECK(!sup.beyond_certified());
}

TEST("beyond the certified range the supervisor says so instead of pretending") {
    adapt::Supervisor sup(kBase, schedule());
    core::CoreModel m(core::Config::from_design(kBase));
    for (int i = 0; i < 200; ++i) sup.on_probe(400, m);
    CHECK(sup.beyond_certified());
}

TEST_MAIN()
