// Closed-loop digital twin: device design -> link delay -> sessions -> dose -> outcome.
#include "apt/twin.hpp"
#include "check.hpp"

using namespace apt;

struct World {
    EvidenceBase eb = EvidenceBase::load_default();
    ScheduleModel sm{eb, RecoveryParams::load_default(), SensitiveWindow::fit(eb)};
    LatencyLearningLaw law = LatencyLearningLaw::from_evidence(eb);
    core::DesignPoint base{};
    adapt::GainSchedule sched = adapt::certify_schedule(base);
    twin::StabilityMap fixed = twin::StabilityMap::measure(base, "fixed", [](double) { return 1.0; }, 300.0, 300.0, 10.0, 3.0);
    twin::StabilityMap adapt_ = twin::StabilityMap::measure(
        base, "adaptive", [this](double d) { return sched.lookup(d); }, sched.certified_max_ms(), 300.0, 10.0, 3.0);
    World() { sm.calibrate_scale(eb); }
};

static World& world() { static World w; return w; }

TEST("stability maps: fixed loses sessions near 50 ms, the certified policy holds to its certified range") {
    auto& w = world();
    CHECK(w.fixed.completes(0) && w.fixed.completes(30));
    CHECK(!w.fixed.completes(60));
    CHECK(w.fixed.first_failure_ms() >= 40 && w.fixed.first_failure_ms() <= 60);
    CHECK(w.adapt_.completes(120) && w.adapt_.completes(190));
    CHECK(!w.adapt_.completes(260));          // beyond the certified range: refused, not attempted
    CHECK(w.adapt_.first_failure_ms() > w.fixed.first_failure_ms() + 100);
    CHECK(w.adapt_.cmd_follow_mm(120) > w.fixed.cmd_follow_mm(0));  // the cost is visible
}

TEST("with common random numbers, policies agree where the link is healthy") {
    auto& w = world();
    const twin::Twin tw(w.eb, w.sm, w.law);
    twin::LinkScenario link;
    link.median_ms = 8;
    link.spike_prob = 0;
    const auto r = tw.run({&w.fixed, &w.adapt_}, link, {}, 800, 5);
    CHECK_NEAR(r[0].mean_fm_gain, r[1].mean_fm_gain, 0.05);
    CHECK(r[1].completed_fraction >= r[0].completed_fraction);
}

TEST("the adaptive policy never completes fewer sessions or delivers less dose, and the gap grows with delay") {
    auto& w = world();
    const twin::Twin tw(w.eb, w.sm, w.law);
    double prev_gap = -1;
    for (double med : {20.0, 60.0, 120.0}) {
        twin::LinkScenario link;
        link.median_ms = med;
        const auto r = tw.run({&w.fixed, &w.adapt_}, link, {}, 800, 5);
        CHECK(r[1].completed_fraction >= r[0].completed_fraction);
        CHECK(r[1].mean_effective_hours >= r[0].mean_effective_hours);
        CHECK(r[1].mean_fm_gain >= r[0].mean_fm_gain - 1e-9);
        const double gap = r[1].completed_fraction - r[0].completed_fraction;
        CHECK(gap >= prev_gap - 1e-9);
        prev_gap = gap;
    }
}

TEST("dose saturates: losing a minority of sessions barely changes the outcome, losing nearly all does") {
    auto& w = world();
    const twin::Twin tw(w.eb, w.sm, w.law);
    twin::LinkScenario healthy, lossy, dead;
    healthy.median_ms = 10;
    lossy.median_ms = 40;
    dead.median_ms = 250;
    const auto a = tw.run({&w.fixed}, healthy, {}, 800, 5)[0], b = tw.run({&w.fixed}, lossy, {}, 800, 5)[0],
               c = tw.run({&w.fixed}, dead, {}, 800, 5)[0];
    CHECK(b.completed_fraction < a.completed_fraction - 0.2);
    CHECK(a.mean_fm_gain - b.mean_fm_gain < 0.6 * a.mean_fm_gain);
    CHECK(c.mean_fm_gain < 0.2 * a.mean_fm_gain);
    CHECK(c.zero_dose_fraction > 0.8);
}

TEST("slower links lower the per-hour learning efficiency monotonically (the verified latency law)") {
    auto& w = world();
    CHECK(w.law.rate(10) > w.law.rate(60));
    CHECK(w.law.rate(60) > w.law.rate(200));
    CHECK_NEAR(w.law.tau_half_ms(), 152.05, 0.5);
}

TEST_MAIN()
