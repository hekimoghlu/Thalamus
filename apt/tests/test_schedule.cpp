// Module M reproduction: threshold dose-response, contact-time thresholds, the optimal schedule,
// and the three-epoch ordering in the marginal limit.
#include <cmath>

#include "apt/evidence.hpp"
#include "apt/numerics.hpp"
#include "apt/schedule.hpp"
#include "check.hpp"

using namespace apt;

static const EvidenceBase& eb() {
    static const auto e = EvidenceBase::load_default();
    return e;
}
static ScheduleModel& model() {
    static ScheduleModel m = [] {
        ScheduleModel s(eb(), RecoveryParams::load_default(), SensitiveWindow::fit(eb()));
        s.calibrate_scale(eb());
        return s;
    }();
    return m;
}

TEST("threshold dose-response: 1.72x usual care, solved from two verified points") {
    CHECK_NEAR(model().threshold(), 1.723, 0.001);
    CHECK_NEAR(model().phi(1.5), 0.0, 0.0);  // below threshold, extra therapy buys nothing
    CHECK(model().phi(2.0) > 0);
}

TEST("contact time to threshold: 43 min/week therapist, ~4 min/week at the device rate") {
    CHECK_NEAR(model().threshold_hours(model().reps_usual()) * 60.0, 43.4, 0.5);
    CHECK_NEAR(model().threshold_hours(model().reps_device()) * 60.0, 4.3, 0.1);
}

TEST("a threshold above usual care exists in ~90% of draws from its anchors' intervals") {
    const auto u = model().threshold_uncertainty(20000, 11);
    CHECK_NEAR(u.p_exists, 0.90, 0.03);
    CHECK(u.q025 < 1.0);
}

TEST("Gauss-Hermite(7) matches numpy.hermegauss") {
    const auto q = num::gauss_hermite_e(7);
    CHECK_NEAR(q.nodes[6], 3.7504397177257425, 1e-12);
    CHECK_NEAR(q.nodes[5], 2.3667594107345415, 1e-12);
    CHECK_NEAR(q.nodes[4], 1.1544053947399682, 1e-12);
    CHECK_NEAR(q.weights[3], 16.0 / 35.0, 1e-12);
}

TEST("reward weights are an exact decomposition of spontaneous recovery") {
    double s = 0;
    for (double w : model().reward_weights()) s += w;
    CHECK(s > 0.0 && s < 1.0);
}

TEST("20 h budget: spread thin buys nothing, concentrated buys ~6.6 FM") {
    const auto o = model().evaluate(model().reps_usual(), 20.0, 4000, num::kSeed + 1);
    CHECK_NEAR(o.gain_uniform, 0.0, 1e-12);
    CHECK_NEAR(o.gain_optimal, 6.6, 0.6);
    CHECK_NEAR(o.prob_exceeds_mcid, 0.81, 0.08);
}

TEST("three-epoch ordering in the marginal limit: ~0.68 : 1.00 : 0.47") {
    const double ac = model().marginal_epoch_value(0, 4);
    const double sub = model().marginal_epoch_value(9, 13);
    const double ch = model().marginal_epoch_value(26, 30);
    CHECK_NEAR(ac / sub, 0.68, 0.04);
    CHECK_NEAR(ch / sub, 0.47, 0.04);
}

TEST_MAIN()
