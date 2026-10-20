// Front-end filters, group delay, the fixed-point biquad and the latency budget.
#include <algorithm>
#include <cmath>
#include <numbers>

#include "apt/dsp.hpp"
#include "check.hpp"

using namespace apt;
using namespace apt::dsp;

TEST("Butterworth: -3 dB at cut-off, unity at DC, monotone roll-off") {
    for (int n : {1, 2, 3, 4, 6}) {
        const auto c = butterworth_lowpass(n, 80, 1000);
        CHECK_NEAR(c.magnitude_db(0.0), 0.0, 1e-9);
        CHECK_NEAR(c.magnitude_db(80.0), -3.0103, 1e-3);
        CHECK(c.magnitude_db(200.0) < c.magnitude_db(100.0));
        for (const auto& s : c.sections) CHECK(s.stable());
    }
}

TEST("second-order Butterworth DC group delay ~ sqrt(2) / (2 pi fc)") {
    const auto c = butterworth_lowpass(2, 20, 10000);  // fc << fs: analogue limit
    CHECK_REL(c.group_delay_ms(0.0), 1000.0 * std::sqrt(2.0) / (2 * std::numbers::pi * 20), 0.01);
}

TEST("group delay grows with order and with lower cut-off: the trade the budget constrains") {
    const double d2 = butterworth_lowpass(2, 80, 1000).max_group_delay_ms(30);
    const double d4 = butterworth_lowpass(4, 80, 1000).max_group_delay_ms(30);
    const double d2lo = butterworth_lowpass(2, 40, 1000).max_group_delay_ms(30);
    CHECK(d4 > d2);
    CHECK(d2lo > d2);
}

TEST("notch removes mains, passes DC") {
    const auto n = notch(50, 5, 1000);
    CHECK(std::abs(n.response(50, 1000)) < 1e-9);
    CHECK_NEAR(std::abs(n.response(0, 1000)), 1.0, 1e-12);
}

TEST("fixed-point biquad tracks the double-precision filter") {
    const auto bq = butterworth_lowpass(2, 80, 1000).sections[0];
    FixedBiquad fq(FixedBiquadCoefs::quantise(bq));
    double x1 = 0, x2 = 0, y1 = 0, y2 = 0, worst = 0;
    for (int i = 0; i < 3000; ++i) {
        const double x = 40.0 * std::sin(2 * std::numbers::pi * 7 * i / 1000.0) + 5.0 * ((i / 97) % 2);
        const double y = bq.b0 * x + bq.b1 * x1 + bq.b2 * x2 - bq.a1 * y1 - bq.a2 * y2;
        x2 = x1, x1 = x, y2 = y1, y1 = y;
        worst = std::max(worst, std::abs(fx::from_sig(fq.step(fx::to_sig(x))) - y));
    }
    CHECK(worst < 1e-4);  // coefficient quantisation, not rounding, dominates
}

TEST("electronics are negligible; the filter dominates the 10 ms afferent budget") {
    ElectronicsTiming t;
    const auto fe = butterworth_lowpass(2, 80, 1000);
    const auto b = afferent_budget(fe, 30, t, 0.0);
    CHECK(b.total_ms() < 10.0);
    CHECK(b.stages[2].ms < 0.01);  // datapath compute: microseconds
    CHECK(b.stages[3].ms > b.stages[1].ms);
}

TEST("Pareto front exists and is monotone: more rejection costs more delay") {
    auto all = filter_tradeoff(1000, 30, 150, 6);
    std::vector<FilterCandidate> front;
    for (auto& c : all)
        if (c.pareto) front.push_back(c);
    CHECK(front.size() >= 3);
    std::sort(front.begin(), front.end(), [](auto& a, auto& b) { return a.group_delay_ms < b.group_delay_ms; });
    for (std::size_t i = 1; i < front.size(); ++i)
        CHECK(front[i].reject_artefact_db >= front[i - 1].reject_artefact_db);
}

TEST_MAIN()
