// Module D / E reproduction: sensitive period, dose exponent, design envelope, effect sizes.
#include <cmath>

#include "apt/evidence.hpp"
#include "apt/io.hpp"
#include "apt/numerics.hpp"
#include "apt/recovery.hpp"
#include "check.hpp"

using namespace apt;

static const EvidenceBase& eb() {
    static const auto e = EvidenceBase::load_default();
    return e;
}
static const DeviceDesignModel& model() {
    static const DeviceDesignModel m(eb(), RecoveryParams::load_default(), num::kSeed + 5);
    return m;
}
static std::vector<std::vector<std::string>> ref(const char* name) {
    return read_csv(EvidenceBase::default_data_dir() / "reference" / name);
}

TEST("sensitive period: inverted U opening at ~2.4 w, closing at ~57 w, peaking near week 8") {
    const auto w = SensitiveWindow::fit(eb());
    CHECK_NEAR(w.tau_open(), 2.4, 0.1);
    CHECK_NEAR(w.tau_close(), 57.0, 1.5);
    CHECK_NEAR(w.peak_week(), 7.7, 0.2);
    // the two constraining series disagree sevenfold; both single-series fits are reported
    CHECK_NEAR(w.tau_close_epoch_only, 9.9, 0.5);
    CHECK_NEAR(w.tau_close_gradients_only, 67.5, 2.0);
    // the whole shipped curve, week by week
    const auto r = ref("D_sensitive_period.csv");
    double worst = 0;
    for (std::size_t i = 1; i < r.size(); ++i)
        worst = std::max(worst, std::abs(w.responsiveness(std::stod(r[i][0])) - std::stod(r[i][1])));
    CHECK(worst < 5e-3);
}

TEST("dose exponent calibrated on 322 reps/session -> +8 ARAT") {
    const auto j = io::read_json_numbers(EvidenceBase::default_data_dir() / "reference" / "E_control_summary.json");
    CHECK_NEAR(model().gamma(), j.at("dose_exponent_gamma"), 0.03);  // Monte-Carlo cohort
    CHECK_REL(model().pooled_sd(), j.at("pooled_sd_used_for_smd"), 1e-12);
}

TEST("dose needed for the MDC at each latency (E_dose_for_mcid.csv)") {
    const auto r = ref("E_dose_for_mcid.csv");
    std::vector<double> lat;
    for (std::size_t i = 1; i < r.size(); ++i) lat.push_back(std::stod(r[i][0]));
    const auto got = model().dose_for_mcid(lat);
    for (std::size_t i = 0; i < got.size(); ++i) {
        // the search walks a 1.08x ladder, so agreement within one rung is exact agreement
        CHECK(std::abs(std::log(got[i].dose_multiple / std::stod(r[i + 1][1]))) <= std::log(1.08) + 1e-9);
    }
    // ~4x usual care at 20 ms, ~5.9x at 110 ms; every latency stays at or below the
    // published 322-rep protocol (the 300 ms row sits on that boundary)
    CHECK_NEAR(got[2].dose_multiple, 4.0, 0.35);
    CHECK_NEAR(got[6].dose_multiple, 5.9, 0.5);
}

TEST("matched-dose latency contrast (reference 60 ms) and dose contrast (E_predicted_effect_sizes)") {
    const auto r = ref("E_predicted_effect_sizes.csv");
    for (std::size_t i = 1; i < r.size(); ++i) {
        const double L = std::stod(r[i][0]);
        const auto c = model().latency_contrast(L, 60.0, 4.0);
        const auto d = model().dose_contrast(L, 60.0, model().high_dose_multiple());
        CHECK_NEAR(c.smd, std::stod(r[i][2]), 0.02);
        CHECK_NEAR(d.smd, std::stod(r[i][4]), 0.06);
    }
    // the contrast crosses zero exactly at its own reference - the reason no single
    // device-versus-therapy number is reported
    CHECK_NEAR(model().latency_contrast(60.0, 60.0, 4.0).smd, 0.0, 1e-12);
    CHECK_NEAR(model().latency_contrast(150.0, 150.0, 4.0).smd, 0.0, 1e-12);
}

TEST("the 10 ms contrast slides with the reference latency (Section 4.6: 0.12 to 1.08)") {
    const double lo = model().latency_contrast(10, 30, 4.0).smd;
    const double hi = model().latency_contrast(10, 300, 4.0).smd;
    CHECK_NEAR(lo, 0.12, 0.03);
    CHECK_NEAR(hi, 1.08, 0.08);
}

TEST("which lever wins depends on the excursion compared (Section 4.6)") {
    const double dh = model().high_dose_multiple();
    // usual care -> 322-rep protocol vs 220 ms -> 10 ms: dose wins by ~2.6-2.7x
    const auto a = model().lever_ratio(220, 10, 1.0, dh, 60);
    CHECK(a.ratio > 2.4 && a.ratio < 2.9);
    // at the doubled-duration contrast, the only dose excursion the evidence base measures,
    // the ratio falls below one and latency is the larger lever - reported because it favours
    // the declared interest
    const auto c = model().lever_ratio(220, 10, 1.0, 2.0, 60);
    CHECK(c.ratio < 1.0);
    CHECK(c.ratio > 0.4);
}

TEST_MAIN()
