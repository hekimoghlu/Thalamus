// apt/recovery.hpp - Module D (the recovery law and the sensitive period) and Module E (4, 5):
// how the device's two design variables, loop latency and practice dose, propagate into the
// Fugl-Meyer units trials report.
//
// The boundary the manuscript draws is kept in the API: `latency_contrast()` compares two device
// configurations at matched dose and *requires* a reference latency, because no verified value
// fixes the feedback latency of conventional therapy. There is deliberately no function that
// returns a device-versus-therapy effect size.
#pragma once

#include <filesystem>
#include <vector>

#include "apt/afferent.hpp"
#include "apt/evidence.hpp"

namespace apt {

// Module D's calibrated two-timescale law (fitted upstream against the verified two-weekly
// Fugl-Meyer gains and per-week treatment gradients; imported from D_recovery_summary.json).
struct RecoveryParams {
    double eta_fast, tau_fast, eta_slow, tau_slow, floor;
    double plasticity_floor_ratio;
    static RecoveryParams load(const std::filesystem::path& json);
    static RecoveryParams load_default();
};

double cumulative_drive(double t_weeks, const RecoveryParams& p, double scale = 1.0);

// Fraction of the still-unrecovered spontaneous range remaining at week t (Module M's Markov form).
double remaining_spontaneous(double t_weeks, const RecoveryParams& p);

// The inverted-U sensitive period w(t) ~ (1 - e^{-t/tau_open}) e^{-t/tau_close}, fitted jointly
// to the verified three-epoch trial and the three verified per-week treatment gradients, each
// series normalised within itself so only the shape is fitted.
class SensitiveWindow {
public:
    static SensitiveWindow fit(const EvidenceBase& eb);
    SensitiveWindow(double tau_open, double tau_close);

    double shape(double t_weeks) const;
    double responsiveness(double t_weeks) const { return shape(t_weeks) / peak_value_; }
    double tau_open() const { return tau_open_; }
    double tau_close() const { return tau_close_; }
    double peak_week() const { return peak_week_; }
    double fit_loss() const { return loss_; }

    // single-series fits, reported so the disagreement between the two series stays visible
    double tau_close_epoch_only = 0, tau_close_gradients_only = 0;

private:
    double tau_open_, tau_close_, peak_week_ = 0, peak_value_ = 1, loss_ = 0;
};

// A virtual chronic cohort, drawn by one shared constructor (Module E).
struct Cohort {
    std::vector<double> fm0, capacity;
    static Cohort draw(std::size_t n, std::uint64_t seed, double fm_mu = 30.0, double fm_sd = 9.0,
                       double cap_mu = 0.9, double cap_sd = 0.45);
};

class DeviceDesignModel {
public:
    static constexpr double kFmMax = 66.0;
    static constexpr double kChronicStartWeek = 26.0;  // device trials recruit >= 6 months
    static constexpr double kProgrammeWeeks = 8.6;     // 60-day programme

    DeviceDesignModel(const EvidenceBase& eb, RecoveryParams par, std::uint64_t seed);

    // Fugl-Meyer gain accrued over `weeks` of training starting in the chronic phase.
    double fm_gain(double fm0, double cap, double weeks, double scale) const;
    double mean_gain(const Cohort& c, double weeks, double scale) const;
    double dose_scale(double dose_multiple) const;

    double gamma() const { return gamma_; }
    double pooled_sd() const { return sd_pool_; }
    double mdc() const { return mdc_; }
    double high_dose_multiple() const { return dose_high_; }
    const LatencyLearningLaw& learning() const { return law_; }
    const Cohort& cohort() const { return cohort_; }

    struct EnvelopePoint {
        double latency_ms, dose_multiple, mean_gain, gain_over_usual, pct_exceeding_mdc;
    };
    std::vector<EnvelopePoint> design_envelope(const std::vector<double>& lat,
                                               const std::vector<double>& dose) const;

    struct DoseForMcid {
        double latency_ms, dose_multiple, reps_per_session;
        bool exceeds_published_high_dose;
    };
    std::vector<DoseForMcid> dose_for_mcid(const std::vector<double>& lat) const;

    // Matched-dose contrast between a device closing its loop at `latency_ms` and the same
    // device at `reference_ms`. Not a device-versus-therapy effect.
    struct Contrast {
        double latency_ms, reference_ms, fm_difference, smd;
    };
    Contrast latency_contrast(double latency_ms, double reference_ms, double dose_multiple) const;

    struct DoseContrast {
        double latency_ms, fm_difference, smd;
    };
    // high-repetition protocol versus usual care at a given latency (commensurable with trials)
    DoseContrast dose_contrast(double latency_ms, double reference_ms, double dose_multiple) const;

    struct LeverRatio {
        double latency_lever_fm, dose_lever_fm, ratio;
    };
    LeverRatio lever_ratio(double lat_slow, double lat_fast, double dose_lo, double dose_hi,
                           double reference_ms) const;

private:
    LatencyLearningLaw law_;
    RecoveryParams par_;
    Cohort cohort_, calib_;
    double gamma_ = 1.0, sd_pool_, mdc_, reps_usual_, reps_high_, dose_high_, base_gain_;
};

}  // namespace apt
