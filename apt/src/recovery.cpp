#include "apt/recovery.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include "apt/io.hpp"
#include "apt/numerics.hpp"

namespace apt {

RecoveryParams RecoveryParams::load(const std::filesystem::path& p) {
    const auto j = io::read_json_numbers(p);
    auto at = [&](const char* k) {
        auto it = j.find(k);
        if (it == j.end()) throw std::runtime_error(std::string("missing ") + k + " in " + p.string());
        return it->second;
    };
    return {at("recovery_parameters.eta_fast"), at("recovery_parameters.tau_fast"),
            at("recovery_parameters.eta_slow"), at("recovery_parameters.tau_slow"),
            at("recovery_parameters.floor"),    at("plasticity_floor_ratio")};
}

RecoveryParams RecoveryParams::load_default() {
    return load(EvidenceBase::default_data_dir() / "reference" / "D_recovery_summary.json");
}

double cumulative_drive(double t, const RecoveryParams& p, double scale) {
    return scale * (p.eta_fast * p.tau_fast * (1 - std::exp(-t / p.tau_fast)) +
                    p.eta_slow * p.tau_slow * (1 - std::exp(-t / p.tau_slow)) + p.floor * t);
}

double remaining_spontaneous(double t, const RecoveryParams& p) {
    const double w = p.eta_fast / (p.eta_fast + p.eta_slow);
    return w * std::exp(-t / p.tau_fast) + (1.0 - w) * std::exp(-t / p.tau_slow);
}

// ------------------------------------------------------------------------- sensitive period
namespace {

double win_shape(double t, double to, double tc) {
    return (1.0 - std::exp(-t / to)) * std::exp(-t / tc);
}

// sum of squared log-ratio residuals, each series normalised at its index-1 point
double series_loss(const std::array<double, 3>& t, const std::array<double, 3>& y, double to,
                   double tc) {
    std::array<double, 3> m;
    for (int i = 0; i < 3; ++i) m[i] = win_shape(t[i], to, tc);
    if (*std::max_element(m.begin(), m.end()) <= 0) return 1e6;
    double loss = 0;
    for (int i = 0; i < 3; ++i) {
        const double r = std::log(m[i] / m[1]) - std::log(y[i] / y[1]);
        loss += r * r;
    }
    return loss;
}

bool in_support(double to, double tc) { return 0.2 < to && to < 60 && 5 < tc && tc < 400; }

}  // namespace

SensitiveWindow::SensitiveWindow(double to, double tc) : tau_open_(to), tau_close_(tc) {
    // peak located on the same grid Module D uses (linspace(0.1, 120, 3000))
    const auto grid = num::linspace(0.1, 120.0, 3000);
    double best = -1;
    for (double t : grid)
        if (const double v = win_shape(t, to, tc); v > best) best = v, peak_week_ = t;
    peak_value_ = best;
}

double SensitiveWindow::shape(double t) const { return win_shape(t, tau_open_, tau_close_); }

SensitiveWindow SensitiveWindow::fit(const EvidenceBase& eb) {
    const std::array<double, 3> cp_t{3.0, 10.7, 26.0};  // epoch mid-points of the verified trial
    const std::array<double, 3> cp_y{eb.get("cpass_arat_acute"), eb.get("cpass_arat_subacute"),
                                     eb.get("cpass_arat_chronic")};
    const std::array<double, 3> gr_t{10.3, 52.0, 100.0};
    const std::array<double, 3> gr_y{eb.get("window_gradient_subacute"),
                                     eb.get("window_gradient_early_chronic"),
                                     eb.get("window_gradient_late_chronic")};
    auto joint = [&](const std::vector<double>& p) {
        const double to = std::exp(p[0]), tc = std::exp(p[1]);
        if (!in_support(to, tc)) return 1e6;
        return series_loss(cp_t, cp_y, to, tc) + series_loss(gr_t, gr_y, to, tc);
    };
    const auto r = num::nelder_mead(joint, {std::log(5.0), std::log(60.0)}, 4000, 1e-6, 1e-9);
    SensitiveWindow w(std::exp(r.x[0]), std::exp(r.x[1]));
    w.loss_ = r.fx;
    auto single = [&](const auto& t, const auto& y) {
        auto f = [&](const std::vector<double>& p) {
            const double to = std::exp(p[0]), tc = std::exp(p[1]);
            return in_support(to, tc) ? series_loss(t, y, to, tc) : 1e6;
        };
        return std::exp(num::nelder_mead(f, {std::log(5.0), std::log(60.0)}, 4000, 1e-4, 1e-4).x[1]);
    };
    w.tau_close_epoch_only = single(cp_t, cp_y);
    w.tau_close_gradients_only = single(gr_t, gr_y);
    return w;
}

// ---------------------------------------------------------------------------------- cohort
Cohort Cohort::draw(std::size_t n, std::uint64_t seed, double fm_mu, double fm_sd, double cap_mu,
                    double cap_sd) {
    num::Rng rng(seed);
    Cohort c;
    c.fm0.resize(n);
    c.capacity.resize(n);
    for (auto& v : c.fm0) v = std::clamp(rng.normal(fm_mu, fm_sd), 4.0, 58.0);
    for (auto& v : c.capacity) v = std::max(rng.normal(cap_mu, cap_sd), 0.02);
    return c;
}

// ------------------------------------------------------------------------- device design
DeviceDesignModel::DeviceDesignModel(const EvidenceBase& eb, RecoveryParams par,
                                     std::uint64_t seed)
    : law_(LatencyLearningLaw::from_evidence(eb)),
      par_(par),
      cohort_(Cohort::draw(3000, seed)),
      calib_(Cohort::draw(3000, seed + 1)) {
    const double sd_t = eb.get("vns_fm_sd_treatment"), sd_c = eb.get("vns_fm_sd_control");
    sd_pool_ = std::sqrt((sd_t * sd_t + sd_c * sd_c) / 2.0);
    mdc_ = eb.get("fmue_mdc");
    reps_usual_ = eb.get("reps_usual_care");
    reps_high_ = eb.get("reps_high_dose_protocol");
    dose_high_ = reps_high_ / reps_usual_;
    // Calibrate the dose exponent against the one verified high-repetition observation
    // (322 repetitions per session for six weeks -> +8 ARAT), on a separate calibration cohort.
    const double arat_to_fm = STRUCTURAL("arat_to_fm_scale", kFmMax / 57.0, "FM/ARAT",
                                         "scale-range approximation, not a validated conversion");
    const double target = eb.get("reps_high_dose_arat_gain") * arat_to_fm;
    double lo = 0.01, hi = 3.0;
    for (int i = 0; i < 60; ++i) {
        const double mid = 0.5 * (lo + hi);
        (mean_gain(calib_, 6.0, std::pow(dose_high_, mid)) > target ? hi : lo) = mid;
    }
    gamma_ = 0.5 * (lo + hi);
    base_gain_ = mean_gain(cohort_, kProgrammeWeeks, 1.0);
}

double DeviceDesignModel::fm_gain(double fm0, double cap, double weeks, double scale) const {
    const double d0 = cumulative_drive(kChronicStartWeek, par_, 1.0);
    const double d1 = d0 + scale * (cumulative_drive(kChronicStartWeek + weeks, par_, 1.0) - d0);
    return (kFmMax - fm0) * (std::exp(-cap * d0) - std::exp(-cap * d1));
}

double DeviceDesignModel::mean_gain(const Cohort& c, double weeks, double scale) const {
    double s = 0;
    for (std::size_t i = 0; i < c.fm0.size(); ++i) s += fm_gain(c.fm0[i], c.capacity[i], weeks, scale);
    return s / double(c.fm0.size());
}

double DeviceDesignModel::dose_scale(double dose) const { return std::pow(dose, gamma_); }

std::vector<DeviceDesignModel::EnvelopePoint> DeviceDesignModel::design_envelope(
    const std::vector<double>& lat, const std::vector<double>& dose) const {
    std::vector<EnvelopePoint> out;
    const std::size_t n = cohort_.fm0.size();
    for (double L : lat) {
        const double lr = law_.relative(L, 60.0);
        for (double D : dose) {
            double sum = 0, over = 0;
            std::size_t exceed = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const double g = fm_gain(cohort_.fm0[i], cohort_.capacity[i], kProgrammeWeeks,
                                         lr * dose_scale(D));
                const double b = fm_gain(cohort_.fm0[i], cohort_.capacity[i], kProgrammeWeeks, 1.0);
                sum += g;
                over += g - b;
                exceed += (g - b) > mdc_;
            }
            out.push_back({L, D, sum / double(n), over / double(n), 100.0 * double(exceed) / double(n)});
        }
    }
    return out;
}

std::vector<DeviceDesignModel::DoseForMcid> DeviceDesignModel::dose_for_mcid(
    const std::vector<double>& lat) const {
    std::vector<DoseForMcid> out;
    for (double L : lat) {
        const double lr = law_.relative(L, 60.0);
        double dose = 1.0;
        for (int i = 0; i < 200; ++i) {
            if (mean_gain(cohort_, kProgrammeWeeks, lr * dose_scale(dose)) - base_gain_ >= mdc_) break;
            dose *= 1.08;
        }
        out.push_back({L, dose, dose * reps_usual_, dose * reps_usual_ > reps_high_});
    }
    return out;
}

DeviceDesignModel::Contrast DeviceDesignModel::latency_contrast(double L, double ref,
                                                                double dose) const {
    const double lr = law_.relative(L, ref);
    const double diff = mean_gain(cohort_, kProgrammeWeeks, lr * dose_scale(dose)) -
                        mean_gain(cohort_, kProgrammeWeeks, dose_scale(dose));
    return {L, ref, diff, diff / sd_pool_};
}

DeviceDesignModel::DoseContrast DeviceDesignModel::dose_contrast(double L, double ref,
                                                                 double dose) const {
    const double lr = law_.relative(L, ref);
    const double diff = mean_gain(cohort_, kProgrammeWeeks, lr * dose_scale(dose)) -
                        mean_gain(cohort_, kProgrammeWeeks, dose_scale(1.0));
    return {L, diff, diff / sd_pool_};
}

DeviceDesignModel::LeverRatio DeviceDesignModel::lever_ratio(double lat_slow, double lat_fast,
                                                             double dose_lo, double dose_hi,
                                                             double ref) const {
    // latency lever: slow -> fast loop at the high dose; dose lever: low -> high dose at the
    // fast loop. The ratio is a property of the two excursions chosen (and, through the
    // saturating recovery law, weakly of the reference), which is why the report sweeps them
    // rather than quoting one pair.
    auto g = [&](double L, double D) {
        return mean_gain(cohort_, kProgrammeWeeks, law_.relative(L, ref) * dose_scale(D));
    };
    const double lat_lever = g(lat_fast, dose_hi) - g(lat_slow, dose_hi);
    const double dose_lever = g(lat_fast, dose_hi) - g(lat_fast, dose_lo);
    return {lat_lever, dose_lever, dose_lever / lat_lever};
}

}  // namespace apt
