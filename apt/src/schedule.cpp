#include "apt/schedule.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "apt/numerics.hpp"

namespace apt {

namespace {
constexpr double kRLow = 1.75, kRHigh = 3.00;  // midpoints of the verified dose bands (+50-100%, >=+200%)
constexpr double kZ95 = 1.959964;
}  // namespace

ScheduleModel::ScheduleModel(const EvidenceBase& eb, const RecoveryParams& par,
                             const SensitiveWindow& win) {
    dose_low_ = eb.get("dose_arat_50_100pct");
    dose_low_lo_ = eb.get("dose_arat_50_100pct", Field::Lo);
    dose_low_hi_ = eb.get("dose_arat_50_100pct", Field::Hi);
    dose_high_ = eb.get("dose_arat_ge200pct");
    dose_high_lo_ = eb.get("dose_arat_ge200pct", Field::Lo);
    dose_high_hi_ = eb.get("dose_arat_ge200pct", Field::Hi);
    r_thr_ = (kRHigh * dose_low_ - kRLow * dose_high_) / (dose_low_ - dose_high_);
    a_dose_ = dose_low_ / (kRLow - r_thr_);

    reps_usual_ = eb.get("reps_usual_care");
    mcid_ = eb.get("fmue_mcid");
    reps_device_ = eb.get("reps_high_dose_protocol");
    const double prr = eb.get("prr_slope_original");
    c0_ = prr;  // spontaneous ceiling on the impairment scale
    fm_scale_ = eb.get("fm_total_recovery_units") / prr;
    sigma_w_ = (eb.get("vns_fm_sd_control") / fm_scale_) / std::sqrt(13.0);

    // Responsiveness: Module D's inverted-U window, scaled by the verified subacute gradient and
    // normalised to its maximum over the year (the scale cancels; only the shape is used).
    resp_.resize(kWeeks);
    for (int t = 0; t < kWeeks; ++t) resp_[t] = win.responsiveness(double(t));
    const double gmax = *std::max_element(resp_.begin(), resp_.end());
    for (auto& r : resp_) r /= gmax;

    // Exact running-reward weights from the Markov form of Module D's spontaneous recovery.
    std::vector<double> r(kWeeks + 1), h(kWeeks);
    for (int t = 0; t <= kWeeks; ++t) r[t] = remaining_spontaneous(double(t), par);
    for (int t = 0; t < kWeeks; ++t)
        h[t] = std::clamp((r[t] - r[t + 1]) / std::max(r[t], 1e-12), 0.0, 0.999);
    w_reward_.assign(kWeeks, 0.0);
    double tail = 1.0;
    for (int t = kWeeks - 1; t >= 0; --t) {
        w_reward_[t] = h[t] * tail;
        tail *= 1.0 - h[t];
    }
    const auto q = num::gauss_hermite_e(7);
    qz_ = q.nodes;
    qw_ = q.weights;
}

double ScheduleModel::phi(double R) const {
    return phi_scale_ * a_dose_ * std::max(0.0, R - r_thr_);
}

double ScheduleModel::threshold_hours(double reps_per_hour) const {
    return (r_thr_ - 1.0) * reps_usual_ / reps_per_hour;
}

ScheduleModel::ThresholdUncertainty ScheduleModel::threshold_uncertainty(int draws,
                                                                         std::uint64_t seed) const {
    num::Rng rng(seed);
    const double se_lo = (dose_low_hi_ - dose_low_lo_) / (2 * kZ95);
    const double se_hi = (dose_high_hi_ - dose_high_lo_) / (2 * kZ95);
    std::vector<double> thr;
    int exists = 0;
    for (int i = 0; i < draws; ++i) {
        const double yl = rng.normal(dose_low_, se_lo), yh = rng.normal(dose_high_, se_hi);
        if (yh <= yl) continue;  // no increasing response through these two points
        const double t = (kRHigh * yl - kRLow * yh) / (yl - yh);
        thr.push_back(t);
        exists += t > 1.0;
    }
    return {double(exists) / draws, num::quantile(thr, 0.025), num::quantile(thr, 0.975)};
}

double ScheduleModel::ceiling_step(double c, int t, double hours, double reps_per_hour) const {
    const double rel = 1.0 + hours * reps_per_hour / reps_usual_;
    return 1.0 - (1.0 - c) * std::exp(-resp_[t] * phi(rel));
}

ScheduleModel::Solution ScheduleModel::solve(double reps, double budget) const {
    Solution s;
    s.C = num::linspace(c0_, 1.0, kNc);
    s.B = num::linspace(0.0, budget, kNb);
    s.reps_per_hour = reps;
    s.budget = budget;
    s.V.assign(kNc * kNb, 0.0);
    s.policy.assign(std::size_t(kWeeks) * kNc * kNb, 0.f);
    const double dc = s.C[1] - s.C[0];
    const double db = std::max(s.B[1] - s.B[0], 1e-12);
    const int nh = int(hours_.size());
    std::vector<double> Vn(kNc * kNb);
    std::vector<double> cn(kNc * nh);
    for (int t = kWeeks - 1; t >= 0; --t) {
        for (int ic = 0; ic < kNc; ++ic)
            for (int a = 0; a < nh; ++a) cn[ic * nh + a] = ceiling_step(s.C[ic], t, hours_[a], reps);
        for (int ic = 0; ic < kNc; ++ic) {
            for (int ib = 0; ib < kNb; ++ib) {
                double best = -std::numeric_limits<double>::infinity();
                int best_a = 0;
                for (int a = 0; a < nh; ++a) {
                    const double bn_raw = s.B[ib] - hours_[a];
                    if (bn_raw < -1e-9) continue;
                    const double bn = std::clamp(bn_raw, 0.0, budget);
                    auto jb = num::clip_index(num::searchsorted(s.B, bn) - 1, 0, kNb - 2);
                    const double fb = (bn - s.B[jb]) / db;
                    double ev = 0;
                    for (std::size_t q = 0; q < qz_.size(); ++q) {
                        const double cz = std::clamp(cn[ic * nh + a] + sigma_w_ * qz_[q], c0_, 1.0);
                        auto jc = num::clip_index(num::searchsorted(s.C, cz) - 1, 0, kNc - 2);
                        const double fc = (cz - s.C[jc]) / dc;
                        const double* V = s.V.data();
                        ev += qw_[q] * ((1 - fc) * (1 - fb) * V[jc * kNb + jb] +
                                        fc * (1 - fb) * V[(jc + 1) * kNb + jb] +
                                        (1 - fc) * fb * V[jc * kNb + jb + 1] +
                                        fc * fb * V[(jc + 1) * kNb + jb + 1]);
                    }
                    const double val = w_reward_[t] * cn[ic * nh + a] + ev;
                    if (val > best) best = val, best_a = a;  // first maximum, as numpy.argmax
                }
                Vn[ic * kNb + ib] = best;
                s.policy[(std::size_t(t) * kNc + ic) * kNb + ib] = float(hours_[best_a]);
            }
        }
        s.V.swap(Vn);
    }
    return s;
}

ScheduleModel::Rollout ScheduleModel::rollout(const Solution& s, int n, std::uint64_t seed,
                                              double fixed) const {
    num::Rng rng(seed);
    Rollout r;
    std::vector<double> c(n, c0_), b(n, s.budget), x(n, 0.0);
    r.hours_used.assign(n, 0.0);
    r.schedule.assign(kWeeks, std::vector<double>(n, 0.0));
    for (int t = 0; t < kWeeks; ++t) {
        for (int i = 0; i < n; ++i) {
            double h;
            if (fixed < 0) {
                const auto ic = num::clip_index(num::searchsorted(s.C, c[i]) - 1, 0, kNc - 1);
                const auto ib = num::clip_index(num::searchsorted(s.B, b[i]) - 1, 0, kNb - 1);
                h = s.hours_at(t, int(ic), int(ib));
            } else {
                h = fixed;
            }
            h = std::min(h, b[i]);
            // therapy acts on the same week's recovery: raise the ceiling, then accrue reward
            c[i] = std::clamp(ceiling_step(c[i], t, h, s.reps_per_hour) + sigma_w_ * rng.normal(),
                              c0_, 1.0);
            x[i] += w_reward_[t] * c[i];
            b[i] -= h;
            r.hours_used[i] += h;
            r.schedule[t][i] = h;
        }
    }
    for (auto& v : x) v = std::clamp(v, 0.0, 1.0);
    r.final_capacity = std::move(x);
    return r;
}

ScheduleModel::Outcome ScheduleModel::evaluate(double reps, double budget, int n,
                                               std::uint64_t seed) const {
    const auto sol = solve(reps, budget);
    const auto opt = rollout(sol, n, seed);
    const auto uni = rollout(sol, n, seed, budget / kWeeks);
    const auto none = rollout(sol, n, seed, 0.0);
    Outcome o{};
    std::vector<double> first, last;
    for (int i = 0; i < n; ++i) {
        const double g = (opt.final_capacity[i] - none.final_capacity[i]) * fm_scale_;
        o.gain_optimal += g;
        o.gain_uniform += (uni.final_capacity[i] - none.final_capacity[i]) * fm_scale_;
        o.prob_exceeds_mcid += g >= mcid_;
        o.hours_used += opt.hours_used[i];
        int f = kWeeks, l = 0, cnt = 0;
        for (int t = 0; t < kWeeks; ++t)
            if (opt.schedule[t][i] > 0) f = std::min(f, t), l = t, ++cnt;
        o.weeks_treated += cnt;
        first.push_back(f);
        last.push_back(l);
    }
    o.gain_optimal /= n;
    o.gain_uniform /= n;
    o.prob_exceeds_mcid /= n;
    o.hours_used /= n;
    o.weeks_treated /= n;
    o.median_first_week = num::quantile(first, 0.5);
    o.median_last_week = num::quantile(last, 0.5);
    return o;
}

double ScheduleModel::marginal_epoch_value(int a, int b) const {
    double v = 0;
    for (int t = a; t < b; ++t) {
        double remain = 0;
        for (int s = t; s < kWeeks; ++s) remain += w_reward_[s];
        v += resp_[t] * remain;
    }
    return v;
}

void ScheduleModel::calibrate_scale(const EvidenceBase& eb) {
    const double target = eb.get("dose_fm_gain_increased_duration");
    constexpr double kHoursPerWeek = 2.0;  // the verified doubled-duration protocol
    auto gain = [&](double scale) {
        phi_scale_ = scale;
        const auto s = solve(reps_usual_, kHoursPerWeek * kWeeks);
        const auto r = rollout(s, 1500, num::kSeed + 3, kHoursPerWeek);
        const auto base = rollout(s, 1500, num::kSeed + 3, 0.0);
        double g = 0;
        for (int i = 0; i < 1500; ++i) g += (r.final_capacity[i] - base.final_capacity[i]) * fm_scale_;
        return g / 1500;
    };
    phi_scale_ = num::bisect(gain, target, 1e-9, 20.0, 40, /*geometric=*/true);
}

}  // namespace apt
