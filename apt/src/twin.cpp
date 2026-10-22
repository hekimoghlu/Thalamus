#include "apt/twin.hpp"

#include <algorithm>
#include <cmath>

#include "apt/numerics.hpp"

namespace apt::twin {

StabilityMap StabilityMap::measure(const core::DesignPoint& base, std::string name,
                                   const std::function<double(double)>& gain_scale, double certified_ms,
                                   double max_ms, double step_ms, double seconds) {
    StabilityMap m;
    m.name_ = std::move(name);
    m.step_ = step_ms;
    for (double d = 0.0; d <= max_ms + 1e-9; d += step_ms) {
        // a session that the supervisor will refuse (beyond the certified range) never starts
        const double s = gain_scale(d);
        const auto t = adapt::trial(base, s, d, 7, seconds);
        m.stable_.push_back(d <= certified_ms && t.stable);
        m.follow_.push_back(t.summary.cmd_follow_rms_mm);
    }
    return m;
}

bool StabilityMap::completes(double delay_ms) const {
    const double g = std::max(0.0, delay_ms) / step_;
    const auto lo = static_cast<std::size_t>(std::floor(g)), hi = static_cast<std::size_t>(std::ceil(g));
    if (hi >= stable_.size()) return false;
    return stable_[lo] && stable_[hi];
}

double StabilityMap::cmd_follow_mm(double delay_ms) const {
    const double g = std::clamp(delay_ms / step_, 0.0, double(follow_.size() - 1));
    const auto lo = static_cast<std::size_t>(std::floor(g));
    const auto hi = std::min(lo + 1, follow_.size() - 1);
    const double f = g - double(lo);
    return follow_[lo] * (1 - f) + follow_[hi] * f;
}

double StabilityMap::first_failure_ms() const {
    for (std::size_t i = 0; i < stable_.size(); ++i)
        if (!stable_[i]) return double(i) * step_;
    return double(stable_.size()) * step_;
}

std::vector<PolicyOutcome> Twin::run(const std::vector<const StabilityMap*>& policies, const LinkScenario& link,
                                     const Protocol& proto, int patients, std::uint64_t seed) const {
    const int W = ScheduleModel::kWeeks;
    const int S = proto.sessions_per_week;
    const double c0 = m_.ceiling_floor(), sig = m_.sigma_week();
    const double mcid = eb_.get("fmue_mcid");
    const double rph = m_.reps_device();
    const double ref = law_.rate(proto.reference_latency_ms);

    // common random numbers: link draws and recovery noise are drawn once and reused by every policy
    num::Rng rng(seed);
    std::vector<std::vector<double>> delay(patients), noise(patients);
    for (int i = 0; i < patients; ++i) {
        const double med = link.median_ms * std::exp(link.patient_sigma * rng.normal());
        delay[i].reserve(std::size_t(proto.weeks) * S);
        for (int k = 0; k < proto.weeks * S; ++k) {
            double d = med * std::exp(link.session_sigma * rng.normal());
            if (rng.uniform() < link.spike_prob) d *= link.spike_factor;
            delay[i].push_back(d);
        }
        noise[i].resize(W);
        for (int t = 0; t < W; ++t) noise[i][t] = sig * rng.normal();
    }

    auto capacity = [&](const StabilityMap* pol, int i, double* eff_hours, int* done, int* total,
                        double* follow) {
        double c = c0, x = 0.0;
        for (int t = 0; t < W; ++t) {
            double h = 0.0;
            if (pol && t >= proto.start_week && t < proto.start_week + proto.weeks) {
                for (int s = 0; s < S; ++s) {
                    const double d = delay[i][std::size_t(t - proto.start_week) * S + s];
                    ++*total;
                    if (!pol->completes(d)) continue;
                    ++*done;
                    const double tau = d + link.electronics_ms;
                    h += proto.hours_per_session * law_.rate(tau) / ref;
                    *follow += pol->cmd_follow_mm(d);
                }
            }
            *eff_hours += h;
            c = std::clamp(m_.ceiling_step(c, t, h, rph) + noise[i][t], c0, 1.0);
            x += m_.reward_weights()[t] * c;
        }
        return std::clamp(x, 0.0, 1.0);
    };

    std::vector<double> none(patients);
    for (int i = 0; i < patients; ++i) {
        double e = 0, f = 0;
        int a = 0, b = 0;
        none[i] = capacity(nullptr, i, &e, &a, &b, &f);
    }

    std::vector<PolicyOutcome> out;
    for (const auto* pol : policies) {
        PolicyOutcome o{};
        o.policy = pol->name();
        long done_all = 0, total_all = 0;
        double follow_sum = 0, gain = 0, eff = 0;
        int exceed = 0, zero = 0;
        for (int i = 0; i < patients; ++i) {
            double e = 0, f = 0;
            int d = 0, t = 0;
            const double x = capacity(pol, i, &e, &d, &t, &f);
            const double g = (x - none[i]) * m_.fm_scale();
            gain += g;
            exceed += g >= mcid;
            eff += e;
            done_all += d;
            total_all += t;
            follow_sum += f;
            zero += (d == 0);
        }
        o.completed_fraction = total_all ? double(done_all) / double(total_all) : 0.0;
        o.mean_effective_hours = eff / patients;
        o.mean_fm_gain = gain / patients;
        o.prob_exceeds_mcid = double(exceed) / patients;
        o.mean_cmd_follow_mm = done_all ? follow_sum / double(done_all) : 0.0;
        o.zero_dose_fraction = double(zero) / patients;
        out.push_back(o);
    }
    return out;
}

}  // namespace apt::twin
