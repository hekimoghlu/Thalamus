#include "apt/adapt.hpp"

#include <algorithm>
#include <cmath>

namespace apt::adapt {

double DelayEstimator::update(double probe_ms, double dt_s) {
    if (!init_) { smooth_ = est_ = probe_ms; init_ = true; return est_; }
    smooth_ += (probe_ms - smooth_) * std::min(1.0, dt_s / tau_);
    est_ = std::max(smooth_, est_ - release_ * dt_s);
    return est_;
}

double GainSchedule::lookup(double delay_ms, bool* covered) const {
    for (std::size_t i = 0; i < edge_ms.size(); ++i)
        if (delay_ms <= edge_ms[i]) { if (covered) *covered = true; return scale[i]; }
    if (covered) *covered = false;
    return scale.empty() ? 1.0 : scale.back();
}

GainSchedule analytic_schedule(const std::vector<double>& edges_ms, double tau0_ms) {
    GainSchedule g;
    for (double e : edges_ms) {
        g.edge_ms.push_back(e);
        g.scale.push_back(e <= tau0_ms ? 1.0 : (tau0_ms / e) * (tau0_ms / e));
    }
    return g;
}

static core::DesignPoint scaled(const core::DesignPoint& b, double s) {
    core::DesignPoint d = b;
    d.inner_kp_N_mm *= s;
    d.inner_kd_Ns_mm *= s;
    return d;
}

Trial trial(const core::DesignPoint& base, double inner_scale, double delay_ms, std::uint64_t seed,
            double seconds, double energy_limit_mJ) {
    hil::PlantParams p;
    p.extra_latency_ms = delay_ms;
    const auto r = hil::run(core::Config::from_design(scaled(base, inner_scale)), p, hil::ReachingTask{},
                            seconds, seed);
    return {!r.faulted && r.min_energy_mJ > -energy_limit_mJ, r};
}

GainSchedule certify_schedule(const core::DesignPoint& base, const CertifySpec& spec) {
    GainSchedule g;
    double cap = 1.0;  // keep the schedule monotone non-increasing
    for (double edge : spec.edges_ms) {
        double s = 1.0, chosen = spec.min_scale;
        for (; s >= spec.min_scale; s *= spec.ladder_ratio) {
            if (s > cap) continue;
            bool ok = true;
            for (int k = 0; k < spec.seeds && ok; ++k) {
                const auto seed = static_cast<std::uint64_t>(101 + k);
                ok = trial(base, s, edge * spec.delay_margin, seed, spec.seconds, spec.energy_limit_mJ).stable &&
                     trial(base, std::min(1.0, s * spec.gain_margin), edge, seed, spec.seconds,
                           spec.energy_limit_mJ).stable;
            }
            if (ok) { chosen = s; break; }
        }
        cap = chosen;
        g.edge_ms.push_back(edge);
        g.scale.push_back(chosen);
    }
    return g;
}

Supervisor::Supervisor(const core::DesignPoint& base, GainSchedule sched, double hold_s, double dt_s)
    : base_(base), sched_(std::move(sched)), hold_s_(hold_s), dt_(dt_s) {}

void Supervisor::on_probe(double probe_ms, core::CoreModel& model) {
    est_.update(probe_ms, dt_);
    bool covered = true;
    const double want = sched_.lookup(est_.estimate_ms(), &covered);
    beyond_ = !covered;
    if (want < scale_) {  // more delay: act at once
        scale_ = candidate_ = want;
        held_ = 0.0;
    } else if (want > scale_) {  // less delay: wait for sustained evidence
        if (want != candidate_) { candidate_ = want; held_ = 0.0; }
        held_ += dt_;
        if (held_ >= hold_s_) { scale_ = want; held_ = 0.0; }
    } else {
        candidate_ = want;
        held_ = 0.0;
        return;
    }
    const auto c = core::Config::from_design(scaled(base_, scale_));
    if (model.config().kp != c.kp || model.config().kd != c.kd) {
        model.config().kp = c.kp;
        model.config().kd = c.kd;
        ++changes_;
    }
}

}  // namespace apt::adapt
