// apt/adapt.hpp - latency-adaptive inner-loop gain scheduling, certified in closed loop.
//
// Finding that motivates it (see `aptctl adapt` and tests/test_adapt.cpp): in the closed-loop
// HIL the first thing to go unstable as the transport delay grows is not the rendered
// admittance (m, b, k) but the *inner position loop*, which closes a PD law around a delayed
// measurement. Raising the rendered damping buys ~20 ms; scaling the inner gains down buys
// more than 150 ms, at the price of a looser device-to-command tracking (the hand is barely
// affected, because the inner loop only has to follow a command the admittance law already
// shaped). The scale that is safe falls roughly as 1/tau^2, which is the classical
// delay-limited crossover frequency of a PD loop (omega_c * tau ~ const).
//
// The schedule is therefore *certified*, not guessed: for every delay bin the largest scale on
// a ladder is chosen such that the closed-loop HIL stays stable (a) at 1.25x the delay of the
// bin and (b) at 1.5x the scale at the delay of the bin, over several noise seeds. The result is
// a small table that the supervisory MCU can apply with two CSR writes.
#pragma once

#include <cstdint>
#include <vector>

#include "apt/core_model.hpp"
#include "apt/plant.hpp"

namespace apt::adapt {

// One-sided delay estimator: smooths the probe, attacks instantly and releases slowly. A delay
// estimate that is too low is the unsafe direction, so it errs on the high side.
class DelayEstimator {
public:
    explicit DelayEstimator(double smooth_tau_s = 0.02, double release_ms_per_s = 50.0)
        : tau_(smooth_tau_s), release_(release_ms_per_s) {}
    double update(double probe_ms, double dt_s);
    double estimate_ms() const { return est_; }
    void reset() { smooth_ = est_ = 0.0; init_ = false; }

private:
    double tau_, release_, smooth_ = 0.0, est_ = 0.0;
    bool init_ = false;
};

struct GainSchedule {
    std::vector<double> edge_ms;  // upper delay edge of each bin (ascending)
    std::vector<double> scale;    // inner-gain scale for the bin (non-increasing)

    // Scale for an estimated delay: the first bin whose edge covers it. Beyond the last
    // certified edge the last (smallest) scale is returned and `covered` is false.
    double lookup(double delay_ms, bool* covered = nullptr) const;
    double certified_max_ms() const { return edge_ms.empty() ? 0.0 : edge_ms.back(); }
};

// The analytic starting point: scale = min(1, (tau0 / tau)^2).
GainSchedule analytic_schedule(const std::vector<double>& edges_ms, double tau0_ms = 40.0);

struct CertifySpec {
    std::vector<double> edges_ms = {0, 10, 20, 30, 40, 60, 80, 100, 120, 160, 200};
    double delay_margin = 1.25;     // must also be stable at this multiple of the bin delay
    double gain_margin = 1.5;       // ... and at this multiple of the chosen scale
    int seeds = 3;
    double seconds = 5.0;
    double ladder_ratio = 0.8;      // candidate scales 1, 0.8, 0.64, ...
    double min_scale = 0.02;
    double energy_limit_mJ = 50.0;  // port energy below minus this is unstable
};

GainSchedule certify_schedule(const core::DesignPoint& base, const CertifySpec& spec = {});

// A closed-loop trial: stable means no latched fault and the port energy stayed bounded.
struct Trial {
    bool stable;
    hil::RunSummary summary;
};
Trial trial(const core::DesignPoint& base, double inner_scale, double delay_ms, std::uint64_t seed,
            double seconds, double energy_limit_mJ = 50.0);

// Applies the schedule online: feeds delay probes to the estimator and rewrites the two
// inner-loop gains (KP, KD) of a running model whenever the bin changes. A step down is
// immediate; a step up waits `hold_s` of continuous evidence.
class Supervisor {
public:
    Supervisor(const core::DesignPoint& base, GainSchedule sched, double hold_s = 0.5, double dt_s = 1e-3);
    void on_probe(double probe_ms, core::CoreModel& model);
    double scale() const { return scale_; }
    double estimate_ms() const { return est_.estimate_ms(); }
    int changes() const { return changes_; }
    bool beyond_certified() const { return beyond_; }

private:
    core::DesignPoint base_;
    GainSchedule sched_;
    DelayEstimator est_;
    double hold_s_, dt_;
    double scale_ = 1.0, candidate_ = 1.0, held_ = 0.0;
    int changes_ = 0;
    bool beyond_ = false;
};

}  // namespace apt::adapt
