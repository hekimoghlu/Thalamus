// apt/schedule.hpp - Module M: the device's schedule as a stochastic dynamic program.
//
// State: the achievable ceiling c and the remaining hour budget b. Action: hours this week.
// Dynamics: c' = 1 - (1 - c) exp(-g(t) phi(R)) + noise, with R the relative dose and phi the
// verified *threshold* dose-response phi(R) = s A max(0, R - R_thr). Because spontaneous recovery
// is linear in the ceiling trajectory, final capacity is an exact linear functional of it and the
// per-week weight is a running reward, so backward induction over (c, b) is exact.
#pragma once

#include <cstdint>
#include <vector>

#include "apt/evidence.hpp"
#include "apt/recovery.hpp"

namespace apt {

class ScheduleModel {
public:
    static constexpr int kWeeks = 52;
    static constexpr int kNc = 81;  // ceiling grid
    static constexpr int kNb = 41;  // budget grid

    ScheduleModel(const EvidenceBase& eb, const RecoveryParams& par, const SensitiveWindow& win);

    // Threshold dose-response, solved from the two verified points with nothing left over.
    double threshold() const { return r_thr_; }
    double slope() const { return a_dose_; }
    double phi(double rel_dose) const;
    double threshold_hours(double reps_per_hour) const;
    double reps_usual() const { return reps_usual_; }
    double reps_device() const { return reps_device_; }

    // Propagate both anchors' 95% intervals; fraction of draws with a threshold above usual care.
    struct ThresholdUncertainty {
        double p_exists, q025, q975;
    };
    ThresholdUncertainty threshold_uncertainty(int draws, std::uint64_t seed) const;

    struct Solution {
        std::vector<double> V;       // [kNc * kNb]
        std::vector<float> policy;   // [kWeeks][kNc][kNb] hours
        std::vector<double> C, B;
        double reps_per_hour, budget;
        float hours_at(int t, int ic, int ib) const { return policy[(t * kNc + ic) * kNb + ib]; }
    };
    Solution solve(double reps_per_hour, double budget_hours) const;

    struct Rollout {
        std::vector<double> final_capacity, hours_used;
        std::vector<std::vector<double>> schedule;  // [week][patient]
    };
    // fixed_hours < 0 => follow the optimal policy
    Rollout rollout(const Solution& s, int n, std::uint64_t seed, double fixed_hours = -1.0) const;

    // Mean treatment-attributable FM gain of a schedule against no treatment (same noise path).
    struct Outcome {
        double gain_optimal, gain_uniform, prob_exceeds_mcid, weeks_treated, hours_used;
        double median_first_week, median_last_week;
    };
    Outcome evaluate(double reps_per_hour, double budget, int n, std::uint64_t seed) const;

    // Scale-free value of an infinitesimal ceiling increment over a window [a, b) of weeks.
    double marginal_epoch_value(int a, int b) const;

    void calibrate_scale(const EvidenceBase& eb);  // solve PHI_SCALE on doubled-duration trial
    double phi_scale() const { return phi_scale_; }
    double fm_scale() const { return fm_scale_; }
    double sigma_week() const { return sigma_w_; }
    double ceiling_floor() const { return c0_; }
    const std::vector<double>& responsiveness() const { return resp_; }
    const std::vector<double>& reward_weights() const { return w_reward_; }

    double ceiling_step(double c, int t, double hours, double reps_per_hour) const;

private:
    double r_thr_, a_dose_, phi_scale_ = 1.0, reps_usual_, reps_device_, fm_scale_, c0_, sigma_w_, mcid_;
    double dose_low_, dose_low_lo_, dose_low_hi_, dose_high_, dose_high_lo_, dose_high_hi_;
    std::vector<double> resp_;      // g(t)/max g, weeks 0..51
    std::vector<double> w_reward_;  // exact running-reward weights
    std::vector<double> hours_ = {0, 1, 2, 3, 4, 6, 8};
    std::vector<double> qz_, qw_;   // Gauss-Hermite (7)
};

}  // namespace apt
