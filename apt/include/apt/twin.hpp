// apt/twin.hpp - closed-loop digital twin: device design -> link delay -> sessions -> dose -> outcome.
//
// The chain, each link taken from a layer that is verified elsewhere in this suite:
//
//   link delay of a session  --(closed-loop HIL, stability map)-->  does the session complete?
//   link delay of a session  --(latency-to-learning law a(tau))-->  how effective is it per hour?
//   completed, effective hours --(Module M threshold dose-response)--> ceiling raise that week
//   weekly ceilings          --(Module D sensitive window, running reward)--> final capacity
//   final capacity           --(calibrated FM scale)--> Fugl-Meyer gain
//
// Everything is a *contrast between device configurations* under one explicit link scenario and
// one explicit reference latency, drawn on the same patients and the same noise path. No function
// here returns a device-versus-therapy effect, for the reason `recovery.hpp` gives: the evidence
// base does not fix the feedback latency of conventional therapy.
//
// What the twin does not model, and says so in its output: how much the loosened device-to-command
// tracking of the adaptive controller matters to learning (no verified quantity relates the two;
// it is reported as a separate transparency axis), and any effect of session failure on adherence.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "apt/adapt.hpp"
#include "apt/afferent.hpp"
#include "apt/evidence.hpp"
#include "apt/schedule.hpp"

namespace apt::twin {

// Per-delay closed-loop behaviour of one controller policy, measured in the HIL on a 5 ms grid.
class StabilityMap {
public:
    // gain_scale(delay_ms) gives the inner-gain scale the policy would apply at that delay;
    // delays above `certified_ms` are treated as not completing (the supervisor trips or locks out).
    static StabilityMap measure(const core::DesignPoint& base, std::string name,
                                const std::function<double(double)>& gain_scale, double certified_ms,
                                double max_ms = 300.0, double step_ms = 5.0, double seconds = 4.0);

    // A session at this delay completes only if the grid points on both sides are stable
    // (conservative between grid points).
    bool completes(double delay_ms) const;
    double cmd_follow_mm(double delay_ms) const;  // rendered-dynamics fidelity cost at that delay
    double first_failure_ms() const;              // smallest grid delay that does not complete
    const std::string& name() const { return name_; }

private:
    std::string name_;
    double step_ = 5.0;
    std::vector<char> stable_;
    std::vector<double> follow_;
};

struct LinkScenario {
    double median_ms = 30.0;       // population median one-way delay
    double patient_sigma = 0.5;    // between-patient log-sd of the home-link median
    double session_sigma = 0.35;   // session-to-session log-sd
    double spike_prob = 0.02;      // chance that a session runs in a congested link ...
    double spike_factor = 5.0;     // ... at this multiple of its usual delay
    double electronics_ms = 3.2;   // sample -> actuation inside the device (filter + electronics)
};

struct Protocol {
    int start_week = 8;             // weeks since stroke (near the fitted sensitive-period peak)
    int weeks = 12;
    int sessions_per_week = 5;
    double hours_per_session = 0.5;
    double reference_latency_ms = 30.0;  // learning efficiency is stated relative to this
};

struct PolicyOutcome {
    std::string policy;
    double completed_fraction;       // sessions that ran to completion
    double mean_effective_hours;     // per patient over the programme, latency-weighted
    double mean_fm_gain;             // treatment-attributable, vs. no treatment on the same noise
    double prob_exceeds_mcid;
    double mean_cmd_follow_mm;       // transparency axis (completed sessions only)
    double zero_dose_fraction;       // patients who completed no session at all
};

struct Twin {
    Twin(const EvidenceBase& eb, const ScheduleModel& model, const LatencyLearningLaw& law)
        : eb_(eb), m_(model), law_(law) {}

    // All policies are evaluated on the same patients, the same link draws and the same noise.
    std::vector<PolicyOutcome> run(const std::vector<const StabilityMap*>& policies, const LinkScenario& link,
                                   const Protocol& proto, int patients, std::uint64_t seed) const;

private:
    const EvidenceBase& eb_;
    const ScheduleModel& m_;
    const LatencyLearningLaw& law_;
};

}  // namespace apt::twin
