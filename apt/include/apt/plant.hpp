// apt/plant.hpp - hardware-in-the-loop plant for the golden model.
//
// A one-axis hand interface: the device carriage (mass, viscous friction, a motor with an
// electrical lag) is coupled through a compliant load cell to the user's hand, which is driven
// by a muscle-like impedance tracking a minimum-jerk reaching intent. Sensors are quantised
// exactly as the hardware quantises them (16-bit ADC codes, encoder counts), with additive noise
// and mains pickup, and a configurable transport delay is inserted between the plant and the
// controller - the knob the manuscript's stability and learning envelopes are drawn against.
#pragma once

#include <cstdint>
#include <deque>
#include <vector>

#include "apt/core_model.hpp"
#include "apt/numerics.hpp"

namespace apt::hil {

struct PlantParams {
    double device_mass_kg = 0.8;
    double device_friction_Ns_m = 2.0;
    double motor_fmax_N = 50.0;
    double motor_tau_s = 1e-3;
    double coupling_k_N_m = 5000.0;   // load cell + strap compliance
    double coupling_b_Ns_m = 5.0;
    double hand_mass_kg = 1.5;
    double hand_k_N_m = 300.0;        // muscle-like impedance about the intent
    double hand_b_Ns_m = 20.0;
    double force_noise_N = 0.05;
    double mains_N = 0.2;
    double mains_hz = 50.0;
    double extra_latency_ms = 0.0;    // transport delay between plant and controller
    double sim_hz = 20000.0;
};

// Minimum-jerk reach-and-return between two positions (mm), repeated.
struct ReachingTask {
    double start_mm = 30.0, end_mm = 90.0;
    double reach_s = 0.8, hold_s = 0.3;
    double intent_mm(double t) const;
    double intent_velocity_mm_s(double t) const;
    double period_s() const { return 2 * (reach_s + hold_s); }
};

struct Sample {
    double t, x_hand_mm, x_dev_mm, intent_mm, force_true_N;
    core::Outputs out;
};

struct RunSummary {
    double latency_ms;
    bool passivity_control;
    double tracking_rms_mm;        // hand vs intent
    double interaction_rms_N;      // what the user feels
    double min_energy_mJ;          // passivity observer minimum
    double pc_active_fraction;
    double peak_duty;
    double oscillation_index;      // late / early high-frequency velocity power
    bool faulted;
    std::uint16_t reps;
    double afferent_rate_hz;
    double gated_fraction;
};

class Plant {
public:
    Plant(PlantParams p, std::uint64_t seed);

    // Advance the plant by one control period under motor duty `duty` (in [-1, 1]); returns the
    // sensor inputs the controller will see `extra_latency_ms` later.
    core::Inputs advance(double ctrl_dt, double duty, const ReachingTask& task);

    double t() const { return t_; }
    double x_hand_mm() const { return 1000 * s_[2]; }
    double x_device_mm() const { return 1000 * s_[0]; }
    double force_true_N() const;

private:
    void deriv(const double* s, double t, const ReachingTask& task, double* ds) const;

    PlantParams p_;
    num::Rng rng_;
    double t_ = 0, motor_force_ = 0, duty_ = 0;
    double s_[4] = {0.03, 0, 0.03, 0};  // x_dev, v_dev, x_hand, v_hand [m, m/s]
    std::deque<core::Inputs> pipe_;
};

RunSummary run(const core::Config& cfg, const PlantParams& pp, const ReachingTask& task,
               double seconds, std::uint64_t seed, std::vector<Sample>* trace = nullptr);

}  // namespace apt::hil
