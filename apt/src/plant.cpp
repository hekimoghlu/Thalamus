#include "apt/plant.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace apt::hil {

namespace {
double min_jerk(double tau) {  // 0 -> 1 over tau in [0, 1]
    tau = std::clamp(tau, 0.0, 1.0);
    return tau * tau * tau * (10 - 15 * tau + 6 * tau * tau);
}
double min_jerk_rate(double tau) {
    if (tau <= 0 || tau >= 1) return 0;
    return 30 * tau * tau * (1 - tau) * (1 - tau);
}
std::int16_t to_code(double value, double full_scale) {
    const double c = std::nearbyint(value / full_scale * 32768.0);
    return static_cast<std::int16_t>(std::clamp(c, -32768.0, 32767.0));
}
}  // namespace

double ReachingTask::intent_mm(double t) const {
    const double ph = std::fmod(t, period_s());
    const double span = end_mm - start_mm;
    if (ph < reach_s) return start_mm + span * min_jerk(ph / reach_s);
    if (ph < reach_s + hold_s) return end_mm;
    if (ph < 2 * reach_s + hold_s) return end_mm - span * min_jerk((ph - reach_s - hold_s) / reach_s);
    return start_mm;
}

double ReachingTask::intent_velocity_mm_s(double t) const {
    const double ph = std::fmod(t, period_s());
    const double span = end_mm - start_mm;
    if (ph < reach_s) return span / reach_s * min_jerk_rate(ph / reach_s);
    if (ph < reach_s + hold_s) return 0;
    if (ph < 2 * reach_s + hold_s) return -span / reach_s * min_jerk_rate((ph - reach_s - hold_s) / reach_s);
    return 0;
}

Plant::Plant(PlantParams p, std::uint64_t seed) : p_(p), rng_(seed) {}

double Plant::force_true_N() const {
    // force the hand exerts on the device through the coupling
    return p_.coupling_k_N_m * (s_[2] - s_[0]) + p_.coupling_b_Ns_m * (s_[3] - s_[1]);
}

void Plant::deriv(const double* s, double t, const ReachingTask& task, double* ds) const {
    const double fc = p_.coupling_k_N_m * (s[2] - s[0]) + p_.coupling_b_Ns_m * (s[3] - s[1]);
    const double x_int = task.intent_mm(t) / 1000.0, v_int = task.intent_velocity_mm_s(t) / 1000.0;
    const double f_musc = p_.hand_k_N_m * (x_int - s[2]) + p_.hand_b_Ns_m * (v_int - s[3]);
    ds[0] = s[1];
    ds[1] = (motor_force_ + fc - p_.device_friction_Ns_m * s[1]) / p_.device_mass_kg;
    ds[2] = s[3];
    ds[3] = (f_musc - fc) / p_.hand_mass_kg;
}

core::Inputs Plant::advance(double ctrl_dt, double duty, const ReachingTask& task) {
    duty_ = std::clamp(duty, -1.0, 1.0);
    const int sub = std::max(1, static_cast<int>(std::lround(ctrl_dt * p_.sim_hz)));
    const double h = ctrl_dt / sub;
    for (int i = 0; i < sub; ++i) {
        // first-order electrical lag of the motor force, exact discretisation
        motor_force_ += (duty_ * p_.motor_fmax_N - motor_force_) * (1 - std::exp(-h / p_.motor_tau_s));
        double k1[4], k2[4], k3[4], k4[4], tmp[4];
        deriv(s_, t_, task, k1);
        for (int j = 0; j < 4; ++j) tmp[j] = s_[j] + 0.5 * h * k1[j];
        deriv(tmp, t_ + 0.5 * h, task, k2);
        for (int j = 0; j < 4; ++j) tmp[j] = s_[j] + 0.5 * h * k2[j];
        deriv(tmp, t_ + 0.5 * h, task, k3);
        for (int j = 0; j < 4; ++j) tmp[j] = s_[j] + h * k3[j];
        deriv(tmp, t_ + h, task, k4);
        for (int j = 0; j < 4; ++j) s_[j] += h / 6.0 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
        t_ += h;
    }
    // sensors, quantised as the hardware quantises them
    core::Inputs in;
    const double mains = p_.mains_N * std::sin(2 * std::numbers::pi * p_.mains_hz * t_);
    const double f_meas = force_true_N() + rng_.normal(0, p_.force_noise_N) + mains;
    in.adc[core::kInteractionForce] = to_code(f_meas, 100.0);
    // grip tightens during the reach phase; finger joints flex with it
    const double reach_phase = (task.intent_mm(t_) - task.start_mm) / (task.end_mm - task.start_mm);
    in.adc[core::kGripForce] = to_code(20.0 + 60.0 * reach_phase + rng_.normal(0, 0.5), 400.0);
    in.adc[core::kElbowTorque] = to_code(0.3 * f_meas * 0.03, 5.0);
    in.adc[core::kMcpAngle] = to_code(10.0 + 40.0 * reach_phase, 180.0);
    in.adc[core::kPipAngle] = to_code(15.0 + 30.0 * reach_phase, 180.0);
    in.adc[core::kEmgAgonist] = to_code(std::abs(rng_.normal(0, 0.3)), 5.0);
    in.adc[core::kEmgAntagonist] = to_code(std::abs(rng_.normal(0, 0.2)), 5.0);
    in.adc[core::kMotorCurrent] = to_code(motor_force_ / 5.0, 10.0);  // 5 N/A motor constant
    in.encoder = static_cast<std::int32_t>(std::floor(1000.0 * s_[0] / 0.005));
    in.heartbeat = true;
    // transport delay
    const auto lag = static_cast<std::size_t>(std::lround(p_.extra_latency_ms / 1000.0 / ctrl_dt));
    pipe_.push_back(in);
    if (pipe_.size() <= lag) return pipe_.front();
    const core::Inputs out = pipe_.front();
    pipe_.pop_front();
    return out;
}

RunSummary run(const core::Config& cfg, const PlantParams& pp, const ReachingTask& task,
               double seconds, std::uint64_t seed, std::vector<Sample>* trace) {
    core::CoreModel m(cfg);
    Plant plant(pp, seed);
    const double dt = 1.0 / cfg.fs_hz;
    const int n = static_cast<int>(seconds / dt);
    RunSummary r{};
    r.latency_ms = pp.extra_latency_ms;
    r.passivity_control = (cfg.ctrl & core::kCtrlPassivity) != 0;
    double duty = 0, se = 0, sf = 0, min_e = 0, early = 0, late = 0, prev_v = 0;
    int pc = 0, spikes = 0, gated = 0;
    for (int i = 0; i < n; ++i) {
        const auto in = plant.advance(dt, duty, task);
        const auto o = m.step(in);
        duty = fx::from_sig(o.duty);
        const double err = plant.x_hand_mm() - task.intent_mm(plant.t());
        se += err * err;
        sf += plant.force_true_N() * plant.force_true_N();
        min_e = std::min(min_e, fx::from_sig(o.energy));
        pc += o.pc_active;
        r.peak_duty = std::max(r.peak_duty, std::abs(duty));
        r.faulted |= o.fault;
        for (int k = 0; k < core::kStimChannels; ++k) {
            spikes += (o.spikes >> k) & 1;
            gated += (o.gated >> k) & 1;
        }
        // high-frequency velocity power: first difference of measured velocity
        const double v = fx::from_sig(o.v_meas), dv = v - prev_v;
        prev_v = v;
        (i < n / 2 ? early : late) += dv * dv;
        if (trace && i % 2 == 0)
            trace->push_back({plant.t(), plant.x_hand_mm(), plant.x_device_mm(),
                              task.intent_mm(plant.t()), plant.force_true_N(), o});
        r.reps = o.rep_count;
    }
    r.tracking_rms_mm = std::sqrt(se / n);
    r.interaction_rms_N = std::sqrt(sf / n);
    r.min_energy_mJ = min_e;
    r.pc_active_fraction = double(pc) / n;
    r.oscillation_index = late / std::max(early, 1e-12);
    r.afferent_rate_hz = spikes / seconds / core::kStimChannels;
    r.gated_fraction = spikes ? double(gated) / spikes : 0.0;
    return r;
}

}  // namespace apt::hil
