#include "apt/core_model.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string_view>

#include "apt/regmap.hpp"

namespace apt::core {

using namespace apt::fx;

Config Config::from_design(const DesignPoint& d) {
    const double dt = 1.0 / d.ctrl_hz;
    Config c;
    c.fs_hz = static_cast<std::uint32_t>(std::lround(d.ctrl_hz));
    c.bq = dsp::FixedBiquadCoefs::quantise(
        dsp::butterworth_lowpass(2, d.filter_fc_hz, d.ctrl_hz).sections.at(0));
    c.enc_gain = to_coef(d.encoder_mm_per_count);
    for (int i = 0; i < kAdcChannels; ++i) c.adc_fs[i] = to_sig(d.adc_full_scale[i]);
    c.v_alpha = to_coef(1.0 - std::exp(-2.0 * std::numbers::pi * d.velocity_fc_hz * dt));
    // kinematics in millimetres: a [mm/s^2] = 1000 F / m, and the b, k terms are scale-free
    c.adm_cb = to_coef(1.0 - dt * d.virtual_damping_Ns_m / d.virtual_mass_kg);
    c.adm_cf = to_coef(dt * 1000.0 / d.virtual_mass_kg);
    c.adm_ck = to_coef(dt * d.virtual_stiffness_N_m / d.virtual_mass_kg);
    c.x_eq = to_sig(d.x_equilibrium_mm);
    c.c_dt = to_coef(dt);
    c.kp = to_coef(d.inner_kp_N_mm);
    c.kd = to_coef(d.inner_kd_Ns_mm);
    c.c_duty = to_coef(1.0 / d.motor_fmax_N);
    c.slew = to_sig(d.duty_slew_per_tick);
    c.pc_alpha_max = to_sig(d.pc_alpha_max_Ns_mm);
    c.pc_v_eps = to_sig(d.pc_v_eps_mm_s);
    c.lim_f = to_sig(d.limit_force_N);
    c.lim_v = to_sig(d.limit_velocity_mm_s);
    c.lim_xmin = to_sig(d.travel_min_mm);
    c.lim_xmax = to_sig(d.travel_max_mm);
    c.lat_budget = d.latency_budget_cycles;
    c.wdog_ticks = d.watchdog_ticks;
    c.lif_leak = to_coef(d.lif_leak_per_tick);
    for (int k = 0; k < kStimChannels; ++k) c.lif_gain[k] = to_coef(d.lif_gain[k]);
    c.lif_theta = to_sig(d.lif_theta);
    c.gate_isi = static_cast<std::uint32_t>(std::lround(d.gate_isi_ms * d.ctrl_hz / 1000.0));
    c.gate_ratio_q8 = static_cast<std::uint32_t>(std::lround(d.gate_ratio * 256.0));
    c.stim_amp = 0;
    for (int k = 0; k < kStimChannels; ++k) c.stim_amp |= std::uint32_t{d.stim_amp[k]} << (8 * k);
    c.stim_width = d.stim_width_cycles;
    c.stim_gap = d.stim_gap_cycles;
    c.stim_refract = d.stim_refract_cycles;
    c.rep_lo = to_sig(d.rep_lo_mm);
    c.rep_hi = to_sig(d.rep_hi_mm);
    return c;
}

std::vector<std::pair<std::uint8_t, std::uint32_t>> Config::to_registers() const {
    auto u = [](std::int32_t v) { return static_cast<std::uint32_t>(v); };
    auto value = [&](std::string_view n) -> std::uint32_t {
        if (n == "CTRL") return ctrl;
        if (n == "WDOG_TICKS") return wdog_ticks;
        if (n == "BQ_B0") return u(bq.b0);
        if (n == "BQ_B1") return u(bq.b1);
        if (n == "BQ_B2") return u(bq.b2);
        if (n == "BQ_A1") return u(bq.a1);
        if (n == "BQ_A2") return u(bq.a2);
        if (n == "FILT_BYPASS") return filt_bypass;
        if (n == "FS_HZ") return fs_hz;
        if (n == "ENC_GAIN") return u(enc_gain);
        if (n.starts_with("ADC_FS")) return u(adc_fs[n.back() - '0']);
        if (n == "F_OFFSET") return u(f_offset);
        if (n == "V_ALPHA") return u(v_alpha);
        if (n == "ADM_CB") return u(adm_cb);
        if (n == "ADM_CF") return u(adm_cf);
        if (n == "ADM_CK") return u(adm_ck);
        if (n == "X_EQ") return u(x_eq);
        if (n == "C_DT") return u(c_dt);
        if (n == "KP") return u(kp);
        if (n == "KD") return u(kd);
        if (n == "C_DUTY") return u(c_duty);
        if (n == "SLEW") return u(slew);
        if (n == "PC_ALPHA_MAX") return u(pc_alpha_max);
        if (n == "PC_V_EPS") return u(pc_v_eps);
        if (n == "LIM_F") return u(lim_f);
        if (n == "LIM_V") return u(lim_v);
        if (n == "LIM_XMIN") return u(lim_xmin);
        if (n == "LIM_XMAX") return u(lim_xmax);
        if (n == "LAT_BUDGET") return lat_budget;
        if (n == "LIF_LEAK") return u(lif_leak);
        if (n.starts_with("LIF_GAIN")) return u(lif_gain[n.back() - '0']);
        if (n == "LIF_THETA") return u(lif_theta);
        if (n == "GATE_ISI") return gate_isi;
        if (n == "GATE_RATIO") return gate_ratio_q8;
        if (n == "STIM_AMP") return stim_amp;
        if (n == "STIM_WIDTH") return stim_width;
        if (n == "STIM_GAP") return stim_gap;
        if (n == "STIM_REFRACT") return stim_refract;
        if (n == "REP_LO") return u(rep_lo);
        if (n == "REP_HI") return u(rep_hi);
        throw std::logic_error("no configuration field for RW register " + std::string(n));
    };
    std::vector<std::pair<std::uint8_t, std::uint32_t>> out;
    for (const auto& r : regmap::kRegs)
        if (r.access == regmap::Access::RW) out.emplace_back(r.addr, value(r.name));
    return out;
}

void CoreModel::reset() {
    x1_ = x2_ = y1_ = y2_ = {};
    x_prev_ = v_ = e_ = x_cmd_ = v_cmd_ = duty_ = 0;
    lif_v_ = {};
    since_.fill(neuro::GateFixed::kSinceMax);
    tick_ = wdog_ = spike_count_ = 0;
    latched_ = 0;
    rep_state_ = false;
    rep_count_ = 0;
    primed_ = armed_ = false;
}

Outputs CoreModel::step(const Inputs& in) {
    const Config& c = cfg_;
    Outputs o;

    // 1-2  scale and filter every channel (one coefficient set, per-channel history)
    for (int ch = 0; ch < kAdcChannels; ++ch) {
        const sig_t phys = mul_code(in.adc[ch], c.adc_fs[ch]);
        const acc_t acc = acc_t{c.bq.b0} * phys + acc_t{c.bq.b1} * x1_[ch] +
                          acc_t{c.bq.b2} * x2_[ch] - acc_t{c.bq.a1} * y1_[ch] -
                          acc_t{c.bq.a2} * y2_[ch];
        const sig_t y = (c.filt_bypass >> ch) & 1u ? phys : rnd_sc(acc);
        x2_[ch] = x1_[ch];
        x1_[ch] = phys;
        y2_[ch] = y1_[ch];
        y1_[ch] = y;
        o.filt[ch] = y;
    }

    // 3-4  force and kinematics
    const sig_t F = sub(o.filt[kInteractionForce], c.f_offset);
    const sig_t x = mul_ic(in.encoder, c.enc_gain);
    if (!primed_) {  // first sample after reset: no velocity transient
        x_prev_ = x;
        primed_ = true;
    }
    const sig_t v_raw = sat32(acc_t{sub(x, x_prev_)} * acc_t{c.fs_hz & 0xFFFFu});  // 16-bit field
    v_ = add(v_, mul_sc(sub(v_raw, v_), c.v_alpha));
    x_prev_ = x;

    // 5  safety supervisor
    std::uint8_t cause = 0;
    if (in.estop) cause |= kFaultEstop;
    if (fx::abs(F) > c.lim_f) cause |= kFaultForce;
    if (fx::abs(v_) > c.lim_v) cause |= kFaultVelocity;
    if (x < c.lim_xmin || x > c.lim_xmax) cause |= kFaultPosition;
    if (c.wdog_ticks == 0) {
        wdog_ = 0;
    } else {
        if (in.heartbeat) wdog_ = 0;
        else if (wdog_ != 0xFFFFFFFFu) ++wdog_;
        if (wdog_ >= c.wdog_ticks) cause |= kFaultWatchdog;
    }
    if (in.latency_fault) cause |= kFaultLatency;
    if (in.adc_fault) cause |= kFaultAdc;
    latched_ = in.fault_clear ? cause : static_cast<std::uint8_t>(latched_ | cause);
    const bool fault = latched_ != 0;
    const bool active = (c.ctrl & kCtrlEnable) && !fault;

    // 6  interaction control
    sig_t f_pc = 0;
    bool pc_active = false;
    if (!active) {
        e_ = 0;
        x_cmd_ = x;
        v_cmd_ = 0;
        duty_ = 0;
        armed_ = false;
    } else {
        if (!armed_) {  // bumpless entry: the virtual mass starts where the device is
            e_ = 0;
            x_cmd_ = x;
            v_cmd_ = 0;
            armed_ = true;
        }
        // 6a passivity observer and controller
        e_ = add(e_, mul_sc(mul_ss(F, v_), c.c_dt));
        if ((c.ctrl & kCtrlPassivity) && e_ < 0 && fx::abs(v_) > c.pc_v_eps) {
            const sig_t v2 = mul_ss(v_, v_);
            const sig_t den = mul_sc(v2, c.c_dt);
            if (den > 0) {
                const std::uint64_t num = static_cast<std::uint64_t>(-acc_t{e_});
                const std::uint64_t q = (num << kSigFrac) / static_cast<std::uint64_t>(den);
                // the ceiling compares as an unsigned 32-bit quantity, as in the RTL
                const sig_t alpha = q > std::uint64_t{static_cast<std::uint32_t>(c.pc_alpha_max)}
                                        ? c.pc_alpha_max
                                        : static_cast<sig_t>(q);
                f_pc = neg(mul_ss(alpha, v_));
                e_ = add(e_, mul_sc(mul_ss(alpha, v2), c.c_dt));
                pc_active = true;
            }
        }
        // 6b admittance
        const sig_t err = sub(x_cmd_, c.x_eq);
        const sig_t f_in = add(F, f_pc);
        v_cmd_ = rnd_sc(acc_t{v_cmd_} * c.adm_cb + acc_t{f_in} * c.adm_cf - acc_t{err} * c.adm_ck);
        x_cmd_ = add(x_cmd_, mul_sc(v_cmd_, c.c_dt));
        // 6c inner position/velocity loop, duty clamp and slew limit
        const sig_t u = rnd_sc(acc_t{sub(x_cmd_, x)} * c.kp + acc_t{sub(v_cmd_, v_)} * c.kd);
        const sig_t d = clamp(mul_sc(u, c.c_duty), -kSigOne, kSigOne);
        duty_ = add(duty_, clamp(sub(d, duty_), neg(c.slew), c.slew));
    }

    // 7  afferent relay: LIF encoder and paired-pulse gate on each stimulus channel
    for (int k = 0; k < kStimChannels; ++k) {
        const neuro::LifFixed lif{c.lif_leak, c.lif_gain[k], c.lif_theta};
        const bool spike = lif.step(lif_v_[k], fx::abs(o.filt[kStimSource[k]]));
        if (spike) {
            bool gated = false;
            const neuro::GateFixed gate{c.gate_isi, c.gate_ratio_q8};
            o.stim_amp[k] = gate.on_spike(since_[k], static_cast<std::uint8_t>(c.stim_amp >> (8 * k)), gated);
            o.spikes |= static_cast<std::uint8_t>(1u << k);
            if (gated) o.gated |= static_cast<std::uint8_t>(1u << k);
            ++spike_count_;
        } else {
            neuro::GateFixed::idle(since_[k]);
        }
    }

    // 8  repetition detector
    if (!rep_state_ && x > c.rep_hi) {
        rep_state_ = true;
    } else if (rep_state_ && x < c.rep_lo) {
        rep_state_ = false;
        ++rep_count_;
    }

    // 9  outputs
    o.force = F;
    o.x_meas = x;
    o.v_meas = v_;
    o.x_cmd = x_cmd_;
    o.v_cmd = v_cmd_;
    o.energy = e_;
    o.duty = duty_;
    o.f_pc = f_pc;
    o.pc_active = pc_active;
    o.fault = fault;
    o.active = active;
    o.causes = cause;
    o.latched = latched_;
    o.rep_count = rep_count_;
    o.pwm_compare = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(fx::abs(duty_)) * kPwmPeriod) >> kSigFrac);
    o.pwm_reverse = duty_ < 0;

    auto& t = o.telemetry;
    t.tick = tick_;
    t.status = static_cast<std::uint16_t>(((c.ctrl & kCtrlEnable) ? 1u : 0u) | (fault ? 2u : 0u) |
                                          (pc_active ? 4u : 0u) | ((c.ctrl & kCtrlStim) ? 8u : 0u) |
                                          (std::uint32_t{latched_} << 8));
    t.force = F;
    t.x_meas = x;
    t.v_meas = v_;
    t.x_cmd = x_cmd_;
    t.v_cmd = v_cmd_;
    t.energy = e_;
    t.duty = duty_;
    t.spikes = static_cast<std::uint8_t>(o.spikes | (o.gated << 4));
    t.rep_count = rep_count_;
    ++tick_;
    return o;
}

}  // namespace apt::core
