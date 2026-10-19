// apt/core_model.hpp - bit-exact golden model of the FPGA datapath (rtl/apt_core.v).
//
// This file *is* the specification of one control tick. The RTL is verified against it frame by
// frame: tb/tb_apt_top.v drives the same ADC codes and encoder counts, captures the SLIP/UART
// telemetry, and compares every byte except the two cycle-latency fields.
//
// One tick, in order:
//   1  scale       phys[c] = mul_code(adc[c], ADC_FS[c])
//   2  filter      filt[c] = DF-I biquad(phys[c])  (or phys[c] if bypassed; history always runs)
//   3  force       F = filt[0] - F_OFFSET
//   4  kinematics  x = mul_ic(enc, ENC_GAIN); (first tick after reset: x_prev = x)
//                  v_raw = (x - x_prev) * FS_HZ;
//                  v += mul_sc(v_raw - v, V_ALPHA)
//   5  safety      causes from estop, |F|, |v|, x range, watchdog, latency, ADC; latched;
//                  FAULT_CLEAR replaces the latch with the causes present this tick
//   6  if !(enable && !fault): E = 0, x_cmd = x, v_cmd = 0, duty = 0, disarm
//      else: on the first active tick E = 0, x_cmd = x, v_cmd = 0 (bumpless entry), then
//   6a passivity   E += mul_sc(mul_ss(F, v), C_DT)                (observer, energy into device)
//                  if PC && E < 0 && |v| > eps: alpha = min((-E << 20) / mul_sc(v^2, C_DT), max)
//                  F_pc = -alpha v;  E += alpha v^2 dt             (Hannaford-Ryu controller)
//   6b admittance  v_cmd = rnd(v_cmd CB + (F + F_pc) CF - (x_cmd - X_EQ) CK); x_cmd += v_cmd dt
//   6c inner loop  u = rnd((x_cmd - x) KP + (v_cmd - v) KD); d = clamp(u C_DUTY, +-1);
//                  duty += clamp(d - duty, +-SLEW)
//   7  afferent    for k in 0..3: LIF(|filt[src[k]]|) -> spike -> paired-pulse gate
//   8  reps        Schmitt detector on x between REP_LO and REP_HI
//   9  emit        telemetry(tick), PWM compare = |duty| * PERIOD >> 20; tick += 1
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "apt/dsp.hpp"
#include "apt/fixed.hpp"
#include "apt/link.hpp"
#include "apt/neuromorphic.hpp"

namespace apt::core {

inline constexpr int kAdcChannels = 8;
inline constexpr int kStimChannels = 4;

enum AdcChannel : int {
    kInteractionForce = 0,
    kGripForce = 1,
    kElbowTorque = 2,
    kMcpAngle = 3,
    kPipAngle = 4,
    kEmgAgonist = 5,
    kEmgAntagonist = 6,
    kMotorCurrent = 7,
};

// afferent channel k is driven by ADC channel kStimSource[k]
inline constexpr std::array<int, kStimChannels> kStimSource = {kGripForce, kInteractionForce,
                                                               kMcpAngle, kPipAngle};

enum FaultBit : std::uint8_t {
    kFaultEstop = 1u << 0,
    kFaultForce = 1u << 1,
    kFaultVelocity = 1u << 2,
    kFaultPosition = 1u << 3,
    kFaultWatchdog = 1u << 4,
    kFaultLatency = 1u << 5,
    kFaultAdc = 1u << 6,
};

enum CtrlBit : std::uint32_t {
    kCtrlEnable = 1u << 0,
    kCtrlPassivity = 1u << 1,
    kCtrlStim = 1u << 2,
    kCtrlTelemetry = 1u << 3,
    kCtrlRun = 1u << 4,  // RTL only: starts the tick generator and soft-resets the datapath
};

// The physical design point a configuration is derived from. The gating parameters are the
// verified ones. The rendered mass and damping are NOT the manuscript's 0.25 kg / 6 N s/m: those
// are unsourced constants of the stability analysis, and the HIL plant shows that rendering a
// virtual mass below the 0.8 kg physical carriage is unstable at every latency (the classic
// admittance-control coupled-stability limit). 1 kg / 20 N s/m is stable to ~40 ms of transport
// delay on that plant; `aptctl hil` reproduces the sweep.
struct DesignPoint {
    double ctrl_hz = 1000.0;
    double virtual_mass_kg = 1.0;
    double virtual_damping_Ns_m = 20.0;
    double virtual_stiffness_N_m = 50.0;
    double x_equilibrium_mm = 40.0;
    double filter_fc_hz = 80.0;
    double velocity_fc_hz = 50.0;
    double inner_kp_N_mm = 0.5;
    double inner_kd_Ns_mm = 0.02;
    double motor_fmax_N = 50.0;
    double duty_slew_per_tick = 0.05;
    double pc_alpha_max_Ns_mm = 0.2;
    double pc_v_eps_mm_s = 1.0;
    double limit_force_N = 80.0;
    double limit_velocity_mm_s = 1500.0;
    double travel_min_mm = -10.0;
    double travel_max_mm = 160.0;
    double encoder_mm_per_count = 0.005;
    std::array<double, kAdcChannels> adc_full_scale = {100, 400, 5, 180, 180, 5, 5, 10};
    double lif_leak_per_tick = 0.01;
    std::array<double, kStimChannels> lif_gain = {0.005, 0.01, 0.002, 0.002};
    double lif_theta = 1.0;
    double gate_isi_ms = 30.0;   // verified somatosensory gating ISI
    double gate_ratio = 0.4;     // verified healthy second/first response ratio
    std::array<std::uint8_t, kStimChannels> stim_amp = {0x40, 0x40, 0x40, 0x40};
    std::uint32_t stim_width_cycles = 10000;  // 200 us at 50 MHz
    std::uint32_t stim_gap_cycles = 2500;     // 50 us
    std::uint32_t stim_refract_cycles = 250000;  // 5 ms
    double rep_lo_mm = 40.0, rep_hi_mm = 80.0;  // inside the 30-90 mm reaching task
    std::uint32_t watchdog_ticks = 100;
    std::uint32_t latency_budget_cycles = 5000;  // 100 us of electronics at 50 MHz
};

// Register-level configuration: exactly what the CSRs hold.
struct Config {
    std::uint32_t ctrl = kCtrlEnable | kCtrlPassivity | kCtrlStim | kCtrlTelemetry;
    dsp::FixedBiquadCoefs bq{};
    std::uint32_t filt_bypass = 0;
    std::uint32_t fs_hz = 1000;
    fx::coef_t enc_gain = 0;
    std::array<fx::sig_t, kAdcChannels> adc_fs{};
    fx::sig_t f_offset = 0;
    fx::coef_t v_alpha = 0;
    fx::coef_t adm_cb = 0, adm_cf = 0, adm_ck = 0;
    fx::sig_t x_eq = 0;
    fx::coef_t c_dt = 0;
    fx::coef_t kp = 0, kd = 0, c_duty = 0;
    fx::sig_t slew = 0;
    fx::sig_t pc_alpha_max = 0, pc_v_eps = 0;
    fx::sig_t lim_f = 0, lim_v = 0, lim_xmin = 0, lim_xmax = 0;
    std::uint32_t lat_budget = 0, wdog_ticks = 0;
    fx::coef_t lif_leak = 0;
    std::array<fx::coef_t, kStimChannels> lif_gain{};
    fx::sig_t lif_theta = 0;
    std::uint32_t gate_isi = 0, gate_ratio_q8 = 0;
    std::uint32_t stim_amp = 0, stim_width = 0, stim_gap = 0, stim_refract = 0;
    fx::sig_t rep_lo = 0, rep_hi = 0;

    static Config from_design(const DesignPoint& d);
    // (address, value) pairs in register-map order, for codegen and for SPI configuration
    std::vector<std::pair<std::uint8_t, std::uint32_t>> to_registers() const;
};

struct Inputs {
    std::array<std::int16_t, kAdcChannels> adc{};
    std::int32_t encoder = 0;
    bool estop = false;
    bool heartbeat = false;
    bool fault_clear = false;
    bool latency_fault = false;
    bool adc_fault = false;
};

struct Outputs {
    std::array<fx::sig_t, kAdcChannels> filt{};
    fx::sig_t force = 0, x_meas = 0, v_meas = 0, x_cmd = 0, v_cmd = 0, energy = 0, duty = 0;
    fx::sig_t f_pc = 0;
    bool pc_active = false, fault = false, active = false;
    std::uint8_t causes = 0, latched = 0;
    std::uint8_t spikes = 0, gated = 0;
    std::array<std::uint8_t, kStimChannels> stim_amp{};
    std::uint32_t pwm_compare = 0;
    bool pwm_reverse = false;
    std::uint16_t rep_count = 0;
    link::Telemetry telemetry;
};

class CoreModel {
public:
    static constexpr std::uint32_t kPwmPeriod = 1250;  // 20 kHz centre-aligned at 50 MHz

    explicit CoreModel(const Config& c) : cfg_(c) { reset(); }
    void reset();
    Outputs step(const Inputs& in);

    Config& config() { return cfg_; }
    const Config& config() const { return cfg_; }
    std::uint32_t tick() const { return tick_; }
    std::uint32_t spike_count() const { return spike_count_; }

private:
    Config cfg_;
    std::array<fx::sig_t, kAdcChannels> x1_{}, x2_{}, y1_{}, y2_{};
    fx::sig_t x_prev_ = 0, v_ = 0, e_ = 0, x_cmd_ = 0, v_cmd_ = 0, duty_ = 0;
    std::array<fx::sig_t, kStimChannels> lif_v_{};
    std::array<std::uint32_t, kStimChannels> since_{};
    std::uint32_t tick_ = 0, wdog_ = 0, spike_count_ = 0;
    std::uint8_t latched_ = 0;
    bool rep_state_ = false;
    std::uint16_t rep_count_ = 0;
    bool primed_ = false;  // first encoder sample seen (velocity estimator)
    bool armed_ = false;   // admittance state initialised from the measured position
};

}  // namespace apt::core
