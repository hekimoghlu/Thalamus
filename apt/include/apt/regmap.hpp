// apt/regmap.hpp - the control/status register map: the single source of truth from which
// `aptctl codegen` emits rtl/generated/apt_regmap.vh and fw/include/apt_regmap.h.
//
// 7-bit word address space, 32-bit registers, accessed over the SPI slave port
// (one transaction = [R/W | addr6..0] then 32 data bits, MSB first).
#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace apt::regmap {

enum class Access : std::uint8_t { RO, RW, WO };

enum class Format : std::uint8_t { Raw, Sig, Coef };  // how the value is interpreted

struct Reg {
    std::string_view name;
    std::uint8_t addr;
    Access access;
    Format format;
    std::string_view description;
};

// clang-format off
inline constexpr std::array kRegs = {
    Reg{"ID",             0x00, Access::RO, Format::Raw,  "device identifier 'APT1'"},
    Reg{"VERSION",        0x01, Access::RO, Format::Raw,  "RTL version major.minor.patch"},
    Reg{"CTRL",           0x02, Access::RW, Format::Raw,  "b0 enable, b1 passivity ctrl, b2 stim enable, b3 telemetry"},
    Reg{"STATUS",         0x03, Access::RO, Format::Raw,  "b0 enabled b1 fault b2 pc_active b3 stim; [15:8] latched fault causes"},
    Reg{"FAULT_CLEAR",    0x04, Access::WO, Format::Raw,  "write 1: clear latched faults whose cause is gone"},
    Reg{"HEARTBEAT",      0x05, Access::WO, Format::Raw,  "any write kicks the host watchdog"},
    Reg{"WDOG_TICKS",     0x06, Access::RW, Format::Raw,  "watchdog timeout in control ticks (0 disables)"},
    Reg{"TICK",           0x07, Access::RO, Format::Raw,  "control tick counter"},
    Reg{"BQ_B0",          0x08, Access::RW, Format::Coef, "front-end biquad b0"},
    Reg{"BQ_B1",          0x09, Access::RW, Format::Coef, "front-end biquad b1"},
    Reg{"BQ_B2",          0x0A, Access::RW, Format::Coef, "front-end biquad b2"},
    Reg{"BQ_A1",          0x0B, Access::RW, Format::Coef, "front-end biquad a1"},
    Reg{"BQ_A2",          0x0C, Access::RW, Format::Coef, "front-end biquad a2"},
    Reg{"FILT_BYPASS",    0x0D, Access::RW, Format::Raw,  "per-channel filter bypass mask"},
    Reg{"FS_HZ",          0x0E, Access::RW, Format::Raw,  "control rate in Hz (velocity estimator)"},
    Reg{"ENC_GAIN",       0x0F, Access::RW, Format::Coef, "encoder: mm per count"},
    Reg{"ADC_FS0",        0x10, Access::RW, Format::Sig,  "ch0 interaction force full scale [N]"},
    Reg{"ADC_FS1",        0x11, Access::RW, Format::Sig,  "ch1 grip force full scale [N]"},
    Reg{"ADC_FS2",        0x12, Access::RW, Format::Sig,  "ch2 elbow torque full scale [N m]"},
    Reg{"ADC_FS3",        0x13, Access::RW, Format::Sig,  "ch3 MCP angle full scale [deg]"},
    Reg{"ADC_FS4",        0x14, Access::RW, Format::Sig,  "ch4 PIP angle full scale [deg]"},
    Reg{"ADC_FS5",        0x15, Access::RW, Format::Sig,  "ch5 EMG agonist full scale [mV]"},
    Reg{"ADC_FS6",        0x16, Access::RW, Format::Sig,  "ch6 EMG antagonist full scale [mV]"},
    Reg{"ADC_FS7",        0x17, Access::RW, Format::Sig,  "ch7 motor current full scale [A]"},
    Reg{"F_OFFSET",       0x18, Access::RW, Format::Sig,  "interaction force tare [N]"},
    Reg{"V_ALPHA",        0x19, Access::RW, Format::Coef, "velocity estimator smoothing"},
    Reg{"ADM_CB",         0x1A, Access::RW, Format::Coef, "admittance 1 - dt b / m"},
    Reg{"ADM_CF",         0x1B, Access::RW, Format::Coef, "admittance dt / m (mm/s per N)"},
    Reg{"ADM_CK",         0x1C, Access::RW, Format::Coef, "admittance dt k / m"},
    Reg{"X_EQ",           0x1D, Access::RW, Format::Sig,  "virtual spring equilibrium [mm]"},
    Reg{"C_DT",           0x1E, Access::RW, Format::Coef, "control period [s]"},
    Reg{"KP",             0x20, Access::RW, Format::Coef, "inner loop stiffness [N/mm]"},
    Reg{"KD",             0x21, Access::RW, Format::Coef, "inner loop damping [N s/mm]"},
    Reg{"C_DUTY",         0x22, Access::RW, Format::Coef, "duty per newton (1 / F_max)"},
    Reg{"SLEW",           0x23, Access::RW, Format::Sig,  "max duty change per tick"},
    Reg{"PC_ALPHA_MAX",   0x24, Access::RW, Format::Sig,  "passivity controller damping ceiling [N s/mm]"},
    Reg{"PC_V_EPS",       0x25, Access::RW, Format::Sig,  "passivity controller velocity floor [mm/s]"},
    Reg{"LIM_F",          0x28, Access::RW, Format::Sig,  "interaction force limit [N]"},
    Reg{"LIM_V",          0x29, Access::RW, Format::Sig,  "velocity limit [mm/s]"},
    Reg{"LIM_XMIN",       0x2A, Access::RW, Format::Sig,  "travel lower limit [mm]"},
    Reg{"LIM_XMAX",       0x2B, Access::RW, Format::Sig,  "travel upper limit [mm]"},
    Reg{"LAT_BUDGET",     0x2C, Access::RW, Format::Raw,  "sample-to-actuation budget [clock cycles]"},
    Reg{"LIF_LEAK",       0x30, Access::RW, Format::Coef, "LIF membrane leak per tick"},
    Reg{"LIF_GAIN0",      0x31, Access::RW, Format::Coef, "LIF gain, stim ch0 (grip force)"},
    Reg{"LIF_GAIN1",      0x32, Access::RW, Format::Coef, "LIF gain, stim ch1 (interaction force)"},
    Reg{"LIF_GAIN2",      0x33, Access::RW, Format::Coef, "LIF gain, stim ch2 (MCP angle)"},
    Reg{"LIF_GAIN3",      0x34, Access::RW, Format::Coef, "LIF gain, stim ch3 (PIP angle)"},
    Reg{"LIF_THETA",      0x35, Access::RW, Format::Sig,  "LIF threshold"},
    Reg{"GATE_ISI",       0x36, Access::RW, Format::Raw,  "sensory gating interval [ticks]"},
    Reg{"GATE_RATIO",     0x37, Access::RW, Format::Raw,  "second-response amplitude ratio, Q0.8"},
    Reg{"STIM_AMP",       0x38, Access::RW, Format::Raw,  "per-channel stimulus amplitude, 4 x 8 bit"},
    Reg{"STIM_WIDTH",     0x39, Access::RW, Format::Raw,  "biphasic phase width [cycles]"},
    Reg{"STIM_GAP",       0x3A, Access::RW, Format::Raw,  "interphase gap [cycles]"},
    Reg{"STIM_REFRACT",   0x3B, Access::RW, Format::Raw,  "minimum pulse interval [cycles]"},
    Reg{"REP_LO",         0x3C, Access::RW, Format::Sig,  "repetition detector return threshold [mm]"},
    Reg{"REP_HI",         0x3D, Access::RW, Format::Sig,  "repetition detector reach threshold [mm]"},
    Reg{"X_MEAS",         0x40, Access::RO, Format::Sig,  "measured position [mm]"},
    Reg{"V_MEAS",         0x41, Access::RO, Format::Sig,  "measured velocity [mm/s]"},
    Reg{"F_MEAS",         0x42, Access::RO, Format::Sig,  "interaction force [N]"},
    Reg{"E_OBS",          0x43, Access::RO, Format::Sig,  "passivity observer energy [mJ]"},
    Reg{"DUTY",           0x44, Access::RO, Format::Sig,  "motor duty [-1, 1]"},
    Reg{"X_CMD",          0x45, Access::RO, Format::Sig,  "admittance command [mm]"},
    Reg{"LAT_MOTOR_LAST", 0x48, Access::RO, Format::Raw,  "sample-to-PWM latency, last [cycles]"},
    Reg{"LAT_MOTOR_MAX",  0x49, Access::RO, Format::Raw,  "sample-to-PWM latency, max [cycles]"},
    Reg{"LAT_AFF_LAST",   0x4A, Access::RO, Format::Raw,  "sample-to-stimulus latency, last [cycles]"},
    Reg{"LAT_AFF_MAX",    0x4B, Access::RO, Format::Raw,  "sample-to-stimulus latency, max [cycles]"},
    Reg{"REP_COUNT",      0x4C, Access::RO, Format::Raw,  "completed reach-and-return repetitions"},
    Reg{"SPIKE_COUNT",    0x4D, Access::RO, Format::Raw,  "afferent events emitted"},
    Reg{"STIM_OVERRUN",   0x4E, Access::RO, Format::Raw,  "afferent events dropped (stimulator busy)"},
    Reg{"ENC_ERR",        0x4F, Access::RO, Format::Raw,  "illegal quadrature transitions"},
    Reg{"FILT0",          0x50, Access::RO, Format::Sig,  "filtered ch0"},
    Reg{"FILT1",          0x51, Access::RO, Format::Sig,  "filtered ch1"},
    Reg{"FILT2",          0x52, Access::RO, Format::Sig,  "filtered ch2"},
    Reg{"FILT3",          0x53, Access::RO, Format::Sig,  "filtered ch3"},
    Reg{"FILT4",          0x54, Access::RO, Format::Sig,  "filtered ch4"},
    Reg{"FILT5",          0x55, Access::RO, Format::Sig,  "filtered ch5"},
    Reg{"FILT6",          0x56, Access::RO, Format::Sig,  "filtered ch6"},
    Reg{"FILT7",          0x57, Access::RO, Format::Sig,  "filtered ch7"},
};
// clang-format on

inline constexpr std::uint32_t kDeviceId = 0x41505431;  // "APT1"
inline constexpr std::uint32_t kVersion = 0x00010000;   // 1.0.0

constexpr const Reg& find(std::string_view name) {
    for (const auto& r : kRegs)
        if (r.name == name) return r;
    throw "unknown register";  // compile-time error when used in a constant expression
}

constexpr bool addresses_unique() {
    for (std::size_t i = 0; i < kRegs.size(); ++i)
        for (std::size_t j = i + 1; j < kRegs.size(); ++j)
            if (kRegs[i].addr == kRegs[j].addr) return false;
    for (const auto& r : kRegs)
        if (r.addr > 0x7F) return false;
    return true;
}
static_assert(addresses_unique(), "register map has a duplicate or out-of-range address");

}  // namespace apt::regmap
