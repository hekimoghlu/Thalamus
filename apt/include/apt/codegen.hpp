// apt/codegen.hpp - emit the hardware and firmware headers from the single C++ source of truth.
//
//   rtl/generated/apt_regmap.vh   CSR addresses           (from regmap.hpp)
//   rtl/generated/apt_params.vh   CSR reset values + timing constants (from a Config)
//   rtl/generated/apt_csr.vh      CSR write-enable decode and reset statements
//   fw/include/apt_regmap.h       the same, for C
//   fw/include/apt_params.h       verified dosing constants and the reset configuration
//   fw/include/apt_policy.h       the Module M optimal policy, quantised for the MCU
//
// Every emitted file carries the evidence ledger it was built from.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "apt/core_model.hpp"
#include "apt/evidence.hpp"
#include "apt/schedule.hpp"

namespace apt::codegen {

struct HardwareTiming {
    std::uint32_t clk_hz = 50'000'000;
    std::uint32_t ctrl_hz = 1000;
    std::uint32_t uart_div = 20;         // 2.5 Mbaud telemetry
    std::uint32_t adc_sclk_half = 2;     // SCLK = clk / 4
    std::uint32_t adc_conv_cycles = 200; // 4 us conversion
    std::uint32_t adc_timeout_cycles = 2000;
    std::uint32_t pwm_deadtime_cycles = 25;  // 500 ns
};

std::string verilog_regmap();
// is_rw() function and the CSR reset statements, included by rtl/apt_top.v
std::string verilog_csr_logic();
std::string verilog_params(const core::Config& reset_cfg, const HardwareTiming& t);
std::string c_regmap();
std::string c_params(const EvidenceBase& eb, const ScheduleModel& m, const core::Config& reset_cfg);
std::string c_policy(const ScheduleModel& m, const ScheduleModel::Solution& s, int c_stride,
                     int b_stride);

// Writes all five files under `root` (the apt/ directory); returns the paths written.
std::vector<std::filesystem::path> emit_all(const std::filesystem::path& root,
                                            const EvidenceBase& eb);

}  // namespace apt::codegen
