#!/usr/bin/env bash
# Bit-exact RTL co-simulation: regenerate headers and golden vectors, then run Icarus Verilog.
#   scripts/sim.sh [BUILD_DIR] [TICKS]
# Needs iverilog >= 11 on PATH (or IVERILOG / VVP / IVL_LIBDIR in the environment).
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${1:-build}
TICKS=${2:-1200}
APTCTL=${APTCTL:-$BUILD/aptctl}
if [ -x "$APTCTL" ] || [ -x "$APTCTL.exe" ]; then
  "$APTCTL" codegen --root .
  "$APTCTL" vectors --out tb/vectors --ticks "$TICKS"
fi
IVERILOG=${IVERILOG:-iverilog}
VVP=${VVP:-vvp}
BFLAG=(); MFLAG=()
if [ -n "${IVL_LIBDIR:-}" ]; then BFLAG=(-B "$IVL_LIBDIR"); MFLAG=(-M "$IVL_LIBDIR"); fi
mkdir -p sim_build
"$IVERILOG" "${BFLAG[@]}" -g2012 -Wall -Wno-timescale -Wno-sensitivity-entire-array -I rtl/generated -s tb_apt_top -o sim_build/tb_apt_top.vvp \
  tb/tb_apt_top.v rtl/apt_top.v rtl/apt_core.v rtl/apt_udiv.v rtl/apt_spi_adc.v rtl/apt_quad.v \
  rtl/apt_pwm.v rtl/apt_stim.v rtl/apt_interlock.v rtl/apt_latency.v rtl/apt_spi_slave.v rtl/apt_telemetry.v rtl/apt_uart_tx.v
"$VVP" "${MFLAG[@]}" -n sim_build/tb_apt_top.vvp | tee sim_build/tb_apt_top.log
grep -q '^PASS' sim_build/tb_apt_top.log
