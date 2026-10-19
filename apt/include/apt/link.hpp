// apt/link.hpp - the telemetry link from the FPGA to the host.
//
// One frame per control tick: a 42-byte little-endian payload, CRC-16/CCITT-FALSE appended
// (little-endian), then SLIP-framed (RFC 1055: END = 0xC0 before and after, ESC = 0xDB). SLIP
// is used rather than COBS because it encodes as a pure byte stream with no look-ahead, which
// is what a 1 kHz FPGA framer wants.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "apt/fixed.hpp"

namespace apt::link {

std::uint16_t crc16_ccitt(std::span<const std::uint8_t> data, std::uint16_t crc = 0xFFFF);

inline constexpr std::uint8_t kSlipEnd = 0xC0, kSlipEsc = 0xDB, kSlipEscEnd = 0xDC,
                              kSlipEscEsc = 0xDD;

std::vector<std::uint8_t> slip_encode(std::span<const std::uint8_t> frame);

// Streaming decoder: feed bytes, receive complete frames (empty frames between ENDs are skipped).
class SlipDecoder {
public:
    std::optional<std::vector<std::uint8_t>> feed(std::uint8_t b);
    std::size_t protocol_errors() const { return errors_; }

private:
    std::vector<std::uint8_t> buf_;
    bool esc_ = false;
    std::size_t errors_ = 0;
};

inline constexpr std::uint8_t kTelemetryType = 0xA1;
inline constexpr std::size_t kTelemetryPayload = 42;

struct Telemetry {
    std::uint32_t tick = 0;
    std::uint16_t status = 0;
    fx::sig_t force = 0, x_meas = 0, v_meas = 0, x_cmd = 0, v_cmd = 0, energy = 0, duty = 0;
    std::uint8_t spikes = 0;  // b0..3 afferent events, b4..7 gated (second-response) flags
    std::uint16_t rep_count = 0, lat_motor = 0, lat_afferent = 0;

    std::array<std::uint8_t, kTelemetryPayload> pack() const;
    static std::optional<Telemetry> unpack(std::span<const std::uint8_t> payload);
    // full wire frame: SLIP(payload || crc16_le)
    std::vector<std::uint8_t> frame() const;
    bool operator==(const Telemetry&) const = default;
};

// Offsets of the two latency fields, which depend on RTL cycle timing and are masked when the
// golden model's frames are compared with the simulated RTL's.
inline constexpr std::size_t kLatencyFieldOffset = 38, kLatencyFieldBytes = 4;

// Decode a byte stream into validated telemetry frames (CRC checked).
struct DecodeStats {
    std::size_t frames = 0, crc_errors = 0, length_errors = 0;
};
std::vector<Telemetry> decode_stream(std::span<const std::uint8_t> bytes, DecodeStats* stats = nullptr);

}  // namespace apt::link
