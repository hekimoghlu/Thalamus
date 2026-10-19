// CRC-16/CCITT-FALSE, SLIP framing and telemetry packing.
#include <cstring>

#include "apt/link.hpp"
#include "apt/numerics.hpp"
#include "check.hpp"

using namespace apt;
using namespace apt::link;

TEST("CRC-16/CCITT-FALSE check value") {
    const char* s = "123456789";
    CHECK(crc16_ccitt({reinterpret_cast<const std::uint8_t*>(s), 9}) == 0x29B1);
}

TEST("SLIP escapes END and ESC and round-trips") {
    const std::vector<std::uint8_t> raw = {0x01, 0xC0, 0x02, 0xDB, 0xC0, 0xDB, 0x03};
    const auto enc = slip_encode(raw);
    CHECK(enc.front() == kSlipEnd && enc.back() == kSlipEnd);
    for (std::size_t i = 1; i + 1 < enc.size(); ++i) CHECK(enc[i] != kSlipEnd);
    SlipDecoder d;
    std::optional<std::vector<std::uint8_t>> got;
    for (auto b : enc)
        if (auto f = d.feed(b)) got = f;
    CHECK(got && *got == raw);
}

TEST("telemetry pack/unpack round trip and stream decode with corruption") {
    num::Rng rng(3);
    std::vector<std::uint8_t> stream;
    std::vector<Telemetry> sent;
    for (int i = 0; i < 200; ++i) {
        Telemetry t;
        t.tick = static_cast<std::uint32_t>(i);
        t.status = static_cast<std::uint16_t>(rng.uniform(0, 65535));
        t.force = static_cast<fx::sig_t>(rng.normal(0, 1e8));
        t.x_meas = static_cast<fx::sig_t>(0xC0C0C0C0u);  // worst case for SLIP stuffing
        t.energy = -static_cast<fx::sig_t>(0x00DBDBDB);
        t.spikes = static_cast<std::uint8_t>(i);
        t.rep_count = static_cast<std::uint16_t>(i * 7);
        CHECK(Telemetry::unpack(t.pack()) == t);
        auto f = t.frame();
        if (i == 50) f[10] ^= 0x01;  // corrupt one frame
        stream.insert(stream.end(), f.begin(), f.end());
        if (i != 50) sent.push_back(t);
    }
    DecodeStats st;
    const auto got = decode_stream(stream, &st);
    CHECK(st.crc_errors == 1);
    CHECK(got.size() == sent.size());
    CHECK(got == sent);
}

TEST("payload layout: 42 bytes, latency fields at the documented offset") {
    Telemetry t;
    t.lat_motor = 0x1234;
    t.lat_afferent = 0xABCD;
    const auto p = t.pack();
    CHECK(p.size() == 42);
    CHECK(p[kLatencyFieldOffset] == 0x34 && p[kLatencyFieldOffset + 1] == 0x12);
    CHECK(p[kLatencyFieldOffset + 2] == 0xCD && p[kLatencyFieldOffset + 3] == 0xAB);
}

TEST_MAIN()
