#include "apt/link.hpp"

namespace apt::link {

std::uint16_t crc16_ccitt(std::span<const std::uint8_t> data, std::uint16_t crc) {
    for (std::uint8_t b : data) {
        crc = static_cast<std::uint16_t>(crc ^ (std::uint16_t{b} << 8));
        for (int i = 0; i < 8; ++i)
            crc = static_cast<std::uint16_t>(crc & 0x8000 ? (crc << 1) ^ 0x1021 : crc << 1);
    }
    return crc;
}

std::vector<std::uint8_t> slip_encode(std::span<const std::uint8_t> frame) {
    std::vector<std::uint8_t> out;
    out.reserve(frame.size() + 8);
    out.push_back(kSlipEnd);
    for (std::uint8_t b : frame) {
        if (b == kSlipEnd) {
            out.push_back(kSlipEsc);
            out.push_back(kSlipEscEnd);
        } else if (b == kSlipEsc) {
            out.push_back(kSlipEsc);
            out.push_back(kSlipEscEsc);
        } else {
            out.push_back(b);
        }
    }
    out.push_back(kSlipEnd);
    return out;
}

std::optional<std::vector<std::uint8_t>> SlipDecoder::feed(std::uint8_t b) {
    if (b == kSlipEnd) {
        esc_ = false;
        if (buf_.empty()) return std::nullopt;
        auto f = std::move(buf_);
        buf_.clear();
        return f;
    }
    if (esc_) {
        esc_ = false;
        if (b == kSlipEscEnd) buf_.push_back(kSlipEnd);
        else if (b == kSlipEscEsc) buf_.push_back(kSlipEsc);
        else ++errors_, buf_.push_back(b);
        return std::nullopt;
    }
    if (b == kSlipEsc) esc_ = true;
    else buf_.push_back(b);
    return std::nullopt;
}

namespace {
template <class T>
void put(std::uint8_t*& p, T v) {
    using U = std::make_unsigned_t<T>;
    const U u = static_cast<U>(v);
    for (std::size_t i = 0; i < sizeof(T); ++i) *p++ = static_cast<std::uint8_t>(u >> (8 * i));
}
template <class T>
T get(const std::uint8_t*& p) {
    using U = std::make_unsigned_t<T>;
    U u = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i) u = static_cast<U>(u | (U(*p++) << (8 * i)));
    return static_cast<T>(u);
}
}  // namespace

std::array<std::uint8_t, kTelemetryPayload> Telemetry::pack() const {
    std::array<std::uint8_t, kTelemetryPayload> a{};
    std::uint8_t* p = a.data();
    put<std::uint8_t>(p, kTelemetryType);
    put(p, tick);
    put(p, status);
    for (fx::sig_t v : {force, x_meas, v_meas, x_cmd, v_cmd, energy, duty}) put(p, v);
    put(p, spikes);
    put(p, rep_count);
    put(p, lat_motor);
    put(p, lat_afferent);
    return a;
}

std::optional<Telemetry> Telemetry::unpack(std::span<const std::uint8_t> payload) {
    if (payload.size() != kTelemetryPayload || payload[0] != kTelemetryType) return std::nullopt;
    const std::uint8_t* p = payload.data() + 1;
    Telemetry t;
    t.tick = get<std::uint32_t>(p);
    t.status = get<std::uint16_t>(p);
    for (fx::sig_t* v : {&t.force, &t.x_meas, &t.v_meas, &t.x_cmd, &t.v_cmd, &t.energy, &t.duty})
        *v = get<std::int32_t>(p);
    t.spikes = get<std::uint8_t>(p);
    t.rep_count = get<std::uint16_t>(p);
    t.lat_motor = get<std::uint16_t>(p);
    t.lat_afferent = get<std::uint16_t>(p);
    return t;
}

std::vector<std::uint8_t> Telemetry::frame() const {
    const auto pl = pack();
    std::vector<std::uint8_t> raw(pl.begin(), pl.end());
    const std::uint16_t crc = crc16_ccitt(pl);
    raw.push_back(static_cast<std::uint8_t>(crc & 0xFF));
    raw.push_back(static_cast<std::uint8_t>(crc >> 8));
    return slip_encode(raw);
}

std::vector<Telemetry> decode_stream(std::span<const std::uint8_t> bytes, DecodeStats* stats) {
    SlipDecoder dec;
    std::vector<Telemetry> out;
    DecodeStats st;
    for (std::uint8_t b : bytes) {
        auto f = dec.feed(b);
        if (!f) continue;
        if (f->size() != kTelemetryPayload + 2) {
            ++st.length_errors;
            continue;
        }
        const std::uint16_t got = static_cast<std::uint16_t>((*f)[kTelemetryPayload] |
                                                             ((*f)[kTelemetryPayload + 1] << 8));
        if (crc16_ccitt(std::span(*f).first(kTelemetryPayload)) != got) {
            ++st.crc_errors;
            continue;
        }
        if (auto t = Telemetry::unpack(std::span(*f).first(kTelemetryPayload))) {
            out.push_back(*t);
            ++st.frames;
        }
    }
    if (stats) *stats = st;
    return out;
}

}  // namespace apt::link
