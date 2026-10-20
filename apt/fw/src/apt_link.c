/* apt_link.c - CRC-16, SLIP and the telemetry payload, byte-compatible with link.cpp. */
#include "apt_link.h"

uint16_t apt_crc16(const uint8_t *data, size_t len, uint16_t crc)
{
    size_t i;
    int k;
    for (i = 0; i < len; ++i) {
        crc = (uint16_t)(crc ^ ((uint16_t)data[i] << 8));
        for (k = 0; k < 8; ++k) {
            crc = (uint16_t)((crc & 0x8000u) ? ((crc << 1) ^ 0x1021u) : (crc << 1));
        }
    }
    return crc;
}

void apt_slip_init(apt_slip_rx_t *rx)
{
    rx->len = 0u;
    rx->esc = 0u;
    rx->overflow = 0u;
    rx->frames = 0u;
    rx->crc_errors = 0u;
    rx->length_errors = 0u;
}

static uint32_t le32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t le16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

int apt_telemetry_unpack(const uint8_t *p, size_t len, apt_telemetry_t *t)
{
    if (len != APT_TLM_PAYLOAD || p[0] != APT_TLM_TYPE) return 0;
    t->tick = le32(p + 1);
    t->status = le16(p + 5);
    t->force = (int32_t)le32(p + 7);
    t->x_meas = (int32_t)le32(p + 11);
    t->v_meas = (int32_t)le32(p + 15);
    t->x_cmd = (int32_t)le32(p + 19);
    t->v_cmd = (int32_t)le32(p + 23);
    t->energy = (int32_t)le32(p + 27);
    t->duty = (int32_t)le32(p + 31);
    t->spikes = p[35];
    t->rep_count = le16(p + 36);
    t->lat_motor = le16(p + 38);
    t->lat_afferent = le16(p + 40);
    return 1;
}

int apt_slip_feed(apt_slip_rx_t *rx, uint8_t b, apt_telemetry_t *out)
{
    if (b == 0xC0u) {
        int ok = 0;
        if (rx->len > 0u && !rx->overflow) {
            if (rx->len != APT_TLM_PAYLOAD + 2u) {
                rx->length_errors++;
            } else if (apt_crc16(rx->buf, APT_TLM_PAYLOAD, 0xFFFFu) != le16(rx->buf + APT_TLM_PAYLOAD)) {
                rx->crc_errors++;
            } else if (apt_telemetry_unpack(rx->buf, APT_TLM_PAYLOAD, out)) {
                rx->frames++;
                ok = 1;
            }
        } else if (rx->overflow) {
            rx->length_errors++;
        }
        rx->len = 0u;
        rx->esc = 0u;
        rx->overflow = 0u;
        return ok;
    }
    if (b == 0xDBu && !rx->esc) {
        rx->esc = 1u;
        return 0;
    }
    if (rx->esc) {
        b = (b == 0xDCu) ? 0xC0u : (b == 0xDDu) ? 0xDBu : b;
        rx->esc = 0u;
    }
    if (rx->len < sizeof rx->buf) rx->buf[rx->len++] = b;
    else rx->overflow = 1u;
    return 0;
}
