/*
 * apt_link.h - telemetry decoding on the MCU: CRC-16/CCITT-FALSE, streaming SLIP decoder and
 * the 42-byte payload layout of include/apt/link.hpp.
 */
#ifndef APT_LINK_H
#define APT_LINK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define APT_TLM_PAYLOAD 42u
#define APT_TLM_TYPE 0xA1u

typedef struct {
    uint32_t tick;
    uint16_t status;
    int32_t force, x_meas, v_meas, x_cmd, v_cmd, energy, duty;
    uint8_t spikes;
    uint16_t rep_count, lat_motor, lat_afferent;
} apt_telemetry_t;

typedef struct {
    uint8_t buf[64];
    size_t len;
    uint8_t esc;
    uint8_t overflow;
    uint32_t frames, crc_errors, length_errors;
} apt_slip_rx_t;

uint16_t apt_crc16(const uint8_t *data, size_t len, uint16_t crc);
void apt_slip_init(apt_slip_rx_t *rx);
/* Feed one byte; returns 1 and fills *out when a valid telemetry frame completes. */
int apt_slip_feed(apt_slip_rx_t *rx, uint8_t b, apt_telemetry_t *out);
int apt_telemetry_unpack(const uint8_t *p, size_t len, apt_telemetry_t *t);

#ifdef __cplusplus
}
#endif

#endif /* APT_LINK_H */
