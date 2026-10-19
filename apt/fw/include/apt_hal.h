/*
 * apt_hal.h - supervisory-MCU access to the APT FPGA over the SPI CSR bus.
 *
 * One transaction is 5 bytes, MSB first: [RW|addr6..0][d31..24][d23..16][d15..8][d7..0].
 * The board layer supplies a single full-duplex transfer function; everything else (fixed-point
 * conversion, configuration with read-back verification, heartbeat) lives here and is
 * host-testable against fw/tests/mock_bus.c.
 *
 * C99, no dynamic allocation, no floating point on the control path (conversion helpers only).
 */
#ifndef APT_HAL_H
#define APT_HAL_H

#include <stdint.h>

#include "apt_regmap.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*apt_xfer_fn)(void *ctx, const uint8_t tx[5], uint8_t rx[5]);

typedef struct {
    apt_xfer_fn xfer;
    void *ctx;
    uint32_t transfers;
    uint32_t errors;
} apt_bus_t;

typedef enum {
    APT_OK = 0,
    APT_ERR_BUS = -1,
    APT_ERR_ID = -2,
    APT_ERR_VERIFY = -3,
    APT_ERR_FAULT = -4,
    APT_ERR_STATE = -5,
    APT_ERR_LATENCY = -6
} apt_status_t;

/* Q12.20 signals and Q8.24 coefficients, as in include/apt/fixed.hpp */
#define APT_SIG_ONE (1L << 20)
#define APT_COEF_ONE (1L << 24)
static inline int32_t apt_to_sig(double v) { return (int32_t)(v * (double)APT_SIG_ONE + (v < 0 ? -0.5 : 0.5)); }
static inline double apt_from_sig(int32_t v) { return (double)v / (double)APT_SIG_ONE; }
static inline int32_t apt_to_coef(double v) { return (int32_t)(v * (double)APT_COEF_ONE + (v < 0 ? -0.5 : 0.5)); }

void apt_bus_init(apt_bus_t *bus, apt_xfer_fn xfer, void *ctx);
apt_status_t apt_write(apt_bus_t *bus, uint8_t addr, uint32_t value);
apt_status_t apt_read(apt_bus_t *bus, uint8_t addr, uint32_t *value);

/* Check identity, write the generated reset configuration, read every register back. */
apt_status_t apt_configure(apt_bus_t *bus);
apt_status_t apt_heartbeat(apt_bus_t *bus);

#ifdef __cplusplus
}
#endif

#endif /* APT_HAL_H */
