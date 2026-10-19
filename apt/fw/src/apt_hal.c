/* apt_hal.c - SPI CSR access, configuration with read-back verification. */
#include "apt_hal.h"

#define APT_PARAMS_RESET_CONFIG
#include "apt_params.h"

void apt_bus_init(apt_bus_t *bus, apt_xfer_fn xfer, void *ctx)
{
    bus->xfer = xfer;
    bus->ctx = ctx;
    bus->transfers = 0u;
    bus->errors = 0u;
}

static apt_status_t xfer(apt_bus_t *bus, uint8_t hdr, uint32_t wdata, uint32_t *rdata)
{
    uint8_t tx[5];
    uint8_t rx[5] = {0u, 0u, 0u, 0u, 0u};
    tx[0] = hdr;
    tx[1] = (uint8_t)(wdata >> 24);
    tx[2] = (uint8_t)(wdata >> 16);
    tx[3] = (uint8_t)(wdata >> 8);
    tx[4] = (uint8_t)wdata;
    bus->transfers++;
    if (bus->xfer(bus->ctx, tx, rx) != 0) {
        bus->errors++;
        return APT_ERR_BUS;
    }
    if (rdata != 0) {
        *rdata = ((uint32_t)rx[1] << 24) | ((uint32_t)rx[2] << 16) | ((uint32_t)rx[3] << 8) | (uint32_t)rx[4];
    }
    return APT_OK;
}

apt_status_t apt_write(apt_bus_t *bus, uint8_t addr, uint32_t value)
{
    return xfer(bus, (uint8_t)(0x80u | (addr & 0x7Fu)), value, 0);
}

apt_status_t apt_read(apt_bus_t *bus, uint8_t addr, uint32_t *value)
{
    return xfer(bus, (uint8_t)(addr & 0x7Fu), 0u, value);
}

apt_status_t apt_configure(apt_bus_t *bus)
{
    uint32_t v = 0u;
    size_t i;
    if (apt_read(bus, APT_REG_ID, &v) != APT_OK) return APT_ERR_BUS;
    if (v != APT_DEVICE_ID) return APT_ERR_ID;
    if (apt_read(bus, APT_REG_VERSION, &v) != APT_OK) return APT_ERR_BUS;
    if ((v >> 16) != (APT_RTL_VERSION >> 16)) return APT_ERR_ID; /* major version must match */

    /* disarm before touching anything else */
    if (apt_write(bus, APT_REG_CTRL, 0u) != APT_OK) return APT_ERR_BUS;
    for (i = 0; i < APT_RESET_CONFIG_LEN; ++i) {
        if (apt_write(bus, APT_RESET_CONFIG[i].addr, APT_RESET_CONFIG[i].value) != APT_OK) return APT_ERR_BUS;
    }
    for (i = 0; i < APT_RESET_CONFIG_LEN; ++i) {
        if (apt_read(bus, APT_RESET_CONFIG[i].addr, &v) != APT_OK) return APT_ERR_BUS;
        if (v != APT_RESET_CONFIG[i].value) return APT_ERR_VERIFY;
    }
    return APT_OK;
}

apt_status_t apt_heartbeat(apt_bus_t *bus)
{
    return apt_write(bus, APT_REG_HEARTBEAT, 1u);
}
