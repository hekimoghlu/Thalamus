/* mock_bus.c - CSR semantics of rtl/apt_top.v, reproduced in C for firmware tests. */
#include "mock_bus.h"

#include <string.h>

#include "apt_regmap.h"

void mock_init(mock_fpga_t *m)
{
    memset(m, 0, sizeof *m);
    m->regs[APT_REG_ID] = APT_DEVICE_ID;
    m->regs[APT_REG_VERSION] = APT_RTL_VERSION;
    m->corrupt_reg = -1;
}

static int is_ro(uint8_t a)
{
    return a == APT_REG_ID || a == APT_REG_VERSION || a == APT_REG_STATUS || a == APT_REG_TICK ||
           (a >= 0x40u && a <= 0x57u);
}

void mock_raise(mock_fpga_t *m, uint32_t cause)
{
    m->cause_present |= cause;
    m->latched |= cause;
}

void mock_drop(mock_fpga_t *m, uint32_t cause) { m->cause_present &= ~cause; }

int mock_xfer(void *ctx, const uint8_t tx[5], uint8_t rx[5])
{
    mock_fpga_t *m = (mock_fpga_t *)ctx;
    uint8_t a = (uint8_t)(tx[0] & 0x7Fu);
    uint32_t d = ((uint32_t)tx[1] << 24) | ((uint32_t)tx[2] << 16) | ((uint32_t)tx[3] << 8) | tx[4];
    uint32_t v;
    if (m->fail_next) {
        m->fail_next = 0;
        return -1;
    }
    rx[0] = 0u;
    if (tx[0] & 0x80u) {
        m->writes++;
        if (a == APT_REG_FAULT_CLEAR) {
            if (d & 1u) m->latched = m->cause_present;
        } else if (a == APT_REG_HEARTBEAT) {
            m->heartbeats++;
        } else if (!is_ro(a)) {
            m->regs[a] = d;
        }
        v = 0u;
    } else {
        m->reads++;
        if (a == APT_REG_STATUS) {
            uint32_t ctrl = m->regs[APT_REG_CTRL];
            v = (m->latched << 8) | ((ctrl & APT_CTRL_STIM) ? 8u : 0u) | (m->latched ? 2u : 0u) |
                (ctrl & APT_CTRL_ENABLE);
        } else {
            v = m->regs[a];
        }
        if ((int)a == m->corrupt_reg) v ^= 1u;
    }
    rx[1] = (uint8_t)(v >> 24);
    rx[2] = (uint8_t)(v >> 16);
    rx[3] = (uint8_t)(v >> 8);
    rx[4] = (uint8_t)v;
    return 0;
}
