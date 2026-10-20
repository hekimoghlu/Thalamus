/* mock_bus.h - a software model of the FPGA's CSR side of the SPI protocol, for host tests. */
#ifndef MOCK_BUS_H
#define MOCK_BUS_H

#include <stdint.h>

typedef struct {
    uint32_t regs[128];
    uint32_t cause_present;   /* fault causes currently asserted by the "hardware" */
    uint32_t latched;         /* latched fault causes */
    uint32_t heartbeats;
    uint32_t writes, reads;
    int fail_next;            /* make the next transfer fail */
    int corrupt_reg;          /* address whose read-back is corrupted (-1: none) */
} mock_fpga_t;

void mock_init(mock_fpga_t *m);
int mock_xfer(void *ctx, const uint8_t tx[5], uint8_t rx[5]);
void mock_raise(mock_fpga_t *m, uint32_t cause);
void mock_drop(mock_fpga_t *m, uint32_t cause);

#endif
