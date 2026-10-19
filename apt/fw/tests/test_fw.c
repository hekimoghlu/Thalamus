/* test_fw.c - host tests of the supervisory firmware against the CSR mock. */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "apt_hal.h"
#include "apt_link.h"
#include "apt_params.h"
#include "apt_policy.h"
#include "apt_session.h"
#include "mock_bus.h"

static int failures = 0, passes = 0;
#define EXPECT(c)                                                            \
    do {                                                                     \
        if (c) ++passes;                                                     \
        else { ++failures; printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } \
    } while (0)

static void boot(mock_fpga_t *m, apt_bus_t *bus, apt_session_t *s)
{
    mock_init(m);
    apt_bus_init(bus, mock_xfer, m);
    EXPECT(apt_session_boot(s, bus, 8u, 60.0) == APT_OK);
}

static void test_configure(void)
{
    mock_fpga_t m;
    apt_bus_t bus;
    apt_session_t s;
    boot(&m, &bus, &s);
    EXPECT(s.state == APT_S_IDLE);
    EXPECT(m.regs[APT_REG_ADM_CF] != 0u);   /* configuration landed */
    EXPECT((m.regs[APT_REG_CTRL] & APT_CTRL_ENABLE) == 0u);  /* booted disarmed */
    EXPECT(m.regs[APT_REG_CTRL] & APT_CTRL_RUN);

    /* wrong device */
    mock_init(&m);
    m.regs[APT_REG_ID] = 0xDEADBEEFu;
    apt_bus_init(&bus, mock_xfer, &m);
    EXPECT(apt_session_boot(&s, &bus, 8u, 60.0) == APT_ERR_ID);

    /* a register that does not read back is a configuration failure, not a warning */
    mock_init(&m);
    m.corrupt_reg = APT_REG_KP;
    apt_bus_init(&bus, mock_xfer, &m);
    EXPECT(apt_session_boot(&s, &bus, 8u, 60.0) == APT_ERR_VERIFY);
    printf("[ ok ] configure: identity, reset image, read-back verification\n");
}

static void test_session_and_faults(void)
{
    mock_fpga_t m;
    apt_bus_t bus;
    apt_session_t s;
    uint32_t t = 1000u, hb0;
    boot(&m, &bus, &s);
    EXPECT(apt_session_start(&s, t) == APT_ERR_STATE);  /* must arm first */
    EXPECT(apt_session_arm(&s) == APT_OK);
    EXPECT(apt_session_start(&s, t) == APT_OK);
    EXPECT(m.regs[APT_REG_CTRL] & APT_CTRL_ENABLE);

    hb0 = m.heartbeats;
    for (t = 1000u; t < 2000u; t += 10u) EXPECT(apt_session_poll(&s, t) == APT_OK);
    EXPECT(m.heartbeats - hb0 >= 45u);  /* >= 50 Hz against a 100 ms hardware watchdog */

    /* a force-limit fault latches: supervisor drops the enable and refuses to clear */
    mock_raise(&m, APT_FAULT_FORCE);
    EXPECT(apt_session_poll(&s, t) == APT_ERR_FAULT);
    EXPECT(s.state == APT_S_FAULT);
    EXPECT((m.regs[APT_REG_CTRL] & APT_CTRL_ENABLE) == 0u);
    EXPECT(s.fault_causes == APT_FAULT_FORCE);
    EXPECT(apt_session_ack_fault(&s) == APT_ERR_FAULT);  /* cause still present */
    mock_drop(&m, APT_FAULT_FORCE);
    EXPECT(apt_session_ack_fault(&s) == APT_OK);
    EXPECT(s.state == APT_S_IDLE);

    /* three faults in one session lock the device out */
    EXPECT(apt_session_arm(&s) == APT_OK);
    EXPECT(apt_session_start(&s, t) == APT_OK);
    {
        int i;
        for (i = 0; i < 3; ++i) {
            mock_raise(&m, APT_FAULT_ESTOP);
            (void)apt_session_poll(&s, t += 10u);
            mock_drop(&m, APT_FAULT_ESTOP);
            if (s.state == APT_S_FAULT) {
                EXPECT(apt_session_ack_fault(&s) == APT_OK);
                EXPECT(apt_session_arm(&s) == APT_OK);
                s.state = APT_S_ACTIVE; /* resume the same session */
            }
        }
    }
    EXPECT(s.state == APT_S_LOCKOUT);
    printf("[ ok ] session: arming order, heartbeat rate, fault latch, ack only when clear, lockout\n");
}

static void test_bus_loss_starves_watchdog(void)
{
    mock_fpga_t m;
    apt_bus_t bus;
    apt_session_t s;
    uint32_t hb0;
    boot(&m, &bus, &s);
    EXPECT(apt_session_arm(&s) == APT_OK);
    EXPECT(apt_session_start(&s, 0u) == APT_OK);
    hb0 = m.heartbeats;
    m.fail_next = 1;
    EXPECT(apt_session_poll(&s, 100u) == APT_ERR_BUS);
    EXPECT(m.heartbeats == hb0);  /* no heartbeat from a supervisor that cannot see the device */
    printf("[ ok ] bus loss: supervisor stops feeding the hardware watchdog\n");
}

static void test_latency_audit(void)
{
    mock_fpga_t m;
    apt_bus_t bus;
    apt_session_t s;
    boot(&m, &bus, &s);
    EXPECT(s.latency_budget_cycles == 5000u);
    EXPECT(apt_session_arm(&s) == APT_OK);
    EXPECT(apt_session_start(&s, 0u) == APT_OK);
    m.regs[APT_REG_LAT_MOTOR_MAX] = 2720u;  /* the RTL simulation's measured worst case */
    EXPECT(apt_session_stop(&s, 600000u) == APT_OK);
    EXPECT(apt_session_arm(&s) == APT_OK);
    EXPECT(apt_session_start(&s, 600000u) == APT_OK);
    m.regs[APT_REG_LAT_MOTOR_MAX] = 6000u;
    EXPECT(apt_session_stop(&s, 700000u) == APT_ERR_LATENCY);
    EXPECT(s.state == APT_S_LOCKOUT);
    printf("[ ok ] latency audit: a session over the budget locks the device out\n");
}

static void test_dose_ledger(void)
{
    mock_fpga_t m;
    apt_bus_t bus;
    apt_session_t s;
    boot(&m, &bus, &s);
    /* the verified threshold: 1.72x usual care -> 23 repetitions per week, ~4.3 min at 322/h */
    EXPECT(fabs(APT_DOSE_THRESHOLD_REL - 1.7233) < 1e-3);
    EXPECT(fabs(APT_THRESHOLD_MIN_PER_WEEK_DEVICE - 4.31) < 0.05);
    EXPECT(fabs(APT_THRESHOLD_MIN_PER_WEEK_USUAL - 43.4) < 0.5);
    EXPECT(apt_dose_reps_to_threshold(&s.week) == 24u);
    EXPECT(!apt_dose_above_threshold(&s.week));

    EXPECT(apt_session_arm(&s) == APT_OK);
    EXPECT(apt_session_start(&s, 0u) == APT_OK);
    m.regs[APT_REG_REP_COUNT] = 15u;
    EXPECT(apt_session_stop(&s, 180000u) == APT_OK);  /* 3 minutes */
    EXPECT(s.week.reps == 15u);
    EXPECT(!apt_dose_above_threshold(&s.week));
    EXPECT(apt_dose_reps_to_threshold(&s.week) == 9u);
    EXPECT(apt_session_arm(&s) == APT_OK);
    EXPECT(apt_session_start(&s, 200000u) == APT_OK);
    m.regs[APT_REG_REP_COUNT] = 40u;
    EXPECT(apt_session_stop(&s, 290000u) == APT_OK);
    EXPECT(s.week.reps == 40u);
    EXPECT(apt_dose_above_threshold(&s.week));
    EXPECT(s.week.sessions == 2u);
    EXPECT(fabs(s.budget_hours_left - (60.0 - 270000.0 / 3.6e6)) < 1e-9);

    /* 16-bit repetition counter wraps without losing count */
    s.week.reps = 0u;
    EXPECT(apt_session_arm(&s) == APT_OK);
    m.regs[APT_REG_REP_COUNT] = 65530u;
    EXPECT(apt_session_start(&s, 300000u) == APT_OK);
    m.regs[APT_REG_REP_COUNT] = 4u;
    EXPECT(apt_session_stop(&s, 310000u) == APT_OK);
    EXPECT(s.week.reps == 10u);
    printf("[ ok ] dose ledger: verified threshold, weekly repetitions, budget, counter wrap\n");
}

static void test_policy(void)
{
    /* the generated table is the Module M device policy at a 60 h budget */
    double early = apt_policy_hours(6u, APT_POLICY_C_MIN, 60.0);
    double late = apt_policy_hours(50u, APT_POLICY_C_MIN, 60.0);
    double none = apt_policy_hours(6u, APT_POLICY_C_MIN, 0.0);
    EXPECT(early >= 0.0 && early <= 8.0);
    EXPECT(none == 0.0);
    EXPECT(apt_policy_hours(6u, APT_POLICY_C_MIN, 0.1) <= 0.1);  /* never beyond the budget */
    EXPECT(late >= 0.0);
    printf("[ ok ] policy table lookup (week 6: %.2f h, week 50: %.2f h)\n", early, late);
}

static void test_link(void)
{
    const uint8_t check[] = "123456789";
    uint8_t payload[APT_TLM_PAYLOAD], frame[128];
    size_t n = 0, i;
    uint16_t crc;
    apt_slip_rx_t rx;
    apt_telemetry_t t;
    int got = 0;
    EXPECT(apt_crc16(check, 9, 0xFFFFu) == 0x29B1u);
    memset(payload, 0, sizeof payload);
    payload[0] = APT_TLM_TYPE;
    payload[1] = 0x2A;                     /* tick 42 */
    payload[7] = 0xC0; payload[8] = 0xDB;  /* force field with both special bytes */
    payload[36] = 7;                       /* reps */
    crc = apt_crc16(payload, APT_TLM_PAYLOAD, 0xFFFFu);
    frame[n++] = 0xC0;
    for (i = 0; i < APT_TLM_PAYLOAD + 2u; ++i) {
        uint8_t b = (i < APT_TLM_PAYLOAD) ? payload[i] : (i == APT_TLM_PAYLOAD ? (uint8_t)crc : (uint8_t)(crc >> 8));
        if (b == 0xC0) { frame[n++] = 0xDB; frame[n++] = 0xDC; }
        else if (b == 0xDB) { frame[n++] = 0xDB; frame[n++] = 0xDD; }
        else frame[n++] = b;
    }
    frame[n++] = 0xC0;
    apt_slip_init(&rx);
    for (i = 0; i < n; ++i) got |= apt_slip_feed(&rx, frame[i], &t);
    EXPECT(got && t.tick == 42u && t.rep_count == 7u && (uint32_t)t.force == 0xDBC0u);
    frame[5] ^= 1u;  /* corrupt and replay */
    got = 0;
    for (i = 0; i < n; ++i) got |= apt_slip_feed(&rx, frame[i], &t);
    EXPECT(!got && rx.crc_errors == 1u);
    printf("[ ok ] link: CRC check value, SLIP escapes, corrupted frame rejected\n");
}

int main(void)
{
    test_configure();
    test_session_and_faults();
    test_bus_loss_starves_watchdog();
    test_latency_audit();
    test_dose_ledger();
    test_policy();
    test_link();
    printf("%d checks passed, %d failed\n", passes, failures);
    return failures ? 1 : 0;
}
