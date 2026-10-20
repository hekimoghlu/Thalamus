/*
 * apt_session.h - therapy-session supervisor for the Artificial Peripheral Thalamus.
 *
 *   BOOT --configure+selftest--> IDLE --arm--> ARMED --start--> ACTIVE --stop--> IDLE
 *                                  ^                               |
 *                                  +------- ack (cause gone) ---- FAULT <-- any latched cause
 *   three faults in one session, or a latency-budget violation  ->  LOCKOUT (service required)
 *
 * The FPGA is the safety layer (it latches faults and cuts the drives within nanoseconds); this
 * supervisor is the policy layer: it keeps the hardware watchdog fed only while it is itself
 * healthy, refuses to re-arm while a cause persists, audits the measured loop latency, and keeps
 * the dose ledger that the framework's Module M says matters - weekly repetitions against the
 * verified dose threshold (1.72x usual care), below which extra therapy buys nothing.
 */
#ifndef APT_SESSION_H
#define APT_SESSION_H

#include <stdint.h>

#include "apt_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    APT_S_BOOT = 0,
    APT_S_IDLE,
    APT_S_ARMED,
    APT_S_ACTIVE,
    APT_S_FAULT,
    APT_S_LOCKOUT
} apt_state_t;

typedef struct {
    uint32_t week_since_stroke;
    uint32_t reps;           /* completed reach-and-return repetitions this week */
    uint32_t active_ms;      /* enabled time this week */
    uint32_t sessions;
} apt_dose_week_t;

typedef struct {
    apt_bus_t *bus;
    apt_state_t state;
    uint32_t last_hb_ms;
    uint32_t session_start_ms;
    uint16_t rep_base;
    uint32_t fault_causes;      /* STATUS[15:8] at the last fault */
    uint32_t faults_this_session;
    uint32_t latency_budget_cycles;
    uint32_t latency_worst_cycles;
    apt_dose_week_t week;
    double budget_hours_left;   /* remaining annual therapy budget */
} apt_session_t;

#define APT_HEARTBEAT_PERIOD_MS 20u /* the FPGA watchdog is 100 ticks = 100 ms */
#define APT_MAX_FAULTS_PER_SESSION 3u

apt_status_t apt_session_boot(apt_session_t *s, apt_bus_t *bus, uint32_t week_since_stroke,
                              double annual_budget_hours);
apt_status_t apt_session_arm(apt_session_t *s);
apt_status_t apt_session_start(apt_session_t *s, uint32_t now_ms);
apt_status_t apt_session_stop(apt_session_t *s, uint32_t now_ms);
/* Call at >= 50 Hz: feeds the watchdog while healthy and moves to FAULT on any latched cause. */
apt_status_t apt_session_poll(apt_session_t *s, uint32_t now_ms);
/* Acknowledge a fault: clears it only if the cause is gone. */
apt_status_t apt_session_ack_fault(apt_session_t *s);
void apt_session_new_week(apt_session_t *s, uint32_t week_since_stroke);
const char *apt_state_name(apt_state_t st);

/* --- dosing (Module M) -------------------------------------------------------------------- */
double apt_dose_relative(const apt_dose_week_t *w);       /* R = 1 + reps / reps per usual session */
int apt_dose_above_threshold(const apt_dose_week_t *w);
uint32_t apt_dose_reps_to_threshold(const apt_dose_week_t *w);
/* Optimal hours this week from the Module M policy (device delivery, quantised table). */
double apt_policy_hours(uint32_t week_since_stroke, double ceiling, double budget_hours_left);

#ifdef __cplusplus
}
#endif

#endif /* APT_SESSION_H */
