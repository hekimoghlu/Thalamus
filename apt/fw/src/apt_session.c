/* apt_session.c - therapy-session supervisor, safety policy and dose ledger. */
#include "apt_session.h"

#include "apt_params.h"
#define APT_POLICY_TABLE
#include "apt_policy.h"

#define CTRL_IDLE (APT_CTRL_RUN | APT_CTRL_TELEMETRY)
#define CTRL_ACTIVE (APT_CTRL_RUN | APT_CTRL_TELEMETRY | APT_CTRL_ENABLE | APT_CTRL_PASSIVITY | APT_CTRL_STIM)

const char *apt_state_name(apt_state_t st)
{
    switch (st) {
    case APT_S_BOOT: return "BOOT";
    case APT_S_IDLE: return "IDLE";
    case APT_S_ARMED: return "ARMED";
    case APT_S_ACTIVE: return "ACTIVE";
    case APT_S_FAULT: return "FAULT";
    case APT_S_LOCKOUT: return "LOCKOUT";
    default: return "?";
    }
}

static apt_status_t read_status(apt_session_t *s, uint32_t *status)
{
    return apt_read(s->bus, APT_REG_STATUS, status);
}

static void enter_fault(apt_session_t *s, uint32_t status)
{
    s->fault_causes = (status >> 8) & 0xFFu;
    s->faults_this_session++;
    /* the FPGA has already cut the drives; drop the enable so a clear cannot re-energise them */
    (void)apt_write(s->bus, APT_REG_CTRL, CTRL_IDLE);
    s->state = (s->faults_this_session >= APT_MAX_FAULTS_PER_SESSION) ? APT_S_LOCKOUT : APT_S_FAULT;
}

apt_status_t apt_session_boot(apt_session_t *s, apt_bus_t *bus, uint32_t week, double annual_budget_hours)
{
    apt_status_t rc;
    uint32_t v = 0u;
    s->bus = bus;
    s->state = APT_S_BOOT;
    s->last_hb_ms = 0u;
    s->session_start_ms = 0u;
    s->rep_base = 0u;
    s->fault_causes = 0u;
    s->faults_this_session = 0u;
    s->latency_worst_cycles = 0u;
    s->week.week_since_stroke = week;
    s->week.reps = 0u;
    s->week.active_ms = 0u;
    s->week.sessions = 0u;
    s->budget_hours_left = annual_budget_hours;

    rc = apt_configure(bus);
    if (rc != APT_OK) return rc;
    if (apt_read(bus, APT_REG_LAT_BUDGET, &v) != APT_OK) return APT_ERR_BUS;
    s->latency_budget_cycles = v;
    /* start the tick generator with the drives disabled: telemetry flows, nothing moves */
    if (apt_write(bus, APT_REG_CTRL, CTRL_IDLE) != APT_OK) return APT_ERR_BUS;
    if (apt_write(bus, APT_REG_FAULT_CLEAR, 1u) != APT_OK) return APT_ERR_BUS;
    s->state = APT_S_IDLE;
    return APT_OK;
}

apt_status_t apt_session_arm(apt_session_t *s)
{
    uint32_t status = 0u;
    if (s->state != APT_S_IDLE) return APT_ERR_STATE;
    if (read_status(s, &status) != APT_OK) return APT_ERR_BUS;
    if (status & 0x2u) return APT_ERR_FAULT;
    s->state = APT_S_ARMED;
    return APT_OK;
}

apt_status_t apt_session_start(apt_session_t *s, uint32_t now_ms)
{
    uint32_t reps = 0u;
    if (s->state != APT_S_ARMED) return APT_ERR_STATE;
    if (apt_heartbeat(s->bus) != APT_OK) return APT_ERR_BUS;
    if (apt_read(s->bus, APT_REG_REP_COUNT, &reps) != APT_OK) return APT_ERR_BUS;
    s->rep_base = (uint16_t)reps;
    s->session_start_ms = now_ms;
    s->last_hb_ms = now_ms;
    s->faults_this_session = 0u;
    if (apt_write(s->bus, APT_REG_CTRL, CTRL_ACTIVE) != APT_OK) return APT_ERR_BUS;
    s->state = APT_S_ACTIVE;
    return APT_OK;
}

static void account(apt_session_t *s, uint32_t now_ms)
{
    uint32_t reps = 0u;
    uint32_t elapsed = now_ms - s->session_start_ms;
    if (apt_read(s->bus, APT_REG_REP_COUNT, &reps) == APT_OK) {
        s->week.reps += (uint16_t)((uint16_t)reps - s->rep_base); /* 16-bit wrap-safe */
        s->rep_base = (uint16_t)reps;
    }
    s->week.active_ms += elapsed;
    s->session_start_ms = now_ms;
    s->budget_hours_left -= (double)elapsed / 3.6e6;
    if (s->budget_hours_left < 0.0) s->budget_hours_left = 0.0;
}

apt_status_t apt_session_stop(apt_session_t *s, uint32_t now_ms)
{
    uint32_t lat = 0u;
    if (s->state != APT_S_ACTIVE) return APT_ERR_STATE;
    if (apt_write(s->bus, APT_REG_CTRL, CTRL_IDLE) != APT_OK) return APT_ERR_BUS;
    account(s, now_ms);
    s->week.sessions++;
    /* latency audit: the specification is a budget, so a session that exceeded it is a defect */
    if (apt_read(s->bus, APT_REG_LAT_MOTOR_MAX, &lat) != APT_OK) return APT_ERR_BUS;
    if (lat > s->latency_worst_cycles) s->latency_worst_cycles = lat;
    if (s->latency_budget_cycles != 0u && lat > s->latency_budget_cycles) {
        s->state = APT_S_LOCKOUT;
        return APT_ERR_LATENCY;
    }
    s->state = APT_S_IDLE;
    return APT_OK;
}

apt_status_t apt_session_poll(apt_session_t *s, uint32_t now_ms)
{
    uint32_t status = 0u;
    if (s->state == APT_S_BOOT || s->state == APT_S_LOCKOUT) return APT_OK;
    if (read_status(s, &status) != APT_OK) {
        /* a supervisor that cannot see the device stops feeding the watchdog: the FPGA then
           faults on its own within 100 ms, which is the intended fail-safe */
        return APT_ERR_BUS;
    }
    if ((status & 0x2u) && s->state != APT_S_FAULT) {
        if (s->state == APT_S_ACTIVE) account(s, now_ms);
        enter_fault(s, status);
        return APT_ERR_FAULT;
    }
    if (now_ms - s->last_hb_ms >= APT_HEARTBEAT_PERIOD_MS) {
        if (apt_heartbeat(s->bus) != APT_OK) return APT_ERR_BUS;
        s->last_hb_ms = now_ms;
    }
    return APT_OK;
}

apt_status_t apt_session_ack_fault(apt_session_t *s)
{
    uint32_t status = 0u;
    if (s->state != APT_S_FAULT) return APT_ERR_STATE;
    if (apt_heartbeat(s->bus) != APT_OK) return APT_ERR_BUS;
    if (apt_write(s->bus, APT_REG_FAULT_CLEAR, 1u) != APT_OK) return APT_ERR_BUS;
    if (read_status(s, &status) != APT_OK) return APT_ERR_BUS;
    if (status & 0x2u) {
        s->fault_causes = (status >> 8) & 0xFFu; /* still present: stay in FAULT */
        return APT_ERR_FAULT;
    }
    s->state = APT_S_IDLE;
    return APT_OK;
}

void apt_session_new_week(apt_session_t *s, uint32_t week)
{
    s->week.week_since_stroke = week;
    s->week.reps = 0u;
    s->week.active_ms = 0u;
    s->week.sessions = 0u;
}

/* ---------------------------------------------------------------------------- dosing */

double apt_dose_relative(const apt_dose_week_t *w)
{
    return 1.0 + (double)w->reps / APT_REPS_USUAL_PER_SESSION;
}

int apt_dose_above_threshold(const apt_dose_week_t *w)
{
    return apt_dose_relative(w) >= APT_DOSE_THRESHOLD_REL;
}

uint32_t apt_dose_reps_to_threshold(const apt_dose_week_t *w)
{
    const double need = (APT_DOSE_THRESHOLD_REL - 1.0) * APT_REPS_USUAL_PER_SESSION;
    const double have = (double)w->reps;
    if (have >= need) return 0u;
    return (uint32_t)(need - have + 0.999999);
}

double apt_policy_hours(uint32_t week, double ceiling, double budget_left)
{
    /* floor lookup on the coarse grids, as Module M's rollout reads its policy */
    int ic, ib;
    double c_step = (APT_POLICY_C_MAX - APT_POLICY_C_MIN) / (double)(APT_POLICY_NC - 1);
    double b_step = APT_POLICY_BUDGET_H / (double)(APT_POLICY_NB - 1);
    double h;
    if (week >= APT_POLICY_WEEKS) week = APT_POLICY_WEEKS - 1u;
    ic = (int)((ceiling - APT_POLICY_C_MIN) / c_step);
    ib = (int)(budget_left / b_step);
    if (ic < 0) ic = 0;
    if (ic > APT_POLICY_NC - 1) ic = APT_POLICY_NC - 1;
    if (ib < 0) ib = 0;
    if (ib > APT_POLICY_NB - 1) ib = APT_POLICY_NB - 1;
    h = (double)APT_POLICY_Q4[week][ic][ib] / 4.0;
    return h < budget_left ? h : budget_left;
}
