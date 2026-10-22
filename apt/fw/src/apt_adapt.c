/* apt_adapt.c - delay estimator and certified inner-gain schedule. */
#include "apt_adapt.h"

#define APT_PARAMS_RESET_CONFIG
#include "apt_params.h"
#define APT_ADAPT_TABLE
#include "apt_adapt_table.h"

void apt_adapt_init(apt_adapt_t *a)
{
    size_t i;
    a->base_kp = a->base_kd = 0u;
    for (i = 0; i < APT_RESET_CONFIG_LEN; ++i) {
        if (APT_RESET_CONFIG[i].addr == APT_REG_KP) a->base_kp = APT_RESET_CONFIG[i].value;
        if (APT_RESET_CONFIG[i].addr == APT_REG_KD) a->base_kd = APT_RESET_CONFIG[i].value;
    }
    a->smooth_us = a->est_us = 0u;
    a->scale_q16 = a->cand_q16 = 65536u;
    a->held_ms = 0u;
    a->changes = 0u;
    a->beyond_certified = 0;
    a->primed = 0;
}

uint32_t apt_adapt_lookup(uint32_t delay_us, int *covered)
{
    unsigned i;
    for (i = 0; i < APT_ADAPT_BINS; ++i) {
        if (delay_us <= (uint32_t)APT_ADAPT_EDGE_MS[i] * 1000u) {
            if (covered) *covered = 1;
            return APT_ADAPT_SCALE_Q16[i];
        }
    }
    if (covered) *covered = 0;
    return APT_ADAPT_SCALE_Q16[APT_ADAPT_BINS - 1u];
}

uint32_t apt_adapt_scale_coef(uint32_t coef, uint32_t scale_q16)
{
    return (uint32_t)(((uint64_t)coef * scale_q16 + 32768u) >> 16);
}

static apt_status_t apply(apt_adapt_t *a, apt_bus_t *bus)
{
    const uint32_t kp = apt_adapt_scale_coef(a->base_kp, a->scale_q16);
    const uint32_t kd = apt_adapt_scale_coef(a->base_kd, a->scale_q16);
    if (apt_write(bus, APT_REG_KP, kp) != APT_OK) return APT_ERR_BUS;
    if (apt_write(bus, APT_REG_KD, kd) != APT_OK) return APT_ERR_BUS;
    a->changes++;
    return APT_OK;
}

apt_status_t apt_adapt_probe(apt_adapt_t *a, apt_bus_t *bus, uint32_t probe_us, uint32_t dt_ms)
{
    uint32_t want, rel;
    int covered = 1;
    if (!a->primed) {
        a->smooth_us = a->est_us = probe_us;
        a->primed = 1;
    } else {
        /* first-order smoothing with alpha = min(1, dt / tau), in Q8 */
        uint32_t alpha_q8 = dt_ms >= APT_ADAPT_SMOOTH_MS ? 256u : (dt_ms * 256u) / APT_ADAPT_SMOOTH_MS;
        if (probe_us >= a->smooth_us) a->smooth_us += (uint32_t)(((uint64_t)(probe_us - a->smooth_us) * alpha_q8) >> 8);
        else a->smooth_us -= (uint32_t)(((uint64_t)(a->smooth_us - probe_us) * alpha_q8) >> 8);
        rel = (uint32_t)(((uint64_t)APT_ADAPT_RELEASE_US_PER_S * dt_ms) / 1000u);
        a->est_us = (a->est_us > rel) ? a->est_us - rel : 0u;
        if (a->smooth_us > a->est_us) a->est_us = a->smooth_us;  /* instant attack */
    }
    want = apt_adapt_lookup(a->est_us, &covered);
    a->beyond_certified = !covered;
    if (want < a->scale_q16) {
        a->scale_q16 = a->cand_q16 = want;
        a->held_ms = 0u;
        return apply(a, bus);
    }
    if (want > a->scale_q16) {
        if (want != a->cand_q16) { a->cand_q16 = want; a->held_ms = 0u; }
        a->held_ms += dt_ms;
        if (a->held_ms >= APT_ADAPT_HOLD_MS) {
            a->scale_q16 = want;
            a->held_ms = 0u;
            return apply(a, bus);
        }
        return APT_OK;
    }
    a->cand_q16 = want;
    a->held_ms = 0u;
    return APT_OK;
}
