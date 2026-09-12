#!/usr/bin/env python3
"""Module E - the closed-loop sensorimotor layer (the Artificial Peripheral Thalamus).

This is the terminal application layer of the framework: an admittance-controlled hand
interface whose design variables are loop latency, virtual stiffness/damping and sensing
resolution, and whose therapeutic output is fed back into the Module D recovery law.

Computed here:
  1. An afferent-channel information rate derived from verified psychophysical resolution
     (force Weber fraction, torque JND) and verified conduction/cortical-arrival latencies.
  2. The delay-dependent stability envelope of the admittance law, simulated rather than
     asserted, with the verified haptic-perception thresholds overlaid.
  3. A latency-to-learning-rate function fitted to the three verified visuomotor adaptation
     rates (60 ms, 259 ms, and 259 ms after prior exposure).
  4. Propagation of that function through the Module D recovery law to give predicted
     Fugl-Meyer gains as a function of device latency and practice dose - the design envelope -
     and the dose needed to clear the verified MCID at each latency.
  5. Conversion of those gains to standardised effect sizes using the verified VNS-REHAB
     within-group SDs, and comparison with the verified pooled effect sizes for robot-assisted
     training, BCI-based training, mirror therapy, FES and sensory retraining.
"""
import numpy as np, pandas as pd
import kcore as K

rng = np.random.default_rng(K.SEED + 5)
G = K.getp
FM_MAX = 66.0

# ---------------------------------------------------------- 1. afferent channel information rate
W_FORCE = G("weber_fraction_force")              # 0.085 over 0.5-200 N
W_POS = G("weber_fraction_position")             # 0.08
JND_TORQUE = G("jnd_elbow_torque_static")        # 0.0975 Nm
N20 = G("n20_latency")                           # 19.4 ms cortical arrival
TRANSIT = G("peripheral_to_cortex_transit")      # 9.5 ms
CV_ABETA = G("abeta_conduction_velocity")        # 50 m/s
GATE_RATIO = G("somatosensory_gating_ratio_healthy")   # <0.4 second-response suppression

def channel_rate(snr, bandwidth_hz):
    """Shannon rate of a Gaussian channel (bit/s)."""
    return bandwidth_hz * np.log2(1.0 + snr)

# Physiological resolution sets the intrinsic SNR of the biological afferent channel: a Weber
# fraction w implies an amplitude resolution of w over the operating range, i.e. SNR = 1/w^2.
SNR_BIO = 1.0 / W_FORCE ** 2
# The mechatronic channel's resolution is set by the transducer quantum (0.01 N on a 0-20 N span
# for the FSR-class sensors specified for the interface) rather than by psychophysics.
SENSOR_Q, SPAN_N = 0.01, 20.0
SNR_DEV = (SPAN_N / SENSOR_Q) ** 2
BW_PROP = 30.0          # proprioceptive bandwidth (Hz) of voluntary hand movement
rate_bio = channel_rate(SNR_BIO, BW_PROP)
rate_dev = channel_rate(SNR_DEV, BW_PROP)

# Delay does not reduce Shannon rate but destroys the causal usefulness of feedback: the fraction
# of the movement bandwidth for which feedback still arrives within one cycle is 1/(2*tau*f_max).
def usable_fraction(tau_ms, bw=BW_PROP):
    return np.clip(1.0 / (2.0 * (tau_ms / 1000.0) * bw), 0.0, 1.0)

tau_grid = np.array([5, 10, 20, 30, 50, 80, 110, 150, 220, 300, 500])
chan = pd.DataFrame(dict(latency_ms=tau_grid,
                         usable_bandwidth_fraction=usable_fraction(tau_grid)))
chan["effective_rate_bit_s"] = rate_dev * chan.usable_bandwidth_fraction
chan["biological_reference_rate_bit_s"] = rate_bio
K.save_table(chan, "E_channel_rate")

# ------------------------------------------------ 2. delay-dependent admittance stability limit
# Virtual mass-spring-damper admittance with a pure loop delay: simulate the discrete loop and
# find the largest virtual stiffness that remains stable at each delay.
def admittance_stable(kstiff, tau_ms, m=0.25, b=6.0, dt=1e-3, tmax=8.0, kick=1.0):
    """Stability is judged by whether the oscillation envelope grows, not by settling time, so
    that heavily damped (slow) but stable configurations are not misclassified."""
    n = int(tmax / dt); d = max(int((tau_ms / 1000.0) / dt), 1)
    x = np.zeros(n); v = np.zeros(n); f = np.zeros(n)
    x[0] = kick
    for i in range(1, n):
        xd = x[i - d] if i - d >= 0 else x[0]
        f[i] = -kstiff * xd - b * (v[i - 1] if i - d < 0 else v[max(i - d, 0)])
        a = f[i] / m
        v[i] = v[i - 1] + dt * a
        x[i] = x[i - 1] + dt * v[i]
        if not np.isfinite(x[i]) or abs(x[i]) > 50 * abs(kick):
            return False
    third = n // 3
    return abs(x[-third:]).max() <= abs(x[third:2 * third]).max() + 1e-9

rows = []
for tau in [1, 2, 5, 10, 20, 30, 50, 80, 110, 150, 220, 300]:
    k_hi = 0.0
    for k in np.geomspace(0.5, 20000.0, 80):
        if admittance_stable(k, tau):
            k_hi = k
        else:
            break
    rows.append(dict(latency_ms=tau, max_stable_stiffness_N_per_m=k_hi,
                     usable_bandwidth_fraction=float(usable_fraction(tau))))
stab = pd.DataFrame(rows)
K.save_table(stab, "E_stability_envelope")

# ------------------------------------------------- 3. latency -> motor-learning-rate function
d_obs = np.array([60.0, 259.2])                       # ms (verified measured total delays)
a_obs = np.array([G("adaptation_rate_60ms"), G("adaptation_rate_259ms")])
# two-point fit of a saturating decay a(tau) = a0 / (1 + (tau/tau_half)); solved exactly
tau_half = float((d_obs[1] - d_obs[0] * a_obs[0] / a_obs[1]) / (a_obs[0] / a_obs[1] - 1.0))
a0 = float(a_obs[0] * (1.0 + d_obs[0] / tau_half))
def adapt_rate(tau_ms):
    return a0 / (1.0 + np.asarray(tau_ms, float) / tau_half)
print(f"Module E: latency-learning function a(tau) = {a0:.3f}/(1+tau/{tau_half:.0f} ms) "
      f"reproduces the two verified adaptation rates exactly "
      f"({adapt_rate(60):.3f} at 60 ms, {adapt_rate(259.2):.3f} at 259 ms)")
print(f"   half-rate latency = {tau_half:.0f} ms; verified 'imperceptible' haptic delay is "
      f"{G('haptic_delay_imperceptible'):.0f} ms and agency collapses by 200 ms")

# ------------------------------------- 4. design envelope: FM gain vs latency and dose
# Module D recovery law, re-used unchanged; the device acts by scaling the effective plasticity
# drive with (i) the latency-dependent learning rate and (ii) the practice dose.
DPAR = __import__("json").loads((K.RESULTS / "D_recovery_summary.json").read_text())["recovery_parameters"]
MDC = G("fmue_mdc")   # minimal DETECTABLE change (5.2 FM points); the MCID is 5.25
REPS_USUAL = G("reps_usual_care")            # 32 movements per session
REPS_HIGH = G("reps_high_dose_protocol")     # 322 repetitions per hour

def cumulative_drive(t_w, par, scale=1.0):
    return scale * (par["eta_fast"] * par["tau_fast"] * (1 - np.exp(-t_w / par["tau_fast"]))
                    + par["eta_slow"] * par["tau_slow"] * (1 - np.exp(-t_w / par["tau_slow"]))
                    + par["floor"] * t_w)

T_CHRONIC = 26.0            # device trials recruit chronic patients (>=6 months post-stroke)

def fm_gain(fm0, cap, weeks, scale):
    """Fugl-Meyer gain accrued over `weeks` of training that STARTS in the chronic phase, so the
    residual spontaneous drive is small and almost all of the gain is training-driven."""
    d0 = cumulative_drive(T_CHRONIC, DPAR, 1.0)
    d1 = d0 + scale * (cumulative_drive(T_CHRONIC + weeks, DPAR, 1.0) - d0)
    return (FM_MAX - fm0) * (np.exp(-cap * d0) - np.exp(-cap * d1))


def dose_scale(dose, gamma):
    """Practice dose enters the plasticity drive sub-linearly; the exponent is calibrated below
    against the one verified high-dose observation (322 repetitions/session -> +8 ARAT)."""
    return np.asarray(dose, float) ** gamma

# cohort for the design study: chronic, moderate-to-severe, as in the device trials
NP = 3000
fm0 = np.clip(rng.normal(30.0, 9.0, NP), 4, 58)
cap = np.clip(rng.normal(0.9, 0.45, NP), 0.02, None)
WEEKS = 8.6                                   # 60-day programme
# --- calibrate the dose exponent to the verified high-dose repetition study
ARAT_TO_FM = FM_MAX / 57.0        # stated scale-range approximation, not a validated conversion
TARGET_FM = G("reps_high_dose_arat_gain") * ARAT_TO_FM
DOSE_HIGH = REPS_HIGH / REPS_USUAL
fm0_c = np.clip(rng.normal(30.0, 9.0, 3000), 4, 58)
cap_c = np.clip(rng.normal(0.9, 0.45, 3000), 0.02, None)
GAMMA = None
lo_g, hi_g = 0.01, 3.0
for _ in range(60):
    mid = 0.5 * (lo_g + hi_g)
    g = fm_gain(fm0_c, cap_c, 6.0, dose_scale(DOSE_HIGH, mid)).mean()
    if g > TARGET_FM:
        hi_g = mid
    else:
        lo_g = mid
GAMMA = 0.5 * (lo_g + hi_g)
print(f"   dose exponent gamma = {GAMMA:.3f} calibrated so that {REPS_HIGH:.0f} repetitions per "
      f"session for 6 weeks yields {fm_gain(fm0_c, cap_c, 6.0, dose_scale(DOSE_HIGH, GAMMA)).mean():.2f} "
      f"FM points (verified +{G('reps_high_dose_arat_gain'):.0f} ARAT = {TARGET_FM:.2f} FM-equivalent)")

dose_grid = np.array([1.0, 2.0, 4.0, DOSE_HIGH, 15.0])
lat_grid = np.array([5, 10, 20, 30, 50, 80, 110, 150, 220, 300])
env = []
base_scale = 1.0
base = fm_gain(fm0, cap, WEEKS, base_scale)
for lat in lat_grid:
    lr = adapt_rate(lat) / adapt_rate(60.0)           # relative to the reference condition
    for dose in dose_grid:
        # NOTE: the latency effect enters ONLY through the measured adaptation-rate function;
        # the usable-bandwidth fraction is reported as an information-theoretic descriptor and is
        # deliberately not multiplied in again, which would double-count delay.
        g = fm_gain(fm0, cap, WEEKS, base_scale * lr * dose_scale(dose, GAMMA))
        env.append(dict(latency_ms=lat, dose_multiple=dose,
                        mean_fm_gain=float(g.mean()),
                        mean_fm_gain_over_usual_care=float((g - base).mean()),
                        pct_exceeding_mcid=float(100 * np.mean(g - base > MDC))))
env = pd.DataFrame(env)
K.save_table(env, "E_design_envelope")

# dose needed to clear the MCID at each latency
need = []
for lat in lat_grid:
    lr = adapt_rate(lat) / adapt_rate(60.0)
    dose = 1.0
    for _ in range(200):
        g = fm_gain(fm0, cap, WEEKS, lr * dose_scale(dose, GAMMA))
        if (g - base).mean() >= MDC:
            break
        dose *= 1.08
    need.append(dict(latency_ms=lat, dose_multiple_for_mcid=float(dose),
                     reps_per_session_required=float(dose * REPS_USUAL),
                     exceeds_published_high_dose_protocol=bool(dose * REPS_USUAL > REPS_HIGH)))
need = pd.DataFrame(need)
K.save_table(need, "E_dose_for_mcid")

# ------------------------------------------ 5. predicted effect sizes vs published meta-analyses
# Two distinct contrasts are predicted, because the published literature contains both:
#   (a) MATCHED-DOSE device trials (robot, BCI, mirror, FES vs dose-matched conventional therapy)
#       where the only difference is the quality and timing of feedback, and
#   (b) DOSE CONTRASTS (high-repetition protocols vs usual care), where the difference is amount.
# Conflating them is the commonest way the rehabilitation-robotics literature overstates effects.
SD_POOL = float(np.sqrt((G("vns_fm_sd_treatment") ** 2 + G("vns_fm_sd_control") ** 2) / 2))
REF_LAT = 60.0                    # the reference latency of the verified adaptation measurement
DOSE_TYPICAL = 4.0                # a conventional robot session relative to usual-care reps
pred = []
for lat in (10, 20, 30, 50, 80, 110, 150, 220):
    lr = adapt_rate(lat) / adapt_rate(REF_LAT)
    g_dev = fm_gain(fm0, cap, WEEKS, lr * dose_scale(DOSE_TYPICAL, GAMMA))
    g_ctl = fm_gain(fm0, cap, WEEKS, 1.0 * dose_scale(DOSE_TYPICAL, GAMMA))
    g_hi = fm_gain(fm0, cap, WEEKS, lr * dose_scale(DOSE_HIGH, GAMMA))
    g_usual = fm_gain(fm0, cap, WEEKS, 1.0 * dose_scale(1.0, GAMMA))
    pred.append(dict(latency_ms=lat,
                     matched_dose_fm_difference=float((g_dev - g_ctl).mean()),
                     matched_dose_smd=float((g_dev - g_ctl).mean() / SD_POOL),
                     dose_amplified_fm_difference=float((g_hi - g_usual).mean()),
                     dose_amplified_smd=float((g_hi - g_usual).mean() / SD_POOL),
                     matched_dose_exceeds_mcid=bool((g_dev - g_ctl).mean() > MDC)))
pred = pd.DataFrame(pred)
comp = pd.DataFrame([
    dict(comparator="robot-assisted training, arm function", contrast="matched dose",
         published_smd=G("robot_smd_arm_function"), ci_lo=G("robot_smd_arm_function", "value_lo"),
         ci_hi=G("robot_smd_arm_function", "value_hi"), src=K.param_src("robot_smd_arm_function")),
    dict(comparator="BCI-based training, FM-UE", contrast="matched dose",
         published_smd=G("bci_smd_fmue"), ci_lo=G("bci_smd_fmue", "value_lo"),
         ci_hi=G("bci_smd_fmue", "value_hi"), src=K.param_src("bci_smd_fmue")),
    dict(comparator="mirror therapy, motor function", contrast="matched dose",
         published_smd=G("mirror_smd_motor_function"),
         ci_lo=G("mirror_smd_motor_function", "value_lo"),
         ci_hi=G("mirror_smd_motor_function", "value_hi"),
         src=K.param_src("mirror_smd_motor_function")),
    dict(comparator="electrical stimulation, FM-UE", contrast="matched dose",
         published_smd=G("fes_smd_fmue"), ci_lo=G("fes_smd_fmue", "value_lo"),
         ci_hi=G("fes_smd_fmue", "value_hi"), src=K.param_src("fes_smd_fmue")),
    dict(comparator="VNS-paired rehabilitation, FM-UE day 90", contrast="matched dose",
         published_smd=G("vns_fmue_day90") / SD_POOL,
         ci_lo=G("vns_fmue_day90", "value_lo") / SD_POOL,
         ci_hi=G("vns_fmue_day90", "value_hi") / SD_POOL, src=K.param_src("vns_fmue_day90")),
    dict(comparator="sensory retraining, FM points", contrast="matched dose",
         published_smd=G("sensory_retraining_md_fm") / SD_POOL,
         ci_lo=G("sensory_retraining_md_fm", "value_lo") / SD_POOL,
         ci_hi=G("sensory_retraining_md_fm", "value_hi") / SD_POOL,
         src=K.param_src("sensory_retraining_md_fm")),
    dict(comparator="increased therapy duration, FM", contrast="dose contrast",
         published_smd=G("dose_fm_gain_increased_duration") / SD_POOL,
         ci_lo=G("dose_fm_gain_increased_duration", "value_lo") / SD_POOL,
         ci_hi=G("dose_fm_gain_increased_duration", "value_hi") / SD_POOL,
         src=K.param_src("dose_fm_gain_increased_duration")),
    dict(comparator="high-repetition protocol, ARAT->FM equivalent", contrast="dose contrast",
         published_smd=G("reps_high_dose_arat_gain") * ARAT_TO_FM / SD_POOL,
         ci_lo=np.nan, ci_hi=np.nan, src=K.param_src("reps_high_dose_arat_gain")),
])
K.save_table(pred, "E_predicted_effect_sizes")
K.save_table(comp, "E_published_comparators")
matched_pub = comp[comp.contrast == "matched dose"]
in_range = pred[(pred.matched_dose_smd >= matched_pub.published_smd.min()) &
                (pred.matched_dose_smd <= matched_pub.published_smd.max())]
print(f"   matched-dose predictions span SMD {pred.matched_dose_smd.min():.2f}-"
      f"{pred.matched_dose_smd.max():.2f}; published matched-dose effects span "
      f"{matched_pub.published_smd.min():.2f}-{matched_pub.published_smd.max():.2f}; "
      f"latencies compatible with that band: "
      f"{', '.join(str(int(x)) + ' ms' for x in in_range.latency_ms)}")

K.save_json(dict(dose_exponent_gamma=float(GAMMA),
                 afferent_rate_biological_bit_s=float(rate_bio),
                 afferent_rate_device_bit_s=float(rate_dev),
                 device_to_biological_rate_ratio=float(rate_dev / rate_bio),
                 cortical_arrival_latency_ms=N20,
                 learning_half_rate_latency_ms=tau_half, learning_rate_at_zero_delay=a0,
                 max_stable_stiffness_at_1ms=float(stab.max_stable_stiffness_N_per_m.iloc[0]),
                 max_stable_stiffness_at_110ms=float(
                     stab.loc[stab.latency_ms == 110, "max_stable_stiffness_N_per_m"].iloc[0]),
                 pooled_sd_used_for_smd=SD_POOL,
                 matched_dose_smd_range=[float(pred.matched_dose_smd.min()), float(pred.matched_dose_smd.max())],
                 dose_amplified_smd_range=[float(pred.dose_amplified_smd.min()), float(pred.dose_amplified_smd.max())],
                 published_matched_dose_smd_range=[float(matched_pub.published_smd.min()), float(matched_pub.published_smd.max())]),
            "E_control_summary")
print(f"   device channel rate {rate_dev:.0f} bit/s vs biological reference {rate_bio:.0f} bit/s "
      f"({rate_dev/rate_bio:.1f}x) at matched bandwidth")
print(f"   stability: max virtual stiffness falls from {stab.max_stable_stiffness_N_per_m.iloc[0]:.0f} N/m "
      f"at 1 ms to {stab.loc[stab.latency_ms==110,'max_stable_stiffness_N_per_m'].iloc[0]:.0f} N/m at 110 ms")
print(need.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print(pred.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(comp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
