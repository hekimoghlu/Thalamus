#!/usr/bin/env python3
"""Module I - the therapeutic layer: why the neuroprotection programme failed, quantified.

This module does three things that require the rest of the framework to already exist:

  I1  It converts the cascade model's own salvage-versus-time curve (Module C) into the outcome
      scale the trials report, calibrating a single mapping constant on ONE verified reperfusion
      data point and then predicting the rest of the verified time-benefit ladder for both
      intravenous thrombolysis and thrombectomy.

  I2  It asks, for every failed neuroprotection trial, what effect size was ATTAINABLE given
      (a) the treatment window actually used, (b) the fraction of injury still preventable at
      that time in the cascade model, and (c) the fraction of the cascade the agent's target can
      touch. It then compares the attainable effect with the effect the trial was powered to
      detect. The claim being tested is specific: that most of these trials could not have
      succeeded even if the drug worked exactly as advertised in animals.

  I3  It ranks targets by tractability, combining verified potency, CNS penetration, the model's
      own sensitivity to blocking that node, and the node's therapeutic window.
"""
import json
import numpy as np, pandas as pd
import kcore as K

rng = np.random.default_rng(K.SEED + 15)
T = K.load_verified("trials_pharmacology",
                    numeric=["n_randomised", "effect", "effect_lo", "effect_hi", "time_window_h"])
P = K.load_verified("target_pharmacology")
rep = pd.read_csv(K.RESULTS / "C_reperfusion_timing.csv").dropna(subset=["reperfusion_h"])

# --------------------------------------------------- I1 salvage -> outcome odds ratio mapping
# The cascade model gives the fraction of tissue salvaged as a function of treatment time. The
# only free constant is the slope that converts salvaged fraction into log-odds of a good
# outcome; it is solved on ONE verified point (thrombolysis within 3 h, OR 1.75).
sal = np.interp(np.linspace(0.25, 24, 200), rep.reperfusion_h, rep.salvage_fraction)
th = np.linspace(0.25, 24, 200)

def salvage_at(hours):
    return float(np.interp(hours, th, sal))

anchor_or = float(T.loc[T["item"] == "IVT <=3.0 h", "effect"].iloc[0])
BETA = np.log(anchor_or) / salvage_at(3.0)

def predicted_or(hours, coverage=1.0):
    """Odds ratio predicted for an intervention that salvages `coverage` of the salvageable
    tissue when delivered at `hours` from onset."""
    return float(np.exp(BETA * coverage * salvage_at(hours)))

ivt_rows = []
for item, hours in (("IVT <=3.0 h", 3.0), ("IVT 3.0-4.5 h", 4.5), ("IVT >4.5 h", 6.0)):
    r = T[T["item"] == item].iloc[0]
    ivt_rows.append(dict(intervention=item, hours=hours, model_or=predicted_or(hours),
                         published_or=float(r.effect), pub_lo=float(r.effect_lo),
                         pub_hi=float(r.effect_hi), used_in_calibration=(item == "IVT <=3.0 h"),
                         src=int(r.src)))
# thrombectomy achieves a larger salvage per unit time because it reperfuses the whole territory;
# its coverage constant is solved on the 3-hour point and the 6- and 8-hour points are predicted
evt3 = float(T.loc[T["item"] == "EVT onset-to-puncture 3 h", "effect"].iloc[0])
COV_EVT = np.log(evt3) / (BETA * salvage_at(3.0))
for item, hours in (("EVT onset-to-puncture 3 h", 3.0), ("EVT onset-to-puncture 6 h", 6.0),
                    ("EVT onset-to-puncture 8 h", 8.0)):
    r = T[T["item"] == item].iloc[0]
    ivt_rows.append(dict(intervention=item, hours=hours,
                         model_or=predicted_or(hours, COV_EVT), published_or=float(r.effect),
                         pub_lo=float(r.effect_lo), pub_hi=float(r.effect_hi),
                         used_in_calibration=(hours == 3.0), src=int(r.src)))
rl = pd.DataFrame(ivt_rows)
rl["within_published_ci"] = [(r.pub_lo <= r.model_or <= r.pub_hi) if np.isfinite(r.pub_lo) else np.nan
                             for r in rl.itertuples()]
K.save_table(rl, "I_reperfusion_time_ladder")

# --------------------------------------------------------- the model's time of zero benefit
# The cascade grid stops at 24 h, where the model still salvages 45% of the at-risk tissue, so the
# crossing time has to be obtained by extrapolating the LATE decay of the salvage curve rather
# than read off the grid. Salvage falls close to log-linearly over 9-24 h, so the extrapolation is
# a two-point fit on that segment; it is reported as an extrapolation, never as a simulated value.
lo_h, hi_h = 9.0, 24.0
decay = np.log(salvage_at(lo_h) / salvage_at(hi_h)) / (hi_h - lo_h)   # per hour
def salvage_extrap(hours):
    return salvage_at(hi_h) * np.exp(-decay * (np.asarray(hours, float) - hi_h))
OR_NULL = 1.05
target_sal = np.log(OR_NULL) / BETA
zero_t = float(hi_h + np.log(salvage_at(hi_h) / target_sal) / decay)
zero_t_extrapolated = zero_t > hi_h

# ---------------------------------------- the time gradient the published ladder itself implies
# Inverting the same mapping on the published odds ratios gives the salvage fraction the trials
# behave as though were available at each time. Comparing that with the model's own curve turns a
# qualitative "the model decays too slowly" into a number, per time point.
gap_rows = []
for r in rl.itertuples():
    if not np.isfinite(r.published_or) or r.published_or <= 1.0:
        implied = np.nan
    else:
        cov = COV_EVT if r.intervention.startswith("EVT") else 1.0
        implied = float(np.log(r.published_or) / (BETA * cov))
    gap_rows.append(dict(intervention=r.intervention, hours=r.hours,
                         model_salvage=salvage_at(r.hours), implied_salvage=implied,
                         ratio=(salvage_at(r.hours) / implied) if np.isfinite(implied) else np.nan,
                         used_in_calibration=r.used_in_calibration, src=r.src))
gap = pd.DataFrame(gap_rows)
K.save_table(gap, "I_implied_salvage_gap")
ivt_gap = gap[(gap.intervention.str.startswith("IVT")) & (~gap.used_in_calibration)]
IVT_OVERSTATEMENT = float(ivt_gap.ratio.mean())
# the decay rate each source implies over its own observed span, for the honest comparison
def loglin_rate(frame):
    f = frame.dropna(subset=["implied_salvage"]).sort_values("hours")
    if len(f) < 2:
        return np.nan
    return float(np.polyfit(f.hours, np.log(f.implied_salvage), 1)[0])
IMPLIED_IVT_RATE = loglin_rate(gap[gap.intervention.str.startswith("IVT")])
IMPLIED_EVT_RATE = loglin_rate(gap[gap.intervention.str.startswith("EVT")])
MODEL_RATE = float(np.polyfit(th[(th >= 3) & (th <= 8)],
                              np.log(np.array([salvage_at(x) for x in th[(th >= 3) & (th <= 8)]])),
                              1)[0])
print(f"Module I time gradient: the model's salvage curve decays at {-100*MODEL_RATE:.1f}%/h over "
      f"3-8 h; the published thrombolysis ladder implies {-100*IMPLIED_IVT_RATE:.1f}%/h and the "
      f"thrombectomy ladder {-100*IMPLIED_EVT_RATE:.1f}%/h. The model therefore overstates "
      f"salvageable tissue beyond the thrombolysis window by a factor of "
      f"{IVT_OVERSTATEMENT:.2f} and the extrapolated time of zero benefit, {zero_t:.1f} h, is "
      f"{zero_t/6.3:.1f}x the verified thrombolysis estimate of 6.3 h.")

# ------------------------------------------------- I2 attainable effect for each failed trial
# A neuroprotectant can only act on the injury that is (i) still preventable at the treatment
# time, and (ii) mediated by the pathway it blocks. Pathway coverage is assigned from the
# cascade model's own structure, not from the trial literature.
COVERAGE = {   # fraction of modelled injury the target pathway can influence
    "excitotoxicity": 0.35, "oxidative": 0.20, "oedema": 0.15, "inflammation": 0.25,
    "barrier": 0.10, "membrane": 0.20, "unknown": 0.20}
AGENT_PATHWAY = {
    "ESCAPE-NA1 nerinetide": "excitotoxicity", "ESCAPE-NEXT nerinetide": "excitotoxicity",
    "CHARM IV glibenclamide": "oedema", "URICO-ICTUS uric acid": "oxidative",
    "ICTUS citicoline": "membrane", "ACTION II natalizumab": "inflammation",
    "SAINT I NXY-059": "oxidative", "SAINT II NXY-059": "oxidative",
    "Tirilazad pooled": "oxidative", "Selfotel ASSIST": "excitotoxicity",
    "Gavestinel GAIN Americas": "excitotoxicity", "EAA antagonists class": "excitotoxicity",
    "FAST-MAG magnesium": "excitotoxicity", "EuroHYP-1 hypothermia": "unknown",
    "EMPHASIS minocycline": "inflammation", "ApTOLL phase 1b/2a": "inflammation",
}
rows = []
for item, pathway in AGENT_PATHWAY.items():
    r = T[T["item"] == item]
    if r.empty:
        continue
    r = r.iloc[0]
    tw = float(r.time_window_h) if np.isfinite(r.time_window_h) else np.nan
    if not np.isfinite(tw):
        continue
    cov = COVERAGE[pathway]
    attainable_or = predicted_or(tw, cov)
    n = float(r.n_randomised) if np.isfinite(r.n_randomised) else np.nan
    # smallest odds ratio a two-arm trial of this size can detect at 80% power, 5% two-sided,
    # for a control-arm good-outcome rate of 0.40
    p0 = 0.40
    if np.isfinite(n):
        se = np.sqrt(2 * (1 / (p0 * (1 - p0))) / (n / 2))
        detectable_or = float(np.exp(2.802 * se))
    else:
        detectable_or = np.nan
    rows.append(dict(trial=item, pathway=pathway, pathway_coverage=cov, window_h=tw,
                     salvageable_fraction_at_window=salvage_at(tw), n=n,
                     attainable_or=attainable_or, detectable_or=detectable_or,
                     could_have_succeeded=bool(np.isfinite(detectable_or)
                                               and attainable_or >= detectable_or),
                     observed_effect=float(r.effect) if np.isfinite(r.effect) else np.nan,
                     observed_lo=float(r.effect_lo) if np.isfinite(r.effect_lo) else np.nan,
                     observed_hi=float(r.effect_hi) if np.isfinite(r.effect_hi) else np.nan,
                     src=int(r.src)))
tr = pd.DataFrame(rows).sort_values("attainable_or", ascending=False)

# Robustness to the one place the framework disagrees with the literature. The model's salvage
# curve decays more slowly than the published thrombolysis ladder implies (above), so the
# attainable effects computed from it are UPPER bounds. Recomputing them on the steeper curve the
# published ladder itself implies can only shrink them; if the conclusion survives on the model's
# own generous curve it survives on the steeper one, and this records that explicitly.
implied_sal_3h = float(np.log(1.75) / BETA)
def salvage_implied(hours):
    """The steeper curve, taken as a strict tightening: never allowed to exceed either 1.0 or the
    model's own value, so the recomputed attainable effects can only fall, never rise."""
    x = float(implied_sal_3h * np.exp(IMPLIED_IVT_RATE * (float(hours) - 3.0)))
    return float(min(x, 1.0, salvage_at(hours)))
tr["attainable_or_steep"] = [float(np.exp(BETA * r.pathway_coverage * salvage_implied(r.window_h)))
                             for r in tr.itertuples()]
tr["could_have_succeeded_steep"] = [bool(np.isfinite(r.detectable_or)
                                         and r.attainable_or_steep >= r.detectable_or)
                                    for r in tr.itertuples()]
K.save_table(tr, "I_trial_attainability")
n_impossible = int((~tr.could_have_succeeded).sum())
n_impossible_steep = int((~tr.could_have_succeeded_steep).sum())

# ---------------------------------- the preclinical-to-clinical translation arithmetic
pre = float(T.loc[T["item"] == "preclinical_pooled_efficacy", "effect"].iloc[0]) / 100
pre_corr = float(T.loc[T["item"] == "preclinical_corrected_efficacy", "effect"].iloc[0]) / 100
tir_pre_time = float(T.loc[T["item"] == "tirilazad_preclinical_dosing_time", "effect"].iloc[0]) / 60
tir_window = float(T.loc[T["item"] == "Tirilazad pooled", "time_window_h"].iloc[0])
translation = dict(
    preclinical_pooled_infarct_reduction=pre,
    publication_bias_corrected=pre_corr,
    salvage_at_preclinical_dosing_time=salvage_at(max(tir_pre_time, th[0])),
    salvage_at_clinical_window=salvage_at(tir_window),
    timing_attenuation_factor=float(salvage_at(tir_window) / salvage_at(max(tir_pre_time, th[0]))),
    combined_expected_clinical_fraction=float(
        pre_corr / pre * salvage_at(tir_window) / salvage_at(max(tir_pre_time, th[0]))))
print(f"Module I: a preclinical agent dosed at {tir_pre_time*60:.0f} min and tested clinically at "
      f"{tir_window:.0f} h retains {100*translation['timing_attenuation_factor']:.1f}% of its "
      f"salvage opportunity; with publication-bias correction the expected clinical fraction of the "
      f"reported preclinical effect is {100*translation['combined_expected_clinical_fraction']:.1f}%")

# ------------------------------------------------------------------- I3 target tractability
sens = pd.read_csv(K.RESULTS / "V_global_sensitivity.csv")
NODE_LEVER = {"excitotoxicity": 0.35, "oedema": 0.15, "inflammation": 0.25, "barrier": 0.10,
              "oxidative": 0.20, "oxidative stress": 0.20, "energy failure": 0.15,
              "thrombosis": 0.45, "fibrinolysis": 0.40, "haemorrhage iron toxicity": 0.10}
assert set(P.cascade_node) <= set(NODE_LEVER), sorted(set(P.cascade_node) - set(NODE_LEVER))
def potency_score(row):
    v, u = row.potency_value, str(row.potency_unit)
    try:
        x = float(v)
    except (TypeError, ValueError):
        return np.nan
    nm = x * 1000.0 if u.startswith("uM") else x
    return float(np.clip(1.0 - np.log10(max(nm, 1e-3)) / 5.0, 0, 1))

P = P.assign(potency_score=[potency_score(r) for r in P.itertuples()])
P["cns_score"] = [1.0 if str(c).startswith("yes") else (0.3 if "UNVERIFIED" in str(c) else 0.1)
                  for c in P.cns_penetrant]
P["cascade_leverage"] = [NODE_LEVER.get(n, 0.15) for n in P.cascade_node]
# the therapeutic window of a node is the time at which the model still has that pathway's share
# of injury left to prevent
P["window_h"] = [float(th[np.argmax(np.array(sal) < (1 - lev))]) if (np.array(sal) < (1 - lev)).any()
                 else float(th[-1]) for lev in P.cascade_leverage]
P["tractability"] = (P.potency_score.fillna(0.3) * 0.3 + P.cns_score * 0.3
                     + P.cascade_leverage * 0.4)
rank = P.sort_values("tractability", ascending=False)[
    ["target_gene", "tool_compound", "potency_value", "potency_unit", "cns_penetrant",
     "cascade_node", "cascade_leverage", "window_h", "tractability", "clinical_phase_in_stroke"]]
K.save_table(rank, "I_target_tractability")

K.save_json(dict(outcome_mapping_beta=BETA, evt_coverage_constant=COV_EVT,
                 model_zero_benefit_time_h=zero_t,
                 model_zero_benefit_is_extrapolated=bool(zero_t_extrapolated),
                 late_salvage_decay_per_h=float(decay),
                 model_salvage_decay_3_8h_per_h=MODEL_RATE,
                 implied_ivt_salvage_decay_per_h=IMPLIED_IVT_RATE,
                 implied_evt_salvage_decay_per_h=IMPLIED_EVT_RATE,
                 ivt_salvage_overstatement_factor=IVT_OVERSTATEMENT,
                 published_ivt_zero_benefit_h=float(T.loc[T["item"] == "IVT time of zero effect",
                                                          "effect"].iloc[0]),
                 published_evt_benefit_boundary_h=float(T.loc[T["item"] == "EVT benefit boundary",
                                                              "effect"].iloc[0]),
                 n_trials_assessed=int(len(tr)), n_trials_underpowered_for_attainable=n_impossible,
                 translation=translation,
                 top_targets=rank.head(5)[["target_gene", "tractability"]].to_dict("records")),
            "I_therapeutics_summary")
print(rl.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print(tr[["trial", "window_h", "attainable_or", "detectable_or", "could_have_succeeded"]]
      .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print(f"  {n_impossible} of {len(tr)} trials were powered for an effect larger than the largest "
      f"effect attainable at their own treatment window "
      f"({n_impossible_steep}/{len(tr)} on the steeper salvage curve the published ladder implies)")
