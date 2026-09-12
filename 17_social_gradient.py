#!/usr/bin/env python3
"""Module K - the social and geographic layer (L0): what sits upstream of inherited risk.

Module A quantifies what a person's genome contributes to stroke risk. That is a small number, and
leaving the framework there would be a distortion: the verified public data say that where someone
lives, and how far they are from a thrombectomy-capable centre, move the same outcome further than
a top-decile polygenic score does. This module puts those factors on the same quantitative footing
as the rest of the framework, using only verified public estimates, and it does arithmetic with
them rather than tabulating them.

  K1  Risk-factor clustering. The verified single-factor population attributable fractions, if the
      factors were independent, imply a combined fraction that is much larger than the verified
      combined fraction. The gap is a measurement of how much the risk factors overlap.

  K2  Mediation of the geographic gradient. How much of the verified stroke-belt prevalence excess
      can be accounted for by the verified excess of hypertension in the belt, given the verified
      hypertension attributable fraction? The answer is obtained by propagating prevalences, not
      by assertion.

  K3  Distance as a therapeutic variable. The verified share of the population beyond 60 minutes
      of a thrombectomy-capable centre is converted, through Module I's salvage-to-outcome map and
      Module C's salvage curve, into lost odds of a good outcome and an annual national count.

  K4  Two attribution systems compared. The verified GBD share of stroke disability-adjusted life
      years attributable to high systolic pressure and the verified NHANES hypertension
      attributable fraction do not agree; the discrepancy is quantified and explained.

  K5  The age structure of the racial disparity, fitted to the two verified REGARDS relative risks
      and extrapolated to the age at which the model says they converge.

  K6  Genome against postcode. Every exposure in this module is converted to the discrimination
      scale used in Module A, so a polygenic score, a stroke-belt address and a modifiable
      risk-factor burden can be compared in the same units.
"""
import json
import numpy as np, pandas as pd
from scipy.stats import norm
import kcore as K

SEX = ("women", "men")
FACTORS = ("hypertension", "obesity", "diabetes", "smoking", "inactivity", "af")

def paf(factor, sex):
    for name in (f"paf_{factor}_{sex}", f"paf_{factor}_men", f"paf_{factor}_women"):
        try:
            return K.getp(name) / 100.0, K.param_src(name), name
        except KeyError:
            continue
    return np.nan, "", ""

# ------------------------------------------------------------------- K1 risk-factor clustering
rows = []
for sex in SEX:
    pafs, srcs, names = [], [], []
    for f in FACTORS:
        v, s, n = paf(f, sex)
        if np.isfinite(v):
            pafs.append(v); srcs.append(s); names.append(n)
    pafs = np.array(pafs)
    naive_sum = float(pafs.sum())
    independent = float(1.0 - np.prod(1.0 - pafs))
    observed = K.getp(f"paf_combined_{sex}") / 100.0
    obs_lo = K.getp(f"paf_combined_{sex}", "value_lo") / 100.0
    obs_hi = K.getp(f"paf_combined_{sex}", "value_hi") / 100.0
    # the single shrinkage factor on each attributable fraction that reproduces the verified
    # combined fraction under the same independence algebra: a summary of the overlap
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if 1.0 - np.prod(1.0 - mid * pafs) < observed:
            lo = mid
        else:
            hi = mid
    overlap_shrinkage = 0.5 * (lo + hi)
    rows.append(dict(sex=sex, n_factors=len(pafs), sum_of_single_factor_pafs=100 * naive_sum,
                     independent_combination=100 * independent,
                     verified_combined=100 * observed, verified_lo=100 * obs_lo,
                     verified_hi=100 * obs_hi,
                     double_counted_points=100 * (independent - observed),
                     fraction_of_independent_estimate_that_is_overlap=
                     100 * (independent - observed) / independent,
                     implied_shrinkage_per_factor=overlap_shrinkage,
                     src=";".join(sorted(set(srcs)))))
clust = pd.DataFrame(rows)
K.save_table(clust, "K_risk_factor_clustering")

# ------------------------------------------------- K2 mediation of the geographic gradient
# Prevalence ratios, not odds ratios, are what propagate through an attributable-fraction
# decomposition, so both verified odds ratios are first converted using their own verified
# baseline prevalences.
def or_to_rr(odds_ratio, p0):
    p1 = odds_ratio * p0 / (1.0 - p0 + odds_ratio * p0)
    return p1 / p0, p1

p_stroke = K.getp("stroke_prevalence_within_60min") / 100.0     # 2.8% verified
p_ht = K.epi("hypertension_prevalence") / 100.0                 # 47.7% verified
or_belt = K.getp("stroke_belt_or_prevalence")
or_belt_ht = K.getp("stroke_belt_or_hypertension")
rr_belt, p_belt = or_to_rr(or_belt, p_stroke)
rr_ht, p_ht_belt = or_to_rr(or_belt_ht, p_ht)
paf_ht = np.mean([K.getp("paf_hypertension_women"), K.getp("paf_hypertension_men")]) / 100.0
# stroke prevalence scales as (1 - PAF) + PAF x (relative prevalence of the exposure)
rr_pred_from_ht = (1.0 - paf_ht) + paf_ht * rr_ht
mediated = float(np.log(rr_pred_from_ht) / np.log(rr_belt))
med = pd.DataFrame([
    dict(quantity="stroke-belt prevalence odds ratio (verified)", value=or_belt,
         src=K.param_src("stroke_belt_or_prevalence")),
    dict(quantity="baseline stroke prevalence (%)", value=100 * p_stroke,
         src=K.param_src("stroke_prevalence_within_60min")),
    dict(quantity="stroke-belt prevalence RATIO implied", value=rr_belt, src="derived"),
    dict(quantity="stroke-belt hypertension odds ratio (verified)", value=or_belt_ht,
         src=K.param_src("stroke_belt_or_hypertension")),
    dict(quantity="US hypertension prevalence (%)", value=100 * p_ht, src="epidemiology table"),
    dict(quantity="hypertension prevalence RATIO in the belt", value=rr_ht, src="derived"),
    dict(quantity="hypertension attributable fraction (%)", value=100 * paf_ht,
         src=K.param_src("paf_hypertension_women") + ";" + K.param_src("paf_hypertension_men")),
    dict(quantity="stroke prevalence ratio predicted from hypertension alone",
         value=rr_pred_from_ht, src="derived"),
    dict(quantity="share of the geographic gradient mediated by hypertension (%)",
         value=100 * mediated, src="derived"),
    dict(quantity="share unexplained by hypertension prevalence (%)",
         value=100 * (1 - mediated), src="derived"),
])
K.save_table(med, "K_geographic_mediation")

# ------------------------------------------------- K3 distance to thrombectomy as lost outcome
I = json.load(open(K.RESULTS / "I_therapeutics_summary.json"))
BETA, COV_EVT = I["outcome_mapping_beta"], I["evt_coverage_constant"]
rep = pd.read_csv(K.RESULTS / "C_reperfusion_timing.csv").dropna(subset=["reperfusion_h"])
th = np.linspace(0.25, 24, 200)
sal = np.interp(th, rep.reperfusion_h, rep.salvage_fraction)

def evt_or(hours):
    return float(np.exp(BETA * COV_EVT * np.interp(hours, th, sal)))

# Module I established that the cascade model's salvage curve decays more slowly with time than the
# published reperfusion ladder implies. Every time-cost calculation in this section therefore
# carries a second version computed on the steeper curve the published ladder itself implies, so a
# delay's cost is bracketed rather than asserted from the side of the discrepancy that flatters it.
RATE_STEEP = I["implied_ivt_salvage_decay_per_h"]
SAL_3H = float(np.interp(base_h_ref := 3.0, th, sal))

def salvage_steep(hours):
    return float(min(SAL_3H * np.exp(RATE_STEEP * (float(hours) - 3.0)), 1.0,
                     float(np.interp(hours, th, sal))))

def evt_or_steep(hours):
    return float(np.exp(BETA * COV_EVT * salvage_steep(hours)))

beyond = 1.0 - K.getp("access_60min_evt_capable") / 100.0     # 20% of US adults
base_h = 3.0                                                  # verified EVT ladder anchor point
rows = []
for extra_min in (30, 60, 90, 120):
    or_near, or_far = evt_or(base_h), evt_or(base_h + extra_min / 60.0)
    p0 = 0.40                                                 # control-arm good-outcome rate
    def p_from_or(o):
        odds = o * p0 / (1 - p0)
        return odds / (1 + odds)
    d_abs = p_from_or(or_near) - p_from_or(or_far)
    or_near_s, or_far_s = evt_or_steep(base_h), evt_or_steep(base_h + extra_min / 60.0)
    d_abs_s = p_from_or(or_near_s) - p_from_or(or_far_s)
    rows.append(dict(additional_transport_min=extra_min, or_at_reference=or_near,
                     or_after_delay=or_far, odds_ratio_lost=or_near / or_far,
                     absolute_good_outcome_lost_pct=100 * d_abs,
                     absolute_good_outcome_lost_pct_steep_curve=100 * d_abs_s))
acc = pd.DataFrame(rows)
# The national count of avoidable poor outcomes cannot be computed from this evidence base: the
# large-vessel-occlusion share of ischaemic stroke is not among the verified parameters, and it
# will not be invented here. The result is therefore expressed per 1000 thrombectomy-eligible
# patients, together with the two verified national denominators that a reader can combine with a
# share of their own choosing.
n_stroke = K.epi("annual_strokes_total")
isch = K.epi("ischemic_share") / 100.0
acc["eligible_patients_beyond_60min_per_1000_eligible"] = 1000.0 * beyond
acc["excess_poor_outcomes_per_1000_eligible"] = (acc.absolute_good_outcome_lost_pct / 100.0
                                                * 1000.0 * beyond)
acc["annual_ischaemic_strokes_us"] = n_stroke * isch
acc["share_beyond_60min_evt"] = beyond
K.save_table(acc, "K_access_geography")

# the same calculation run on the two verified access-related disparities, expressed as the
# equivalent transport delay each corresponds to
def delay_equivalent(odds_ratio_loss, steep=False):
    """Minutes of additional onset-to-puncture time that would produce this loss of odds."""
    f = evt_or_steep if steep else evt_or
    target = f(base_h) / odds_ratio_loss
    grid = np.linspace(base_h, 23.9, 800)
    vals = np.array([f(x) for x in grid])
    if vals.min() > target:
        return np.nan
    return float((grid[np.argmax(vals <= target)] - base_h) * 60.0)

equiv = pd.DataFrame([
    dict(disparity="10 units of area deprivation (thrombectomy)",
         odds_ratio=K.getp("adi_or_thrombectomy"), src=K.param_src("adi_or_thrombectomy")),
    dict(disparity="10 units of area deprivation (thrombolysis)",
         odds_ratio=K.getp("adi_or_thrombolysis"), src=K.param_src("adi_or_thrombolysis")),
    dict(disparity="Black versus White patients (thrombectomy)",
         odds_ratio=K.getp("evt_or_black_vs_white"), src=K.param_src("evt_or_black_vs_white")),
    dict(disparity="lowest versus highest income quartile (thrombectomy)",
         odds_ratio=1.0 / K.getp("evt_or_income_q4_vs_q1"),
         src=K.param_src("evt_or_income_q4_vs_q1")),
    dict(disparity="Medicare versus private insurance (thrombectomy)",
         odds_ratio=1.0 / K.getp("evt_or_private_vs_medicare"),
         src=K.param_src("evt_or_private_vs_medicare")),
])
equiv["treatment_odds_lost"] = 1.0 / equiv.odds_ratio
equiv["equivalent_extra_transport_min"] = [delay_equivalent(x) for x in equiv.treatment_odds_lost]
equiv["equivalent_extra_transport_min_steep"] = [delay_equivalent(x, steep=True)
                                                 for x in equiv.treatment_odds_lost]
K.save_table(equiv, "K_disparity_delay_equivalents")

# ---------------------------------------------- K4 two attribution systems that do not agree
daly_total = K.getp("us_stroke_daly_rate")
daly_sbp = K.getp("dalys_attributable_high_sbp")
daly_ldl = K.getp("dalys_attributable_high_ldl")
gbd_share_sbp = daly_sbp / daly_total
attr = pd.DataFrame([
    dict(system="GBD 2021, high systolic pressure as a continuous exposure",
         stroke_burden_share=100 * gbd_share_sbp, basis="DALY rate per 100 000",
         src=K.param_src("dalys_attributable_high_sbp")),
    dict(system="NHANES 2015-2018, diagnosed hypertension as a binary exposure",
         stroke_burden_share=100 * paf_ht, basis="population attributable fraction",
         src=K.param_src("paf_hypertension_women")),
    dict(system="GBD 2021, high LDL cholesterol", stroke_burden_share=100 * daly_ldl / daly_total,
         basis="DALY rate per 100 000", src=K.param_src("dalys_attributable_high_ldl")),
])
attr_gap = dict(gbd_sbp_share=100 * gbd_share_sbp, nhanes_hypertension_paf=100 * paf_ht,
                ratio=float(gbd_share_sbp / paf_ht),
                explanation=("the two systems are not measuring the same thing: GBD attributes "
                             "burden against a theoretical-minimum systolic pressure, so every "
                             "millimetre above about 110 mmHg counts, while the attributable "
                             "fraction counts only people who cross the diagnostic threshold. The "
                             "ratio, %.1f, is the price of the binary definition, and it means any "
                             "model that uses hypertension as a yes/no exposure - including "
                             "Module A of this framework - understates the modifiable share of "
                             "stroke burden by roughly that factor."
                             % float(gbd_share_sbp / paf_ht)))
K.save_table(attr, "K_attribution_systems")

# ---------------------------------------- K5 the age structure of the racial disparity in risk
a1, a2 = 45.0, 65.0
rr1 = K.getp("regards_rr_black_white_age45")
rr2 = K.getp("regards_rr_black_white_age65")
b = float(np.log(rr1 / rr2) / (a2 - a1))               # per year decline in log relative risk
a0 = float(np.log(rr1) + b * a1)
def rr_age(age):
    return np.exp(a0 - b * np.asarray(age, float))
converge_age = float(a0 / b)
expl_45 = K.getp("regards_explained_by_risk_factors_45") / 100.0
expl_65 = K.getp("regards_explained_by_risk_factors_65") / 100.0
expl_ses = K.getp("regards_explained_adding_ses") / 100.0
# the residual relative risk after removing a stated share of the EXCESS risk
def residual(rr, share):
    return 1.0 + (rr - 1.0) * (1.0 - share)
ages = np.arange(45, 86, 5)
dis = pd.DataFrame(dict(age=ages, relative_risk=rr_age(ages)))
dis["residual_after_risk_factors"] = [residual(r, expl_45 + (expl_65 - expl_45)
                                               * np.clip((a - 45) / 20.0, 0, 1))
                                      for a, r in zip(dis.age, dis.relative_risk)]
dis["residual_after_risk_factors_and_ses"] = [residual(r, expl_ses) for r in dis.relative_risk]
K.save_table(dis, "K_disparity_age_structure")
dis_check = pd.DataFrame([
    dict(quantity="relative risk at 45 y", model=float(rr_age(45)), published=rr1,
         used_in_calibration=True, src=K.param_src("regards_rr_black_white_age45")),
    dict(quantity="relative risk at 65 y", model=float(rr_age(65)), published=rr2,
         used_in_calibration=True, src=K.param_src("regards_rr_black_white_age65")),
    dict(quantity="residual relative risk at 45 y after risk factors",
         model=residual(rr1, expl_45), published=2.14, used_in_calibration=False,
         src=K.param_src("regards_explained_by_risk_factors_45")),
    dict(quantity="residual relative risk at 65 y after risk factors",
         model=residual(rr2, expl_65), published=1.35, used_in_calibration=False,
         src=K.param_src("regards_explained_by_risk_factors_65")),
    dict(quantity="age at which the modelled relative risk reaches 1", model=converge_age,
         published=np.nan, used_in_calibration=False, src="derived"),
])
dis_check["ratio_model_to_published"] = dis_check.model / dis_check.published
K.save_table(dis_check, "K_disparity_validation")

# ------------------------------------------- K6 genome against postcode, in the same units
# A single binary exposure with prevalence p and odds ratio R implies a maximum discrimination
# that can be written down exactly; this puts a polygenic score, an address and a risk-factor
# burden on one axis. The liability-scale effect size is the other common currency.
A = json.load(open(K.RESULTS / "A_risk_architecture_summary.json"))

def auc_binary(p, R):
    """Area under the curve of a single binary marker with prevalence p and odds ratio R."""
    p1 = R * p / (1.0 - p + R * p)            # exposure prevalence among cases (approximately)
    se, sp = p1, 1.0 - p
    return float(0.5 * (se + sp))

def liability_sd(R):
    """Liability-scale shift equivalent to an odds ratio, on the probit scale."""
    return float(np.log(R) / 1.81)            # logistic-to-probit scale factor pi/sqrt(3)

cmp_rows = [
    dict(exposure="polygenic score, top decile versus bottom decile",
         odds_ratio=A["or_decile10_vs_decile1"], prevalence=0.10,
         source_layer="Module A (genome)", src="Module A"),
    dict(exposure="atrial fibrillation", odds_ratio=A["af_implied_or_derived"],
         prevalence=K.epi("af_prevalence_pct") / 100.0, source_layer="Module A (clinical)",
         src="Module A"),
    dict(exposure="residence in the stroke belt", odds_ratio=or_belt, prevalence=0.20,
         source_layer="Module K (geography)", src=K.param_src("stroke_belt_or_prevalence")),
    dict(exposure="residence in the stroke belt, ages 18-64",
         odds_ratio=K.getp("stroke_belt_or_prevalence_18_64"), prevalence=0.20,
         source_layer="Module K (geography)",
         src=K.param_src("stroke_belt_or_prevalence_18_64")),
    dict(exposure="county in the lowest college-attainment band",
         odds_ratio=K.getp("county_or_no_college"), prevalence=0.25,
         source_layer="Module K (social)", src=K.param_src("county_or_no_college")),
    dict(exposure="county in the highest blood-pressure non-adherence band",
         odds_ratio=K.getp("county_or_bp_nonadherence"), prevalence=0.25,
         source_layer="Module K (social)", src=K.param_src("county_or_bp_nonadherence")),
    dict(exposure="living beyond 60 minutes of any stroke care",
         odds_ratio=(K.getp("stroke_prevalence_beyond_60min")
                     / K.getp("stroke_prevalence_within_60min")),
         prevalence=1.0 - K.getp("access_60min_any_stroke_care") / 100.0,
         source_layer="Module K (access)", src=K.param_src("stroke_prevalence_beyond_60min")),
]
cmp = pd.DataFrame(cmp_rows)
cmp["auc_alone"] = [auc_binary(r.prevalence, r.odds_ratio) if np.isfinite(r.odds_ratio) else np.nan
                    for r in cmp.itertuples()]
cmp["liability_sd_equivalent"] = [liability_sd(r.odds_ratio) if np.isfinite(r.odds_ratio)
                                  else np.nan for r in cmp.itertuples()]
cmp["population_attributable_fraction_pct"] = [
    100 * K.paf(r.prevalence, r.odds_ratio) if np.isfinite(r.odds_ratio) else np.nan
    for r in cmp.itertuples()]
cmp = cmp.sort_values("population_attributable_fraction_pct", ascending=False)
K.save_table(cmp, "K_exposure_comparison")

prs_var = 100 * A["liability_scale_variance_explained"]
modifiable = 100 * np.mean([K.getp("paf_combined_women"), K.getp("paf_combined_men")]) / 100
K.save_json(dict(
    clustering=dict(independent_combination_pct=float(clust.independent_combination.mean()),
                    verified_combined_pct=float(clust.verified_combined.mean()),
                    overlap_share_of_independent_pct=float(
                        clust.fraction_of_independent_estimate_that_is_overlap.mean()),
                    implied_shrinkage_per_factor=float(clust.implied_shrinkage_per_factor.mean())),
    geographic_mediation=dict(belt_prevalence_ratio=rr_belt,
                              hypertension_prevalence_ratio_in_belt=rr_ht,
                              share_mediated_by_hypertension_pct=100 * mediated),
    access=dict(share_beyond_60min_evt_capable=beyond,
                excess_poor_outcomes_per_1000_eligible_at_60min_steep=float(
                    acc.loc[acc.additional_transport_min == 60,
                            "absolute_good_outcome_lost_pct_steep_curve"].iloc[0]
                    / 100.0 * 1000.0 * beyond),
                excess_poor_outcomes_per_1000_eligible_at_60min=float(
                    acc.loc[acc.additional_transport_min == 60,
                            "excess_poor_outcomes_per_1000_eligible"].iloc[0]),
                largest_disparity_delay_equivalent_min=float(
                    equiv.equivalent_extra_transport_min.max())),
    attribution_gap=attr_gap,
    disparity=dict(log_rr_decline_per_year=b, modelled_convergence_age=converge_age,
                   residual_rr_45=residual(rr1, expl_45), residual_rr_65=residual(rr2, expl_65)),
    comparison=dict(polygenic_liability_variance_pct=prs_var,
                    modifiable_risk_factor_paf_pct=modifiable,
                    ratio_modifiable_to_genomic=float(modifiable / prs_var),
                    top_exposure_by_paf=str(cmp.iloc[0].exposure),
                    top_paf_pct=float(cmp.iloc[0].population_attributable_fraction_pct)),
), "K_social_summary")

print("Module K: risk-factor clustering")
print(clust[["sex", "sum_of_single_factor_pafs", "independent_combination", "verified_combined",
             "fraction_of_independent_estimate_that_is_overlap",
             "implied_shrinkage_per_factor"]].to_string(index=False,
                                                        float_format=lambda x: f"{x:.1f}"))
print(f"  geography: the stroke belt's verified prevalence odds ratio {or_belt:.2f} is a "
      f"prevalence ratio of {rr_belt:.2f}; the verified hypertension excess in the belt accounts "
      f"for {100*mediated:.1f}% of it, leaving {100*(1-mediated):.1f}% unexplained by "
      f"hypertension prevalence")
print(acc[["additional_transport_min", "or_after_delay", "absolute_good_outcome_lost_pct",
           "absolute_good_outcome_lost_pct_steep_curve",
           "excess_poor_outcomes_per_1000_eligible"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print(equiv[["disparity", "odds_ratio", "equivalent_extra_transport_min",
             "equivalent_extra_transport_min_steep"]]
      .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print(f"  attribution: GBD attributes {100*gbd_share_sbp:.1f}% of US stroke DALYs to high systolic "
      f"pressure, {float(gbd_share_sbp/paf_ht):.1f}x the verified binary hypertension attributable "
      f"fraction of {100*paf_ht:.1f}%")
print(f"  disparity: relative risk falls {100*(1-np.exp(-b)):.1f}% per year of age and reaches 1 at "
      f"{converge_age:.0f} y; after risk factors it is {residual(rr1, expl_45):.2f} at 45 "
      f"(published 2.14) and {residual(rr2, expl_65):.2f} at 65 (published 1.35)")
print(cmp[["exposure", "odds_ratio", "auc_alone", "liability_sd_equivalent",
           "population_attributable_fraction_pct"]].to_string(index=False,
                                                              float_format=lambda x: f"{x:.3f}"))
print(f"  the modifiable risk-factor burden ({modifiable:.1f}% attributable) is "
      f"{modifiable/prs_var:.1f}x the liability variance the polygenic score explains "
      f"({prs_var:.2f}%)")
