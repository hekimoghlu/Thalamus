#!/usr/bin/env python3
"""Module A - pre-stroke risk architecture (layer L1).

What is computed here, from verified inputs only:
  1. A genome-wide additive log-odds polygenic score built from 30 stroke GWAS loci whose
     per-allele odds ratios and risk-allele frequencies were read from the source publications.
  2. Its population distribution, decile risk gradient, discrimination (AUC) and population
     attributable fraction, under a liability-calibrated logistic model whose intercept is set
     so that simulated lifetime-equivalent prevalence matches the CDC/NHIS adult stroke
     prevalence (3.1%).
  3. Subtype-resolved scores (cardioembolic, large-artery, small-vessel, any/undifferentiated)
     showing that discrimination is not uniform across the stroke syndromes.
  4. A derived effect size for atrial fibrillation obtained by inverting the PAF expression on
     two independently verified public numbers (AF prevalence in US adults; the share of
     strokes attributable to AF), used as the single non-genomic comparator.
  5. Monte-Carlo propagation of the published CI on every odds ratio into every summary.

Nothing here is fitted to patient-level data; no patient-level data is used anywhere.
"""
import numpy as np, pandas as pd
import kcore as K

rng = np.random.default_rng(K.SEED)
N = 200_000
NBOOT = 400

loci = K.load_verified("gwas_loci", numeric=["or_est", "or_lo", "or_hi", "raf"],
                       require=["or_est", "raf"])
loci = loci[loci["use_in_prs"] == "1"].reset_index(drop=True)
loci["beta"] = np.log(loci["or_est"])
loci["se"] = [K.se_from_ci(lo, hi) if np.isfinite(lo) and np.isfinite(hi) else np.nan
              for lo, hi in zip(loci["or_lo"], loci["or_hi"])]
print(f"Module A: {len(loci)} loci with verified OR and RAF "
      f"({(loci['verified']=='primary').sum()} primary, {(loci['verified']=='secondary').sum()} secondary)")

# ---------------------------------------------------------------- genotype simulation (HWE)
def simulate_dosages(raf, n, rng):
    return rng.binomial(2, raf[None, :], size=(n, len(raf))).astype(np.int8)

G = simulate_dosages(loci["raf"].to_numpy(), N, rng)

def prs_of(G, beta, idx=None):
    if idx is None:
        return G @ beta
    return G[:, idx] @ beta[idx]

beta = loci["beta"].to_numpy()
prs_all = prs_of(G, beta)

# Variance of the score attributable to each locus: 2p(1-p)beta^2 (additive, HWE)
loci["var_contrib"] = 2 * loci["raf"] * (1 - loci["raf"]) * loci["beta"] ** 2
Vg = loci["var_contrib"].sum()
# Liability-scale heritability implied by a log-odds score under the probit approximation
K_prev = K.epi("stroke_prevalence_pct") / 100.0
h2_liab = Vg / (Vg + np.pi ** 2 / 3)   # logistic residual variance = pi^2/3

# ------------------------------------------------- liability-calibrated absolute risk model
def calibrate_intercept(score, target_prev):
    lo, hi = -20.0, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if K.expit(mid + score).mean() > target_prev:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

a0 = calibrate_intercept(prs_all, K_prev)
risk = K.expit(a0 + prs_all)
y = rng.binomial(1, risk)

auc_all = K.auc_from_scores(prs_all, y)
dec = pd.qcut(prs_all, 10, labels=False)
dec_tab = pd.DataFrame({
    "decile": np.arange(1, 11),
    "mean_prs": [prs_all[dec == d].mean() for d in range(10)],
    "abs_risk_pct": [100 * risk[dec == d].mean() for d in range(10)],
})
ref = dec_tab.loc[4:5, "abs_risk_pct"].mean()          # deciles 5-6 as reference
dec_tab["risk_ratio_vs_mid"] = dec_tab["abs_risk_pct"] / ref
or_top_vs_bottom = ((dec_tab.loc[9, "abs_risk_pct"] / (100 - dec_tab.loc[9, "abs_risk_pct"])) /
                    (dec_tab.loc[0, "abs_risk_pct"] / (100 - dec_tab.loc[0, "abs_risk_pct"])))
top1 = prs_all >= np.quantile(prs_all, 0.99)
rr_top1 = risk[top1].mean() / np.median(risk)

# ------------------------------------------------------------------- subtype-resolved scores
subtype_map = {"CES": "cardioembolic", "LAS": "large-artery", "SVS": "small-vessel",
               "AIS": "any ischemic", "AS": "any stroke"}
sub_rows = []
for code, label in subtype_map.items():
    idx = np.where(loci["subtype"].to_numpy() == code)[0]
    if len(idx) == 0:
        continue
    s = prs_of(G, beta, idx)
    a0s = calibrate_intercept(s, K_prev)
    rs = K.expit(a0s + s)
    ys = rng.binomial(1, rs)
    q = np.quantile(s, [0.1, 0.9])
    sub_rows.append(dict(subtype=code, label=label, n_loci=len(idx),
                         var_explained_logodds=float(loci.loc[idx, "var_contrib"].sum()),
                         auc=K.auc_from_scores(s, ys),
                         decile10_vs_1_rr=float(rs[s >= q[1]].mean() / rs[s <= q[0]].mean()),
                         max_single_or=float(loci.loc[idx, "or_est"].max()),
                         top_locus=loci.loc[idx, "gene_symbol"][loci.loc[idx, "or_est"].idxmax()]))
sub_tab = pd.DataFrame(sub_rows).sort_values("auc", ascending=False)

# ------------------------------------------------------- Monte-Carlo over published OR CIs
have_ci = np.isfinite(loci["se"].to_numpy())
boot = {"auc": [], "h2_liab": [], "or_top_bottom": [], "Vg": []}
Gsub = G[:40_000]
for _ in range(NBOOT):
    b = beta.copy()
    b[have_ci] = rng.normal(beta[have_ci], loci["se"].to_numpy()[have_ci])
    s = Gsub @ b
    v = float((2 * loci["raf"] * (1 - loci["raf"]) * b ** 2).sum())
    a0b = calibrate_intercept(s, K_prev)
    rb = K.expit(a0b + s)
    yb = rng.binomial(1, rb)
    d = pd.qcut(s, 10, labels=False)
    p10, p1 = rb[d == 9].mean(), rb[d == 0].mean()
    boot["auc"].append(K.auc_from_scores(s, yb))
    boot["Vg"].append(v)
    boot["h2_liab"].append(v / (v + np.pi ** 2 / 3))
    boot["or_top_bottom"].append((p10 / (1 - p10)) / (p1 / (1 - p1)))
ci = {k: (float(np.quantile(v, .025)), float(np.quantile(v, .975))) for k, v in boot.items()}

# -------------------------------------------- verified-number-derived comparator: atrial fib
af_prev = K.epi("af_prevalence_pct") / 100
af_paf = K.epi("af_attributable_stroke_share") / 100
af_or = K.or_from_paf(af_prev, af_paf)

# PAF of the polygenic component: compare observed prevalence to the counterfactual in which
# every individual carries the population-minimum genotype score.
risk_min = K.expit(a0 + prs_all.min())
paf_poly = 1 - risk_min / risk.mean()
paf_top20 = (np.mean(risk) - np.mean(K.expit(a0 + np.minimum(prs_all, np.quantile(prs_all, .8))))) / np.mean(risk)

summary = dict(
    n_loci=int(len(loci)), n_simulated=N,
    total_score_variance_logodds=float(Vg), total_score_variance_ci=ci["Vg"],
    liability_scale_variance_explained=float(h2_liab), liability_ci=ci["h2_liab"],
    auc_polygenic=float(auc_all), auc_ci=ci["auc"],
    or_decile10_vs_decile1=float(or_top_vs_bottom), or_decile_ci=ci["or_top_bottom"],
    risk_ratio_top1pct_vs_median=float(rr_top1),
    calibrated_prevalence_pct=float(100 * risk.mean()),
    target_prevalence_pct=float(100 * K_prev),
    paf_whole_polygenic_component=float(paf_poly),
    paf_top20pct_of_score=float(paf_top20),
    af_prevalence=af_prev, af_attributable_share=af_paf,
    af_implied_or_derived=float(af_or),
    af_implied_or_note=("derived by inverting PAF = p(OR-1)/(1+p(OR-1)) on two public figures, both "
                        "flagged secondary in the provenance table; not a measured odds ratio"),
    polygenic_equivalent_of_af=float(np.log(af_or) / prs_all.std()),
    polygenic_sd_logodds=float(prs_all.std()),
)

# ------------------------------------------------- sensitivity: rare loci and winner's curse
# Discovery-stage effect estimates are upward-biased (winner's curse), so the score variance
# computed from published ORs is an upper bound. Two sensitivity axes are reported rather than
# a single point estimate.
sens_rows = []
common = np.where(loci["raf"].to_numpy() >= 0.05)[0]
for shrink in (1.0, 0.9, 0.8, 0.7):
    for label, idx in (("all loci", None), ("RAF>=0.05 only", common)):
        b = beta * shrink
        s_ = prs_of(Gsub, b, idx)
        a0_ = calibrate_intercept(s_, K_prev)
        r_ = K.expit(a0_ + s_)
        y_ = rng.binomial(1, r_)
        d_ = pd.qcut(s_, 10, labels=False)
        p10_, p1_ = r_[d_ == 9].mean(), r_[d_ == 0].mean()
        sens_rows.append(dict(effect_shrinkage=shrink, locus_set=label,
                              n_loci=len(beta) if idx is None else len(idx),
                              score_sd=float(s_.std()),
                              per_sd_or=float(np.exp(s_.std())),
                              auc=K.auc_from_scores(s_, y_),
                              or_decile10_vs_1=float((p10_/(1-p10_))/(p1_/(1-p1_)))))
sens = pd.DataFrame(sens_rows)
K.save_table(sens, "A_prs_sensitivity")
summary["per_sd_or_unshrunk"] = float(np.exp(prs_all.std()))
summary["per_sd_or_shrink0.7"] = float(sens.query("effect_shrinkage==0.7 and locus_set=='all loci'")["per_sd_or"].iloc[0])
summary["auc_range_over_sensitivity"] = [float(sens["auc"].min()), float(sens["auc"].max())]
K.save_json(summary, "A_risk_architecture_summary")
print(f"  per-SD OR = {summary['per_sd_or_unshrunk']:.3f} (unshrunk) to {summary['per_sd_or_shrink0.7']:.3f} (30% shrinkage)")
print(f"  AUC across sensitivity grid: {summary['auc_range_over_sensitivity'][0]:.3f}-{summary['auc_range_over_sensitivity'][1]:.3f}")

K.save_json(summary, "A_risk_architecture_summary")
K.save_table(dec_tab, "A_prs_deciles")
K.save_table(sub_tab, "A_subtype_discrimination")
K.save_table(loci[["locus", "gene_symbol", "rs_id", "subtype", "or_est", "or_lo", "or_hi",
                   "raf", "beta", "var_contrib", "src", "verified"]], "A_locus_contributions")
np.save(K.RESULTS / "A_prs_distribution.npy", prs_all[:50_000].astype(np.float32))
np.save(K.RESULTS / "A_risk_distribution.npy", risk[:50_000].astype(np.float32))

print(f"  AUC (polygenic, {len(loci)} loci) = {auc_all:.4f}  95% MC CI {ci['auc'][0]:.4f}-{ci['auc'][1]:.4f}")
print(f"  liability-scale variance explained = {100*h2_liab:.2f}%  CI {100*ci['h2_liab'][0]:.2f}-{100*ci['h2_liab'][1]:.2f}%")
print(f"  OR decile 10 vs 1 = {or_top_vs_bottom:.2f}  CI {ci['or_top_bottom'][0]:.2f}-{ci['or_top_bottom'][1]:.2f}")
print(f"  AF implied OR (derived) = {af_or:.2f}; equals {summary['polygenic_equivalent_of_af']:.2f} SD of the polygenic score")
print(f"  PAF of top 20% of the score = {100*paf_top20:.1f}%")
