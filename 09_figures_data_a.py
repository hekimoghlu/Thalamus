#!/usr/bin/env python3
"""Figures 3-6: risk architecture, perfusion and tissue fate, cascade dynamics, biomarker timeline."""
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import kcore as K
import kfig as F

R = K.RESULTS
S = F.SERIES

# ============================================================ Figure 3: genomic risk architecture
loci = pd.read_csv(R / "A_locus_contributions.csv")
dec = pd.read_csv(R / "A_prs_deciles.csv")
sub = pd.read_csv(R / "A_subtype_discrimination.csv")
sens = pd.read_csv(R / "A_prs_sensitivity.csv")
sm = json.loads((R / "A_risk_architecture_summary.json").read_text())
prs = np.load(R / "A_prs_distribution.npy")

fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.9))
ax = axs[0, 0]
sub_colors = {"CES": S[0], "LAS": S[1], "AIS": S[2], "AS": S[3], "SVS": S[4]}
for st, g in loci.groupby("subtype"):
    ax.scatter(g.raf, g.or_est, s=8000 * g.var_contrib, color=sub_colors.get(st, F.MUTED),
               alpha=0.85, edgecolor="white", linewidth=0.6, zorder=3, label=st)
for _, r in loci.nlargest(3, "var_contrib").iterrows():
    ax.annotate(r.gene_symbol, (r.raf, r.or_est), textcoords="offset points", xytext=(6, 4),
                fontsize=6.4, color=F.INK2)
ax.set_xlabel("risk-allele frequency"); ax.set_ylabel("odds ratio per allele")
ax.set_yscale("log"); ax.set_yticks([1.05, 1.1, 1.2, 1.5, 2.0, 2.5])
ax.set_yticklabels(["1.05", "1.1", "1.2", "1.5", "2.0", "2.5"])
ax.minorticks_off()
F.legend(ax, title="subtype", ncols=2, loc="upper right", title_fontsize=6.8)
F.grid(ax, "both"); F.panel(ax, "a", "30 verified loci: effect vs frequency")

ax = axs[0, 1]
ax.hist(prs, bins=70, color=S[0], alpha=0.9, zorder=3)
q = np.quantile(prs, [0.1, 0.9])
for xq, lab in ((q[0], "decile 1"), (q[1], "decile 10")):
    ax.axvline(xq, color=F.INK2, lw=1.0, ls=(0, (4, 3)), zorder=4)
    ax.annotate(lab, (xq, ax.get_ylim()[1] * 0.92), fontsize=6.4, color=F.INK2,
                ha="center", backgroundcolor=F.SURFACE)
ax.set_xlabel("polygenic score (log-odds)"); ax.set_ylabel("simulated individuals")
F.grid(ax); F.panel(ax, "b", "score distribution (200 000 genotypes)")
ax.text(0.02, 0.86, f"per-SD OR {sm['per_sd_or_unshrunk']:.2f}\nAUC {sm['auc_polygenic']:.3f}",
        transform=ax.transAxes, fontsize=6.8, color=F.INK2, va="top")

ax = axs[1, 0]
ax.bar(dec.decile, dec.abs_risk_pct, color=[S[0] if d < 10 else S[1] for d in dec.decile],
       width=0.72, zorder=3)
for _, r in dec.iterrows():
    if r.decile in (1, 5, 10):
        ax.annotate(f"{r.abs_risk_pct:.1f}%", (r.decile, r.abs_risk_pct), ha="center",
                    va="bottom", fontsize=6.5, color=F.INK2, xytext=(0, 2),
                    textcoords="offset points")
ax.axhline(K.epi("stroke_prevalence_pct"), color=F.INK2, lw=1.0, ls=(0, (4, 3)), zorder=4)
ax.annotate(f"US adult prevalence {K.epi('stroke_prevalence_pct')}%", (10.4, K.epi("stroke_prevalence_pct")),
            fontsize=6.4, color=F.INK2, ha="right", va="bottom")
ax.set_xlabel("polygenic score decile"); ax.set_ylabel("absolute risk (%)")
ax.set_xticks(range(1, 11))
F.grid(ax); F.panel(ax, "c", "calibrated risk gradient across deciles")

ax = axs[1, 1]
y = np.arange(len(sub))
ax.barh(y, sub.auc, color=[sub_colors.get(s, F.MUTED) for s in sub.subtype], height=0.6, zorder=3)
ax.set_yticks(y); ax.set_yticklabels([f"{r.label}\n({int(r.n_loci)} loci)" for r in sub.itertuples()],
                                     fontsize=6.8)
for i, r in enumerate(sub.itertuples()):
    ax.annotate(f"{r.auc:.3f}", (r.auc, i), xytext=(3, 0), textcoords="offset points",
                va="center", fontsize=6.6, color=F.INK2)
ax.axvline(0.5, color=F.MUTED, lw=0.9)
ax.set_xlim(0.45, max(sub.auc) + 0.06); ax.set_xlabel("AUC of the subtype-restricted score")
ax.invert_yaxis(); F.grid(ax, "x")
F.panel(ax, "d", "discrimination is concentrated in cardioembolic stroke")
fig.suptitle("Figure 3  ·  Layer 1: the genomic risk architecture", fontsize=10.5, x=0.005,
             ha="left", y=1.005)
F.note(fig, f"All odds ratios and allele frequencies are verified published values (Table S2). The score is "
            f"calibrated so that simulated prevalence equals the CDC/NHIS adult prevalence. Effect estimates are "
            f"discovery-stage and therefore upward biased: under 30% shrinkage the per-SD OR falls from "
            f"{sm['per_sd_or_unshrunk']:.2f} to {sm['per_sd_or_shrink0.7']:.2f} and AUC spans "
            f"{sm['auc_range_over_sensitivity'][0]:.2f}-{sm['auc_range_over_sensitivity'][1]:.2f} (Table S10). "
            f"For comparison, inverting the PAF expression on two independent CDC figures (AF prevalence "
            f"{100*sm['af_prevalence']:.0f}%, {100*sm['af_attributable_share']:.1f}% of strokes attributable) "
            f"implies an atrial-fibrillation odds ratio of {sm['af_implied_or_derived']:.1f} - "
            f"{sm['polygenic_equivalent_of_af']:.1f} SD of the polygenic score.", y=0.0)
fig.tight_layout()
F.save(fig, "figure_03_risk_architecture")

# ====================================================== Figure 4: perfusion and tissue-fate kinetics
dd = pd.read_csv(R / "B_depth_duration_curve.csv")
val = pd.read_csv(R / "B_tissue_fate_validation.csv")
igr = np.load(R / "B_igr.npy"); igr_alt = np.load(R / "B_igr_proportion_matched.npy")
coll = np.load(R / "B_collateral.npy"); traj = np.load(R / "B_trajectories.npy")
tg = np.load(R / "B_time_grid.npy")
bs = json.loads((R / "B_perfusion_summary.json").read_text())

fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.8))
ax = axs[0, 0]
m = dd.t_crit_h < 30
ax.plot(dd.cbf[m], dd.t_crit_h[m], color=S[0], zorder=3)
pts = [(K.getp("depth_duration_30min"), 0.5), (K.getp("depth_duration_6h"), 6.0),
       (K.getp("depth_duration_12h"), 12.0)]
ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=34, color=F.RED, zorder=5,
           edgecolor="white", linewidth=0.8, label="verified anchor points")
ax.axvspan(K.getp("cbf_penumbra_lower"), K.getp("cbf_penumbra_upper"), color=S[1], alpha=0.12,
           zorder=1)
ax.annotate("penumbra band\n12-22 mL/100 g/min", (17, 19), fontsize=6.4, color=F.INK2, ha="center")
ax.annotate("oligemia ceiling:\ntolerance unbounded", (46, 21), fontsize=6.4, color=F.INK2, ha="right")
ax.set_xlabel("cerebral blood flow (mL/100 g/min)"); ax.set_ylabel("tolerated duration (h)")
ax.set_ylim(0, 26); ax.set_xlim(6, 50); F.legend(ax, loc="upper left")
F.grid(ax); F.panel(ax, "a", "depth-duration tolerance curve")

ax = axs[0, 1]
ax.hist(igr, bins=60, color=S[0], alpha=0.9, zorder=3, label="model (CTP-median calibration)")
ax.hist(igr_alt, bins=60, color=S[1], alpha=0.55, zorder=4,
        label="model (progressor-proportion calibration)")
for v, lab, c in ((K.getp("igr_median"), "published median 8.1", F.INK2),
                  (K.getp("igr_fast_cut"), "fast-progressor cut 10", F.RED)):
    ax.axvline(v, color=c, lw=1.0, ls=(0, (4, 3)), zorder=5)
ax.annotate("published\nmedian 8.1", (K.getp("igr_median") + 0.6, ax.get_ylim()[1] * 0.62),
            fontsize=6.4, color=F.INK2, ha="left", va="top")
ax.set_xlim(0, 40); ax.set_xlabel("infarct growth rate (mL/h)"); ax.set_ylabel("simulated patients")
F.legend(ax, loc="upper right", fontsize=6.6)
F.grid(ax); F.panel(ax, "b", "growth rate: two irreconcilable published targets")

ax = axs[1, 0]
for i in range(0, 240, 3):
    ax.plot(tg, traj[i], color=S[0], alpha=0.08, lw=0.7, zorder=2)
ax.plot(tg, np.median(traj, axis=0), color=F.RED, lw=2.2, zorder=4, label="median trajectory")
ax.axhline(K.getp("typical_lvo_infarct_volume"), color=F.INK2, ls=(0, (4, 3)), lw=1.0, zorder=5)
ax.annotate("published typical LVO infarct 54 mL", (23.5, K.getp("typical_lvo_infarct_volume")),
            fontsize=6.4, color=F.INK2, ha="right", va="bottom")
ax.set_xlabel("hours from occlusion"); ax.set_ylabel("infarct volume (mL)")
F.legend(ax, loc="lower right")
F.grid(ax); F.panel(ax, "c", "individual infarct-volume trajectories")

ax = axs[1, 1]
oos = val[~val.used_in_calibration]
yv = np.arange(len(oos))
ax.scatter(oos.ratio_model_to_published, yv, s=30, color=S[0], zorder=4)
ax.axvline(1.0, color=F.INK2, lw=1.0)
ax.axvspan(0.75, 1.25, color=S[2], alpha=0.13, zorder=1)
ax.set_yticks(yv); ax.set_yticklabels(
    [q.replace(" (mL/h)", "").replace(" (%)", "").replace(" (h)", "").replace(" (mL)", "")
     for q in oos.quantity], fontsize=6.5)
ax.set_xscale("log"); ax.set_xticks([0.25, 0.5, 1, 2, 4])
ax.set_xticklabels(["0.25x", "0.5x", "1x", "2x", "4x"])
ax.set_xlabel("model / published"); ax.invert_yaxis()
F.grid(ax, "x"); F.panel(ax, "d", "out-of-sample comparisons (shaded: within 25%)")
fig.suptitle("Figure 4  ·  Layer 2a: perfusion, collateral supply and tissue fate", fontsize=10.5,
             x=0.005, ha="left", y=1.005)
F.note(fig, f"One free parameter (the at-risk territory volume, calibrated to {bs['median_igr']:.1f} mL/h) and one "
            f"inferred parameter (territory-volume dispersion sigma={bs['territory_volume_dispersion_sigma_inferred']:.2f}, "
            f"set by the published collateral-growth correlation r={bs['collateral_igr_correlation_published']}) fix the "
            f"model; collateral status then explains {100*bs['variance_in_igr_explained_by_collateral']:.0f}% of the "
            f"variance in growth rate and territory size {100*bs['variance_in_igr_explained_by_territory_volume']:.0f}%. "
            f"Panel b shows why no single population can satisfy both published summaries "
            f"(a median of 8.1 mL/h is incompatible with 63.5% of patients below 5 mL/h).", y=0.0)
fig.tight_layout()
F.save(fig, "figure_04_perfusion")
