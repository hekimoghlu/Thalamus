#!/usr/bin/env python3
"""Figures 15-18 and the rebuilt Figures 2 and 10: social gradient, posteriors, control policy,
statistical hardening, the framework architecture and the validation accounting."""
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import kcore as K
import kfig as F

S = F.SERIES
R = K.RESULTS
load = lambda n: pd.read_csv(R / n)
js = lambda n: json.load(open(R / n))

def ratio_axis(ax, lo=0.05, hi=20):
    ax.axvline(1.0, color=F.INK2, lw=1.0, zorder=3)
    ax.axvspan(0.75, 1.25, color=S[2], alpha=0.13, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4, 10])
    ax.set_xticklabels(["0.1x", "0.25x", "0.5x", "1x", "2x", "4x", "10x"])

# ==================================================== Figure 15 - the social and access layer
KS = js("K_social_summary.json")
clust = load("K_risk_factor_clustering.csv")
acc = load("K_access_geography.csv")
eq = load("K_disparity_delay_equivalents.csv")
cmp_ = load("K_exposure_comparison.csv")
dis = load("K_disparity_age_structure.csv")

fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.8), layout='constrained')
fig.suptitle("Figure 15  ·  Where someone lives, and how far they are from care",
             x=0.007, ha="left", fontsize=10.5, fontweight="bold")

ax = axs[0, 0]
F.panel(ax, "a", "risk factors overlap")
x = np.arange(len(clust))
w = 0.26
ax.bar(x - w, clust.sum_of_single_factor_pafs, w, color=F.GRID, label="sum of single factors")
ax.bar(x, clust.independent_combination, w, color=S[0], label="if independent")
ax.bar(x + w, clust.verified_combined, w, color=S[2], label="verified combined")
ax.errorbar(x + w, clust.verified_combined,
            yerr=[clust.verified_combined - clust.verified_lo,
                  clust.verified_hi - clust.verified_combined],
            fmt="none", ecolor=F.INK2, capsize=3, lw=1.0)
ax.set_xticks(x); ax.set_xticklabels(clust.sex)
ax.set_ylabel("attributable fraction (%)")
F.legend(ax, loc="upper right", fontsize=6.2); F.grid(ax)

ax = axs[0, 1]
F.panel(ax, "b", "what a delay costs")
ax.plot(acc.additional_transport_min, acc.absolute_good_outcome_lost_pct, "o-", color=S[0],
        label="framework's salvage curve")
ax.plot(acc.additional_transport_min, acc.absolute_good_outcome_lost_pct_steep_curve, "s--",
        color=S[1], label="curve the published ladder implies")
ax.set_xlabel("additional minutes to a thrombectomy centre")
ax.set_xticks([30, 60, 90, 120])
ax.set_ylabel("good outcomes lost (percentage points)")
F.legend(ax, loc="upper left", fontsize=6.3); F.grid(ax)

ax = axs[1, 0]
F.panel(ax, "c", "disparities as an equivalent delay")
e = eq.sort_values("equivalent_extra_transport_min_steep")
y = np.arange(len(e))
ax.hlines(y, e.equivalent_extra_transport_min_steep, e.equivalent_extra_transport_min,
          color=F.GRID, lw=3.0, zorder=1)
ax.scatter(e.equivalent_extra_transport_min_steep, y, color=S[1], s=34, zorder=4,
           label="steeper curve (lower bound)")
ax.scatter(e.equivalent_extra_transport_min, y, color=S[0], s=34, zorder=4,
           label="framework's curve")
ax.set_yticks(y); ax.set_yticklabels(F.trim(e.disparity, 44), fontsize=6.0)
ax.set_xscale("log"); ax.set_xlabel("equivalent extra transport time (min)")
ax.set_xticks([10, 30, 60, 120, 300, 600])
ax.set_xticklabels(["10", "30", "60", "120", "300", "600"])
ax.set_ylim(-1.35, len(e) - 0.2)
F.legend(ax, loc="lower right", fontsize=6.0); F.grid(ax, "x")

ax = axs[1, 1]
F.panel(ax, "d", "genome against postcode")
c = cmp_.sort_values("population_attributable_fraction_pct")
col = [S[0] if "genome" in s or "clinical" in s else S[3] for s in c.source_layer]
ax.barh(np.arange(len(c)), c.population_attributable_fraction_pct, color=col, height=0.6)
ax.set_yticks(np.arange(len(c))); ax.set_yticklabels(F.trim(c.exposure, 46), fontsize=5.8)
ax.set_xlabel("population attributable fraction (%)")
for i, (v, a) in enumerate(zip(c.population_attributable_fraction_pct, c.auc_alone)):
    ax.text(v + 0.4, i, f"AUC {a:.2f}", va="center", fontsize=5.8, color=F.MUTED)
ax.set_xlim(0, c.population_attributable_fraction_pct.max() * 1.40)
F.grid(ax, "x")

F.note(fig, f"Every number here is computed from verified public estimates rather than asserted. "
            f"The verified single-factor attributable fractions, combined as if independent, give "
            f"{clust.independent_combination.mean():.0f}% against a verified combined figure of "
            f"{clust.verified_combined.mean():.0f}%, so "
            f"{clust.fraction_of_independent_estimate_that_is_overlap.mean():.0f}% of the naive "
            f"estimate is overlap. Hypertension prevalence accounts for only "
            f"{KS['geographic_mediation']['share_mediated_by_hypertension_pct']:.0f}% of the "
            f"stroke belt's excess. The modifiable burden is "
            f"{KS['comparison']['ratio_modifiable_to_genomic']:.0f} times the liability variance "
            f"the polygenic score explains.", y=0.005)
F.save(fig, "figure_15_social_gradient")

# =================================================== Figure 16 - posteriors (Module L)
post = load("L_tolerance_posterior.csv")
band = load("L_tolerance_credible_band.csv")
pred = load("L_tolerance_predictive.csv")

fig, axs = plt.subplots(1, 3, figsize=(7.5, 2.9), layout='constrained')
fig.suptitle("Figure 16  ·  The tolerance curve with its uncertainty, and what that implies",
             x=0.007, ha="left", fontsize=10.5, fontweight="bold")

ax = axs[0]
F.panel(ax, "a", "depth-duration tolerance")
ax.fill_between(band.cbf, band.t_crit_lo, band.t_crit_hi, color=S[0], alpha=0.20, lw=0,
                label="95% credible band")
ax.plot(band.cbf, band.t_crit_median, color=S[0], label="posterior median")
for f, t in ((K.getp("depth_duration_30min"), 0.5), (K.getp("depth_duration_6h"), 6.0),
             (K.getp("depth_duration_12h"), 12.0)):
    ax.plot(f, t, "o", color=F.RED, ms=5, zorder=5)
ax.set_ylim(0, 26); ax.set_xlim(6, 50)
ax.set_xlabel("cerebral blood flow (mL/100 g/min)"); ax.set_ylabel("tolerated duration (h)")
ax.text(0.03, 0.93, "red: verified anchor points", transform=ax.transAxes, fontsize=6.3,
        color=F.RED, va="top")
F.legend(ax, loc="upper left", fontsize=6.3); F.grid(ax)

ax = axs[1]
F.panel(ax, "b", "posterior summaries")
p = post[post.parameter.isin(["slope_low", "slope_high", "sigma_V", "log_tau_between_source"])]
y = np.arange(len(p))
ax.hlines(y, p.q025, p.q975, color=S[0], lw=2.4)
ax.plot(p["median"], y, "o", color=S[0], ms=5)
ax.set_yticks(y)
ax.set_yticklabels(["low-flow slope", "high-flow slope", "territory dispersion",
                    "log between-source SD"], fontsize=6.6)
ax.set_xlabel("posterior median and 95% credible interval")
F.grid(ax, "x")

ax = axs[2]
F.panel(ax, "c", "convergence")
ax.barh(np.arange(len(post)), post.ess, color=S[2], height=0.6)
ax.set_yticks(np.arange(len(post)))
ax.set_yticklabels(F.trim(post.parameter, 22), fontsize=6.2)
ax.set_xlabel("effective sample size")
for i, r in enumerate(post.rhat):
    ax.text(post.ess.iloc[i] + 25, i, f"R-hat {r:.3f}", va="center", fontsize=5.8, color=F.MUTED)
ax.set_xlim(0, post.ess.max() * 1.45)
F.grid(ax, "x")

F.note(fig, "Four chains of an adaptive random-walk Metropolis sampler, written out in full "
            "because no probabilistic-programming package was installable in the environment this "
            "work was produced in. The likelihood combines the three verified flow-versus-time "
            "infarction points with the verified growth-rate median, its interquartile range and "
            "the progressor proportions; because Module B established that the last two published "
            "series are mutually incompatible, a between-source standard deviation is estimated "
            "rather than assumed away, and its posterior is the measurement of the disagreement.",
       y=0.005)
F.save(fig, "figure_16_posteriors")

# ============================================ Figure 17 - the control policy (Module M)
M = js("M_control_summary.json")
pol = load("M_policy_outcomes.csv")
sch = load("M_optimal_schedule.csv")
epq = load("M_epoch_efficacy.csv")
prc = load("M_device_equivalence.csv")

fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.6), layout='constrained')
fig.suptitle("Figure 17  ·  When to spend the therapy hours, solved rather than assumed",
             x=0.007, ha="left", fontsize=10.5, fontweight="bold")

ax = axs[0, 0]
F.panel(ax, "a", "the optimal schedule")
for bud, col in zip(sorted(sch.budget_hours.unique()), (S[0], S[1], S[2])):
    d = sch[(sch.budget_hours == bud) & (sch.delivery == "therapist")]
    ax.plot(d.week, d.mean_hours, color=col, label=f"{bud:.0f} h available")
ax2 = ax.twinx()
d0 = sch[(sch.budget_hours == sch.budget_hours.min()) & (sch.delivery == "therapist")]
ax2.fill_between(d0.week, d0.remaining_value, color=F.MUTED, alpha=0.10, lw=0)
ax2.set_ylabel("value still recoverable", fontsize=6.6, color=F.MUTED)
ax2.tick_params(labelsize=6.2, colors=F.MUTED)
ax2.spines["right"].set_visible(False)
ax.set_xlabel("weeks after stroke"); ax.set_ylabel("therapy hours in that week")
F.legend(ax, loc="upper right", fontsize=6.3); F.grid(ax)

ax = axs[0, 1]
F.panel(ax, "b", "concentrating the same hours")
t = pol[pol.delivery == "therapist-delivered"]
x = np.arange(len(t)); w = 0.35
ax.bar(x - w / 2, t.treatment_attributable_fm_gain, w, color=S[0], label="optimal timing")
ax.bar(x + w / 2, t.gain_if_dose_spread_uniformly, w, color=F.GRID, label="spread evenly")
ax.errorbar(x - w / 2, t.treatment_attributable_fm_gain,
            yerr=[t.treatment_attributable_fm_gain - t.gain_ci_lo,
                  t.gain_ci_hi - t.treatment_attributable_fm_gain],
            fmt="none", ecolor=F.INK2, capsize=3, lw=1.0)
ax.axhline(K.getp("fmue_mcid"), color=F.RED, ls="--", lw=1.1)
ax.text(len(t) - 0.4, K.getp("fmue_mcid") + 0.12, "minimal clinically\nimportant difference",
        fontsize=6.0, color=F.RED, ha="right")
ax.set_xticks(x); ax.set_xticklabels([f"{b:.0f} h" for b in t.budget_hours])
ax.set_ylabel("treatment-attributable FM-UE points")
F.legend(ax, loc="lower right", fontsize=6.3); F.grid(ax)

ax = axs[1, 0]
F.panel(ax, "c", "same dose, three epochs")
x = np.arange(len(epq)); w = 0.35
ax.bar(x - w / 2, epq.model_ratio_to_subacute, w, color=S[0], label="model")
ax.bar(x + w / 2, epq.published_ratio_to_subacute, w, color=S[1], label="published trial")
ax.set_xticks(x); ax.set_xticklabels(epq.epoch)
ax.set_ylabel("efficacy relative to subacute")
ax.set_ylim(0, 2.05)
ax.text(0.98, 0.97, "the framework's window declines monotonically;\nthe trial says it PEAKS "
                    "subacutely - a structural failure",
        transform=ax.transAxes, va="top", ha="right", fontsize=6.1, color=F.RED)
F.legend(ax, loc="upper left", fontsize=6.3); F.grid(ax)

ax = axs[1, 1]
F.panel(ax, "d", "the dose threshold, and what the device does to it")
p = prc.copy()
thr_t, thr_d = float(p.threshold_hours_therapist.iloc[0]), float(p.threshold_hours_device.iloc[0])
x = np.logspace(np.log10(0.02), np.log10(4.0), 240)
ax.plot(x * 60, [max(0.0, 1 + h - 1.723) for h in x], color=S[0], label="therapist-delivered")
ax.plot(x * 60, [max(0.0, 1 + h * p.verified_repetition_ratio.iloc[0] - 1.723) for h in x],
        color=S[1], label="device-delivered")
ax.axvline(thr_t * 60, color=S[0], ls="--", lw=1.0)
ax.axvline(thr_d * 60, color=S[1], ls="--", lw=1.0)
ax.set_xscale("log")
ax.set_xticks([3, 10, 43, 120, 240])
ax.set_xticklabels(["3", "10", "43", "120", "240"])
ax.set_xlabel("minutes of practice per week")
ax.set_ylabel("dose above the verified threshold")
ax.text(thr_d * 60 * 1.1, ax.get_ylim()[1] * 0.86, f"{thr_d*60:.0f} min", fontsize=6.2, color=S[1])
ax.text(thr_t * 60 * 1.1, ax.get_ylim()[1] * 0.70, f"{thr_t*60:.0f} min", fontsize=6.2, color=S[0])
ax.text(0.03, 0.05, "below its threshold each mode buys nothing at all",
        transform=ax.transAxes, fontsize=6.0, color=F.MUTED)
F.legend(ax, loc="upper left", fontsize=6.3); F.grid(ax)

F.note(fig, "The program is solved exactly by backward induction over the achievable ceiling and "
            "the remaining hour budget, with the verified treatment-window gradients setting how "
            "much a week is worth and the verified threshold dose-response setting how little a "
            "thin dose buys. Panel b is the result that matters to a service: at a 20-hour budget "
            "the same hours spread evenly across the year fall below the verified dose threshold "
            "and buy nothing at all, while concentrated they clear the minimal clinically "
            "important difference. Panel c is the framework's own failure, reported rather than "
            "tuned away.", y=0.005)
F.save(fig, "figure_17_control_policy")

# ================================== Figure 18 - the statistical hardening pass (Module N)
N = js("N_statistics_summary.json")
prof = load("N_profile_likelihood.csv")
sob = load("N_sobol_indices.csv")
bench = load("N_benchmark_multiplicity.csv")
boot = load("N_bootstrap_intervals.csv")

fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.8), layout='constrained')
fig.suptitle("Figure 18  ·  Intervals, identifiability, interaction and multiplicity",
             x=0.007, ha="left", fontsize=10.5, fontweight="bold")

ax = axs[0, 0]
F.panel(ax, "a", "bootstrap intervals")
b = boot.copy()
b["rel_lo"] = b.ci_lo / b.value
b["rel_hi"] = b.ci_hi / b.value
b = b.sort_values("rel_hi")
y = np.arange(len(b))
ax.hlines(y, b.rel_lo, b.rel_hi, color=S[0], lw=2.6)
ax.plot(np.ones(len(b)), y, "|", color=F.INK, ms=8)
ax.set_yticks(y); ax.set_yticklabels(F.trim(b.quantity, 40), fontsize=5.8)
ax.set_xlabel("95% interval, relative to the point estimate")
F.grid(ax, "x")

ax = axs[0, 1]
F.panel(ax, "b", "profile-likelihood identifiability")
pf = prof.dropna(subset=["ci_lo", "ci_hi"]).copy()
pf["width"] = (pf.ci_hi - pf.ci_lo) / np.abs(pf.point_estimate).clip(lower=1e-6)
pf = pf.sort_values("width")
y = np.arange(len(pf))
col = [S[2] if i else F.RED for i in pf.identified]
ax.barh(y, pf.width.clip(upper=40), color=col, height=0.6)
ax.set_yticks(y); ax.set_yticklabels(F.trim(pf.parameter, 30), fontsize=6.0)
ax.set_xscale("log")
ax.set_xlabel("interval width, relative to the estimate")
ax.text(0.98, 0.04, "red: not identified by the data used to fit it", transform=ax.transAxes,
        ha="right", fontsize=6.0, color=F.RED)
F.grid(ax, "x")

ax = axs[1, 0]
F.panel(ax, "c", "first-order against total-order")
for mod, col, mk in (("tissue fate", S[0], "o"), ("rehabilitation control", S[1], "s")):
    d = sob[sob.model == mod]
    ax.scatter(d.first_order, d.total_order, color=col, s=34, label=mod, marker=mk, alpha=0.85)
lim = [0, 1.02]
ax.plot(lim, lim, color=F.MUTED, lw=0.9, ls="--")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("first-order Sobol index"); ax.set_ylabel("total-order Sobol index")
ax.text(0.04, 0.93, "distance above the diagonal is interaction", transform=ax.transAxes,
        fontsize=6.2, color=F.MUTED)
F.legend(ax, loc="lower right", fontsize=6.3); F.grid(ax)

ax = axs[1, 1]
F.panel(ax, "d", "benchmark disagreements after multiplicity control")
t = bench[np.isfinite(bench.p_value)].sort_values("p_value").copy()
y = np.arange(len(t))
ax.scatter(t.p_value, y, color=F.MUTED, s=26, label="uncorrected p", zorder=3)
ax.scatter(t.p_bh, y, color=S[0], s=30, label="Benjamini-Hochberg", zorder=4)
ax.axvline(0.05, color=F.RED, ls="--", lw=1.1)
ax.set_xscale("log")
ax.set_yticks(y); ax.set_yticklabels(F.trim(t.module + " · " + t.quantity, 42), fontsize=5.6)
ax.set_xlabel("p-value for disagreement with the published value")
ax.text(0.05, 1.02, f"{N['benchmarks']['discrepant_fdr5']} of "
                    f"{N['benchmarks']['testable']} testable comparisons disagree at 5% FDR",
        transform=ax.transAxes, fontsize=6.3, color=F.INK, va="bottom")
F.legend(ax, loc="lower right", fontsize=6.2)
F.grid(ax, "x")

F.note(fig, f"Of {N['benchmarks']['total']} comparisons with the published literature, "
            f"{N['benchmarks']['out_of_sample']} are out-of-sample and "
            f"{N['benchmarks']['testable']} carry a published interval from which a standard error "
            f"can be recovered, which is what makes a test possible at all. "
            f"{N['benchmarks']['discrepant_uncorrected']} disagree uncorrected and "
            f"{N['benchmarks']['discrepant_fdr5']} survive control of the false-discovery rate. "
            f"Panel b reports that {N['n_parameters_identified']} of {N['n_parameters_profiled']} "
            "calibrated parameters are identified; one failure is structural rather than "
            "empirical, because only the mixing weight of the two recovery amplitudes enters the "
            "trajectory at all.", y=0.005)
F.save(fig, "figure_18_statistics")

# ============================ Figure 2 (rebuilt) - the architecture, centred on the invention
NS = js("N_statistics_summary.json")

def rbox(ax, x, y, w, h, text, fc, ec=None, fs=6.6, weight="normal", tc=None, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.014",
                                linewidth=lw, facecolor=fc, edgecolor=ec or fc, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4,
            color=tc or F.INK, fontweight=weight, linespacing=1.3)

def arrow(ax, p1, p2, color=F.MUTED, lw=1.0, rad=0.0, ls="-", style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=8, linewidth=lw,
                                 color=color, zorder=2, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))

fig, ax = plt.subplots(figsize=(7.5, 6.4))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
fig.suptitle("Figure 2  ·  Fourteen modules, one shared state, and the device the chain is built "
             "towards", x=0.007, ha="left", fontsize=10.5, fontweight="bold")

rbox(ax, 0.03, 0.885, 0.94, 0.090,
     "SHARED STATE VECTOR\n"
     "collateral index · oxygen-extraction reserve · inflammatory set-point\n"
     "corticospinal capacity · afferent signal-to-noise · plasticity gain",
     "#e9e6f8", ec="#cdc6ef", fs=6.2, weight="bold")

rows = [
    (0.700, "#eaf2fd", "#2a78d6", "BEFORE  ·  who is at risk, and where they live",
     [("K", "social and\naccess layer"), ("A", "genomic risk\narchitecture"),
      ("F", "multi-layer\nmolecular network")]),
    (0.520, "#fdece5", "#eb6834", "DURING  ·  what the occlusion does, in time and in space",
     [("B", "perfusion and\ntissue fate"), ("C", "ictal cascade\n(14-state ODE)"),
      ("J", "spatial stochastic\ngrowth"), ("G", "haemorrhage and\nexpansion")]),
    (0.340, "#e9f7f1", "#1baf7a", "AFTER  ·  what the brain and the body do next",
     [("H", "small-vessel disease\nand cognition"), ("I", "therapeutics and\ntrial attainability"),
      ("D", "recovery law\n(two time scales)")]),
]
BH = 0.118
for y, fc, ec, header, mods in rows:
    ax.add_patch(FancyBboxPatch((0.02, y), 0.96, BH,
                                boxstyle="round,pad=0.004,rounding_size=0.010",
                                facecolor=fc, edgecolor="none", zorder=0))
    ax.text(0.028, y + BH + 0.012, header, fontsize=6.8, color=ec, fontweight="bold", va="bottom")
    w = 0.94 / len(mods)
    for i, (letter, name) in enumerate(mods):
        x = 0.035 + i * w
        rbox(ax, x, y + 0.013, w - 0.028, 0.076, name, "white", ec=ec, fs=6.3)
        ax.add_patch(FancyBboxPatch((x - 0.002, y + 0.073), 0.026, 0.024,
                                    boxstyle="round,pad=0.003,rounding_size=0.008",
                                    facecolor=ec, edgecolor=ec, zorder=5))
        ax.text(x + 0.011, y + 0.085, letter, fontsize=6.6, color="white", fontweight="bold",
                ha="center", va="center", zorder=6)
    if y > 0.35:
        arrow(ax, (0.5, y), (0.5, y - 0.032), color=ec, lw=1.3)

arrow(ax, (0.5, 0.340), (0.5, 0.300), color=F.VIOLET, lw=1.6)
rbox(ax, 0.055, 0.188, 0.89, 0.108,
     "THE ARTIFICIAL PERIPHERAL THALAMUS\n"
     "a cybernetic neuromorphic relay standing in for the damaged afferent pathway\n"
     "Module E sets its stability and information envelope; Module M solves its policy",
     F.VIOLET, ec=F.VIOLET, fs=6.8, weight="bold", tc="white")
for letter, name, x in (("E", "closed-loop interface:\nadmittance and latency", 0.075),
                        ("M", "stochastic optimal\ncontrol of the schedule", 0.645)):
    rbox(ax, x, 0.088, 0.28, 0.068, name, "white", ec=F.VIOLET, fs=6.2)
    ax.add_patch(FancyBboxPatch((x - 0.002, 0.140), 0.026, 0.024,
                                boxstyle="round,pad=0.003,rounding_size=0.008",
                                facecolor=F.VIOLET, edgecolor=F.VIOLET, zorder=5))
    ax.text(x + 0.011, 0.152, letter, fontsize=6.6, color="white", fontweight="bold",
            ha="center", va="center", zorder=6)
    arrow(ax, (x + 0.14, 0.156), (x + 0.14, 0.188), color=F.VIOLET, lw=1.1)

rbox(ax, 0.385, 0.088, 0.23, 0.068,
     "L · posteriors\nN · intervals, Sobol indices,\nmultiplicity control",
     "white", ec=F.INK2, fs=6.0)
arrow(ax, (0.50, 0.156), (0.50, 0.188), color=F.INK2, lw=1.0, ls="--")

stats = [(f"{NS['benchmarks']['total']}", "comparisons with\nthe literature", S[0]),
         (f"{NS['benchmarks']['out_of_sample']}", "of them\nout-of-sample", S[2]),
         (f"{NS['benchmarks']['discrepant_fdr5']}", "disagreements surviving\nfalse-discovery control",
          F.RED),
         (f"{NS['n_predictions']}", "pre-registered\npredictions", S[3])]
for i, (num, lab, col) in enumerate(stats):
    x = 0.035 + i * 0.245
    ax.text(x, 0.030, num, fontsize=15, fontweight="bold", color=col, va="center", ha="left")
    ax.text(x + 0.055, 0.030, lab, fontsize=6.0, color=F.INK2, va="center", linespacing=1.3)

ax.annotate("", xy=(0.992, 0.800), xytext=(0.992, 0.360),
            arrowprops=dict(arrowstyle="<|-", color=F.MUTED, lw=1.0, linestyle="--",
                            connectionstyle="arc3,rad=0.32"))
ax.text(1.028, 0.580, "recurrence and secondary prevention", fontsize=5.8, color=F.MUTED,
        rotation=90, ha="center", va="center")

F.note(fig, "Solid arrows carry the shared state forward; the dashed return path is recurrence, "
            "which puts a survivor back into the first row with an altered substrate. The device "
            "is not an application bolted on at the end: its stability envelope is set by the "
            "afferent state variable that Modules C and D determine, and its schedule is set by "
            "the recovery kinetics of Module D, so a change anywhere upstream moves it.",
       y=0.020)
F.save(fig, "figure_02_architecture")

# ==================== Figure 10 (rebuilt) - the whole validation accounting with multiplicity
bench = load("N_benchmark_multiplicity.csv")
prof = load("N_profile_likelihood.csv")
sob = load("N_sobol_indices.csv")

fig = plt.figure(figsize=(7.5, 8.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], left=0.30, right=0.975, top=0.935,
                      bottom=0.115, wspace=0.62)
fig.suptitle("Figure 10  ·  Every comparison the framework makes with the published literature",
             x=0.007, ha="left", fontsize=10.5, fontweight="bold")

b = bench.dropna(subset=["model", "published"]).copy()
b = b[b.published != 0]
b["ratio"] = b.model / b.published
b = b.sort_values("ratio")
y = np.arange(len(b))

ax = fig.add_subplot(gs[0, 0])
for i, r in enumerate(b.itertuples()):
    if r.used_in_calibration:
        c, m = F.MUTED, "s"
    elif np.isfinite(getattr(r, "discrepant_fdr5", np.nan)) and r.discrepant_fdr5 == 1:
        c, m = F.RED, "o"
    elif abs(np.log(max(r.ratio, 1e-9))) <= np.log(1.25):
        c, m = S[2], "o"
    else:
        c, m = S[3], "o"
    ax.scatter(r.ratio, i, color=c, marker=m, s=30, zorder=4)
ax.axvline(1.0, color=F.INK2, lw=1.0, zorder=3)
ax.axvspan(0.75, 1.25, color=S[2], alpha=0.12, zorder=0)
ax.set_xscale("log")
ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1, 2, 4, 10])
ax.set_xticklabels(["0.05x", "0.1x", "0.25x", "0.5x", "1x", "2x", "4x", "10x"], fontsize=6.6)
ax.set_yticks(y)
ax.set_yticklabels(F.trim(b.module + " · " + b.quantity, 52), fontsize=4.9)
ax.set_ylim(-1, len(b))
ax.set_xlabel("model / published")
ax.set_title("a   every point comparison, by module", loc="left", fontsize=9.0,
             fontweight="bold")
from matplotlib.lines import Line2D
F.legend(ax, loc="lower left", fontsize=6.0, handles=[
    Line2D([], [], marker="s", ls="", color=F.MUTED, label="calibration target"),
    Line2D([], [], marker="o", ls="", color=S[2], label="out-of-sample, within 25%"),
    Line2D([], [], marker="o", ls="", color=S[3], label="out-of-sample, outside 25%"),
    Line2D([], [], marker="o", ls="", color=F.RED, label="disagrees at 5% FDR")])

axr = fig.add_subplot(gs[0, 1])
axr.set_title("b   what each conclusion depends on", loc="left", fontsize=9.0, fontweight="bold")
piv = sob.pivot_table(index="parameter", columns="output", values="total_order")
piv = piv.reindex(piv.mean(axis=1).sort_values().index)
im = axr.imshow(piv.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
axr.set_xticks(np.arange(piv.shape[1]))
axr.set_xticklabels(F.trim(piv.columns, 22), rotation=35, ha="right", fontsize=5.8)
axr.set_yticks(np.arange(piv.shape[0]))
axr.set_yticklabels(F.trim(piv.index, 26), fontsize=5.8)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        if np.isfinite(v) and v >= 0.10:
            axr.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.4,
                     color="white" if v > 0.55 else F.INK)
cb = fig.colorbar(im, ax=axr, fraction=0.035, pad=0.03)
cb.set_label("total-order Sobol index", fontsize=6.0)
cb.ax.tick_params(labelsize=5.8)

F.note(fig, f"{NS['benchmarks']['total']} comparisons in all, of which "
            f"{NS['benchmarks']['out_of_sample']} are out-of-sample. A published interval, and "
            f"therefore a testable standard error, exists for {NS['benchmarks']['testable']} of "
            f"them; {NS['benchmarks']['discrepant_uncorrected']} disagree uncorrected and "
            f"{NS['benchmarks']['discrepant_fdr5']} survive Benjamini-Hochberg control across the "
            "whole set. Panel b replaces the earlier rank-regression sensitivity pass with "
            "variance-based total-order indices, which is what makes the interaction visible: a "
            "parameter whose total-order index exceeds its first-order index matters through its "
            "interactions, and rank regression cannot see that at all.", y=0.012)
F.save(fig, "figure_10_validation")

# ================================================================ the graphical abstract
JS = js("J_spatial_summary.json")
IS = js("I_therapeutics_summary.json")
KSU = js("K_social_summary.json")
MS = js("M_control_summary.json")
HS = js("H_svd_cognition_summary.json")

fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
fig.suptitle("A Cybernetic Neuromorphic Framework for the Stroke Continuum", x=0.007, ha="left",
             fontsize=12.5, fontweight="bold")
ax.text(0.0, 0.955, "an artificial peripheral thalamus, and the model of the disease that decides "
                    "what it has to do", fontsize=8.0, color=F.INK2, va="top")

stages = [("who is at risk", "#eaf2fd", "#2a78d6",
           f"{KSU['comparison']['ratio_modifiable_to_genomic']:.0f}x",
           "modifiable burden against\nthe polygenic score"),
          ("what the occlusion does", "#fdece5", "#eb6834",
           f"{JS['median_effective_front_speed']:.1f}",
           "mm/min depolarisation front,\nsolved on a cortical sheet"),
          ("why the drugs failed", "#fdf6e6", "#eda100",
           f"{IS['n_trials_underpowered_for_attainable']}/{IS['n_trials_assessed']}",
           "trials powered for an effect\nlarger than was attainable"),
          ("what the brain does next", "#e9f7f1", "#1baf7a",
           f"{HS.get('language_tests_failed', 4)}/4",
           "language benchmarks the motor\nrecovery law fails to reproduce"),
          ("what the device must do", "#ece9f8", F.VIOLET,
           f"{list(MS['optimal_timing_advantage_fm'].values())[0]:.1f}",
           "Fugl-Meyer points from timing\nthe same hours correctly")]
w = 0.186
for i, (title, fc, ec, hero, sub) in enumerate(stages):
    x = 0.005 + i * (w + 0.010)
    ax.add_patch(FancyBboxPatch((x, 0.235), w, 0.63,
                                boxstyle="round,pad=0.006,rounding_size=0.02",
                                facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=2))
    import textwrap as _tw
    ax.text(x + w / 2, 0.800, "\n".join(_tw.wrap(title, 18)), ha="center", va="center",
            fontsize=6.6, fontweight="bold", color=ec, zorder=4, linespacing=1.3)
    ax.text(x + w / 2, 0.615, hero, ha="center", va="center", fontsize=20, fontweight="bold",
            color=ec, zorder=4)
    ax.text(x + w / 2, 0.385, "\n".join(_tw.wrap(sub.replace("\n", " "), 25)), ha="center",
            va="center", fontsize=5.7, color=F.INK2, linespacing=1.5, zorder=4)
    if i < len(stages) - 1:
        ax.add_patch(FancyArrowPatch((x + w + 0.001, 0.55), (x + w + 0.009, 0.55),
                                     arrowstyle="-|>", mutation_scale=7, color=F.MUTED, lw=1.0))

ax.add_patch(FancyBboxPatch((0.005, 0.075), 0.99, 0.125,
                            boxstyle="round,pad=0.006,rounding_size=0.02",
                            facecolor="#f3f2ee", edgecolor="none", zorder=1))
ax.text(0.5, 0.138,
        f"{NS['benchmarks']['total']} comparisons with the published literature  ·  "
        f"{NS['benchmarks']['out_of_sample']} out-of-sample  ·  "
        f"{NS['benchmarks']['discrepant_fdr5']} disagreements surviving false-discovery control\n"
        f"{NS['n_predictions']} pre-registered predictions  ·  "
        "every input traceable to a public accession  ·  the whole pipeline ships as code",
        ha="center", va="center", fontsize=6.4, color=F.INK, zorder=4, linespacing=1.6)
F.note(fig, "No patient-level or institutional data are used anywhere in this work: every "
            "biological, epidemiological and clinical input is a published public value carried "
            "with its source identifier, and anything that could not be verified at source is "
            "written as UNVERIFIED and excluded from computation by the loader itself.",
       y=0.045, width=118)
F.save(fig, "figure_00_graphical_abstract")

# ==================================== Figure 19 - delivery physics and the carrier specification
O = js("O_delivery_summary.json")
odiff = load("O_interstitial_diffusion.csv")
odel = load("O_delivery_deficit.csv")
ospec = load("O_carrier_specification.csv")
odes = load("O_design_curve.csv")
obio = load("O_biologic_addressable_targets.csv")

fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.8), layout='constrained')
fig.suptitle("Figure 19  ·  Delivery physics: what a carrier would actually have to do",
             x=0.007, ha="left", fontsize=10.5, fontweight="bold")

ax = axs[0, 0]
F.panel(ax, "a", "penetration falls as the square root of size")
for tissue, col, ls in (("normally perfused", S[0], "-"), ("ischaemic", S[1], "--")):
    d = odiff[(odiff.tissue == tissue) & (odiff.cbf == odiff.cbf.max())]
    ax.plot(d.radius_nm, d.penetration_um / 1000.0, ls, color=col, marker="o", ms=3.5,
            label=tissue)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("carrier hydrodynamic radius (nm)")
ax.set_ylabel("penetration in the tolerated time (mm)")
ax.axvline(0.17, color=F.MUTED, lw=0.8, ls=":")
ax.annotate("hydrated K$^+$, recovered from\nthe verified free diffusivity", xy=(0.17, 4.5),
            xytext=(1.6, 6.0), fontsize=5.8, color=F.MUTED,
            arrowprops=dict(arrowstyle="-", color=F.MUTED, lw=0.7))
F.legend(ax, loc="lower left", fontsize=6.3); F.grid(ax)

ax = axs[0, 1]
F.panel(ax, "b", "the delivery deficit compounds the time deficit")
mid = odel[odel.delivery_site == "mid-penumbra"].sort_values("attainable_or")
y = np.arange(len(mid))
ax.hlines(y, mid.attainable_or_flow_limited, mid.attainable_or, color=F.GRID, lw=2.4, zorder=1)
ax.scatter(mid.attainable_or, y, color=S[2], s=26, zorder=4, label="attainable, time only")
ax.scatter(mid.attainable_or_flow_limited, y, color=S[1], s=26, zorder=4,
           label="attainable, flow-limited delivery")
ax.scatter(mid.detectable_or, y, color=F.RED, s=26, zorder=4, label="powered to detect")
ax.set_yticks(y); ax.set_yticklabels(F.trim(mid.trial, 24), fontsize=5.6)
ax.set_xlabel("odds ratio"); ax.set_xscale("log")
ax.set_xticks([1.0, 1.2, 1.5, 2.0, 3.0]); ax.set_xticklabels(["1.0", "1.2", "1.5", "2.0", "3.0"])
F.legend(ax, loc="lower right", fontsize=5.9); F.grid(ax, "x")

ax = axs[1, 0]
F.panel(ax, "c", "the specification, and the design curve")
sp = ospec.sort_values("required_enrichment")
ax.barh(np.arange(len(sp)), sp.required_enrichment, color=S[0], height=0.62)
ax.axvline(O["required_enrichment"]["median"], color=F.RED, ls="--", lw=1.2)
ax.text(O["required_enrichment"]["median"] * 1.05, len(sp) - 0.5,
        f"median {O['required_enrichment']['median']:.0f}x", fontsize=6.2, color=F.RED, va="top")
ax.set_yticks(np.arange(len(sp))); ax.set_yticklabels(F.trim(sp.trial, 24), fontsize=5.6)
ax.set_xlabel("selective enrichment in penumbral tissue required")
F.grid(ax, "x")
axin = ax.inset_axes([0.52, 0.20, 0.44, 0.40])
d1 = odes[odes.trial_n == 1000]
axin.plot(d1.enrichment, d1.latest_useful_window_h.fillna(0), "o-", color=S[1], ms=3)
axin.set_xscale("log")
axin.set_xlabel("enrichment", fontsize=5.6); axin.set_ylabel("useful window (h)", fontsize=5.6)
axin.tick_params(labelsize=5.2)
axin.set_title("what it buys, n=1000", fontsize=5.8, loc="left", x=0.0)

ax = axs[1, 1]
F.panel(ax, "d", "which targets a biologic can reach")
b = obio.dropna(subset=["node_peak_hours"]).sort_values("node_peak_hours")
col = [S[1] if sm else S[2] for sm in b.small_molecule_only]
ax.barh(np.arange(len(b)), b.node_peak_hours, color=col, height=0.62)
ax.set_xscale("log")
ax.set_yticks(np.arange(len(b)))
ax.set_yticklabels([f"{g} · {n}" for g, n in zip(b.target_gene, F.trim(b.cascade_node, 18))],
                   fontsize=5.6)
ax.axvline(6.0, color=F.INK2, ls="--", lw=1.0)
ax.text(6.6, 0.2, "earliest plausible onset\nfor an expression-based therapy", fontsize=5.8,
        color=F.INK2)
ax.set_xlabel("hours from onset at which the target's cascade node peaks")
F.grid(ax, "x")

F.note(fig, "Panel a validates the treatment before using it: the Stokes-Einstein relation applied "
            f"to the verified free diffusion coefficient returns a hydrodynamic radius of "
            f"{O['stokes_einstein_radius_pm']:.0f} pm, the hydrated potassium ion, which was not "
            "fitted. Panel b is the compounding penalty - the tissue with most to gain from a "
            f"neuroprotectant receives {100*O['delivery_fraction']['mid-penumbra']:.0f}% of the "
            "delivery normally perfused cortex receives - and panel c the specification that "
            "follows: a selective enrichment of single-digit magnitude, which is within reach of a "
            "targeted carrier and entirely out of reach of a better molecule.", y=0.005)
F.save(fig, "figure_19_delivery_physics")
