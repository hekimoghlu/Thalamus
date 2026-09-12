#!/usr/bin/env python3
"""Figures 11-14: haemorrhage, small-vessel disease and cognition, therapeutics, spatial growth."""
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import kcore as K
import kfig as F

S = F.SERIES
R = K.RESULTS

def load(n):
    return pd.read_csv(R / n)

def js(n):
    return json.load(open(R / n))

def ratio_axis(ax):
    ax.axvline(1.0, color=F.INK2, lw=1.0, zorder=3)
    ax.axvspan(0.75, 1.25, color=S[2], alpha=0.13, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4, 10])
    ax.set_xticklabels(["0.1x", "0.25x", "0.5x", "1x", "2x", "4x", "10x"])

# ===================================================== Figure 11 - intracerebral haemorrhage
G = js("G_haemorrhage_summary.json")
gv = load("G_validation.csv")
bp = load("G_bp_lowering_scenarios.csv")
vols = np.load(R / "G_volumes.npy")          # row 0 baseline, row 1 after expansion
fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.6), layout='constrained')
fig.suptitle("Figure 11  ·  Haemorrhagic stroke: expansion, volume and what blood-pressure "
             "lowering can reach", x=0.007, ha="left", fontsize=10.5, fontweight="bold")

ax = axs[0, 0]
F.panel(ax, "a", "baseline haematoma volume")
bins = np.logspace(np.log10(0.5), np.log10(300), 46)
ax.hist(vols[0], bins=bins, color=S[0], alpha=0.85, edgecolor="none", label="at presentation")
ax.hist(vols[1], bins=bins, histtype="step", color=S[1], lw=1.4, label="after expansion")
ax.set_xscale("log")
ax.set_xticks([1, 3, 10, 30, 100, 300]); ax.set_xticklabels(["1", "3", "10", "30", "100", "300"])
for q in (4, 10, 30):
    ax.axvline(q, color=F.INK2, ls="--", lw=0.9, zorder=1)
ax.annotate("verified median 10 mL (IQR 4-30)", xy=(10, ax.get_ylim()[1] * 0.62),
            xytext=(34, ax.get_ylim()[1] * 0.62), fontsize=6.4, color=F.INK2, va="center",
            arrowprops=dict(arrowstyle="-", color=F.INK2, lw=0.8))
F.legend(ax, loc="upper right")
ax.set_xlabel("haematoma volume (mL, log scale)"); ax.set_ylabel("simulated patients")
F.grid(ax)

ax = axs[0, 1]
F.panel(ax, "b", "expansion and outcome benchmarks")
lab = F.trim(gv.quantity, 40)
y = np.arange(len(gv))
col = [S[2] if p else F.RED for p in gv.passes]
for i, r in enumerate(gv.itertuples()):
    ax.scatter(r.model / r.published, i, c=col[i], s=46, zorder=4,
               marker="s" if r.used_in_calibration else "o")
ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=6.4); ax.invert_yaxis()
ratio_axis(ax)
ax.set_xlabel("model / published")
ax.text(0.02, 1.02, "squares: calibration   circles: out-of-sample   green: within range",
        transform=ax.transAxes, fontsize=6.2, color=F.MUTED, va="bottom")

ax = axs[1, 0]
F.panel(ax, "c", "blood-pressure lowering scenarios")
ax.barh(np.arange(len(bp)), bp.model_mortality_pct, color=S[0], height=0.55)
ax.set_yticks(np.arange(len(bp))); ax.set_yticklabels(F.trim(bp.scenario, 26), fontsize=6.6)
ax.invert_yaxis(); ax.set_xlabel("modelled case fatality (%)")
for i, (m, o) in enumerate(zip(bp.model_mortality_pct, bp.model_odds_ratio)):
    ax.text(m + 0.4, i, f"OR {o:.3f}", va="center", fontsize=6.4, color=F.INK2)
ax.set_xlim(0, max(bp.model_mortality_pct) * 1.25)
F.grid(ax, "x")

ax = axs[1, 1]
F.panel(ax, "d", "how much growth mediates")
frac = 100.0 * G["growth_mediated_fraction"]
ax.barh([0, 1], [frac, 100 - frac], color=[S[1], F.GRID], height=0.5)
ax.set_yticks([0, 1])
ax.set_yticklabels(["mediated by\nhaematoma growth", "not mediated by\ngrowth"], fontsize=6.8)
ax.invert_yaxis(); ax.set_xlabel("share of the intensive-lowering effect (%)")
ax.text(frac + 2, 0, f"{frac:.0f}%", va="center", fontsize=8, fontweight="bold", color=F.INK)
F.grid(ax, "x")

F.note(fig, "The expansion model is built from verified odds ratios for time from onset, baseline "
            "volume, anticoagulation and the spot sign, with the intercept solved to reproduce the "
            "verified 20% overall expansion rate; the volume-to-mortality map is fitted to two "
            "verified anchors. Panel d compares the odds ratio the model produces when intensive "
            "lowering acts only through growth against the verified trial odds ratio.", y=0.005)
F.save(fig, "figure_11_haemorrhage")

# ======================================= Figure 12 - small-vessel disease, cognition, language
H = js("H_svd_cognition_summary.json")
dem = load("H_dementia_incidence.csv")
psci = load("H_psci_threshold_curve.csv")
lang = load("H_language_recovery.csv")
ldose = load("H_language_dose_response.csv")
traj = np.load(R / "H_cognitive_trajectories.npy")
ty = np.load(R / "H_time_years.npy")

fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.6), layout='constrained')
fig.suptitle("Figure 12  ·  Small-vessel disease, post-stroke cognition, and a law that does not "
             "transfer to language", x=0.007, ha="left", fontsize=10.5, fontweight="bold")

ax = axs[0, 0]
F.panel(ax, "a", "cognitive trajectory after stroke")
band = np.percentile(traj, [10, 50, 90], axis=0)
ax.fill_between(ty, band[0], band[2], color=S[0], alpha=0.18, lw=0)
ax.plot(ty, band[1], color=S[0], label="model median")
ax.axhline(-0.5, color=F.RED, ls="--", lw=1.2, zorder=1)
ax.text(ty[-1], -0.42, "impairment cut-point", ha="right", va="bottom", fontsize=6.4,
        color=F.RED)
ax.set_xlabel("years after stroke"); ax.set_ylabel("global cognition (SD units)")
F.legend(ax, loc="upper right"); F.grid(ax)

ax = axs[0, 1]
F.panel(ax, "b", "cumulative dementia incidence")
w = 0.38
ax.bar(dem.year - w / 2, dem.model_pct, w, color=S[0], label="model")
ax.bar(dem.year + w / 2, dem.published_pct, w, color=S[1], label="published")
ax.set_xticks(dem.year.values)
ax.set_xlabel("years after stroke"); ax.set_ylabel("cumulative incidence (%)")
F.legend(ax, loc="upper left"); F.grid(ax)

ax = axs[1, 0]
F.panel(ax, "c", "how the impairment threshold sets prevalence")
ax.plot(psci.threshold_sd, psci.model_pct, color=S[0], label="model")
ax.axhline(psci.published_3mo_pct.iloc[0], color=S[1], ls="--",
           label=f"published 3-month prevalence {psci.published_3mo_pct.iloc[0]:.0f}%")
ax.set_xlabel("cut-point (SD below the norm)"); ax.set_ylabel("prevalence of impairment (%)")
F.legend(ax, loc="upper left"); F.grid(ax)

ax = axs[1, 1]
F.panel(ax, "d", "the motor recovery law applied to aphasia")
y = np.arange(len(lang))
ax.scatter(lang.model / lang.published, y, c=[S[2] if p else F.RED for p in lang.passes], s=48,
           zorder=4)
ax.set_yticks(y); ax.set_yticklabels(F.trim(lang.quantity, 34), fontsize=6.5); ax.invert_yaxis()
ratio_axis(ax)
ax.set_xlabel("model / published")
ax.text(0.02, 1.02, "every comparison fails: language does not follow the motor law",
        transform=ax.transAxes, fontsize=6.4, color=F.RED, fontweight="bold", va="bottom")

F.note(fig, "The recovery law of Module D is transferred to aphasia WITHOUT refitting, which is "
            "the point of panel d: a law that describes motor recovery does not describe language "
            "recovery, and the size of the failure is the measurement. Cognitive trajectories use "
            "the verified step-and-slope decomposition; dementia incidence and the impairment "
            "prevalence are then predicted rather than fitted.", y=0.005)
F.save(fig, "figure_12_svd_cognition")

# ============================================ Figure 13 - the therapeutic layer (Module I)
ther = js("I_therapeutics_summary.json")
lad = load("I_reperfusion_time_ladder.csv")
att = load("I_trial_attainability.csv")
trg = load("I_target_tractability.csv")
gap = load("I_implied_salvage_gap.csv")

fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.8), layout='constrained')
fig.suptitle("Figure 13  ·  Why the neuroprotection programme could not have worked",
             x=0.007, ha="left", fontsize=10.5, fontweight="bold")

ax = axs[0, 0]
F.panel(ax, "a", "time-benefit ladder")
for key, col, mk in (("IVT", S[0], "o"), ("EVT", S[1], "s")):
    d = lad[lad.intervention.str.startswith(key)]
    ax.errorbar(d.hours, d.published_or, yerr=[d.published_or - d.pub_lo, d.pub_hi - d.published_or],
                fmt=mk, color=col, ms=5, capsize=3, lw=1.2, label=f"{key} published")
    ax.plot(d.hours, d.model_or, color=col, ls="--", lw=1.6, marker="x",
            label=f"{key} model")
ax.axhline(1.0, color=F.MUTED, lw=0.9)
ax.set_xlabel("hours from onset to treatment"); ax.set_ylabel("odds ratio, good outcome")
F.legend(ax, loc="upper right", fontsize=6.2, ncol=2)
F.grid(ax)

ax = axs[0, 1]
F.panel(ax, "b", "the salvage curve decays too slowly")
for key, ls, mk in (("IVT", "-", "o"), ("EVT", "--", "s")):
    d = gap[gap.intervention.str.startswith(key)].sort_values("hours")
    ax.plot(d.hours, d.model_salvage, ls, marker=mk, color=S[0], label=f"{key}: model salvage")
    ax.plot(d.hours, d.implied_salvage, ls, marker=mk, color=S[1],
            label=f"{key}: implied by the published ladder")
    for r in d.itertuples():
        if np.isfinite(r.implied_salvage) and not r.used_in_calibration:
            ax.annotate("", xy=(r.hours, r.implied_salvage), xytext=(r.hours, r.model_salvage),
                        arrowprops=dict(arrowstyle="-|>", color=F.MUTED, lw=0.9))
ax.set_xlabel("hours from onset"); ax.set_ylabel("salvageable fraction")
ax.set_ylim(0, 1.05)
F.legend(ax, loc="lower left", fontsize=5.9, ncol=2)
F.grid(ax)

ax = axs[1, 0]
F.panel(ax, "c", "attainable against detectable effect")
a = att.sort_values("attainable_or")
y = np.arange(len(a))
ax.hlines(y, a.attainable_or, a.detectable_or, color=F.GRID, lw=2.4, zorder=1)
ax.scatter(a.attainable_or, y, color=S[2], s=34, zorder=4, label="largest attainable")
ax.scatter(a.detectable_or, y, color=F.RED, s=34, zorder=4, label="powered to detect")
ax.set_yticks(y); ax.set_yticklabels(F.trim(a.trial, 26), fontsize=6.0)
ax.set_xlabel("odds ratio")
F.legend(ax, loc="lower right", fontsize=6.3)
F.grid(ax, "x")

ax = axs[1, 1]
F.panel(ax, "d", "target tractability")
t = trg.sort_values("tractability")
ax.barh(np.arange(len(t)), t.tractability, color=[S[2] if v > 0.5 else S[0] for v in t.tractability],
        height=0.6)
ax.set_yticks(np.arange(len(t))); ax.set_yticklabels(t.target_gene, fontsize=6.6)
ax.set_xlabel("tractability score")
for i, (v, n) in enumerate(zip(t.tractability, t.cascade_node)):
    ax.text(v + 0.008, i, n, va="center", fontsize=5.8, color=F.MUTED)
ax.set_xlim(0, max(t.tractability) * 1.75)
F.grid(ax, "x")

F.note(fig, f"Panel c is the central claim of this layer: {ther['n_trials_underpowered_for_attainable']} "
            f"of {ther['n_trials_assessed']} trials were powered to detect an effect larger than the "
            "largest effect attainable at their own treatment window, given the injury still "
            "preventable at that time and the share of the cascade their target can influence. The "
            "conclusion is unchanged on the steeper salvage curve of panel b, which is the direction "
            "the published ladder demands.", y=0.005)
F.save(fig, "figure_13_therapeutics")

# ================================= Figure 14 - spatial stochastic growth (Module J)
JS = js("J_spatial_summary.json")
JC = load("J_cohort.csv")
JV = load("J_validation.csv")
SDC = load("J_sd_contribution.csv").set_index("quantity")["value"]
ex = np.load(R / "J_exemplar_fields.npz")
trj = np.load(R / "J_trajectories.npy")
th = np.load(R / "J_time_hours.npy")

flow_cm = LinearSegmentedColormap.from_list("flow", ["#241436", "#2a78d6", "#9ec9f5", "#f6f6f1"])
infarct_cm = ListedColormap([(0, 0, 0, 0), (0.89, 0.24, 0.24, 0.34)])

fig = plt.figure(figsize=(7.5, 8.2))
# two gridspecs: the map panels use the full width, the benchmark ladder needs a wide left
# margin for its category labels, and sharing one grid made the maps float in whitespace
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.40, wspace=0.24,
                      left=0.055, right=0.90, top=0.915, bottom=0.455)
gs2 = fig.add_gridspec(1, 1, left=0.29, right=0.965, top=0.375, bottom=0.115)
fig.suptitle("Figure 14  ·  Infarct growth solved in space: collateral geometry and "
             "depolarisation waves", x=0.007, ha="left", fontsize=10.5, fontweight="bold")

cell_ml = float(JC.cell_ml.iloc[0])
for j, (lab, ttl) in enumerate((("poor", "poor collaterals"), ("intermediate", "intermediate"),
                                ("good", "good collaterals"))):
    ax = fig.add_subplot(gs[0, j])
    im = ax.imshow(ex[f"{lab}_F0"], cmap=flow_cm, vmin=0, vmax=50, origin="lower")
    dead = ex[f"{lab}_dead_24h"]
    ax.imshow(dead.astype(float), cmap=infarct_cm, vmin=0, vmax=1, origin="lower")
    ax.contour(dead.astype(float), levels=[0.5], colors=[F.RED], linewidths=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(ttl, fontsize=7.6)
    ax.text(0.03, 0.03, f"{dead.sum() * cell_ml:.0f} mL infarcted at 24 h", transform=ax.transAxes,
            fontsize=6.2, color="white", fontweight="bold",
            bbox=dict(facecolor="#00000055", edgecolor="none", pad=1.6))
    if j == 0:
        F.panel(ax, "a", None, dx=-0.06, dy=1.10)
cax = fig.add_axes([0.915, 0.70, 0.012, 0.19])
cb = fig.colorbar(im, cax=cax)
cb.set_label("cerebral blood flow (mL/100 g/min)", fontsize=6.2)
cb.ax.tick_params(labelsize=6.0)

ax = fig.add_subplot(gs[1, 0])
F.panel(ax, "b", None, dx=-0.06, dy=1.10)
imk = ax.imshow(np.log10(np.clip(ex["intermediate_K_6h"], 3.1, 65)), cmap="magma",
                vmin=np.log10(3.1), vmax=np.log10(65), origin="lower")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("K$^+$ elevation is confined to the\ndepolarised patch (6 h)", fontsize=6.8)
cax2 = fig.add_axes([0.915, 0.475, 0.012, 0.15])
cb2 = fig.colorbar(imk, cax=cax2)
cb2.set_ticks(np.log10([3.1, 6, 12, 25, 65]))
cb2.set_ticklabels(["3.1", "6", "12", "25", "65"])
cb2.set_label("extracellular K$^+$ (mM, log)", fontsize=6.0)
cb2.ax.tick_params(labelsize=5.8)

ax = fig.add_subplot(gs[1, 1])
F.panel(ax, "c", None, dx=-0.06, dy=1.10)
ph = ex["intermediate_phase_6h"]
ax.imshow(ph, cmap=ListedColormap(["#f0efe9", F.RED, "#bdbab0"]), vmin=0, vmax=2, origin="lower")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("depolarised (red), refractory (grey)", fontsize=7.0)

ax = fig.add_subplot(gs[1, 2])
F.panel(ax, "d", "volume trajectories", dx=-0.20, dy=1.08)
for row in trj[:26]:
    ax.plot(th, row, color=S[0], alpha=0.22, lw=0.9)
ax.plot(th, np.median(trj, axis=0), color=F.RED, lw=2.4, label="median")
ax.set_xlabel("hours from occlusion"); ax.set_ylabel("infarct volume (mL)")
F.legend(ax, loc="lower right", fontsize=6.4); F.grid(ax)
adv_min = 60 * float(SDC["time to half the final infarct, waves off (h)"]
                     - SDC["time to half the final infarct, waves on (h)"])
rec_pct = float(SDC["volume recruited by waves (% of final)"])
ax.text(0.03, 0.97, f"waves recruit {rec_pct:.2f}% of the final volume\n"
                    f"and advance the half-way point by {adv_min:.0f} min",
        transform=ax.transAxes, va="top", fontsize=6.2, color=F.INK2,
        bbox=dict(facecolor=F.SURFACE, edgecolor="none", alpha=0.9, pad=2.0))

ax = fig.add_subplot(gs2[0, 0])
F.panel(ax, "e", "every spatial benchmark, model against published", dx=-0.055, dy=1.04)
v = JV.dropna(subset=["model", "published"]).copy()
v["ratio"] = v.model / v.published
v = v.sort_values("ratio")
yy = np.arange(len(v))
for i, r in enumerate(v.itertuples()):
    ax.scatter(r.ratio, i, s=46, zorder=4,
               color=S[2] if r.within_published_range else F.RED,
               marker="s" if r.used_in_calibration else "o")
ax.set_yticks(yy); ax.set_yticklabels(F.trim(v.quantity, 52), fontsize=6.2)
ax.set_ylim(-0.8, len(v) - 0.2)
ratio_axis(ax)
ax.set_xlabel("model / published")
ax.text(0.99, 0.03, "squares: calibration   circles: out-of-sample   green: within range",
        transform=ax.transAxes, fontsize=6.2, color=F.MUTED, ha="right")

F.note(fig, "The domain is Module B's calibrated at-risk territory and the flow field's marginal is "
            "Module B's lognormal, rearranged so the lowest flows sit at the centre of the "
            "territory and the highest where leptomeningeal collaterals enter at its border. "
            "Depolarisation is integrated in the sharp-interface limit, because the "
            "reaction-diffusion front the verified constants imply is 36 um wide - a twenty-sixth "
            "of one grid cell - and no cell-averaged potassium gradient can cross the verified "
            "9.4 mM threshold. The two failures at the left of panel e are the core depolarisation "
            "duration, where a pump law calibrated in perfused tissue does not extend to the core, "
            "and the penumbral share of the final infarct, where the tolerance curve is too sharp.",
       y=0.005)
F.save(fig, "figure_14_spatial_growth")
