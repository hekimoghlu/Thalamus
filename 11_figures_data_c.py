#!/usr/bin/env python3
"""Figures 7-10: recovery law, closed-loop design envelope, molecular network, validation."""
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import kcore as K
import kfig as F

R = K.RESULTS; S = F.SERIES; G = K.getp

# ==================================================================== Figure 7: recovery dynamics
traj = np.load(R / "D_trajectories.npy"); tw = np.load(R / "D_time_weeks.npy")
fm0 = np.load(R / "D_fm0.npy"); cap = np.load(R / "D_cap.npy")
prr = pd.read_csv(R / "D_proportional_recovery_by_cohort.csv")
pv = pd.read_csv(R / "D_prr_validation.csv")
scan = pd.read_csv(R / "D_prr_identifiability_scan.csv")
selq = pd.read_csv(R / "D_fitter_selection_effect.csv")
tc = pd.read_csv(R / "D_time_course_validation.csv")
ds = json.loads((R / "D_recovery_summary.json").read_text())
DP = ds["recovery_parameters"]

fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.7))
ax = axs[0, 0]
for i in range(0, 150, 2):
    ax.plot(tw, traj[i], color=S[0], alpha=0.10, lw=0.7)
ax.plot(tw, traj.mean(0), color=F.RED, lw=2.2, label="cohort mean")
gain = np.diff(traj.mean(0)) / np.diff(tw)
ax.set_xlim(0, 52); ax.set_xlabel("weeks after stroke"); ax.set_ylabel("Fugl-Meyer upper extremity")
for wk, lab in ((3.0, f"fast component\nτ = {DP['tau_fast']:.0f} weeks"),
                (30.0, f"slow window\nτ = {DP['tau_slow']:.0f} weeks")):
    ax.axvline(wk, color=F.MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.annotate(lab, (wk + 1.2, 8), fontsize=6.4, color=F.INK2)
F.legend(ax, loc="lower right")
F.grid(ax); F.panel(ax, "a", "two-timescale recovery trajectories")

ax = axs[0, 1]
x = 66.0 - fm0
y = np.asarray([np.interp(26.0, tw, traj[i]) for i in range(len(traj))]) - fm0[:len(traj)]
xs = x[:len(traj)]
ax.scatter(xs, y, s=7, color=S[0], alpha=0.55, zorder=3)
b = np.polyfit(xs, y, 1)
xx = np.linspace(0, 66, 50)
ax.plot(xx, np.polyval(b, xx), color=F.RED, lw=1.8, zorder=4,
        label=f"model fit: slope {b[0]:.2f}")
ax.plot(xx, 0.7 * xx, color=F.INK2, lw=1.4, ls=(0, (5, 3)), zorder=4,
        label="published 0.70 rule")
ax.set_xlabel("initial impairment (66 - FM$_0$)"); ax.set_ylabel("observed change at 26 weeks")
F.legend(ax, loc="upper left")
F.grid(ax, "both"); F.panel(ax, "b", "proportional recovery emerges, but shallower")

ax = axs[1, 0]
for i, (lab, col) in enumerate((("slope", S[0]), ("r2", S[1]), ("variance_ratio_k", S[2]))):
    ax.scatter(scan.cap_sd, scan[lab], s=14, color=col, alpha=0.8, zorder=3, label=lab)
targets = {"slope": (G("prr_slope_range_lo"), G("prr_slope_range_hi")),
           "r2": (min(G("prr_r2_dataset3"), G("prr_r2_dataset1")), G("prr_r2_dataset2")),
           "variance_ratio_k": (G("variance_ratio_k_winters"), G("variance_ratio_k_sb"))}
for i, (lab, (lo, hi)) in enumerate(targets.items()):
    ax.fill_between([0.02, 0.9], lo, hi, color=[S[0], S[1], S[2]][i], alpha=0.10, zorder=1)
ax.set_xscale("log"); ax.set_xticks([0.05, 0.1, 0.2, 0.45, 0.8])
ax.set_xticklabels(["0.05", "0.1", "0.2", "0.45", "0.8"])
ax.set_xlabel("dispersion of surviving corticospinal capacity")
ax.set_ylabel("statistic")
F.legend(ax, loc="upper right", fontsize=6.6)
F.grid(ax); F.panel(ax, "c", "what the published statistics constrain")

ax = axs[1, 1]
lab = ["all\npatients", "residual-defined\nfitters", "initial FM\n10-45"]
ax.bar(range(3), selq.slope, color=[S[0], S[1], S[3]], width=0.6, zorder=3)
for i, r in enumerate(selq.itertuples()):
    ax.annotate(f"{r.slope:.2f}", (i, r.slope), ha="center", va="bottom", fontsize=6.8,
                color=F.INK2, xytext=(0, 2), textcoords="offset points")
ax.axhspan(G("prr_slope_range_lo"), G("prr_slope_range_hi"), color=F.MUTED, alpha=0.15, zorder=1)
ax.annotate("published slope range", (2.42, 0.7), fontsize=6.4, color=F.INK2, ha="right")
ax.set_xticks(range(3)); ax.set_xticklabels(lab, fontsize=6.8)
ax.set_ylabel("fitted recovery proportion"); ax.set_ylim(0, 0.95)
F.grid(ax); F.panel(ax, "d", "fitter selection inflates the slope")
fig.suptitle("Figure 7  ·  Layer 4: recovery dynamics and the proportional-recovery law",
             fontsize=10.5, x=0.005, ha="left", y=1.005)
F.note(fig, f"Only two constants are calibrated, both against verified TIME-COURSE data (the two-weekly Fugl-Meyer "
            f"gains and the per-week treatment gradients); nothing from the proportional-recovery literature enters "
            f"the fit. The model then reproduces {int(pv.model_overlaps_published_range.sum())} of "
            f"{len(pv)} published proportional-recovery statistics, but under-predicts the regression slope "
            f"({selq.slope.iloc[0]:.2f} against a published {G('prr_slope_range_lo')}-{G('prr_slope_range_hi')}). "
            f"Panel d quantifies one published explanation for that gap: restricting to fitters raises the model's own "
            f"slope from {selq.slope.iloc[0]:.2f} to {selq.slope.iloc[1]:.2f} without any change in biology.", y=0.0)
fig.tight_layout()
F.save(fig, "figure_07_recovery")

# ============================================================== Figure 8: closed-loop design envelope
stab = pd.read_csv(R / "E_stability_envelope.csv")
chan = pd.read_csv(R / "E_channel_rate.csv")
env = pd.read_csv(R / "E_design_envelope.csv")
need = pd.read_csv(R / "E_dose_for_mcid.csv")
pred = pd.read_csv(R / "E_predicted_effect_sizes.csv")
comp = pd.read_csv(R / "E_published_comparators.csv")
es = json.loads((R / "E_control_summary.json").read_text())

fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.8))
ax = axs[0, 0]
ok = stab.max_stable_stiffness_N_per_m > 0
ax.plot(stab.latency_ms[ok], stab.max_stable_stiffness_N_per_m[ok], color=S[0], marker="o",
        markersize=3.6, zorder=3)
ax.fill_between(stab.latency_ms[ok], 0.5, stab.max_stable_stiffness_N_per_m[ok], color=S[0],
                alpha=0.12, zorder=2)
for v, lab in ((G("haptic_delay_imperceptible"), "delay imperceptible"),
               (G("haptic_delay_threshold_discrete"), "delay detected")):
    ax.axvline(v, color=F.MUTED, lw=0.9, ls=(0, (4, 3)), zorder=4)
    ax.annotate(lab, (v * 1.06, 3), fontsize=6.3, color=F.INK2, rotation=90, va="bottom")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("loop latency (ms)"); ax.set_ylabel("largest stable virtual stiffness (N/m)")
ax.set_xticks([1, 3, 10, 30, 100, 300]); ax.set_xticklabels(["1", "3", "10", "30", "100", "300"])
F.grid(ax, "both"); F.panel(ax, "a", "the stability envelope closes with delay")

ax = axs[0, 1]
lat = np.geomspace(5, 400, 200)
a0 = es["learning_rate_at_zero_delay"]; th = es["learning_half_rate_latency_ms"]
ax.plot(lat, a0 / (1 + lat / th), color=S[0], zorder=3)
ax.scatter([60, 259.2], [G("adaptation_rate_60ms"), G("adaptation_rate_259ms")], s=32,
           color=F.RED, zorder=5, edgecolor="white", linewidth=0.8, label="verified measurements")
ax.axvline(th, color=F.MUTED, lw=0.9, ls=(0, (4, 3)))
ax.annotate(f"half-rate latency\n{th:.0f} ms", (th * 1.08, 0.11), fontsize=6.4, color=F.INK2)
ax.set_xscale("log"); ax.set_xticks([10, 30, 100, 300])
ax.set_xticklabels(["10", "30", "100", "300"])
ax.set_xlabel("feedback latency (ms)"); ax.set_ylabel("visuomotor adaptation rate")
F.legend(ax, loc="upper right")
F.grid(ax, "both"); F.panel(ax, "b", "latency sets the learning rate")

ax = axs[1, 0]
piv = env.pivot(index="dose_multiple", columns="latency_ms", values="mean_fm_gain_over_usual_care")
im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="Blues",
               extent=[-0.5, len(piv.columns) - 0.5, -0.5, len(piv.index) - 0.5])
cs = ax.contour(np.arange(len(piv.columns)), np.arange(len(piv.index)), piv.values,
                levels=[G("fmue_mdc")], colors=[F.RED], linewidths=1.6)
ax.clabel(cs, fmt={G("fmue_mdc"): "MDC 5.2 points"}, fontsize=6.4)
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([int(c) for c in piv.columns], fontsize=6.6)
ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f"{v:.0f}x" for v in piv.index], fontsize=6.6)
ax.set_xlabel("loop latency (ms)"); ax.set_ylabel("practice dose vs usual care")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("FM-UE gain over usual care", fontsize=6.6); cb.ax.tick_params(labelsize=6.4)
F.panel(ax, "c", "design envelope: dose buys more than latency")

ax = axs[1, 1]
matched = comp[comp.contrast == "matched dose"].copy()
yv = np.arange(len(matched))
ax.hlines(yv, matched.ci_lo, matched.ci_hi, color=F.MUTED, lw=1.6, zorder=2)
ax.scatter(matched.published_smd, yv, s=30, color=S[0], zorder=4)
ax.set_yticks(yv)
ax.set_yticklabels([c.split(",")[0] for c in matched.comparator], fontsize=6.6)
lo, hi = pred.matched_dose_smd.min(), pred.matched_dose_smd.max()
ax.axvspan(lo, hi, color=S[1], alpha=0.16, zorder=1)
ax.axvline(0, color=F.INK2, lw=0.9)
import matplotlib.patches as mpatches
F.legend(ax, handles=[mpatches.Patch(color=S[1], alpha=0.16,
                                  label=f"model prediction, matched dose\n(latency "
                                        f"{int(pred.latency_ms.iloc[0])}-{int(pred.latency_ms.iloc[-1])} ms)")],
          loc="lower left", fontsize=6.3)
ax.set_xlabel("standardised mean difference"); ax.invert_yaxis()
F.grid(ax, "x"); F.panel(ax, "d", "predicted vs published matched-dose effects")
fig.suptitle("Figure 8  ·  The closed-loop layer: an explicit design envelope",
             fontsize=10.5, x=0.005, ha="left", y=1.005)
F.note(fig, f"The latency-to-learning-rate function is fixed exactly by two verified adaptation measurements; the dose "
            f"exponent is fixed by one verified high-repetition study. The model then predicts that at MATCHED dose the "
            f"entire achievable benefit of better feedback is SMD {pred.matched_dose_smd.max():.2f} at "
            f"{int(pred.latency_ms.iloc[0])} ms, falling to {pred.matched_dose_smd.min():.2f} by "
            f"{int(pred.latency_ms.iloc[-1])} ms - bracketing the published robot-training effect of "
            f"{float(matched.published_smd.iloc[0]):.2f} and falling short of the larger published effects, which the "
            f"model can only reproduce as dose contrasts (SMD {pred.dose_amplified_smd.min():.1f}-"
            f"{pred.dose_amplified_smd.max():.1f}). The four largest published comparators in panel d are conversions of published point differences using one trial's within-group SD (see Methods), not published standardised effects.", y=0.0)
fig.tight_layout()
F.save(fig, "figure_08_closed_loop")

# ================================================================= Figure 9: multi-layer network
A = np.load(R / "F_adjacency.npy")
nodes = K.load_verified("genes_nodes")
prop = pd.read_csv(R / "F_rwr_propagation.csv")
conn = pd.read_csv(R / "F_cross_layer_connectors.csv")
lp = pd.read_csv(R / "F_layer_pair_enrichment.csv")
fs = json.loads((R / "F_network_summary.json").read_text())
genes = nodes.gene_symbol.tolist()
broad = {g: l.split("_")[0] for g, l in zip(nodes.gene_symbol, nodes.layer)}
LCOL = {"L1": S[0], "L2": S[1], "L3": S[2], "L4": S[3]}
LNAME = {"L1": "L1 pre-stroke risk", "L2": "L2 ictal cascade",
         "L3": "L3 systemic response", "L4": "L4 recovery"}

fig = plt.figure(figsize=(7.5, 6.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1], hspace=0.32, wspace=0.26)
ax = fig.add_subplot(gs[0, :])
order = ["L1", "L2", "L3", "L4"]
pos, cnt = {}, {l: 0 for l in order}
per = {l: sum(1 for g in genes if broad[g] == l) for l in order}
for g in genes:
    l = broad[g]
    n = per[l]
    i = cnt[l]; cnt[l] += 1
    pos[g] = (0.03 + 0.94 * (i + 0.5) / n, {"L1": 3, "L2": 2, "L3": 1, "L4": 0}[l])
score = prop.set_index("gene")["rwr_score"].to_dict()
for i in range(len(genes)):
    for j in range(i + 1, len(genes)):
        if A[i, j]:
            x1, y1 = pos[genes[i]]; x2, y2 = pos[genes[j]]
            same = broad[genes[i]] == broad[genes[j]]
            ax.plot([x1, x2], [y1, y2], color=F.MUTED if same else F.INK2,
                    alpha=0.18 if same else 0.5, lw=0.55 if same else 0.9, zorder=1)
for g in genes:
    x, y = pos[g]
    ax.scatter([x], [y], s=16 + 5200 * score.get(g, 0), color=LCOL[broad[g]], zorder=3,
               edgecolor="white", linewidth=0.5)
top = set(conn.head(6).gene) | {"NOS3"}
lab_i = 0
for g in genes:
    x, y = pos[g]
    if g in top:
        dy = 7 if lab_i % 2 == 0 else 16
        lab_i += 1
        ax.annotate(g, (x, y), fontsize=6.2, color=F.INK, ha="center", va="bottom",
                    xytext=(0, dy), textcoords="offset points", fontweight="bold")
for l in order:
    ax.annotate(LNAME[l], (0.0, {"L1": 3, "L2": 2, "L3": 1, "L4": 0}[l]), fontsize=6.8,
                color=LCOL[l], ha="left", va="center", fontweight="bold",
                xytext=(0, -16), textcoords="offset points")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.45, 3.45); ax.axis("off")
F.panel(ax, "a", "93 verified genes, 111 curated relations (marker area = reachability from the risk layer)")

ax = fig.add_subplot(gs[1, 0])
lp = lp.sort_values("observed", ascending=False)
yv = np.arange(len(lp))
ax.barh(yv - 0.19, lp.observed, height=0.36, color=S[0], zorder=3, label="observed")
ax.barh(yv + 0.19, lp.null_mean, height=0.36, color=F.MUTED, alpha=0.6, zorder=3,
        label="degree-preserving null")
for i, r in enumerate(lp.itertuples()):
    if r.p_two_sided < 0.05:
        ax.annotate("*", (max(r.observed, r.null_mean) + 1.5, i), fontsize=10, color=F.RED,
                    va="center")
ax.set_yticks(yv); ax.set_yticklabels(lp.layer_pair, fontsize=6.8)
ax.set_xlabel("number of relations"); ax.invert_yaxis(); F.legend(ax, loc="lower right", fontsize=6.5)
F.grid(ax, "x"); F.panel(ax, "b", "the mechanism literature is layer-siloed")

ax = fig.add_subplot(gs[1, 1])
cc = conn.head(10).iloc[::-1]
ax.barh(np.arange(len(cc)), cc.participation, color=[LCOL[b] for b in cc.broad_layer],
        height=0.62, zorder=3)
for i, r in enumerate(cc.itertuples()):
    ax.annotate(f"{int(r.n_layers_touched)} layers", (r.participation + 0.01, i), fontsize=6.2,
                color=F.INK2, va="center")
ax.set_yticks(np.arange(len(cc))); ax.set_yticklabels(cc.gene, fontsize=6.8)
ax.set_xlim(0, 0.95); ax.set_xlabel("participation across layers")
F.grid(ax, "x"); F.panel(ax, "c", "cross-layer connector genes")
fig.suptitle("Figure 9  ·  The molecular layer: what the curated mechanism set does and does not connect",
             fontsize=10.5, x=0.005, ha="left", y=0.995)
F.note(fig, f"Random-walk propagation from the {sum(1 for g in genes if broad[g]=='L1')} risk-layer genes reaches only "
            f"{fs['n_downstream_significant']} downstream gene ({', '.join(fs['downstream_significant_genes'])}) more "
            f"strongly than a degree-preserving null predicts. Within-layer relations are over-represented and "
            f"risk-to-ictal relations are UNDER-represented ({int(lp.loc[lp.layer_pair=='L1-L2','observed'].iloc[0])} "
            f"observed versus {float(lp.loc[lp.layer_pair=='L1-L2','null_mean'].iloc[0]):.0f} expected). A minimal cut of "
            f"{fs['min_cut_size']} relations separates the risk layer from the recovery layer entirely. This describes "
            f"the structure of the curated mechanism literature - and of our curation of it - not of biology.", y=-0.01)
F.save(fig, "figure_09_network")

# ==================================================================== Figure 10: validation summary
bench = pd.read_csv(R / "V_benchmark_master.csv")
sens = pd.read_csv(R / "V_global_sensitivity.csv")
vs = json.loads((R / "V_validation_summary.json").read_text())

fig, axs = plt.subplots(1, 2, figsize=(7.5, 5.2), width_ratios=[1.25, 1])
ax = axs[0]
b = bench.dropna(subset=["ratio"]).copy()
b = b[np.isfinite(b.ratio)]
b["kind"] = np.where(b.used_in_calibration, "calibration target", "out-of-sample test")
b = b.sort_values(["kind", "ratio"])
yv = np.arange(len(b))
cols = [F.MUTED if k == "calibration target" else (S[2] if p else F.RED)
        for k, p in zip(b.kind, b.passes)]
ax.scatter(b.ratio.clip(0.08, 12), yv, s=26, color=cols, zorder=4)
ax.axvline(1.0, color=F.INK2, lw=1.0, zorder=3)
ax.axvspan(0.75, 1.25, color=S[2], alpha=0.10, zorder=1)
ax.set_yticks(yv)
ax.set_yticklabels([f"{q[:46]}" for q in b.quantity], fontsize=6.0)
ax.set_xscale("log"); ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4, 10])
ax.set_xticklabels(["0.1x", "0.25x", "0.5x", "1x", "2x", "4x", "10x"])
ax.set_xlabel("model / published value")
for lab, col in (("calibration target", F.MUTED), ("out-of-sample pass", S[2]),
                 ("out-of-sample failure", F.RED)):
    ax.scatter([], [], s=26, color=col, label=lab)
F.legend(ax, loc="lower right", fontsize=6.4)
F.grid(ax, "x"); F.panel(ax, "a", "every point comparison in the framework")

ax = axs[1]
piv = sens.pivot(index="parameter", columns="output", values="srrc")
piv = piv.reindex(piv.abs().max(axis=1).sort_values().index)
im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(piv.columns)))
ax.set_xticklabels([c.replace("_", " ") for c in piv.columns], rotation=38, ha="right", fontsize=6.4)
ax.set_yticks(range(len(piv.index)))
ax.set_yticklabels([i.replace("_", " ") for i in piv.index], fontsize=6.6)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        if abs(v) > 0.2:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.8,
                    color="white" if abs(v) > 0.55 else F.INK)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.set_label("standardised rank-regression coefficient", fontsize=6.4)
cb.ax.tick_params(labelsize=6.2)
F.panel(ax, "b", "what the conclusions actually depend on")
fig.suptitle("Figure 10  ·  Validation accounting and global sensitivity", fontsize=10.5,
             x=0.005, ha="left", y=1.0)
F.note(fig, f"{vs['n_calibration_met']} of {vs['n_calibration_targets']} calibration targets are met by construction; "
            f"{vs['n_out_of_sample_passed']} of {vs['n_out_of_sample']} out-of-sample comparisons "
            f"({100*vs['out_of_sample_pass_rate']:.0f}%) fall within the published range or within 25% of the published "
            f"value. The failures are listed in Table S12 and discussed in the text; they are concentrated in the "
            f"progressor-proportion series (which is internally inconsistent with the growth-rate series it is compared "
            f"against), depolarisation frequency, and plateau timing. Panel b shows that no conclusion depends on more "
            f"than two parameters.", y=-0.01)
fig.tight_layout()
F.save(fig, "figure_10_validation")
