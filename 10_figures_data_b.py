#!/usr/bin/env python3
"""Figures 5-6: ictal cascade dynamics and the systemic biomarker timeline."""
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import kcore as K
import kfig as F

R = K.RESULTS; S = F.SERIES; G = K.getp
cs = json.loads((R / "C_cascade_summary.json").read_text())
IDX = cs["state_index"]
t = np.load(R / "C_time.npy")
core = np.load(R / "C_trace_core.npy"); sev = np.load(R / "C_trace_sev.npy")
pen = np.load(R / "C_trace_pen.npy")
scen = pd.read_csv(R / "C_scenarios.csv"); rep = pd.read_csv(R / "C_reperfusion_timing.csv")
val = pd.read_csv(R / "C_validation.csv")

# ================================================================ Figure 5: cascade dynamics
fig, axs = plt.subplots(2, 3, figsize=(7.5, 4.9))
tm = t * 60.0                                   # minutes for the fast panels
fast = tm <= 60

ax = axs[0, 0]
ax.plot(tm[fast], core[fast, IDX["A"]], color=S[0], label="core (5% of normal flow)")
ax.plot(tm[fast], sev[fast, IDX["A"]], color=S[1], label="severe penumbra (15%)")
ax.plot(tm[fast], pen[fast, IDX["A"]], color=S[2], label="penumbra (25%)")
ax.axhline(G("atp_baseline"), color=F.MUTED, lw=0.8, ls=(0, (4, 3)))
ax.annotate("baseline 2 mM", (58, G("atp_baseline")), fontsize=6.2, color=F.MUTED, ha="right",
            va="bottom")
ax.set_xlabel("minutes"); ax.set_ylabel("ATP (mM)")
F.legend(ax, loc="upper right", fontsize=6.4)
F.grid(ax); F.panel(ax, "a", "energy failure")

ax = axs[0, 1]
ax.plot(tm[fast], core[fast, IDX["Ko"]], color=S[0])
ax.plot(tm[fast], sev[fast, IDX["Ko"]], color=S[1])
ax.plot(tm[fast], pen[fast, IDX["Ko"]], color=S[2])
ax.axhline(G("k_extracellular_sd_peak"), color=F.RED, lw=0.9, ls=(0, (4, 3)))
ax.annotate("verified anoxic plateau 60 mM", (58, 61), fontsize=6.2, color=F.RED, ha="right")
ax.set_xlabel("minutes"); ax.set_ylabel("extracellular K$^+$ (mM)")
F.grid(ax); F.panel(ax, "b", "ionic collapse")

ax = axs[0, 2]
ax.plot(tm[fast], core[fast, IDX["Ca"]], color=S[0], label="Ca$^{2+}$ (fold), core")
ax.plot(tm[fast], sev[fast, IDX["Ca"]], color=S[1], label="Ca$^{2+}$ (fold), 15% flow")
ax2 = ax.twiny() if False else None
ax.plot(tm[fast], core[fast, IDX["pH"]] - 6.0, color=S[3], ls=(0, (5, 2)),
        label="pH - 6.0, core")
ax.scatter([8.0], [G("ca_ogd_fold_rise")], s=28, color=F.RED, zorder=5, edgecolor="white",
           linewidth=0.7, label="verified Ca$^{2+}$ at 8 min")
ax.set_xlabel("minutes"); ax.set_ylabel("fold of rest  /  pH offset")
F.legend(ax, loc="upper left", fontsize=6.2)
F.grid(ax); F.panel(ax, "c", "excitotoxic load and acidosis")

ax = axs[1, 0]
for Y, lab, c in ((core, "core", S[0]), (sev, "severe penumbra", S[1]), (pen, "penumbra", S[2])):
    ax.plot(t, Y[:, IDX["Nsd"]], color=c, label=lab)
ax.axhline(G("sd_events_first_6h"), color=F.RED, lw=0.9, ls=(0, (4, 3)))
ax.annotate("verified ≈6 events\nin the first 6 h", (23, G("sd_events_first_6h") + 1), fontsize=6.2,
            color=F.RED, ha="right", va="bottom")
ax.set_xlim(0, 24); ax.set_ylim(0, 40)
ax.set_xlabel("hours"); ax.set_ylabel("cumulative depolarisations")
F.legend(ax, loc="upper left", fontsize=6.4)
F.grid(ax); F.panel(ax, "d", "depolarisation count")

ax = axs[1, 1]
for Y, lab, c in ((core, "core", S[0]), (sev, "severe penumbra", S[1]), (pen, "penumbra", S[2])):
    ax.plot(t, 100 * Y[:, IDX["Dead"]], color=c, label=lab)
ax.axvspan(G("time_to_complete_infarction", "value_lo"), G("time_to_complete_infarction", "value_hi"),
           color=F.RED, alpha=0.10, zorder=1)
ax.annotate("published window for\ncomplete infarction", (18, 22), fontsize=6.2, color=F.RED,
            ha="left")
ax.set_xlim(0, 48); ax.set_xlabel("hours"); ax.set_ylabel("irreversibly injured (%)")
F.grid(ax); F.panel(ax, "e", "injury accrual")

ax = axs[1, 2]
r = rep.dropna(subset=["reperfusion_h"])
ax.plot(r.reperfusion_h, 100 * r.salvage_fraction, color=S[0], marker="o", markersize=4)
ax.axvline(4.5, color=F.MUTED, lw=0.9, ls=(0, (4, 3)))
ax.annotate("4.5 h", (4.5, 8), fontsize=6.4, color=F.MUTED, ha="left")
ax.set_xscale("log"); ax.set_xticks([0.25, 1, 3, 6, 12, 24])
ax.set_xticklabels(["0.25", "1", "3", "6", "12", "24"])
ax.set_xlabel("hour of reperfusion"); ax.set_ylabel("tissue salvaged (%)")
F.grid(ax); F.panel(ax, "f", "salvage versus reperfusion time")
fig.suptitle("Figure 5  ·  Layer 2b: the ictal cascade as a 14-state dynamical system",
             fontsize=10.5, x=0.005, ha="left", y=1.01)
F.note(fig, "Three free constants (metabolic scale, K$^+$ conductance, injury hazard) and one Ca$^{2+}$ influx "
            "coefficient are solved inside the model against four verified observations; the K$^+$ ceiling, the Na$^+$ "
            "ceiling and the pH floor are verified anchors rather than fitted. Panel d shows a genuine failure: the "
            f"model generates {int(scen.n_sd_6h.iloc[0])} depolarisations in the first 6 h against a verified ≈6, and "
            "the two verified onset figures themselves disagree threefold (1-2 min after in-vivo flow arrest versus 4.3-8.2 min in slice OGD), and the model can satisfy only one.", y=0.0)
fig.tight_layout()
F.save(fig, "figure_05_cascade")

# ========================================================== Figure 6: systemic biomarker timeline
fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.4), width_ratios=[1.3, 1])
ax = axs[0]
series = [("Infl", "parenchymal cytokine (IL-6-like)", S[0], "il6_tissue_peak"),
          ("Mmp", "MMP-9 activity", S[1], "mmp9_tissue_peak"),
          ("Bbb", "barrier permeability", S[2], "bbb_phase2_peak"),
          ("Wat", "tissue water (oedema)", S[3], "edema_max_time"),
          ("WBC", "circulating leukocytes", S[4], "wbc_peak_time"),
          ("CRP", "C-reactive protein", F.VIOLET, "crp_peak_time")]
for name, lab, c, pk in series:
    y = core[:, IDX[name]]
    if y.max() <= 0:
        continue
    ax.plot(t, y / y.max(), color=c, label=lab, lw=1.8)
    ax.scatter([G(pk)], [1.0], s=22, color=c, zorder=5, edgecolor="white", linewidth=0.8)
ax.set_xscale("log"); ax.set_xlim(0.5, 260); ax.set_ylim(0, 1.28)
ax.set_xticks([1, 6, 12, 24, 48, 72, 168])
ax.set_xticklabels(["1", "6", "12", "24", "48", "72", "168"])
ax.set_xlabel("hours from onset (log scale)"); ax.set_ylabel("normalised to own peak")
F.legend(ax, loc="upper left", fontsize=6.3, ncols=2, columnspacing=1.0, handlelength=1.4)
F.grid(ax); F.panel(ax, "a", "a timed sequence, not one inflammatory wave")

ax = axs[1]
clock = [("circulating leukocytes", "wbc_peak_time", S[4]),
         ("MMP-9 activity", "mmp9_tissue_peak", S[1]),
         ("parenchymal cytokine", "il6_tissue_peak", S[0]),
         ("barrier permeability", "bbb_phase2_peak", S[2]),
         ("tissue water (oedema)", "edema_max_time", S[3]),
         ("C-reactive protein", "crp_peak_time", F.VIOLET)]
for i, (lab, pname, c) in enumerate(clock):
    try:
        lo, hi = G(pname, "value_lo"), G(pname, "value_hi")
    except ValueError:
        lo = hi = G(pname)
    ax.plot([lo, hi], [i, i], color=c, lw=6, solid_capstyle="round", alpha=0.35, zorder=2)
    ax.scatter([G(pname)], [i], s=26, color=c, zorder=4)
    key = {"wbc_peak_time": "wbc", "mmp9_tissue_peak": "mmp", "il6_tissue_peak": "infl",
           "bbb_phase2_peak": "bbb", "edema_max_time": "wat", "crp_peak_time": "crp"}[pname]
    mv = float(scen.iloc[0][f"{key}_peak_h"])
    ax.scatter([mv], [i], s=40, facecolor="none", edgecolor=F.INK, linewidth=1.1, zorder=5)
ax.set_yticks(range(len(clock)))
ax.set_yticklabels([c[0] for c in clock], fontsize=6.8)
ax.set_xscale("log"); ax.set_xticks([6, 12, 24, 48, 96, 168])
ax.set_xticklabels(["6", "12", "24", "48", "96", "168"])
ax.set_xlim(5, 300); ax.set_xlabel("hours from onset"); ax.invert_yaxis()
ax.scatter([], [], s=26, color=F.MUTED, label="verified peak (bar: published range)")
ax.scatter([], [], s=40, facecolor="none", edgecolor=F.INK, linewidth=1.1, label="model peak")
F.legend(ax, loc="lower right", fontsize=6.4)
F.grid(ax, "x"); F.panel(ax, "b", "the acute-phase clock")
fig.suptitle("Figure 6  ·  Layer 3: the systemic response as a measurable clock",
             fontsize=10.5, x=0.005, ha="left", y=1.02)
F.note(fig, "Each compartment carries one free constant - the scale of its response kernel - solved so that its peak "
            "coincides with the verified peak time; panel b is therefore a calibration check, not a test. The result "
            "that is NOT free is the ordering constraint: MMP-9 (18 h) and the leukocyte peak (12 h) precede the IL-6 "
            "peak (24 h), so neither can be a response to accumulated injury or to the cytokine state. In the model "
            "only the early ionic-collapse signal is fast enough to place them where the data put them.", y=-0.02)
fig.tight_layout()
F.save(fig, "figure_06_biomarker_clock")
