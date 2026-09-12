#!/usr/bin/env python3
"""Figures 1-2: the stroke continuum tree and the architecture (schematics)."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import kcore as K
import kfig as F

# ---------------------------------------------------------------------------- Figure 1: the tree
def box(ax, x, y, w, h, text, fc, ec=None, fs=7.0, weight="normal", tc=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                       linewidth=0.9, facecolor=fc, edgecolor=ec or fc, zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc or F.INK, zorder=4, fontweight=weight, linespacing=1.35)
    return (x + w / 2, y, x + w / 2, y + h)


def arrow(ax, p1, p2, color=F.MUTED, lw=1.0, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=8, linewidth=lw,
                                 color=color, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))


epi = {r.split(",")[0]: r.split(",")[1] for r in
       open(K.DATA / "epidemiology.csv").read().strip().split("\n")[1:]}
fig, ax = plt.subplots(figsize=(7.5, 8.3))
ax.set_xlim(-0.055, 1.0); ax.set_ylim(0.0, 1.0); ax.axis("off")

# The band label is split: a short rotated identifier in the margin, which fits inside even the
# shortest band, and the time scale set horizontally inside the band's own top-left corner. A
# single long rotated string overflowed the shorter bands and collided with its neighbours.
bands = [(0.726, 0.269, "#f2f6fd", "LAYER 1\nPRE-STROKE", "years to decades"),
         (0.430, 0.266, "#fdf1ec", "LAYER 2\nICTAL", "seconds to days"),
         (0.228, 0.190, "#eef9f4", "LAYER 3\nSYSTEMIC", "hours to weeks"),
         (0.012, 0.204, "#fdf6e6", "LAYER 4\nRECOVERY", "weeks to years")]
for y, h, c, label, scale in bands:
    ax.add_patch(Rectangle((0.005, y), 0.99, h, facecolor=c, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((-0.050, y), 0.032, h, facecolor=c, edgecolor="none", zorder=0))
    ax.text(-0.034, y + h / 2, label, fontsize=6.6, color=F.INK2, fontweight="bold",
            va="center", ha="center", rotation=90, zorder=5, linespacing=1.15)
    ax.text(0.012, y + h - 0.008, scale, fontsize=6.0, color=F.MUTED, va="top", ha="left",
            zorder=5, style="italic")

# --- Layer 1: fixed and modifiable substrate
box(ax, 0.03, 0.946, 0.94, 0.042,
    "CEREBROVASCULAR RESERVE — the state the brain brings to the occlusion",
    "#2a78d6", fs=8.0, weight="bold", tc="white")
l1 = [
    (0.025, "Inherited vessel wall", "NOTCH3 · HTRA1\nCOL4A1/2 · ACTA2\nRNF213 · FOXF2"),
    (0.268, "Cardiac / embolic", "PITX2 · ZFHX3\nNKX2-5\nAF in ≈5% of adults"),
    (0.511, "Haemostatic set-point", "F5 · F2 · F11 · ABO\nFGA/FGB · SERPINC1\nPROC · PROS1"),
    (0.754, "Metabolic and lipid", "LPA · MTHFR · GLA\nSH2B3\nLDL · HbA1c · hs-CRP"),
]
tops = []
for x, head, body in l1:
    box(ax, x, 0.884, 0.221, 0.030, head, "white", ec="#2a78d6", fs=7.2, weight="bold")
    box(ax, x, 0.806, 0.221, 0.070, body, "#eaf2fd", ec="#eaf2fd", fs=6.4)
    arrow(ax, (x + 0.11, 0.884), (x + 0.11, 0.876))
    tops.append(x + 0.11)
    arrow(ax, (x + 0.11, 0.946), (x + 0.11, 0.914))

box(ax, 0.17, 0.752, 0.66, 0.026,
    f"population burden: {epi['hypertension_prevalence']}% hypertension · "
    f"{epi['diabetes_total_prevalence']}% diabetes · {epi['current_smoking']}% smoking · "
    f"{epi['obesity_prevalence']}% obesity",
    "#dbe9fb", fs=6.5)
for t in tops:
    arrow(ax, (t, 0.806), (0.5, 0.778), rad=0.0)

# --- the occlusion
# the occlusion sits in the gap between the two bands, clear of both
box(ax, 0.330, 0.700, 0.34, 0.026, "ARTERIAL OCCLUSION · 87% ischaemic",
    F.RED, fs=7.6, weight="bold", tc="white")
arrow(ax, (0.5, 0.752), (0.5, 0.728), color=F.RED, lw=1.4)

# --- Layer 2 cascade chain
chain = [("Flow below the\npump threshold", "CBF < 8–12\nmL/100 g/min"),
         ("ATP collapse,\nionic failure", "K⁺ 3 → 60 mM\nwithin minutes"),
         ("Spreading\ndepolarisation", "2–3 mm/min\n20–30 min in core"),
         ("Excitotoxicity\nand acidosis", "GRIN2B · DLG4\nNOS1 · pH 6.5"),
         ("Irreversible\ninjury grows", "1.9×10⁶\nneurons/min")]
y0 = 0.560
w = 0.178
for i, (head, sub) in enumerate(chain):
    x = 0.012 + i * 0.197
    box(ax, x, y0, w, 0.046, head, "white", ec="#eb6834", fs=6.9, weight="bold")
    box(ax, x, y0 - 0.048, w, 0.044, sub, "#fdeee7", ec="#fdeee7", fs=6.2)
    if i:
        arrow(ax, (x - 0.019, y0 + 0.023), (x, y0 + 0.023), color="#eb6834", lw=1.2)
arrow(ax, (0.42, 0.700), (0.1, 0.606), color=F.RED, lw=1.2, rad=-0.18)

box(ax, 0.012, 0.447, 0.478, 0.040,
    "COLLATERAL SUPPLY sets the growth rate\nfast ≥ 10 mL/h versus slow < 5 mL/h",
    "#fbe0d4", ec="#fbe0d4", fs=6.5)
box(ax, 0.510, 0.447, 0.478, 0.040,
    "OXYGEN-EXTRACTION RESERVE buys time\nOEF 0.44 → 0.85 before the penumbra fails",
    "#fbe0d4", ec="#fbe0d4", fs=6.5)

# --- Layer 3 systemic
l3 = [("Barrier opening", "MMP9\npeak 18–24 h"),
      ("Innate immune\nactivation", "HMGB1 → TLR4\nNLRP3 · IL-6 24 h"),
      ("Blood signature", "ARG1 · S100A12\nCSF3R · WBC 0–24 h"),
      ("Oedema", "AQP4 · SUR1-TRPM4\nmaximum 48–72 h"),
      ("Acute-phase\nresponse", "CRP peak\n120–144 h")]
for i, (head, sub) in enumerate(l3):
    x = 0.012 + i * 0.197
    box(ax, x, 0.330, w, 0.044, head, "white", ec="#1baf7a", fs=6.9, weight="bold")
    box(ax, x, 0.282, w, 0.044, sub, "#e6f7f0", ec="#e6f7f0", fs=6.2)
    arrow(ax, (x + w / 2, 0.500), (x + w / 2, 0.374), color="#1baf7a", lw=0.9, ls=(0, (3, 2)))

# --- Layer 4 recovery
box(ax, 0.03, 0.176, 0.94, 0.034,
    "CORTICOSPINAL CAPACITY × PLASTICITY WINDOW × PRACTICE DOSE × AFFERENT SIGNAL QUALITY",
    "#eda100", fs=7.2, weight="bold")
l4 = [("Spontaneous\nrecovery", "fast component\nτ ≈ 3 weeks"),
      ("Sensitive period\nfor training", "slow component\nτ ≈ 55 weeks"),
      ("Degraded afferent\nfeedback", "learning rate halves\nat ≈150 ms delay"),
      ("Closed-loop\ninterface", "Module E\nlatency envelope"),
      ("Residual\nimpairment", "≈50% impaired\nat 6 months")]
for i, (head, sub) in enumerate(l4):
    x = 0.012 + i * 0.197
    box(ax, x, 0.104, w, 0.044, head, "white", ec="#eda100", fs=6.9, weight="bold")
    box(ax, x, 0.056, w, 0.044, sub, "#fdf1d5", ec="#fdf1d5", fs=6.2)
    arrow(ax, (x + w / 2, 0.176), (x + w / 2, 0.148), color="#eda100", lw=0.9)
arrow(ax, (0.5, 0.282), (0.5, 0.210), color="#1baf7a", lw=1.2)
box(ax, 0.18, 0.016, 0.64, 0.028,
    "recurrence returns the survivor to Layer 1 · ≈185 000 recurrent strokes per year (US)",
    "#f3edd8", ec="#f3edd8", fs=6.5)
arrow(ax, (0.18, 0.030), (0.022, 0.800), color=F.MUTED, lw=0.9, rad=0.32, ls=(0, (4, 3)))

ax.set_title("Figure 1  ·  The stroke continuum as four coupled layers", fontsize=10.5, pad=14)
F.note(fig, "Every quantitative annotation is a verified public value (Supplementary Tables S1-S9); "
            "gene symbols carry NCBI Gene identifiers in Table S3. Layer labels give the time scale on which "
            "each layer's state variables move. The dashed return path is recurrence, which returns the survivor "
            "to Layer 1 with an altered substrate.", y=0.035)
F.save(fig, "figure_01_continuum")

# ------------------------------------------------------------------- Figure 2: architecture
va = json.loads((K.RESULTS / "V_validation_summary.json").read_text())
fig, ax = plt.subplots(figsize=(7.3, 5.3))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

mods = [
    ("A", "Genomic risk\narchitecture", "30 verified GWAS loci\nliability-calibrated score", "#2a78d6", 0.045, 0.70),
    ("B", "Perfusion &\nhaemorheology", "depth–duration curve\ncollateral-set growth", "#eb6834", 0.275, 0.70),
    ("C", "Ictal cascade\n(14-state ODE)", "ions · SD · inflammation\nbarrier · oedema", "#eb6834", 0.505, 0.70),
    ("D", "Recovery law\n(two time scales)", "τ ≈ 3 w and ≈ 55 w\nproportional recovery", "#eda100", 0.735, 0.70),
    ("E", "Closed-loop interface\n(artificial peripheral thalamus)", "admittance control\nlatency-bounded envelope", "#4a3aa7", 0.505, 0.36),
    ("F", "Multi-layer molecular\nnetwork", "93 genes · 111 relations\nrandom walk + null models", "#1baf7a", 0.045, 0.36),
]
cent = {}
for tag, head, sub, col, x, y in mods:
    box(ax, x, y, 0.22, 0.056, head, "white", ec=col, fs=7.3, weight="bold")
    box(ax, x, y - 0.062, 0.22, 0.056, sub, "#f4f4f1", ec="#f4f4f1", fs=6.4)
    ax.add_patch(FancyBboxPatch((x, y + 0.056), 0.032, 0.026,
                                boxstyle="round,pad=0.004,rounding_size=0.01",
                                facecolor=col, edgecolor=col, zorder=4))
    ax.text(x + 0.016, y + 0.065, tag, ha="center", va="center", color="white",
            fontsize=7.4, fontweight="bold", zorder=5)
    cent[tag] = (x + 0.11, y, x + 0.11, y + 0.052)

for a, b in (("A", "B"), ("B", "C"), ("C", "D")):
    arrow(ax, (cent[a][0] + 0.115, cent[a][1] + 0.026), (cent[b][0] - 0.115, cent[b][1] + 0.026),
          color=F.INK2, lw=1.2)
arrow(ax, (cent["D"][0], cent["D"][1] - 0.062), (cent["E"][0] + 0.115, cent["E"][1] + 0.028),
      color=F.INK2, lw=1.2, rad=-0.25)
arrow(ax, (cent["E"][0] - 0.115, cent["E"][1] + 0.026), (cent["F"][0] + 0.115, cent["F"][1] + 0.026),
      color=F.MUTED, lw=0.9, ls=(0, (4, 3)))
arrow(ax, (cent["F"][0], cent["F"][1] + 0.056), (cent["A"][0], cent["A"][1] - 0.062),
      color=F.MUTED, lw=0.9, ls=(0, (4, 3)))

ax.text(0.045, 0.945, "shared state vector carried across layers", fontsize=7.6,
        fontweight="bold", color=F.INK)
shared = ["collateral index", "oxygen-extraction reserve", "inflammatory set-point",
          "corticospinal capacity", "afferent signal-to-noise", "plasticity gain"]
for i, s in enumerate(shared):
    x = 0.045 + (i % 3) * 0.315
    y = 0.886 - (i // 3) * 0.040
    box(ax, x, y, 0.295, 0.030, s, "#eceaf6", ec="#eceaf6", fs=6.6)

ax.text(0.045, 0.245, "calibration and validation accounting", fontsize=7.6, fontweight="bold",
        color=F.INK)
stats = [(f"{va['n_calibration_met']}/{va['n_calibration_targets']}", "calibration targets met",
          F.SERIES[2]),
         (f"{va['n_out_of_sample_passed']}/{va['n_out_of_sample']}",
          "out-of-sample benchmarks\nwithin published range", F.SERIES[3]),
         (f"{va['prr_range_targets_met']}/{va['prr_range_targets']}",
          "proportional-recovery\nstatistics reproduced", F.SERIES[3]),
         (f"{va['best_matched_dose_prediction']:.2f}",
          f"best matched-dose SMD predicted\n(published robot SMD {va['robot_smd_published']:.2f})",
          F.SERIES[0])]
for i, (num, lab, col) in enumerate(stats):
    x = 0.045 + i * 0.238
    ax.text(x, 0.19, num, fontsize=15, fontweight="bold", color=col, va="top")
    ax.text(x, 0.125, lab, fontsize=6.6, color=F.INK2, va="top", linespacing=1.4)

ax.text(0.045, 0.045, "Solid arrows: state passed forward. Dashed: annotation and feedback only — "
        "the molecular layer is literature-curated and therefore hypothesis-generating.",
        fontsize=6.6, color=F.MUTED, va="top")
ax.set_title("Figure 2  ·  KAIROS: module structure, shared state and validation accounting",
             fontsize=10.5, pad=12)
F.save(fig, "figure_02_architecture")
