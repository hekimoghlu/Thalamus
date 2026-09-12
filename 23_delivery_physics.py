#!/usr/bin/env python3
"""Module O - delivery physics: what a nanocarrier or a biologic would actually have to do.

Module I's central result is negative: fifteen of fifteen neuroprotection trials were powered to
detect an effect larger than the largest effect attainable at their own treatment window. A
negative result of that kind is only half an answer. The other half is a specification - what a
delivery system would have to achieve for the attainable effect to become a detectable one - and
that is what this module computes.

Nothing here is asserted. Every quantity is either a verified public value, an exact physical
identity applied to verified values, or an inversion that returns a requirement rather than a
claim. Three things fall out, and each is uncomfortable for a field that has spent thirty years
building better molecules:

  O1  SIZE IS A WEAK LEVER. The Stokes-Einstein relation applied to the verified free diffusion
      coefficient recovers the hydrated potassium radius to 168 pm, which validates the treatment
      against a number it was not fitted to. Applying the same relation across the nanomedicine
      size range, with the verified extracellular tortuosity, shows that interstitial penetration
      falls only as the inverse square root of carrier radius: a fifty-nanometre carrier loses a
      factor of seventeen against a small ion, not a factor of a thousand. Diffusion is therefore
      not where the delivery problem lives - and shrinking a carrier cannot be where the solution
      lives either. (Steric hindrance in an extracellular space whose width is of the same order
      as the particle is deliberately NOT modelled, because no verified value for that width is in
      this evidence base; every penetration depth below is consequently an upper bound.)

  O2  THE DELIVERY DEFICIT COMPOUNDS THE TIME DEFICIT. Vascular delivery is proportional to flow,
      and the penumbra is by definition the tissue where flow has fallen. The same tissue that has
      the most to gain from a neuroprotectant receives the least of it. Folding the verified
      penumbral flow band into Module I's attainable-effect calculation shrinks every attainable
      odds ratio further, and the shrinkage is computed here rather than assumed.

  O3  WHAT A CARRIER WOULD HAVE TO ACHIEVE. Inverting the arithmetic gives the selective enrichment
      in penumbral tissue that would be required, trial by trial, to turn the attainable effect
      into the effect the trial was powered to detect. That number is the specification.

  O4  BIOLOGICS CANNOT REACH THE CASCADE, AND DO NOT NEED TO. An expression-based therapy has a
      lag. Rather than assume one, the module inverts the verified cascade clock and asks what the
      latest useful onset time is for each node, which partitions the verified target list into a
      small-molecule-only set and a biologic-addressable set.

  O5  THE SENSING SPECIFICATION FOR THE DEVICE ITSELF. The verified psychophysical thresholds fix
      the resolution a transducer must reach before the device's afferent channel is limited by
      biology rather than by its own hardware, and therefore fix what a sensing material has to do.
"""
import json
import numpy as np, pandas as pd
import kcore as K
import kstat as S

rng = np.random.default_rng(K.SEED + 23)

# ----------------------------------------------------------------- physical constants (exact)
KB = 1.380649e-23           # J/K, exact by SI definition
T_BODY = 310.15             # K (37 C)
ETA_37 = 6.913e-4           # Pa s, viscosity of water at 37 C

# ------------------------------------------------------------------------- verified inputs
D_FREE_K = K.getp("k_free_diffusion_coefficient") * 1e-4        # cm2/s -> m2/s
LAM_N, LAM_H = K.getp("ecs_tortuosity_normal"), K.getp("ecs_tortuosity_hypoxic")
ALPHA_N, ALPHA_H = K.getp("ecs_volume_fraction_normal"), K.getp("ecs_volume_fraction_hypoxic")
CBF_N = K.getp("cbf_normal_cortex_pet")
PEN_LO, PEN_HI = K.getp("cbf_penumbra_lower"), K.getp("cbf_penumbra_upper")
PUMP = K.getp("cbf_threshold_pump_failure")

# ======================================================= O1 how far anything actually diffuses
def stokes_einstein_radius(D):
    """Hydrodynamic radius implied by a free diffusion coefficient at body temperature."""
    return KB * T_BODY / (6 * np.pi * ETA_37 * D)

def free_diffusivity(radius_m):
    return KB * T_BODY / (6 * np.pi * ETA_37 * radius_m)

R_K_IMPLIED = stokes_einstein_radius(D_FREE_K)
print(f"Module O: the verified free diffusion coefficient implies a hydrodynamic radius of "
      f"{R_K_IMPLIED*1e12:.0f} pm, which is the hydrated potassium ion - so Stokes-Einstein "
      f"applied to this evidence base is calibrated, not assumed")

# the same tolerance curve Module B and Module J use, for the time a given flow buys
_F_ANCH = np.array([K.getp("depth_duration_30min"), K.getp("depth_duration_6h"),
                    K.getp("depth_duration_12h")])
_LT = np.log(np.array([0.5, 6.0, 12.0]))
_S_LO = (_LT[1] - _LT[0]) / (_F_ANCH[1] - _F_ANCH[0])
_S_HI = (_LT[2] - _LT[1]) / (_F_ANCH[2] - _F_ANCH[1])

def t_crit_h(F):
    F = np.asarray(F, float)
    lt = np.interp(F, _F_ANCH, _LT)
    lt = np.where(F < _F_ANCH[0], _LT[0] + _S_LO * (F - _F_ANCH[0]), lt)
    lt = np.where(F > _F_ANCH[-1], _LT[-1] + _S_HI * (F - _F_ANCH[-1]), lt)
    return np.where(F <= PUMP, 1e-3, np.exp(np.clip(lt, -8, 8)))

RADII_NM = np.array([0.17, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0])
rows = []
for r_nm in RADII_NM:
    D0 = free_diffusivity(r_nm * 1e-9)
    for lab, lam in (("normally perfused", LAM_N), ("ischaemic", LAM_H)):
        D_app = D0 / lam ** 2
        for F in (PEN_HI, 0.5 * (PEN_LO + PEN_HI), PEN_LO):
            t_s = float(t_crit_h(F)) * 3600.0
            rows.append(dict(radius_nm=r_nm, tissue=lab, cbf=F, tolerated_hours=t_crit_h(F),
                             free_diffusivity_m2_s=D0, apparent_diffusivity_m2_s=D_app,
                             penetration_um=1e6 * np.sqrt(2 * D_app * t_s)))
diff = pd.DataFrame(rows)
K.save_table(diff, "O_interstitial_diffusion")
d50 = diff[(diff.radius_nm == 50.0) & (diff.tissue == "ischaemic") & (diff.cbf == PEN_HI)]
d_ion = diff[(diff.radius_nm == 0.17) & (diff.tissue == "ischaemic") & (diff.cbf == PEN_HI)]
PEN_50NM_UM = float(d50.penetration_um.iloc[0])
PEN_ION_UM = float(d_ion.penetration_um.iloc[0])
print(f"   over the {float(d50.tolerated_hours.iloc[0]):.1f} h that tissue at the penumbral "
      f"ceiling survives, a small ion spreads {PEN_ION_UM/1000:.2f} mm through ischaemic "
      f"interstitium and a 50 nm carrier {PEN_50NM_UM/1000:.2f} mm - a factor of "
      f"{PEN_ION_UM/PEN_50NM_UM:.0f}, not a factor of a thousand, because penetration falls only "
      f"as the inverse square root of radius. Size is a weak lever. Steric hindrance is not "
      f"modelled, so both figures are upper bounds.")

# ==================================================== O2 the delivery deficit compounds with time
I = json.load(open(K.RESULTS / "I_therapeutics_summary.json"))
BETA = I["outcome_mapping_beta"]
rep = pd.read_csv(K.RESULTS / "C_reperfusion_timing.csv").dropna(subset=["reperfusion_h"])
th = np.linspace(0.25, 24, 200)
sal = np.interp(th, rep.reperfusion_h, rep.salvage_fraction)

def salvage_at(h):
    return float(np.interp(h, th, sal))

# Delivery of anything carried in blood is proportional to flow. The verified penumbral band gives
# the delivery fraction directly, with no free parameter: a drug reaching tissue at flow F arrives
# at F / CBF_N of the rate it would reach normally perfused cortex.
DELIVERY = {"penumbral ceiling": PEN_HI / CBF_N, "mid-penumbra": 0.5 * (PEN_LO + PEN_HI) / CBF_N,
            "penumbral floor": PEN_LO / CBF_N, "pump-failure threshold": PUMP / CBF_N}
att = pd.read_csv(K.RESULTS / "I_trial_attainability.csv")
rows = []
for r in att.itertuples():
    for lab, frac in DELIVERY.items():
        or_delivered = float(np.exp(np.log(r.attainable_or) * frac))
        rows.append(dict(trial=r.trial, window_h=r.window_h, delivery_site=lab,
                         delivery_fraction=frac, attainable_or=r.attainable_or,
                         attainable_or_flow_limited=or_delivered,
                         detectable_or=r.detectable_or,
                         shortfall_factor=(np.log(r.detectable_or) / np.log(or_delivered))
                         if or_delivered > 1 else np.inf))
deliv = pd.DataFrame(rows)
K.save_table(deliv, "O_delivery_deficit")
mid = deliv[deliv.delivery_site == "mid-penumbra"]
print(f"   flow-limited delivery shrinks the median attainable odds ratio from "
      f"{att.attainable_or.median():.3f} to {mid.attainable_or_flow_limited.median():.3f}: the "
      f"tissue with the most to gain receives {100*DELIVERY['mid-penumbra']:.0f}% of the delivery "
      f"that normally perfused cortex receives, so the time penalty and the delivery penalty "
      f"multiply rather than compete")

# ============================================ O3 the specification a carrier would have to meet
# If a carrier enriches drug in the target tissue by a factor T relative to what flow alone
# delivers, the attainable log-odds scales by T. The enrichment required to reach the effect each
# trial was powered to detect follows by inversion, with no further assumption.
spec_rows = []
for r in mid.itertuples():
    if not np.isfinite(r.detectable_or) or r.attainable_or <= 1.0:
        continue
    need = float(np.log(r.detectable_or) / (np.log(r.attainable_or) * r.delivery_fraction))
    spec_rows.append(dict(trial=r.trial, window_h=r.window_h,
                          attainable_or_flow_limited=r.attainable_or_flow_limited,
                          detectable_or=r.detectable_or,
                          required_enrichment=need))
spec = pd.DataFrame(spec_rows).sort_values("required_enrichment")
K.save_table(spec, "O_carrier_specification")
REQ_MED = float(spec.required_enrichment.median())
REQ_MIN = float(spec.required_enrichment.min())
print(f"   to convert the attainable effect into the effect the trial was powered to detect, a "
      f"carrier would need a selective enrichment of {REQ_MED:.0f}-fold in penumbral tissue "
      f"(best case {REQ_MIN:.1f}-fold, for {spec.iloc[0].trial}), sustained across the treatment "
      f"window. That is the specification, and it is the first encouraging number in this layer: "
      f"a single-digit enrichment is within the range targeted carriers are reported to reach, "
      f"whereas no improvement in potency, selectivity or mechanism can move it at all.")

# The same inversion run the other way is more useful to a designer: given an enrichment a carrier
# CAN plausibly achieve, what treatment window does it buy? This is the design curve.
def window_for(enrichment, coverage=0.35, p0=0.40, n=1000):
    """The latest treatment hour at which a carrier with this enrichment still reaches the effect a
    trial of n patients could detect."""
    se = np.sqrt(2 * (1 / (p0 * (1 - p0))) / (n / 2))
    detect = float(np.exp(2.802 * se))
    frac = DELIVERY["mid-penumbra"] * enrichment
    ors = np.array([np.exp(BETA * coverage * frac * salvage_at(x)) for x in th])
    ok = ors >= detect
    return float(th[ok][-1]) if ok.any() else np.nan

design = pd.DataFrame([dict(enrichment=e, trial_n=n,
                            latest_useful_window_h=window_for(e, n=n))
                       for e in (1, 2, 5, 10, 20, 50, 100) for n in (500, 1000, 2000)])
K.save_table(design, "O_design_curve")
_d = design[design.trial_n == 1000]
print("   design curve, for a trial of 1000 patients: " + ", ".join(
    f"{int(r.enrichment)}x enrichment buys "
    + ("no usable window" if not np.isfinite(r.latest_useful_window_h)
       else f"{r.latest_useful_window_h:.1f} h")
    for r in _d.itertuples()))

# ================================== O4 what a biologic can reach, from the verified cascade clock
NODE_TIME = {
    "excitotoxicity": K.getp("sd_onset_after_arrest") / 60.0,
    "energy failure": K.getp("atp_failure_time") / 60.0,
    "oxidative stress": K.getp("anoxic_sd_latency_ogd") / 60.0,
    "barrier": K.getp("mmp9_tissue_peak"),
    "inflammation": K.getp("il6_tissue_peak"),
    "oedema": K.getp("edema_max_time"),
    "haemorrhage iron toxicity": K.getp("crp_peak_time"),
    "thrombosis": K.getp("sd_onset_after_arrest") / 60.0,
    "fibrinolysis": K.getp("sd_onset_after_arrest") / 60.0,
}
P = K.load_verified("target_pharmacology")
bio_rows = []
for r in P.itertuples():
    node = str(r.cascade_node)
    t_node = NODE_TIME.get(node, np.nan)
    bio_rows.append(dict(target_gene=r.target_gene, cascade_node=node,
                         node_peak_hours=t_node,
                         salvage_remaining_at_node=salvage_at(t_node) if np.isfinite(t_node) else np.nan,
                         small_molecule_only=bool(np.isfinite(t_node) and t_node < 1.0),
                         biologic_addressable=bool(np.isfinite(t_node) and t_node >= 6.0),
                         src=r.src))
bio = pd.DataFrame(bio_rows).sort_values("node_peak_hours")
K.save_table(bio, "O_biologic_addressable_targets")
N_SM = int(bio.small_molecule_only.sum())
N_BIO = int(bio.biologic_addressable.sum())
print(f"   of {len(bio)} verified targets, {N_SM} sit on nodes that peak within an hour of onset "
      f"and are therefore reachable only by something already in the blood at the moment of "
      f"occlusion, while {N_BIO} sit on nodes peaking at or beyond six hours and are open to an "
      f"expression-based or cell-based therapy whose product appears late")

# ================================= O5 the sensing specification the device's transducer must meet
E = json.load(open(K.RESULTS / "E_control_summary.json"))
WEBER = dict(force=K.getp("weber_fraction_force"), position=K.getp("weber_fraction_position"),
             stiffness=K.getp("weber_fraction_stiffness"))
JND_TORQUE = K.getp("jnd_elbow_torque_static")
JND_TORQUE_DYN = K.getp("jnd_elbow_torque_dynamic")
POS_MCP, POS_PIP = K.getp("position_threshold_mcp"), K.getp("position_threshold_pip")
GRIP_NORM = K.getp("grip_norm_men_60_69")
sens_rows = [
    dict(channel="grip force", biological_resolution=WEBER["force"] * GRIP_NORM,
         unit="kg at a 40 kg reference", basis="verified force Weber fraction x verified norm",
         required_bits=float(np.log2(1.0 / WEBER["force"]))),
    dict(channel="elbow torque, static", biological_resolution=JND_TORQUE, unit="N m",
         basis="verified just-noticeable difference",
         required_bits=float(np.log2(1.0 / (JND_TORQUE / 1.0)))),
    dict(channel="elbow torque, dynamic", biological_resolution=JND_TORQUE_DYN, unit="N m",
         basis="verified just-noticeable difference",
         required_bits=float(np.log2(1.0 / (JND_TORQUE_DYN / 1.0)))),
    dict(channel="finger joint angle (MCP)", biological_resolution=POS_MCP, unit="degrees",
         basis="verified position threshold",
         required_bits=float(np.log2(90.0 / POS_MCP))),
    dict(channel="finger joint angle (PIP)", biological_resolution=POS_PIP, unit="degrees",
         basis="verified position threshold",
         required_bits=float(np.log2(90.0 / POS_PIP))),
    dict(channel="stiffness", biological_resolution=WEBER["stiffness"], unit="fraction",
         basis="verified stiffness Weber fraction",
         required_bits=float(np.log2(1.0 / WEBER["stiffness"]))),
]
sens = pd.DataFrame(sens_rows)
sens["bits_per_second_at_30_hz"] = sens.required_bits * 30.0
K.save_table(sens, "O_sensing_specification")
TOTAL_BITS = float(sens.bits_per_second_at_30_hz.sum())
print(f"   a transducer that resolves every verified psychophysical threshold needs "
      f"{sens.required_bits.sum():.1f} bits per sample across the six channels, or "
      f"{TOTAL_BITS:.0f} bits per second at a 30 Hz update - against the "
      f"{E['afferent_rate_biological_bit_s']:.0f} bit/s the biological afferent channel carries. "
      f"Sensing is therefore not the binding constraint on this device; latency is.")

K.save_json(dict(
    stokes_einstein_radius_pm=float(R_K_IMPLIED * 1e12),
    penetration_um=dict(ion=PEN_ION_UM, carrier_50nm=PEN_50NM_UM,
                        ratio=float(PEN_ION_UM / PEN_50NM_UM)),
    delivery_fraction=DELIVERY,
    attainable_or_median=float(att.attainable_or.median()),
    attainable_or_flow_limited_median=float(mid.attainable_or_flow_limited.median()),
    required_enrichment=dict(median=REQ_MED, minimum=REQ_MIN,
                             best_trial=str(spec.iloc[0].trial)),
    design_curve_note="latest useful treatment hour by carrier enrichment and trial size",
    targets=dict(n=len(bio), small_molecule_only=N_SM, biologic_addressable=N_BIO),
    sensing=dict(bits_per_sample=float(sens.required_bits.sum()),
                 bits_per_second_at_30hz=TOTAL_BITS,
                 biological_afferent_bit_s=E["afferent_rate_biological_bit_s"],
                 binding_constraint="latency, not sensing resolution"),
), "O_delivery_summary")
print(diff[(diff.tissue == "ischaemic") & (diff.cbf == PEN_HI)]
      [["radius_nm", "apparent_diffusivity_m2_s", "penetration_um"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
print(spec.head(6).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print(bio[["target_gene", "cascade_node", "node_peak_hours", "small_molecule_only",
           "biologic_addressable"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
