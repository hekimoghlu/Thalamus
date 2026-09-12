#!/usr/bin/env python3
"""validation aggregator and global sensitivity analysis.

Part 1 collects every model-versus-published comparison produced by Modules A-F into a single
benchmark table, separating quantities that were used to calibrate the model from quantities that
were not, and reporting the out-of-sample pass rate - the only number that carries evidential
weight.

Part 2 runs a Latin-hypercube global sensitivity analysis over the model's structural parameters
(not the calibrated ones), reporting standardised rank-regression coefficients for each output, so
that readers can see which assumptions the conclusions actually rest on.
"""
import json
import numpy as np, pandas as pd
import kcore as K

G = K.getp
rng = np.random.default_rng(K.SEED + 7)

# ------------------------------------------------------------------ Part 1: benchmark assembly
frames = []

b = K.load_verified("../../results/B_tissue_fate_validation".replace("../../results/", ""), ) if False else pd.read_csv(K.RESULTS / "B_tissue_fate_validation.csv")
b["module"] = "B perfusion / tissue fate"
frames.append(b.rename(columns={"ratio_model_to_published": "ratio"}))

c = pd.read_csv(K.RESULTS / "C_validation.csv")
c["module"] = "C ictal cascade"
c["ratio"] = c.model / c.published.replace(0, np.nan)
frames.append(c)

d = pd.read_csv(K.RESULTS / "D_time_course_validation.csv")
d["module"] = "D recovery time course"
d["ratio"] = d.model / d.published
d["pub_lo"] = np.nan; d["pub_hi"] = np.nan
frames.append(d)

bench = pd.concat(frames, ignore_index=True, sort=False)

def passed(r):
    if not np.isnan(r.get("pub_lo", np.nan)) and not np.isnan(r.get("pub_hi", np.nan)):
        return bool(min(r.pub_lo, r.pub_hi) <= r.model <= max(r.pub_lo, r.pub_hi))
    if r.published == 0:
        return bool(abs(r.model) < 1e-3)
    return bool(abs(r.model - r.published) / abs(r.published) <= 0.25)

bench["passes"] = [passed(r) for _, r in bench.iterrows()]
bench = bench[["module", "quantity", "model", "published", "pub_lo", "pub_hi",
               "used_in_calibration", "passes", "ratio", "src"]]
K.save_table(bench, "V_benchmark_master")

oos = bench[~bench.used_in_calibration.astype(bool)]
cal = bench[bench.used_in_calibration.astype(bool)]

# range-valued comparisons from Module D (proportional recovery) and Module E (effect sizes)
prr = pd.read_csv(K.RESULTS / "D_prr_validation.csv")
epred = pd.read_csv(K.RESULTS / "E_predicted_effect_sizes.csv")
ecomp = pd.read_csv(K.RESULTS / "E_published_comparators.csv")
matched = ecomp[ecomp.contrast == "matched dose"]

summary = dict(
    n_point_benchmarks=int(len(bench)),
    n_calibration_targets=int(len(cal)),
    n_calibration_met=int(cal.passes.sum()),
    n_out_of_sample=int(len(oos)),
    n_out_of_sample_passed=int(oos.passes.sum()),
    out_of_sample_pass_rate=float(oos.passes.mean()),
    out_of_sample_failures=oos.loc[~oos.passes, ["module", "quantity", "model", "published"]]
        .to_dict("records"),
    prr_range_targets_met=int(prr.model_overlaps_published_range.sum()),
    prr_range_targets=int(len(prr)),
    matched_dose_smd_prediction_span=[float(epred.matched_dose_smd.min()),
                                      float(epred.matched_dose_smd.max())],
    published_matched_dose_smd_span=[float(matched.published_smd.min()),
                                     float(matched.published_smd.max())],
    robot_smd_published=float(matched.loc[matched.comparator.str.startswith("robot"),
                                         "published_smd"].iloc[0]),
    best_matched_dose_prediction=float(epred.matched_dose_smd.max()),
)
print(f"Validation: {summary['n_calibration_met']}/{summary['n_calibration_targets']} calibration "
      f"targets met; {summary['n_out_of_sample_passed']}/{summary['n_out_of_sample']} out-of-sample "
      f"benchmarks within range ({100*summary['out_of_sample_pass_rate']:.0f}%)")
for f in summary["out_of_sample_failures"]:
    print(f"   FAILS: {f['quantity']} - model {f['model']:.2f} vs published {f['published']:.2f}")

# ------------------------------------------------------- Part 2: global sensitivity (LHS + SRRC)
def lhs(n, bounds):
    k = len(bounds)
    u = (rng.permutation(np.arange(n))[:, None] + rng.random((n, k))) / n
    for j in range(1, k):
        u[:, j] = (rng.permutation(np.arange(n)) + rng.random(n)) / n
    return np.array([lo + u[:, j] * (hi - lo) for j, (lo, hi) in enumerate(bounds)]).T

# structural parameters that were NOT calibrated, with ranges spanning the verified uncertainty
CBF_N = G("cbf_normal_cortex_pet")
bounds = [
    (0.30, 0.80),      # collateral coupling strength in the perfusion model
    (0.35, 0.85),      # perfusion heterogeneity (lognormal sigma) within the territory
    (12.0, 22.0),      # penumbra lower threshold (verified range 12-22)
    (22.0, 50.0),      # oligemia ceiling (verified range 22-50)
    (0.44 / 0.44, 0.85 / 0.44),  # oxygen-extraction reserve (verified OEF 0.44 -> up to 0.85)
    (0.05, 0.80),      # corticospinal capacity dispersion in the recovery cohort
    (100.0, 250.0),    # latency at which the learning rate halves (verified fit +- range)
    (0.2, 1.0),        # dose exponent gamma
]
names = ["collateral_coupling", "perfusion_sigma", "penumbra_lower_cbf", "oligemia_ceiling",
         "oef_reserve", "capacity_dispersion", "learning_half_latency_ms", "dose_exponent"]
X = lhs(240, bounds)

DPAR = json.loads((K.RESULTS / "D_recovery_summary.json").read_text())["recovery_parameters"]
FM_MAX = 66.0

def drive(t, par, scale=1.0):
    return scale * (par["eta_fast"] * par["tau_fast"] * (1 - np.exp(-t / par["tau_fast"]))
                    + par["eta_slow"] * par["tau_slow"] * (1 - np.exp(-t / par["tau_slow"]))
                    + par["floor"] * t)

def run(x):
    coll_k, sig, pen_lo, olig, oef, cap_sd, tau_half, gamma = x
    # --- perfusion / tissue fate: median infarct growth rate over 3 h
    n = 600
    coll = rng.beta(2.2, 2.6, n)
    vr = rng.lognormal(np.log(50.0), 0.63, n)
    mu = (0.10 + coll_k * coll) * CBF_N * min(1.0, oef)
    F = rng.lognormal(np.log(np.maximum(mu, 0.5)), sig)
    # tolerance curve through the verified depth-duration points, with scanned thresholds
    tol = np.where(F <= pen_lo, 0.5, np.where(F >= olig, np.inf, 0.5 + 11.5 * (F - pen_lo) /
                                              max(olig - pen_lo, 1e-6)))
    dead3 = (tol <= 3.0).astype(float)
    igr = dead3 * vr / 3.0
    # --- recovery: mean FM gain at 26 weeks and the PRR slope
    fm0 = rng.beta(1.0, 1.0, 1500) * FM_MAX
    z = rng.normal(0, 1, 1500)
    cap = np.clip(0.9 + cap_sd * z, 0.02, None)
    y = (FM_MAX - fm0) * (1 - np.exp(-cap * drive(26.0, DPAR)))
    A_ = np.vstack([FM_MAX - fm0, np.ones(1500)]).T
    slope = float(np.linalg.lstsq(A_, y, rcond=None)[0][0])
    # --- device: matched-dose SMD at 20 ms with the scanned latency sensitivity and dose exponent
    a20 = 1.0 / (1.0 + 20.0 / tau_half); a60 = 1.0 / (1.0 + 60.0 / tau_half)
    lr = a20 / a60
    fm0c = np.clip(rng.normal(30.0, 9.0, 1500), 4, 58)
    capc = np.clip(rng.normal(0.9, 0.45, 1500), 0.02, None)
    d0 = drive(26.0, DPAR)
    def gain(scale):
        d1 = d0 + scale * (drive(34.6, DPAR) - d0)
        return ((FM_MAX - fm0c) * (np.exp(-capc * d0) - np.exp(-capc * d1))).mean()
    smd = (gain(lr * 4.0 ** gamma) - gain(4.0 ** gamma)) / np.sqrt((4.4 ** 2 + 3.8 ** 2) / 2)
    return dict(median_igr=float(np.median(igr[igr > 0]) if (igr > 0).any() else 0.0),
                pct_fast_progressors=float(100 * np.mean(igr >= 10)),
                prr_slope=slope, matched_dose_smd=float(smd),
                mean_fm_gain_26w=float(y.mean()))

out = pd.DataFrame([run(x) for x in X])
sens_rows = []
for target in out.columns:
    y = out[target].to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    R = np.array([pd.Series(X[:, j]).rank().to_numpy() for j in range(X.shape[1])]).T
    R = (R - R.mean(0)) / R.std(0)
    ry_s = (ry - ry.mean()) / ry.std()
    beta, *_ = np.linalg.lstsq(R, ry_s, rcond=None)
    r2 = 1 - float(((ry_s - R @ beta) ** 2).sum()) / float((ry_s ** 2).sum())
    for nm, bb in zip(names, beta):
        sens_rows.append(dict(output=target, parameter=nm, srrc=float(bb), model_r2=r2))
sens = pd.DataFrame(sens_rows)
K.save_table(sens, "V_global_sensitivity")
K.save_table(pd.concat([pd.DataFrame(X, columns=names), out], axis=1), "V_sensitivity_samples")

piv = sens.pivot(index="parameter", columns="output", values="srrc")
print("\nStandardised rank-regression coefficients (|SRRC| > 0.2 in bold in the figure):")
print(piv.round(2).to_string())

summary["sensitivity_r2_by_output"] = {t: float(sens.loc[sens.output == t, "model_r2"].iloc[0])
                                       for t in out.columns}
summary["dominant_parameter_by_output"] = {
    t: str(sens.loc[sens.output == t].iloc[sens.loc[sens.output == t].srrc.abs().argmax()].parameter)
    for t in out.columns}
K.save_json(summary, "V_validation_summary")
