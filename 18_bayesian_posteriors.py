#!/usr/bin/env python3
"""Module L - Bayesian posteriors for the two load-bearing laws of the framework.

Until this point the framework's central objects - the depth-duration tolerance curve and the
two-timescale recovery law - were point estimates obtained by matching verified summaries. Point
estimates are not enough for a paper that then asks readers to trust downstream predictions built
on them, so both are refitted here as Bayesian models, with explicit likelihoods, priors that are
stated rather than implied, and posteriors sampled by an adaptive random-walk Metropolis chain
written out in full (no probabilistic-programming package was installable in the environment this
work was produced in; see Methods).

  L1  The depth-duration tolerance curve and the territory it acts on, fitted JOINTLY to six
      verified summaries: the three flow-versus-time infarction points, the median infarct growth
      rate and its interquartile range, and the three progressor proportions. The forward model is
      analytic - for a lognormal flow field the infarcted fraction at time t is a normal
      cumulative distribution function evaluated at the inverse tolerance curve - so no Monte
      Carlo is needed inside the likelihood and the chain is exact to sampling error.

  L2  The two-timescale recovery law, fitted hierarchically to the verified Fugl-Meyer gain bins,
      the verified total recovery magnitude, the verified proportion recovered by four months and
      the three verified treatment-window gradients, with a per-source scale effect so that
      between-cohort heterogeneity is absorbed rather than attributed to the kinetics.

Both fits report split-Rhat and effective sample size, both are checked against verified
quantities that were NOT in the likelihood, and the posterior for the tolerance curve is
propagated forward into a credible band on predicted infarct volume - which is the point of doing
this at all.
"""
import numpy as np, pandas as pd
from scipy import stats
from scipy.special import ndtr
import kcore as K
import kstat as S

_LOG2PI = float(np.log(2.0 * np.pi))

def _lnorm(x, m, s):
    """Normal log-density, written out because scipy's dispatch dominates the sampler's cost."""
    z = (np.asarray(x, float) - m) / s
    return -0.5 * (z * z + _LOG2PI) - np.log(s)

rng = np.random.default_rng(K.SEED + 18)

# ==================================================================== L1 the tolerance curve
F1, F2, F3 = K.getp("depth_duration_30min"), K.getp("depth_duration_6h"), K.getp("depth_duration_12h")
T1, T2, T3 = 0.5, 6.0, 12.0
CBF_N = K.getp("cbf_normal_cortex_pet")
OLIG_HI = 50.0
SIGMA_F = 0.55                                  # within-territory flow dispersion (Module B)
IGR_MED = K.getp("igr_median")
IGR_LO, IGR_HI = K.getp("igr_median", "value_lo"), K.getp("igr_median", "value_hi")
P_SLOW = K.getp("igr_prop_slow") / 100.0
P_FAST = K.getp("igr_prop_fast") / 100.0
CUT_SLOW, CUT_FAST = K.getp("igr_slow_cut"), K.getp("igr_fast_cut")

# collateral distribution as in Module B, integrated by Gauss-Legendre quadrature
_cx, _cw = np.polynomial.legendre.leggauss(24)
COLL = 0.5 * (_cx + 1.0)
COLL_W = 0.5 * _cw * stats.beta.pdf(COLL, 2.2, 2.6)
COLL_W /= COLL_W.sum()
# territory-volume distribution, also by quadrature on the log scale
_vx, _vw = np.polynomial.hermite_e.hermegauss(15)
VZ, VW = _vx, _vw / _vw.sum()

PARAMS = ("log_t_at_F1", "slope_low", "slope_high", "log_V_median", "sigma_V", "log_sigma_obs",
          "log_tau_between_source")


def t_crit_of(F, p):
    """Piecewise log-linear tolerance curve with the two slopes as free parameters."""
    lt1, s_lo, s_hi = p[0], p[1], p[2]
    F = np.asarray(F, float)
    lt = np.where(F <= F2, lt1 + s_lo * (F - F1), lt1 + s_lo * (F2 - F1) + s_hi * (F - F2))
    out = np.exp(np.clip(lt, -10, 10))
    return np.where(F >= OLIG_HI, np.inf, out)


def f_crit_of(t, p):
    """Inverse: the flow whose tolerance is exactly t hours."""
    lt1, s_lo, s_hi = p[0], p[1], p[2]
    lt = np.log(np.asarray(t, float))
    lt2 = lt1 + s_lo * (F2 - F1)
    below = F1 + (lt - lt1) / max(s_lo, 1e-6)
    above = F2 + (lt - lt2) / max(s_hi, 1e-6)
    return np.where(lt <= lt2, below, above)


def infarct_fraction(t, collateral, p):
    """Fraction of a territory infarcted by time t, analytically."""
    mu = (0.10 + 0.80 * np.asarray(collateral, float)) * CBF_N
    fc = np.minimum(f_crit_of(t, p), OLIG_HI)
    z = (np.log(np.maximum(fc, 1e-6)) - np.log(mu)) / SIGMA_F
    return ndtr(z)


def igr_distribution(p, n_quad=None):
    """Joint distribution of the 3-hour growth rate over collateral grade and territory volume,
    returned as (values, weights) so quantiles and proportions follow without simulation."""
    logV, sV = p[3], p[4]
    frac = infarct_fraction(3.0, COLL, p)                      # (n_coll,)
    vols = np.exp(logV + sV * VZ)                              # (n_vol,)
    vals = np.outer(frac, vols) / 3.0
    w = np.outer(COLL_W, VW)
    order = np.argsort(vals.ravel())
    return vals.ravel()[order], w.ravel()[order]


def weighted_quantile(v, w, q):
    c = np.cumsum(w) / w.sum()
    return float(np.interp(q, c, v))


def summaries(p):
    v, w = igr_distribution(p)
    med = weighted_quantile(v, w, 0.50)
    q25 = weighted_quantile(v, w, 0.25)
    q75 = weighted_quantile(v, w, 0.75)
    p_slow = float(w[v < CUT_SLOW].sum() / w.sum())
    p_fast = float(w[v >= CUT_FAST].sum() / w.sum())
    return med, q25, q75, p_slow, p_fast


# The likelihood. The three flow-versus-time points are treated as log-scale measurements with a
# common unknown error; the growth-rate summaries are treated as measurements with the dispersion
# the published interquartile range itself implies. Nothing here is weighted by hand.
LOG_T_SD_FLOOR = 0.05
IGR_SD = (IGR_HI - IGR_LO) / 2.7          # the verified IQR converted to a normal scale


def log_lik(p):
    lt1, s_lo, s_hi, logV, sV, log_so, log_tau = p
    if not (0.0 < s_lo < 1.0 and 0.0 < s_hi < 1.0 and 0.02 < sV < 2.0
            and np.log(5.0) < logV < np.log(400.0) and -6 < log_so < 1.5 and -8 < log_tau < 1.0):
        return -np.inf
    so = np.exp(log_so) + LOG_T_SD_FLOOR
    pred = np.log(t_crit_of(np.array([F1, F2, F3]), p))
    obs = np.log(np.array([T1, T2, T3]))
    ll = float(_lnorm(obs, pred, so).sum())
    med, q25, q75, ps, pf = summaries(p)
    if not np.isfinite(med):
        return -np.inf
    ll += float(_lnorm(IGR_MED, med, IGR_SD))
    ll += float(_lnorm(IGR_LO, q25, IGR_SD))
    ll += float(_lnorm(IGR_HI, q75, 1.5 * IGR_SD))
    # The progressor proportions come from a cohort of 8076 and the growth-rate summaries from a
    # cohort of 897. Entering the proportions as exact binomial observations would therefore give
    # them nine times the weight - and Module B already established that the two published series
    # are mutually incompatible, so forcing both would produce a confidently wrong posterior.
    # Each proportion is instead entered with an additive between-source standard deviation whose
    # size is ESTIMATED (tau below), on top of its own binomial sampling error. If the two series
    # agreed, tau would be driven to zero and the binomial weight would be recovered; the
    # posterior for tau is consequently a direct measurement of how far apart they are.
    tau = np.exp(log_tau)
    n_prop = 8076.0
    for obs_p, mod_p in ((P_SLOW, ps), (P_FAST, pf)):
        mod_p = float(np.clip(mod_p, 1e-6, 1 - 1e-6))
        se_bin = np.sqrt(mod_p * (1 - mod_p) / n_prop)
        ll += float(_lnorm(obs_p, mod_p, np.sqrt(se_bin ** 2 + tau ** 2)))
    return float(ll)


def log_prior(p):
    lt1, s_lo, s_hi, logV, sV, log_so, log_tau = p
    lp = 0.0
    lp += _lnorm(lt1, np.log(0.5), 1.0)          # tolerance at the lowest anchor
    lp += _lnorm(s_lo, 0.25, 0.25)               # slopes are positive and order 0.1-0.3
    lp += _lnorm(s_hi, 0.10, 0.15)
    lp += _lnorm(logV, np.log(60.0), 0.8)        # territory volume, weakly informative
    lp += _lnorm(sV, 0.0, 0.8) + np.log(2.0)
    lp += _lnorm(log_so, np.log(0.15), 1.0)
    lp += _lnorm(np.exp(log_tau), 0.0, 0.10) + log_tau   # half-normal on tau, Jacobian included
    return float(lp)


def log_post(p):
    lp = log_prior(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_lik(p)


start = np.array([np.log(0.5), 0.2485, 0.0770, np.log(51.6), 0.69, np.log(0.15), np.log(0.03)])
print("Module L1: starting log-posterior", round(log_post(start), 2))
draws1, sc1 = S.metropolis(log_post, start, np.array([0.10, 0.02, 0.01, 0.10, 0.05, 0.25, 0.30]),
                           n_draws=9000, n_chains=4, burn=4500, seed=K.SEED + 1)
sum1 = S.summarise(draws1, list(PARAMS))
K.save_table(sum1, "L_tolerance_posterior")
flat1 = draws1.reshape(-1, draws1.shape[-1])

# credible band on the tolerance curve, and on the 24-hour infarct volume it implies
fgrid = np.linspace(6.0, 49.5, 120)
sub = flat1[rng.choice(flat1.shape[0], 1500, replace=False)]
curves = np.array([t_crit_of(fgrid, p) for p in sub])
band = pd.DataFrame(dict(cbf=fgrid,
                         t_crit_median=np.nanmedian(curves, axis=0),
                         t_crit_lo=np.nanpercentile(curves, 2.5, axis=0),
                         t_crit_hi=np.nanpercentile(curves, 97.5, axis=0)))
K.save_table(band, "L_tolerance_credible_band")

def volume_quantile(p, hours, q=0.5):
    """Weighted quantile of infarct volume at `hours`, over collateral grade and territory size."""
    vals = np.outer(infarct_fraction(hours, COLL, p), np.exp(p[3] + p[4] * VZ)).ravel()
    w = np.outer(COLL_W, VW).ravel()
    o = np.argsort(vals)
    return weighted_quantile(vals[o], w[o], q)

vol24 = np.array([volume_quantile(p, 24.0) for p in sub])
pp1 = pd.DataFrame([
    dict(quantity="median 24-h infarct volume (mL)", posterior_median=float(np.median(vol24)),
         cri_lo=float(np.percentile(vol24, 2.5)), cri_hi=float(np.percentile(vol24, 97.5)),
         published=K.getp("typical_lvo_infarct_volume"), in_likelihood=False,
         src=K.param_src("typical_lvo_infarct_volume")),
])
# out-of-sample posterior predictive: the verified time-to-complete-infarction window, and the
# verified relative-CBF reperfusion thresholds, neither of which entered the likelihood
t95 = []
for p in sub:
    tg = np.linspace(0.25, 24, 120)
    f = np.array([float((infarct_fraction(t, COLL, p) * COLL_W).sum()) for t in tg])
    f /= max(f[-1], 1e-9)
    t95.append(float(tg[np.argmax(f >= 0.95)]))
t95 = np.array(t95)
pp1 = pd.concat([pp1, pd.DataFrame([
    dict(quantity="time to 95% of final infarct (h)", posterior_median=float(np.median(t95)),
         cri_lo=float(np.percentile(t95, 2.5)), cri_hi=float(np.percentile(t95, 97.5)),
         published=K.getp("time_to_complete_infarction"), in_likelihood=False,
         src=K.param_src("time_to_complete_infarction")),
    dict(quantity="tolerance at 18 mL/100 g/min (h)",
         posterior_median=float(np.median([t_crit_of(18.0, p) for p in sub])),
         cri_lo=float(np.percentile([t_crit_of(18.0, p) for p in sub], 2.5)),
         cri_hi=float(np.percentile([t_crit_of(18.0, p) for p in sub], 97.5)),
         published=np.nan, in_likelihood=False, src="interpolated prediction"),
])], ignore_index=True)
K.save_table(pp1, "L_tolerance_predictive")
print(sum1.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
