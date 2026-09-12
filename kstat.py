#!/usr/bin/env python3
"""Statistical toolkit - the inference machinery the framework's claims are made with.

Written against numpy and scipy only, because the sandbox this work was produced in has no
network access and statsmodels could not be installed (see Methods). Every routine here is
therefore explicit, which is in any case preferable for a paper: the reader can see exactly what
the interval, the multiplicity correction and the sensitivity index are.

Contents
    bootstrap_ci        percentile and BCa bootstrap intervals for any statistic
    paired_bootstrap    interval for a difference between paired model and reference series
    benjamini_hochberg  false-discovery-rate control with adjusted p-values
    holm_bonferroni     familywise control, for the confirmatory subset
    profile_likelihood  one-dimensional profile with a chi-square interval
    sobol_indices       first-order and total-order variance indices (Saltelli estimators)
    metropolis          adaptive random-walk Metropolis with multiple chains
    rhat, ess           Gelman-Rubin split-Rhat and effective sample size
"""
from __future__ import annotations
import numpy as np
from scipy import stats

DEFAULT_B = 4000


# ------------------------------------------------------------------------------- intervals
def bootstrap_ci(x, stat=np.median, B=DEFAULT_B, alpha=0.05, seed=0, method="bca"):
    """Bootstrap confidence interval for `stat` of a one-dimensional sample.

    method="percentile" gives the plain percentile interval; "bca" applies the bias-correction
    and acceleration adjustment, which is what is reported in the paper because several of the
    statistics involved (medians, ratios, proportions of a skewed distribution) are biased.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return (np.nan, np.nan, np.nan)
    rg = np.random.default_rng(seed)
    idx = rg.integers(0, n, size=(B, n))
    boots = np.array([stat(x[i]) for i in idx])
    theta = float(stat(x))
    if method == "percentile":
        lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return theta, float(lo), float(hi)
    # BCa
    z0 = stats.norm.ppf(np.clip((boots < theta).mean(), 1e-6, 1 - 1e-6))
    jack = np.array([stat(np.delete(x, i)) for i in range(n)])
    jbar = jack.mean()
    num = ((jbar - jack) ** 3).sum()
    den = 6.0 * (((jbar - jack) ** 2).sum() ** 1.5)
    a = num / den if den > 0 else 0.0
    zl, zu = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
    def adj(z):
        return stats.norm.cdf(z0 + (z0 + z) / max(1 - a * (z0 + z), 1e-6))
    ql, qu = adj(zl), adj(zu)
    lo, hi = np.percentile(boots, [100 * np.clip(ql, 0, 1), 100 * np.clip(qu, 0, 1)])
    return theta, float(lo), float(hi)


def paired_bootstrap(a, b, stat=np.median, B=DEFAULT_B, alpha=0.05, seed=0):
    """Interval for stat(a) - stat(b) with the pairing preserved."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    n = a.size
    rg = np.random.default_rng(seed)
    idx = rg.integers(0, n, size=(B, n))
    d = np.array([stat(a[i]) - stat(b[i]) for i in idx])
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(a) - stat(b)), float(lo), float(hi)


def ratio_ci_from_ci(model, published, pub_lo, pub_hi, alpha=0.05):
    """Interval for the model-to-published ratio, propagating only the PUBLISHED uncertainty.
    Used for benchmark comparisons where the model value is a deterministic simulation output."""
    if not (np.isfinite(pub_lo) and np.isfinite(pub_hi)) or published <= 0:
        return (np.nan, np.nan)
    return (float(model / pub_hi), float(model / pub_lo))


# ------------------------------------------------------------------------------ multiplicity
def benjamini_hochberg(p, q=0.05):
    """Return (rejected, adjusted p-values) under Benjamini-Hochberg FDR control."""
    p = np.asarray(p, float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(m)
    out[order] = adj
    return out <= q, out


def holm_bonferroni(p, alpha=0.05):
    p = np.asarray(p, float)
    m = p.size
    order = np.argsort(p)
    adj_sorted = np.maximum.accumulate(p[order] * (m - np.arange(m)))
    adj_sorted = np.clip(adj_sorted, 0, 1)
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj <= alpha, adj


def equivalence_p(model, published, se):
    """Two-sided p-value for the null that the model equals the published value, given the
    published standard error. Used to place every benchmark comparison on a common p scale so
    that a multiplicity correction can be applied across the benchmark set."""
    if not np.isfinite(se) or se <= 0:
        return np.nan
    return float(2 * stats.norm.sf(abs(model - published) / se))


# ------------------------------------------------------------------------ profile likelihood
def profile_likelihood(nll, theta_hat, index, grid, args=()):
    """Profile a single parameter of a negative-log-likelihood.

    `nll(theta, *args)` must accept the full parameter vector. At each value on `grid` the
    remaining parameters are re-optimised by Nelder-Mead from the full-model optimum. Returns
    the profile deviance and the chi-square(1) interval, which is what identifiability means in
    practice: a parameter whose profile is flat has no interval.
    """
    from scipy.optimize import minimize
    theta_hat = np.asarray(theta_hat, float)
    base = float(nll(theta_hat, *args))
    prof = np.empty(len(grid))
    for k, v in enumerate(grid):
        free = np.delete(np.arange(theta_hat.size), index)
        def f(z):
            th = theta_hat.copy()
            th[index] = v
            th[free] = z
            return nll(th, *args)
        if free.size:
            r = minimize(f, theta_hat[free], method="Nelder-Mead",
                         options=dict(maxiter=2000, xatol=1e-4, fatol=1e-4))
            prof[k] = r.fun
        else:
            prof[k] = f(np.array([]))
    dev = 2.0 * (prof - base)
    crit = stats.chi2.ppf(0.95, 1)
    inside = dev <= crit
    if inside.any():
        lo, hi = float(np.min(np.asarray(grid)[inside])), float(np.max(np.asarray(grid)[inside]))
    else:
        lo = hi = np.nan
    identified = bool(inside.any() and not (inside[0] and inside[-1]))
    return dict(grid=np.asarray(grid, float), deviance=dev, ci=(lo, hi),
                identified=identified, base_nll=base)


# --------------------------------------------------------------------------- sensitivity
def sobol_indices(f, bounds, n=1024, seed=0, second_order=False):
    """First-order and total-order Sobol indices by the Saltelli estimators.

    `f` must accept an (m, d) array of parameter rows and return an (m,) array of outputs, so
    that the whole design is evaluated in one vectorised call where possible. Sampling is plain
    Monte Carlo with a scrambled Sobol sequence when scipy provides one, which it does.
    """
    bounds = np.asarray(bounds, float)
    d = bounds.shape[0]
    try:
        eng = stats.qmc.Sobol(d=2 * d, scramble=True, seed=seed)
        u = eng.random(n)
    except Exception:
        u = np.random.default_rng(seed).random((n, 2 * d))
    lo, hi = bounds[:, 0], bounds[:, 1]
    A = lo + (hi - lo) * u[:, :d]
    B = lo + (hi - lo) * u[:, d:]
    yA, yB = np.asarray(f(A), float), np.asarray(f(B), float)
    varY = np.var(np.concatenate([yA, yB]), ddof=1)
    scale = max(abs(np.mean(np.concatenate([yA, yB]))), 1e-12)
    if not np.isfinite(varY) or np.sqrt(varY) / scale < 1e-3:
        # a design over which the output barely moves cannot support a variance decomposition;
        # returning near-zero denominators would produce indices outside [0, 1]
        return dict(S1=np.full(d, np.nan), ST=np.full(d, np.nan), variance=float(varY), n=n,
                    degenerate=True)
    S1, ST = np.zeros(d), np.zeros(d)
    for i in range(d):
        AB = A.copy()
        AB[:, i] = B[:, i]
        yAB = np.asarray(f(AB), float)
        # Saltelli 2010 estimators
        S1[i] = np.mean(yB * (yAB - yA)) / varY
        ST[i] = 0.5 * np.mean((yA - yAB) ** 2) / varY
    return dict(S1=S1, ST=ST, variance=float(varY), n=n, degenerate=False)


# --------------------------------------------------------------------------------- sampling
def metropolis(log_post, start, scale, n_draws=20000, n_chains=4, burn=None, seed=0,
               target_accept=0.30, adapt_until=None):
    """Adaptive random-walk Metropolis. Returns draws of shape (chain, draw, parameter).

    The proposal covariance is diagonal and each component's scale is tuned during the burn-in
    towards the target acceptance rate, then frozen, so the retained draws come from a fixed
    kernel and the usual convergence diagnostics apply.
    """
    start = np.atleast_2d(np.asarray(start, float))
    d = start.shape[-1]
    if start.shape[0] == 1:
        start = np.repeat(start, n_chains, axis=0)
    burn = burn if burn is not None else n_draws // 2
    adapt_until = adapt_until if adapt_until is not None else burn
    rg = np.random.default_rng(seed)
    sc = np.tile(np.asarray(scale, float), (n_chains, 1))
    cur = start.copy()
    lp = np.array([log_post(c) for c in cur])
    keep = np.empty((n_chains, n_draws - burn, d))
    acc = np.zeros((n_chains, d))
    tried = np.zeros((n_chains, d))
    for it in range(n_draws):
        for j in range(d):
            prop = cur.copy()
            prop[:, j] += sc[:, j] * rg.standard_normal(n_chains)
            lpp = np.array([log_post(p) for p in prop])
            u = np.log(rg.random(n_chains))
            take = (lpp - lp) > u
            cur[take] = prop[take]
            lp[take] = lpp[take]
            acc[:, j] += take
            tried[:, j] += 1
            if it < adapt_until and tried[0, j] >= 25:
                rate = acc[:, j] / tried[:, j]
                sc[:, j] *= np.exp(np.clip(rate - target_accept, -0.3, 0.3))
                acc[:, j] = 0.0
                tried[:, j] = 0.0
        if it >= burn:
            keep[:, it - burn] = cur
    return keep, sc


def rhat(draws):
    """Split-Rhat per parameter for draws of shape (chain, draw, parameter)."""
    c, n, d = draws.shape
    half = n // 2
    s = np.concatenate([draws[:, :half], draws[:, half:2 * half]], axis=0)
    m, nn = s.shape[0], s.shape[1]
    means = s.mean(axis=1)
    variances = s.var(axis=1, ddof=1)
    W = variances.mean(axis=0)
    B = nn * means.var(axis=0, ddof=1)
    var_hat = ((nn - 1) / nn) * W + B / nn
    return np.sqrt(np.where(W > 0, var_hat / W, np.nan))


def ess(draws, max_lag=200):
    """Effective sample size per parameter, from the summed positive autocorrelations."""
    c, n, d = draws.shape
    out = np.empty(d)
    for j in range(d):
        rho_sum = 0.0
        for k in range(c):
            x = draws[k, :, j] - draws[k, :, j].mean()
            v = np.dot(x, x) / n
            if v <= 0:
                continue
            rho = np.array([np.dot(x[:-l], x[l:]) / (n * v) for l in range(1, min(max_lag, n - 2))])
            stop = np.argmax(rho < 0.05) if (rho < 0.05).any() else rho.size
            rho_sum += rho[:stop].sum()
        out[j] = c * n / max(1.0 + 2.0 * rho_sum / c, 1e-6)
    return out


def summarise(draws, names):
    import pandas as pd
    flat = draws.reshape(-1, draws.shape[-1])
    r, e = rhat(draws), ess(draws)
    return pd.DataFrame(dict(
        parameter=names,
        posterior_mean=flat.mean(axis=0),
        posterior_sd=flat.std(axis=0, ddof=1),
        q025=np.percentile(flat, 2.5, axis=0),
        median=np.percentile(flat, 50, axis=0),
        q975=np.percentile(flat, 97.5, axis=0),
        rhat=r, ess=e))
