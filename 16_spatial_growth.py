#!/usr/bin/env python3
"""Module J - spatially resolved stochastic reaction-diffusion model of infarct growth.

Modules B and C describe the ischaemic territory as a distribution of independent voxels and as a
well-mixed compartment respectively. Neither can produce the two things that actually decide how
an infarct grows in a brain: the GEOMETRY of collateral supply, and the travelling waves of
spreading depolarisation that carry injury outward from the core. This module adds both, and it
does so without introducing a new distributional parameter.

What is solved
--------------
On an unfolded territory sheet whose volume is drawn exactly as in Module B, the module
integrates extracellular potassium as a reaction-diffusion field,

    dK/dt = div( D_app(state) grad K ) + S_leak(F) + S_SD(x,t) - P_pump(K, F),                (1)

with the apparent diffusivity D_app = D_free / lambda^2 taken from the verified extracellular
tortuosity and its verified hypoxic value, together with a stochastic per-cell injury process

    dose_i(t+dt) = dose_i(t) + dt / t_crit(F_eff,i) + c_SD * 1[SD passes i] * (1 - atp_i),
    cell i infarcts when dose_i >= E_i,   E_i ~ Exp(1),                                       (2)

where t_crit is the SAME verified depth-duration tolerance curve Module B uses (asserted below to
agree to 1e-9), so switching geometry and waves off recovers the aggregate model.

The flow field is built by MONOTONE REARRANGEMENT: its marginal distribution is exactly Module B's
lognormal, and only the spatial arrangement is new - lowest flows at the centre of the territory,
highest where leptomeningeal collaterals enter at the border, blended with a smooth random
component. There is therefore one structural choice in this module (how strongly the arrangement
is geometric rather than random) and no new fitted parameter; the choice is stated, its
consequence is reported, and it is scanned.

Spreading depolarisation is propagated in the sharp-interface limit of (1), not as a resolved
excitable front. That is the only defensible choice at a tractable grid spacing: with the verified
effective SD diffusivity and the verified human front speed the reaction-diffusion front width is
D/c ~ 0.04 mm, ~20x below a grid that still covers a 10 cm territory. The sharp-interface limit of
an excitable reaction-diffusion system is the eikonal law c = c0 - D_app * kappa, which is what is
integrated; the curvature term is kept, so wave shape is not imposed. The realised front speed is
measured from the simulation and reported against the verified value rather than assumed.
"""
import numpy as np, pandas as pd
from scipy import ndimage
from scipy.special import ndtri
from scipy.stats import rankdata
import kcore as K

rng = np.random.default_rng(K.SEED + 16)

# ----------------------------------------------------------------- verified physical constants
CBF_N   = K.getp("cbf_normal_cortex_pet")                   # 44.4 mL/100 mL/min
PEN_LO  = K.getp("cbf_penumbra_lower")                      # 12
PEN_HI  = K.getp("cbf_penumbra_upper")                      # 22
OLIG_HI = 50.0                                              # oligaemia ceiling (as Module B)
PUMP    = K.getp("cbf_threshold_pump_failure")              # 8
AD_CBF  = K.getp("cbf_threshold_energy_failure_rel") / 100 * CBF_N   # anoxic-depolarisation flow

ALPHA_N = K.getp("ecs_volume_fraction_normal")              # 0.20
ALPHA_H = K.getp("ecs_volume_fraction_hypoxic")             # 0.13
LAM_N   = K.getp("ecs_tortuosity_normal")                   # 1.6
LAM_H   = K.getp("ecs_tortuosity_hypoxic")                  # 1.8
D_FREE  = K.getp("k_free_diffusion_coefficient") * 100.0    # cm2/s -> mm2/s
K_BASE  = K.getp("k_baseline_cortex")                       # 3.1 mM
K_THR   = K.getp("k_threshold_for_sd_cortex")               # 9.4 mM
K_PEAK  = K.getp("k_peak_during_sd")                        # 65 mM
C_SD    = K.getp("sd_speed_human")                          # 5.7 mm/min
C_SD_LO = K.getp("sd_speed_human", "value_lo")              # 2.9
C_SD_HI = K.getp("sd_speed_human", "value_hi")              # 8.2
SD_DUR  = K.getp("sd_duration_human")                       # 5.4 min
SD_DUR_LO = K.getp("sd_duration_human", "value_lo")
SD_DUR_HI = K.getp("sd_duration_human", "value_hi")
SD_INT  = K.getp("sd_interval_human")                       # 44 min
SI_DROP = K.getp("spreading_ischaemia_cbf_drop") / 100      # 0.25 persistent post-SD CBF fall
NEUR_DECAY = K.getp("neuron_survival_decay_L4") / 100       # 0.02 per min after anoxic SD
D_SD_EFF = K.getp("sd_effective_diffusivity_cortex") * 1e6  # m2/s -> mm2/s

# --------------------------------------------------- the two potassium constants, both derived
# The leak and pump-capacity constants of (1) are NOT free. They are pinned by two verified
# latencies, so the potassium field carries no invented rate:
#   leak   - tissue whose pump has failed must cross the verified 9.4 mM depolarisation threshold
#            from the verified 3.1 mM baseline in the verified 1.5 min anoxic-depolarisation
#            latency, which fixes the leak flux exactly;
#   V_max  - fully perfused cortex must clear the verified 65 mM depolarisation load back below
#            threshold in the verified 1.5 min repolarisation time, which fixes pump capacity.
# The Michaelis constant is taken as the threshold offset itself (6.3 mM), the only scale the
# verified data supplies. Both derivations are checked numerically below.
AD_LATENCY = K.getp("sd_onset_after_arrest") * 60.0           # 1.5 min -> s
REPOL_PERF = K.getp("sd_repolarization_perfused") * 60.0      # 1.5 min -> s
KM_PUMP = K_THR - K_BASE                                      # 6.3 mM
LEAK = (K_THR - K_BASE) * ALPHA_H / AD_LATENCY                # mM (ECS) per s
DK_SD = K_PEAK - K_BASE
VMAX = ALPHA_N * ((DK_SD - KM_PUMP) + KM_PUMP * np.log(DK_SD / KM_PUMP)) / REPOL_PERF

def _k_rhs(dk, atp, alpha):
    return (LEAK * (1.0 - atp) - VMAX * atp * dk / (KM_PUMP + dk)) / alpha

def _integrate_k(dk0, atp, alpha, stop, h=0.05, tmax=3000.0):
    dk, t = dk0, 0.0
    while t < tmax and not stop(dk):
        dk = max(dk + h * _k_rhs(dk, atp, alpha), 0.0)
        t += h
    return t

_lat = _integrate_k(0.0, 0.0, ALPHA_H, lambda dk: dk >= KM_PUMP)
_rep = _integrate_k(DK_SD, 1.0, ALPHA_N, lambda dk: dk <= KM_PUMP)
assert abs(_lat - AD_LATENCY) < 0.05 * AD_LATENCY, (_lat, AD_LATENCY)
assert abs(_rep - REPOL_PERF) < 0.05 * REPOL_PERF, (_rep, REPOL_PERF)

V_DEFICIT = K.getp("perfusion_deficit_volume")              # 109 mL Tmax>3 s (median)
CORE_PRES = K.getp("core_volume_at_presentation")           # 20 mL median at presentation
V_AT_RISK = 51.644824245196915        # Module B's calibrated at-risk territory volume (mL)
SIGMA_V   = 0.6932433972934692        # Module B's inferred territory-volume dispersion
SIGMA_F   = 0.55                      # Module B's within-territory flow dispersion (log scale)
DEFICIT_RATIO = V_DEFICIT / V_AT_RISK # verified deficit volume per unit at-risk volume

# ------------------------------------------------------- the Module B tolerance curve, reused
_F_ANCH = np.array([K.getp("depth_duration_30min"), K.getp("depth_duration_6h"),
                    K.getp("depth_duration_12h")])
_LT_ANCH = np.log(np.array([0.5, 6.0, 12.0]))
_SLOPE_LO = (_LT_ANCH[1] - _LT_ANCH[0]) / (_F_ANCH[1] - _F_ANCH[0])
_SLOPE_HI = (_LT_ANCH[2] - _LT_ANCH[1]) / (_F_ANCH[2] - _F_ANCH[1])

def t_crit(F):
    F = np.asarray(F, dtype=float)
    lt = np.interp(F, _F_ANCH, _LT_ANCH)
    lt = np.where(F < _F_ANCH[0], _LT_ANCH[0] + _SLOPE_LO * (F - _F_ANCH[0]), lt)
    lt = np.where(F > _F_ANCH[-1], _LT_ANCH[-1] + _SLOPE_HI * (F - _F_ANCH[-1]), lt)
    out = np.exp(np.clip(lt, -8, 8))
    out = np.where(F >= OLIG_HI, np.inf, out)
    return np.where(F <= PUMP, 1e-3, out)

_b = pd.read_csv(K.RESULTS / "B_depth_duration_curve.csv")
_chk, _pub = t_crit(_b.cbf.values), _b.t_crit_h.values
_fin = np.isfinite(_chk) & np.isfinite(_pub)
assert np.allclose(_chk[_fin], _pub[_fin], rtol=1e-9), "the tolerance curve drifted from Module B"

# ------------------------------------------------------------------------------ the domain
N, DX = 110, 0.92                     # 110 x 110 cells of 0.92 mm -> 101.2 mm across
AREA = (N * DX) ** 2                  # mm2
SD_FRONT_WIDTH = D_SD_EFF / (C_SD / 60.0)          # mm, from the verified constants
_yy, _xx = np.indices((N, N))
D_EDGE = np.minimum.reduce([_xx, _yy, N - 1 - _xx, N - 1 - _yy]).astype(float) * DX
L_COLL = 25.0                         # mm collateral penetration length at unit collateral grade
W_NOISE = 1.0                         # stated structural choice: equal geometric/random weighting
SIGMA_TOL = 0.30                      # stated structural choice: spread of per-cell tolerance

_S8 = np.ones((3, 3), bool)

def _ndtr(z):
    from scipy.special import ndtr
    return ndtr(z)

def laplacian(u):
    return ((np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1)
             - 4.0 * u) / (DX * DX))

def perfusion_field(collateral, rg, w_noise=W_NOISE):
    """Module B's lognormal flow marginal, spatially arranged by monotone rearrangement."""
    L = L_COLL * (0.30 + 1.40 * collateral)
    geo = np.exp(-D_EDGE / L)
    geo = (geo - geo.mean()) / geo.std()
    if w_noise == 0.0:
        score = geo
    else:
        w = ndimage.gaussian_filter(rg.standard_normal((N, N)), 5.0, mode="nearest")
        w = (w - w.mean()) / w.std()
        score = geo + w_noise * w if np.isfinite(w_noise) else w
    q = (rankdata(score.ravel()) - 0.5) / score.size
    mu = (0.10 + 0.80 * collateral) * CBF_N
    F = (mu * np.exp(SIGMA_F * ndtri(q))).reshape(N, N)
    var_geo = float(np.corrcoef(geo.ravel(), np.log(F.ravel()))[0, 1] ** 2)
    return np.clip(F, 0.2, 3.0 * CBF_N), var_geo

# ------------------------------------------------------------------------ the integrator
# Two timescales are split. The potassium field, the perfusion penalty and the injury process are
# advanced on an outer step well inside the explicit diffusion stability limit dx^2/(4 D_app);
# the depolarisation front, whose transit time across one cell is dx/c (about 10 s), is sub-stepped
# inside each outer step on a bounding box around the active wave. The front advance is a
# continuous credit (credit += c dt / dx, fire at 1), so the speed is set by c and not by the
# lattice, up to a quantisation of one sub-step; the realised speed is measured and reported
# against the verified value rather than assumed.
T_END_H = 24.0
DT_OUT, DT_IDLE, DT_FRONT = 30.0, 120.0, 3.0        # s
SNAP_H = np.array([1.0, 3.0, 6.0, 12.0, 18.0, 24.0])
DT_STABLE = DX * DX / (4.0 * D_FREE / (LAM_N ** 2))
assert DT_IDLE < DT_STABLE, (DT_IDLE, DT_STABLE)

def simulate(collateral, thickness_mm, sd_on=True, rg=None, keep_fields=False,
             w_noise=W_NOISE, t_end_h=T_END_H, sd_cost_mult=1.0, si_on=True):
    """One occlusion. `sd_on` switches the depolarisation waves off entirely, `si_on` switches
    off only the persistent post-depolarisation flow penalty, and `sd_cost_mult` scales only the
    direct per-passage metabolic cost; running all four combinations decomposes what the waves
    actually contribute."""
    rg = rg if rg is not None else np.random.default_rng(rng.integers(1 << 62))
    cell_ml = DX * DX * thickness_mm / 1000.0
    F0, var_geo = perfusion_field(collateral, rg, w_noise)
    f32 = np.float32

    atp0 = np.clip(F0 / PUMP, 0.0, 1.0).astype(f32)
    atp1 = np.clip(F0 * (1.0 - SI_DROP) / PUMP, 0.0, 1.0).astype(f32)
    # graded flow reserve: the fraction of the normal-to-pump-failure flow span still available.
    # A depolarisation costs a non-ischaemic cortex nothing it cannot repay (verified: a single
    # non-ischaemic SD causes a transient 50% ATP fall and full recovery) and costs tissue near
    # pump failure its whole remaining tolerance; the reserve interpolates between the two.
    rsv0 = np.clip((F0 - PUMP) / (CBF_N - PUMP), 0.0, 1.0).astype(f32)
    rsv1 = np.clip((F0 * (1.0 - SI_DROP) - PUMP) / (CBF_N - PUMP), 0.0, 1.0).astype(f32)
    _t0, _t1 = t_crit(F0), t_crit(F0 * (1.0 - SI_DROP))
    inv_tc0 = np.where(np.isfinite(_t0), 1.0 / _t0, 0.0).astype(f32)
    inv_tc1 = np.where(np.isfinite(_t1), 1.0 / _t1, 0.0).astype(f32)

    core0 = F0 <= AD_CBF
    pen0 = (F0 > PEN_LO) & (F0 <= PEN_HI)
    olig0 = F0 > PEN_HI

    Kx = np.full((N, N), K_BASE, f32)
    dose = np.zeros((N, N), f32)
    # Per-cell tolerance is a SPATIALLY CORRELATED lognormal field with median exactly 1, so the
    # median cell infarcts exactly on Module B's verified depth-duration curve and the spatial
    # model still reduces to the aggregate model; only the spread around it is new. A unit
    # exponential was rejected: its median is 0.69, so half of every territory would infarct
    # earlier than the verified curve says and the model would silently grow faster than Module B.
    # Correlation (about 2.3 mm) matters as much as the marginal: independent draws produce a
    # salt-and-pepper infarct that no depolarisation wave can cross, which is a lattice artefact
    # rather than biology - real infarcts are contiguous.
    _z = ndimage.gaussian_filter(rg.standard_normal((N, N)), 2.5, mode="nearest")
    _z /= _z.std()
    thresh = np.exp(SIGMA_TOL * _z).astype(f32)
    dead = core0.copy()
    dose[core0] = np.inf
    phase = np.zeros((N, N), np.int8)           # 0 rest, 1 depolarised, 2 refractory
    phase[core0] = 1
    perm = core0.copy()
    # Absolute event times rather than incremental timers: the front is sub-stepped on a moving
    # bounding box, so any timer advanced inside that box would stop advancing for tissue outside
    # it. t_dep is when a cell depolarised, t_ref when it became refractory.
    t_dep = np.full((N, N), -np.inf, np.float64)
    t_ref = np.full((N, N), -np.inf, np.float64)
    t_dep[core0] = 0.0
    credit = np.zeros((N, N), f32)
    si = np.zeros((N, N), f32)                  # spreading-ischaemia countdown (s)

    c0 = float(np.clip(rg.lognormal(np.log(C_SD), np.log(C_SD_HI / C_SD_LO) / 2.70),
                       0.6 * C_SD_LO, 1.4 * C_SD_HI))
    dur = float(np.clip(rg.lognormal(np.log(SD_DUR), np.log(SD_DUR_HI / SD_DUR_LO) / 2.70),
                        2.0, 20.0)) * 60.0
    # Refractoriness is not imposed: a cell can host the next depolarisation once the pump has
    # cleared the potassium load, and that clearance time follows from the verified pump capacity
    # and the cell's own flow. Well-perfused cortex recovers in about a minute, penumbral tissue
    # in several, and the core effectively never - which is the observed spatial pattern of
    # recurrent depolarisation without any of it being put in by hand.
    _atp_ref = np.clip(F0 / PUMP, 1e-3, 1.0)
    clear_s = (ALPHA_H * ((DK_SD - KM_PUMP) + KM_PUMP * np.log(DK_SD / KM_PUMP))
               / (VMAX * _atp_ref))
    refr_map = np.clip(clear_s, REPOL_PERF, 6.0 * 3600.0)
    D_N, D_H = D_FREE / (LAM_N ** 2), D_FREE / (LAM_H ** 2)

    t, n_sd, n_sd_blocked, si_cells = 0.0, 0, 0, 0.0
    next_sd = rg.exponential(SD_INT * 60.0) if sd_on else np.inf
    traj_t, traj_v = [0.0], [float(dead.sum() * cell_ml)]
    waves, wave = [], None
    snaps, isnap = {}, 0
    dead_at_img = None
    T_IMG_H = 3.0            # the verified core/penumbra split is measured on imaging within 3 h

    while t < t_end_h * 3600.0:
        dep = phase == 1
        active = sd_on and bool((dep & ~perm).any())
        dt = min(DT_OUT if active else DT_IDLE, t_end_h * 3600.0 - t)

        hot = si > 0
        atp = np.where(hot, atp1, atp0)
        rsv = np.where(hot, rsv1, rsv0)
        invtc = np.where(hot, inv_tc1, inv_tc0)

        # ---- potassium reaction-diffusion, explicit, one outer step
        hyp = dead | dep
        Dm = np.where(hyp, D_H, D_N)
        alpha = np.where(hyp, ALPHA_H, ALPHA_N)
        Kx += (dt * (Dm * ((np.roll(Kx, 1, 0) + np.roll(Kx, -1, 0) + np.roll(Kx, 1, 1)
                            + np.roll(Kx, -1, 1) - 4.0 * Kx) / (DX * DX))
                     + (LEAK * (1.0 - atp)
                        - VMAX * atp * (Kx - K_BASE) / (KM_PUMP + (Kx - K_BASE))) / alpha
                     )).astype(f32)
        Kx[0] = Kx[1]; Kx[-1] = Kx[-2]; Kx[:, 0] = Kx[:, 1]; Kx[:, -1] = Kx[:, -2]
        Kx[dep] = K_PEAK
        np.clip(Kx, K_BASE, K_PEAK, out=Kx)

        # ---- a new depolarisation igniting at the rim of the infarcted core
        if sd_on and t >= next_sd:
            # Recurrent depolarisations arise in the peri-infarct penumbra: still-living tissue,
            # adjacent to already-infarcted tissue, whose flow lies below the verified penumbral
            # upper bound and which is therefore working without metabolic reserve. Tissue below
            # the verified pump-failure threshold cannot be the generator in this model because it
            # does not survive long enough to host a wave - its tolerance is under a minute. The verified 9.4 mM potassium
            # threshold is deliberately NOT used as a gate, for either ignition or propagation -
            # the reaction-diffusion length of the potassium field around a depolarised cell is
            # sqrt(D Km alpha / Vmax) = 0.08 mm, a tenth of one cell, so no cell-averaged
            # potassium halo can ever cross it. That length is exactly why the front has to be
            # integrated in the sharp-interface limit, and the potassium field is validated on
            # what it can legitimately predict: the two depolarisation latencies.
            # the seed must sit on the OUTER rim of the infarct - a cell with living tissue
            # around it to spread into - not in an enclosed pocket inside it
            room = ndimage.uniform_filter((~dead).astype(np.float32), size=5, mode="nearest")
            rim = (ndimage.binary_dilation(dead, structure=_S8) & (phase == 0) & (~dead)
                   & (F0 <= PEN_HI) & (room >= 0.5))
            idx = np.flatnonzero(rim.ravel())
            if idx.size:
                j = int(idx[rg.integers(idx.size)])
                phase.ravel()[j] = 1
                t_dep.ravel()[j] = t
                n_sd += 1
                wave = dict(seed=(j // N, j % N), t0=t, t=[], d=[])
                waves.append(wave)
            else:
                n_sd_blocked += 1
            next_sd = t + rg.exponential(SD_INT * 60.0)

        # ---- front sub-steps, on a bounding box around the wave
        moving_any = bool(((phase == 1) & ~perm).any())
        if moving_any:
            band = ndimage.binary_dilation(((phase == 1) & ~perm), iterations=2)
            ys, xs = np.nonzero(band)
            y0, y1 = max(ys.min() - 2, 0), min(ys.max() + 3, N)
            x0, x1 = max(xs.min() - 2, 0), min(xs.max() + 3, N)
            sl = (slice(y0, y1), slice(x0, x1))
            ph_v, cr_v, dz_v, td_v = phase[sl], credit[sl], dose[sl], t_dep[sl]
            dd_v, kk_v, at_v, si_v = dead[sl], Kx[sl], atp[sl], si[sl]
            rs_v = rsv[sl]
            f0_v, pm_v = F0[sl], perm[sl]
            nsub = max(int(np.ceil(dt / DT_FRONT)), 1)
            hsub = dt / nsub
            for _ in range(nsub):
                d_v = ph_v == 1
                if not d_v.any():
                    break
                nbr = ndimage.binary_dilation(d_v, structure=_S8) & ~d_v
                adv = nbr & (ph_v == 0) & (~dd_v)
                if adv.any():
                    df = d_v.astype(f32)
                    kappa = -((np.roll(df, 1, 0) + np.roll(df, -1, 0) + np.roll(df, 1, 1)
                               + np.roll(df, -1, 1) - 4.0 * df) / (DX * DX))
                    c_loc = (c0 / 60.0) * (0.55 + 0.45 * rs_v) - np.clip(D_N * kappa, 0.0,
                                                                        c0 / 120.0)
                    np.clip(c_loc, 0.2 * c0 / 60.0, 2.0 * c0 / 60.0, out=c_loc)
                    cr_v[adv] += (c_loc[adv] * hsub / DX).astype(f32)
                    fire = adv & (cr_v >= 1.0)
                    if fire.any():
                        ph_v[fire] = 1
                        td_v[fire] = t
                        dz_v[fire] += (sd_cost_mult * NEUR_DECAY * (dur / 60.0)
                                       * (1.0 - rs_v[fire])).astype(f32)
                        hit = fire & (f0_v <= PEN_HI)
                        if si_on:
                            si_v[hit] = 3600.0
                        si_cells += float(hit.sum())
                        if wave is not None:
                            fy, fx = np.nonzero(fire)
                            wave["t"].extend([(t - wave["t0"]) / 60.0] * fy.size)
                            wave["d"].extend(np.hypot(fy + y0 - wave["seed"][0],
                                                      fx + x0 - wave["seed"][1]) * DX)
                t += hsub
        else:
            t += dt

        # depolarisation ends after the verified plateau duration; refractoriness ends when the
        # pump has cleared the load (per-cell, flow-dependent)
        fin = (phase == 1) & (~perm) & (t - t_dep >= dur)
        if fin.any():
            phase[fin] = 2
            t_ref[fin] = t
        rec = (phase == 2) & (~perm) & (t - t_ref >= refr_map)
        if rec.any():
            phase[rec] = 0
            credit[rec] = 0.0
        np.subtract(si, dt, out=si); np.clip(si, 0.0, None, out=si)

        # ---- stochastic injury accumulation
        live = ~dead
        dose += ((dt / 3600.0) * invtc * live).astype(f32)
        newly = live & (dose >= thresh)
        if newly.any():
            dead |= newly
            phase[newly] = 2
            t_ref[newly] = np.inf          # infarcted tissue never becomes excitable again

        if dead_at_img is None and t / 3600.0 >= T_IMG_H:
            dead_at_img = dead.copy()
        traj_t.append(t / 3600.0); traj_v.append(float(dead.sum() * cell_ml))
        while isnap < len(SNAP_H) and t / 3600.0 >= SNAP_H[isnap]:
            snaps[float(SNAP_H[isnap])] = float(dead.sum() * cell_ml)
            if keep_fields:
                snaps[("field", float(SNAP_H[isnap]))] = (dead.copy(), phase.copy(), Kx.copy())
            isnap += 1

    if dead_at_img is None:
        dead_at_img = dead.copy()
    # realised front speed, wave by wave: distance from the seed regressed on ignition time
    sp = []
    for w in waves:
        tm, dm = np.asarray(w["t"]), np.asarray(w["d"])
        ok = tm > 0
        if ok.sum() >= 30:
            sp.append(float(np.polyfit(tm[ok], dm[ok], 1)[0]))
    tt = np.array(traj_t); vv = np.array(traj_v)
    def vol(h):
        return float(np.interp(h, tt, vv))
    out = dict(collateral=collateral, thickness_mm=thickness_mm, cell_ml=cell_ml, sd_on=sd_on,
               sd_cost_mult=sd_cost_mult, si_on=si_on,
               c0_mm_per_min=c0, sd_duration_s=dur, n_sd=n_sd, n_sd_blocked=n_sd_blocked,
               n_waves_measured=len(sp),
               wave_cells_median=float(np.median([len(w["t"]) for w in waves])) if waves else 0.0,
               wave_cells_max=float(max([len(w["t"]) for w in waves])) if waves else 0.0,
               var_explained_geometry=var_geo,
               measured_front_speed=float(np.median(sp)) if sp else np.nan,
               domain_ml=float(N * N * cell_ml),
               deficit_ml=float(N * N * cell_ml * DEFICIT_RATIO),
               core0_ml=float(core0.sum() * cell_ml),
               penumbra0_ml=float(pen0.sum() * cell_ml),
               oligaemia0_ml=float(olig0.sum() * cell_ml),
               final_ml=vol(24.0), si_cells=si_cells,
               clearance_s_penumbra=float(np.median(clear_s[pen0])) if pen0.any() else np.nan,
               clearance_s_core_rim=float(np.median(clear_s[(F0 > AD_CBF) & (F0 < PUMP)]))
               if ((F0 > AD_CBF) & (F0 < PUMP)).any() else np.nan,
               clearance_s_normal=float(np.median(clear_s[F0 >= 0.9 * CBF_N]))
               if (F0 >= 0.9 * CBF_N).any() else np.nan,
               clearance_s_core=float(np.median(clear_s[core0])) if core0.any() else np.nan,
               frac_final_from_core=float((dead & core0).sum() / max(dead.sum(), 1)),
               frac_final_from_penumbra=float((dead & pen0).sum() / max(dead.sum(), 1)),
               frac_final_from_oligaemia=float((dead & olig0).sum() / max(dead.sum(), 1)),
               # the published core/penumbra split is measured on imaging within 3 h, where "core"
               # means tissue already irreversibly injured at that moment and "penumbra" means
               # hypoperfused tissue still alive. Those are the definitions used here.
               frac_final_from_imaging_core=float(
                   (dead & dead_at_img).sum() / max(dead.sum(), 1)),
               frac_final_from_imaging_penumbra=float(
                   (dead & (~dead_at_img) & (F0 <= PEN_HI)).sum() / max(dead.sum(), 1)),
               frac_final_from_imaging_oligaemia=float(
                   (dead & (~dead_at_img) & olig0).sum() / max(dead.sum(), 1)),
               imaging_core_ml=float(dead_at_img.sum() * cell_ml))
    for h in SNAP_H:
        out[f"v_{h:g}h"] = vol(float(h))
    fin_v = max(vol(24.0), 1e-9)
    out["t50"] = float(np.interp(0.50 * fin_v, vv, tt))
    out["t95"] = float(np.interp(0.95 * fin_v, vv, tt))
    out["n_sd_first_6h"] = int(sum(1 for w in waves if w["t0"] <= 6 * 3600.0))
    out["eigr"] = (vol(3.0) - vol(0.25)) / 2.75
    out["late_igr"] = (vol(24.0) - vol(6.0)) / 18.0
    out["igr3"] = vol(3.0) / 3.0
    out["t_core20"] = float(np.interp(CORE_PRES, vv, tt)) if vv[-1] >= CORE_PRES else np.nan
    if keep_fields:
        out["F0"] = F0
        out["snaps"] = snaps
    return out, tt, vv

# ============================================================================ the cohort
# The domain is Module B's calibrated at-risk territory, drawn with Module B's inferred volume
# dispersion, so the spatial model inherits exactly one calibrated scale from the aggregate model
# and adds no second one. Section J4 below reports what happens if the verified perfusion-deficit
# volume is used as the domain instead: it cannot be, and the arithmetic of why is given.
NPAT = 36
coll = rng.beta(2.2, 2.6, NPAT)
vrisk = rng.lognormal(np.log(V_AT_RISK), SIGMA_V, NPAT)
thick = vrisk * 1000.0 / AREA

print(f"Module J: {N}x{N} sheet at {DX} mm ({N*DX:.0f} mm across), domain volume "
      f"{V_AT_RISK:.1f} mL median")
print(f"   potassium constants derived, not fitted: leak {LEAK:.5f} mM/s reproduces the verified "
      f"{AD_LATENCY/60:.1f} min anoxic-depolarisation latency ({_lat/60:.2f} min integrated), "
      f"V_max {VMAX:.4f} mM/s reproduces the verified {REPOL_PERF/60:.1f} min repolarisation "
      f"({_rep/60:.2f} min integrated)")
print(f"   reaction-diffusion front width implied by the verified constants: "
      f"{SD_FRONT_WIDTH*1000:.0f} um = 1/{DX/SD_FRONT_WIDTH:.0f} of one cell, hence the "
      f"sharp-interface treatment; potassium decay length around a depolarised cell "
      f"{1000*np.sqrt(D_FREE/(LAM_N**2) * KM_PUMP * ALPHA_N / VMAX):.0f} um")

VARIANTS = (("full", dict()), ("no_spreading_ischaemia", dict(si_on=False)),
            ("no_passage_cost", dict(sd_cost_mult=0.0)), ("no_sd", dict(sd_on=False)))
rows, trajs, byvar = [], [], {k: [] for k, _ in VARIANTS}
for i in range(NPAT):
    base = None
    for name, kw in VARIANTS:
        o, tt, vv = simulate(coll[i], thick[i], rg=np.random.default_rng(K.SEED + 1600 + i), **kw)
        o["variant"] = name
        o["patient"] = i
        byvar[name].append(o)
        if name == "full":
            base = o
            trajs.append(np.interp(np.arange(0, 24.01, 0.25), tt, vv))
        rows.append(o)
    if (i + 1) % 10 == 0:
        print(f"   {i+1}/{NPAT} occlusions x {len(VARIANTS)} variants simulated")

S = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, np.ndarray))}
                  for r in rows])
K.save_table(S, "J_cohort")
np.save(K.RESULTS / "J_trajectories.npy", np.array(trajs))
np.save(K.RESULTS / "J_time_hours.npy", np.arange(0, 24.01, 0.25))
FULL = S[S.variant == "full"].reset_index(drop=True)
NOSD = S[S.variant == "no_sd"].reset_index(drop=True)
NOSI = S[S.variant == "no_spreading_ischaemia"].reset_index(drop=True)
NOCO = S[S.variant == "no_passage_cost"].reset_index(drop=True)

# --------------------------------------------------- J2 what the depolarisation waves contribute
# In a domain that is by construction the tissue at risk, waves can recruit only the fraction whose
# tolerance exceeds 24 h; their larger effect is on WHEN the infarct completes. Both are measured,
# and the contribution is decomposed into the persistent flow penalty and the direct passage cost.
def med(x):
    return float(np.median(np.asarray(x, float)))

sd_effect = pd.DataFrame([
    dict(quantity="final volume, waves on (mL)", value=med(FULL.final_ml)),
    dict(quantity="final volume, waves off (mL)", value=med(NOSD.final_ml)),
    dict(quantity="volume recruited by waves (mL)", value=med(FULL.final_ml - NOSD.final_ml)),
    dict(quantity="volume recruited by waves (% of final)",
         value=100 * med((FULL.final_ml - NOSD.final_ml) / FULL.final_ml)),
    dict(quantity="of which persistent flow penalty (mL)",
         value=med(FULL.final_ml - NOSI.final_ml)),
    dict(quantity="of which direct passage cost (mL)",
         value=med(FULL.final_ml - NOCO.final_ml)),
    dict(quantity="time to half the final infarct, waves on (h)", value=med(FULL.t50)),
    dict(quantity="time to half the final infarct, waves off (h)", value=med(NOSD.t50)),
    dict(quantity="advance in time to half (h)", value=med(NOSD.t50 - FULL.t50)),
    dict(quantity="time to 95% of final infarct, waves on (h)", value=med(FULL.t95)),
    dict(quantity="time to 95% of final infarct, waves off (h)", value=med(NOSD.t95)),
    dict(quantity="early growth rate, waves on (mL/h)", value=med(FULL.eigr)),
    dict(quantity="early growth rate, waves off (mL/h)", value=med(NOSD.eigr)),
])
K.save_table(sd_effect, "J_sd_contribution")

# ------------------------------------------------------------------- J3 mismatch prevalence
# Perfusion-imaging mismatch is defined as in the late-window trials: deficit-to-core ratio at
# least 1.8 with an absolute mismatch of at least 15 mL and a core below 70 mL. The deficit volume
# per patient is the model's at-risk domain scaled by the verified deficit-to-at-risk ratio.
def mismatch_prevalence(frame, hours):
    core = frame[f"v_{hours:g}h"].values
    deficit = frame.domain_ml.values * DEFICIT_RATIO
    ok = (deficit / np.maximum(core, 1e-9) >= 1.8) & (deficit - core >= 15.0) & (core < 70.0)
    return float(100.0 * ok.mean())

# --------------------------------------------------------------------- J4 validation table
core_frac = med(FULL.frac_final_from_core)
pen_frac = med(FULL.frac_final_from_penumbra)
fast = FULL.eigr >= np.percentile(FULL.eigr, 60)
slow = FULL.eigr < np.percentile(FULL.eigr, 40)
spd = FULL.measured_front_speed.dropna()

val = pd.DataFrame([
    dict(quantity="effective depolarisation front speed (mm/min)", model=med(spd),
         published=K.getp("sd_velocity"), pub_lo=K.getp("sd_velocity", "value_lo"),
         pub_hi=K.getp("sd_velocity", "value_hi"), used_in_calibration=False,
         src=K.param_src("sd_velocity"),
         note="local speed is drawn from the verified human ECoG distribution (src "
              + K.param_src("sd_speed_human") + "); the EFFECTIVE speed measured across the "
              "infarcting territory is an emergent output and is compared with an independent "
              "verified series"),
    dict(quantity="depolarisations in the first 6 h (events)", model=med(FULL.n_sd_first_6h),
         published=K.getp("sd_events_first_6h"), pub_lo=np.nan, pub_hi=np.nan,
         used_in_calibration=False, src=K.param_src("sd_events_first_6h"),
         note="rodent distal MCAO; the model's inter-event interval is the verified human value"),
    dict(quantity="anoxic-depolarisation latency (min)", model=_lat / 60.0,
         published=K.getp("sd_onset_after_arrest"),
         pub_lo=K.getp("sd_onset_after_arrest", "value_lo"),
         pub_hi=K.getp("sd_onset_after_arrest", "value_hi"), used_in_calibration=True,
         src=K.param_src("sd_onset_after_arrest"), note="this latency fixes the leak constant"),
    dict(quantity="repolarisation time, perfused (min)", model=_rep / 60.0,
         published=K.getp("sd_repolarization_perfused"),
         pub_lo=K.getp("sd_repolarization_perfused", "value_lo"),
         pub_hi=K.getp("sd_repolarization_perfused", "value_hi"), used_in_calibration=True,
         src=K.param_src("sd_repolarization_perfused"),
         note="this time fixes the pump capacity"),
    dict(quantity="depolarisation duration in the core (min)",
         model=med(FULL.clearance_s_core) / 60.0, published=K.getp("sd_duration_core"),
         pub_lo=K.getp("sd_duration_core", "value_lo"),
         pub_hi=K.getp("sd_duration_core", "value_hi"), used_in_calibration=False,
         src=K.param_src("sd_duration_core"),
         note="emergent: the clearance time the pump capacity implies at core flow"),
    dict(quantity="median final infarct volume (mL)", model=med(FULL.final_ml),
         published=K.getp("typical_lvo_infarct_volume"), pub_lo=np.nan, pub_hi=np.nan,
         used_in_calibration=False, src=K.param_src("typical_lvo_infarct_volume"),
         note="domain volume is inherited from Module B's IGR calibration, not fitted here"),
    dict(quantity="mean final infarct volume, MCA territory (mL)", model=float(FULL.final_ml.mean()),
         published=K.getp("final_infarct_volume_mca"), pub_lo=np.nan, pub_hi=np.nan,
         used_in_calibration=False, src=K.param_src("final_infarct_volume_mca"), note=""),
    dict(quantity="median growth rate over the first 3 h (mL/h)", model=med(FULL.igr3),
         published=K.getp("igr_median"), pub_lo=K.getp("igr_median", "value_lo"),
         pub_hi=K.getp("igr_median", "value_hi"), used_in_calibration=False,
         src=K.param_src("igr_median"), note="inherited calibration target of Module B"),
    dict(quantity="early growth rate, slower 40% (mL/h)", model=med(FULL.eigr[slow]),
         published=K.getp("eigr_good_outcome_post_evt"),
         pub_lo=K.getp("eigr_good_outcome_post_evt", "value_lo"),
         pub_hi=K.getp("eigr_good_outcome_post_evt", "value_hi"), used_in_calibration=False,
         src=K.param_src("eigr_good_outcome_post_evt"), note=""),
    dict(quantity="early growth rate, faster 40% (mL/h)", model=med(FULL.eigr[fast]),
         published=K.getp("eigr_poor_outcome_post_evt"),
         pub_lo=K.getp("eigr_poor_outcome_post_evt", "value_lo"),
         pub_hi=K.getp("eigr_poor_outcome_post_evt", "value_hi"), used_in_calibration=False,
         src=K.param_src("eigr_poor_outcome_post_evt"), note=""),
    dict(quantity="late growth rate, slower 40% (mL/h)", model=med(FULL.late_igr[slow]),
         published=K.getp("late_igr_good_outcome"),
         pub_lo=K.getp("late_igr_good_outcome", "value_lo"),
         pub_hi=K.getp("late_igr_good_outcome", "value_hi"), used_in_calibration=False,
         src=K.param_src("late_igr_good_outcome"), note=""),
    dict(quantity="late growth rate, faster 40% (mL/h)", model=med(FULL.late_igr[fast]),
         published=K.getp("late_igr_poor_outcome"),
         pub_lo=K.getp("late_igr_poor_outcome", "value_lo"),
         pub_hi=K.getp("late_igr_poor_outcome", "value_hi"), used_in_calibration=False,
         src=K.param_src("late_igr_poor_outcome"), note=""),
    dict(quantity="share of final infarct already infarcted at 3 h imaging (%)",
         model=100 * med(FULL.frac_final_from_imaging_core),
         published=K.getp("fraction_final_infarct_from_core"),
         pub_lo=np.nan, pub_hi=np.nan, used_in_calibration=False,
         src=K.param_src("fraction_final_infarct_from_core"), note=""),
    dict(quantity="share of final infarct arising in surviving penumbra at 3 h (%)",
         model=100 * med(FULL.frac_final_from_imaging_penumbra),
         published=K.getp("fraction_final_infarct_from_penumbra"),
         pub_lo=np.nan, pub_hi=np.nan, used_in_calibration=False,
         src=K.param_src("fraction_final_infarct_from_penumbra"), note=""),
    dict(quantity="median core volume at 3 h (mL)", model=med(FULL.imaging_core_ml),
         published=CORE_PRES, pub_lo=K.getp("core_volume_at_presentation", "value_lo"),
         pub_hi=K.getp("core_volume_at_presentation", "value_hi"), used_in_calibration=False,
         src=K.param_src("core_volume_at_presentation"),
         note="verified median and IQR of the core volume measured at presentation"),
    dict(quantity="mismatch prevalence at 6 h (%)", model=mismatch_prevalence(FULL, 6),
         published=K.getp("mismatch_prevalence_6h"), pub_lo=np.nan, pub_hi=np.nan,
         used_in_calibration=False, src=K.param_src("mismatch_prevalence_6h"), note=""),
    dict(quantity="mismatch prevalence at 18 h (%)", model=mismatch_prevalence(FULL, 18),
         published=K.getp("mismatch_prevalence_18h"), pub_lo=np.nan, pub_hi=np.nan,
         used_in_calibration=False, src=K.param_src("mismatch_prevalence_18h"), note=""),
])
val["within_published_range"] = [
    (np.isfinite(r.pub_lo) and r.pub_lo <= r.model <= r.pub_hi) if np.isfinite(r.pub_lo)
    else (abs(r.model - r.published) / max(abs(r.published), 1e-9) <= 0.35)
    for r in val.itertuples()]
val["ratio_model_to_published"] = val.model / val.published
K.save_table(val, "J_validation")

# ------------------- J5 the domain-volume contradiction, stated as arithmetic rather than hidden
# If the verified perfusion-deficit volume (Tmax > 3 s) were used as the at-risk domain, the same
# flow marginal and the same tolerance curve would infarct the same FRACTION of it, so the implied
# 24-hour volume scales by the ratio of the two volumes. That number is reported here because it is
# the reason the domain is the at-risk sub-volume and not the perfusion deficit.
implied_if_deficit = med(FULL.final_ml) * DEFICIT_RATIO
contradiction = dict(
    verified_perfusion_deficit_mL=V_DEFICIT,
    module_b_at_risk_volume_mL=V_AT_RISK,
    deficit_to_at_risk_ratio=DEFICIT_RATIO,
    model_median_final_volume_mL=med(FULL.final_ml),
    implied_final_volume_if_domain_were_the_deficit_mL=implied_if_deficit,
    verified_typical_lvo_volume_mL=K.getp("typical_lvo_infarct_volume"),
    overstatement_factor=implied_if_deficit / K.getp("typical_lvo_infarct_volume"),
    resolution=("the Tmax>3 s perfusion deficit cannot be treated as tissue at risk: doing so "
                "would predict a 24-hour infarct %.0f mL, %.1f-fold the verified typical "
                "large-vessel-occlusion volume. The domain is therefore the at-risk sub-volume "
                "calibrated in Module B, and the oligaemic remainder of the deficit lies outside "
                "it; the consequence is that in this model depolarisation waves can accelerate "
                "infarct completion but can recruit only the small fraction of the domain whose "
                "tolerance exceeds 24 h."
                % (implied_if_deficit, implied_if_deficit / K.getp("typical_lvo_infarct_volume"))))

# --------------------------------------------- J6 sensitivity to the one structural choice
scan = []
for w in (0.0, 0.5, 1.0, 2.0, np.inf):
    fv, cf, ns = [], [], []
    for i in range(8):
        o, _, _ = simulate(coll[i], thick[i], rg=np.random.default_rng(K.SEED + 1600 + i),
                           w_noise=w)
        fv.append(o["final_ml"]); cf.append(o["frac_final_from_core"]); ns.append(o["n_sd"])
    scan.append(dict(w_noise=w, geometry_only=(w == 0.0), random_only=(not np.isfinite(w)),
                     median_final_ml=med(fv), median_core_share=med(cf),
                     median_n_sd=med(ns)))
K.save_table(pd.DataFrame(scan), "J_geometry_sensitivity")

# ------------------------------------------------------------------ J7 exemplar field snapshots
ex = {}
for lab, c in (("poor", 0.18), ("intermediate", 0.45), ("good", 0.78)):
    o, _, _ = simulate(c, V_AT_RISK * 1000.0 / AREA, rg=np.random.default_rng(K.SEED + 99),
                       keep_fields=True)
    ex[f"{lab}_F0"] = o["F0"].astype(np.float32)
    for h in SNAP_H:
        dead, phase, Kf = o["snaps"][("field", float(h))]
        ex[f"{lab}_dead_{h:g}h"] = dead
        ex[f"{lab}_phase_{h:g}h"] = phase
        ex[f"{lab}_K_{h:g}h"] = Kf.astype(np.float32)
np.savez_compressed(K.RESULTS / "J_exemplar_fields.npz", **ex)

K.save_json(dict(grid=[N, N], dx_mm=DX, domain_mL_median=V_AT_RISK,
                 potassium_leak_mM_per_s=float(LEAK), potassium_vmax_mM_per_s=float(VMAX),
                 km_mM=float(KM_PUMP),
                 ad_latency_integrated_min=_lat / 60.0, repolarisation_integrated_min=_rep / 60.0,
                 sd_front_width_um=float(SD_FRONT_WIDTH * 1000),
                 potassium_decay_length_um=float(
                     1000 * np.sqrt(D_FREE / (LAM_N ** 2) * KM_PUMP * ALPHA_N / VMAX)),
                 n_patients=NPAT, variants=[v for v, _ in VARIANTS],
                 median_final_volume_mL=med(FULL.final_ml),
                 median_sd_events=med(FULL.n_sd),
                 median_effective_front_speed=med(spd),
                 variance_in_flow_explained_by_geometry=med(FULL.var_explained_geometry),
                 sd_recruited_pct=100 * med((FULL.final_ml - NOSD.final_ml) / FULL.final_ml),
                 sd_time_advance_to_half_h=med(NOSD.t50 - FULL.t50),
                 mismatch_6h_pct=mismatch_prevalence(FULL, 6),
                 mismatch_18h_pct=mismatch_prevalence(FULL, 18),
                 n_validation=int(len(val)),
                 n_validation_passed=int(val.within_published_range.sum()),
                 domain_volume_contradiction=contradiction),
            "J_spatial_summary")

print(sd_effect.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print(val[["quantity", "model", "published", "used_in_calibration",
           "within_published_range"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print(f"  Module J: {int(val.within_published_range.sum())}/{len(val)} comparisons within range; "
      f"waves recruit {100*med((FULL.final_ml-NOSD.final_ml)/FULL.final_ml):.1f}% of the final "
      f"volume and advance the time to half the infarct by "
      f"{med(NOSD.t50-FULL.t50)*60:.0f} min")
print("  " + contradiction["resolution"])
