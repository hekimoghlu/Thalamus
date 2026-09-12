"""framework - shared configuration, verified-data loaders and numerics helpers.

= Kinetic Architecture of Ischemic Risk, Onset and recovery Systems.

Design rule enforced here: a value enters a computation only if it was verified against a
public source. `load_verified()` drops any row whose requested numeric field is the literal
string UNVERIFIED, and `getp()` raises if a parameter was never verified. Nothing is imputed.
"""
from __future__ import annotations
import json, math, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "verified"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
for d in (RESULTS, FIGS):
    d.mkdir(parents=True, exist_ok=True)

SEED = 20260912
UNV = "UNVERIFIED"


def load_verified(name: str, numeric: list[str] | None = None,
                  require: list[str] | None = None) -> pd.DataFrame:
    """Load a verified-data table, coercing `numeric` columns and dropping rows whose
    `require` columns are UNVERIFIED/empty."""
    df = pd.read_csv(DATA / f"{name}.csv", dtype=str, keep_default_na=False)
    for c in numeric or []:
        df[c] = pd.to_numeric(df[c].replace({UNV: np.nan, "": np.nan, "NA": np.nan}),
                              errors="coerce")
    for c in require or []:
        df = df[df[c].notna()] if c in (numeric or []) else df[df[c] != UNV]
    return df.reset_index(drop=True)


_PARAMS = None


def getp(param: str, field: str = "value") -> float:
    """Fetch a verified cascade/sensorimotor/rehab parameter by name. Raises KeyError if the
    parameter is absent and ValueError if it exists but was never verified."""
    global _PARAMS
    if _PARAMS is None:
        frames = []
        for tab, key in (("cascade_params", "param"), ("sensorimotor_params", "param"),
                         ("rehab_benchmarks", "benchmark"), ("haemorrhage_params", "param"),
                         ("svd_cognition_params", "param"), ("spatial_social_params", "param")):
            d = pd.read_csv(DATA / f"{tab}.csv", dtype=str, keep_default_na=False)
            d = d.rename(columns={key: "param"})
            d["table"] = tab
            frames.append(d[["param", "value", "value_lo", "value_hi", "unit", "src", "table"]])
        _ALL = pd.concat(frames, ignore_index=True)
        _dup = _ALL[_ALL.duplicated("param", keep=False)]
        if len(_dup):
            raise ValueError("duplicate parameter names across verified tables: "
                             + ", ".join(sorted(set(_dup["param"]))))
        _PARAMS = _ALL.set_index("param")
    if param not in _PARAMS.index:
        raise KeyError(f"{param} is not in the verified parameter tables")
    v = _PARAMS.loc[param, field]
    v = v.iloc[0] if hasattr(v, "iloc") else v
    if v == UNV or v == "":
        raise ValueError(f"{param}.{field} was never verified - it must not be used")
    return float(v)


def param_src(param: str) -> str:
    getp(param)  # populates cache and validates
    s = _PARAMS.loc[param, "src"]
    return str(s.iloc[0] if hasattr(s, "iloc") else s)


def epi(measure: str) -> float:
    d = pd.read_csv(DATA / "epidemiology.csv", dtype=str, keep_default_na=False)
    row = d[d["measure"] == measure]
    if row.empty:
        raise KeyError(measure)
    return float(row.iloc[0]["value"])


def save_table(df: pd.DataFrame, name: str) -> Path:
    p = RESULTS / f"{name}.csv"
    df.to_csv(p, index=False)
    print(f"  -> {p.relative_to(ROOT)}  ({len(df)} rows)")
    return p


def save_json(obj: dict, name: str) -> Path:
    p = RESULTS / f"{name}.json"
    p.write_text(json.dumps(obj, indent=2, default=float))
    print(f"  -> {p.relative_to(ROOT)}")
    return p


def logit(p):
    return np.log(p / (1 - p))


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def se_from_ci(lo: float, hi: float) -> float:
    """Standard error of a log-scale effect from its 95% CI on the ratio scale."""
    return (math.log(hi) - math.log(lo)) / (2 * 1.959964)


def auc_from_scores(score: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney U based AUC (no sklearn dependency, handles ties)."""
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    yy = y[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = yy.sum()
    n0 = len(yy) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def paf(prev: float, or_: float) -> float:
    """Population attributable fraction under a rare-outcome OR-as-RR approximation."""
    return prev * (or_ - 1) / (1 + prev * (or_ - 1))


def or_from_paf(prev: float, pafrac: float) -> float:
    """Invert the PAF expression to recover the implied effect size."""
    return 1 + (pafrac / (prev * (1 - pafrac)))
