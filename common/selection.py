# -*- coding: utf-8 -*-
"""
selection.py -- the candidate-selection rule in one place (the real pipeline and the simulation both use this code).

Main v3 framework (Path B, FDR-null compatible; no Top-N and no BH condition):
  Step N : fixed nominal screen: p < NOMINAL_P_MAX and n per group >= MAIN_MIN_N_PER_GROUP  (main_candidates)
  Step S : stability = bootstrap (CI excludes 0 and has the same sign as the observed effect) AND split-half (is_stable)
  Step A : pass the exploratory co-mutation adjustment (common/adjust.py; robust_to_adjustment only)
None of the steps controls FDR on the final list; the output is an "exploratory prioritisation" and may contain zero candidates.

Sensitivity analyses (never merged with the main rule): bh_gate_candidates (BH q < SELECT_FDR_Q), threshold_stable (Top-N),
passes_effect_precision (conservative effect + narrow CI).

Definitions of the stability measures (each measures something different):
  bootstrap  : "estimation precision" on all n Discovery samples. pct_negative/pct_positive are over B bootstrap draws (denominator = B);
               when the 95% percentile CI excludes 0, >=97.5% of draws share the sign; direction% is therefore not independent information and is not a gate.
  split-half : "direction reproducibility at half the n": each group is split into two disjoint halves and both halves must share the sign of the full-sample effect.
               Denominator = all attempted splits; an invalid split counts as a failure.
  Both are INTERNAL stability checks on the same data; neither is independent replication nor a selection-bias correction.
"""
import numpy as np
from . import config_official as cfg
from . import stats


# ------------------------------------------------------------------ main rule (nominal screen)
def main_candidates(df, p_max=None, min_n=None):
    """df: the whole screen family (columns gene, drug, p_value, n_mutated_with_AUC, n_wt_with_AUC). Independent of rank/N/BH."""
    p_max = cfg.NOMINAL_P_MAX if p_max is None else p_max
    min_n = cfg.MAIN_MIN_N_PER_GROUP if min_n is None else min_n
    if df is None or len(df) == 0:
        return []
    s = df[(df["p_value"] < p_max) & (df["n_mutated_with_AUC"] >= min_n) & (df["n_wt_with_AUC"] >= min_n)]
    return sorted(set(zip(s["gene"], s["drug"])))


# ------------------------------------------------------------------ sensitivity: BH gate (v2 rule)
def bh_gate_candidates(df, q=None):
    """BH-FDR_q (from stage 01, over the whole screen family) < SELECT_FDR_Q. Sensitivity analysis only."""
    q = cfg.SELECT_FDR_Q if q is None else q
    if df is None or len(df) == 0 or "FDR_q" not in df.columns:
        return []
    s = df[df["FDR_q"] < q]
    return sorted(set(zip(s["gene"], s["drug"])))


# ------------------------------------------------------------------ sensitivity: Top-N threshold-stable (legacy rule)
def threshold_stable(df, thresholds=None, top_n=None):
    thresholds = cfg.THRESHOLDS if thresholds is None else thresholds
    top_n = cfg.TOP_N_FOR_OVERLAP if top_n is None else top_n
    tops = {}
    for t in thresholds:
        s = df[(df["n_mutated_with_AUC"] >= t) & (df["n_wt_with_AUC"] >= t)].sort_values("p_value").head(top_n)
        tops[t] = set(zip(s["gene"], s["drug"]))
    always = sorted(set.intersection(*tops.values())) if tops else []
    return always, tops


# ------------------------------------------------------------------ stability
def stability_row(mut_auc, wt_auc, B=None, n_splits=None, min_group_n=None, seed=None):
    B = cfg.BOOTSTRAP_B if B is None else B
    n_splits = cfg.N_SPLIT_HALF if n_splits is None else n_splits
    min_group_n = cfg.SPLIT_HALF_MIN_GROUP_N if min_group_n is None else min_group_n
    obs = stats.rank_biserial(mut_auc, wt_auc)
    d = stats.bootstrap_distribution(mut_auc, wt_auc, B, seed)
    if len(d) == 0:
        lo = hi = med = pn = pp = np.nan
        crosses = True
    else:
        lo, hi = np.percentile(d, [2.5, 97.5])
        med, pn, pp = float(np.median(d)), float((d < 0).mean() * 100), float((d > 0).mean() * 100)
        crosses = bool(lo <= 0 <= hi)
    sp = stats.split_half_stability(mut_auc, wt_auc, n_splits, min_group_n, seed=seed)
    return {"observed_effect": obs, "bootstrap_median_effect": med, "bootstrap_CI_low": lo, "bootstrap_CI_high": hi,
            "CI_crosses_zero": crosses, "n_boot_draws": int(len(d)),
            "pct_negative_direction": pn, "pct_positive_direction": pp,
            "n_split_half_attempted": sp["n_splits_attempted"], "n_split_half_valid": sp["n_splits_valid"],
            "pct_split_half_valid": sp["pct_valid_splits"],
            "pct_split_half_same_direction": sp["pct_both_halves_same_direction"],
            "pct_split_half_match_full": sp["pct_both_halves_match_full"],
            "pct_split_half_match_full_among_valid": sp["pct_both_halves_match_full_among_valid"],
            "pct_split_half_nominal_p": sp["pct_both_halves_nominal_p"]}


def _get(r, k):
    try:
        v = r.get(k, np.nan)
    except AttributeError:
        v = np.nan
    return np.nan if v is None else v


def is_stable(r):
    """(bootstrap_ok, split_half_ok); "stable" = both.
    bootstrap_ok: finite CI that excludes 0 (lo<=0<=hi => reject) and has the same sign as the observed effect.
    split_half_ok: at least one valid split and pct_match_full (denominator = all splits) >= SPLIT_HALF_DIRECTION_MIN."""
    lo, hi = r["bootstrap_CI_low"], r["bootstrap_CI_high"]
    obs = _get(r, "observed_effect")
    ok_ci = bool(np.isfinite(lo) and np.isfinite(hi) and not bool(r["CI_crosses_zero"]))
    sign_ok = (
            np.isfinite(obs)
            and obs != 0
            and np.sign(obs) == np.sign(lo)
    )
    boot = ok_ci and bool(sign_ok)
    split = (r["n_split_half_valid"] > 0) and (r["pct_split_half_match_full"] >= cfg.SPLIT_HALF_DIRECTION_MIN)
    return bool(boot), bool(split)


def direction_ok(r):
    """Diagnostic (not a gate): >= DIRECTION_CONSISTENCY_MIN% of bootstrap draws share the sign. Almost redundant given a CI that excludes 0."""
    a, b = r["pct_negative_direction"], r["pct_positive_direction"]
    return bool(np.isfinite(a) and np.isfinite(b) and max(a, b) >= cfg.DIRECTION_CONSISTENCY_MIN)


# ------------------------------------------------------------------ sensitivity: effect + uncertainty
def conservative_effect(r):
    """The bootstrap CI end closest to zero. 0 if the CI is unbounded, covers 0 (lo<=0<=hi),
    or its sign disagrees with the observed effect (when r is given)."""
    lo, hi = r["bootstrap_CI_low"], r["bootstrap_CI_high"]
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0 <= hi:
        return 0.0
    obs = _get(r, "observed_effect")
    if np.isfinite(obs) and np.sign(obs) != np.sign(lo):
        return 0.0
    return float(min(abs(lo), abs(hi)))


def ci_width(r):
    lo, hi = r["bootstrap_CI_low"], r["bootstrap_CI_high"]
    return float(hi - lo) if np.isfinite(lo) and np.isfinite(hi) else np.inf


def passes_effect_precision(r, n_mut, n_wt):
    """Sensitivity rule: conservative_effect >= EFFECT_MIN_CONSERVATIVE AND ci_width <= CI_WIDTH_MAX AND n per group >= MIN_N_EFFECT_RULE."""
    return bool(conservative_effect(r) >= cfg.EFFECT_MIN_CONSERVATIVE and ci_width(r) <= cfg.CI_WIDTH_MAX
                and n_mut >= cfg.MIN_N_EFFECT_RULE and n_wt >= cfg.MIN_N_EFFECT_RULE)
