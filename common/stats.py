# -*- coding: utf-8 -*-
"""
stats.py v3 -- the only statistics implementation of the project (FDR, effect size, stability, OLS, selection-aware).

Sign convention (identical in every stage):
    rank_biserial = 1 - 2*U1/(n1*n2)  ,  U1 for the mutated group
    > 0 : AUC lower in mutated   (mutated more sensitive)
    < 0 : AUC higher in mutated  (mutated more resistant)
Note: the OLS coefficient on AUC has the opposite sign (beta > 0 means mutated is more resistant).

Reference test: Mann-Whitney asymptotic + tie correction + continuity correction.
test_pair (scipy, method="asymptotic") and mwu_columns (vectorised) must agree; test: test_pipeline.py::test_mwu_vs_scipy
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, norm, t as t_dist
from statsmodels.stats.multitest import multipletests


def _finite(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def _pos_int(x, name):
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x <= 0:
        raise ValueError(f"{name} باید عدد صحیح مثبت باشد (گرفت: {x!r})")
    return int(x)


# ------------------------------------------------------------------ FDR
def fdr(p, method="fdr_bh"):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        out[ok] = multipletests(p[ok], method=method)[1]
    return out


# ------------------------------------------------------------------ two-group test
def rank_biserial(mut, wt):
    mut, wt = _finite(mut), _finite(wt)
    n1, n2 = len(mut), len(wt)
    if n1 == 0 or n2 == 0:
        return np.nan
    r = rankdata(np.concatenate([mut, wt]))
    u1 = r[:n1].sum() - n1 * (n1 + 1) / 2
    return float(1 - 2 * u1 / (n1 * n2))


def test_pair(mut, wt):
    """method="asymptotic" is explicit so that it matches mwu_columns (with n>=10 per group, 'auto' is asymptotic as well)."""
    mut, wt = _finite(mut), _finite(wt)
    if len(mut) == 0 or len(wt) == 0:
        return {"median_AUC_mutated": np.nan, "median_AUC_wt": np.nan, "delta_median_AUC": np.nan,
                "rank_biserial_effect": np.nan, "p_value": np.nan}
    _, p = mannwhitneyu(mut, wt, alternative="two-sided", method="asymptotic", use_continuity=True)
    return {"median_AUC_mutated": np.median(mut), "median_AUC_wt": np.median(wt),
            "delta_median_AUC": np.median(mut) - np.median(wt),
            "rank_biserial_effect": rank_biserial(mut, wt), "p_value": p}


def mwu_columns(values, G):
    """
    Vectorised Mann-Whitney: one vector of values (n) against several 0/1 columns in G (n x g).
    Equivalent to mannwhitneyu(method='asymptotic', use_continuity=True, tie-corrected).
    Returns: n1, n2, U1, p, rank_biserial
    An empty group (n1=0 or n2=0) or n<2 => rank_biserial and p = NaN (no division by zero).
    sd=0 (all values equal) => p=1.
    """
    v = np.asarray(values, float)
    G = np.asarray(G, float)
    n = len(v)
    r = rankdata(v)
    n1 = G.sum(0)
    n2 = n - n1
    U1 = r @ G - n1 * (n1 + 1) / 2
    _, cnt = np.unique(v, return_counts=True)
    tie = float((cnt ** 3 - cnt).sum())
    valid = (n1 > 0) & (n2 > 0) & (n >= 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = n1 * n2 / 2
        var = n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1) if n > 1 else np.nan))
        sd = np.sqrt(var)
        U = np.maximum(U1, n1 * n2 - U1)
        z = (U - mu - 0.5) / sd
        p_raw = np.clip(2 * norm.sf(z), 0, 1)
        rb_raw = 1 - 2 * U1 / (n1 * n2)
    p = np.where(valid, np.where(np.isfinite(sd) & (sd > 0), p_raw, 1.0), np.nan)
    rb = np.where(valid, rb_raw, np.nan)
    return n1, n2, U1, p, rb


def rb_p(mut, wt):
    mut, wt = _finite(mut), _finite(wt)
    if len(mut) < 2 or len(wt) < 2:
        return np.nan, np.nan
    v = np.concatenate([mut, wt])
    g = np.r_[np.ones(len(mut)), np.zeros(len(wt))][:, None]
    _, _, _, p, rb = mwu_columns(v, g)
    return float(rb[0]), float(p[0])


# ------------------------------------------------------------------ Bootstrap
def bootstrap_distribution(mut, wt, B, seed=None):
    """
    Non-parametric percentile bootstrap (each group independently, with replacement) for the rank-biserial effect.
    Limitations: (1) a percentile CI has no optimal coverage for small n / skewed distributions.
    (2) This CI does not correct for selecting pairs by nominal p: it is not a selection-adjusted CI and does not remove winner's curse.
    """
    B = _pos_int(B, "B")
    mut, wt = _finite(mut), _finite(wt)
    if len(mut) < 3 or len(wt) < 3:
        return np.array([])
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for i in range(B):
        out[i] = rank_biserial(rng.choice(mut, len(mut)), rng.choice(wt, len(wt)))
    return out


def bootstrap_ci(mut, wt, B, seed=None):
    d = bootstrap_distribution(mut, wt, B, seed)
    if len(d) == 0:
        return (np.nan, np.nan)
    return tuple(np.percentile(d, [2.5, 97.5]))


# ------------------------------------------------------------------ Split-half
def split_half_stability(mut, wt, n_splits=1000, min_group_n=5, alpha=0.05, seed=None):
    """
    Internal directional stability using disjoint halves (without replacement).
    Each split: every group is divided into two non-overlapping halves; the effect of each half is estimated separately.
      same_direction : both halves have the same non-zero sign
      match_full     : both halves have the same sign as the full-sample effect  (gating criterion)
      nominal_p      : match_full and p<alpha in both halves
    Denominator of all percentages = n_splits (all attempted splits); an invalid split counts as a "failure", it is not dropped.
    split_status:
      insufficient_group_size : the sample size did not allow a split (n_splits_attempted = 0)
      full_effect_zero_or_invalid : the full-sample effect is zero/invalid (n_splits_attempted = 0)
      no_valid_split : splits were run but all were invalid (n_splits_attempted = n_splits)
      completed : at least one valid split
    This is an internal stability analysis, not independent replication and not a selection-bias correction.
    The splits are dependent (they reuse the same observations).
    """
    n_splits = _pos_int(n_splits, "n_splits")
    min_group_n = _pos_int(min_group_n, "min_group_n")
    mut, wt = _finite(mut), _finite(wt)

    def _empty(status, attempted=0):
        return {"split_status": status, "n_splits_attempted": attempted, "n_splits_valid": 0,
                "pct_valid_splits": np.nan, "pct_both_halves_same_direction": np.nan,
                "pct_both_halves_match_full": np.nan, "pct_both_halves_match_full_among_valid": np.nan,
                "pct_both_halves_nominal_p": np.nan}

    n1, n2 = len(mut), len(wt)
    if n1 < 2 * min_group_n or n2 < 2 * min_group_n:
        return _empty("insufficient_group_size")
    full = rank_biserial(mut, wt)
    s = np.sign(full)
    if not np.isfinite(full) or s == 0:
        return _empty("full_effect_zero_or_invalid")
    rng = np.random.default_rng(seed)
    h1, h2 = n1 // 2, n2 // 2
    valid = same = match = nom = 0
    for _ in range(n_splits):
        i, j = rng.permutation(n1), rng.permutation(n2)
        ea, pa = rb_p(mut[i[:h1]], wt[j[:h2]])
        eb, pb = rb_p(mut[i[h1:2 * h1]], wt[j[h2:2 * h2]])
        if not (np.isfinite(ea) and np.isfinite(eb) and np.isfinite(pa) and np.isfinite(pb)):
            continue
        valid += 1
        sa, sb = np.sign(ea), np.sign(eb)
        if sa != 0 and sa == sb:
            same += 1
            if sa == s:
                match += 1
                if pa < alpha and pb < alpha:
                    nom += 1
    if valid == 0:
        return _empty("no_valid_split", attempted=n_splits)
    return {"split_status": "completed", "n_splits_attempted": n_splits, "n_splits_valid": valid,
            "pct_valid_splits": 100 * valid / n_splits,
            "pct_both_halves_same_direction": 100 * same / n_splits,
            "pct_both_halves_match_full": 100 * match / n_splits,
            "pct_both_halves_match_full_among_valid": 100 * match / valid,
            "pct_both_halves_nominal_p": 100 * nom / n_splits}


# ------------------------------------------------------------------ Outlier / robust
def winsorize(arr, limits=(0.05, 0.05)):
    arr = _finite(arr)
    if len(arr) < 3:
        return arr.copy()
    lo, hi = np.percentile(arr, [limits[0] * 100, (1 - limits[1]) * 100])
    return np.clip(arr, lo, hi)


def trim(arr, proportion=0.05):
    arr = np.sort(_finite(arr))
    k = int(np.floor(len(arr) * proportion))
    if len(arr) < 5 or k == 0 or len(arr) - 2 * k < 1:
        return arr
    return arr[k:-k]


def remove_outliers_iqr(mut, wt, k=1.5):
    def _rm(a):
        a = _finite(a)
        if len(a) < 4:
            return a, 0
        q1, q3 = np.percentile(a, [25, 75])
        c = a[(a >= q1 - k * (q3 - q1)) & (a <= q3 + k * (q3 - q1))]
        return c, len(a) - len(c)
    cm, rm = _rm(mut)
    cw, rw = _rm(wt)
    return cm, cw, rm + rw


# ------------------------------------------------------------------ permutation (two groups)
def permutation_test(mut, wt, n_perm=2000, seed=None):
    n_perm = _pos_int(n_perm, "n_perm")
    mut, wt = _finite(mut), _finite(wt)
    if len(mut) < 3 or len(wt) < 3:
        return np.nan, np.nan
    obs = rank_biserial(mut, wt)
    pooled = np.concatenate([mut, wt])
    r = rankdata(pooled)
    n1, n = len(mut), len(pooled)
    n2 = n - n1
    rng = np.random.default_rng(seed)
    idx = np.argsort(rng.random((n_perm, n)), axis=1)[:, :n1]
    u1 = r[idx].sum(1) - n1 * (n1 + 1) / 2
    perm = 1 - 2 * u1 / (n1 * n2)
    return float(obs), float((np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (n_perm + 1))


# ------------------------------------------------------------------ OLS
OLS_MAX_COND = 1e10          # condition number of the design matrix; larger => None (ill-conditioned)
OLS_MAX_LEVERAGE = 1 - 1e-8  # larger leverage => HC3 unstable => None


def ols(y, X):
    """
    OLS with intercept. y: Series indexed like X (DataFrame).
    p_value: classical SE; p_value_HC3: robust SE (HC3). HC3 only makes the SE robust; it does not make the model robust to confounding/misspecification.
    None when: invalid input, small n, rank-deficient, condition number > OLS_MAX_COND, leverage ~ 1,
    or non-finite coefficients/SEs.
    Returns: n, n_input, n_dropped (dropped for NaN/inf), condition_number, max_leverage, df_resid, table.
    """
    if not isinstance(X, pd.DataFrame) or X.empty or X.columns.duplicated().any():
        return None
    y = pd.Series(y, dtype=float)
    d0 = pd.concat([y.rename("__y"), X.astype(float)], axis=1).replace([np.inf, -np.inf], np.nan)
    n_input = len(d0)
    d = d0.dropna()
    n = len(d)
    cols = list(X.columns)
    if n < len(cols) + 2:
        return None
    Xm = np.column_stack([np.ones(n), d[cols].to_numpy()])
    yv = d["__y"].to_numpy()
    k = Xm.shape[1]
    beta, _, rank, _ = np.linalg.lstsq(Xm, yv, rcond=None)
    if rank < k:
        return None
    cond = float(np.linalg.cond(Xm))
    if not np.isfinite(cond) or cond > OLS_MAX_COND:
        return None
    dfr = n - k
    res = yv - Xm @ beta
    try:
        XtXi = np.linalg.inv(Xm.T @ Xm)
    except np.linalg.LinAlgError:
        return None
    se = np.sqrt(np.diag(XtXi) * (res @ res) / dfr)
    h = np.einsum("ij,jk,ik->i", Xm, XtXi, Xm)
    if not np.all(np.isfinite(h)) or h.max() > OLS_MAX_LEVERAGE:
        return None
    meat = Xm.T @ (Xm * ((res / (1 - h)) ** 2)[:, None])
    se3 = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    if not (np.all(np.isfinite(beta)) and np.all(np.isfinite(se)) and np.all(np.isfinite(se3))):
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t3 = beta / se, beta / se3
    p1, p3 = 2 * t_dist.sf(np.abs(t1), dfr), 2 * t_dist.sf(np.abs(t3), dfr)
    if len(cols) == 1:
        vif = np.array([1.0])
    else:
        try:
            vif = np.diag(np.linalg.inv(np.corrcoef(d[cols].to_numpy().T)))
        except np.linalg.LinAlgError:
            vif = np.full(len(cols), np.inf)
    tab = pd.DataFrame({"term": ["Intercept"] + cols, "coef": beta, "std_err": se, "t_value": t1,
                        "p_value": p1, "p_value_HC3": p3, "VIF": np.r_[np.nan, vif]})
    return {"n": int(n), "n_input": int(n_input), "n_dropped": int(n_input - n), "df_resid": int(dfr),
            "condition_number": cond, "max_leverage": float(h.max()), "table": tab}


# ------------------------------------------------------------------ Selection-aware tools
def signed_z(effect, p):
    """Signed z from a two-sided p. Requires effect and p to come from the same test / the same sign convention. p outside [0,1] or NaN => NaN."""
    effect = np.asarray(effect, float)
    p = np.asarray(p, float)
    if effect.shape != p.shape:
        raise ValueError("effect و p باید هم‌شکل باشند")
    p = np.where((p >= 0) & (p <= 1), p, np.nan)
    p = np.clip(p, 1e-300, 1)
    return np.sign(effect) * norm.isf(p / 2)


def eb_shrink(z, n_iter=300, tol=1e-10, min_n=10):
    """
    Exploratory shrinkage (effect_EB): normal mixture (point null + N(0,tau2)) on the z-scores of the whole screen (EM),
    posterior mean (Tweedie). Returns: factor = E[mu|z]/z, local_fdr = P(null|z), info.
    NOT a "selection-adjusted effect"; in the paper: shrinkage-based exploratory estimate. It does not enter G1/G2/G3.
    Empty input, fewer than min_n finite values, non-convergence or an invalid solution => NaN output and info["converged"]=False.
    """
    z = np.asarray(z, float)
    ok = np.isfinite(z)
    zz = z[ok]
    factor = np.full(len(z), np.nan)
    lfdr = np.full(len(z), np.nan)
    bad = {"pi0": np.nan, "tau2": np.nan, "converged": False, "n_iter": 0, "n_finite": int(ok.sum())}
    if len(zz) < min_n:
        return factor, lfdr, bad
    pi0, tau2 = 0.9, 1.0
    converged, it = False, 0
    for it in range(1, n_iter + 1):
        f0 = norm.pdf(zz, 0, 1)
        f1 = norm.pdf(zz, 0, np.sqrt(1 + tau2))
        w1 = (1 - pi0) * f1 / (pi0 * f0 + (1 - pi0) * f1 + 1e-300)
        pi0_new = float(np.clip(1 - w1.mean(), 0.01, 0.999))
        tau2_new = max((w1 * zz ** 2).sum() / max(w1.sum(), 1e-12) - 1, 1e-6)
        delta = abs(pi0_new - pi0) + abs(tau2_new - tau2)
        pi0, tau2 = pi0_new, tau2_new
        if delta < tol:
            converged = True
            break
    if not (np.isfinite(pi0) and np.isfinite(tau2)):
        return factor, lfdr, bad
    f0 = norm.pdf(zz, 0, 1)
    f1 = norm.pdf(zz, 0, np.sqrt(1 + tau2))
    w1 = (1 - pi0) * f1 / (pi0 * f0 + (1 - pi0) * f1 + 1e-300)
    post = w1 * zz * tau2 / (1 + tau2)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor[ok] = np.where(np.abs(zz) > 1e-9, post / zz, 0.0)
    lfdr[ok] = 1 - w1
    return factor, lfdr, {"pi0": pi0, "tau2": tau2, "converged": converged, "n_iter": it, "n_finite": int(ok.sum())}


def mde_rank_biserial(n1, n2, power=0.8, alpha=0.05):
    """
    The smallest |rank-biserial| detectable with a given power; normal approximation for Mann-Whitney with the variance of U under H0 without ties
    (var(U)=n1*n2*(n1+n2+1)/12 , rb = 1-2U/(n1 n2)). Assumptions: independent samples, normal approximation, power with the same SE as under H0 (approximate);
    ties and the distribution shape are not accounted for.
    """
    if n1 < 2 or n2 < 2 or not (0 < power < 1) or not (0 < alpha < 1):
        return np.nan
    z = norm.isf(alpha / 2) + norm.ppf(power)
    se = np.sqrt((n1 + n2 + 1) / (12.0 * n1 * n2))
    return float(min(1.0, 2 * z * se))


def power_resample(mut_ref, wt_ref, n1, n2, w=1.0, B=2000, alpha=0.05, seed=None):
    """
    Power to detect a same-direction effect with p<alpha in a sample of size (n1,n2) -- exploratory power analysis, not the actual power of the study.
    Mutated values: with probability w drawn from the reference mutated distribution, otherwise from the reference WT (w<1 shrinks the effect by about a factor w).
    NaN if the reference effect is zero/invalid (the same-direction criterion is meaningless).
    """
    B = _pos_int(B, "B")
    n1, n2 = _pos_int(n1, "n1"), _pos_int(n2, "n2")
    if not (0 <= w <= 1):
        raise ValueError("w باید در [0,1] باشد")
    if not (0 < alpha < 1):
        raise ValueError("alpha باید در (0,1) باشد")
    mut_ref, wt_ref = _finite(mut_ref), _finite(wt_ref)
    if len(mut_ref) < 2 or len(wt_ref) < 2:
        return np.nan
    e0 = rank_biserial(mut_ref, wt_ref)
    s0 = np.sign(e0)
    if not np.isfinite(e0) or s0 == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    hit = 0
    for _ in range(B):
        use_mut = rng.random(n1) < w
        a = np.where(use_mut, rng.choice(mut_ref, n1), rng.choice(wt_ref, n1))
        e, p = rb_p(a, rng.choice(wt_ref, n2))
        hit += int(np.isfinite(p) and p < alpha and np.sign(e) == s0)
    return hit / B
