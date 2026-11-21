# -*- coding: utf-8 -*-
"""adjust.py -- exploratory, data-adaptive co-mutation screening + adjusted OLS (Stage 08 and the simulation).
The label robust_to_adjustment means "the effect survived this exploratory adjustment", not proof that confounding is absent."""
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from . import config_official as cfg
from . import stats


def screen_confounders(gene, mut, y):
    """
    y: Series AUC (index = patient). mut: patients x genes.
    candidate = a gene that both co-occurs with the tested gene (Fisher) and is associated with AUC (Mann-Whitney U); both p < threshold.
    Sorted by the mean of the two p-values (ascending).
    """
    sub = mut.loc[mut.index.isin(y.index)]
    freq = sub.sum(0)
    cands = freq[(freq >= cfg.MIN_GENE_MUTATION_COUNT) & (freq.index != gene)].index
    g = sub[gene].to_numpy() == 1
    yy = y.reindex(sub.index).to_numpy(float)
    rows = []
    for c in cands:
        cm = sub[c].to_numpy() == 1
        a, b = int((g & cm).sum()), int((g & ~cm).sum())
        cc, d = int((~g & cm).sum()), int((~g & ~cm).sum())
        _, pf = fisher_exact([[a, b], [cc, d]])
        if pf >= cfg.COOCCUR_SCREEN_P:
            continue
        if cm.sum() < 3 or (~cm).sum() < 3:
            continue
        po = mannwhitneyu(yy[cm], yy[~cm], alternative="two-sided")[1]
        if po < cfg.OUTCOME_SCREEN_P:
            rows.append({"candidate_gene": c, "n_mutated_in_cohort": int(freq[c]),
                         "cooccurrence_fisher_p": pf, "outcome_association_p": po,
                         "evidence_score": (pf + po) / 2})
    cols = ["candidate_gene", "n_mutated_in_cohort", "cooccurrence_fisher_p", "outcome_association_p", "evidence_score"]
    return pd.DataFrame(rows, columns=cols).sort_values("evidence_score") if rows else pd.DataFrame(columns=cols)


def classify(unadj_coef, unadj_p, adj_coef, adj_p, pct_change):
    if np.sign(adj_coef) != np.sign(unadj_coef):
        return "sign_flip_after_adjustment_likely_confounded"
    # Only a REDUCTION of the effect size is evidence of confounding. An increase after adjustment (suppression) is not attenuation.
    shrink = abs(adj_coef) / abs(unadj_coef) if unadj_coef not in (0, None) else np.nan
    if np.isfinite(shrink) and shrink <= 1 - cfg.ADJ_ATTENUATION_PCT / 100:
        return "attenuated_>=50pct_possible_partial_confounding"
    if adj_p >= 0.05 and unadj_p < 0.05:
        return "loses_nominal_significance_after_adjustment"
    return "robust_to_adjustment"


def _adjust_pair_core(gene, y, mut):
    """
    y: AUC Series (patients that are profiled and have AUC); mut: patients x genes (aligned).
    Returns a dict with unadjusted/adjusted beta and the interpretation label (ADJ_PASS_LABELS = pass).
    """
    y = y[y.index.isin(mut.index)].astype(float)
    order = list(y.index)
    tgt = pd.Series(mut.loc[order, gene].to_numpy(), index=order, name=gene)
    n_mut, n_wt = int(tgt.sum()), int((tgt == 0).sum())
    max_cov = min(cfg.MAX_COVARIATES_HARD_CAP, min(n_mut, n_wt) // cfg.MIN_N_PER_PREDICTOR)
    out = {"n_mutated": n_mut, "n_wt": n_wt, "max_covariates_allowed": max_cov,
           "unadjusted_beta": np.nan, "unadjusted_p": np.nan, "confounders_included": "",
           "adjusted_beta": np.nan, "adjusted_p": np.nan, "adjusted_p_HC3": np.nan,
           "max_VIF": np.nan, "pct_change_beta": np.nan, "interpretation": "unadjusted_model_not_estimable"}
    un = stats.ols(y, tgt.to_frame())
    if un is None:
        return out
    ur = un["table"].query("term == @gene").iloc[0]
    out.update(unadjusted_beta=ur["coef"], unadjusted_p=ur["p_value"])
    if max_cov == 0:
        out["interpretation"] = "insufficient_n_for_adjustment"
        return out
    cand = screen_confounders(gene, mut, y)
    if cand.empty:
        out["interpretation"] = "no_confounder_candidate_found"
        return out
    chosen = cand["candidate_gene"].head(max_cov).tolist()
    out["confounders_included"] = ";".join(chosen)
    X = tgt.to_frame()
    for c in chosen:
        X[c] = mut.loc[order, c].to_numpy()
    adj = stats.ols(y, X)
    if adj is None:
        out["interpretation"] = "adjusted_model_not_estimable"
        return out
    ar = adj["table"].query("term == @gene").iloc[0]
    hc3 = cfg.ADJ_P_SOURCE == "HC3"
    pu, pa = (ur["p_value_HC3"], ar["p_value_HC3"]) if hc3 else (ur["p_value"], ar["p_value"])
    pct = 100 * (ar["coef"] - ur["coef"]) / abs(ur["coef"]) if ur["coef"] != 0 else np.nan
    out.update(adjusted_beta=ar["coef"], adjusted_p=ar["p_value"], adjusted_p_HC3=ar["p_value_HC3"],
               max_VIF=float(adj["table"].loc[adj["table"]["term"] != "Intercept", "VIF"].max()),
               pct_change_beta=pct,
               interpretation=classify(ur["coef"], pu, ar["coef"], pa, pct))
    return out


# Coarse status (4 states). Only robust_to_adjustment passes; no_confounder and not_assessable do not pass.
_NOT_ASSESSABLE = {"insufficient_n_for_adjustment", "unadjusted_model_not_estimable", "adjusted_model_not_estimable"}


def coarse_status(label):
    if label == "robust_to_adjustment":
        return "robust_to_adjustment"
    if label == "no_confounder_candidate_found":
        return "no_confounder_candidate_found"
    if label in _NOT_ASSESSABLE:
        return "adjustment_not_assessable"
    return "attenuated_or_not_robust"


def adjust_pair(gene, y, mut):
    """Exploratory, data-adaptive. Returns a dict + adjustment_status (coarse_status)."""
    out = _adjust_pair_core(gene, y, mut)
    out["adjustment_status"] = coarse_status(out["interpretation"])
    return out
