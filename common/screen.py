# -*- coding: utf-8 -*-
"""screen.py -- vectorised gene x drug screen + Westfall-Young min-p (selection-aware)."""
import numpy as np
import pandas as pd
from . import config_official as cfg
from . import stats


def screen_arrays(D, G, min_drug=None, min_mut_auc=None, min_wt_auc=None):
    """D: patients x drugs (float, NaN allowed); G: patients x genes (0/1). Returns a dict of arrays."""
    min_drug = cfg.MIN_DRUG_SAMPLES if min_drug is None else min_drug
    min_mut_auc = cfg.MIN_MUTATED_WITH_AUC if min_mut_auc is None else min_mut_auc
    min_wt_auc = cfg.MIN_WT_WITH_AUC if min_wt_auc is None else min_wt_auc
    gi, di, n1s, n2s, ps, rbs = [], [], [], [], [], []
    for j in range(D.shape[1]):
        v = D[:, j]
        ok = ~np.isnan(v)
        if ok.sum() < min_drug:
            continue
        n1, n2, _, p, rb = stats.mwu_columns(v[ok], G[ok])
        keep = (n1 >= min_mut_auc) & (n2 >= min_wt_auc)
        if keep.any():
            k = np.where(keep)[0]
            gi.append(k); di.append(np.full(len(k), j))
            n1s.append(n1[k]); n2s.append(n2[k]); ps.append(p[k]); rbs.append(rb[k])
    if not gi:
        e = np.array([])
        return dict(gi=e.astype(int), di=e.astype(int), n1=e, n2=e, p=e, rb=e)
    return dict(gi=np.concatenate(gi), di=np.concatenate(di), n1=np.concatenate(n1s),
                n2=np.concatenate(n2s), p=np.concatenate(ps), rb=np.concatenate(rbs))


def prepare(drug, mut, min_mutated=None):
    """Align the matrices and filter genes. Returns D, G, genes, drugs, index."""
    min_mutated = cfg.MIN_MUTATED if min_mutated is None else min_mutated
    idx = drug.index.intersection(mut.index)
    m = mut.loc[idx]
    genes = m.columns[m.sum(0) >= min_mutated]
    return (drug.loc[idx].to_numpy(float), m[genes].to_numpy(float),
            np.asarray(genes), np.asarray(drug.columns), idx)


def screen_matrices(drug, mut, **kw):
    D, G, genes, drugs, _ = prepare(drug, mut)
    a = screen_arrays(D, G, **kw)
    return pd.DataFrame({"gene": genes[a["gi"]], "drug": drugs[a["di"]],
                         "n_mutated_with_AUC": a["n1"].astype(int), "n_wt_with_AUC": a["n2"].astype(int),
                         "p_value": a["p"], "rank_biserial_effect": a["rb"]})


def westfall_young_minp(D, G, gi, di, p_obs, B, seed):
    """
    Westfall-Young single-step min-p (FWER) over the observed family.
    Permutation: rows of the drug matrix are shuffled jointly (the co-mutation structure, drug-drug correlation and NaN pattern are preserved;
    only the gene->AUC link is broken). p_adj = (1 + #{b: minp_b <= p_k}) / (B + 1).
    """
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    by_drug = {j: gi[di == j] for j in np.unique(di)}
    minp = np.ones(B)
    for b in range(B):
        Dp = D[rng.permutation(n)]
        m = 1.0
        for j, cols in by_drug.items():
            v = Dp[:, j]
            ok = ~np.isnan(v)
            if ok.sum() < 6:
                continue
            n1, n2, _, p, _ = stats.mwu_columns(v[ok], G[ok][:, cols])
            p = np.where((n1 >= 3) & (n2 >= 3), p, 1.0)
            m = min(m, p.min())
        minp[b] = m
    return np.array([(1 + np.sum(minp <= pk)) / (B + 1) for pk in p_obs]), minp
