# -*- coding: utf-8 -*-
"""
posthoc_validation_analysis.py -- post hoc validation analyses (Methods 2.10).

These analyses are POST HOC: they run after the freeze and after VALIDATION, they do not take part in
selection, and they do not modify any frozen output.

  2.10.1  perm : chance baseline. The AUC rows of the VALIDATION cohort are permuted jointly across patients
                 (genotype structure fixed, gene->AUC link broken) and the number of "validated" main-track pairs
                 (BH q < 0.10 within the reproducible subset AND same direction as Discovery) is counted.
  2.10.2  flt3 : FLT3 validation with WES-only labels (clinical ITD removed) instead of the composite FLT3 definition,
                 in two variants: ITD-only patients treated as non-mutated, or excluded.

Usage (run from anywhere; outputs of `perm` are written to the current directory):
    export BEATAML_OUT=/path/to/output_report     # must contain FULL/, manifest/ and VALIDATION/
    python posthoc_validation_analysis.py perm 2000     # number of permutations (default 2000)
    python posthoc_validation_analysis.py flt3

Inputs : FULL/analysis_population.csv, manifest/patient_level_cohort_manifest.csv,
         manifest/flt3_composite_provenance_VALIDATION.csv (flt3 only),
         VALIDATION/validation_report.csv, VALIDATION/validation_report_sensitivity.csv
Outputs: permutation_baseline_summary.json, permutation_null_counts.csv   (perm only)
Needs  : numpy, pandas, scipy, statsmodels
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm, rankdata
from statsmodels.stats.multitest import multipletests

SEED = 20261112          # same as RANDOM_SEED in common/config_official.py
MIN_N = 10               # minimum n per group (MIN_MUTATED_WITH_AUC / MIN_WT_WITH_AUC)
VALIDATION_FDR = 0.10    # same as VALIDATION_FDR in common/config_official.py


def _read(rel):
    return pd.read_csv(os.path.join(BASE, rel), encoding="utf-8-sig")


def _load():
    """Load the VALIDATION mutation/AUC matrices and the table of evaluated frozen pairs."""
    pop = _read("FULL/analysis_population.csv")
    man = _read("manifest/patient_level_cohort_manifest.csv")
    pats = sorted(man[man.in_VALIDATION == True].patient_id)  # noqa: E712
    v = pop[pop.patient_id.isin(pats)]

    mut = (v.drop_duplicates(["patient_id", "gene"])
             .pivot(index="patient_id", columns="gene", values="mutation_status").reindex(pats))
    # every patient-drug pair must carry a single AUC value
    assert (v.groupby(["patient_id", "drug"]).AUC.nunique() <= 1).all()
    auc = (v.drop_duplicates(["patient_id", "drug"])
             .pivot(index="patient_id", columns="drug", values="AUC").reindex(pats))

    report = pd.concat([_read("VALIDATION/validation_report.csv"),
                        _read("VALIDATION/validation_report_sensitivity.csv")], ignore_index=True)
    evaluated = report[report.flag_evaluated == True].reset_index(drop=True)  # noqa: E712
    return mut, auc, evaluated


def test(mut, auc):
    """Mann-Whitney (scipy, asymptotic). Returns (rank-biserial, p, n_mutated, n_wt); NaN if a group has n < MIN_N."""
    ok = ~np.isnan(auc) & ~np.isnan(mut)
    a, b = auc[ok & (mut == 1)], auc[ok & (mut == 0)]
    if len(a) < MIN_N or len(b) < MIN_N:
        return np.nan, np.nan, len(a), len(b)
    u = mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
    return 1 - 2 * u.statistic / (len(a) * len(b)), u.pvalue, len(a), len(b)


def mwu_fast(a, b):
    """Fast asymptotic Mann-Whitney with tie and continuity correction (matches scipy's asymptotic two-sided test).
    Returns (rank-biserial, p)."""
    n1, n2 = len(a), len(b)
    x = np.concatenate([a, b])
    r = rankdata(x)
    u1 = r[:n1].sum() - n1 * (n1 + 1) / 2
    _, cnt = np.unique(x, return_counts=True)
    n = n1 + n2
    tie = (cnt ** 3 - cnt).sum()
    s = np.sqrt(n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1))))
    mu = n1 * n2 / 2
    z = (max(u1, n1 * n2 - u1) - mu - 0.5) / s
    p = min(1.0, 2 * norm.sf(z))
    return 1 - 2 * u1 / (n1 * n2), p


def _validated(effects, pvals, discovery):
    """BH within the given subset; validated = q < VALIDATION_FDR and same direction as Discovery."""
    ok = ~np.isnan(pvals)
    q = np.full(len(pvals), np.nan)
    q[ok] = multipletests(pvals[ok], method="fdr_bh")[1]
    concordant = (np.sign(effects) == np.sign(discovery)) & (effects != 0)
    return (q < VALIDATION_FDR) & concordant & ok, q


def run_perm(mut, auc, ev):
    """2.10.1 -- chance baseline for the number of validated pairs."""
    rows = []
    for _, r in ev.iterrows():
        if r.gene in mut and r.drug in auc:
            rows.append(test(mut[r.gene].values.astype(float), auc[r.drug].values)[:2])
        else:
            rows.append((np.nan, np.nan))
    ev["rb"] = [x[0] for x in rows]
    ev["p"] = [x[1] for x in rows]
    ev["match"] = (ev.rb - ev.validation_effect).abs() < 1e-9      # only pairs that are reproduced exactly
    main = ev[(ev.rule_track == "main") & ev.match].reset_index(drop=True)
    print("main pairs reproducible:", len(main))

    disc = main.discovery_effect.values
    v0, _ = _validated(main.rb.values, main.p.values, disc)
    print("observed validated in subset (BH within subset):", v0.sum(), "| original flag:", main.flag_validated.sum())

    genes = sorted(set(main.gene))
    drugs = sorted(set(main.drug))
    gi = {g: i for i, g in enumerate(genes)}
    di = {d: i for i, d in enumerate(drugs)}
    mat_mut = mut[genes].values.astype(float)
    mat_auc = auc[drugs].values
    g_idx = main.gene.map(gi).values
    d_idx = main.drug.map(di).values

    rng = np.random.default_rng(SEED)
    n_perm = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    counts, conc_frac, gene_counts = [], [], []
    for _ in range(n_perm):
        a_perm = mat_auc[rng.permutation(len(mat_auc))]            # joint row permutation of the drug matrix
        es = np.full(len(main), np.nan)
        ps = es.copy()
        for k in range(len(main)):
            y, m = a_perm[:, d_idx[k]], mat_mut[:, g_idx[k]]
            ok = ~np.isnan(y) & ~np.isnan(m)
            a, b = y[ok & (m == 1)], y[ok & (m == 0)]
            if len(a) < MIN_N or len(b) < MIN_N:
                continue
            es[k], ps[k] = mwu_fast(a, b)
        v, _ = _validated(es, ps, disc)
        counts.append(int(v.sum()))
        ok = ~np.isnan(ps)
        conc_frac.append(int((((np.sign(es) == np.sign(disc)) & (es != 0)) & ok).sum()) / max(ok.sum(), 1))
        gene_counts.append(len(set(main.gene[v])))

    counts = np.array(counts)
    out = dict(
        n_pairs=len(main), observed=int(v0.sum()), NP=n_perm,
        null_mean=float(counts.mean()), null_median=float(np.median(counts)), null_max=int(counts.max()),
        null_p95=float(np.percentile(counts, 95)), null_p99=float(np.percentile(counts, 99)),
        p_ge_obs=float(((counts >= v0.sum()).sum() + 1) / (n_perm + 1)),
        null_mean_concordance=float(np.mean(conc_frac)), P_any=float((counts > 0).mean()),
        null_mean_genes=float(np.mean(gene_counts)),
    )
    print(out)
    with open("permutation_baseline_summary.json", "w") as fh:
        json.dump(out, fh)
    pd.Series(counts).to_csv("permutation_null_counts.csv", index=False)


def run_flt3(mut, auc, ev):
    """2.10.2 -- FLT3 validation with WES-only labels."""
    prov = (_read("manifest/flt3_composite_provenance_VALIDATION.csv")
            .set_index("patient_id").reindex(mut.index))
    composite = mut["FLT3"].values.astype(float)
    wes = prov.wes_positive.astype(float).values
    itd_only = (composite == 1) & (wes == 0)
    wes_itd_as_neg = wes.copy()                 # ITD-only patients become non-mutated
    wes_itd_excluded = wes.copy()               # ITD-only patients are removed from both groups
    wes_itd_excluded[itd_only] = np.nan

    main = ev[ev.rule_track == "main"].reset_index(drop=True)
    disc = main.discovery_effect.values
    is_flt3 = (main.gene == "FLT3").values

    def one(r, labels):
        if r.drug not in auc or r.gene not in mut:
            return np.nan, np.nan
        m = mut[r.gene].values.astype(float) if r.gene != "FLT3" else labels
        return test(m, auc[r.drug].values)[:2]

    def summarise(labels, name):
        res = np.array([one(r, labels) for _, r in main.iterrows()])
        e, p = res[:, 0].copy(), res[:, 1].copy()
        # non-FLT3 pairs keep their original validation result (they enter the BH family unchanged)
        miss = (~is_flt3) & np.isnan(p)
        e[miss] = main.validation_effect.values[miss]
        p[miss] = main.validation_p.values[miss]
        ok = ~np.isnan(p)
        q = np.full(len(p), np.nan)
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
        conc = (np.sign(e) == np.sign(disc)) & (e != 0)
        fo = is_flt3 & ok
        val = (q < VALIDATION_FDR) & conc & ok
        print(name, "| FLT3 evaluable", fo.sum(), "| concordant", (conc & fo).sum(),
              "| validated", (val & is_flt3).sum(),
              "| median sign-aligned r %.3f" % np.nanmedian((np.sign(disc) * e)[fo]))

    summarise(composite, "composite")
    summarise(wes_itd_as_neg, "WES-only (ITD-only -> non-mutated)")
    summarise(wes_itd_excluded, "WES-only (ITD-only excluded)")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("perm", "flt3"):
        print(__doc__)
        return
    mut, auc, ev = _load()
    (run_perm if sys.argv[1] == "perm" else run_flt3)(mut, auc, ev)


if "BEATAML_OUT" not in os.environ and __name__ == "__main__" and len(sys.argv) > 1:
    raise SystemExit("Set the environment variable BEATAML_OUT to the output folder "
                     "(it must contain FULL/, manifest/ and VALIDATION/).")
BASE = os.environ.get("BEATAML_OUT", ".") + os.sep

if __name__ == "__main__":
    main()
