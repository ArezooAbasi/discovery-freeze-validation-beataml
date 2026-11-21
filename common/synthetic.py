# -*- coding: utf-8 -*-
"""synthetic.py -- synthetic data for code testing only (BEATAML_SYNTHETIC=1). It has no relation to the real data."""
import numpy as np
import pandas as pd

_CACHE = {}


def build(seed=7):
    if seed in _CACHE:
        return _CACHE[seed]
    rng = np.random.default_rng(seed)
    n = 530
    pid = [f"{2000 + i}" for i in range(n)]
    wave = np.array(["W12"] * 307 + ["W34"] * 190 + ["BOTH"] * 33)
    names = ["FLT3", "NPM1", "DNMT3A", "RUNX1", "TET2", "IDH1", "IDH2", "NRAS", "KRAS", "TP53", "ASXL1", "SRSF2",
             "WT1", "PTPN11", "CEBPA", "STAG2", "SF3B1", "U2AF1", "BCOR", "EZH2"] + [f"GENE{i}" for i in range(40)]
    freq = np.r_[[.25, .30, .22, .12, .12, .07, .10, .12, .05, .08, .08, .08, .06, .06, .06, .05, .04, .04, .04, .03],
                 rng.uniform(.025, .07, 40)]
    M = (rng.random((n, len(names))) < freq).astype(int)
    M[:, 3] = np.where(rng.random(n) < .6, M[:, 0], M[:, 3])          # RUNX1 co-occurs with FLT3
    mut = pd.DataFrame(M, index=pid, columns=names)
    drugs = ["Midostaurin", "Gilteritinib (ASP2215)", "Venetoclax", "Drug_NPM1", "Drug_W12only"] + [f"Inhib_{i}" for i in range(75)]
    A = rng.normal(150, 45, (n, len(drugs))) + rng.normal(0, 25, (n, 1))     # patient-level shared effect
    A[:, 0] -= 70 * M[:, 0]                                        # FLT3 -> Midostaurin
    A[:, 1] -= 60 * M[:, 0]
    A[:, 3] -= 45 * M[:, 1]                                        # NPM1 -> Drug_NPM1
    w12 = wave == "W12"
    A[:, 4] -= 60 * M[:, 2] * w12                                  # DNMT3A effect only in W12 (should fail validation)
    miss = rng.random((n, len(drugs))) < rng.uniform(.1, .5, len(drugs))
    A[miss] = np.nan
    drug = pd.DataFrame(A, index=pid, columns=drugs)
    profiled = rng.random(n) > .04
    _CACHE[seed] = (mut.loc[profiled], drug, pd.Series(wave, index=pid))
    return _CACHE[seed]


def cohort_matrices(cohort):
    mut, drug, wave = build()
    keep = {"FULL": wave.index, "DISCOVERY": wave.index[wave == "W12"], "VALIDATION": wave.index[wave == "W34"]}[cohort]
    return mut.loc[mut.index.isin(keep)], drug.loc[drug.index.isin(keep)]
