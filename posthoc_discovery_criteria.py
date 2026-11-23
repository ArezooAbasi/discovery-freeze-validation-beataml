# -*- coding: utf-8 -*-
"""
posthoc_discovery_criteria.py -- post hoc descriptive analyses (Methods 2.9.3 - 2.9.8).

POST HOC: run after the freeze and after VALIDATION; no frozen list or classification is changed.

  2.9.3  Table 11   validation rate by DISCOVERY-stage criterion (Wilson 95% CI for the q strata)
  2.9.4  Table 9    adequately powered discordant pairs (+ whether the VALIDATION CI contains the DISCOVERY effect)
  2.9.5  Table 19   estimated false-declaration proportion (null replicates + Storey pi0)
  2.9.6  Table 12/13 logistic regression of validation on G3 (M1-M3) and rates within q strata
  2.9.7  Table 18   BH false declarations involving co-mutated genes (multi-signal simulations)
  2.9.8             directional concordance by gene

Usage:   export BEATAML_OUT=/path/to/output      # must contain DISCOVERY/ and VALIDATION/
         python posthoc_discovery_criteria.py     # writes posthoc_table*.csv to the current directory
Inputs : DISCOVERY/freeze_audit.csv, DISCOVERY/mutation_drug_associations.csv,
         DISCOVERY/simulation_replicates.csv, DISCOVERY/simulation_multi_replicates.csv,
         VALIDATION/validation_report.csv, VALIDATION/validation_report_sensitivity.csv
Needs  : numpy, pandas, scipy, statsmodels
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

BASE = os.environ.get("BEATAML_OUT", ".")


def rd(rel):
    return pd.read_csv(os.path.join(BASE, rel), encoding="utf-8-sig")


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def load():
    rep = pd.concat([rd("VALIDATION/validation_report.csv"),
                     rd("VALIDATION/validation_report_sensitivity.csv")], ignore_index=True)
    aud = rd("DISCOVERY/freeze_audit.csv")[["gene", "drug", "G1_in_main", "G3_adjustment_pass"]]
    rep = rep.merge(aud, on=["gene", "drug"], how="left")
    assert rep.G1_in_main.notna().all()
    ev = rep[rep.flag_evaluated == True].copy()           # noqa: E712
    ev["validated"] = ev.flag_validated == True           # noqa: E712
    ev["G1"] = ev.G1_in_main == True                      # noqa: E712
    ev["G3"] = ev.G3_adjustment_pass == True              # noqa: E712
    ev["is_main"] = ev.in_main == True                    # noqa: E712
    return rep, ev


WILSON_ROWS = {"BH q <= 0.01", "0.01 < q <= 0.05", "0.05 < q <= 0.10"}      # Wilson 95% CI for the q strata only (Section 2.9.3)


def table11(ev):
    assert len(ev) == 246, len(ev)                         # Section 2.9.3
    q = ev.discovery_FDR_q
    crit = [
        ("Whole BH pool (q < 0.10)", ev.index == ev.index),
        ("BH q <= 0.05", q <= 0.05),
        ("BH q <= 0.01", q <= 0.01),
        ("0.01 < q <= 0.05", (q > 0.01) & (q <= 0.05)),
        ("0.05 < q <= 0.10", q > 0.05),
        ("Single-step min-p < 0.05", ev.p_WY < 0.05),
        ("G1 (nominal rule)", ev.G1),
        ("G1 and G3 (frozen main)", ev.G1 & ev.G3),
        ("G1, failing G3", ev.G1 & ~ev.G3),
        ("G3 passed (any G1)", ev.G3),
        ("G3 failed", ~ev.G3),
        ("Failed G1", ~ev.G1),
        ("Failed G1, excluding KRAS", ~ev.G1 & (ev.gene != "KRAS")),
    ]
    rows = []
    for name, m in crit:
        d = ev[m]
        assess = d[d.flag_direction_assessable == True]    # noqa: E712
        lo, hi = wilson(d.validated.sum(), len(d)) if name in WILSON_ROWS else (np.nan, np.nan)
        rows.append(dict(criterion=name, evaluated=len(d), validated=int(d.validated.sum()),
                         validated_pct=round(100 * d.validated.mean(), 1),
                         concordant_pct=round(100 * (assess.flag_concordant == True).mean(), 1),   # noqa: E712
                         wilson_lo=round(lo, 1), wilson_hi=round(hi, 1)))
    t = pd.DataFrame(rows)
    t.to_csv("posthoc_table11_validation_rate_by_criterion.csv", index=False)
    print("\n== Table 11\n", t.to_string(index=False))


def table9(ev):
    d = ev[(ev.interpretation == "discordant_not_significant_adequately_powered") & ev.is_main].copy()
    d["VAL_CI_contains_DISC"] = (d.rank_biserial_ci_low <= d.discovery_effect) & (d.discovery_effect <= d.rank_biserial_ci_high)
    cols = ["gene", "drug", "n_mut_disc", "n_mut_val", "discovery_effect", "discovery_FDR_q", "validation_effect",
            "rank_biserial_ci_low", "rank_biserial_ci_high", "validation_p", "MDE_rank_biserial_80pct",
            "power_at_shrunk_effect", "VAL_CI_contains_DISC"]
    d[cols].to_csv("posthoc_table9_powered_discordant.csv", index=False)
    print("\n== Table 9 (adequately powered discordant, main)\n", d[cols].round(3).to_string(index=False))


def table19():
    s = rd("DISCOVERY/simulation_replicates.csv")
    g = s[(s.scenario == "global_null") & (s.family == "H20")]
    rows = []
    for meth, obs, label in (("MAIN_N", 210, "G1 nominal rule"), ("MAIN_NSA", 139, "G1 + G2 + G3 (frozen main)")):
        x = g[g.method == meth].n_declared.values
        rows.append(dict(rule=label, observed=obs, n_reps=len(x), null_mean=round(x.mean(), 2),
                         null_p95=float(np.percentile(x, 95)), null_max=int(x.max()),
                         est_proportion_pct=round(100 * x.mean() / obs, 1),
                         upper_bound_pct=round(100 * np.percentile(x, 95) / obs, 1)))
    print("\n== Table 19 (null replicates)\n", pd.DataFrame(rows).to_string(index=False))
    h20 = g.n_H20.drop_duplicates()
    print("H20 family size per replicate: mean %.0f, range %d-%d" % (g.groupby("rep").n_H20.first().mean(),
                                                                    g.groupby("rep").n_H20.first().min(),
                                                                    g.groupby("rep").n_H20.first().max()))
    print("MAIN_N share of H20 declared (nominal 1%%): %.2f%%" % (100 * g[g.method == "MAIN_N"].n_declared.mean() / g.groupby("rep").n_H20.first().mean()))
    # Storey pi0 on the observed H20 family
    a = rd("DISCOVERY/mutation_drug_associations.csv")
    h = a[(a.n_mutated_with_AUC >= 20) & (a.n_wt_with_AUC >= 20)]
    m = len(h)
    pv = h.p_value.values
    st = []
    for lam in (0.5, 0.6, 0.7, 0.8, 0.9):
        pi0 = min(1.0, (pv > lam).sum() / ((1 - lam) * m))
        st.append(dict(lambda_=lam, pi0=round(pi0, 3), expected_false=round(pi0 * m * 0.01, 2),
                       pct_of_210=round(100 * pi0 * m * 0.01 / 210, 1)))
    st = pd.DataFrame(st)
    print("\n== Storey, m =", m, "\n", st.to_string(index=False), "\nmean pi0 = %.3f" % st.pi0.mean())
    pd.DataFrame(rows).to_csv("posthoc_table19_null_counts.csv", index=False)
    st.to_csv("posthoc_table19_storey.csv", index=False)


def logistic(ev):
    y = ev.validated.astype(float).values
    g3 = ev.G3.astype(float).values
    lq = np.log10(ev.discovery_FDR_q.values)
    ln = np.log(ev.n_mut_val.values.astype(float))
    models = {"M1: G3 only": [g3], "M2: + log10 discovery BH q": [g3, lq], "M3: M2 + log mutated n in VALIDATION": [g3, lq, ln]}
    rows = []
    for name, cols in models.items():
        X = sm.add_constant(np.column_stack(cols))
        r = sm.Logit(y, X).fit(disp=0)
        ci = r.conf_int()[1]
        rows.append(dict(model=name, OR=round(float(np.exp(r.params[1])), 2),
                         ci_lo=round(float(np.exp(ci[0])), 2), ci_hi=round(float(np.exp(ci[1])), 2),
                         p=round(float(r.pvalues[1]), 3)))
    t = pd.DataFrame(rows)
    t.to_csv("posthoc_table12_logistic_G3.csv", index=False)
    print("\n== Table 12\n", t.to_string(index=False))
    q = ev.discovery_FDR_q
    strata = [("<= 0.01", q <= 0.01), ("0.01-0.05", (q > 0.01) & (q <= 0.05)), ("0.05-0.10", q > 0.05), ("All", q == q)]
    r13 = []
    for nm, m in strata:
        d = ev[m]
        a, b = d[d.G3], d[~d.G3]
        r13.append(dict(q_stratum=nm, G3_passed="%d/%d (%.1f%%)" % (a.validated.sum(), len(a), 100 * a.validated.mean()),
                        G3_failed="%d/%d (%.1f%%)" % (b.validated.sum(), len(b), 100 * b.validated.mean())))
    t13 = pd.DataFrame(r13)
    t13.to_csv("posthoc_table13_G3_within_q_strata.csv", index=False)
    print("\n== Table 13\n", t13.to_string(index=False))


def table18():
    m = rd("DISCOVERY/simulation_multi_replicates.csv")
    b = m[(m.method == "BH") & (m.family == "H20")]
    rows = []
    for sc in ("multi_alt_m10", "multi_alt_m20", "multi_alt_m30"):
        d = b[b.scenario == sc]
        R, V, Vco = int(d.n_declared.sum()), int(d.n_false.sum()), int(d.n_comutation_associated_false.sum())
        fdp = d.fdp.values
        adj = (d.n_false - d.n_comutation_associated_false) / np.maximum(d.n_declared, 1)
        rows.append(dict(m=int(d.n_planted.iloc[0]), reps=len(d), R=R, V=V, V_co=Vco, Vco_over_V_pct=round(100 * Vco / V, 1),
                         reps_with_Vco=int((d.n_comutation_associated_false > 0).sum()),
                         BH_FDR=round(fdp.mean(), 3), FDR_if_Vco_not_false=round(adj.mean(), 3),
                         SE=round(adj.std(ddof=1) / np.sqrt(len(d)), 3), remaining_false=V - Vco))
    t = pd.DataFrame(rows)
    t.to_csv("posthoc_table18_BH_comutated_false.csv", index=False)
    print("\n== Table 18\n", t.to_string(index=False))


def concord_by_gene(ev):
    d = ev[ev.flag_direction_assessable == True]            # noqa: E712
    t = d.groupby("gene").agg(assessable=("flag_concordant", "size"),
                              concordant=("flag_concordant", lambda s: int((s == True).sum()))).reset_index()  # noqa: E712
    t["pct"] = (100 * t.concordant / t.assessable).round(0)
    t = t[t.assessable >= 5].sort_values("pct", ascending=False)
    t.to_csv("posthoc_concordance_by_gene.csv", index=False)
    print("\n== Concordance by gene (>=5 assessable); both tracks total: %d/%d" % (
        (d.flag_concordant == True).sum(), len(d)))                                          # noqa: E712
    print(t.to_string(index=False))


def main():
    rep, ev = load()
    table11(ev)
    table9(ev)
    logistic(ev)
    concord_by_gene(ev)
    table19()
    table18()


if __name__ == "__main__":
    if "BEATAML_OUT" not in os.environ:
        sys.exit("Set BEATAML_OUT to the output folder (it must contain DISCOVERY/ and VALIDATION/).")
    main()
