# -*- coding: utf-8 -*-
"""
08c_global_sensitivity_diagnostic.py   (diagnostic; DISCOVERY only; not a gate)
Question: how much of a gene's effect on one drug is a "shared effect across all drugs" (overall sample sensitivity)?
  global_index(patient) = mean column-wise z-score of AUC over all OTHER drugs (the tested drug is left out).
Output: AUC ~ gene  and  AUC ~ gene + global_index  (OLS, HC3).
Interpretation warning: if the gene itself affects many drugs, global_index absorbs part of the gene effect (over-adjustment).
So a smaller beta is not conclusive evidence of confounding; it only shows how much of the effect is shared across drugs.
  python 08c_global_sensitivity_diagnostic.py --cohort DISCOVERY
"""
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from common import data, stats

MIN_OTHER_DRUGS = 20


def global_index(drug, exclude):
    z = (drug - drug.mean()) / drug.std(ddof=0)
    z = z.drop(columns=[exclude])
    idx = z.mean(axis=1)
    idx[z.notna().sum(axis=1) < MIN_OTHER_DRUGS] = np.nan
    return idx


def main():
    cohort = data.parse_cohort("global sensitivity diagnostic")
    st = data.read(cohort, "stable_pairs.csv", hint="اول 04_export_stable_pairs.py")
    cols = ["gene", "drug", "n_mutated", "n_wt", "n_model_unadjusted", "n_model_adjusted",
            "n_dropped_missing_global_index", "n_eligible_mutant", "n_eligible_wt",
            "n_mutant_missing_global", "n_wt_missing_global", "pct_mutant_missing_global", "pct_wt_missing_global",
            "missingness_difference_pp", "missingness_fisher_p", "beta_unadj", "p_unadj", "beta_global_adj", "p_HC3_global_adj",
            "pct_change_abs_beta", "sign_flip", "gene_vs_global_index_rb", "gene_vs_global_index_p", "n_stable_pairs_same_gene"]
    if st.empty:
        data.write(pd.DataFrame(columns=cols), cohort, "global_sensitivity_diagnostic.csv"); return
    mut, drug = data.load_cohort_matrices(cohort)
    per_gene = st.groupby("gene").size().to_dict()
    rows = []
    for _, r in st.iterrows():
        g, d = r["gene"], r["drug"]
        y = drug[d].reindex(mut.index)
        gi = global_index(drug, d).reindex(mut.index)
        n_with_auc = int(y.notna().sum())
        el = pd.DataFrame({"AUC": y, "mut": mut[g].astype(float), "gi": gi}).dropna(subset=["AUC", "mut"])
        mm, wm = el[el.mut == 1], el[el.mut == 0]
        n_mm, n_wm = int(mm.gi.isna().sum()), int(wm.gi.isna().sum())
        pct_m, pct_w = 100 * n_mm / max(len(mm), 1), 100 * n_wm / max(len(wm), 1)
        p_miss = fisher_exact([[n_mm, len(mm) - n_mm], [n_wm, len(wm) - n_wm]])[1]
        # both models are fitted on exactly ONE patient set (complete-case)
        cc = pd.concat([y.rename("AUC"), mut[g].astype(float).rename(g), gi.rename("global_sensitivity")], axis=1) \
            .replace([np.inf, -np.inf], np.nan).dropna()
        if cc.empty:
            continue
        o0 = stats.ols(cc["AUC"], cc[[g]])
        o1 = stats.ols(cc["AUC"], cc[[g, "global_sensitivity"]])
        if o0 is None or o1 is None:
            continue
        assert o0["n"] == o1["n"], "n مدل‌ها برابر نیست"
        assert n_with_auc - o1["n"] == n_mm + n_wm, "شمارش missingness با complete-case نمی‌خواند"
        t0, t1 = o0["table"].set_index("term"), o1["table"].set_index("term")
        m = cc[g] == 1
        rb, p = stats.rb_p(cc.loc[m, "global_sensitivity"].to_numpy(), cc.loc[~m, "global_sensitivity"].to_numpy())
        b0, b1 = t0.loc[g, "coef"], t1.loc[g, "coef"]
        rows.append({"gene": g, "drug": d, "n_mutated": int(m.sum()), "n_wt": int((~m).sum()), "n_model_unadjusted": o0["n"],
                     "n_model_adjusted": o1["n"], "n_dropped_missing_global_index": n_with_auc - o1["n"],
                     "n_eligible_mutant": len(mm), "n_eligible_wt": len(wm), "n_mutant_missing_global": n_mm,
                     "n_wt_missing_global": n_wm, "pct_mutant_missing_global": pct_m, "pct_wt_missing_global": pct_w,
                     "missingness_difference_pp": pct_m - pct_w, "missingness_fisher_p": p_miss,
                     "beta_unadj": b0, "p_unadj": t0.loc[g, "p_value_HC3"],
                     "beta_global_adj": b1, "p_HC3_global_adj": t1.loc[g, "p_value_HC3"],
                     "pct_change_abs_beta": 100 * (abs(b1) - abs(b0)) / abs(b0) if b0 else np.nan,
                     "sign_flip": bool(np.sign(b1) != np.sign(b0)),
                     "gene_vs_global_index_rb": rb, "gene_vs_global_index_p": p,
                     "n_stable_pairs_same_gene": per_gene[g]})
    out = pd.DataFrame(rows, columns=cols)
    data.write(out, cohort, "global_sensitivity_diagnostic.csv")
    # panel structure: number of drugs measured per patient, split by gene status
    nmeas = drug.notna().sum(axis=1)
    prow = []
    for g in sorted(st["gene"].unique()):
        m = mut[g].reindex(nmeas.index) == 1
        pp = mannwhitneyu(nmeas[m], nmeas[~m])[1] if m.sum() > 1 and (~m).sum() > 1 else np.nan
        prow.append({"gene": g, "n_mut": int(m.sum()), "n_wt": int((~m).sum()),
                     "median_drugs_measured_mut": float(nmeas[m].median()), "median_drugs_measured_wt": float(nmeas[~m].median()),
                     "n_mut_below_min": int((nmeas[m] <= MIN_OTHER_DRUGS).sum()), "n_wt_below_min": int((nmeas[~m] <= MIN_OTHER_DRUGS).sum()),
                     "MWU_p_panel_size": pp})
    data.write(pd.DataFrame(prow), cohort, "panel_size_by_gene.csv")
    print(out[["gene", "drug", "beta_unadj", "beta_global_adj", "pct_change_abs_beta", "sign_flip", "p_HC3_global_adj",
               "gene_vs_global_index_rb"]].round(3).to_string(index=False))
    print("\nبتا<0 یعنی mutant حساس‌تر (AUC پایین‌تر). pct_change_abs_beta نزدیک -100 (اندازه‌ی اثر نصف/صفر شده) یعنی اثر تقریباً کاملاً مشترک بین داروهاست.")


if __name__ == "__main__":
    main()
