# -*- coding: utf-8 -*-
"""
08_adjusted_analysis.py
=========================================================================================
For each stable pair: AUC ~ gene after adjusting for co-mutation confounders (automatic screening; logic in common/adjust.py).
Mind the sign: beta is on AUC, so beta > 0 means mutated is more resistant (opposite to the rank_biserial sign).
The `interpretation` label is the single source of truth (the master table does not recompute it).
  python 08_adjusted_analysis.py --cohort DISCOVERY
"""
import pandas as pd
from common import data, adjust


def main():
    cohort = data.parse_cohort("adjusted analysis")
    st = data.read(cohort, "stable_pairs.csv", hint="اول 04_export_stable_pairs.py")
    cols = ["gene", "drug", "n_mutated", "n_wt", "max_covariates_allowed", "unadjusted_beta", "unadjusted_p",
            "confounders_included", "adjusted_beta", "adjusted_p", "adjusted_p_HC3", "max_VIF", "pct_change_beta",
            "interpretation", "adjustment_status"]   # adjustment_status = coarse_status(interpretation); needed by 08b/13
    if st.empty:
        data.write(pd.DataFrame(columns=cols), cohort, "adjusted_analysis_results.csv")
        return
    mut, drug = data.load_cohort_matrices(cohort)
    rows = []
    for _, r in st.iterrows():
        res = adjust.adjust_pair(r["gene"], drug[r["drug"]].dropna(), mut)
        rows.append({"gene": r["gene"], "drug": r["drug"], **res})
        print(f"{r['gene']} x {r['drug']}: conf=[{res['confounders_included']}] -> {res['interpretation']}")
    data.write(pd.DataFrame(rows)[cols], cohort, "adjusted_analysis_results.csv")


if __name__ == "__main__":
    main()
