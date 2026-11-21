# -*- coding: utf-8 -*-
"""
11_analysis_population.py
=========================================================================================
Patient-level file for figures/audit: for every stable pair, all profiled patients with an AUC and mutation_status 0/1.
(WT means only a profiled patient; a patient without any mutation record never enters the matrix.)
  python 11_analysis_population.py --cohort DISCOVERY
"""
import pandas as pd
from common import data


def main():
    cohort = data.parse_cohort("analysis population")
    st = data.read(cohort, "stable_pairs.csv", hint="run 04_export_stable_pairs.py first")
    mut, drug = data.load_cohort_matrices(cohort)
    names = data.drug_names()
    rows = []
    for _, r in st.iterrows():
        y = drug[r["drug"]].dropna()
        y = y[y.index.isin(mut.index)]
        status = (mut[r["gene"]].reindex(y.index) == 1).astype(int)
        rows.append(pd.DataFrame({"patient_id": y.index, "gene": r["gene"], "drug": r["drug"],
                                  "drug_name": names.get(r["drug"], r["drug"]),
                                  "mutation_status": status.to_numpy(), "AUC": y.to_numpy()}))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["patient_id", "gene", "drug", "drug_name", "mutation_status", "AUC"])
    data.write(df, cohort, "analysis_population.csv")
    if len(df):
        print(df.groupby(["gene", "drug_name", "mutation_status"]).size().rename("n").reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
