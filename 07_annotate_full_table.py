# -*- coding: utf-8 -*-
"""
07_annotate_full_table.py
=========================================================================================
Annotates every association of the cohort with name/family from the same lookup table (stage 05).
  python 07_annotate_full_table.py --cohort DISCOVERY
"""
import pandas as pd
from common import config_official as cfg, data


def main():
    cohort = data.parse_cohort("annotate full table", choices=("DISCOVERY", "FULL"))
    a = data.read(cohort, "mutation_drug_associations.csv", hint="اول 01_association_screen.py")
    lk = pd.read_csv(cfg.cohort_dir("shared") / "drug_lookup_table.csv")[["inhibitor", "standard_name", "family"]]
    a = a.merge(lk, left_on="drug", right_on="inhibitor", how="left").drop(columns="inhibitor")
    a = a.rename(columns={"standard_name": "drug_name"}).sort_values("p_value")
    print(f"بدون نام: {a['drug_name'].isna().sum()} / {len(a)}")
    print(a[["gene", "drug_name", "n_mutated_with_AUC", "n_wt_with_AUC", "rank_biserial_effect", "p_value", "FDR_q"]]
          .head(20).to_string(index=False))
    data.write(a, cohort, "mutation_drug_associations_annotated.csv")


if __name__ == "__main__":
    main()
