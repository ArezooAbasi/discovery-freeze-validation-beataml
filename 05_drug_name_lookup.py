# -*- coding: utf-8 -*-
"""
05_drug_name_lookup.py
=========================================================================================
In the official data the drug identifier is the `inhibitor` column (not improve_drug_id); beataml_drugs.tsv.gz is no longer used.
This stage only builds a display table: inhibitor -> standard_name (parenthesised alias removed) + family + target genes
taken from beataml_drug_families.xlsx. The join key of every stage is always the original `inhibitor`.
Output: output/shared/drug_lookup_table.csv
"""
import re
import pandas as pd
from common import config_official as cfg, data


def clean(name):
    c = re.sub(r"\s*\([^)]*\)\s*$", "", str(name)).strip()
    return c or str(name)


def main():
    inhibitors = sorted(data.load_cohort_matrices("FULL")[1].columns)
    lk = pd.DataFrame({"inhibitor": inhibitors})
    lk["standard_name"] = lk["inhibitor"].map(clean)
    dup = lk["standard_name"].duplicated(keep=False)
    lk.loc[dup, "standard_name"] = lk.loc[dup, "inhibitor"]      # name collision => fall back to the original name
    for k, v in cfg.DRUG_NAME_OVERRIDES.items():
        lk.loc[lk["inhibitor"] == k, "standard_name"] = v
    lk["family"], lk["target_genes"] = "", ""
    f = cfg.DRUG_FAMILIES_FILE
    if f.exists():
        fam = pd.read_excel(f, sheet_name="drug_family")
        tg = pd.read_excel(f, sheet_name="drug_gene")
        fm = fam.groupby("inhibitor")["family"].agg(lambda s: "; ".join(sorted(set(s.dropna().astype(str)))))
        tm = tg.groupby("inhibitor")["Symbol"].agg(lambda s: "; ".join(sorted(set(s.dropna().astype(str)))[:12]))
        lk["family"] = lk["inhibitor"].map(fm).fillna("")
        lk["target_genes"] = lk["inhibitor"].map(tm).fillna("")
    else:
        print(f"[!] {f} پیدا نشد؛ family/target خالی ماند.")
    print(f"unmapped to family: {(lk.family == '').sum()} / {len(lk)}")
    data.write(lk, "shared", "drug_lookup_table.csv")


if __name__ == "__main__":
    main()
