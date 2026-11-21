# -*- coding: utf-8 -*-
"""
06_drug_provenance.py
=========================================================================================
Provenance table (Methods-ready) for the drugs of the stable candidates: name, family, target, source files and hashes,
the repeat-record rule, and the number of patient-drug pairs with repeated records.
  python 06_drug_provenance.py --cohort DISCOVERY
"""
import json
from datetime import date
import pandas as pd
from common import config_official as cfg, data


def main():
    cohort = data.parse_cohort("drug provenance")
    st = data.read(cohort, "stable_pairs.csv", hint="اول 04_export_stable_pairs.py")
    lk = pd.read_csv(cfg.cohort_dir("shared") / "drug_lookup_table.csv").set_index("inhibitor")
    meta_p = cfg.MANIFEST_DIR / "manifest_metadata.json"
    hashes = json.loads(meta_p.read_text()).get("source_sha256", {}) if meta_p.exists() else {}
    rep_p = cfg.MANIFEST_DIR / "drug_repeat_pairs.csv"
    rep = pd.read_csv(rep_p) if rep_p.exists() else pd.DataFrame(columns=["inhibitor", "n_records", "auc_range"])
    _, drug = data.load_cohort_matrices(cohort)
    rows = []
    for d in sorted(st["drug"].unique()):
        r = rep[rep["inhibitor"] == d]
        rows.append({"inhibitor": d, "standard_name": lk["standard_name"].get(d, d), "family": lk["family"].get(d, ""),
                     "target_genes": lk["target_genes"].get(d, ""), "n_patients_with_AUC_in_cohort": int(drug[d].notna().sum()),
                     "n_patient_drug_pairs_with_repeats": len(r),
                     "median_repeat_auc_range": r["auc_range"].median() if len(r) else None,
                     "repeat_policy": cfg.DRUG_REPEAT_POLICY, "metric": cfg.DRUG_METRIC,
                     "source_file": cfg.FILES["drug"].name, "source_sha256": hashes.get("drug"),
                     "date_checked": date.today().isoformat()})
    data.write(pd.DataFrame(rows), cohort, "drug_provenance.csv")


if __name__ == "__main__":
    main()
