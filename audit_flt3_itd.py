# -*- coding: utf-8 -*-
"""audit_flt3_itd.py -- read-only. Run it before filling the config (FLT3_ITD_POSITIVE/NEGATIVE_VALUES) and again afterwards.
Checks: (1) raw column values and mapping  (2) link to the manifest patients  (3) discordance between the clinical rows of one patient / timing
(4) number of FLT3-mutant patients per cohort with the composite definition and with WES-only."""
import pandas as pd
from common import config_official as cfg, io_official as io, cohort

src = io.load_sources()
cl, col = src["clinical"], cfg.FLT3_ITD_COLUMN
man = pd.read_csv(cfg.MANIFEST_FILE, dtype={"patient_id": "string"})

print("== 1) raw values", col)
print(cl[col].astype("string").str.strip().value_counts(dropna=False).to_string())
pos = {str(x).strip().lower() for x in cfg.FLT3_ITD_POSITIVE_VALUES}
neg = {str(x).strip().lower() for x in cfg.FLT3_ITD_NEGATIVE_VALUES}
print("config positive:", sorted(pos), "| negative:", sorted(neg))
if not pos:
    raise SystemExit("The config is not filled in yet; choose the positive values from the table above and put them in config_official.py.")

g = io.flt3_itd_status(cl)                                   # unknown value => error
print("\n== 2) link")
ids = set(man.patient_id.dropna())
print("clinical patients:", len(g), "| in manifest:", len(set(g.index) & ids), "| ITD-positive:", int(g.any_pos.sum()),
      "| ITD-positive not in manifest:", len(set(g.index[g.any_pos]) - ids))
print("manifest patients without any clinical ITD row:", len(ids - set(g.index)))

print("\n== 3) timing/discordance")
print("patients with >1 clinical row:", int((g.n_rows > 1).sum()), "| discordant (pos and neg rows):", int(g.discordant.sum()))
tc = [c for c in ["isRelapse", "diseaseStageAtSpecimenCollection", "timeOfSampleCollectionRelativeToInclusion"] if c in cl.columns]
x = cl[["dbgap_subject_id", col] + tc].copy()
x["itd_pos"] = x[col].astype("string").str.strip().str.lower().isin(pos)
if "diseaseStageAtSpecimenCollection" in x:
    print(pd.crosstab(x["diseaseStageAtSpecimenCollection"].fillna("NA"), x["itd_pos"]).to_string())
bad = g[g.discordant]
if len(bad):
    out = cfg.MANIFEST_DIR / "flt3_itd_discordant_patients.csv"
    x[x.dbgap_subject_id.isin(bad.index)].sort_values("dbgap_subject_id").to_csv(out, index=False, encoding="utf-8-sig")
    print("saved", out)

print("\n== 4) FLT3-mutant per cohort (composite vs WES-only)")
for c in ["FULL", "DISCOVERY", "VALIDATION"]:
    pats = cohort.patients_of(man, c)
    a = io.build_patient_mutation_matrix(src, pats, flt3_composite=True)[cfg.FLT3_GENE]
    b = io.build_patient_mutation_matrix(src, pats, flt3_composite=False)[cfg.FLT3_GENE]
    print(f"{c}: profiled={len(a)} | WES-only={int(b.sum())} | composite={int(a.sum())} | ITD-only added={int(a.sum() - b.sum())}")

print("\n== 5) provenance (patient-level)")
for c in ["FULL", "DISCOVERY", "VALIDATION"]:
    pats = cohort.patients_of(man, c)
    wes = io.build_patient_mutation_matrix(src, pats, flt3_composite=False)
    pv = io.flt3_composite_audit(wes, cl)
    out = cfg.MANIFEST_DIR / f"flt3_composite_provenance_{c}.csv"
    pv.to_csv(out, index=False, encoding="utf-8-sig")
    print(c, "| source:", pv["source"].value_counts().to_dict(), "| itd_status:", pv["itd_status"].value_counts().to_dict())
    print("  saved", out)
print("\nLimitation: composite = ever-detected patient-level FLT3 (WES or ITD at any time); neither baseline status nor simultaneous with the AUC sample.")
