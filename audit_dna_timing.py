# -*- coding: utf-8 -*-
"""audit_dna_timing.py  read-only: DNA sample time relative to the AUC sample, and the FLT3-ITD status"""
import pandas as pd
from common import config_official as cfg, io_official as io

src = io.load_sources()
man = pd.read_csv(cfg.MANIFEST_FILE, dtype={"patient_id": "string"})
man = man[man.eligible_for_association]
dna = io.eligible_dna_table(src["mapping"])
prof = set(src["mutations"]["dbgap_sample_id"].dropna())
dna = dna[dna.dna_sample.isin(prof) & dna.patient_id.isin(man.patient_id)]
n_dna = dna.groupby("patient_id").dna_sample.nunique()
multi = set(n_dna[n_dna > 1].index)
print("patients with >1 profiled DNA:", len(multi), "of", len(n_dna))

cl = src["clinical"]
cols = ["dbgap_dnaseq_sample", "isRelapse", "isDenovo", "specimenType",
        "diseaseStageAtSpecimenCollection", "timeOfSampleCollectionRelativeToInclusion"]
c = dna.merge(cl[cols].drop_duplicates("dbgap_dnaseq_sample"), left_on="dna_sample",
              right_on="dbgap_dnaseq_sample", how="left")
cm = c[c.patient_id.isin(multi)].sort_values(["patient_id", "timeOfSampleCollectionRelativeToInclusion"])
print("\nmulti-DNA patients: DNA samples without clinical row:", int(cm.isRelapse.isna().sum()))
print(cm.groupby("patient_id").isRelapse.nunique().value_counts().rename("n_distinct_isRelapse").to_string())
print(cm.groupby("patient_id").diseaseStageAtSpecimenCollection.nunique().value_counts()
      .rename("n_distinct_stage").to_string())
cm.to_csv(cfg.MANIFEST_DIR / "multi_dna_patients_timing.csv", index=False, encoding="utf-8-sig")

# Is the drug sample the same as the DNA sample?
d = io.usable_drug_records(src["drug"])
d = d[d.dbgap_subject_id.isin(man.patient_id)]
d_s = d[["dbgap_subject_id", "dbgap_dnaseq_sample"]].drop_duplicates()
print("\ndrug records' DNA sample missing:", int(d_s.dbgap_dnaseq_sample.isna().sum()), "of", len(d_s))
ok = d_s.dbgap_dnaseq_sample.isin(set(dna.dna_sample))
print("drug DNA sample is a profiled eligible DNA sample:", int(ok.sum()), "of", len(d_s))
dm = d_s[d_s.dbgap_subject_id.isin(multi)]
print("multi-DNA patients: drug DNA sample among profiled:", int(dm.dbgap_dnaseq_sample.isin(set(dna.dna_sample)).sum()), "of", len(dm))

# FLT3-ITD
f = cl[["dbgap_subject_id", "FLT3-ITD"]].dropna(subset=["dbgap_subject_id"])
print("\nclinical FLT3-ITD values:\n", f["FLT3-ITD"].value_counts(dropna=False).to_string())
mut = io.build_patient_mutation_matrix(src, set(man.patient_id), flt3_composite=False)   # FLT3 column = WES only
itd = io.flt3_itd_status(cl).any_pos
tab = pd.crosstab(mut["FLT3"].rename("FLT3_in_WES_file"), itd.reindex(mut.index).fillna(False).rename("FLT3_ITD_clinical"))
print("\n", tab.to_string())
