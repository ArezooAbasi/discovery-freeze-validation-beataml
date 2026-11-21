# -*- coding: utf-8 -*-
"""cohort.py -- builds the patient-level manifest and the sample-flow table."""
import pandas as pd
from . import config_official as cfg
from . import io_official as io

def _norm_label(x):
    return str(x).replace(" ", "").lower()

def patient_wave_groups(clinical):
    cl = clinical.loc[clinical["dbgap_subject_id"].notna(), ["dbgap_subject_id", "cohort"]].copy()
    cl["label"] = cl["cohort"].map(lambda x: None if pd.isna(x) else _norm_label(x))
    seen = set(cl["label"].dropna())
    unmapped = seen - set(cfg.COHORT_LABEL_TO_WAVEGROUP)
    if unmapped:
        raise ValueError(f"برچسب‌های cohort بدون نگاشت در config: {sorted(unmapped)}")
    cl["group"] = cl["label"].map(cfg.COHORT_LABEL_TO_WAVEGROUP)
    def resolve(g):
        s = set(g.dropna())
        return "UNKNOWN" if not s else (next(iter(s)) if len(s) == 1 else "BOTH")
    out = cl.groupby("dbgap_subject_id").agg(
        n_clinical_rows=("cohort", "size"),
        cohort_labels=("cohort", lambda x: " | ".join(sorted(set(x.dropna())))),
        wave_group=("group", resolve))
    return out

def build_manifest(src):
    mapping, clinical, drug = src["mapping"], src["clinical"], src["drug"]
    dna = io.eligible_dna_table(mapping)
    profiled_dna = set(src["mutations"]["dbgap_sample_id"].dropna())
    dna["has_mutation_records"] = dna["dna_sample"].isin(profiled_dna)
    dna_pat = dna.groupby("patient_id").agg(
        n_dna_eligible=("dna_sample", "nunique"),
        n_dna_with_mutation_records=("has_mutation_records", "sum"))

    recs = io.usable_drug_records(drug)
    drug_pat = recs.groupby("dbgap_subject_id").agg(
        n_drug_records_usable=("inhibitor", "size"),
        n_inhibitors_usable=("inhibitor", "nunique"))

    wave = patient_wave_groups(clinical)

    universe = (set(mapping.loc[~mapping["is_control"], "dbgap_subject_id"].dropna())
                | set(clinical["dbgap_subject_id"].dropna())
                | set(drug["dbgap_subject_id"].dropna())) - cfg.CONTROL_SUBJECT_IDS
    m = pd.DataFrame({"patient_id": sorted(universe)}).set_index("patient_id")
    m = m.join(wave).join(dna_pat).join(drug_pat)
    m["wave_group"] = m["wave_group"].fillna("UNKNOWN")
    for c in ["n_clinical_rows", "n_dna_eligible", "n_dna_with_mutation_records",
              "n_drug_records_usable", "n_inhibitors_usable"]:
        m[c] = m[c].fillna(0).astype(int)
    m["cohort_labels"] = m["cohort_labels"].fillna("")

    m["has_eligible_DNA"] = m["n_dna_eligible"] > 0
    m["mutation_profiled"] = m["n_dna_with_mutation_records"] > 0
    m["has_usable_drug"] = m["n_drug_records_usable"] > 0
    m["eligible_for_association"] = m["mutation_profiled"] & m["has_usable_drug"]

    reason = pd.Series("", index=m.index)
    reason[m["eligible_for_association"] == False] = "no_usable_drug"
    reason[~m["mutation_profiled"]] = "no_mutation_records_for_eligible_DNA"
    reason[~m["has_eligible_DNA"]] = "no_eligible_DNA"
    m["exclusion_reason"] = reason

    elig = m["eligible_for_association"]
    m["in_FULL"] = elig
    m["in_DISCOVERY"] = elig & (m["wave_group"] == "W12")
    m["in_VALIDATION"] = elig & (m["wave_group"] == "W34")
    m["split_note"] = ""
    m.loc[elig & (m["wave_group"] == "BOTH"), "split_note"] = "cross_wave_excluded_from_split"
    m.loc[elig & (m["wave_group"] == "UNKNOWN"), "split_note"] = "wave_unknown_excluded_from_split"
    return m.reset_index()

def sample_flow(m, src):
    rows = [
        ("patients_in_official_files (excl. pooled controls)", len(m)),
        ("with_eligible_DNA", int(m["has_eligible_DNA"].sum())),
        ("with_mutation_profile (>=1 eligible DNA sample with records)", int(m["mutation_profiled"].sum())),
        ("with_usable_drug_response", int(m["has_usable_drug"].sum())),
        ("with_both (eligible_for_association = FULL)", int(m["eligible_for_association"].sum())),
        ("  FULL: wave W12 only", int(((m.wave_group == "W12") & m.in_FULL).sum())),
        ("  FULL: wave W34 only", int(((m.wave_group == "W34") & m.in_FULL).sum())),
        ("  FULL: BOTH waves (excluded from splits)", int(((m.wave_group == "BOTH") & m.in_FULL).sum())),
        ("  FULL: wave UNKNOWN (excluded from splits)", int(((m.wave_group == "UNKNOWN") & m.in_FULL).sum())),
        ("DISCOVERY patients", int(m.in_DISCOVERY.sum())),
        ("VALIDATION patients", int(m.in_VALIDATION.sum())),
        ("DISCOVERY ∩ VALIDATION (must be 0)", int((m.in_DISCOVERY & m.in_VALIDATION).sum())),
    ]
    dna = io.eligible_dna_table(src["mapping"])
    n_dna_all = src["mapping"].loc[~src["mapping"]["is_control"], "dbgap_dnaseq_sample"].nunique()
    rows += [("DNA samples in mapping (non-control)", int(n_dna_all)),
             ("DNA samples eligible", len(dna)),
             ("DNA samples excluded by mapping flag", int(n_dna_all - len(dna))),
             ("usable drug records", len(io.usable_drug_records(src["drug"]))),
             ("raw drug records", len(src["drug"]))]
    return pd.DataFrame(rows, columns=["step", "n"])

def patients_of(manifest, cohort):
    col = {"FULL": "in_FULL", "DISCOVERY": "in_DISCOVERY", "VALIDATION": "in_VALIDATION"}[cohort]
    return set(manifest.loc[manifest[col].astype(bool), "patient_id"].astype(str))
