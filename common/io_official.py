# -*- coding: utf-8 -*-
"""io_official.py -- reads the official files and builds the patient-level matrices."""
import numpy as np
import pandas as pd
from . import config_official as cfg

_NA = ["", "NA", "N/A", "NaN", "nan", "None", "NULL"]
_NA_LOWER = [x.lower() for x in _NA]   # after lower()
_TRUE = {"true", "1", "yes", "y"}
_FALSE = {"false", "0", "no", "n"}

def norm_id(s):
    return s.astype("string").str.strip().replace(_NA, pd.NA)

def norm_flag(s):
    x = s.astype("string").str.strip().str.lower()
    return x.mask(x.isin(_TRUE), "yes").mask(x.isin(_FALSE), "no").replace(_NA, pd.NA)

def to_bool(s):
    return s.astype("string").str.strip().str.lower().isin(_TRUE)

def load_sources():
    for k, p in cfg.FILES.items():
        if not p.exists():
            raise FileNotFoundError(f"{k}: {p}")
    mapping = pd.read_excel(cfg.FILES["mapping"], sheet_name=cfg.SHEETS["mapping"], dtype="string")
    clinical = pd.read_excel(cfg.FILES["clinical"], sheet_name=cfg.SHEETS["clinical"], dtype="string")
    mut = pd.read_csv(cfg.FILES["mutations"], sep="\t", low_memory=False, dtype={"dbgap_sample_id": "string"})
    drug = pd.read_csv(cfg.FILES["drug"], sep="\t", low_memory=False,
                       dtype={"dbgap_subject_id": "string", "dbgap_dnaseq_sample": "string",
                              "dbgap_rnaseq_sample": "string"})
    for df, cols in [(mapping, ["patientId", "dbgap_subject_id", "dbgap_dnaseq_sample", "dbgap_rnaseq_sample"]),
                     (clinical, ["dbgap_subject_id", "dbgap_dnaseq_sample", "dbgap_rnaseq_sample"]),
                     (mut, ["dbgap_sample_id"]),
                     (drug, ["dbgap_subject_id", "dbgap_dnaseq_sample", "dbgap_rnaseq_sample"])]:
        for c in cols:
            if c in df.columns:
                df[c] = norm_id(df[c])
    mapping["dna_include_in_analysis"] = norm_flag(mapping["dna_include_in_analysis"])
    mapping["is_control"] = (mapping["patientId"].isin(cfg.CONTROL_PATIENT_IDS)
                             | mapping["dbgap_subject_id"].isin(cfg.CONTROL_SUBJECT_IDS))
    return {"mapping": mapping, "clinical": clinical, "mutations": mut, "drug": drug}

# ------------------------------------------------------------------ DNA / mutation
def eligible_dna_table(mapping):
    """One row per eligible DNA sample: dna_sample -> patient."""
    m = mapping.loc[~mapping["is_control"] & mapping["dbgap_dnaseq_sample"].notna()]
    if cfg.DNA_REQUIRE_MAPPING_INCLUDE_FLAG:
        m = m.loc[m["dna_include_in_analysis"] == "yes"]
    t = m[["dbgap_dnaseq_sample", "dbgap_subject_id"]].drop_duplicates()
    dup = t["dbgap_dnaseq_sample"].duplicated(keep=False)
    if dup.any():
        raise ValueError("نمونه DNA مربوط به بیش از یک بیمار است:\n" + t[dup].to_string())
    return t.rename(columns={"dbgap_dnaseq_sample": "dna_sample", "dbgap_subject_id": "patient_id"})

def eligible_variants(mut):
    v = mut.copy()
    cls = v[cfg.VARIANT_CLASS_COLUMN].astype("string").str.strip().str.lower().replace(_NA_LOWER, pd.NA)
    # An empty/unknown class is NOT "eligible" (it is not in the excluded set, so without notna() it would silently enter the matrix)
    keep = cls.notna() & ~cls.isin(cfg.EXCLUDED_VARIANT_CLASSES)
    if cfg.MIN_T_VAF is not None:
        keep &= pd.to_numeric(v["t_vaf"], errors="coerce") >= cfg.MIN_T_VAF
    keep &= v[cfg.GENE_COLUMN].notna()
    return v.loc[keep]

def variant_class_table(mut):
    cls = mut[cfg.VARIANT_CLASS_COLUMN].astype("string").str.strip().str.lower()
    t = cls.value_counts(dropna=False).rename_axis("variant_class").reset_index(name="n_records")
    t["excluded_by_config"] = t["variant_class"].isin(cfg.EXCLUDED_VARIANT_CLASSES)
    return t

def variant_filter_report(mut):
    """Effect of the variant filter (read-only report); inspect it before running Stage 01 and record it in Methods.
    Returns: a dict of counts + the class table (including NaN) showing which classes are kept/dropped."""
    cls = mut[cfg.VARIANT_CLASS_COLUMN].astype("string").str.strip().str.lower().replace(_NA_LOWER, pd.NA)
    gene_na = mut[cfg.GENE_COLUMN].isna()
    excl = cls.isin(cfg.EXCLUDED_VARIANT_CLASSES)
    kept = eligible_variants(mut)
    t = cls.value_counts(dropna=False).rename_axis("variant_class").reset_index(name="n_records")
    t["decision"] = np.where(t["variant_class"].isna(), "dropped_empty_class",
                             np.where(t["variant_class"].isin(cfg.EXCLUDED_VARIANT_CLASSES), "dropped_excluded_class", "kept"))
    present = set(cls.dropna())
    rep = {"n_records": len(mut), "n_empty_class": int(cls.isna().sum()), "n_excluded_class": int(excl.sum()),
           "n_empty_gene": int(gene_na.sum()), "n_kept": len(kept), "min_t_vaf": cfg.MIN_T_VAF,
           "excluded_classes_not_present_in_file": sorted(set(cfg.EXCLUDED_VARIANT_CLASSES) - present)}
    return rep, t


def flt3_itd_status(clinical):
    """
    Patient-level summary of the clinical FLT3-ITD column: any_pos / any_neg / discordant (positive and negative rows at once) / n_rows.
    An unknown value => ValueError (positive/negative is never guessed). NaN = unknown (not positive).
    """
    col = cfg.FLT3_ITD_COLUMN
    if col not in clinical.columns:
        raise ValueError(f"ستون {col} در clinical نیست")
    pos = {str(x).strip().lower() for x in cfg.FLT3_ITD_POSITIVE_VALUES}
    neg = {str(x).strip().lower() for x in cfg.FLT3_ITD_NEGATIVE_VALUES}
    if not pos:
        raise ValueError("FLT3_ITD_POSITIVE_VALUES خالی است؛ اول audit_flt3_itd.py را اجرا و config را پر کنید")
    if pos & neg:
        raise ValueError(f"مقدار هم مثبت هم منفی: {sorted(pos & neg)}")
    cl = clinical.loc[clinical["dbgap_subject_id"].notna(), ["dbgap_subject_id", col]].copy()
    v = cl[col].astype("string").str.strip().str.lower().replace(_NA_LOWER, pd.NA)
    unmapped = sorted(set(v.dropna()) - pos - neg)
    if unmapped:
        raise ValueError(f"مقدار FLT3-ITD بدون نگاشت در config: {unmapped}")
    cl["pos"] = v.isin(pos).astype(bool)
    cl["neg"] = v.isin(neg).astype(bool)
    g = cl.groupby("dbgap_subject_id").agg(any_pos=("pos", "any"), any_neg=("neg", "any"), n_rows=("pos", "size"))
    g["discordant"] = g["any_pos"] & g["any_neg"]
    return g


def apply_flt3_composite(mat, clinical):
    """FLT3 = WES OR clinical ITD; only for profiled patients (rows of mat). All other columns are untouched."""
    g = flt3_itd_status(clinical)
    prof = set(mat.index.astype(str))
    pos = {p for p in g.index[g["any_pos"]].astype(str) if p in prof}
    mat = mat.copy()
    if cfg.FLT3_GENE not in mat.columns:
        mat[cfg.FLT3_GENE] = 0
    hit = [str(i) in pos for i in mat.index]
    mat.loc[hit, cfg.FLT3_GENE] = 1
    return mat


def flt3_composite_audit(wes_mat, clinical):
    """
    Patient-level provenance of the composite FLT3 (report only; does not change the matrix).
    wes_mat: the "WES-only" matrix (build_patient_mutation_matrix(..., flt3_composite=False)).
    source column: both | wes_only | itd_only | neither. itd_status: positive_only | negative_only | discordant | no_clinical_value.
    """
    st = flt3_itd_status(clinical)
    st.index = st.index.astype(str)
    wes = (wes_mat[cfg.FLT3_GENE].eq(1) if cfg.FLT3_GENE in wes_mat.columns
           else pd.Series(False, index=wes_mat.index))
    rows = []
    for p, w in zip(wes_mat.index.astype(str), wes.to_numpy()):
        r = st.loc[p] if p in st.index else None
        pos = bool(r["any_pos"]) if r is not None else False
        neg = bool(r["any_neg"]) if r is not None else False
        w = bool(w)
        rows.append({"patient_id": p, "wes_positive": w, "itd_any_pos": pos, "itd_any_neg": neg,
                     "itd_discordant": pos and neg, "itd_n_clinical_rows": int(r["n_rows"]) if r is not None else 0,
                     "itd_status": ("discordant" if pos and neg else "positive_only" if pos
                                    else "negative_only" if neg else "no_clinical_value"),
                     "composite": w or pos,
                     "source": "both" if (w and pos) else "wes_only" if w else "itd_only" if pos else "neither"})
    return pd.DataFrame(rows)


def build_patient_mutation_matrix(src, patients, flt3_composite=None):
    """
    Patient x gene (0/1) matrix, only for profiled patients inside `patients`.
    A patient who is not profiled is absent from the matrix (Unknown, not WT).
    """
    dna = eligible_dna_table(src["mapping"])
    profiled_dna = set(src["mutations"]["dbgap_sample_id"].dropna())
    dna = dna[dna["dna_sample"].isin(profiled_dna) & dna["patient_id"].isin(patients)]
    profiled_patients = sorted(dna["patient_id"].unique())
    v = eligible_variants(src["mutations"]).merge(dna, left_on="dbgap_sample_id", right_on="dna_sample")
    pairs = v[["patient_id", cfg.GENE_COLUMN]].drop_duplicates()
    pairs["x"] = 1
    mat = (pairs.pivot(index="patient_id", columns=cfg.GENE_COLUMN, values="x")
           .reindex(profiled_patients).fillna(0).astype(int))
    use = cfg.FLT3_COMPOSITE if flt3_composite is None else flt3_composite
    if use:
        mat = apply_flt3_composite(mat, src["clinical"]).astype(int)
    return mat

# ------------------------------------------------------------------ drug
def usable_drug_records(drug):
    d = drug.copy()
    d[cfg.DRUG_METRIC] = pd.to_numeric(d[cfg.DRUG_METRIC], errors="coerce")
    d = d[d["dbgap_subject_id"].notna() & d[cfg.DRUG_ID_COLUMN].notna() & d[cfg.DRUG_METRIC].notna()]
    d = d[~d["dbgap_subject_id"].isin(cfg.CONTROL_SUBJECT_IDS)]
    if cfg.DRUG_REQUIRE_CONVERGED:
        d = d[to_bool(d["converged"])]
    if cfg.DRUG_REQUIRE_PAPER_INCLUSION:
        d = d[to_bool(d["paper_inclusion"])]
    if cfg.DRUG_ALLOWED_STATUS is not None:
        allowed = {s.lower() for s in cfg.DRUG_ALLOWED_STATUS}
        d = d[d["status"].astype("string").str.strip().str.lower().isin(allowed)]
    return d

def repeat_report(recs):
    g = recs.groupby(["dbgap_subject_id", cfg.DRUG_ID_COLUMN])[cfg.DRUG_METRIC]
    r = g.agg(n_records="size", auc_min="min", auc_max="max").reset_index()
    r["auc_range"] = r["auc_max"] - r["auc_min"]
    return r[r["n_records"] > 1]

def build_patient_drug_matrix(src, patients):
    """Patient x drug matrix with a single AUC value per pair (repeat rule in config)."""
    recs = usable_drug_records(src["drug"])
    recs = recs[recs["dbgap_subject_id"].isin(patients)]
    g = recs.groupby(["dbgap_subject_id", cfg.DRUG_ID_COLUMN])[cfg.DRUG_METRIC]
    pol = cfg.DRUG_REPEAT_POLICY
    if pol == "median":
        s = g.median()
    elif pol == "mean":
        s = g.mean()
    elif pol == "exclude_repeated":
        s = g.agg(lambda x: x.iloc[0] if len(x) == 1 else np.nan).dropna()
    else:
        raise ValueError(f"DRUG_REPEAT_POLICY نامعتبر: {pol}")
    return s.unstack(cfg.DRUG_ID_COLUMN)
