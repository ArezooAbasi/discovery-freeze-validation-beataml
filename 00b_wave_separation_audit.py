
# -*- coding: utf-8 -*-
"""
00b_wave_separation_audit.py
======================================================================
Purpose
-------
Audit the separation of DISCOVERY and VALIDATION cohorts using:
1. The generated patient manifest.
2. Official mapping and clinical source files.

No statistical analysis is performed.

Hard checks:
    H1: No patient_id overlap between DISCOVERY and VALIDATION.
    H2: No eligible DNA sample is linked to both cohorts.
    H3a: One patientId must not map to multiple dbgap_subject_id values.
    H3b: One dbgap_subject_id must not map to multiple patientId values.
    H4: No mapping patientId may connect subject IDs from both cohorts.

Additional integrity checks:
    H5: Manifest patient_id must be unique.
    H6: Every DISCOVERY/VALIDATION patient must be represented in
        the non-control official mapping.
    H7: Mapping identifiers used in the cohort audit must be valid.

Informational checks:
    I1: Distribution of wave_group in the manifest.
    I2: Number of FULL patients classified as BOTH or UNKNOWN.
    I3: DISCOVERY and VALIDATION sample sizes.
    I4: Patients with more than one distinct clinical cohort label.
    I5: Patients with mixed W12/W34 labels.
    I6: Patients with an explicit "Both" label.

Important:
    Multiple clinical labels are not equivalent to mixed W12/W34 labels.
    These quantities are reported separately.

Exit codes:
    0 = all hard checks passed
    1 = one or more hard checks failed
    2 = input/schema/data integrity error
"""

import sys
import pandas as pd

from common import config_official as cfg
from common import io_official as io
from common import cohort


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def normalize_identifier(series):
    """Normalize identifiers without converting missing values to strings."""
    return series.astype("string").str.strip()


def normalize_cohort_label(value):
    """Normalize clinical cohort labels for consistent comparisons."""
    if pd.isna(value):
        return pd.NA

    return str(value).replace(" ", "").strip().lower()


def add_check(rows, check, value, passed, note="", severity="INFO"):
    rows.append({
        "check": check,
        "value": value,
        "pass": bool(passed),
        "severity": severity,
        "note": note
    })


def require_columns(df, required, source_name):
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {missing}"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    rows = []
    hard_fail = False

    try:
        # -------------------------------------------------------------
        # 1. Load manifest and official sources
        # -------------------------------------------------------------

        manifest = pd.read_csv(
            cfg.MANIFEST_FILE,
            dtype={"patient_id": "string"}
        )

        src = io.load_sources()

        mapping = src["mapping"].copy()
        clinical = src["clinical"].copy()

        require_columns(
            manifest,
            ["patient_id", "in_FULL", "wave_group"],
            "Manifest"
        )

        require_columns(
            mapping,
            ["patientId", "dbgap_subject_id", "is_control"],
            "Official mapping"
        )

        require_columns(
            clinical,
            ["dbgap_subject_id", "cohort"],
            "Clinical source"
        )

        # -------------------------------------------------------------
        # 2. Validate manifest identifiers
        # -------------------------------------------------------------

        manifest["patient_id"] = normalize_identifier(
            manifest["patient_id"]
        )

        if manifest["patient_id"].isna().any():
            raise ValueError(
                "Manifest contains missing patient_id values."
            )

        duplicate_manifest_ids = int(
            manifest["patient_id"].duplicated().sum()
        )

        add_check(
            rows,
            "H5 duplicate patient_id in manifest",
            duplicate_manifest_ids,
            duplicate_manifest_ids == 0,
            "Manifest must contain exactly one row per patient.",
            "HARD"
        )

        if duplicate_manifest_ids > 0:
            hard_fail = True

        if manifest["patient_id"].duplicated().any():
            duplicate_ids = (
                manifest.loc[
                    manifest["patient_id"].duplicated(keep=False),
                    "patient_id"
                ]
                .drop_duplicates()
                .head(20)
                .tolist()
            )

            raise ValueError(
                "Duplicate patient_id values in manifest. "
                f"Examples: {duplicate_ids}"
            )

        # -------------------------------------------------------------
        # 3. Extract cohorts
        # -------------------------------------------------------------

        d = cohort.patients_of(manifest, "DISCOVERY")
        v = cohort.patients_of(manifest, "VALIDATION")

        d = {str(x).strip() for x in d}
        v = {str(x).strip() for x in v}

        # H1: patient-level separation

        patient_overlap = d & v

        add_check(
            rows,
            "H1 patient overlap DISCOVERY∩VALIDATION",
            len(patient_overlap),
            len(patient_overlap) == 0,
            (
                "No patient may belong to both cohorts. "
                f"Examples: {sorted(patient_overlap)[:20]}"
            ),
            "HARD"
        )

        if patient_overlap:
            hard_fail = True

        # -------------------------------------------------------------
        # 4. Prepare official mapping
        # -------------------------------------------------------------

        mapping["patientId"] = normalize_identifier(
            mapping["patientId"]
        )

        mapping["dbgap_subject_id"] = normalize_identifier(
            mapping["dbgap_subject_id"]
        )

        controls = mapping["is_control"].fillna(False).astype(bool)

        mp = mapping.loc[
            ~controls,
            ["patientId", "dbgap_subject_id"]
        ].copy()

        missing_mapping_ids = int(
            (
                mp["patientId"].isna()
                | mp["dbgap_subject_id"].isna()
            ).sum()
        )

        add_check(
            rows,
            "H7 missing identifiers in non-control mapping",
            missing_mapping_ids,
            missing_mapping_ids == 0,
            "Rows with missing patientId or dbgap_subject_id.",
            "HARD"
        )

        if missing_mapping_ids > 0:
            hard_fail = True

        mp = (
            mp.dropna(
                subset=["patientId", "dbgap_subject_id"]
            )
            .drop_duplicates()
        )

        # -------------------------------------------------------------
        # 5. H2: DNA sample separation
        # -------------------------------------------------------------

        dna = io.eligible_dna_table(mapping.copy())

        require_columns(
            dna,
            ["patient_id", "dna_sample"],
            "Eligible DNA table"
        )

        dna["patient_id"] = normalize_identifier(
            dna["patient_id"]
        )

        dna["dna_sample"] = normalize_identifier(
            dna["dna_sample"]
        )

        dna = dna.dropna(
            subset=["patient_id", "dna_sample"]
        )

        d_samples = set(
            dna.loc[
                dna["patient_id"].isin(d),
                "dna_sample"
            ]
        )

        v_samples = set(
            dna.loc[
                dna["patient_id"].isin(v),
                "dna_sample"
            ]
        )

        shared_dna = d_samples & v_samples

        add_check(
            rows,
            "H2 DNA samples linked to both cohorts",
            len(shared_dna),
            len(shared_dna) == 0,
            (
                "No eligible DNA sample may be shared. "
                f"Examples: {sorted(shared_dna)[:20]}"
            ),
            "HARD"
        )

        if shared_dna:
            hard_fail = True

        # -------------------------------------------------------------
        # 6. H3: mapping identifier consistency
        # -------------------------------------------------------------

        patient_to_subject = (
            mp.groupby("patientId")["dbgap_subject_id"]
            .nunique()
        )

        subject_to_patient = (
            mp.groupby("dbgap_subject_id")["patientId"]
            .nunique()
        )

        h3a_ids = patient_to_subject[
            patient_to_subject > 1
        ]

        h3b_ids = subject_to_patient[
            subject_to_patient > 1
        ]

        add_check(
            rows,
            "H3a patientId -> >1 subject_id",
            len(h3a_ids),
            len(h3a_ids) == 0,
            (
                "Each patientId should map to one subject ID. "
                f"Examples: {h3a_ids.index.tolist()[:20]}"
            ),
            "HARD"
        )

        add_check(
            rows,
            "H3b subject_id -> >1 patientId",
            len(h3b_ids),
            len(h3b_ids) == 0,
            (
                "Each subject ID should map to one patientId. "
                f"Examples: {h3b_ids.index.tolist()[:20]}"
            ),
            "HARD"
        )

        if len(h3a_ids) > 0 or len(h3b_ids) > 0:
            hard_fail = True

        # -------------------------------------------------------------
        # 7. H4: patientId connecting both cohorts
        # -------------------------------------------------------------

        pd_map = mp.copy()

        pd_map["in_d"] = pd_map["dbgap_subject_id"].isin(d)
        pd_map["in_v"] = pd_map["dbgap_subject_id"].isin(v)

        mapping_cohort = (
            pd_map.groupby("patientId")[["in_d", "in_v"]]
            .any()
        )

        h4_ids = mapping_cohort.loc[
            mapping_cohort["in_d"] & mapping_cohort["in_v"]
        ].index.tolist()

        add_check(
            rows,
            "H4 patientId in both DISCOVERY and VALIDATION",
            len(h4_ids),
            len(h4_ids) == 0,
            (
                "A mapping patientId must not connect subject IDs "
                "from both cohorts. "
                f"Examples: {h4_ids[:20]}"
            ),
            "HARD"
        )

        if h4_ids:
            hard_fail = True

        # -------------------------------------------------------------
        # 8. H6: cohort patients represented in official mapping
        # -------------------------------------------------------------

        mapped_subjects = set(
            mp["dbgap_subject_id"].dropna()
        )

        missing_d = sorted(d - mapped_subjects)
        missing_v = sorted(v - mapped_subjects)

        missing_cohort_patients = len(missing_d) + len(missing_v)

        add_check(
            rows,
            "H6 cohort patients missing from official mapping",
            missing_cohort_patients,
            missing_cohort_patients == 0,
            (
                f"DISCOVERY missing: {missing_d[:20]}; "
                f"VALIDATION missing: {missing_v[:20]}"
            ),
            "HARD"
        )

        if missing_cohort_patients > 0:
            hard_fail = True

        # -------------------------------------------------------------
        # 9. Informational manifest checks
        # -------------------------------------------------------------

        manifest["wave_group"] = (
            manifest["wave_group"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        wave_counts = (
            manifest["wave_group"]
            .value_counts(dropna=False)
            .to_dict()
        )

        add_check(
            rows,
            "I1 wave_group distribution (all patients)",
            str(wave_counts),
            True,
            "Descriptive distribution; not a hard check."
        )

        full_flag = (
            manifest["in_FULL"]
            .fillna(False)
            .astype(bool)
        )

        full = manifest.loc[full_flag].copy()

        full_both = int(
            (full["wave_group"] == "BOTH").sum()
        )

        full_unknown = int(
            (full["wave_group"] == "UNKNOWN").sum()
        )

        add_check(
            rows,
            "I2 FULL: BOTH / UNKNOWN",
            f"{full_both} / {full_unknown}",
            True,
            "These patients are excluded from the W12/W34 split."
        )

        add_check(
            rows,
            "I3 DISCOVERY / VALIDATION patients",
            f"{len(d)} / {len(v)}",
            True,
            "Patient counts from the manifest."
        )

        # -------------------------------------------------------------
        # 10. Clinical label diagnostics
        # -------------------------------------------------------------

        cl = clinical.loc[
            clinical["dbgap_subject_id"].notna(),
            ["dbgap_subject_id", "cohort"]
        ].copy()

        cl["dbgap_subject_id"] = normalize_identifier(
            cl["dbgap_subject_id"]
        )

        cl["label"] = cl["cohort"].map(
            normalize_cohort_label
        )

        cl = cl.dropna(
            subset=["dbgap_subject_id"]
        )

        grouped = cl.groupby("dbgap_subject_id")["label"]

        label_counts = grouped.nunique(dropna=True)

        multi_label_count = int(
            (label_counts > 1).sum()
        )

        mixed_wave = grouped.apply(
            lambda x: {
                "waves1+2",
                "waves3+4"
            }.issubset(set(x.dropna()))
        )

        mixed_wave_count = int(
            mixed_wave.sum()
        )

        explicit_both = grouped.apply(
            lambda x: (x == "both").any()
        )

        explicit_both_count = int(
            explicit_both.sum()
        )

        add_check(
            rows,
            "I4 patients with >1 distinct clinical cohort label",
            multi_label_count,
            True,
            "Multiple distinct labels; not necessarily mixed W12/W34."
        )

        add_check(
            rows,
            "I5 patients with mixed W12/W34 labels",
            mixed_wave_count,
            True,
            "Patient has both Waves1+2 and Waves3+4 labels."
        )

        add_check(
            rows,
            "I6 patients with explicit Both label",
            explicit_both_count,
            True,
            "Patient has at least one explicit Both label."
        )

        # -------------------------------------------------------------
        # 11. Check clinical patients against manifest
        # -------------------------------------------------------------

        clinical_ids = set(
            cl["dbgap_subject_id"].dropna()
        )

        manifest_ids = set(
            manifest["patient_id"].dropna()
        )

        clinical_not_in_manifest = sorted(
            clinical_ids - manifest_ids
        )

        manifest_not_in_clinical = sorted(
            manifest_ids - clinical_ids
        )

        add_check(
            rows,
            "I7 clinical patients absent from manifest",
            len(clinical_not_in_manifest),
            True,
            f"Examples: {clinical_not_in_manifest[:20]}"
        )

        add_check(
            rows,
            "I8 manifest patients absent from clinical",
            len(manifest_not_in_clinical),
            True,
            f"Examples: {manifest_not_in_clinical[:20]}"
        )

        # -------------------------------------------------------------
        # 12. Save audit
        # -------------------------------------------------------------

        out = pd.DataFrame(rows)

        cfg.MANIFEST_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            cfg.MANIFEST_DIR /
            "wave_separation_audit.csv"
        )

        out.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

        print("\n--- WAVE SEPARATION AUDIT ---\n")
        print(out.to_string(index=False))

        print("\n[saved]", output_path)

        # -------------------------------------------------------------
        # 13. Final status
        # -------------------------------------------------------------

        failed_checks = out.loc[
            (out["severity"] == "HARD")
            & (~out["pass"]),
            "check"
        ].tolist()

        if hard_fail or failed_checks:
            print("\n❌ HARD AUDIT FAILURE")
            print("Failed checks:")
            for item in failed_checks:
                print(" -", item)

            print(
                "\nDo not proceed to downstream cohort analyses "
                "until these failures are resolved."
            )

            sys.exit(1)

        print("\n✅ All hard cohort-separation checks passed.")
        print(
            "Informational clinical-label counts are reported "
            "separately and must not be interpreted as equivalent."
        )

    except Exception as exc:
        print("\n❌ AUDIT ERROR")
        print(f"{type(exc).__name__}: {exc}")
        print(
            "Audit did not complete. Do not proceed to downstream "
            "analysis until the issue is resolved."
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
