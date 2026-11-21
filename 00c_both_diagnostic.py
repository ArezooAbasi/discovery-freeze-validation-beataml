
# -*- coding: utf-8 -*-
"""
00c_both_diagnostic.py
======================================================================
Purpose
-------
Patient-level diagnostic of patients with:
    A: mixed W12/W34 clinical cohort labels
    B: at least one explicit "Both" clinical cohort label

A patient may belong to both A and B.

This script is diagnostic only:
    - It does not modify the patient manifest.
    - It does not change cohort assignments.
    - It does not perform statistical analysis.

Outputs:
    output/manifest/both_patients_breakdown.csv
    output/manifest/both_patients_summary.csv
    output/manifest/both_patients_excluded_from_FULL.csv

The diagnostic distinguishes:
    - all clinical patients
    - patients with A or B
    - patients included in FULL
    - patients with A or B excluded from FULL

Exit codes:
    0 = diagnostic completed successfully
    2 = input/schema/data integrity error
"""

import sys
import pandas as pd

from common import config_official as cfg
from common import io_official as io


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def normalize_id(series):
    """Normalize identifiers while preserving missing values."""
    return series.astype("string").str.strip()


def normalize_label(value):
    """Normalize clinical cohort labels."""
    if pd.isna(value):
        return pd.NA

    return str(value).replace(" ", "").strip().lower()


def require_columns(df, required, source_name):
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {missing}"
        )


def summarize_group(df, label):
    """Return A/B diagnostic counts for a specified patient group."""

    a = df["A_mixed_W12_W34"].fillna(False).astype(bool)
    b = df["B_both_label"].fillna(False).astype(bool)

    return {
        "population": label,
        "n_patients": len(df),
        "A_mixed_W12_W34": int(a.sum()),
        "B_explicit_Both": int(b.sum()),
        "A_and_B": int((a & b).sum()),
        "A_or_B": int((a | b).sum()),
        "A_only": int((a & ~b).sum()),
        "B_only": int((b & ~a).sum()),
        "neither_A_nor_B": int((~a & ~b).sum())
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    try:
        # -------------------------------------------------------------
        # 1. Load sources
        # -------------------------------------------------------------

        src = io.load_sources()

        clinical = src["clinical"].copy()

        manifest = pd.read_csv(
            cfg.MANIFEST_FILE,
            dtype={"patient_id": "string"}
        )

        require_columns(
            clinical,
            ["dbgap_subject_id", "cohort"],
            "Clinical source"
        )

        require_columns(
            manifest,
            ["patient_id", "wave_group", "in_FULL"],
            "Manifest"
        )

        # -------------------------------------------------------------
        # 2. Normalize and validate manifest
        # -------------------------------------------------------------

        manifest["patient_id"] = normalize_id(
            manifest["patient_id"]
        )

        if manifest["patient_id"].isna().any():
            raise ValueError(
                "Manifest contains missing patient_id values."
            )

        duplicated_manifest = manifest.loc[
            manifest["patient_id"].duplicated(keep=False),
            "patient_id"
        ].drop_duplicates()

        if not duplicated_manifest.empty:
            raise ValueError(
                "Manifest contains duplicate patient_id values. "
                f"Examples: {duplicated_manifest.head(20).tolist()}"
            )

        manifest["in_FULL"] = (
            manifest["in_FULL"]
            .fillna(False)
            .astype(bool)
        )

        # -------------------------------------------------------------
        # 3. Prepare clinical patient-level data
        # -------------------------------------------------------------

        clinical = clinical.loc[
            clinical["dbgap_subject_id"].notna(),
            ["dbgap_subject_id", "cohort"]
        ].copy()

        clinical["dbgap_subject_id"] = normalize_id(
            clinical["dbgap_subject_id"]
        )

        clinical["label"] = clinical["cohort"].map(
            normalize_label
        )

        clinical = clinical.dropna(
            subset=["dbgap_subject_id"]
        )

        # -------------------------------------------------------------
        # 4. Identify unexpected clinical labels
        # -------------------------------------------------------------

        expected_labels = {
            "waves1+2",
            "waves3+4",
            "both"
        }

        observed_labels = set(
            clinical["label"].dropna().unique()
        )

        unexpected_labels = sorted(
            observed_labels - expected_labels
        )

        missing_labels = int(
            clinical["label"].isna().sum()
        )

        print("\n--- CLINICAL LABEL AUDIT ---")
        print("Observed labels:", sorted(observed_labels))
        print("Unexpected labels:", unexpected_labels)
        print("Missing cohort labels:", missing_labels)

        # -------------------------------------------------------------
        # 5. Patient-level aggregation
        # -------------------------------------------------------------

        grouped = clinical.groupby(
            "dbgap_subject_id",
            sort=True
        )["label"]

        g = grouped.agg(
            n_rows="size",
            n_distinct_labels=lambda x: x.nunique(dropna=True),
            labels=lambda x: " | ".join(
                sorted(set(x.dropna()))
            ),
            has_W12=lambda x: (x == "waves1+2").any(),
            has_W34=lambda x: (x == "waves3+4").any(),
            has_Both=lambda x: (x == "both").any()
        ).reset_index()

        g = g.rename(
            columns={"dbgap_subject_id": "patient_id"}
        )

        g["A_mixed_W12_W34"] = (
            g["has_W12"] & g["has_W34"]
        )

        g["B_both_label"] = g["has_Both"]

        g["multiple_clinical_labels"] = (
            g["n_distinct_labels"] > 1
        )

        # -------------------------------------------------------------
        # 6. Validate patient-level uniqueness
        # -------------------------------------------------------------

        if g["patient_id"].duplicated().any():
            raise ValueError(
                "Patient-level clinical aggregation produced "
                "duplicate patient IDs."
            )

        # -------------------------------------------------------------
        # 7. Merge with manifest
        # -------------------------------------------------------------

        merged = g.merge(
            manifest[
                ["patient_id", "wave_group", "in_FULL"]
            ],
            on="patient_id",
            how="left",
            validate="one_to_one",
            indicator=True
        )

        unmatched = merged.loc[
            merged["_merge"] == "left_only"
        ].copy()

        if not unmatched.empty:
            print(
                "\nWARNING: Clinical patients absent from manifest:",
                len(unmatched)
            )
            print(
                unmatched["patient_id"]
                .head(20)
                .tolist()
            )

        merged["in_manifest"] = (
            merged["_merge"] == "both"
        )

        merged["in_FULL"] = (
            merged["in_FULL"]
            .fillna(False)
            .astype(bool)
        )

        merged = merged.drop(
            columns=["_merge"]
        )

        # -------------------------------------------------------------
        # 8. Select A/B patients
        # -------------------------------------------------------------

        is_ab = (
            merged["A_mixed_W12_W34"]
            | merged["B_both_label"]
        )

        breakdown = merged.loc[is_ab].copy()

        breakdown["diagnostic_category"] = "A_or_B"

        breakdown.loc[
            breakdown["A_mixed_W12_W34"]
            & ~breakdown["B_both_label"],
            "diagnostic_category"
        ] = "A_only"

        breakdown.loc[
            breakdown["B_both_label"]
            & ~breakdown["A_mixed_W12_W34"],
            "diagnostic_category"
        ] = "B_only"

        breakdown.loc[
            breakdown["A_mixed_W12_W34"]
            & breakdown["B_both_label"],
            "diagnostic_category"
        ] = "A_and_B"

        # -------------------------------------------------------------
        # 9. Identify patients excluded from FULL
        # -------------------------------------------------------------

        excluded_from_full = breakdown.loc[
            ~breakdown["in_FULL"]
        ].copy()

        included_in_full = breakdown.loc[
            breakdown["in_FULL"]
        ].copy()

        # -------------------------------------------------------------
        # 10. Build summaries
        # -------------------------------------------------------------

        summaries = [
            summarize_group(
                merged,
                "ALL_CLINICAL_PATIENTS"
            ),
            summarize_group(
                breakdown,
                "PATIENTS_WITH_A_OR_B"
            ),
            summarize_group(
                included_in_full,
                "A_OR_B_INCLUDED_IN_FULL"
            ),
            summarize_group(
                excluded_from_full,
                "A_OR_B_EXCLUDED_FROM_FULL"
            )
        ]

        summary_df = pd.DataFrame(summaries)

        # -------------------------------------------------------------
        # 11. Additional integrity checks
        # -------------------------------------------------------------

        # A/B patients should be a subset of all clinical patients.
        if len(breakdown) != int(is_ab.sum()):
            raise ValueError(
                "A/B patient count mismatch."
            )

        # Included and excluded groups must partition A/B patients.
        if (
            len(included_in_full)
            + len(excluded_from_full)
            != len(breakdown)
        ):
            raise ValueError(
                "FULL inclusion/exclusion partition is inconsistent."
            )

        # -------------------------------------------------------------
        # 12. Save outputs
        # -------------------------------------------------------------

        cfg.MANIFEST_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        breakdown_path = (
            cfg.MANIFEST_DIR /
            "both_patients_breakdown.csv"
        )

        summary_path = (
            cfg.MANIFEST_DIR /
            "both_patients_summary.csv"
        )

        excluded_path = (
            cfg.MANIFEST_DIR /
            "both_patients_excluded_from_FULL.csv"
        )

        breakdown.to_csv(
            breakdown_path,
            index=False,
            encoding="utf-8-sig"
        )

        summary_df.to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig"
        )

        excluded_from_full.to_csv(
            excluded_path,
            index=False,
            encoding="utf-8-sig"
        )

        # -------------------------------------------------------------
        # 13. Console report
        # -------------------------------------------------------------

        print("\n--- PATIENT-LEVEL BOTH DIAGNOSTIC ---\n")

        print(summary_df.to_string(index=False))

        print("\n--- CLINICAL LABEL DIAGNOSTICS ---")
        print(
            "Patients with >1 distinct clinical label:",
            int(merged["multiple_clinical_labels"].sum())
        )

        print(
            "Patients with mixed W12/W34:",
            int(merged["A_mixed_W12_W34"].sum())
        )

        print(
            "Patients with explicit Both label:",
            int(merged["B_both_label"].sum())
        )

        print("\n--- A/B PATIENTS EXCLUDED FROM FULL ---")

        if excluded_from_full.empty:
            print("None.")
        else:
            cols = [
                "patient_id",
                "labels",
                "wave_group",
                "n_rows",
                "A_mixed_W12_W34",
                "B_both_label"
            ]

            print(
                excluded_from_full[cols].to_string(
                    index=False
                )
            )

        print("\n--- SAVED FILES ---")
        print("[saved]", breakdown_path)
        print("[saved]", summary_path)
        print("[saved]", excluded_path)

        print("\n✅ BOTH diagnostic completed.")
        print(
            "No manifest assignments or cohort definitions were modified."
        )

        if unexpected_labels:
            print(
                "\n⚠️ Review unexpected clinical cohort labels "
                "before interpreting the diagnostic."
            )

        if missing_labels > 0:
            print(
                "\n⚠️ Some clinical rows have missing cohort labels."
            )

        if not unmatched.empty:
            print(
                "\n⚠️ Some clinical patients were not found in the manifest."
            )

    except Exception as exc:
        print("\n❌ BOTH DIAGNOSTIC ERROR")
        print(f"{type(exc).__name__}: {exc}")
        print(
            "Diagnostic did not complete. "
            "Review the error before proceeding."
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
