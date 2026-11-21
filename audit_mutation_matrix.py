
# -*- coding: utf-8 -*-
"""
Read-only audit of patient-level mutation matrices.
Checks:
1. Matrix patient IDs against the expected profiled patients.
2. Provenance of every matrix value == 1.
3. Correct interpretation of matrix zeros.
4. DNA sample-to-patient mapping uniqueness.
5. Exclusion of controls and non-eligible DNA samples.
"""

import pandas as pd

from common import config_official as cfg
from common import io_official as io
from common import cohort


def report(title, value):
    print(f"{title}: {value}")


def audit_cohort(src, manifest, cohort_name):

    print("\n" + "=" * 70)
    print(f"MUTATION MATRIX AUDIT: {cohort_name}")
    print("=" * 70)

    patients = cohort.patients_of(manifest, cohort_name)

    # Reconstruct the expected eligible DNA samples and patients.
    dna = io.eligible_dna_table(src["mapping"]).copy()

    profiled_samples = set(
        src["mutations"]["dbgap_sample_id"].dropna().astype(str)
    )

    dna_in_cohort = dna[
        dna["patient_id"].isin(patients)
    ].copy()

    expected_dna = dna_in_cohort[
        dna_in_cohort["dna_sample"].isin(profiled_samples)
    ].copy()

    expected_patients = set(
        expected_dna["patient_id"].dropna().astype(str)
    )

    # Build the actual matrix using the official function.
    mut = io.build_patient_mutation_matrix(src, patients)

    actual_patients = set(mut.index.astype(str))

    # ---------------------------------------------------------
    # 1. Patient ID consistency
    # ---------------------------------------------------------

    missing_from_matrix = expected_patients - actual_patients
    unexpected_in_matrix = actual_patients - expected_patients

    report("Patients in manifest cohort", len(patients))
    report("Expected profiled patients", len(expected_patients))
    report("Patients in mutation matrix", len(actual_patients))
    report("Expected patients missing from matrix", len(missing_from_matrix))
    report("Unexpected patients in matrix", len(unexpected_in_matrix))

    if missing_from_matrix:
        print("  Missing examples:", sorted(missing_from_matrix)[:10])

    if unexpected_in_matrix:
        print("  Unexpected examples:", sorted(unexpected_in_matrix)[:10])

    # ---------------------------------------------------------
    # 2. Reconstruct expected patient-gene mutation pairs
    # ---------------------------------------------------------

    eligible_variants = io.eligible_variants(src["mutations"])

    variant_rows = eligible_variants.merge(
        expected_dna[["dna_sample", "patient_id"]],
        left_on="dbgap_sample_id",
        right_on="dna_sample",
        how="inner",
        validate="many_to_one"
    )

    expected_pairs = set(
        zip(
            variant_rows["patient_id"].astype(str),
            variant_rows[cfg.GENE_COLUMN].astype(str)
        )
    )

    # composite FLT3: clinical ITD pairs also have a "valid origin"
    itd_pos = set()
    if cfg.FLT3_COMPOSITE:
        itd = io.flt3_itd_status(src["clinical"])
        itd_pos = set(itd.index[itd["any_pos"]].astype(str)) & expected_patients
        expected_pairs |= {(p, cfg.FLT3_GENE) for p in itd_pos}

    observed_pairs = set()

    for patient_id in mut.index:
        for gene in mut.columns:
            value = mut.at[patient_id, gene]

            if value == 1:
                observed_pairs.add((str(patient_id), str(gene)))

    false_positive_ones = observed_pairs - expected_pairs
    missing_ones = expected_pairs - observed_pairs

    report("Expected patient-gene positive pairs", len(expected_pairs))
    report("Observed matrix ones", len(observed_pairs))
    report("Matrix ones without eligible variant provenance", len(false_positive_ones))
    report("Eligible variant pairs absent from matrix", len(missing_ones))

    if false_positive_ones:
        print("  Unexplained matrix ones:", list(false_positive_ones)[:10])

    if missing_ones:
        print("  Missing positive pairs:", list(missing_ones)[:10])

    # ---------------------------------------------------------
    # 3. Zero interpretation
    # ---------------------------------------------------------

    non_binary = int((~mut.isin([0, 1])).to_numpy().sum())

    expected_cells = len(mut.index) * len(mut.columns)
    observed_zeros = int((mut == 0).to_numpy().sum())

    # Every cell not represented by an eligible patient-gene
    # pair should be zero under the current matrix construction.
    expected_zero_count = expected_cells - len(expected_pairs)

    report("Matrix dimensions", mut.shape)
    report("Number of zero cells", observed_zeros)
    report("Expected number of zero cells", expected_zero_count)
    report("Non-binary cells", non_binary)

    # ---------------------------------------------------------
    # 4. DNA sample mapping
    # ---------------------------------------------------------

    duplicate_sample_ids = int(
        dna["dna_sample"].duplicated(keep=False).sum()
    )

    missing_patient_ids = int(dna["patient_id"].isna().sum())

    report("Eligible DNA samples", len(dna))
    report("Duplicate eligible DNA sample rows", duplicate_sample_ids)
    report("Eligible DNA samples without patient ID", missing_patient_ids)

    # ---------------------------------------------------------
    # 5. Controls and non-eligible samples
    # ---------------------------------------------------------

    mapping = src["mapping"]

    used_samples = set(expected_dna["dna_sample"].dropna().astype(str))

    excluded_mapping = mapping.loc[
        mapping["dbgap_dnaseq_sample"].notna()
        & (
            mapping["is_control"]
            | (
                cfg.DNA_REQUIRE_MAPPING_INCLUDE_FLAG
                & mapping["dna_include_in_analysis"].ne("yes")
            )
        ),
        "dbgap_dnaseq_sample"
    ]

    excluded_sample_ids = set(excluded_mapping.dropna().astype(str))

    overlap_excluded = used_samples & excluded_sample_ids

    control_patients_in_cohort = (
        set(patients) & set(cfg.CONTROL_SUBJECT_IDS)
    )

    report("Used DNA samples", len(used_samples))
    report("Excluded DNA samples overlapping used samples", len(overlap_excluded))
    report("Control subject IDs in cohort", len(control_patients_in_cohort))

    if overlap_excluded:
        print("  Excluded samples incorrectly used:", sorted(overlap_excluded)[:10])

    # ---------------------------------------------------------
    # 5b. FLT3 composite == WES OR ITD ; other genes untouched
    # ---------------------------------------------------------

    flt3_ok = True
    if cfg.FLT3_COMPOSITE:
        wes = io.build_patient_mutation_matrix(src, patients, flt3_composite=False)
        exp = wes[cfg.FLT3_GENE].copy() if cfg.FLT3_GENE in wes.columns else pd.Series(0, index=wes.index)
        exp[[str(i) in itd_pos for i in exp.index]] = 1
        got = mut[cfg.FLT3_GENE].reindex(exp.index)
        others = [c for c in wes.columns if c != cfg.FLT3_GENE]
        same_others = bool((wes[others].reindex(mut.index).to_numpy() == mut[others].to_numpy()).all())
        flt3_ok = bool((got.to_numpy() == exp.to_numpy()).all() and same_others and list(wes.index) == list(mut.index))
        report("FLT3 WES-only positives", int(wes[cfg.FLT3_GENE].sum()) if cfg.FLT3_GENE in wes.columns else 0)
        report("FLT3 ITD-positive (profiled)", len([i for i in mut.index if str(i) in itd_pos]))
        report("FLT3 composite positives", int(mut[cfg.FLT3_GENE].sum()))

    # ---------------------------------------------------------
    # 6. Overall status
    # ---------------------------------------------------------

    checks = {
        "patient_set_match": not missing_from_matrix and not unexpected_in_matrix,
        "positive_pair_provenance": not false_positive_ones and not missing_ones,
        "binary_matrix": non_binary == 0,
        "zero_count_consistency": observed_zeros == expected_zero_count,
        "unique_DNA_mapping": duplicate_sample_ids == 0,
        "complete_DNA_patient_IDs": missing_patient_ids == 0,
        "no_excluded_sample_overlap": not overlap_excluded,
        "no_control_subject_in_cohort": not control_patients_in_cohort,
        "flt3_composite_definition": flt3_ok,
    }

    print("\nCHECK SUMMARY")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} | {name}")

    return checks


def main():

    print("Loading official sources...")
    src = io.load_sources()

    manifest = pd.read_csv(
        cfg.MANIFEST_FILE,
        dtype={"patient_id": "string"}
    )

    results = {}

    for cohort_name in ["DISCOVERY", "VALIDATION", "FULL"]:
        results[cohort_name] = audit_cohort(
            src,
            manifest,
            cohort_name
        )

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for cohort_name, checks in results.items():
        passed = sum(checks.values())
        total = len(checks)
        print(f"{cohort_name}: {passed}/{total} checks passed")

    if not all(
        all(checks.values())
        for checks in results.values()
    ):
        print("\n[WARNING] At least one audit check failed.")
    else:
        print("\n[PASS] All mutation matrix audit checks passed.")


if __name__ == "__main__":
    main()
