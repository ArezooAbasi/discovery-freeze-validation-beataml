# -*- coding: utf-8 -*-
"""
config_official.py  --  every methodological decision is recorded in this single file.
Every value here must be fixed (frozen) before Validation is run and reported in Methods.
Confirm the values marked [DECIDE] against your own audit outputs.

v2: the sections "SELECTION RULES", "ADJUSTMENT", "VALIDATION" and "SIMULATION" were added
(they used to be scattered in the old config.py). No previous value was changed.
v4 (provenance review): FREEZE_PARAMS now also covers the "data definition" parameters, and ANALYSIS_DEFINITION_PARAMS was added.
"""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
OFFICIAL_DIR = Path(os.environ.get(
    "BEATAML_DIR", PROJECT_DIR / "biodev-beataml2.0_data-6e403e1"))

FILES = {
    "mapping":   OFFICIAL_DIR / "beataml_waves1to4_sample_mapping.xlsx",
    "clinical":  OFFICIAL_DIR / "beataml_wv1to4_clinical.xlsx",
    "mutations": OFFICIAL_DIR / "beataml_wes_wv1to4_mutations_dbgap.txt",
    "drug":      OFFICIAL_DIR / "beataml_probit_curve_fits_v4_dbgap.txt",
}
DRUG_FAMILIES_FILE = OFFICIAL_DIR / "beataml_drug_families.xlsx"   # annotation only (Stage 05)
SHEETS = {"mapping": "sample_map", "clinical": "summary"}

OUTPUT_ROOT = Path(os.environ.get("BEATAML_OUT", PROJECT_DIR / "output"))
MANIFEST_DIR = OUTPUT_ROOT / "manifest"
MANIFEST_FILE = MANIFEST_DIR / "patient_level_cohort_manifest.csv"

def cohort_dir(cohort):
    d = OUTPUT_ROOT / cohort
    d.mkdir(parents=True, exist_ok=True)
    return d

# ---------------------------------------------------------------- unit of analysis
UNIT = "patient"                 # dbgap_subject_id; no test is ever run at the sample level
CONTROL_PATIENT_IDS = {"0"}      # patientId of the pooled controls (mapping file)
CONTROL_SUBJECT_IDS = {"2096"}   # dbgap_subject_id of the pooled controls

# ---------------------------------------------------------------- Wave
COHORT_LABEL_TO_WAVEGROUP = {
    "waves1+2": "W12",
    "waves3+4": "W34",
    "both": "BOTH",
}

# ---------------------------------------------------------------- DNA eligibility
DNA_REQUIRE_MAPPING_INCLUDE_FLAG = True

# ---------------------------------------------------------------- mutation definition
# A zero in the matrix = "no eligible variant reported", NOT a confirmed wild type. An empty/unknown class is not eligible.
GENE_COLUMN = "symbol"
VARIANT_CLASS_COLUMN = "variant_classification"
EXCLUDED_VARIANT_CLASSES = {
    "silent", "synonymous_variant", "intron", "intron_variant",
    "3'utr", "5'utr", "3_prime_utr_variant", "5_prime_utr_variant",
    "3'flank", "5'flank", "upstream_gene_variant", "downstream_gene_variant",
    "igr", "intergenic_variant", "rna", "non_coding_transcript_exon_variant",
}
MIN_T_VAF = None   # deliberate decision: no VAF threshold; every eligible coding variant (after the class filter) counts as mutation-reported. Report this in Methods.

# ---------------------------------------------------------------- composite FLT3 (v5)
# FLT3 = WES mutation OR clinical FLT3-ITD (WES hardly detects ITD). Only the FLT3 column of the matrix changes; the number of hypotheses stays fixed.
# ITD positive = "at least one clinical row of that patient is positive" (the ITD sample time is not aligned with the DNA/AUC sample; state this in Methods/Limitations).
# Empty/NaN = "unknown" => not counted as positive. Any unknown value (neither in POSITIVE nor in NEGATIVE) => error.
# [DECIDE] Fill the values after running audit_flt3_itd.py (value_counts of the column); all compared after lower().strip().
FLT3_COMPOSITE = True
FLT3_GENE = "FLT3"
FLT3_ITD_COLUMN = "FLT3-ITD"
FLT3_ITD_POSITIVE_VALUES = {"positive"}
FLT3_ITD_NEGATIVE_VALUES = {"negative"}

# ---------------------------------------------------------------- Drug response
DRUG_ID_COLUMN = "inhibitor"
DRUG_METRIC = "auc"           # note: lower AUC = more sensitive
DRUG_REQUIRE_CONVERGED = False
DRUG_REQUIRE_PAPER_INCLUSION = False
DRUG_ALLOWED_STATUS = None
DRUG_REPEAT_POLICY = "median"

# ---------------------------------------------------------------- Stage 1 thresholds
MIN_DRUG_SAMPLES = 30
MIN_MUTATED = 10
MIN_MUTATED_WITH_AUC = 10
MIN_WT_WITH_AUC = 10

# ---------------------------------------------------------------- statistics
FDR_METHOD = "fdr_bh"
SENSITIVITY_FDR_METHODS = {"FDR_BY": "fdr_by"}
BOOTSTRAP_B = 2000
RANDOM_SEED = 20261112

# ================================================================ MAIN SELECTION RULE (no Top-N; DISCOVERY only)
# Main rule (v3 / Path B): (1) fixed nominal screen: p < NOMINAL_P_MAX and n >= MAIN_MIN_N_PER_GROUP  (2) bootstrap + split-half
# (3) exploratory co-mutation adjustment  (4) freeze  (5) evaluation of the same list in VALIDATION.
# BH-FDR (SELECT_FDR_Q) is the v2 rule and a sensitivity analysis only; it plays no role in the main selection.
# "Reproducibility in Validation" is not an input of selection; it is an output of the evaluation.
# --- v3 / Path B (FDR-null framework): entry to the later stages is a "nominal screen with a fixed threshold", not BH.
# No FDR control is claimed on the final list; the output = exploratory prioritisation; zero candidates is a valid result.
NOMINAL_P_MAX = 0.01              # [DECIDED v3] fixed nominal threshold (raw p, two-sided) over the whole eligible family
MAIN_MIN_N_PER_GROUP = 20         # [DECIDED v3] minimum n per group for the main rule (= REFERENCE_THRESHOLD)
# BH gate (v2 rule) is a sensitivity analysis only; it never replaces or merges with the main rule.
SELECT_FDR_Q = 0.10               # [SENSITIVITY ONLY]
# Sensitivity rule "effect size + uncertainty" (not the main rule).
# conservative_effect = min(|CI_low|,|CI_high|) if the CI excludes 0 and has the sign of the observed effect; otherwise 0.
EFFECT_MIN_CONSERVATIVE = 0.20    # [DECIDED v3] sensitivity only
CI_WIDTH_MAX = 0.60               # [DECIDED v3] bootstrap CI width (rank-biserial units)
MIN_N_EFFECT_RULE = 20            # [DECIDED v3] aligned with MAIN_MIN_N_PER_GROUP
SENSITIVITY_TOP_N = [10, 20, 30]  # legacy threshold-stable Top-N rule; sensitivity only

# ================================================================ SELECTION RULES (stability; DISCOVERY only)
# Sign convention (identical across the project): rank_biserial_effect = 1 - 2*U1/(n1*n2), U1 for the mutated group.
#   effect > 0  => AUC of the mutated group is lower  => mutated more sensitive
#   effect < 0  => AUC of the mutated group is higher => mutated more resistant
THRESHOLDS = [10, 20, 30]          # minimum n per group for threshold-stability
REFERENCE_THRESHOLD = MAIN_MIN_N_PER_GROUP
TOP_N_FOR_OVERLAP = 20
DIRECTION_CONSISTENCY_MIN = 95.0   # diagnostic column only (boot_direction_ok): holds automatically when the 95% CI excludes 0; not a gate
N_SPLIT_HALF = 1000
SPLIT_HALF_MIN_GROUP_N = 5
SPLIT_HALF_DIRECTION_MIN = 90.0    # gate: % of ALL attempted splits (denominator = N_SPLIT_HALF; an invalid split = failure) in which both halves share the full-sample sign
N_PERMUTATIONS = 2000              # single-pair permutation test (Stage 09)
N_WY_PERMUTATIONS = 1000           # Westfall-Young min-p (Stage 08b)

# ================================================================ ADJUSTMENT (co-mutation)
MIN_GENE_MUTATION_COUNT = 5     # minimum NUMBER of carrier patients (not a percentage)
COOCCUR_SCREEN_P = 0.05
OUTCOME_SCREEN_P = 0.05
MAX_COVARIATES_HARD_CAP = 5
MIN_N_PER_PREDICTOR = 10
ADJ_ATTENUATION_PCT = 50.0
# Only these labels pass the adjustment stage and enter the frozen list
# Main rule: only robust_to_adjustment passes. "No confounder candidate found" is not evidence of robustness.
ADJ_PASS_LABELS = {"robust_to_adjustment"}
# Old/lenient version; used for sensitivity analyses only (master-table column and simulation)
ADJ_PASS_LABELS_PERMISSIVE = {"robust_to_adjustment", "no_confounder_candidate_found"}
# Source of p in classify: "classic" (previous behaviour; current freeze) or "HC3". Changing it changes the freeze.
ADJ_P_SOURCE = "classic"
# "Predefined/exploratory" confounding questions (gene -> confounders); do not call them prespecified unless you have a dated protocol
EXTRA_CONFOUNDING_CHECKS = {
    "NPM1": ["FLT3", "DNMT3A", "IDH1", "IDH2", "TP53", "ASXL1", "STAG2"],
    "NRAS": ["KRAS", "PTPN11", "FLT3"],
    "RUNX1": ["FLT3"],
}
# Manual display names of drugs (inhibitor -> name)
DRUG_NAME_OVERRIDES = {"Lze": "AT7519", "Qnz": "QNZ"}

# ================================================================ VALIDATION
VALIDATION_FDR = 0.10     # BH over m = number of evaluable candidates
POWER_TARGET = 0.80
POWER_B = 2000

# ================================================================ SIMULATION (Stage 16)
SIM_R_NULL = 500                     # v3: for tighter CIs (rule of three: zero events in 500 => 95% upper bound ~ 0.6%)
SIM_R_ALT = 300
SIM_MC_CI_LEVEL = 0.95             # Clopper-Pearson for all proportions
SIM_DELTAS_SD = [0.6, 0.9, 1.2]      # AUC shift in SD units of the same drug (official AUC is on a ~0-300 scale)
SIM_N_BOOTSTRAP = 300
SIM_N_SPLIT_HALF = 200
SIM_NOMINAL_P = 1e-3
SIM_COMPARATOR_Q = SELECT_FDR_Q   # BH comparator = gate of the main rule
SIM_TOP_K = 20
SIM_R_NULL_MISSING = 300          # replicates for the null with genotype-dependent missingness (per type)
SIM_MISS_DRUG_FRAC = 0.25         # fraction of drugs to which missingness is applied
SIM_MISS_DROP_P = 0.5             # probability of dropping an eligible value
SIM_FAMILY_MIN_N = MAIN_MIN_N_PER_GROUP   # shared H20 family for all methods in the simulation (n per group >= this value)

# ================================================================ FREEZE BINDING
# (a) "Data/analysis definition" parameters: changing them changes the number of patients, the test family or the effects.
# Besides the params hash, a separate analysis_definition_sha256 is recorded and re-checked in VALIDATION.
ANALYSIS_DEFINITION_PARAMS = [
    "UNIT", "CONTROL_PATIENT_IDS", "CONTROL_SUBJECT_IDS", "COHORT_LABEL_TO_WAVEGROUP",
    "DNA_REQUIRE_MAPPING_INCLUDE_FLAG", "GENE_COLUMN", "VARIANT_CLASS_COLUMN", "EXCLUDED_VARIANT_CLASSES", "MIN_T_VAF",
    "DRUG_ID_COLUMN", "DRUG_METRIC", "DRUG_REPEAT_POLICY", "DRUG_REQUIRE_CONVERGED", "DRUG_REQUIRE_PAPER_INCLUSION",
    "DRUG_ALLOWED_STATUS", "EXTRA_CONFOUNDING_CHECKS",
    "FLT3_COMPOSITE", "FLT3_GENE", "FLT3_ITD_COLUMN", "FLT3_ITD_POSITIVE_VALUES", "FLT3_ITD_NEGATIVE_VALUES",
]
# (b) Selection/evaluation parameters. Their hash is stored in freeze_manifest.json and re-checked in VALIDATION.
FREEZE_PARAMS = [
    "NOMINAL_P_MAX", "MAIN_MIN_N_PER_GROUP", "SELECT_FDR_Q", "EFFECT_MIN_CONSERVATIVE", "CI_WIDTH_MAX", "MIN_N_EFFECT_RULE",
    "SENSITIVITY_TOP_N", "THRESHOLDS", "REFERENCE_THRESHOLD", "TOP_N_FOR_OVERLAP", "DIRECTION_CONSISTENCY_MIN",
    "SPLIT_HALF_DIRECTION_MIN", "SPLIT_HALF_MIN_GROUP_N", "N_SPLIT_HALF", "BOOTSTRAP_B", "ADJ_PASS_LABELS",
    "ADJ_PASS_LABELS_PERMISSIVE", "ADJ_ATTENUATION_PCT", "ADJ_P_SOURCE", "MAX_COVARIATES_HARD_CAP", "MIN_N_PER_PREDICTOR",
    "MIN_GENE_MUTATION_COUNT", "COOCCUR_SCREEN_P", "OUTCOME_SCREEN_P", "RANDOM_SEED", "MIN_DRUG_SAMPLES", "MIN_MUTATED",
    "MIN_MUTATED_WITH_AUC", "MIN_WT_WITH_AUC", "FDR_METHOD", "VALIDATION_FDR", "POWER_TARGET", "N_WY_PERMUTATIONS",
] + ANALYSIS_DEFINITION_PARAMS


def require_thresholds():
    missing = [k for k in ("MIN_DRUG_SAMPLES", "MIN_MUTATED",
                           "MIN_MUTATED_WITH_AUC", "MIN_WT_WITH_AUC")
               if globals()[k] is None]
    if missing:
        raise SystemExit(f"[config] این آستانه‌ها باید در config_official.py تعیین شوند: {missing}")
