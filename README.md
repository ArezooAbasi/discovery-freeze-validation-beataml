# BeatAML mutation–drug association pipeline

A **Discovery → Freeze → Validation** framework for the *exploratory prioritisation* of nominal
mutation–drug-response signals in a large ex-vivo drug screen. The BeatAML 2.0 cohort
(acute myeloid leukaemia; ex-vivo inhibitor AUC + whole-exome mutations + clinical data) is the illustrating case.

> **What this is — and is not.**
> The pipeline reports mutation–drug pairs that are *stable under the checks performed* and then evaluates the
> **same frozen list, one time, in a separate set of waves**. It is an exploratory prioritisation tool.
> **No FDR/FDP control is claimed for the final list, and zero candidates is a valid result.**
> The validation is *independent cross-wave validation within BeatAML* — not external validation and not an
> independent cohort.

---

## 1. Design at a glance

```
official BeatAML files
        │
        ▼
 00  patient manifest  ──►  00b/00c  wave-separation + "Both"-label audits
        │
        ├── DISCOVERY = waves 1+2  (W12)          ├── VALIDATION = waves 3+4 (W34)
        │                                         │   (patients labelled in both groups are excluded from both)
        ▼                                         │
 01  gene × drug screen (Mann-Whitney, rank-biserial effect, BH/BY)
 02  candidate pool        (nominal screen + sensitivity rules, kept in separate columns)
 03  bootstrap / split-half / winsorise-trim stability
 04  stability gates  →  stable pairs
 05-07  drug names, provenance, annotation
 08  co-mutation adjustment (exploratory, data-adaptive)
 08b FREEZE  →  frozen_candidate_list.csv  + freeze_manifest.json (hashes)      ◄── locks Discovery
        │
        ▼
 01 --cohort VALIDATION --frozen …   (one shot; a lock marker is written before any computation)
        │
        ▼
 09-11 sensitivity checks · 14 validation report · 12 audit · 13 master table · 15 figures
 16  simulation benchmark of the selection rule against standard multiplicity procedures
```

* **Unit of analysis:** the patient (`dbgap_subject_id`); no test is run at the sample level.
* **Sign convention (whole project):** `rank_biserial_effect = 1 − 2·U₁/(n₁·n₂)`, `U₁` for the mutated group.
  `effect > 0` ⇒ mutated AUC is **lower** ⇒ mutated is **more sensitive**; `effect < 0` ⇒ more resistant.
  OLS coefficients are on AUC and therefore have the *opposite* sign (`beta > 0` ⇒ more resistant).
* **Zeros in the mutation matrix** mean "no eligible variant reported", not a confirmed wild type.
* **FLT3** is a composite: WES mutation **OR** clinical FLT3-ITD (WES hardly detects ITD). Only the FLT3 column changes.

### Selection rule (main result)

| Gate | Definition (values from `common/config_official.py`) |
|---|---|
| **G1** nominal screen | raw two-sided `p < 0.01` and `n ≥ 20` per group, over the whole eligible family (no Top-N, no BH gate) |
| **G2** stable | bootstrap 95 % CI excludes 0 with the observed sign **and** split-half direction match in ≥ 90 % of all attempted splits |
| **G3** adjustment | exploratory co-mutation adjustment label is `robust_to_adjustment` (only this label passes) |

`frozen_candidate_list.csv = G1 ∧ G2 ∧ G3`. BH-gate, Top-N and effect-size/CI rules are **sensitivity analyses only**
and are never merged into the main list. The validation effect (never used in selection) is the primary effect
estimate for frozen candidates; `effect_EB` is an exploratory shrinkage estimate, not a selection-adjusted effect.

---

## 2. Repository layout

```
.
├── README.md
├── requirements.txt
├── common/                         shared library (freeze-bound; see §6)
│   ├── config_official.py          ALL methodological decisions / parameters
│   ├── io_official.py              reads the official files, builds patient-level matrices
│   ├── cohort.py                   patient manifest + sample-flow table
│   ├── data.py                     cohort matrices, I/O helpers, hashing, freeze/validation lock
│   ├── stats.py                    the only statistics implementation (tests, bootstrap, split-half, OLS, EB, power)
│   ├── screen.py                   vectorised gene × drug screen + single-step min-p
│   ├── selection.py                the selection rule (shared by the pipeline and the simulation)
│   ├── adjust.py                   co-mutation screening + adjusted OLS
│   └── synthetic.py                synthetic data for code testing only
├── 00_build_patient_manifest.py    … 16_simulation_benchmark.py    pipeline stages (table below)
├── post_hoc/                       post hoc analyses, Methods 2.9.1–2.9.8 (see §5.3)
│   ├── posthoc_validation_analysis.py   2.9.1 permutation baseline, 2.9.2 FLT3 WES-only validation
│   └── posthoc_discovery_criteria.py    2.9.3–2.9.8 (Tables 9, 11, 12, 13, 18, 19; concordance by gene)
├── audit_*.py, compare_repro.py, safe_io.py, fix_simulation_null_label.py   utilities
└── test_pipeline.py                tests on synthetic data (no real data needed)
```

### Stages

| Stage | Purpose | Main output(s) |
|---|---|---|
| `00_build_patient_manifest.py` | Patient-level manifest from the official files; no statistics | `output/manifest/*` |
| `00b_wave_separation_audit.py` | Hard checks that DISCOVERY and VALIDATION are disjoint | `wave_separation_audit.csv` |
| `00c_both_diagnostic.py` | Diagnostic of patients with mixed/"Both" wave labels (does not modify the manifest) | `output/manifest/both_patients_*.csv` |
| `01_association_screen.py` | Gene × drug screen (`--cohort DISCOVERY\|FULL`) **or** one-shot evaluation of a frozen list (`--cohort VALIDATION --frozen …`) | `mutation_drug_associations.csv`, `validation_results.csv` |
| `02_threshold_robustness.py` | Threshold sensitivity + candidate pool | `candidate_pairs.csv`, `mutation_drug_threshold_robustness.csv` |
| `03_stability_checks.py` | Bootstrap, split-half, winsorise/trim | `bootstrap_robustness_results.csv` |
| `04_export_stable_pairs.py` | The only place "stable" is defined; all gates logged | `stability_gate_audit.csv`, `stable_pairs.csv` |
| `05_drug_name_lookup.py` · `06_drug_provenance.py` · `07_annotate_full_table.py` | Drug names/families/targets, provenance table, annotation | `output/shared/drug_lookup_table.csv`, `drug_provenance.csv`, `mutation_drug_associations_annotated.csv` |
| `08_adjusted_analysis.py` | Co-mutation-adjusted AUC ~ gene for every stable pair | `adjusted_analysis_results.csv` |
| `08b_freeze_candidates.py` | Gates for **all** candidates, freeze, hashes | `freeze_audit.csv`, `frozen_candidate_list.csv`, `frozen_sensitivity_list.csv`, `freeze_manifest.json` |
| `08c_global_sensitivity_diagnostic.py` | Diagnostic: how much of an effect is shared across all drugs (not a gate) | `global_sensitivity_diagnostic.csv`, `panel_size_by_gene.csv` |
| `09_sensitivity_analyses.py` | Outliers, permutation, confounder removal, repeat-record rules, WES-only FLT3 | `sensitivity_analyses_results.csv` |
| `10_predefined_confounding.py` | Predefined/exploratory confounding questions (`EXTRA_CONFOUNDING_CHECKS`) | `predefined_confounding_results.csv` |
| `11_analysis_population.py` | Patient-level analysis population for every stable pair | `analysis_population.csv` |
| `12_audit.py` | Automatic consistency audit (hash chains, BH recomputation, coverage, signs) | console report |
| `13_master_table.py` | Joins earlier outputs; recomputes nothing | `final_master_association_table.csv` |
| `14_validation_report.py` | Validation report with MDE/power; "not confirmed" ≠ "rejected" | `validation_report.csv`, `validation_summary.txt` |
| `15_make_figures.py` · `15b_make_figures_detail.py` | Paper figures (PNG 300 dpi + PDF) · per-pair figures for internal review | `output/figures/` |
| `16_simulation_benchmark.py` | Simulation benchmark (see §7) | `simulation_*.csv` |
| `post_hoc/posthoc_validation_analysis.py` | Post hoc 2.9.1–2.9.2: permutation chance baseline; WES-only FLT3 validation (§5.3) | `permutation_baseline_summary.json`, `permutation_null_counts.csv`, `flt3_wes_only_validation.csv`, `flt3_not_evaluable_pairs.csv` |
| `post_hoc/posthoc_discovery_criteria.py` | Post hoc 2.9.3–2.9.8: validation rate by discovery criterion, G3 logistic models, false-declaration proportion, co-mutated false declarations, concordance by gene (§5.3) | `posthoc_table*.csv`, `posthoc_concordance_by_gene.csv` |

Utilities: `audit_flt3_itd.py` (run before/after filling the FLT3-ITD values in the config), `audit_dna_timing.py`,
`audit_mutation_matrix.py` (read-only audits), `compare_repro.py` (content-level comparison of two output roots),
`safe_io.py` (`read_csv_safe`, which does not turn the string `"null"` into NaN),
`fix_simulation_null_label.py` (one-off converter for old simulation outputs).

---

## 3. Requirements

* Python ≥ 3.9 (developed on 3.9; the test suite also passes on 3.12)
* `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib ≥ 3.6`, `openpyxl`

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` gives lower bounds. For exact reproduction of a manuscript run, record the versions you used
(`pip freeze > requirements-lock.txt`).

---

## 4. Data

The data are **not** included in this repository. The pipeline reads these official BeatAML 2.0 files
(see `FILES`, `SHEETS`, `DRUG_FAMILIES_FILE` in `common/config_official.py`):

| Config key | File | Sheet |
|---|---|---|
| `mapping` | `beataml_waves1to4_sample_mapping.xlsx` | `sample_map` |
| `clinical` | `beataml_wv1to4_clinical.xlsx` | `summary` |
| `mutations` | `beataml_wes_wv1to4_mutations_dbgap.txt` | – |
| `drug` | `beataml_probit_curve_fits_v4_dbgap.txt` | – |
| (annotation only) | `beataml_drug_families.xlsx` | – |

They accompany the BeatAML 2.0 manuscript and are distributed openly under **CC BY 4.0**:

* Zenodo (release v3.0): <https://zenodo.org/records/10654808> — DOI [10.5281/zenodo.10654808](https://doi.org/10.5281/zenodo.10654808)
* GitHub: [`biodev/beataml2.0_data`](https://github.com/biodev/beataml2.0_data)

This pipeline was run on the GitHub snapshot `biodev-beataml2.0_data-6e403e1`; the Zenodo v3.0 release notes state that
no existing file was changed (it only adds a license file and malignant cell type scores).
**Data are not redistributed in this repository.** Download them from the source above and cite the original work (see Section 11).
Put them in one folder and point the pipeline to it:

```bash
export BEATAML_DIR=/path/to/beataml_data      # default: ./biodev-beataml2.0_data-6e403e1
export BEATAML_OUT=/path/to/output            # default: ./output
```

(Windows PowerShell: `$env:BEATAML_DIR="C:\path\to\beataml_data"`.)

---

## 5. Usage

### 5.1 Smoke test without real data

```bash
export BEATAML_SYNTHETIC=1 BEATAML_OUT=./_out
python test_pipeline.py                 # 20 tests, synthetic data only
python 16_simulation_benchmark.py --quick
```

Synthetic data (`common/synthetic.py`) are for code testing only and have no relation to the real data.

### 5.2 Full run on the real data

Run from the repository root, in this order:

```bash
# --- manifest and audits
python 00_build_patient_manifest.py
python 00b_wave_separation_audit.py
python 00c_both_diagnostic.py
python audit_flt3_itd.py                # check the FLT3-ITD values, then fill them in common/config_official.py

# --- DISCOVERY (waves 1+2)
python 01_association_screen.py --cohort DISCOVERY
python 02_threshold_robustness.py --cohort DISCOVERY
python 03_stability_checks.py     --cohort DISCOVERY
python 04_export_stable_pairs.py  --cohort DISCOVERY
python 05_drug_name_lookup.py
python 06_drug_provenance.py      --cohort DISCOVERY
python 07_annotate_full_table.py  --cohort DISCOVERY
python 08_adjusted_analysis.py    --cohort DISCOVERY
python 08b_freeze_candidates.py                                   # FREEZE: hashes lists, parameters, manifest, sources, code

# --- VALIDATION (waves 3+4): one shot, locked
python 01_association_screen.py --cohort VALIDATION --frozen output/DISCOVERY/frozen_candidate_list.csv
python 01_association_screen.py --cohort VALIDATION --frozen output/DISCOVERY/frozen_sensitivity_list.csv --tag sensitivity   # optional

# --- checks, report, tables, figures
python 09_sensitivity_analyses.py --cohort DISCOVERY
python 10_predefined_confounding.py     --cohort DISCOVERY
python 11_analysis_population.py          --cohort DISCOVERY
python 14_validation_report.py                                    # add --tag sensitivity for the sensitivity list
python 12_audit.py --cohort DISCOVERY
python 12_audit.py --cohort VALIDATION
python 13_master_table.py
python 16_simulation_benchmark.py --cohort DISCOVERY              # expensive; --quick for a test run
python 15_make_figures.py

# --- descriptive side analysis (whole cohort)
python 01_association_screen.py --cohort FULL
```

Optional: `08c_global_sensitivity_diagnostic.py`, `15b_make_figures_detail.py`,
`16_simulation_benchmark.py --multi-m1 10 20 30` (multi-pair scenario), `--with-wy` (single-step min-p comparator).

### 5.3 Post hoc analyses (Methods 2.9)

Run after Validation. These analyses are **post hoc**: they take no part in selection and modify no frozen output.
Each script writes its tables to the current directory.

```bash
export BEATAML_OUT=/path/to/output     # the main run folder; must contain DISCOVERY/, FULL/, manifest/ and VALIDATION/
cd post_hoc
python posthoc_validation_analysis.py perm 2000   # 2.9.1 chance baseline (joint permutation of Validation AUC rows)
python posthoc_validation_analysis.py flt3        # 2.9.2 FLT3 validated with WES-only labels (ITD-only as non-mutated / excluded)
python posthoc_discovery_criteria.py              # 2.9.3-2.9.8 and Tables 9, 11, 12, 13, 18, 19
```

| Paper section | Script | Output file(s) |
|---|---|---|
| 2.9.1, 3.6.6 | `posthoc_validation_analysis.py perm` | `permutation_baseline_summary.json`, `permutation_null_counts.csv` |
| 2.9.2, 3.6.7, Table 10 | `posthoc_validation_analysis.py flt3` | `flt3_wes_only_validation.csv`, `flt3_not_evaluable_pairs.csv` |
| 2.9.3, 3.6.8, Table 11 | `posthoc_discovery_criteria.py` | `posthoc_table11_validation_rate_by_criterion.csv` |
| 2.9.4, 3.6.4, Table 9 | `posthoc_discovery_criteria.py` | `posthoc_table9_powered_discordant.csv` |
| 2.9.5, 3.7.7, Table 19 | `posthoc_discovery_criteria.py` | `posthoc_table19_null_counts.csv`, `posthoc_table19_storey.csv` |
| 2.9.6, 3.6.9, Tables 12-13 | `posthoc_discovery_criteria.py` | `posthoc_table12_logistic_G3.csv`, `posthoc_table13_G3_within_q_strata.csv` |
| 2.9.7, 3.7.4, Table 18 | `posthoc_discovery_criteria.py` | `posthoc_table18_BH_comutated_false.csv` |
| 2.9.8, 3.6.2 | `posthoc_discovery_criteria.py` | `posthoc_concordance_by_gene.csv` |

Extra inputs of the first script: `FULL/analysis_population.csv`, `manifest/flt3_composite_provenance_VALIDATION.csv`,
`VALIDATION/validation_report*.csv`. The script stops with an assertion if the reconstructed data do not give the
published counts (128 reconstructable pairs, 64 validated, 41 ITD-only and 17 WES-positive FLT3 patients).
Results are reproducible only from the main run folder that holds the freeze and the Validation outputs; if you run them on a
re-run folder (for example `output_report`), say so when you cite the numbers.

---

## 6. Freeze binding and the validation lock

`08b_freeze_candidates.py` writes `freeze_manifest.json`, which hashes: the frozen lists, every Discovery file that
shaped the decision, the selection parameters (`FREEZE_PARAMS`), a separate `analysis_definition_sha256`, the patient
manifest, the official source files, and the code (full hash **and** AST hash).

* `01 --cohort VALIDATION` and `14_validation_report.py` re-verify all of this (`common/data.verify_freeze`); a hard
  mismatch stops the run.
* A **lock marker** (`validation_started*.json`) is written *before* any Validation computation. Afterwards DISCOVERY
  and the freeze cannot be re-run or overwritten, and Validation cannot be repeated (there is no `--force`).
  Revising a freeze requires archiving the complete Validation output and recording the reason, outside these scripts.
* **Hard-locked code** = `common/*.py` (except `config_official.py`, whose *parameters* are hashed separately) and
  stage scripts `00` – `08c`. Any change that alters the program (AST) after a freeze is a hard stop and requires a
  new freeze + a Discovery re-run. Changes only in comments/docstrings are allowed and produce a warning that
  records the files. All other files (simulation, reports, figures, utilities) produce a warning only.

If you fork this repository, keep these rules in mind before editing the hard-locked files.

---

## 7. Simulation benchmark (stage 16)

The simulation runs on the **real Discovery matrices**: rows of the drug matrix are permuted jointly, so genotype and
co-mutation structure stay fixed and only the gene → AUC link is broken. Planted pairs are gene–drug pairs, shifted by
`delta × SD` of that drug (the official AUC scale is ≈ 0–300). The adjustment step is part of the rule.

* **Family H20 (main comparison):** all methods run on the same family `{n_mut ≥ 20, n_wt ≥ 20}`:
  `MAIN_N / MAIN_NS / MAIN_NSA`, `NOMINAL_STRICT`, `BH`, `BY`, `HOLM`, `TOPK_MATCHED`, data splitting, optional single-step min-p.
* **Family operational (supplement):** each comparator on its own predefined family (`n ≥ 10`), plus legacy Top-N chains.
* **Scenarios:** `global_null`, `null_gt_missing`, `null_informative_missing` (data artefact, reported separately), `alt`,
  and optionally `multi_alt_m<M>`.
* **Metrics:** `FDP = V / max(R, 1)`; `FDR_hat` = mean FDP over replicates; `Power = S / m1`; Clopper–Pearson CIs for
  proportions and mean ± z·SE for `FDR_hat`. These are *empirical simulation performance*, not guaranteed control.
  No post-hoc FDP bound (Simes) is used or implemented.

---

## 8. Key outputs

Under `$BEATAML_OUT` (default `./output`):

| Location | Files |
|---|---|
| `manifest/` | `patient_level_cohort_manifest.csv`, `sample_flow.csv`, `variant_class_decisions.csv`, `drug_repeat_pairs.csv`, `excluded_patients.csv`, `manifest_metadata.json` |
| `DISCOVERY/` | `mutation_drug_associations.csv`, `candidate_pairs.csv`, `stability_gate_audit.csv`, `stable_pairs.csv`, `adjusted_analysis_results.csv`, `freeze_audit.csv`, `freeze_manifest.json`, `frozen_candidate_list.csv`, `frozen_sensitivity_list.csv`, `supp_selection_flow.csv`, `final_master_association_table.csv`, `simulation_*.csv` |
| `VALIDATION/` | `validation_results.csv`, `validation_report.csv`, `validation_started.json`, `validation_completed.json`, `run_metadata.json` (and `*_sensitivity` variants) |
| `figures/` | paper figures (PNG + PDF) |

`direction_not_assessable` (zero/invalid effect) is reported as such and never counted as concordant.

---

## 9. Reproducibility and tests

* `python test_pipeline.py` — consistency tests on synthetic data (statistics vs SciPy, selection-rule definitions and
  denominators, freeze binding and tagging, validation isolation, FLT3 composite, end-to-end smoke test).
* `python compare_repro.py output_a output_b` — compares two output roots by *content* (the `frozen_at_utc` column is
  ignored, because it differs on every run).
* All random procedures use `RANDOM_SEED` from the config.

---

## 10. Interpretation notes

* Candidates are *stable under the checks performed* — not "most consistent evidence". Stability checks
  (bootstrap, split-half) are internal; they are neither independent replication nor a winner's-curse correction.
* "Not confirmed" in an under-powered Validation means *cannot tell*, not *rejected*. Stage 14 reports the minimum
  detectable effect and power.
* In synthetic tests the sequential rule can declare at least one candidate under a global null, because a shared
  Top-N plus internal stability is not multiplicity control. Treat the rule as a descriptive reporting filter; error
  control comes from FDR procedures, single-step min-p and the separate Validation, not from the rule itself.
* The mutation–drug link is made at the patient level and does not account for sample collection time
  (specimen-matched analyses are not part of the main pipeline).
* A few console messages inside the freeze-bound core (`common/` and stages `00`–`08c`) are still in Persian: translating
  string literals would change the program and therefore the freeze hashes. Comments and docstrings are in English.

---

## 11. License, data attribution and citation

* **Code:** MIT License (see `LICENSE`).
* **Data (not included):** BeatAML 2.0 data, Bottomly D. *et al.*, "biodev/beataml2.0_data: Release of BeatAML 2.0 data",
  Zenodo, v3.0, 2024, DOI [10.5281/zenodo.10654808](https://doi.org/10.5281/zenodo.10654808),
  licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Data were used unmodified as inputs;
  all derived tables are produced locally by this pipeline and are not redistributed here.
* **Please also cite** the BeatAML 2.0 manuscript (Bottomly *et al.*, 2022, *Cancer Cell*) when using the data.
