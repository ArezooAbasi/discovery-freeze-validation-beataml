# -*- coding: utf-8 -*-
"""
compare_repro.py -- compares two output roots (e.g. output_v3 and output_repro) by content, not by file hash.
(The hash of the frozen list differs on every run because of the frozen_at_utc column; this script ignores that column.)

  python compare_repro.py output_v3 output_repro
"""
import json
import sys
from pathlib import Path
import pandas as pd

A, B = Path(sys.argv[1]), Path(sys.argv[2])
# (relative path, columns to ignore)
FILES = [
    ("manifest/patient_level_cohort_manifest.csv", ()),
    ("DISCOVERY/mutation_drug_associations.csv", ()),
    ("DISCOVERY/bootstrap_robustness_results.csv", ()),
    ("DISCOVERY/stable_pairs.csv", ()),
    ("DISCOVERY/adjusted_analysis_results.csv", ()),
    ("DISCOVERY/selection_adjusted_associations.csv", ()),
    ("DISCOVERY/frozen_candidate_list.csv", ("frozen_at_utc",)),
    ("DISCOVERY/frozen_sensitivity_list.csv", ("frozen_at_utc",)),
    ("VALIDATION/validation_results.csv", ()),
    ("VALIDATION/validation_results_sensitivity.csv", ()),
    ("VALIDATION/validation_report.csv", ("frozen_at_utc",)),
    ("VALIDATION/validation_report_sensitivity.csv", ("frozen_at_utc",)),
]
JSON_KEYS = ["selection_params_sha256", "analysis_definition_sha256", "patient_manifest_sha256", "source_sha256",
             "n_candidates", "n_frozen_main", "n_sensitivity"]


def read(p):
    return pd.read_csv(p, keep_default_na=False, na_values=[""], encoding="utf-8-sig")


bad = 0
for rel, drop in FILES:
    pa, pb = A / rel, B / rel
    if not pa.exists() or not pb.exists():
        print(f"[skip] {rel}: {'A' if not pa.exists() else 'B'} is missing")
        continue
    a, b = read(pa).drop(columns=list(drop), errors="ignore"), read(pb).drop(columns=list(drop), errors="ignore")
    if list(a.columns) != list(b.columns):
        print(f"[DIFF] {rel}: columns differ: {sorted(set(a.columns) ^ set(b.columns))}")
        bad += 1
        continue
    key = [c for c in ("gene", "drug", "patient_id") if c in a.columns]
    if key:
        a, b = a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-12, check_dtype=False)
        print(f"[ok]   {rel}: {len(a)} identical rows")
    except AssertionError as e:
        print(f"[DIFF] {rel}: {str(e).splitlines()[0]} ... ({len(a)} vs {len(b)} rows)")
        bad += 1
ja, jb = A / "DISCOVERY/freeze_manifest.json", B / "DISCOVERY/freeze_manifest.json"
if ja.exists() and jb.exists():
    ma, mb = json.loads(ja.read_text()), json.loads(jb.read_text())
    for k in JSON_KEYS:
        same = ma.get(k) == mb.get(k)
        print(f"[{'ok' if same else 'DIFF'}]   freeze_manifest.{k}")
        bad += 0 if same else 1
print("\nResult:", "✅ all identical (reproduction successful)" if bad == 0 else f"❌ {bad} item(s) differ")
sys.exit(1 if bad else 0)
