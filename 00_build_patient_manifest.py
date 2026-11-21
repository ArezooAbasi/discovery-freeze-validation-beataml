# -*- coding: utf-8 -*-
"""
00_build_patient_manifest.py
Reads the official source files only. Runs no statistical tests.
Outputs in output/manifest/: patient manifest, sample flow, variant-class table, drug repeat-record report.
"""
import json, hashlib
import pandas as pd
from common import config_official as cfg, io_official as io, cohort

def main():
    cfg.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    src = io.load_sources()
    m = cohort.build_manifest(src)
    assert not (m.in_DISCOVERY & m.in_VALIDATION).any()

    m.to_csv(cfg.MANIFEST_FILE, index=False, encoding="utf-8-sig")
    flow = cohort.sample_flow(m, src)
    flow.to_csv(cfg.MANIFEST_DIR / "sample_flow.csv", index=False, encoding="utf-8-sig")
    io.variant_class_table(src["mutations"]).to_csv(
        cfg.MANIFEST_DIR / "variant_class_decisions.csv", index=False, encoding="utf-8-sig")
    io.repeat_report(io.usable_drug_records(src["drug"])).to_csv(
        cfg.MANIFEST_DIR / "drug_repeat_pairs.csv", index=False, encoding="utf-8-sig")
    m.loc[m.exclusion_reason != "", ["patient_id", "wave_group", "exclusion_reason"]].to_csv(
        cfg.MANIFEST_DIR / "excluded_patients.csv", index=False, encoding="utf-8-sig")

    meta = {"manifest_sha256": hashlib.sha256(cfg.MANIFEST_FILE.read_bytes()).hexdigest(),
            "source_files": {k: str(v.name) for k, v in cfg.FILES.items()},
            "source_sha256": {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in cfg.FILES.items()},
            "config": {k: (sorted(v) if isinstance(v, set) else str(v)) for k, v in vars(cfg).items()
                       if k.isupper() and k not in ("FILES", "OFFICIAL_DIR", "OUTPUT_ROOT",
                                                    "MANIFEST_DIR", "MANIFEST_FILE", "PROJECT_DIR")}}
    (cfg.MANIFEST_DIR / "manifest_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(flow.to_string(index=False))
    print("\nwave_group counts (all patients):")
    print(m["wave_group"].value_counts().to_string())
    print("\nSaved to", cfg.MANIFEST_DIR)

if __name__ == "__main__":
    main()
