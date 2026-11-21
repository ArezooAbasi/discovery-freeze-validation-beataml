# -*- coding: utf-8 -*-
"""
01_association_screen.py
Stage 1 with the patient as the unit of analysis; input = the official manifest.

  python 01_association_screen.py --cohort DISCOVERY
  python 01_association_screen.py --cohort FULL
  python 01_association_screen.py --cohort VALIDATION --frozen output/DISCOVERY/frozen_candidate_list.csv

Rules:
  * DISCOVERY / FULL : full gene x drug screen. (DISCOVERY is locked once VALIDATION has started.)
  * VALIDATION       : only the pairs of the frozen list are evaluated (no re-selection); cannot run without --frozen.
                       An empty frozen list (a null result) is allowed and yields a valid output with all columns.
                       A lock marker (validation_started*.json) is written before any computation; re-running is impossible.
  * FDR: a single implementation (BH) for the main result; BY is a sensitivity column only.
  * A zero in the matrix means "no eligible variant reported", NOT a confirmed wild type.
"""
import argparse, json, hashlib, datetime
import numpy as np
import pandas as pd
from common import config_official as cfg, io_official as io, cohort, stats, data

VALIDATION_COLS = ["gene", "drug", "status", "n_patients_profiled", "n_mutated_total", "n_wt_total",
                   "n_mutated_with_AUC", "n_wt_with_AUC", "median_AUC_mutated", "median_AUC_wt", "delta_median_AUC",
                   "rank_biserial_effect", "p_value", "rank_biserial_ci_low", "rank_biserial_ci_high",
                   "discovery_rank_biserial", "direction_concordant"]


def eligible_genes_and_drugs(mut, drug):
    qd = drug.notna().sum()
    qd = qd[qd >= cfg.MIN_DRUG_SAMPLES].index.tolist()
    qg = mut.sum(axis=0)
    qg = qg[qg >= cfg.MIN_MUTATED].index.tolist()
    return qg, qd


def evaluate_pair(gene, drug_name, mut, drug):
    s = drug[drug_name].dropna()
    prof = mut.index
    mut_ids = mut.index[mut[gene] == 1]
    wt_ids = mut.index[mut[gene] == 0]
    ma = s[s.index.isin(mut_ids)].values
    wa = s[s.index.isin(wt_ids)].values
    base = {"gene": gene, "drug": drug_name,
            "n_patients_profiled": len(prof),
            "n_mutated_total": len(mut_ids), "n_wt_total": len(wt_ids),
            "n_mutated_with_AUC": len(ma), "n_wt_with_AUC": len(wa)}
    return base, ma, wa


def align_matrices(mut, drug):
    """Check uniqueness and alignment; the number of rows dropped from each matrix is returned and logged (silently changing the analysis set is forbidden)."""
    for nm, df in (("mut", mut), ("drug", drug)):
        if not df.index.is_unique:
            raise SystemExit(f"[!] index ماتریس {nm} یکتا نیست (بیمار تکراری)")
        if not df.columns.is_unique:
            raise SystemExit(f"[!] ستون‌های ماتریس {nm} یکتا نیستند")
    common_ids = mut.index.intersection(drug.index)
    info = {"n_mut_matrix": len(mut), "n_drug_matrix": len(drug), "n_common": len(common_ids),
            "n_dropped_mut_only": len(mut.index.difference(drug.index)),
            "n_dropped_drug_only": len(drug.index.difference(mut.index))}
    print(f"profiled: {info['n_mut_matrix']} | with drug data: {info['n_drug_matrix']} | both: {info['n_common']} "
          f"| dropped mut-only: {info['n_dropped_mut_only']} | dropped drug-only: {info['n_dropped_drug_only']}")
    return mut.loc[common_ids], drug.loc[common_ids], info


def direction_concordant(discovery, validation):
    """Concordance only when both effects are finite and non-zero; otherwise NaN (not interpretable)."""
    try:
        d, v = float(discovery), float(validation)
    except (TypeError, ValueError):
        return np.nan
    if np.isfinite(d) and np.isfinite(v) and d != 0 and v != 0:
        return bool(np.sign(d) == np.sign(v))
    return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["FULL", "DISCOVERY", "VALIDATION"])
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--tag", default=None, choices=["sensitivity"],
                    help="فقط VALIDATION: ارزیابی فهرست sensitivity با خروجی جدا (هرگز با فهرست اصلی ادغام نمی‌شود)")
    a = ap.parse_args()
    if a.tag and a.cohort != "VALIDATION":
        raise SystemExit("--tag فقط برای --cohort VALIDATION معنی دارد.")
    if a.cohort == "DISCOVERY" and data.validation_started():
        raise SystemExit("[!] VALIDATION شروع شده؛ اجرای مجدد DISCOVERY (بازنویسی ورودی‌های freeze) مجاز نیست.")
    binding = marker = None
    cfg.require_thresholds()

    if a.cohort == "VALIDATION":
        if not a.frozen:
            raise SystemExit("VALIDATION بدون --frozen candidate list مجاز نیست (جلوگیری از انتخاب مجدد).")
        binding = data.verify_freeze(a.frozen, a.tag)     # hard mismatch => stop
        marker = data.begin_validation(a.frozen, a.cohort, a.tag, binding)   # lock before any computation

    manifest = pd.read_csv(cfg.MANIFEST_FILE, dtype={"patient_id": str})
    patients = cohort.patients_of(manifest, a.cohort)
    print(f"[{a.cohort}] patients in manifest: {len(patients)}")

    src = io.load_sources()
    mut = io.build_patient_mutation_matrix(src, patients)
    drug = io.build_patient_drug_matrix(src, patients)
    mut, drug, align = align_matrices(mut, drug)     # gene/drug eligibility on the same set used by 08b and 16

    out_dir = cfg.cohort_dir(a.cohort)
    rows = []
    family = None

    if a.cohort == "VALIDATION":
        frozen = pd.read_csv(a.frozen)
        if not {"gene", "drug"} <= set(frozen.columns):
            raise SystemExit("[!] فایل frozen باید ستون‌های gene و drug داشته باشد.")
        if frozen.duplicated(["gene", "drug"]).any():
            raise SystemExit("[!] جفت gene-drug تکراری در فهرست frozen.")
        if len(frozen) == 0:
            print("[info] فهرست frozen خالی است (نتیجه‌ی null مجاز)؛ خروجی Validation خالی ولی معتبر نوشته می‌شود.")
        for r in frozen.itertuples():
            rec = {"gene": r.gene, "drug": r.drug}
            if r.gene not in mut.columns or r.drug not in drug.columns:
                rec.update(status="not_evaluable_gene_or_drug_absent")
                rows.append(rec); continue
            base, ma, wa = evaluate_pair(r.gene, r.drug, mut, drug)
            rec.update(base)
            if len(ma) < cfg.MIN_MUTATED_WITH_AUC or len(wa) < cfg.MIN_WT_WITH_AUC:
                rec.update(status="not_evaluable_low_n"); rows.append(rec); continue
            rec.update(stats.test_pair(ma, wa), status="evaluated")
            lo, hi = stats.bootstrap_ci(ma, wa, cfg.BOOTSTRAP_B, cfg.RANDOM_SEED)
            rec.update(rank_biserial_ci_low=lo, rank_biserial_ci_high=hi)
            if "rank_biserial_effect" in frozen.columns:
                d = getattr(r, "rank_biserial_effect")
                rec["discovery_rank_biserial"] = d
                rec["direction_concordant"] = direction_concordant(d, rec["rank_biserial_effect"])
            rows.append(rec)
        res = pd.DataFrame(rows)
        fdr_cols = [c for c, _ in [("FDR_q", cfg.FDR_METHOD), *cfg.SENSITIVITY_FDR_METHODS.items()]]
        res = res.reindex(columns=VALIDATION_COLS + [c for c in res.columns if c not in VALIDATION_COLS])
        ev = res["status"].eq("evaluated")
        for col, meth in [("FDR_q", cfg.FDR_METHOD), *cfg.SENSITIVITY_FDR_METHODS.items()]:
            res[col] = np.nan
            if ev.sum():
                res.loc[ev, col] = stats.fdr(res.loc[ev, "p_value"], meth)
        family = {"n_frozen": int(len(frozen)), "n_evaluated": int(ev.sum()),
                  "n_not_evaluable_gene_or_drug_absent": int(res["status"].eq("not_evaluable_gene_or_drug_absent").sum()),
                  "n_not_evaluable_low_n": int(res["status"].eq("not_evaluable_low_n").sum()),
                  "fdr_m": int(ev.sum())}
        res["fdr_family"] = (f"frozen candidates evaluated in VALIDATION (m={family['n_evaluated']}; frozen={family['n_frozen']}; "
                             f"absent={family['n_not_evaluable_gene_or_drug_absent']}; low_n={family['n_not_evaluable_low_n']})")
        print("family:", family)
        name = data.tagged("validation_results.csv", a.tag)
    else:
        qg, qd = eligible_genes_and_drugs(mut, drug)
        print(f"genes>= {cfg.MIN_MUTATED}: {len(qg)} | drugs>= {cfg.MIN_DRUG_SAMPLES}: {len(qd)}")
        skipped = 0
        for g in qg:
            for d in qd:
                base, ma, wa = evaluate_pair(g, d, mut, drug)
                if len(ma) < cfg.MIN_MUTATED_WITH_AUC or len(wa) < cfg.MIN_WT_WITH_AUC:
                    skipped += 1; continue
                base.update(stats.test_pair(ma, wa)); rows.append(base)
        res = pd.DataFrame(rows)
        if res.empty:
            raise SystemExit("هیچ جفت واجد شرایطی وجود ندارد؛ آستانه‌ها را بازبینی کنید.")
        res["FDR_q"] = stats.fdr(res["p_value"], cfg.FDR_METHOD)
        for col, meth in cfg.SENSITIVITY_FDR_METHODS.items():
            res[col] = stats.fdr(res["p_value"], meth)
        res["fdr_family"] = f"all evaluable pairs in {a.cohort} (m={len(res)})"
        res = res.sort_values("p_value")
        name = "mutation_drug_associations.csv"
        print(f"pairs tested: {len(res)} | skipped low-n: {skipped} | "
              f"FDR<0.05: {(res.FDR_q < 0.05).sum()} | FDR<0.10: {(res.FDR_q < 0.10).sum()}")

    res.to_csv(out_dir / name, index=False, encoding="utf-8-sig")
    meta = {"cohort": a.cohort, "n_patients_manifest": len(patients), "n_profiled": len(mut),
            "n_with_drug": len(drug), "alignment": align,
            "run_time": datetime.datetime.now().isoformat(timespec="seconds"),
            "manifest_sha256": data.sha256_file(cfg.MANIFEST_FILE),
            "frozen_list_sha256": data.sha256_file(a.frozen) if a.frozen else None,
            "tag": a.tag, "freeze_binding": binding, "validation_run_id": marker["run_id"] if marker else None,
            "validation_family": family,
            "selection_params_sha256": data.params_sha256(),
            "analysis_definition_sha256": data.analysis_definition_sha256(),
            "drug_repeat_policy": cfg.DRUG_REPEAT_POLICY,
            "thresholds": {k: getattr(cfg, k) for k in
                           ("MIN_DRUG_SAMPLES", "MIN_MUTATED", "MIN_MUTATED_WITH_AUC", "MIN_WT_WITH_AUC")}}
    (out_dir / data.tagged("run_metadata.json", a.tag)).write_text(json.dumps(meta, indent=2, default=str))
    print("[saved]", out_dir / name)
    if marker:
        print("[completed]", data.complete_validation(marker, out_dir / name))


if __name__ == "__main__":
    main()
