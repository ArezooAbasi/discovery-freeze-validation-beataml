# -*- coding: utf-8 -*-
"""
08b_freeze_candidates.py   (DISCOVERY only; v3 / Path B)
=========================================================================================
Main-rule gates (all computed for ALL candidates and logged in freeze_audit.csv, including rejected ones):
  G1_in_main            : fixed nominal screen: p < NOMINAL_P_MAX and n >= MAIN_MIN_N_PER_GROUP over the whole screen family (stage 02)
  G2_stable             : bootstrap + split-half (Stage 04)
  G3_adjustment_pass    : exploratory adjustment in ADJ_PASS_LABELS (robust_to_adjustment only; "no confounder candidate found" does not pass)
  frozen_candidate_list.csv   = G1 AND G2 AND G3   <- the only "main result" list; it may be empty (a null result is allowed).
  frozen_sensitivity_list.csv = remaining stable pairs (BH gate, Top-N, failed adjustment, ...) with sensitivity_reason; sensitivity only,
                                evaluated separately with --tag sensitivity; never merged with the main list.
  Exact definition of the sensitivity list (for Methods): "all stable pairs (G2) that are not in the main result";
  not all nominal pairs and not all BH-significant pairs.
None of these lists controls FDR. No FDP/Simes bound is reported (its mathematical validity has not been verified).
freeze_manifest.json: hashes of the lists and of every Discovery file that shaped the decision (including adjusted_analysis_results),
  the selection parameters, analysis_definition_sha256, patient manifest, source files, config and code (full hash + AST hash).
The freeze cannot be overwritten after VALIDATION has started (lock marker). The --force option was removed;
  revising a freeze requires archiving the complete Validation output and recording the reason, outside this script.
p_WY, effect_EB and local_fdr are exploratory; effect_EB is NOT a "selection-adjusted effect".
"""
import argparse
import datetime
import json
import uuid
import numpy as np
import pandas as pd
from common import config_official as cfg, data, screen, stats

KEY = ["gene", "drug"]
COLS = ["gene", "drug", "rank_biserial_effect", "p_value", "FDR_q", "p_WY", "effect_EB", "local_fdr",
        "n_mutated_with_AUC", "n_wt_with_AUC", "adjustment_interpretation", "adjustment_status", "in_main", "in_bh_gate",
        "effect_rule_ok", "sensitivity_reason", "frozen_at_utc"]


def _require_unique(name, df):
    dup = df[df.duplicated(KEY, keep=False)]
    if len(dup):
        raise SystemExit(f"[!] کلید gene-drug تکراری در {name} ({len(dup)} ردیف):\n{dup[KEY].head(10).to_string()}")


def main():
    ap = argparse.ArgumentParser(description="Freeze فهرست کاندیدها (فقط DISCOVERY)")
    ap.add_argument("--force", action="store_true", help=argparse.SUPPRESS)
    if ap.parse_args().force:
        raise SystemExit("[!] --force حذف شده است: بازنویسی freeze بعد از VALIDATION بدون آرشیو واقعی پذیرفته نیست.")
    cohort = "DISCOVERY"
    if data.validation_started():
        raise SystemExit("[!] VALIDATION شروع شده (marker یا نتیجه موجود است)؛ freeze بازنویسی نمی‌شود. "
                         "بازنگری فقط با آرشیو کامل خروجی‌های Validation، ثبت علت و freeze با شناسه‌ی جدید.")

    mm = cfg.MANIFEST_DIR / "manifest_metadata.json"
    source_sha = None
    if not data.is_synthetic():
        if not cfg.MANIFEST_FILE.exists():
            raise SystemExit(f"[!] patient manifest نیست: {cfg.MANIFEST_FILE}")
        if not mm.exists():
            raise SystemExit("[!] manifest_metadata.json is missing (hash منابع رسمی برای freeze لازم است)")
        source_sha = json.loads(mm.read_text()).get("source_sha256")
        if not source_sha:
            raise SystemExit("[!] source_sha256 در manifest_metadata.json ثبت نشده")

    audit = data.read(cohort, "stability_gate_audit.csv", hint="اول 04_export_stable_pairs.py")
    adj = data.read(cohort, "adjusted_analysis_results.csv", hint="اول 08_adjusted_analysis.py")
    assoc = data.read(cohort, "mutation_drug_associations.csv", hint="اول 01_association_screen.py")
    for nm, df in (("stability_gate_audit", audit), ("adjusted_analysis_results", adj), ("mutation_drug_associations", assoc)):
        _require_unique(nm, df)
    top_cols = [f"in_top{N}" for N in cfg.SENSITIVITY_TOP_N]

    mut, drug = data.load_cohort_matrices(cohort)
    D, G, genes, drugs, _ = screen.prepare(drug, mut)
    a = screen.screen_arrays(D, G)
    pw, _ = screen.westfall_young_minp(D, G, a["gi"], a["di"], a["p"], cfg.N_WY_PERMUTATIONS, cfg.RANDOM_SEED)
    z = stats.signed_z(a["rb"], a["p"])
    factor, lfdr, eb = stats.eb_shrink(z)
    if not eb.get("converged", False) and eb.get("n_finite", 0):
        print(f"[warn] EB shrinkage همگرا نشد (n_finite={eb.get('n_finite')})؛ effect_EB/local_fdr اکتشافی و غیرقابل اتکا")
    rec = pd.DataFrame({"gene": genes[a["gi"]], "drug": drugs[a["di"]], "p_recomputed": a["p"],
                        "p_WY": pw, "effect_EB": a["rb"] * factor, "local_fdr": lfdr})
    _require_unique("screen recomputed", rec)
    full = assoc[KEY + ["n_mutated_with_AUC", "n_wt_with_AUC", "rank_biserial_effect", "p_value", "FDR_q"]].merge(
        rec, on=KEY, how="left", validate="one_to_one")
    n_missing = int(full["p_recomputed"].isna().sum())
    chk = full.dropna(subset=["p_recomputed"])
    maxdiff = float((chk["p_value"] - chk["p_recomputed"]).abs().max()) if len(chk) else float("nan")
    if len(rec) != len(assoc) or n_missing or not (maxdiff < 1e-6):
        raise SystemExit(f"[!] ناسازگاری خانواده‌ی screen: recomputed={len(rec)} vs 01={len(assoc)} | "
                         f"pairs بدون p_recomputed={n_missing} | max|dp|={maxdiff}. 01 را با نسخه‌ی اصلاح‌شده دوباره اجرا کنید.")
    print(f"EB: pi0={eb['pi0']:.3f} tau2={eb['tau2']:.3f} converged={eb.get('converged')} | min p_WY={full['p_WY'].min():.4f}")
    data.write(full, cohort, "selection_adjusted_associations.csv")

    # ---------------- input consistency checks
    orphan = audit.merge(full[KEY], on=KEY, how="left", indicator=True)
    orphan = orphan[orphan["_merge"] == "left_only"]
    if len(orphan):
        raise SystemExit(f"[!] {len(orphan)} زوج در stability_gate_audit در خانواده‌ی screen نیستند:\n{orphan[KEY].head(10).to_string()}")

    # ---------------- gates for all candidates
    adj_cols = [c for c in ["unadjusted_beta", "adjusted_beta", "confounders_included", "pct_change_beta"] if c in adj.columns]
    adj_sel = adj[KEY + ["interpretation", "adjustment_status"] + adj_cols].rename(
        columns={"interpretation": "adjustment_interpretation"})
    g = audit.drop(columns=[c for c in adj_sel.columns if c in audit.columns and c not in KEY])
    g = g.merge(adj_sel, on=KEY, how="left", validate="one_to_one")
    for c in ["in_main", "in_bh_gate", "boot_ok", "split_ok", "effect_rule_ok"] + top_cols:
        g[c] = g[c].fillna(False).astype(bool)
    g["G1_in_main"] = g["in_main"]
    g["G2_stable"] = g["boot_ok"] & g["split_ok"]
    g["G3_adjustment_run"] = g["adjustment_interpretation"].notna()
    # Every pair with G1 AND G2 must have an adjustment record; a missing one is a Stage 08 execution defect, not a "rejection".
    miss_adj = g[g["G1_in_main"] & g["G2_stable"] & ~g["G3_adjustment_run"]]
    if len(miss_adj):
        raise SystemExit(f"[!] {len(miss_adj)} زوج G1∧G2 رکورد adjustment ندارند (Stage 08 ناقص):\n{miss_adj[KEY].head(10).to_string()}")
    g["G3_adjustment_pass"] = g["adjustment_interpretation"].isin(cfg.ADJ_PASS_LABELS)
    g["is_frozen_main"] = g["G1_in_main"] & g["G2_stable"] & g["G3_adjustment_pass"]
    g["first_failed_gate"] = np.select([~g["G1_in_main"], ~g["G2_stable"], ~g["G3_adjustment_pass"]],
                                       ["G1_not_nominal_main", "G2_not_stable", "G3_adjustment"], default="")

    def why(r):
        parts = [c.replace("in_", "legacy_") for c in top_cols if r[c]]
        if r["in_bh_gate"]:
            parts.append("bh_gate")
        if r["G1_in_main"] and not r["G3_adjustment_pass"]:
            parts.append(f"main_failed_adjustment:{r['adjustment_status']}")
        if not r["G1_in_main"]:
            parts.append("not_in_main_nominal_rule")
        return ";".join(parts)

    g["sensitivity_reason"] = g.apply(why, axis=1) if len(g) else pd.Series(dtype=str)
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    g["frozen_at_utc"] = now
    # `full` is the authoritative source of p/n/effect; same-named columns in the audit are dropped to avoid _x/_y suffixes
    overlap = [c for c in full.columns if c in g.columns and c not in KEY]
    g = g.drop(columns=overlap).merge(full.drop(columns=["p_recomputed"]), on=KEY, how="left", validate="one_to_one")
    adj_path = data.path(cohort, "adjusted_analysis_results.csv")
    g["G1_threshold_nominal_p_max"] = cfg.NOMINAL_P_MAX
    g["G1_threshold_min_n_per_group"] = cfg.MAIN_MIN_N_PER_GROUP
    g["adjustment_file_sha256"] = data.sha256_file(adj_path)
    frozen = g[g["is_frozen_main"]].copy()
    frozen["sensitivity_reason"] = ""
    sens = g[g["G2_stable"] & ~g["is_frozen_main"]].copy()           # only stable pairs enter the sensitivity evaluation
    assert not (set(zip(frozen.gene, frozen.drug)) & set(zip(sens.gene, sens.drug)))
    _require_unique("frozen", frozen)
    _require_unique("sensitivity", sens)

    print(f"\nکاندیدها: {len(g)} | G1(nominal p<{cfg.NOMINAL_P_MAX}): {int(g.G1_in_main.sum())} | G1∧G2: {int((g.G1_in_main & g.G2_stable).sum())}"
          f" | frozen main (G1∧G2∧G3): {len(frozen)} | sensitivity-only: {len(sens)}")
    if g["first_failed_gate"].ne("").any():
        print(g.loc[g.G1_in_main & g.first_failed_gate.ne(""), "first_failed_gate"].value_counts().to_string())
    if len(frozen) == 0:
        print("  >>> نتیجه‌ی اصلی: صفر کاندیدا (مجاز و قابل گزارش؛ فهرست خالی frozen و hash می‌شود).")
    print("  (هیچ کران FDP/FDR گزارش نمی‌شود؛ فهرست = اولویت‌بندی اکتشافی.)")

    p1 = data.write(frozen.reindex(columns=COLS), cohort, "frozen_candidate_list.csv")
    p2 = data.write(sens.reindex(columns=COLS), cohort, "frozen_sensitivity_list.csv")
    audit_cols = (KEY + ["in_main", "in_bh_gate"] + top_cols +
                  ["G1_in_main", "boot_ok", "split_ok", "G2_stable", "G3_adjustment_run",
                   "adjustment_interpretation", "adjustment_status", "G3_adjustment_pass", "effect_rule_ok", "is_frozen_main",
                   "first_failed_gate", "sensitivity_reason", "p_value", "FDR_q", "p_WY", "rank_biserial_effect",
                   "n_mutated_with_AUC", "n_wt_with_AUC", "observed_effect", "bootstrap_CI_low", "bootstrap_CI_high",
                   "CI_crosses_zero", "n_split_half_valid", "pct_split_half_match_full",
                   "G1_threshold_nominal_p_max", "G1_threshold_min_n_per_group"] + adj_cols + ["adjustment_file_sha256"])
    p3 = data.write(g.reindex(columns=audit_cols), cohort, "freeze_audit.csv")

    manifest = {"freeze_id": uuid.uuid4().hex, "frozen_at_utc": now,
                "n_candidates": len(g), "n_frozen_main": len(frozen), "n_sensitivity": len(sens),
                "frozen_candidate_list_sha256": data.sha256_file(p1), "frozen_sensitivity_list_sha256": data.sha256_file(p2),
                "freeze_audit_sha256": data.sha256_file(p3),
                "stable_pairs_sha256": data.sha256_file(data.path(cohort, "stable_pairs.csv")),
                "stability_gate_audit_sha256": data.sha256_file(data.path(cohort, "stability_gate_audit.csv")),
                "associations_sha256": data.sha256_file(data.path(cohort, "mutation_drug_associations.csv")),
                "adjusted_analysis_results_sha256": data.sha256_file(adj_path),
                "config_official_sha256": data.sha256_file(cfg.__file__),
                "selection_params": data.freeze_params(), "selection_params_sha256": data.params_sha256(),
                "analysis_definition": data.analysis_definition(),
                "analysis_definition_sha256": data.analysis_definition_sha256(),
                "code_sha256": data.code_hashes(), "code_ast_sha256": data.code_ast_hashes(),
                "patient_manifest_sha256": data.sha256_file(cfg.MANIFEST_FILE) if not data.is_synthetic() else None,
                "source_sha256": source_sha}
    data.path(cohort, "freeze_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print("freeze_id:", manifest["freeze_id"])
    print("frozen_candidate_list_sha256:", manifest["frozen_candidate_list_sha256"])
    print("selection_params_sha256:", manifest["selection_params_sha256"])
    print("analysis_definition_sha256:", manifest["analysis_definition_sha256"])
    print("\nحالا:")
    print("  python 01_association_screen.py --cohort VALIDATION --frozen", p1)
    print("  python 01_association_screen.py --cohort VALIDATION --frozen", p2, "--tag sensitivity")


if __name__ == "__main__":
    main()

