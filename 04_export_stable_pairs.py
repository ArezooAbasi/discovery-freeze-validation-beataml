# -*- coding: utf-8 -*-
"""
04_export_stable_pairs.py   (v3)
=========================================================================================
The only place where "stable under the checks performed" is defined (common/selection.is_stable):
   (a) bootstrap  : the 95% CI excludes 0 and has the same sign as the observed effect
   (b) split-half : both halves have the same sign as the full-sample effect in >= SPLIT_HALF_DIRECTION_MIN % of ALL splits
"Stable" does not mean "more credible evidence"; it only means the candidate did not fail the predefined checks.

Outputs:
  stability_gate_audit.csv     : ALL candidates with every individual gate (including rejected ones) -> input of freeze_audit in 08b
  stable_pairs.csv             : stable pairs only + in_main / in_bh_gate / in_top{N} columns (independent of each other) + effect_rule_ok (sensitivity)
  candidates_failed_stability.csv
"""
import pandas as pd
from common import config_official as cfg, data, selection

KEY = ["gene", "drug"]


def main():
    cohort = data.parse_cohort("export stable pairs")
    boot = data.read(cohort, "bootstrap_robustness_results.csv", hint="اول 03_stability_checks.py")
    cands = data.read(cohort, "candidate_pairs.csv", hint="اول 02_threshold_robustness.py")
    top_cols = [f"in_top{N}" for N in cfg.SENSITIVITY_TOP_N]
    flag_cols = ["in_main", "in_bh_gate"] + top_cols
    gate_cols = ["boot_ok", "split_ok", "boot_direction_ok", "effect_rule_ok", "conservative_effect", "ci_width"]
    audit_cols = KEY + flag_cols + gate_cols + ["n_mutated", "n_wt", "observed_effect", "bootstrap_CI_low", "bootstrap_CI_high",
                                                "pct_split_half_match_full", "selection_reason"]
    stable_cols = KEY + ["selection_reason"] + flag_cols + ["boot_ok", "split_ok", "effect_rule_ok", "conservative_effect", "ci_width"]
    if boot.empty:
        for nm, cl in (("stability_gate_audit.csv", audit_cols), ("stable_pairs.csv", stable_cols), ("candidates_failed_stability.csv", audit_cols)):
            data.write(pd.DataFrame(columns=cl), cohort, nm)
        print("[i] هیچ کاندیدی نیست؛ stable_pairs خالی (نتیجه‌ی null).")
        return
    df = boot.merge(cands, on=KEY, how="left")

    def gates(r):
        b, s = selection.is_stable(r)
        return pd.Series({"boot_ok": b, "split_ok": s, "boot_direction_ok": selection.direction_ok(r),
                          "effect_rule_ok": selection.passes_effect_precision(r, r["n_mutated"], r["n_wt"]),
                          "conservative_effect": selection.conservative_effect(r), "ci_width": selection.ci_width(r)})
    df = pd.concat([df, df.apply(gates, axis=1)], axis=1)
    df["stable"] = df["boot_ok"].astype(bool) & df["split_ok"].astype(bool)
    stable, failed = df[df["stable"]], df[~df["stable"]]
    data.write(df[audit_cols], cohort, "stability_gate_audit.csv")
    data.write(failed[audit_cols], cohort, "candidates_failed_stability.csv")
    out = stable.copy()
    out["selection_reason"] = "stable_under_predefined_checks"
    data.write(out[stable_cols], cohort, "stable_pairs.csv")
    print(f"candidates: {len(df)} | stable: {len(stable)} | failed: {len(failed)}")
    print(f"stable ∧ in_main (نتیجه‌ی اصلی پیش از adjustment): {int(stable['in_main'].astype(bool).sum())}")
    if len(failed):
        print(failed[KEY + ["boot_ok", "split_ok", "pct_split_half_match_full"]].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
