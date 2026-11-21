# -*- coding: utf-8 -*-
"""
13_master_table.py   (v3)
=========================================================================================
Master table (the single source of the paper's figures/tables). No statistic is recomputed; it only joins the outputs of the previous stages.
main and sensitivity are NEVER merged:
  frozen_main / validation_* / tier            : main result (G1 nominal AND stable AND robust_to_adjustment); from validation_report.csv
  frozen_sensitivity / sens_validation_* / tier: sensitivity only; from validation_report_sensitivity.csv (prefix sens_, and tier prefixed with sensitivity_)
  adjustment_pass_main / adjustment_pass_permissive : two adjustment definitions, independent of each other
  in_main / in_bh_gate / in_top{N}             : flags of the different rules, kept separate
  frozen (=frozen_main)                        : compatibility with stage 15
  python 13_master_table.py        (DISCOVERY only)
"""
import pandas as pd
from common import config_official as cfg, data

KEY = ["gene", "drug"]


def _label(v):
    if pd.isna(v):
        return "pending_validation"
    v = str(v)
    if v == "validated":
        return "validated_in_cross_wave_replication"
    if v == "direction_not_assessable":
        return "direction_not_assessable_in_validation"
    if v == "not_evaluable":
        return "not_evaluable_in_validation"          # not "rejected"; there was not enough data
    if v.startswith("contradicted"):
        return "contradicted_in_cross_wave_replication"
    if v.endswith("underpowered"):
        return "inconclusive_underpowered"            # "not confirmed" != "rejected"
    if v.startswith("concordant"):
        return "concordant_not_significant_adequately_powered"
    return "discordant_not_significant_adequately_powered"


def tier(r):
    if r["frozen_main"]:
        v = r.get("validation_interpretation")
        return "frozen_pending_validation" if pd.isna(v) else _label(v)
    if r["frozen_sensitivity"]:
        return "sensitivity_" + _label(r.get("sens_validation_interpretation"))
    return "not_frozen"


_VKEEP = ["gene", "drug", "n_mut_val", "n_wt_val", "validation_effect", "rank_biserial_ci_low", "rank_biserial_ci_high",
          "validation_p", "validation_FDR_q", "power_at_shrunk_effect", "interpretation"]


def _vcols(vr, prefix):
    v = vr[[k for k in _VKEEP if k in vr.columns]].rename(columns={"interpretation": "validation_interpretation"})
    return v.rename(columns={c: prefix + c for c in v.columns if c not in KEY}).drop_duplicates(KEY)


def assemble(st, assoc, boot, adj, sa, fz, fs, vr_main, vr_sens, names):
    flag_cols = [c for c in st.columns if c.startswith("in_") or c in ("boot_ok", "split_ok", "effect_rule_ok", "conservative_effect", "ci_width")]
    cols = ["gene", "drug", "n_mutated_with_AUC", "n_wt_with_AUC", "median_AUC_mutated", "median_AUC_wt",
            "delta_median_AUC", "rank_biserial_effect", "p_value", "FDR_q"]
    m = st[["gene", "drug"] + flag_cols].merge(assoc[[x for x in cols if x in assoc.columns]], on=KEY, how="left")
    m = m.merge(boot[["gene", "drug", "bootstrap_median_effect", "bootstrap_CI_low", "bootstrap_CI_high", "pct_split_half_match_full"]],
                on=KEY, how="left")
    m = m.merge(adj[["gene", "drug", "confounders_included", "adjusted_beta", "adjusted_p", "interpretation", "adjustment_status"]]
                .rename(columns={"interpretation": "adjustment_interpretation"}), on=KEY, how="left")
    m = m.merge(sa[["gene", "drug", "p_WY", "effect_EB", "local_fdr"]], on=KEY, how="left")
    for c in ("in_main", "in_bh_gate"):
        if c not in m:
            m[c] = False
    m["adjustment_pass_main"] = m["adjustment_interpretation"].isin(cfg.ADJ_PASS_LABELS)
    m["adjustment_pass_permissive"] = m["adjustment_interpretation"].isin(cfg.ADJ_PASS_LABELS_PERMISSIVE)
    fm = set(zip(fz.gene, fz.drug)) if fz is not None and len(fz) else set()
    fsn = set(zip(fs.gene, fs.drug)) if fs is not None and len(fs) else set()
    m["frozen_main"] = [(g, d) in fm for g, d in zip(m.gene, m.drug)]
    m["frozen_sensitivity"] = [(g, d) in fsn for g, d in zip(m.gene, m.drug)]
    m["frozen"] = m["frozen_main"]
    m.insert(2, "drug_name", m["drug"].map(lambda x: names.get(x, x)))
    # validation: two separate column sets (main without prefix; sensitivity with prefix sens_)
    if vr_main is not None and len(vr_main):
        m = m.merge(_vcols(vr_main, ""), on=KEY, how="left")
        m.loc[~m["frozen_main"], [c for c in m.columns if c in _vcols(vr_main, "").columns and c not in KEY]] = pd.NA
    else:
        m["validation_interpretation"] = pd.NA
    if vr_sens is not None and len(vr_sens):
        m = m.merge(_vcols(vr_sens, "sens_"), on=KEY, how="left")
        sc = [c for c in m.columns if c.startswith("sens_")]
        m.loc[~m["frozen_sensitivity"], sc] = pd.NA
    else:
        m["sens_validation_interpretation"] = pd.NA
    m["tier"] = m.apply(tier, axis=1)
    return m.sort_values("p_value").reset_index(drop=True)


def main():
    c = "DISCOVERY"
    m = assemble(
        data.read(c, "stable_pairs.csv"), data.read(c, "mutation_drug_associations.csv"),
        data.read(c, "bootstrap_robustness_results.csv"), data.read(c, "adjusted_analysis_results.csv"),
        data.read(c, "selection_adjusted_associations.csv", hint="run 08b_freeze_candidates.py first"),
        data.read(c, "frozen_candidate_list.csv"), data.read(c, "frozen_sensitivity_list.csv", required=False),
        data.read("VALIDATION", "validation_report.csv", required=False),
        data.read("VALIDATION", "validation_report_sensitivity.csv", required=False), data.drug_names())
    data.write(m, c, "final_master_association_table.csv")
    show = ["gene", "drug_name", "rank_biserial_effect", "p_value", "FDR_q", "p_WY", "adjustment_status", "tier"]
    print("MAIN (frozen):")
    print(m.loc[m.frozen_main, show].to_string(index=False) if m.frozen_main.any() else "  (zero candidates - a null result is allowed)")
    print(f"\nsensitivity-only frozen: {int(m.frozen_sensitivity.sum())} (tier prefixed with sensitivity_) - do not put these in the main table of the paper.")


if __name__ == "__main__":
    main()
