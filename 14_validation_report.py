# -*- coding: utf-8 -*-
"""
14_validation_report.py   (v3)
=========================================================================================
Cross-wave validation report (W12 -> W34) with an honest power assessment. After running:
    python 01_association_screen.py --cohort VALIDATION --frozen output/DISCOVERY/frozen_candidate_list.csv
    python 14_validation_report.py
and for the sensitivity list (separately, separate files):
    python 01_association_screen.py --cohort VALIDATION --frozen output/DISCOVERY/frozen_sensitivity_list.csv --tag sensitivity
    python 14_validation_report.py --tag sensitivity

Binding to Discovery: verify_freeze (hash of the frozen list + selection parameters + manifest/sources) before every report; mismatch => stop.
For every frozen candidate: effect and CI, direction concordance, q (BH over m = evaluable candidates), MDE, power (observed / EB-shrunk effect),
interpretation label. "Not confirmed" with low power means "cannot tell", not "rejected". If the frozen list is empty, an empty (null) report is written.
v3.1: explicit status columns (flag_*) and not_evaluable_reason were added for every frozen pair; the statistical result and interpretation are unchanged.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from common import config_official as cfg, data, stats


def interpret(r):
    if r["status"] != "evaluated":
        return "not_evaluable"
    if pd.isna(r["direction_concordant"]):          # zero/invalid effect in one of the two cohorts: concordance is not assessable (bool(NaN) is True; it must not count as concordant)
        return "direction_not_assessable"
    conc, sig = bool(r["direction_concordant"]), r["FDR_q"] < cfg.VALIDATION_FDR
    if conc and sig:
        return "validated"
    if (not conc) and r["p_value"] < 0.05:
        return "contradicted_opposite_direction"
    pw = r["power_at_shrunk_effect"]
    tag = "underpowered" if (not np.isfinite(pw) or pw < cfg.POWER_TARGET) else "adequately_powered"
    return f"{'concordant' if conc else 'discordant'}_not_significant_{tag}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, choices=["sensitivity"])
    tag = ap.parse_args().tag
    fname = "frozen_sensitivity_list.csv" if tag == "sensitivity" else "frozen_candidate_list.csv"
    fz = data.read("DISCOVERY", fname, hint="run 08b_freeze_candidates.py first")
    rep_name, sum_name = data.tagged("validation_report.csv", tag), data.tagged("validation_summary.txt", tag)
    if fz.empty:
        print("[i] The frozen list is empty; nothing to validate (null result in Discovery).")
        data.write(pd.DataFrame(columns=["gene", "drug", "interpretation"]), "VALIDATION", rep_name)
        return
    data.verify_freeze(data.path("DISCOVERY", fname), tag)
    val = data.read("VALIDATION", data.tagged("validation_results.csv", tag),
                    hint="run stage 01 with --cohort VALIDATION --frozen ... first.")
    meta_p = data.path("VALIDATION", data.tagged("run_metadata.json", tag))
    if meta_p.exists():
        import json
        md = json.loads(meta_p.read_text())
        if md.get("frozen_list_sha256") != data.sha256_file(data.path("DISCOVERY", fname)) or \
                md.get("selection_params_sha256") != data.params_sha256():
            raise SystemExit("[!] VALIDATION was not run with the current Discovery list/parameters; the result is not valid.")
    fz = fz.rename(columns={"rank_biserial_effect": "discovery_effect", "p_value": "discovery_p", "FDR_q": "discovery_FDR_q",
                            "n_mutated_with_AUC": "n_mut_disc", "n_wt_with_AUC": "n_wt_disc"})
    v = val.rename(columns={"rank_biserial_effect": "validation_effect", "p_value": "validation_p",
                            "FDR_q": "validation_FDR_q", "n_mutated_with_AUC": "n_mut_val", "n_wt_with_AUC": "n_wt_val"})
    keep = [c for c in ["gene", "drug", "status", "n_mut_val", "n_wt_val", "validation_effect", "validation_p",
                        "rank_biserial_ci_low", "rank_biserial_ci_high", "direction_concordant", "validation_FDR_q"] if c in v.columns]
    rep = fz.merge(v[keep], on=["gene", "drug"], how="left")
    mut, drug = data.load_cohort_matrices("DISCOVERY")
    mde, p_full, p_shr = [], [], []
    for i, r in rep.iterrows():
        ok = r["status"] == "evaluated"
        mde.append(stats.mde_rank_biserial(r["n_mut_val"], r["n_wt_val"], cfg.POWER_TARGET) if ok else np.nan)
        if not ok:
            p_full.append(np.nan); p_shr.append(np.nan); continue
        ma, wa = data.group_aucs(r["gene"], r["drug"], mut, drug)
        w = float(np.clip(r["effect_EB"] / r["discovery_effect"], 0, 1)) if r["discovery_effect"] else 1.0
        if not np.isfinite(w):                       # EB did not converge => effect_EB=NaN; without this guard power_resample fails
            p_full.append(stats.power_resample(ma, wa, int(r["n_mut_val"]), int(r["n_wt_val"]), 1.0, cfg.POWER_B, seed=cfg.RANDOM_SEED))
            p_shr.append(np.nan)
            continue
        p_full.append(stats.power_resample(ma, wa, int(r["n_mut_val"]), int(r["n_wt_val"]), 1.0, cfg.POWER_B, seed=cfg.RANDOM_SEED))
        p_shr.append(stats.power_resample(ma, wa, int(r["n_mut_val"]), int(r["n_wt_val"]), w, cfg.POWER_B, seed=cfg.RANDOM_SEED))
    rep["MDE_rank_biserial_80pct"] = mde
    rep["power_at_observed_effect"] = p_full
    rep["power_at_shrunk_effect"] = p_shr
    rep["p_value"], rep["FDR_q"] = rep["validation_p"], rep["validation_FDR_q"]
    rep["interpretation"] = rep.apply(interpret, axis=1)
    rep = rep.drop(columns=["p_value", "FDR_q"])
    rep["rule_track"] = tag or "main"

    # ---- explicit status of every frozen pair (sample-count report for all candidates) ----
    # Only the PRESENCE of a column in the VALIDATION matrix is read, to separate an absent gene/drug; no VALIDATION statistic enters selection.
    try:
        vmut, vdrug = data.load_cohort_matrices("VALIDATION")
        v_genes, v_drugs = set(vmut.columns), set(vdrug.columns)
    except Exception as exc:                          # a missing matrix must not break the report; the reason for "absent" becomes "unspecified"
        print(f"[warn] could not read the VALIDATION matrix to classify absent pairs: {exc}")
        v_genes = v_drugs = None
    st_ = rep["status"]
    rep["flag_evaluated"] = (st_ == "evaluated")
    absent = (st_ == "not_evaluable_gene_or_drug_absent")
    if v_genes is None:
        rep["flag_gene_absent"] = pd.Series(pd.NA, index=rep.index, dtype="boolean")
        rep["flag_drug_absent"] = pd.Series(pd.NA, index=rep.index, dtype="boolean")
        reason_abs = pd.Series("absent_unspecified", index=rep.index)
    else:
        ga = ~rep["gene"].isin(v_genes)
        da = ~rep["drug"].isin(v_drugs)
        rep["flag_gene_absent"] = absent & ga
        rep["flag_drug_absent"] = absent & da
        reason_abs = pd.Series("absent_unspecified", index=rep.index)
        reason_abs[ga & ~da] = "gene_absent"
        reason_abs[~ga & da] = "drug_absent"
        reason_abs[ga & da] = "gene_and_drug_absent"
    rep["flag_low_n"] = (st_ == "not_evaluable_low_n")
    rep["flag_direction_assessable"] = rep["flag_evaluated"] & rep["direction_concordant"].notna()
    rep["flag_concordant"] = rep["flag_direction_assessable"] & (rep["direction_concordant"].astype(object) == True)
    rep["flag_validated"] = (rep["interpretation"] == "validated")
    rep["not_evaluable_reason"] = np.where(absent, reason_abs, np.where(rep["flag_low_n"], "low_n", ""))
    n_ev, n_ab, n_lo = int(rep["flag_evaluated"].sum()), int(absent.sum()), int(rep["flag_low_n"].sum())
    if n_ev + n_ab + n_lo != len(rep):
        raise SystemExit(f"[!] statuses do not add up: frozen={len(rep)} evaluated={n_ev} absent={n_ab} low_n={n_lo}")
    data.write(rep, "VALIDATION", rep_name)

    ev = rep[rep["status"] == "evaluated"]
    ev = ev[ev["direction_concordant"].notna()]      # only pairs whose concordance is assessable
    k = int((ev["direction_concordant"].astype(object) == True).sum()) if len(ev) else 0
    lines = [f"track: {tag or 'main'} | frozen candidates: {len(rep)} | evaluated in VALIDATION: {int((rep['status'] == 'evaluated').sum())} | direction-assessable: {len(ev)}",
             f"direction-concordant: {k}/{len(ev)}",
             "status accounting: frozen={} = evaluated {} + absent {} (gene {}, drug {}) + low_n {}".format(
                 len(rep), n_ev, n_ab, int(rep["flag_gene_absent"].fillna(False).astype(bool).sum()),
                 int(rep["flag_drug_absent"].fillna(False).astype(bool).sum()), n_lo)]
    if len(ev):
        lines.append(f"sign test (H0: p=0.5, one-sided): p = {binomtest(k, len(ev), 0.5, alternative='greater').pvalue:.3g}")
        lines.append(f"validated (concordant & q<{cfg.VALIDATION_FDR}): {(rep.interpretation == 'validated').sum()}")
        vg = rep.loc[rep.interpretation == "validated", "gene"]
        lines.append(f"distinct genes among validated: {vg.nunique()} (pairs sharing a gene/drug are dependent; cross-wave, not fully independent)")
        lines.append(f"underpowered at shrunk effect (<{cfg.POWER_TARGET:.0%}): {(~(ev.power_at_shrunk_effect >= cfg.POWER_TARGET)).sum()}/{len(ev)}")
        lines.append(f"median power at observed/shrunk effect: {ev.power_at_observed_effect.median():.2f} / {ev.power_at_shrunk_effect.median():.2f}")
        lines.append(f"median MDE (|rank-biserial|, 80% power): {ev.MDE_rank_biserial_80pct.median():.2f}")
    lines.append("Interpretation: \"not confirmed\" when underpowered means the data are insufficient to judge, not that the effect is absent.")
    txt = "\n".join(lines)
    print("\n" + txt)
    data.path("VALIDATION", sum_name).write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
