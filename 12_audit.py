# -*- coding: utf-8 -*-
"""
12_audit.py   (v3.1)
=========================================================================================
Automatic validation before the master table/figures.
  python 12_audit.py --cohort DISCOVERY | FULL | VALIDATION

DISCOVERY / FULL (screen):
  C1 n in associations == analysis_population == sensitivity baseline (for every stable pair)
  C2 direction: sign(effect) == sign(bootstrap median) (error); sign(OLS beta) == -sign(effect) (warning; the signs are opposite by convention)
  C3 drug names in the lookup
  C4 frozen is a subset of stable, in_main, adjustment pass; main and sensitivity are disjoint; hash of lists/parameters == freeze_manifest == VALIDATION run (both tracks)
  C7 BH over the whole associations family
  C5 wave separation: DISCOVERY and VALIDATION do not overlap (from the manifest)
  C6 no duplicated/NaN pair in the key outputs
  (v3.1) INFO/WARN: empty freeze_audit columns, role of the stability gate, q of the frozen list

VALIDATION (no screen; evaluation of the frozen list only):
  V0 verify_freeze (hash of hard code, parameters, analysis definition, manifest, sources)
  V1 existence of marker/result/report/metadata files for both tracks
  V2 hash chain: freeze_manifest == started == metadata == frozen file; results sha == completed; same run_id; start time after the freeze
  V3 coverage: every frozen pair appears exactly once in results and report; main and sensitivity are disjoint
  V4 status accounting: frozen = evaluated + absent + low_n; equals run_metadata; fdr_m == evaluated
  V5 BH recomputed on the p-values of evaluated pairs; report == results
  V6 direction: concordance flag == sign; NaN only for a zero/non-evaluated effect
  V7 interpretation recomputed (validated = concordant & q<VALIDATION_FDR) and equal to validation_summary.txt
  V8 n threshold: low_n below the minimum; evaluated at or above the minimum
  V9 patient manifest: zero overlap between DISCOVERY/VALIDATION; hash and n agree with the metadata
"""
import json
import re
import numpy as np
import pandas as pd
from common import config_official as cfg, data, stats

KEY = ["gene", "drug"]


def _pairs(df):
    return set(zip(df.gene, df.drug))


# ====================================================================== VALIDATION
def audit_validation(err, warn, info):
    fm_p = data.path("DISCOVERY", "freeze_manifest.json")
    if not fm_p.exists():
        err.append("freeze_manifest.json is missing; VALIDATION without a freeze is not valid")
        return
    fm = json.loads(fm_p.read_text())
    mfile_ok = cfg.MANIFEST_FILE.exists() and not data.is_synthetic()
    pair_sets = {}
    for tg in (None, "sensitivity"):
        lab = "sensitivity" if tg else "main"
        fn = "frozen_sensitivity_list.csv" if tg else "frozen_candidate_list.csv"
        fkey = "frozen_sensitivity_list_sha256" if tg else "frozen_candidate_list_sha256"
        # V0
        try:
            data.verify_freeze(data.path("DISCOVERY", fn), tg)
        except SystemExit as e:
            err.append(f"[{lab}] V0 verify_freeze: {e}")
        fz = data.read("DISCOVERY", fn, required=False)
        files = {"started": "validation_started.json", "completed": "validation_completed.json",
                 "results": "validation_results.csv", "report": "validation_report.csv",
                 "meta": "run_metadata.json", "summary": "validation_summary.txt"}
        P = {k: data.path("VALIDATION", data.tagged(n, tg)) for k, n in files.items()}
        miss = [files[k] for k, p in P.items() if not p.exists()]
        if miss or fz is None:
            err.append(f"[{lab}] V1 files missing: {miss or fn}")
            continue
        started, done, md = (json.loads(P[k].read_text()) for k in ("started", "completed", "meta"))
        fh = data.sha256_file(data.path("DISCOVERY", fn))
        # V2
        for name, val in (("freeze_manifest", fm.get(fkey)), ("validation_started", started.get("frozen_list_sha256")),
                          ("run_metadata", md.get("frozen_list_sha256"))):
            if val != fh:
                err.append(f"[{lab}] V2 the hash of the frozen list in {name} does not match the current file")
        if not (started.get("run_id") == done.get("run_id") == md.get("validation_run_id")):
            err.append(f"[{lab}] V2 run_id differs between started/completed/metadata")
        if done.get("results_sha256") != data.sha256_file(P["results"]):
            err.append(f"[{lab}] V2 the hash of validation_results.csv does not match validation_completed.json (file changed after the run)")
        if started.get("freeze_id") != fm.get("freeze_id"):
            err.append(f"[{lab}] V2 the freeze_id of the marker differs from freeze_manifest")
        if str(started.get("started_at_utc", "")) < str(fm.get("frozen_at_utc", "")):
            err.append(f"[{lab}] V2 VALIDATION started before the freeze")
        if (md.get("freeze_binding") or {}).get("warnings"):
            warn.append(f"[{lab}] freeze_binding warning in metadata: {md['freeze_binding']['warnings']}")
        res, rep = pd.read_csv(P["results"]), pd.read_csv(P["report"])
        # V3
        for nm, d in (("results", res), ("report", rep)):
            if d.duplicated(KEY).any():
                err.append(f"[{lab}] V3 duplicated pair in {nm}")
            if _pairs(d) != _pairs(fz):
                err.append(f"[{lab}] V3 the pairs of {nm} do not match the frozen list (frozen={len(fz)}, {nm}={len(d)})")
        pair_sets[lab] = _pairs(fz)
        # V4
        st = rep["status"]
        ev, ab, lo = st == "evaluated", st == "not_evaluable_gene_or_drug_absent", st == "not_evaluable_low_n"
        if (~(ev | ab | lo)).any():
            err.append(f"[{lab}] V4 unknown status: {sorted(set(st[~(ev | ab | lo)]))}")
        vf = md.get("validation_family") or {}
        for k, v in (("n_frozen", len(rep)), ("n_evaluated", int(ev.sum())),
                     ("n_not_evaluable_gene_or_drug_absent", int(ab.sum())),
                     ("n_not_evaluable_low_n", int(lo.sum())), ("fdr_m", int(ev.sum()))):
            if vf.get(k) != v:
                err.append(f"[{lab}] V4 {k}: metadata={vf.get(k)} ≠ report={v}")
        if len(rep) != int(ev.sum() + ab.sum() + lo.sum()):
            err.append(f"[{lab}] V4 frozen ≠ evaluated + absent + low_n")
        # V5
        e = rep[ev].copy()
        if e["validation_p"].isna().any():
            err.append(f"[{lab}] V5 NaN p among evaluated pairs")
        else:
            q = stats.fdr(e["validation_p"].to_numpy(float))
            if not np.allclose(q, e["validation_FDR_q"].to_numpy(float), atol=1e-9):
                err.append(f"[{lab}] V5 validation_FDR_q does not match BH over m={len(e)}")
        j = rep.merge(res[KEY + ["p_value", "FDR_q", "rank_biserial_effect", "n_mutated_with_AUC", "n_wt_with_AUC"]], on=KEY, how="left")
        je = j[j.status == "evaluated"]
        if not np.allclose(je.validation_p, je.p_value, equal_nan=True) or \
                not np.allclose(je.validation_FDR_q, je.FDR_q, equal_nan=True):
            err.append(f"[{lab}] V5 p/q in the report do not match results")
        # V6
        eff, disc = e["validation_effect"].to_numpy(float), e["discovery_effect"].to_numpy(float)
        assess = np.isfinite(eff) & np.isfinite(disc) & (eff != 0) & (disc != 0)
        conc = e["direction_concordant"]
        if (conc.notna().to_numpy() != assess).any():
            err.append(f"[{lab}] V6 NaN in direction_concordant does not match assessability")
        exp = np.sign(eff) == np.sign(disc)
        flag = (conc.astype(object) == True).to_numpy()
        if (flag[assess] != exp[assess]).any():
            err.append(f"[{lab}] V6 direction_concordant does not match the signs of the effects")
        # V7
        thr = cfg.VALIDATION_FDR
        val_exp = assess & flag & (e["validation_FDR_q"].to_numpy(float) < thr)
        n_val_exp = int(val_exp.sum())
        n_val_rep = int((rep["interpretation"] == "validated").sum())
        if n_val_exp != n_val_rep:
            err.append(f"[{lab}] V7 validated: recomputed={n_val_exp} != report={n_val_rep}")
        if ((rep["interpretation"] == "validated") & ~ev).any():
            err.append(f"[{lab}] V7 a non-evaluated pair was marked validated")
        txt = P["summary"].read_text(encoding="utf-8")
        got = {"evaluated": re.search(r"evaluated in VALIDATION: (\d+)", txt),
               "assess": re.search(r"direction-assessable: (\d+)", txt),
               "conc": re.search(r"direction-concordant: (\d+)/(\d+)", txt),
               "val": re.search(r"validated \(concordant[^)]*\): (\d+)", txt)}
        exp_n = {"evaluated": int(ev.sum()), "assess": int(assess.sum()), "conc": int((flag & assess).sum()), "val": n_val_exp}
        for k, m in got.items():
            if m is None:
                warn.append(f"[{lab}] V7 line '{k}' not found in validation_summary.txt")
            elif int(m.group(1)) != exp_n[k]:
                err.append(f"[{lab}] V7 validation_summary.txt ({k}={m.group(1)}) does not match the recomputation ({exp_n[k]})")
        # V8
        lo_rows = j[j.status == "not_evaluable_low_n"]
        if len(lo_rows) and not ((lo_rows.n_mutated_with_AUC < cfg.MIN_MUTATED_WITH_AUC) |
                                 (lo_rows.n_wt_with_AUC < cfg.MIN_WT_WITH_AUC)).all():
            err.append(f"[{lab}] V8 low_n pair whose n is not below the minimum")
        if len(je) and not ((je.n_mutated_with_AUC >= cfg.MIN_MUTATED_WITH_AUC) & (je.n_wt_with_AUC >= cfg.MIN_WT_WITH_AUC)).all():
            err.append(f"[{lab}] V8 evaluated pair whose n is below the minimum")
        if not (je.n_mut_val.astype(float).to_numpy() == je.n_mutated_with_AUC.astype(float).to_numpy()).all():
            err.append(f"[{lab}] V8 n in the report does not match results")
        # V9 (each track)
        if mfile_ok:
            m = pd.read_csv(cfg.MANIFEST_FILE)
            if md.get("manifest_sha256") != data.sha256_file(cfg.MANIFEST_FILE):
                err.append(f"[{lab}] V9 the hash of the patient manifest does not match the VALIDATION run")
            if int(m.in_VALIDATION.sum()) != md.get("n_patients_manifest"):
                err.append(f"[{lab}] V9 number of VALIDATION patients in the manifest={int(m.in_VALIDATION.sum())} != metadata={md.get('n_patients_manifest')}")
            if (m.in_DISCOVERY & m.in_VALIDATION).any():
                err.append(f"[{lab}] V9 patient present in both DISCOVERY and VALIDATION")
        al = md.get("alignment") or {}
        if al.get("n_dropped_mut_only") or al.get("n_dropped_drug_only"):
            warn.append(f"[{lab}] V9 patients dropped during alignment: {al}")
        # status table
        gabs = int((rep.get("flag_gene_absent", pd.Series(dtype=bool)).fillna(False).astype(bool)).sum()) if "flag_gene_absent" in rep else None
        info.append(f"[{lab}] frozen={len(rep)} | evaluated={int(ev.sum())} | absent={int(ab.sum())}"
                    f"{'' if gabs is None else f' (gene={gabs})'} | low_n={int(lo.sum())} | "
                    f"direction_assessable={int(assess.sum())} | concordant={int((flag & assess).sum())} | validated={n_val_exp}")
        if "flag_validated" not in rep.columns:
            warn.append(f"[{lab}] the report has no flag_* columns; re-run 14_validation_report.py (v3.1)")
    if len(pair_sets) == 2 and (pair_sets["main"] & pair_sets["sensitivity"]):
        err.append("V3 the main and sensitivity lists overlap")


# ====================================================================== DISCOVERY / FULL
def audit_screen(cohort, err, warn, info):
    assoc = data.read(cohort, "mutation_drug_associations.csv")
    st = data.read(cohort, "stable_pairs.csv")
    bs = data.read(cohort, "bootstrap_robustness_results.csv")
    adj = data.read(cohort, "adjusted_analysis_results.csv")
    sens = data.read(cohort, "sensitivity_negative_control_results.csv")
    pop = data.read(cohort, "analysis_population.csv")
    key = KEY
    if st.duplicated(key).any():
        err.append("stable_pairs contains a duplicated pair")
    for _, r in st.iterrows():
        g, d = r["gene"], r["drug"]
        tag = f"{g} x {d}"
        a = assoc[(assoc.gene == g) & (assoc.drug == d)]
        if a.empty:
            err.append(f"{tag}: not in associations")
            continue
        a = a.iloc[0]
        p = pop[(pop.gene == g) & (pop.drug == d)]
        nm, nw = int((p.mutation_status == 1).sum()), int((p.mutation_status == 0).sum())
        if (a.n_mutated_with_AUC, a.n_wt_with_AUC) != (nm, nw):
            err.append(f"{tag}: n in associations ({a.n_mutated_with_AUC}/{a.n_wt_with_AUC}) != population ({nm}/{nw})")
        s = sens[(sens.gene == g) & (sens.drug == d) & (sens.check == "0_baseline")]
        if not s.empty and (s.iloc[0].n_mutated, s.iloc[0].n_wt) != (nm, nw):
            warn.append(f"{tag}: n in the sensitivity baseline differs from the population")
        b = bs[(bs.gene == g) & (bs.drug == d)]
        if not b.empty and np.sign(b.iloc[0].bootstrap_median_effect) != np.sign(a.rank_biserial_effect):
            err.append(f"{tag}: the sign of the bootstrap median differs from the main effect")
        j = adj[(adj.gene == g) & (adj.drug == d)]
        if not j.empty and np.isfinite(j.iloc[0].unadjusted_beta) and \
                np.sign(j.iloc[0].unadjusted_beta) == np.sign(a.rank_biserial_effect):
            warn.append(f"{tag}: beta and rank-biserial have the same sign (expected: opposite); check the sign convention/data")
    sa = data.read(cohort, "selection_adjusted_associations.csv", required=False)
    if sa is not None and len(st):
        j = st[key].merge(sa[key + ["p_WY"]], on=key, how="left")
        if j["p_WY"].isna().any():
            err.append("p_WY is missing for stable pair(s): the screen family differs between 01 and 08b")
    names = data.drug_names()
    miss = sorted(set(st.drug) - set(names))
    if miss:
        warn.append(f"drugs without a name in the lookup: {miss}")
    # C7: BH over the "whole eligible family" (not over the selected subset)
    q_re = stats.fdr(assoc["p_value"].to_numpy())
    if not np.allclose(q_re, assoc["FDR_q"].to_numpy(), atol=1e-9, equal_nan=True):
        err.append("FDR_q in associations does not match BH over the whole table family")
    if "fdr_family" in assoc.columns and f"m={len(assoc)}" not in str(assoc["fdr_family"].iloc[0]):
        err.append("the fdr_family label does not match the table size")
    if cohort == "DISCOVERY":
        fz = data.read("DISCOVERY", "frozen_candidate_list.csv", required=False)
        fs = data.read("DISCOVERY", "frozen_sensitivity_list.csv", required=False)
        fa = data.read("DISCOVERY", "freeze_audit.csv", required=False)
        if fz is not None and len(fz):
            if not _pairs(fz) <= _pairs(st):
                err.append("frozen contains a non-stable pair")
            bad = fz[~fz.adjustment_interpretation.isin(cfg.ADJ_PASS_LABELS)]
            if len(bad):
                err.append(f"frozen contains a pair that failed adjustment: {bad[key].values.tolist()}")
            if not fz["in_main"].astype(bool).all():
                err.append("frozen contains a pair outside the main (nominal) rule")
            if (fz["p_value"] >= cfg.NOMINAL_P_MAX).any():
                err.append("frozen contains a pair with p >= NOMINAL_P_MAX")
            info.append(f"frozen main: {len(fz)} pairs | max p={fz.p_value.max():.4g} | max BH q={fz.FDR_q.max():.4g} | "
                        f"q<{cfg.SELECT_FDR_Q}: {int((fz.FDR_q < cfg.SELECT_FDR_Q).sum())}/{len(fz)} (selection was not FDR-based; descriptive only)")
        if fz is not None and fs is not None and len(fz) and len(fs) and (_pairs(fz) & _pairs(fs)):
            err.append("the main and sensitivity lists overlap (they must be disjoint)")
        if fa is None:
            warn.append("freeze_audit.csv is missing (was 08b run?)")
        else:
            for c in ("n_split_half_valid", "CI_crosses_zero"):
                if c in fa.columns and fa[c].isna().all():
                    warn.append(f"freeze_audit.csv: column {c} is completely empty (export bug; the real value is in bootstrap_robustness_results.csv; do not modify 08b, take it from the bootstrap file in the supplement)")
            if "G2_stable" in fa.columns:
                info.append(f"stability gate: {int(fa.G2_stable.astype(bool).sum())}/{len(fa)} passed"
                            + (" -> the gate does not discriminate; do not present it as evidence of error reduction" if fa.G2_stable.astype(bool).all() else ""))
        fm_p = data.path("DISCOVERY", "freeze_manifest.json")
        if fm_p.exists():
            fm = json.loads(fm_p.read_text())
            if fz is not None and fm.get("frozen_candidate_list_sha256") != data.sha256_file(data.path("DISCOVERY", "frozen_candidate_list.csv")):
                err.append("the hash of the frozen list does not match freeze_manifest.json (file changed after the freeze)")
            if fm.get("selection_params_sha256") != data.params_sha256():
                err.append("the selection parameters differ from those at freeze time")
            for tg in (None, "sensitivity"):
                mp = data.path("VALIDATION", data.tagged("run_metadata.json", tg))
                if mp.exists():
                    md = json.loads(mp.read_text())
                    fn = "frozen_sensitivity_list.csv" if tg else "frozen_candidate_list.csv"
                    if md.get("frozen_list_sha256") != data.sha256_file(data.path("DISCOVERY", fn)):
                        err.append(f"the hash of {fn} does not match the VALIDATION run{' (sensitivity)' if tg else ''}")
                    if md.get("selection_params_sha256") != data.params_sha256():
                        err.append(f"VALIDATION{' (sensitivity)' if tg else ''} was not run with the current selection parameters")
        elif data.validation_started():
            err.append("VALIDATION was run but freeze_manifest.json is missing")
        if cfg.MANIFEST_FILE.exists() and not data.is_synthetic():
            m = pd.read_csv(cfg.MANIFEST_FILE)
            if (m.in_DISCOVERY & m.in_VALIDATION).any():
                err.append("some patients are in both DISCOVERY and VALIDATION")


def main():
    cohort = data.parse_cohort("audit", choices=("DISCOVERY", "FULL", "VALIDATION"))
    err, warn, info = [], [], []
    if cohort == "VALIDATION":
        audit_validation(err, warn, info)
    else:
        audit_screen(cohort, err, warn, info)
    print("=" * 80)
    print(f"audit cohort: {cohort}")
    if err:
        print("❌ ERRORS:"); [print("  -", e) for e in err]
    else:
        print("✅ no critical error")
    if warn:
        print("⚠️ WARNINGS:"); [print("  -", w) for w in warn]
    if info:
        print("ℹ️ INFO:"); [print("  -", i) for i in info]
    if err:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
