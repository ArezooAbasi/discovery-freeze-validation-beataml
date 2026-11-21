# -*- coding: utf-8 -*-
"""
09_sensitivity_negative_control.py
=========================================================================================
Sensitivity checks for every stable pair (all pairs; confounders from adjusted_analysis_results.csv):
  0 baseline | 1 outlier removal (IQR) | 3 permutation test | 4 drop mutant samples for the Stage 08 confounders
  4b drop mutants for the confounder of the predefined question (cfg.EXTRA_CONFOUNDING_CHECKS)
  6 drug repeat-record rule: mean and exclude_repeated   |  7 converged records only
  8 WES-only FLT3 (without clinical ITD) for FLT3 pairs when cfg.FLT3_COMPOSITE=True
    WARNING: in 8, ITD-only patients (ITD positive, WES negative) fall into the WT group; so 8 is a "misclassification" test.
  8b same WES-only analysis, but ITD-only patients are removed from both groups (post-hoc, after the freeze; report it as such in the text).
  (6, 7, 8 and 8b rebuild the matrix with a temporary cfg override)
  python 09_sensitivity_negative_control.py --cohort DISCOVERY
  python 09_sensitivity_negative_control.py --cohort DISCOVERY --only-8b   # only the 8b rows; merged into the existing file
"""
import argparse
import numpy as np
import pandas as pd
from common import config_official as cfg, data, stats

CHK_8B = "8b_flt3_wes_only_excl_itd_only"
RES_NAME = "sensitivity_negative_control_results.csv"


def row(g, d, check, ma, wa, note="", p_perm=np.nan, effect=None, p=None):
    if ma is None or len(ma) < 3 or len(wa) < 3:
        return {"gene": g, "drug": d, "check": check, "n_mutated": 0 if ma is None else len(ma),
                "n_wt": 0 if wa is None else len(wa), "rank_biserial_effect": np.nan, "p_value": np.nan,
                "p_permutation": np.nan, "delta_median_AUC": np.nan, "delta_mean_AUC": np.nan,
                "note": "insufficient n " + note}
    e, pv = stats.rb_p(ma, wa)
    return {"gene": g, "drug": d, "check": check, "n_mutated": len(ma), "n_wt": len(wa),
            "rank_biserial_effect": e if effect is None else effect, "p_value": pv if p is None else p,
            "p_permutation": p_perm, "delta_median_AUC": np.median(ma) - np.median(wa),
            "delta_mean_AUC": np.mean(ma) - np.mean(wa), "note": note}


def itd_only_patients(mut, wes_mut):
    """ITD-only = positive in the composite FLT3 but WES-negative (patients whose ITD was missed by WES)."""
    comp = set(mut.index[mut[cfg.FLT3_GENE] == 1]) if cfg.FLT3_GENE in mut.columns else set()
    wes = set(wes_mut.index[wes_mut[cfg.FLT3_GENE] == 1]) if cfg.FLT3_GENE in wes_mut.columns else set()
    return comp - wes


def row_8b(g, d, wes_mut, wes_drug, itd_only):
    ma, wa = data.group_aucs(g, d, wes_mut, wes_drug, exclude=itd_only)
    y = wes_drug[d].dropna() if d in wes_drug.columns else pd.Series(dtype=float)
    n_ex = int(y.index.isin(itd_only).sum())
    return row(g, d, CHK_8B, ma, wa, f"FLT3 = WES only; excluded {n_ex} ITD-only patient(s) with AUC (composite+, WES-)")


def only_8b(cohort):
    p = data.path(cohort, RES_NAME)
    if not p.exists():
        raise SystemExit(f"[!] {p} not found; run stage 09 completely first.")
    st = data.read(cohort, "stable_pairs.csv", hint="run 04_export_stable_pairs.py first")
    if not cfg.FLT3_COMPOSITE:
        raise SystemExit("[!] FLT3_COMPOSITE=False; 8b is meaningless.")
    mut, drug = data.load_cohort_matrices(cohort)
    wes_mut, wes_drug = data.load_cohort_matrices(cohort, {"FLT3_COMPOSITE": False})
    itd_only = itd_only_patients(mut, wes_mut)
    print(f"[8b] ITD-only patients in cohort: {len(itd_only)}")
    new = [row_8b(r["gene"], r["drug"], wes_mut, wes_drug, itd_only) for _, r in st.iterrows() if r["gene"] == cfg.FLT3_GENE]
    old = pd.read_csv(p, keep_default_na=False, na_values=[""], encoding="utf-8-sig")        # do not turn 'null'/'NA' into NaN
    old = old[old["check"] != CHK_8B]
    res = pd.concat([old, pd.DataFrame(new)], ignore_index=True)
    data.write(res, cohort, RES_NAME)
    print(f"[8b] {len(new)} new rows; previous rows untouched ({len(old)}).")


def main():
    ap = argparse.ArgumentParser(description="sensitivity")
    ap.add_argument("--cohort", choices=["DISCOVERY", "FULL"], default="DISCOVERY")
    ap.add_argument("--only-8b", action="store_true")
    args = ap.parse_args()
    cohort = args.cohort
    if args.only_8b:
        only_8b(cohort)
        return
    st = data.read(cohort, "stable_pairs.csv", hint="run 04_export_stable_pairs.py first")
    adj = data.read(cohort, "adjusted_analysis_results.csv", hint="run 08_adjusted_analysis.py first")
    conf = {(r.gene, r.drug): [c for c in str(r.confounders_included).split(";") if c and c != "nan"]
            for r in adj.itertuples()}
    mut, drug = data.load_cohort_matrices(cohort)
    alt = {"6_policy_mean": {"DRUG_REPEAT_POLICY": "mean"},
           "6_policy_exclude_repeated": {"DRUG_REPEAT_POLICY": "exclude_repeated"},
           "7_converged_only": {"DRUG_REQUIRE_CONVERGED": True}}
    alt_mats = {k: data.load_cohort_matrices(cohort, v) for k, v in alt.items()} if len(st) else {}
    wes_mats, itd_only = None, set()
    if len(st) and cfg.FLT3_COMPOSITE and (st["gene"] == cfg.FLT3_GENE).any():
        wes_mats = data.load_cohort_matrices(cohort, {"FLT3_COMPOSITE": False})      # 8: WES-only FLT3
        itd_only = itd_only_patients(mut, wes_mats[0])                                # 8b
    rows = []
    for _, r in st.iterrows():
        g, d = r["gene"], r["drug"]
        ma, wa = data.group_aucs(g, d, mut, drug)
        rows.append(row(g, d, "0_baseline", ma, wa, "reference"))
        mc, wc, nrm = stats.remove_outliers_iqr(ma, wa)
        rows.append(row(g, d, "1_outlier_removed_IQR", mc, wc, f"{nrm} outlier(s) removed"))
        obs, pp = stats.permutation_test(ma, wa, cfg.N_PERMUTATIONS, cfg.RANDOM_SEED)
        rows.append(row(g, d, "3_permutation_test", ma, wa, f"{cfg.N_PERMUTATIONS} permutations", p_perm=pp))
        cg = conf.get((g, d), [])
        if cg:
            ex = set().union(*[data.mutated_set(mut, c) for c in cg])
            m2, w2 = data.group_aucs(g, d, mut, drug, exclude=ex)
            rows.append(row(g, d, "4_excl_" + "_".join(cg), m2, w2, f"excluded mutant for {cg}"))
        if g in cfg.EXTRA_CONFOUNDING_CHECKS:
            eg = cfg.EXTRA_CONFOUNDING_CHECKS[g]
            eg = [eg] if isinstance(eg, str) else list(eg)          # the config value is now a list (several confounders)
            m2, w2 = data.group_aucs(g, d, mut, drug, exclude=set().union(*[data.mutated_set(mut, c) for c in eg]))
            rows.append(row(g, d, f"4b_excl_{eg}", m2, w2, "excluded; pre-defined concern"))
        if wes_mats is not None and g == cfg.FLT3_GENE:
            m2, w2 = data.group_aucs(g, d, *wes_mats)
            rows.append(row(g, d, "8_flt3_wes_only", m2, w2, "FLT3 = WES only (no clinical ITD)"))
            rows.append(row_8b(g, d, wes_mats[0], wes_mats[1], itd_only))
        for k, (mu2, dr2) in alt_mats.items():
            m2, w2 = data.group_aucs(g, d, mu2, dr2)
            rows.append(row(g, d, k, m2, w2, str(alt[k])))
    res = pd.DataFrame(rows)
    data.write(res, cohort, RES_NAME)
    print("\nDirection of effect relative to baseline:")
    for (g, d), s in res.dropna(subset=["rank_biserial_effect"]).groupby(["gene", "drug"]):
        b = np.sign(s.loc[s.check == "0_baseline", "rank_biserial_effect"].iloc[0])
        ok = (np.sign(s.loc[s.check != "0_baseline", "rank_biserial_effect"]) == b).all()
        print(f"  {g} x {d}: {'consistent' if ok else 'SIGN CHANGE in >=1 check'}")


if __name__ == "__main__":
    main()
