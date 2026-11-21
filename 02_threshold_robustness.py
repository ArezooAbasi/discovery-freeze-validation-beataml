# -*- coding: utf-8 -*-
"""
02_threshold_robustness.py   (v3 / Path B)
=========================================================================================
Two tasks, both based only on mutation_drug_associations.csv of the same cohort (BH in stage 01 is computed over the full eligible family):
 (1) Sensitivity to the minimum-n threshold: for each t in THRESHOLDS, FDR is recomputed over the family eligible at that t (descriptive; not a gate).
 (2) Build the candidate pool candidate_pairs.csv (no forced Top-N):
       in_main     : main rule = p < NOMINAL_P_MAX and n per group >= MAIN_MIN_N_PER_GROUP   (may be empty)
       in_bh_gate  : sensitivity = BH q < SELECT_FDR_Q (over the whole screen)
       in_top{N}   : sensitivity = legacy threshold-stable Top-N rule
     The pool is the union of the three; the columns stay separate and are never merged.
     eligible_at_t and q_family_t{t} are descriptive. (p does not depend on t; the main rule depends on t only through eligibility.)
  python 02_threshold_robustness.py --cohort DISCOVERY
Outputs: mutation_drug_threshold_robustness.csv, candidate_pairs.csv
"""
import pandas as pd
from common import config_official as cfg, data, selection, stats


def main():
    cohort = data.parse_cohort("threshold sensitivity + candidate pool")
    df = data.read(cohort, "mutation_drug_associations.csv", hint="اول 01_association_screen.py")
    df = df.dropna(subset=["gene", "drug", "p_value", "rank_biserial_effect"]).reset_index(drop=True)
    print(f"[{cohort}] pairs (کل خانواده): {len(df)}")

    rows, fam_q = [], {}
    for t in cfg.THRESHOLDS:
        s = df[(df["n_mutated_with_AUC"] >= t) & (df["n_wt_with_AUC"] >= t)].copy()
        s["q"] = stats.fdr(s["p_value"]) if len(s) else []
        fam_q[t] = s.set_index(["gene", "drug"])["q"]
        rows.append({"minimum_n_per_group": t, "n_feasible_pairs": len(s),
                     "n_nominal_p_below_main": int((s["p_value"] < cfg.NOMINAL_P_MAX).sum()),
                     "n_FDR_below_0.05": int((s["q"] < 0.05).sum()), "n_FDR_below_0.10": int((s["q"] < 0.10).sum()),
                     "min_q": s["q"].min() if len(s) else float("nan")})
    tr = pd.DataFrame(rows)
    print(tr.to_string(index=False))
    data.write(tr, cohort, "mutation_drug_threshold_robustness.csv")

    main_set = set(selection.main_candidates(df))
    bh_set = set(selection.bh_gate_candidates(df))
    tops = {N: set(selection.threshold_stable(df, top_n=N)[0]) for N in cfg.SENSITIVITY_TOP_N}
    pool = sorted(main_set | bh_set | set().union(*tops.values()))
    print(f"\nMAIN (p<{cfg.NOMINAL_P_MAX}, n>={cfg.MAIN_MIN_N_PER_GROUP}/گروه): {len(main_set)}  [نتیجه‌ی صفر مجاز است]")
    print(f"SENSITIVITY: BH q<{cfg.SELECT_FDR_Q}: {len(bh_set)} | " + " | ".join(f"Top-{N} threshold-stable: {len(v)}" for N, v in tops.items()))
    print(f"استخر کاندید (اجتماع): {len(pool)}")

    top_cols = [f"in_top{N}" for N in cfg.SENSITIVITY_TOP_N]
    cols = (["gene", "drug", "in_main", "in_bh_gate"] + top_cols + [f"eligible_at_{t}" for t in cfg.THRESHOLDS]
            + [f"q_family_t{t}" for t in cfg.THRESHOLDS] + ["selection_reason"])
    recs = []
    idx = df.set_index(["gene", "drug"])
    for g, d in pool:
        r = idx.loc[(g, d)]
        rec = {"gene": g, "drug": d, "in_main": (g, d) in main_set, "in_bh_gate": (g, d) in bh_set}
        for N in cfg.SENSITIVITY_TOP_N:
            rec[f"in_top{N}"] = (g, d) in tops[N]
        for t in cfg.THRESHOLDS:
            ok = r["n_mutated_with_AUC"] >= t and r["n_wt_with_AUC"] >= t
            rec[f"eligible_at_{t}"] = bool(ok)
            rec[f"q_family_t{t}"] = float(fam_q[t].get((g, d), float("nan"))) if ok else float("nan")
        why = (["main_nominal"] if rec["in_main"] else []) + (["bh_gate"] if rec["in_bh_gate"] else []) \
            + [f"legacy_top{N}" for N in cfg.SENSITIVITY_TOP_N if rec[f"in_top{N}"]]
        rec["selection_reason"] = ";".join(why)
        recs.append(rec)
    data.write(pd.DataFrame(recs, columns=cols), cohort, "candidate_pairs.csv")


if __name__ == "__main__":
    main()
