# -*- coding: utf-8 -*-
"""
03_stability_checks.py   (v3)
=========================================================================================
Bootstrap + split-half + winsorize/trim for ALL rows of candidate_pairs.csv (main and sensitivity; the in_* columns stay separate).
Both are INTERNAL stability checks on the same Discovery data: neither is independent replication nor a winner's-curse correction.
Exact definitions of the percentages and denominators: docstring of common/selection.py.
  - pct_negative/positive_direction : denominator = n_boot_draws
  - pct_split_half_*                 : denominator = n_split_half_attempted (an invalid split counts as a failure);
                                       pct_split_half_match_full_among_valid is for transparency only (not a gate)
  python 03_stability_checks.py --cohort DISCOVERY
Output: bootstrap_robustness_results.csv
"""
import pandas as pd
from common import config_official as cfg, data, selection, stats

COLS = ["gene", "drug", "n_mutated", "n_wt", "observed_effect", "bootstrap_median_effect", "bootstrap_CI_low",
        "bootstrap_CI_high", "CI_crosses_zero", "n_boot_draws", "pct_negative_direction", "pct_positive_direction",
        "n_split_half_attempted", "n_split_half_valid", "pct_split_half_valid", "pct_split_half_same_direction",
        "pct_split_half_match_full", "pct_split_half_match_full_among_valid", "pct_split_half_nominal_p",
        "effect_winsorized", "effect_trimmed"]


def main():
    cohort = data.parse_cohort("stability checks")
    cands = data.read(cohort, "candidate_pairs.csv", hint="اول 02_threshold_robustness.py")
    if cands.empty:
        print("[i] استخر کاندید خالی است (نتیجه‌ی null)؛ خروجی خالی نوشته می‌شود.")
        data.write(pd.DataFrame(columns=COLS), cohort, "bootstrap_robustness_results.csv")
        return
    mut, drug = data.load_cohort_matrices(cohort)
    rows = []
    for _, r in cands.iterrows():
        g, d = r["gene"], r["drug"]
        ma, wa = data.group_aucs(g, d, mut, drug)
        if ma is None:
            continue
        s = selection.stability_row(ma, wa, seed=cfg.RANDOM_SEED)
        row = {"gene": g, "drug": d, "n_mutated": len(ma), "n_wt": len(wa), **s,
               "effect_winsorized": stats.rank_biserial(stats.winsorize(ma), stats.winsorize(wa)),
               "effect_trimmed": stats.rank_biserial(stats.trim(ma), stats.trim(wa))}
        rows.append(row)
        print(f"{g} x {d}: n={len(ma)}/{len(wa)} effect={row['observed_effect']:+.3f} "
              f"CI=[{s['bootstrap_CI_low']:+.3f},{s['bootstrap_CI_high']:+.3f}] "
              f"split_match={s['pct_split_half_match_full']:.1f}%")
    data.write(pd.DataFrame(rows, columns=COLS), cohort, "bootstrap_robustness_results.csv")


if __name__ == "__main__":
    main()
