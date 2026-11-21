# -*- coding: utf-8 -*-
"""
10_prespecified_confounding.py   (formerly: runx1_flt3_confounding.py)
=========================================================================================
"Predefined / exploratory" confounding questions in cfg.EXTRA_CONFOUNDING_CHECKS (gene -> confounder); not the output of the screening.
For every stable pair whose gene is in this dictionary: co-occurrence (Fisher), the effect in the confounder-WT and confounder-mutant strata,
and the effect of the confounder itself for comparison. If no pair is relevant it prints a message instead of failing.
(Use the word "prespecified" only if you have a time-stamped protocol; otherwise write "predefined" in the text.)
  python 10_prespecified_confounding.py --cohort DISCOVERY
"""
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from common import config_official as cfg, data, stats


def stratum(y, mut, gene, subset, label, drug):
    y = y[y.index.isin(subset)]
    m = mut[gene].reindex(y.index) == 1
    ma, wa = y[m].to_numpy(), y[~m].to_numpy()
    rec = {"drug": drug, "tested_gene": gene, "stratum": label, "n_mutated": len(ma), "n_wt": len(wa),
           "rank_biserial_effect": np.nan, "p_value": np.nan,
           "median_mutated_AUC": np.median(ma) if len(ma) else np.nan, "median_wt_AUC": np.median(wa) if len(wa) else np.nan}
    if len(ma) >= 3 and len(wa) >= 3:
        rec["rank_biserial_effect"], rec["p_value"] = stats.rb_p(ma, wa)
    print(f"  {label}: n={len(ma)}/{len(wa)} effect={rec['rank_biserial_effect']:+.3f} p={rec['p_value']:.3g}")
    return rec



def main():
    cohort = data.parse_cohort("pre-defined confounding checks")

    st = data.read(
        cohort,
        "stable_pairs.csv",
        hint="run 04_export_stable_pairs.py first"
    )

    todo = st[st["gene"].isin(cfg.EXTRA_CONFOUNDING_CHECKS)]

    if todo.empty:
        print(
            f"[i] no stable pair involves any of the genes "
            f"{list(cfg.EXTRA_CONFOUNDING_CHECKS)}."
        )
        data.write(
            pd.DataFrame(columns=[
                "drug", "tested_gene", "stratum",
                "confounder", "pair_gene"
            ]),
            cohort,
            "prespecified_confounding_results.csv"
        )
        return

    mut, drug = data.load_cohort_matrices(cohort)
    out = []

    for _, r in todo.iterrows():
        g, d = r["gene"], r["drug"]

        confounders = cfg.EXTRA_CONFOUNDING_CHECKS[g]

        # Backward compatibility with the old string format
        if isinstance(confounders, str):
            confounders = [confounders]

        y = drug[d].dropna()
        y = y[y.index.isin(mut.index)]

        allp = set(y.index)
        gm = data.mutated_set(mut, g) & allp

        for confounder in confounders:

            if confounder not in mut.columns:
                print(
                    f"[!] {confounder} is not in the mutation matrix; "
                    f"{g} x {d} skipped."
                )
                continue

            cm = data.mutated_set(mut, confounder) & allp

            a = len(gm & cm)
            b = len(gm - cm)
            c = len(cm - gm)
            dd = len(allp - gm - cm)

            orr, fp = fisher_exact([[a, b], [c, dd]])

            print(
                f"\n{g} x {d} | confounder={confounder}: "
                f"a={a} b={b} c={c} d={dd} "
                f"OR={orr:.2f} Fisher p={fp:.3g}"
            )

            rows = [
                stratum(
                    y, mut, g, allp,
                    "overall", d
                ),
                stratum(
                    y, mut, g, allp - cm,
                    f"{confounder}-WT only", d
                ),
                stratum(
                    y, mut, g, allp & cm,
                    f"{confounder}-mutant only", d
                ),
                stratum(
                    y, mut, confounder, allp,
                    f"[{confounder} gene] overall", d
                ),
                stratum(
                    y, mut, confounder, allp - gm,
                    f"[{confounder} gene] {g}-WT only", d
                )
            ]

            for x in rows:
                x.update(
                    pair_gene=g,
                    confounder=confounder,
                    cooccurrence_OR=orr,
                    cooccurrence_fisher_p=fp
                )

            overall = rows[0]["rank_biserial_effect"]
            wt_effect = rows[1]["rank_biserial_effect"]

            if np.isfinite(overall) and np.isfinite(wt_effect):
                if np.sign(wt_effect) != np.sign(overall):
                    msg = "direction FLIPS"
                elif abs(wt_effect) < abs(overall) / 2:
                    msg = "effect more than halves"
                else:
                    msg = "same direction, similar magnitude"

                print(f"  => in {confounder}-WT: {msg}")
            else:
                print(
                    f"  => {confounder}-WT stratum too small; "
                    "descriptive only"
                )

            out.extend(rows)

    result = pd.DataFrame(out)

    data.write(
        result,
        cohort,
        "prespecified_confounding_results.csv"
    )


if __name__ == "__main__":
    main()
