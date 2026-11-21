# -*- coding: utf-8 -*-
"""
15b_make_figures_detail.py  (per-pair figures; internal review only. Paper figures: 15_make_figures.py)
=========================================================================================
Figures come only from the pipeline outputs (no hand-typed numbers). PNG 300 dpi + PDF.
Changes in v2:
  * No figure is wider than 7.2 in (two columns); long figures are automatically split into several figures (a, b, c ...).
  * The in-figure title (suptitle) was removed; put the "frozen_main / ALL stable" description in the caption; it is recorded in figure_index.csv.
  * Legends are placed outside the data area (below the axes).
  F2 volcano | F3 threshold | F4-F6, F8: frozen_main ; FigureS*: all stable pairs | FigureS7 confounding
  F9 Discovery vs Validation | F10a/F10b simulation
  python 15_make_figures.py
"""
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import config_official as cfg, data

OUT = cfg.OUTPUT_ROOT / "figures_detail"   # internal review only, not for the paper
OUT.mkdir(parents=True, exist_ok=True)
RED, BLUE, GREY, GREEN, GOLD = "#C65D5D", "#5B8DB8", "#C9CDD3", "#2C7A57", "#D99A2B"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 7, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
                     "axes.spines.top": False, "axes.spines.right": False})
W2 = 7.2                    # double-column width (in)
BOX_PER_FIG, BOX_COLS = 6, 3            # each boxplot figure has at most 6 panels (3x2)
ROWS_PER_FIG = 10                       # forest / adjustment / heatmap: at most 10 pairs per figure
BAR_ROWS_PER_FIG = 20                   # FigureS7
VAL_ROWS_PER_FIG = 8                    # Figure9
INDEX = []                              # figure_index.csv


def save(fig, name, note=""):
    fig.savefig(OUT / f"{name}.png"); fig.savefig(OUT / f"{name}.pdf"); plt.close(fig)
    INDEX.append({"figure": name, "note": note}); print("[saved]", name)


def lab(g, n, w=26):
    return "\n".join(textwrap.wrap(f"{g} × {n}", w)) or f"{g} × {n}"


def _suffix(i):
    s, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(97 + r) + s
    return s


def chunks(df, k):
    """Consecutive chunks of size k; if there is only one chunk, the name has no suffix."""
    parts = [df.iloc[i:i + k] for i in range(0, len(df), k)]
    return [(("" if len(parts) == 1 else _suffix(i)), p) for i, p in enumerate(parts)]


def fig_height(n_rows, per_row=0.34, base=1.1, wrap_lines=1):
    return base + n_rows * per_row * max(1, wrap_lines)


def figs(mm, pre, tag, pop, adjr, pc, se0):
    if mm.empty:
        print(f"[i] {tag}: no pair; figures were not produced."); return
    mm = mm.reset_index(drop=True)
    # ---- F4 boxplots (6 panels per figure)
    for suf, part in chunks(mm, BOX_PER_FIG):
        n = len(part); nc = min(BOX_COLS, n); nr = int(np.ceil(n / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(W2 * nc / BOX_COLS if nc < BOX_COLS else W2, 2.5 * nr), squeeze=False)
        for ax, (_, r) in zip(axs.ravel(), part.iterrows()):
            s_ = pop[(pop.gene == r.gene) & (pop.drug == r.drug)]
            v = [s_[s_.mutation_status == k].AUC.values for k in (0, 1)]
            ax.boxplot(v, showfliers=False, patch_artist=True, widths=.55, boxprops=dict(facecolor="#eee"),
                       medianprops=dict(color="k"))
            for k, col in zip((1, 2), (BLUE, RED)):
                ax.scatter(np.random.default_rng(k).normal(k, .06, len(v[k - 1])), v[k - 1], s=4, c=col, alpha=.5, lw=0)
            ax.set_xticks([1, 2]); ax.set_xticklabels([f"WT\n(n={len(v[0])})", f"Mut\n(n={len(v[1])})"])
            ax.set_ylabel("AUC")
            ax.set_title(f"{lab(r.gene, r.drug_name, 24)}\nq={r.FDR_q:.2g}, eff={r.rank_biserial_effect:+.2f}", pad=3)
        for ax in axs.ravel()[n:]:
            ax.axis("off")
        fig.tight_layout(h_pad=1.2, w_pad=1.0)
        save(fig, f"Figure{pre}4{suf}_AUC_boxplots", tag)
    # ---- F5 bootstrap forest
    mo_all = mm.sort_values("bootstrap_median_effect").reset_index(drop=True)
    for suf, mo in chunks(mo_all, ROWS_PER_FIG):
        y = np.arange(len(mo))
        fig, ax = plt.subplots(figsize=(W2 * .8, fig_height(len(mo), .38)))
        ax.errorbar(mo.bootstrap_median_effect, y, xerr=[mo.bootstrap_median_effect - mo.bootstrap_CI_low,
                    mo.bootstrap_CI_high - mo.bootstrap_median_effect], fmt="o", ms=3.5, c="k", ecolor="gray", capsize=2, lw=.8)
        ax.axvline(0, ls="--", c="k", lw=.7)
        ax.set_yticks(y); ax.set_yticklabels([lab(g, d, 30) for g, d in zip(mo.gene, mo.drug_name)])
        ax.set_ylim(-.6, len(mo) - .4)
        ax.set_xlabel("Rank-biserial effect (bootstrap median, 95% CI)\ninternal stability, not independent replication")
        fig.tight_layout(); save(fig, f"Figure{pre}5{suf}_bootstrap_forest", tag)
    # ---- F6 adjustment
    a_ = mm.dropna(subset=["adjusted_beta"]).reset_index(drop=True)
    for suf, part in chunks(a_, ROWS_PER_FIG):
        ub = adjr.loc[list(zip(part.gene, part.drug))].unadjusted_beta.values
        y = np.arange(len(part))
        fig, ax = plt.subplots(figsize=(W2 * .8, fig_height(len(part), .38, base=1.5)))
        ax.scatter(ub, y, c="lightgray", edgecolor="k", lw=.5, s=22, label="Unadjusted")
        ok = part.adjustment_interpretation.isin(cfg.ADJ_PASS_LABELS)
        ax.scatter(part.adjusted_beta[ok], y[ok.values], marker="D", c=GREEN, edgecolor="k", lw=.5, s=22, label="Adjusted, passes")
        ax.scatter(part.adjusted_beta[~ok], y[(~ok).values], marker="D", c=RED, edgecolor="k", lw=.5, s=22, label="Adjusted, does not pass")
        ax.axvline(0, ls="--", c="k", lw=.7)
        ax.set_yticks(y); ax.set_yticklabels([lab(g, d, 30) for g, d in zip(part.gene, part.drug_name)])
        ax.set_ylim(-.6, len(part) - .4)
        ax.set_xlabel("OLS β on AUC (β > 0: mutant more resistant)")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.22 if len(part) > 3 else -.4), ncol=3, columnspacing=1)
        fig.tight_layout(); save(fig, f"Figure{pre}6{suf}_adjustment", tag)
    # ---- FigureS7 (supplement only)
    if pre == "S" and pc is not None and len(pc):
        for suf, part in chunks(pc.reset_index(drop=True), BAR_ROWS_PER_FIG):
            y = np.arange(len(part))
            fig, ax = plt.subplots(figsize=(W2 * .8, fig_height(len(part), .24)))
            ax.barh(y, part.rank_biserial_effect, color=BLUE, edgecolor="k", lw=.4)
            ax.set_yticks(y); ax.set_yticklabels([f"{g}/{s_}" for g, s_ in zip(part.pair_gene, part.stratum)], fontsize=6)
            ax.invert_yaxis(); ax.axvline(0, c="k", lw=.7); ax.set_xlabel("Rank-biserial effect")
            fig.tight_layout(); save(fig, f"FigureS7{suf}_predefined_confounding", tag)
    # ---- F8 heatmap
    se = se0.merge(mm[["gene", "drug"]], on=["gene", "drug"])
    if len(se):
        se = se.assign(pair=[f"{g} × {d}" for g, d in zip(se.gene, se.drug)],
                       chk=se.check.str.replace(r"^4_excl_.*", "4_excl_confounders", regex=True))
        hm_all = se.pivot_table(index="pair", columns="chk", values="rank_biserial_effect", aggfunc="first")
        hm_all = hm_all.loc[[p for p in dict.fromkeys(f"{g} × {d}" for g, d in zip(mm.gene, mm.drug)) if p in hm_all.index]]
        for suf, part in chunks(hm_all.reset_index(), ROWS_PER_FIG):
            hm = part.set_index("pair")
            fig, ax = plt.subplots(figsize=(min(W2, .62 * hm.shape[1] + 2.6), fig_height(len(hm), .32, base=1.6)))
            im = ax.imshow(hm.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(hm.shape[1])); ax.set_xticklabels(hm.columns, rotation=45, ha="right", fontsize=6)
            ax.set_yticks(range(hm.shape[0])); ax.set_yticklabels(hm.index, fontsize=6.5)
            for i in range(hm.shape[0]):
                for j in range(hm.shape[1]):
                    if np.isfinite(hm.values[i, j]):
                        ax.text(j, i, f"{hm.values[i, j]:+.2f}", ha="center", va="center", fontsize=5.5)
            cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.02); cb.set_label("Rank-biserial", fontsize=7)
            for s in ax.spines.values():
                s.set_visible(False)
            fig.tight_layout(); save(fig, f"Figure{pre}8{suf}_sensitivity_heatmap", tag)


def main():
    c = "DISCOVERY"
    m = data.read(c, "final_master_association_table.csv", hint="run 13_master_table.py first")
    sa = data.read(c, "selection_adjusted_associations.csv", required=False)
    if sa is not None:
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.scatter(sa.rank_biserial_effect, -np.log10(sa.p_value), s=4, c=GREY, lw=0, label="Other pairs")
        mm = sa.merge(m[["gene", "drug", "frozen"]], on=["gene", "drug"])
        for fz, col, nm in [(False, GOLD, "Stable, not frozen"), (True, RED, "Frozen (main)")]:
            s = mm[mm.frozen == fz]
            if len(s):
                ax.scatter(s.rank_biserial_effect, -np.log10(s.p_value), s=22, c=col, edgecolor="k", lw=.5, label=nm)
        ax.set_xlabel("Rank-biserial effect (> 0: mutant more sensitive)"); ax.set_ylabel(r"$-\log_{10}\,p$")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=3, columnspacing=.8, handletextpad=.2)
        fig.tight_layout(); save(fig, "Figure2_screen_overview", f"DISCOVERY screen, m={len(sa)}")
    t = data.read(c, "mutation_drug_threshold_robustness.csv", required=False)
    if t is not None:
        fig, ax = plt.subplots(figsize=(3.8, 3.0))
        ax.bar([f"≥{v}" for v in t.minimum_n_per_group], t.n_feasible_pairs, color=BLUE, edgecolor="k", lw=.5)
        top = float(t.n_feasible_pairs.max()) if len(t) else 1
        for i, (n, q) in enumerate(zip(t.n_feasible_pairs, t["n_FDR_below_0.10"])):
            ax.text(i, n + .02 * top, f"{n}\nFDR<0.10: {q}", ha="center", va="bottom", fontsize=6.5)
        ax.set_ylim(0, top * 1.25); ax.set_xlabel("Minimum patients per group"); ax.set_ylabel("Feasible pairs")
        fig.tight_layout(); save(fig, "Figure3_threshold")
    if m.empty:
        print("[i] no stable pair; the remaining figures were not produced."); data.write(pd.DataFrame(INDEX), "shared", "figure_index_detail.csv"); return
    pop = data.read(c, "analysis_population.csv")
    adjr = data.read(c, "adjusted_analysis_results.csv").set_index(["gene", "drug"])
    pc = data.read(c, "prespecified_confounding_results.csv", required=False)
    se0 = data.read(c, "sensitivity_negative_control_results.csv")
    fa = data.read(c, "freeze_audit.csv", required=False)
    if fa is not None:
        flow = pd.DataFrame([("candidates_in_pool", len(fa)), ("G1_nominal_main", int(fa.G1_in_main.sum())),
                             ("G2_stable_any_rule", int(fa.G2_stable.sum())), ("G1_and_G2", int((fa.G1_in_main & fa.G2_stable).sum())),
                             ("G1_G2_and_G3_frozen_main", int(fa.is_frozen_main.sum())),
                             ("stable_not_frozen_main (sensitivity/exploratory)", int((fa.G2_stable & ~fa.is_frozen_main).sum()))],
                            columns=["step", "n"])
        data.write(flow, c, "supp_selection_flow.csv")
    figs(m[m.frozen_main], "", "frozen_main (selection rule v3)", pop, adjr, pc, se0)
    figs(m, "S", "ALL stable pairs: exploratory/sensitivity, NOT the main-rule output", pop, adjr, pc, se0)

    vr = data.read("VALIDATION", "validation_report.csv", required=False)
    if vr is not None and len(vr) and "status" in vr.columns:
        ev_all = vr[vr.status == "evaluated"].reset_index(drop=True)
        for suf, ev in chunks(ev_all, VAL_ROWS_PER_FIG):
            y = np.arange(len(ev))
            fig, ax = plt.subplots(figsize=(W2, fig_height(len(ev), .5, base=1.7)))
            ax.scatter(ev.discovery_effect, y + .15, c="lightgray", edgecolor="k", lw=.5, s=20, label="Discovery (observed)")
            ax.scatter(ev.effect_EB, y, marker="s", c=GOLD, edgecolor="k", lw=.5, s=20, label="Discovery (EB-shrunk, exploratory)")
            lo = (ev.validation_effect - ev.rank_biserial_ci_low).clip(lower=0)
            hi = (ev.rank_biserial_ci_high - ev.validation_effect).clip(lower=0)
            ax.errorbar(ev.validation_effect, y - .15, xerr=[lo, hi], fmt="D", ms=3.5, c=RED, ecolor="gray", capsize=2, lw=.8,
                        label="Validation (95% CI)")
            for yi, (_, r) in zip(y, ev.iterrows()):
                pw = r.power_at_shrunk_effect
                ax.text(1.04, yi, f"power={pw:.2f}\nn={int(r.n_mut_val)}/{int(r.n_wt_val)}" if np.isfinite(pw)
                        else f"power=NA\nn={int(r.n_mut_val)}/{int(r.n_wt_val)}", va="center", fontsize=6)
            ax.axvline(0, ls="--", c="k", lw=.7); ax.set_xlim(-1, 1.35); ax.set_ylim(-.6, len(ev) - .4)
            ax.set_yticks(y); ax.set_yticklabels([lab(g, cc, 30) for g, cc in zip(ev.gene, ev.drug)])
            ax.set_xlabel("Rank-biserial effect")
            ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.45, -.2 if len(ev) > 2 else -.45), ncol=3, columnspacing=1)
            fig.tight_layout(); save(fig, f"Figure9{suf}_discovery_vs_validation", "power = at EB-shrunk effect")
    sp = data.path(c, "simulation_summary.csv")
    if sp.exists():
        s = pd.read_csv(sp); s = s[(s.planted_n_mut_bin == "all") & (s.family == "H20")]
        NUL = ["null", "null_gt_missing", "null_informative_missing"]
        NUL_LAB = {"null": "Null", "null_gt_missing": "Null + genotype-dependent missingness",
                   "null_informative_missing": "Null + informative missingness (artefact)"}
        MM = ["MAIN_N", "MAIN_NS", "MAIN_NSA", "BH", "BY", "HOLM", "TOPK_MATCHED", "DATASPLIT"]
        fig, ax = plt.subplots(figsize=(W2 * .75, 3.6)); h = .26
        for k, sc in enumerate(NUL):
            g = s[(s.scenario == sc) & s.method.isin(MM)].set_index("method").reindex(MM).dropna(subset=["P_any_declared"])
            if g.empty:
                continue
            y = np.array([MM.index(i) for i in g.index]) + (k - 1) * h
            ax.barh(y, g.P_any_declared, height=h, label=NUL_LAB[sc],
                    xerr=[(g.P_any_declared - g.P_any_declared_CI_low).clip(lower=0), (g.P_any_declared_CI_high - g.P_any_declared).clip(lower=0)],
                    capsize=1.5, error_kw=dict(lw=.6))
        ax.set_yticks(range(len(MM))); ax.set_yticklabels(MM); ax.invert_yaxis()
        ax.set_xlabel("P(≥1 declared) under null (Clopper–Pearson 95% CI), family H20")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.4, -.18), ncol=1)
        fig.tight_layout(); save(fig, "Figure10a_simulation_null")
        al = s[(s.scenario == "alt") & s.method.isin(MM)]
        if len(al):
            fig, axs = plt.subplots(1, 2, figsize=(W2, 3.0))
            for me, g in al.groupby("method"):
                axs[0].errorbar(g.delta_sd, g.power_recovery, yerr=[(g.power_recovery - g.power_CI_low).clip(lower=0), (g.power_CI_high - g.power_recovery).clip(lower=0)],
                                marker="o", ms=3, lw=.9, label=me, capsize=1.5)
                axs[1].errorbar(g.delta_sd, g.FDR_hat, yerr=[(g.FDR_hat - g.FDR_hat_CI_low).clip(lower=0), (g.FDR_hat_CI_high - g.FDR_hat).clip(lower=0)],
                                marker="o", ms=3, lw=.9, capsize=1.5)
            axs[0].set_xlabel("Planted shift (SD)"); axs[0].set_ylabel("Power = S/m₁")
            axs[1].set_xlabel("Planted shift (SD)"); axs[1].set_ylabel("Empirical FDR (mean FDP, 95% MC CI)")
            h_, l_ = axs[0].get_legend_handles_labels()
            fig.legend(h_, l_, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.12))
            fig.tight_layout(); save(fig, "Figure10b_simulation_alt")
    data.write(pd.DataFrame(INDEX), "shared", "figure_index_detail.csv")


if __name__ == "__main__":
    main()
