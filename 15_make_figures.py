# -*- coding: utf-8 -*-
"""
15_make_figures.py  (v4: the same figures as v3, with a tidier layout and legends)
=========================================================================================
Uses only the pipeline outputs; no statistic is recomputed. PNG 300 dpi + PDF. Maximum width 7.2 in.

Changes in v4 (appearance only; data logic and numbers are untouched):
  - All figures use layout="constrained" (tick/label/title spacing is adjusted automatically).
  - Legends no longer hang below the axes with bbox_to_anchor; each one gets its own dedicated row in the gridspec,
    so it can never overlap the xlabel or the next figure.
  - Uniform panel titles: bold letter + short description (function ptitle).
  - landscape: the colorbar and the marker legend share a separate bottom strip.
  - PDF with editable fonts (fonttype 42) for journals.
Requires matplotlib >= 3.6.

Main text (6 figures):
  Fig1  screen: (a) volcano (b) selection funnel (c) sensitivity to the threshold
  Fig2  landscape: gene x drug, colour = Discovery effect, marker = Validation status   (frozen main)
  Fig3  Discovery vs Validation: (a) effect vs effect (b) status counts by gene
  Fig4  example boxplots: the best pair of each gene
  Fig5  robustness: (a) bootstrap CI of all pairs (b) raw/adjusted beta (c) concordance across the sensitivity checks
  Fig6  simulation: (a) null (b) power (c) empirical FDR
Supplement (4 figures):
  FigS1 bootstrap CI of all stable pairs | FigS2 landscape of the sensitivity pairs | FigS3 predefined confounding
  FigS4 full Validation forest plot (all evaluated pairs)
Full table of every pair (no number is lost): DISCOVERY/supp_table_per_pair_all.csv
Per-pair figures (legacy) are in 15b_make_figures_detail.py; for internal review only, not for the paper.
  python 15_make_figures.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import config_official as cfg, data

OUT = cfg.OUTPUT_ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RED, BLUE, GREY, GREEN, GOLD = "#C65D5D", "#5B8DB8", "#C9CDD3", "#2C7A57", "#D99A2B"
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 6, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelpad": 2.5, "axes.titlepad": 4, "xtick.major.pad": 2, "ytick.major.pad": 2,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
W2 = 7.2
INDEX = []
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf", "#7f7f7f", "#bcbd22"]
GENE_COL = {}

# ---- Validation status (one definition for all figures)
VCLASS = ["validated", "concordant, n.s. (powered)", "inconclusive (underpowered)", "discordant, n.s. (powered)",
          "opposite direction", "not evaluable"]
VCOL = dict(zip(VCLASS, [GREEN, BLUE, GREY, GOLD, RED, "#EEEEEE"]))


def vclass(s):
    s = str(s)
    if s == "validated":
        return VCLASS[0]
    if s.startswith("contradicted"):
        return VCLASS[4]
    if s.endswith("underpowered"):
        return VCLASS[2]
    if s.startswith("concordant"):
        return VCLASS[1]
    if s.startswith("discordant"):
        return VCLASS[3]
    return VCLASS[5]


def gcol(g):
    return GENE_COL.get(g, "#888888")


def short(s, n=26):
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + "…"


# ------------------------------------------------------------------ layout helpers
def new_fig(w, h):
    """Figure with constrained layout; pads in inches, wspace/hspace as a fraction of width/height."""
    fig = plt.figure(figsize=(w, h), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.05, h_pad=0.05, wspace=0.07, hspace=0.07)
    return fig


def ptitle(ax, letter, desc="", fs=7):
    """Uniform panel title: bold letter + description, left-aligned above the axes."""
    ax.set_title(rf"$\mathbf{{{letter}}}$" + (f"  {desc}" if desc else ""), loc="left", fontsize=fs)


def legend_ax(fig, cell, handles, labels, ncol=1, fs=5.5, **kw):
    """A dedicated row for the legend (invisible axes). The legend never overlaps the axes."""
    lax = fig.add_subplot(cell)
    lax.axis("off")
    if handles:
        lax.legend(handles, labels, loc="center", ncol=ncol, frameon=False, fontsize=fs, borderaxespad=0,
                   columnspacing=0.9, handletextpad=0.3, labelspacing=0.35, **kw)
    return lax


def gene_handles(genes):
    return [Line2D([], [], marker="o", ls="", c=gcol(g), ms=4) for g in genes], list(genes)


def save(fig, name, note=""):
    fig.savefig(OUT / f"{name}.png"); fig.savefig(OUT / f"{name}.pdf"); plt.close(fig)
    INDEX.append({"figure": name, "note": note}); print("[saved]", name)


# ------------------------------------------------------------------ Fig1
def fig1(c, m, sa, fa, t):
    fig = new_fig(W2, 3.0)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1], height_ratios=[1, .2])
    ax = fig.add_subplot(gs[0, 0])
    if sa is not None:
        ax.scatter(sa.rank_biserial_effect, -np.log10(sa.p_value), s=3, c=GREY, lw=0, label="Other pairs")
        mm = sa.merge(m[["gene", "drug", "frozen"]], on=["gene", "drug"])
        for fz, col, nm in [(False, GOLD, "Stable, not frozen"), (True, RED, "Frozen (main)")]:
            s = mm[mm.frozen == fz]
            if len(s):
                ax.scatter(s.rank_biserial_effect, -np.log10(s.p_value), s=9, c=col, edgecolor="k", lw=.3, label=nm)
        ax.set_xlabel("Rank-biserial effect\n(>0: mutant more sensitive)"); ax.set_ylabel(r"$-\log_{10}\,p$")
        h, l = ax.get_legend_handles_labels()
        legend_ax(fig, gs[1, 0], h, l, ncol=2)
    ptitle(ax, "a", "Screen")
    ax = fig.add_subplot(gs[0, 1])
    if fa is not None:
        steps = [("Candidate\npool", len(fa)), ("Nominal rule\n(G1)", int(fa.G1_in_main.sum())),
                 ("G1 ∧ stable\n(G2)", int((fa.G1_in_main & fa.G2_stable).sum())),
                 ("G1 ∧ G2 ∧\nadjusted (G3)", int(fa.is_frozen_main.sum()))]
        ax.barh(range(len(steps)), [n for _, n in steps], color=[GREY, BLUE, BLUE, RED], edgecolor="k", lw=.4)
        for i, (_, n) in enumerate(steps):
            ax.text(n, i, f" {n}", va="center", fontsize=6)
        ax.set_yticks(range(len(steps))); ax.set_yticklabels([s for s, _ in steps]); ax.invert_yaxis()
        ax.set_xlabel("Pairs"); ax.set_xlim(0, max(n for _, n in steps) * 1.25)
    ptitle(ax, "b", "Selection funnel")
    ax = fig.add_subplot(gs[0, 2])
    if t is not None:
        ax.bar([f"≥{v}" for v in t.minimum_n_per_group], t.n_feasible_pairs, color=BLUE, edgecolor="k", lw=.4)
        top = float(t.n_feasible_pairs.max())
        for i, (n, q) in enumerate(zip(t.n_feasible_pairs, t["n_FDR_below_0.10"])):
            ax.text(i, n + .02 * top, f"{n}\n(q<.1: {q})", ha="center", va="bottom", fontsize=5)
        ax.set_ylim(0, top * 1.3); ax.set_xlabel("Min. patients per group"); ax.set_ylabel("Feasible pairs")
    ptitle(ax, "c", "Threshold sensitivity")
    fig.add_subplot(gs[1, 1:]).axis("off")
    save(fig, "Fig1_screen_overview", "DISCOVERY screen; a volcano, b selection funnel, c threshold sensitivity")


# ------------------------------------------------------------------ Fig2 / FigS2 landscape
MARK = {VCLASS[0]: dict(marker="o", mfc="k", mec="k", ms=3.2), VCLASS[1]: dict(marker="o", mfc="none", mec="k", ms=3.2, mew=.7),
        VCLASS[2]: dict(marker=".", mfc="k", mec="k", ms=2.2), VCLASS[3]: dict(marker="x", mfc="k", mec="k", ms=3, mew=.7),
        VCLASS[4]: dict(marker="x", mfc="k", mec="#B00020", ms=3.6, mew=1.1), VCLASS[5]: dict(marker="_", mfc="k", mec="gray", ms=3.5, mew=.6)}


def landscape(mm, vr, name, note, per_panel=36):
    if mm is None or mm.empty:
        print(f"[i] {name}: no pairs."); return
    P = mm.pivot_table(index="drug_name", columns="gene", values="rank_biserial_effect", aggfunc="first")
    P = P[list(mm.gene.value_counts().index)]
    P = P.loc[P.mean(axis=1).sort_values(ascending=False).index]
    st = {}
    if vr is not None and len(vr) and "interpretation" in vr.columns:
        j = mm[["gene", "drug", "drug_name"]].merge(vr[["gene", "drug", "interpretation"]], on=["gene", "drug"], how="left")
        st = {(g, d): vclass(i) for g, d, i in zip(j.gene, j.drug_name, j.interpretation) if isinstance(i, str)}
    parts = [P.iloc[i:i + per_panel] for i in range(0, len(P), per_panel)]
    ncol = min(len(parts), 3); nrow = int(np.ceil(len(parts) / ncol))
    h_panel = .125 * min(per_panel, len(P)) + .5          # rows + room for gene labels
    foot = .75                                             # bottom strip: colorbar + legend
    fig = new_fig(W2, nrow * h_panel + foot)
    outer = fig.add_gridspec(2, 1, height_ratios=[nrow * h_panel, foot])
    inner = outer[0].subgridspec(nrow, ncol)
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#F4F4F4")
    im = None
    for k, part in enumerate(parts):
        ax = fig.add_subplot(inner[k // ncol, k % ncol])
        im = ax.imshow(np.ma.masked_invalid(part.values), cmap=cmap, vmin=-.7, vmax=.7, aspect="auto")
        ax.set_xticks(range(part.shape[1])); ax.set_xticklabels(part.columns, rotation=90, fontsize=5.5)
        ax.set_yticks(range(part.shape[0])); ax.set_yticklabels([short(d, 22) for d in part.index], fontsize=5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=1.5)
        for i, d in enumerate(part.index):
            for jx, g in enumerate(part.columns):
                kk = st.get((g, d))
                if kk and np.isfinite(part.values[i, jx]):
                    ax.plot(jx, i, ls="", **MARK[kk])
    foot_gs = outer[1].subgridspec(1, 2, width_ratios=[1, 1.7])
    cax_host = fig.add_subplot(foot_gs[0]); cax_host.axis("off")
    cax = cax_host.inset_axes([.05, .62, .9, .16])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Discovery rank-biserial (>0: mutant more sensitive)", fontsize=5.5, labelpad=2)
    cb.ax.tick_params(labelsize=5, length=1.5)
    present = [k for k in VCLASS if k in set(st.values())]
    h = [Line2D([], [], ls="", label=k, **MARK[k]) for k in present]
    legend_ax(fig, foot_gs[1], h, present, ncol=3)
    save(fig, name, note)


# ------------------------------------------------------------------ Fig3
def fig3(vr, mm):
    if vr is None or vr.empty or "status" not in vr.columns:
        return
    ev = vr[vr.status == "evaluated"].copy()
    if ev.empty:
        return
    ev["cls"] = ev.interpretation.map(vclass)
    fig = new_fig(W2, 3.6)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, .3])
    ax = fig.add_subplot(gs[0, 0])
    e2 = ev.dropna(subset=["discovery_effect", "validation_effect"])
    lim = 1.0
    ax.plot([-lim, lim], [-lim, lim], c="gray", lw=.6, ls="--"); ax.axhline(0, c="k", lw=.4); ax.axvline(0, c="k", lw=.4)
    for g, s in e2.groupby("gene"):
        val = s.cls == VCLASS[0]
        ax.scatter(s.discovery_effect[val], s.validation_effect[val], s=14, c=gcol(g), edgecolor="k", lw=.4)
        ax.scatter(s.discovery_effect[~val], s.validation_effect[~val], s=14, facecolor="white", edgecolor=gcol(g), lw=.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_box_aspect(1)
    ax.set_xlabel("Discovery effect (W1+2)"); ax.set_ylabel("Validation effect (W3+4)")
    conc = e2[e2.direction_concordant.astype(str) == "True"] if "direction_concordant" in e2.columns else e2.iloc[0:0]
    ax.text(.03, .97, f"same direction: {len(conc)}/{len(e2)}\nvalidated (filled): {int((e2.cls == VCLASS[0]).sum())}",
            transform=ax.transAxes, va="top", fontsize=6)
    ptitle(ax, "a", "Discovery vs Validation")
    genes = list(e2.gene.value_counts().index)
    h, l = gene_handles(genes)
    legend_ax(fig, gs[1, 0], h, l, ncol=5)
    ax = fig.add_subplot(gs[0, 1])
    ct = pd.crosstab(ev.gene, ev.cls).reindex(columns=VCLASS, fill_value=0)
    ct = ct.loc[ct.sum(axis=1).sort_values().index]
    left = np.zeros(len(ct))
    for k in VCLASS:
        ax.barh(range(len(ct)), ct[k], left=left, color=VCOL[k], edgecolor="k", lw=.3, label=k); left += ct[k].values
    ax.set_yticks(range(len(ct))); ax.set_yticklabels(ct.index); ax.set_xlabel("Evaluated pairs")
    ptitle(ax, "b", "Outcome per gene")
    h, l = ax.get_legend_handles_labels()
    legend_ax(fig, gs[1, 1], h, l, ncol=2)
    save(fig, "Fig3_validation", "a: Discovery vs Validation effect (filled = validated); b: validation outcome per gene. Pairs sharing a gene/drug are not independent.")


# ------------------------------------------------------------------ Fig4
def fig4(mm, pop, vr):
    if mm is None or mm.empty:
        return
    best = mm.sort_values("p_value").groupby("gene", sort=False).head(1)
    best = best.merge(vr[["gene", "drug", "validation_effect", "interpretation"]], on=["gene", "drug"], how="left") if vr is not None and len(vr) else best
    best = best.head(8)
    n = len(best); nc = min(4, n); nr = int(np.ceil(n / nc))
    fig = new_fig(W2, 2.3 * nr + .2)
    axs = fig.subplots(nr, nc, squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for k, (ax, (_, r)) in enumerate(zip(axs.ravel(), best.iterrows())):
        ax.axis("on")
        s_ = pop[(pop.gene == r.gene) & (pop.drug == r.drug)]
        v = [s_[s_.mutation_status == q].AUC.values for q in (0, 1)]
        ax.boxplot(v, showfliers=False, patch_artist=True, widths=.55, boxprops=dict(facecolor="#eee"), medianprops=dict(color="k"))
        for q, col in zip((1, 2), (BLUE, gcol(r.gene))):
            ax.scatter(np.random.default_rng(q).normal(q, .07, len(v[q - 1])), v[q - 1], s=3, c=col, alpha=.5, lw=0)
        ax.set_xticks([1, 2]); ax.set_xticklabels([f"WT\n(n={len(v[0])})", f"Mut\n(n={len(v[1])})"])
        if k % nc == 0:
            ax.set_ylabel("AUC")
        vtxt = ""
        if "validation_effect" in r and pd.notna(r.get("validation_effect")):
            vtxt = f"\nvalidation: {r.validation_effect:+.2f} ({vclass(r.interpretation).split(' (')[0]})"
        ax.set_title(f"{r.gene} × {short(r.drug_name, 18)}\nDisc. {r.rank_biserial_effect:+.2f}{vtxt}", fontsize=6, pad=3)
    save(fig, "Fig4_examples", "best (lowest p) frozen-main pair per gene; Discovery AUC (lower = more sensitive)")


# ------------------------------------------------------------------ Fig5 / FigS1
def caterpillar_ax(ax, d, alpha_col=None):
    d = d.sort_values("bootstrap_median_effect").reset_index(drop=True)
    y = np.arange(len(d)); cols = [gcol(g) for g in d.gene]
    al = np.where(d[alpha_col].astype(bool), 1, .35) if alpha_col else np.ones(len(d))
    for yi, lo, hi, md, cc, a in zip(y, d.bootstrap_CI_low, d.bootstrap_CI_high, d.bootstrap_median_effect, cols, al):
        ax.plot([lo, hi], [yi, yi], c=cc, lw=.6, alpha=a); ax.plot(md, yi, ".", c=cc, ms=1.8, alpha=a)
    ax.axvline(0, c="k", lw=.5, ls="--"); ax.set_yticks([]); ax.set_ylim(len(d), -1)
    ax.set_xlabel("Rank-biserial (bootstrap median, 95% CI)")


def chk_group(c):
    if c.startswith("4b_"):
        return "4b excl. predefined"
    if c.startswith("4_excl"):
        return "4 excl. confounders"
    return {"1_outlier_removed_IQR": "1 no outliers", "6_policy_mean": "6 repeats: mean", "6_policy_exclude_repeated": "6 repeats: excl.",
            "7_converged_only": "7 converged", "8_flt3_wes_only": "8 FLT3 WES-only"}.get(c)


def fig5(m, mm, adjr, se0):
    fig = new_fig(W2, 6.2)
    gs = fig.add_gridspec(3, 2, width_ratios=[.85, 1.15], height_ratios=[1, 1, .16])
    ax = fig.add_subplot(gs[0:2, 0]); caterpillar_ax(ax, mm.dropna(subset=["bootstrap_CI_low"]))
    ptitle(ax, "a", f"Bootstrap stability ({len(mm)} frozen pairs)")
    genes = list(mm.gene.value_counts().index)
    h, l = gene_handles(genes)
    legend_ax(fig, gs[2, 0], h, l, ncol=3)
    ax = fig.add_subplot(gs[0, 1])
    a_ = m.dropna(subset=["adjusted_beta"]).copy()
    a_["ub"] = adjr.reindex(list(zip(a_.gene, a_.drug))).unadjusted_beta.values
    ok = a_.adjustment_interpretation.isin(cfg.ADJ_PASS_LABELS)
    for g, s in a_.groupby("gene"):
        o = ok[s.index]
        ax.scatter(s.ub[o], s.adjusted_beta[o], s=8, c=gcol(g), lw=0)
        ax.scatter(s.ub[~o], s.adjusted_beta[~o], s=10, marker="x", c=gcol(g), lw=.6)
    lim = float(np.nanmax(np.abs(a_[["ub", "adjusted_beta"]].values))) * 1.05 if len(a_) else 1
    ax.plot([-lim, lim], [-lim, lim], c="gray", ls="--", lw=.5); ax.axhline(0, c="k", lw=.4); ax.axvline(0, c="k", lw=.4)
    ax.set_xlabel("Unadjusted β (AUC units; >0: mutant more resistant)"); ax.set_ylabel("Co-mutation-adjusted β")
    ptitle(ax, "b", "Adjustment (all stable pairs)")
    legend_ax(fig, gs[2, 1], [Line2D([], [], marker="o", ls="", c="gray", ms=3), Line2D([], [], marker="x", ls="", c="gray", ms=3.5)],
              ["passes adjustment", "does not pass"], ncol=2)
    ax = fig.add_subplot(gs[1, 1])
    s = se0.merge(mm[["gene", "drug"]], on=["gene", "drug"])
    s = s.assign(grp=s.check.map(chk_group))
    base = s[s.check == "0_baseline"].set_index(["gene", "drug"]).rank_biserial_effect
    rows = []
    for (g, d, grp), x in s[s.grp.notna()].groupby(["gene", "drug", "grp"]):
        b, e = base.get((g, d), np.nan), x.rank_biserial_effect.iloc[0]
        if np.isfinite(b) and np.isfinite(e):
            rows.append((g, grp, float(np.sign(b) == np.sign(e))))
    if rows:
        r_ = pd.DataFrame(rows, columns=["gene", "grp", "same"])
        pct = r_.pivot_table(index="gene", columns="grp", values="same", aggfunc="mean") * 100
        cnt = r_.pivot_table(index="gene", columns="grp", values="same", aggfunc="size")
        pct = pct.reindex(index=genes)
        ax.imshow(pct.values, cmap="Greens", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(pct.shape[1])); ax.set_xticklabels(pct.columns, rotation=45, ha="right", fontsize=5.5)
        ax.set_yticks(range(pct.shape[0])); ax.set_yticklabels(pct.index)
        for i in range(pct.shape[0]):
            for j in range(pct.shape[1]):
                v = pct.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0f}%\n(n={int(cnt.reindex(index=pct.index).values[i, j])})", ha="center", va="center", fontsize=4.8,
                            color="white" if v > 60 else "k")
        for sp in ax.spines.values():
            sp.set_visible(False)
    ptitle(ax, "c", "Same direction as baseline (% of pairs)", fs=6.5)
    save(fig, "Fig5_robustness", "a: internal bootstrap stability (not independent replication); b: unadjusted vs adjusted beta; c: direction consistency across sensitivity checks (FLT3 WES-only is a diluted comparison)")


def figS1(m):
    d = m.dropna(subset=["bootstrap_CI_low"]).copy()
    if d.empty:
        return
    d = d.sort_values(["frozen_main", "bootstrap_median_effect"], ascending=[False, True]).reset_index(drop=True)
    half = int(np.ceil(len(d) / 2))
    fig = new_fig(W2, 8.6)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, .05])
    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1], sharex=ax1)
    for ax, part in zip((ax1, ax2), (d.iloc[:half], d.iloc[half:])):
        caterpillar_ax(ax, part, alpha_col="frozen_main")
    h, l = gene_handles(list(d.gene.value_counts().index))
    lax = fig.add_subplot(gs[1, :]); lax.axis("off")
    lax.legend(h, l, loc="center", ncol=8, frameon=False, fontsize=5.5, columnspacing=.9, handletextpad=.3, borderaxespad=0)
    save(fig, "FigS1_all_stable_caterpillar", "all stable pairs (opaque = frozen main, faded = sensitivity-only); sorted within each half")


# ------------------------------------------------------------------ FigS3
def figS3(pc):
    if pc is None or pc.empty or "confounder" not in pc.columns:
        return
    ov = pc[pc.stratum == "overall"][["drug", "pair_gene", "confounder", "rank_biserial_effect"]].rename(columns={"rank_biserial_effect": "overall"})
    wt = pc[pc.stratum == pc.confounder + "-WT only"][["drug", "pair_gene", "confounder", "rank_biserial_effect", "n_mutated"]].rename(
        columns={"rank_biserial_effect": "wt_stratum"})
    j = ov.merge(wt, on=["drug", "pair_gene", "confounder"]).dropna(subset=["overall", "wt_stratum"])
    if j.empty:
        return
    fig = new_fig(W2 * .66, 4.4)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, .26])
    ax = fig.add_subplot(gs[0])
    cs = sorted(j.confounder.unique()); cmap = {c: PALETTE[i % 10] for i, c in enumerate(cs)}
    ax.plot([-1, 1], [-1, 1], c="gray", ls="--", lw=.5); ax.axhline(0, c="k", lw=.4); ax.axvline(0, c="k", lw=.4)
    for cn, s in j.groupby("confounder"):
        ax.scatter(s.overall, s.wt_stratum, s=8, c=cmap[cn], lw=0, label=cn)
    ax.set_box_aspect(1)
    ax.set_xlabel("Effect in all patients"); ax.set_ylabel("Effect among confounder-WT patients")
    h, l = ax.get_legend_handles_labels()
    legend_ax(fig, gs[1], h, l, ncol=4)
    save(fig, "FigS3_confounding", f"predefined confounding: effect among confounder-WT vs overall ({len(j)} gene×drug×confounder combinations). Strata with <3 patients omitted.")


# ------------------------------------------------------------------ FigS4
def figS4(vr, names):
    if vr is None or vr.empty or "status" not in vr.columns:
        return
    ev = vr[vr.status == "evaluated"].copy()
    if ev.empty:
        return
    ev["cls"] = ev.interpretation.map(vclass)
    ev["dn"] = ev.drug.map(lambda x: names.get(x, x))
    ev = ev.sort_values(["gene", "discovery_effect"]).reset_index(drop=True)
    half = int(np.ceil(len(ev) / 2))
    body, foot = .095 * half + .35, .45
    fig = new_fig(W2, body + foot + .2)
    gs = fig.add_gridspec(2, 2, height_ratios=[body, foot])
    for k, part in enumerate((ev.iloc[:half], ev.iloc[half:])):
        ax = fig.add_subplot(gs[0, k])
        y = np.arange(len(part))
        lo = (part.validation_effect - part.rank_biserial_ci_low).clip(lower=0); hi = (part.rank_biserial_ci_high - part.validation_effect).clip(lower=0)
        for yi, e, l_, h_, kk in zip(y, part.validation_effect, lo, hi, part.cls):
            ax.errorbar(e, yi, xerr=[[l_], [h_]], fmt="D", ms=2, c=VCOL[kk] if kk != VCLASS[5] else "gray", ecolor="gray", elinewidth=.5, capsize=0)
        ax.scatter(part.discovery_effect, y, marker="|", s=14, c="k", lw=.7, zorder=3)
        ax.axvline(0, c="k", lw=.4, ls="--")
        ax.set_yticks(y); ax.set_yticklabels([f"{g} × {short(d, 16)}" for g, d in zip(part.gene, part.dn)], fontsize=4.6)
        ax.set_ylim(len(part), -1); ax.set_xlim(-1, 1); ax.set_xlabel("Rank-biserial effect")
    hs = [Line2D([], [], marker="D", ls="", c=VCOL[k], ms=3) for k in VCLASS[:5]] + [Line2D([], [], marker="|", ls="", c="k", ms=5)]
    ls = list(VCLASS[:5]) + ["Discovery effect"]
    lax = fig.add_subplot(gs[1, :]); lax.axis("off")
    lax.legend(hs, ls, loc="center", ncol=3, frameon=False, fontsize=5.5, columnspacing=1.0, handletextpad=.3, borderaxespad=0)
    save(fig, "FigS4_validation_forest", f"all {len(ev)} evaluated frozen pairs: Validation effect with 95% bootstrap CI (colour = outcome), Discovery effect (tick)")


# ------------------------------------------------------------------ Fig6
def fig6():
    sp = data.path("DISCOVERY", "simulation_summary.csv")
    if not sp.exists():
        return
    s = pd.read_csv(sp); s = s[(s.planted_n_mut_bin == "all") & (s.family == "H20")]
    NUL = ["null", "null_gt_missing", "null_informative_missing"]
    NL = {"null": "Null", "null_gt_missing": "Null + genotype-dep. missingness", "null_informative_missing": "Null + informative missingness"}
    NC = dict(zip(NUL, [BLUE, GOLD, RED]))
    MM = ["MAIN_N", "MAIN_NS", "MAIN_NSA", "BH", "BY", "HOLM", "TOPK_MATCHED", "DATASPLIT"]
    MC = dict(zip(MM, PALETTE))
    fig = new_fig(W2, 3.5)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1], height_ratios=[1, .3])
    ax = fig.add_subplot(gs[0, 0]); h = .26
    for k, sc in enumerate(NUL):
        g = s[(s.scenario == sc) & s.method.isin(MM)].set_index("method").reindex(MM).dropna(subset=["P_any_declared"])
        if g.empty:
            continue
        y = np.array([MM.index(i) for i in g.index]) + (k - 1) * h
        ax.barh(y, g.P_any_declared, height=h, color=NC[sc], label=NL[sc],
                xerr=[(g.P_any_declared - g.P_any_declared_CI_low).clip(lower=0), (g.P_any_declared_CI_high - g.P_any_declared).clip(lower=0)],
                capsize=1, error_kw=dict(lw=.5))
    ax.set_yticks(range(len(MM))); ax.set_yticklabels(MM); ax.invert_yaxis(); ax.set_xlabel("P(≥1 false declaration), null")
    ptitle(ax, "a", "Null scenarios")
    h_, l_ = ax.get_legend_handles_labels()
    legend_ax(fig, gs[1, 0], h_, l_, ncol=1)
    al = s[(s.scenario == "alt") & s.method.isin(MM)]
    axs = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    for me, g in al.groupby("method"):
        axs[0].errorbar(g.delta_sd, g.power_recovery, yerr=[(g.power_recovery - g.power_CI_low).clip(lower=0), (g.power_CI_high - g.power_recovery).clip(lower=0)],
                        marker="o", ms=2.5, lw=.8, label=me, capsize=1, c=MC.get(me))
        axs[1].errorbar(g.delta_sd, g.FDR_hat, yerr=[(g.FDR_hat - g.FDR_hat_CI_low).clip(lower=0), (g.FDR_hat_CI_high - g.FDR_hat).clip(lower=0)],
                        marker="o", ms=2.5, lw=.8, capsize=1, c=MC.get(me))
    axs[0].set_xlabel("Planted shift (SD)"); axs[0].set_ylabel("Power (recovery)"); ptitle(axs[0], "b", "Power")
    axs[1].set_xlabel("Planted shift (SD)"); axs[1].set_ylabel("Empirical FDR"); ptitle(axs[1], "c", "FDR")
    hl, ll = axs[0].get_legend_handles_labels()
    legend_ax(fig, gs[1, 1:], hl, ll, ncol=4)
    save(fig, "Fig6_simulation", "family H20; Clopper–Pearson 95% Monte-Carlo CI; FDR_hat in alt depends on the number of planted signals")


# ------------------------------------------------------------------ full table
def full_table(m, se0, vr, vrs):
    t = m.copy()
    w = se0.assign(grp=se0.check.map(lambda c: c if c in ("0_baseline", "3_permutation_test") else (chk_group(c) or c)))
    w = w[w.grp != "3_permutation_test"].pivot_table(index=["gene", "drug"], columns="grp", values="rank_biserial_effect", aggfunc="first")
    w.columns = [f"sens_effect_{c}" for c in w.columns]
    t = t.merge(w.reset_index(), on=["gene", "drug"], how="left")
    extra = ["discovery_effect", "effect_EB", "n_mut_val", "n_wt_val", "validation_effect", "rank_biserial_ci_low", "rank_biserial_ci_high",
             "validation_p", "validation_FDR_q", "power_at_shrunk_effect", "MDE_rank_biserial_80pct", "power_at_observed_effect", "interpretation"]
    for nm, v in (("main", vr), ("sens", vrs)):
        if v is not None and len(v):
            keep = [c for c in extra if c in v.columns and (c not in t.columns or nm == "sens")]
            vv = v[["gene", "drug"] + keep].rename(columns={c: f"{nm}_val_{c}" for c in keep})
            t = t.merge(vv, on=["gene", "drug"], how="left")
    data.write(t, "DISCOVERY", "supp_table_per_pair_all.csv")


def main():
    c = "DISCOVERY"
    m = data.read(c, "final_master_association_table.csv", hint="run 13_master_table.py first")
    sa = data.read(c, "selection_adjusted_associations.csv", required=False)
    fa = data.read(c, "freeze_audit.csv", required=False)
    t = data.read(c, "mutation_drug_threshold_robustness.csv", required=False)
    pop = data.read(c, "analysis_population.csv")
    adjr = data.read(c, "adjusted_analysis_results.csv").set_index(["gene", "drug"])
    pc = data.read(c, "prespecified_confounding_results.csv", required=False)
    se0 = data.read(c, "sensitivity_negative_control_results.csv")
    vr = data.read("VALIDATION", "validation_report.csv", required=False)
    vrs = data.read("VALIDATION", "validation_report_sensitivity.csv", required=False)
    mm = m[m.frozen_main].copy()
    for i, g in enumerate(m.gene.value_counts().index):
        GENE_COL[g] = PALETTE[i % len(PALETTE)] if i < len(PALETTE) else "#888888"
    if fa is not None:
        data.write(pd.DataFrame([("candidates_in_pool", len(fa)), ("G1_nominal_main", int(fa.G1_in_main.sum())),
                                 ("G2_stable_any_rule", int(fa.G2_stable.sum())), ("G1_and_G2", int((fa.G1_in_main & fa.G2_stable).sum())),
                                 ("G1_G2_and_G3_frozen_main", int(fa.is_frozen_main.sum())),
                                 ("stable_not_frozen_main", int((fa.G2_stable & ~fa.is_frozen_main).sum()))], columns=["step", "n"]), c, "supp_selection_flow.csv")
    fig1(c, m, sa, fa, t)
    if mm.empty:
        print("[i] frozen main is empty; the remaining figures were not produced.")
    else:
        landscape(mm, vr, "Fig2_landscape", "frozen main: colour = Discovery effect; symbol = Validation outcome")
        fig3(vr, mm); fig4(mm, pop, vr); fig5(m, mm, adjr, se0)
    figS1(m)
    sens = m[m.frozen_sensitivity].copy()
    landscape(sens, vrs, "FigS2_landscape_sensitivity", "sensitivity-only frozen pairs (NOT the main result)")
    figS3(pc)
    names = dict(zip(m.drug, m.drug_name))
    figS4(vr, names)
    fig6()
    full_table(m, se0, vr, vrs)
    data.write(pd.DataFrame(INDEX), "shared", "figure_index.csv")


if __name__ == "__main__":
    main()
