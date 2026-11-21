# -*- coding: utf-8 -*-
"""
16_simulation_benchmark.py   (v4: shared H20 family + no Simes/FDP bound)
=========================================================================================
Design: the simulation runs on the REAL Discovery drugs and genotypes; only the gene->AUC link is broken
(rows of the drug matrix are permuted jointly; G and the co-mutation structure stay fixed). A drug used in the simulation is NOT necessarily outside
the real candidates; what is planted is a "gene-drug pair", and it is logged in simulation_planted_log.csv.

Two comparisons (column family):
  H20          : the main comparison. Shared family H20 = {(g,d): n_mut>=SIM_FAMILY_MIN_N, n_wt>=SIM_FAMILY_MIN_N} (within the same replicate).
                 All methods (MAIN_*, NOMINAL_STRICT, BH, BY, HOLM, TOPK_MATCHED*, DATASPLIT, BHGATE_*, [WY]) run on H20 only.
                 The planted pair is always chosen from H20 (eligibility differences do not enter the comparison).
  operational  : supplementary analysis. Each comparator on its own predefined family (n>=MIN_MUTATED_WITH_AUC), + LEGACY_*.
Top-K:
  TOPK_MATCHED   : K = number declared by the MAIN_NSA rule IN THE SAME replicate (derived from that replicate's data, not the number of planted signals).
  TOPK_MATCHED_N : K = number declared by MAIN_N of the same replicate. K is logged in column K_used. K=0 => nothing declared.
  TOPK (operational): fixed K = SIM_TOP_K.
MAIN rule (the same code as common/; the v3 rule is unchanged):  MAIN_N (p<NOMINAL_P_MAX AND n>=MAIN_MIN_N) -> MAIN_NS (+stable) -> MAIN_NSA (+adjustment robust)
  MAIN_NSA_PERMISSIVE / MAIN_EFFECT / BHGATE* / LEGACY* = sensitivity; never merged with MAIN_*.
Scenarios: global_null | null_gt_missing | null_informative_missing (data artefact) | alt (one planted pair, delta*SD)
  | multi_alt_m<M> (optional, --multi-m1 10 20 30): M planted pairs with co-mutation structure; outputs go to simulation_multi_*.csv.
  (The old name "null" was renamed "global_null": pandas turns the string "null" into NaN in read_csv.)
multi_alt: n_clusters "driver" genes (default 3), each on several distinct drugs (each drug carries at most one pair); the real G and co-mutation stay fixed.
  --multi-unlinked: independent drivers. Default linked: drivers 2..k are chosen among genes co-occurring with the first driver (Fisher p<0.05, OR>1), if available.
  Effect sizes: delta from cfg.SIM_DELTAS_SD is spread in a balanced random way across pairs. Power = S/m1 in each replicate (fraction); CI = mean +/- z*SE over replicates.
  FDR_hat in multi_alt is also "empirical"; V = declared pairs that are not planted (including genes co-occurring with a driver on the planted drug).

Definitions (per replicate):
  R=#declared , V=#false (declared and not a planted pair; in null*: every declared) , S=#true recovered (planted pair declared and effect>0)
  FDP = V/max(R,1)   (one replicate)       FDR_hat = mean_b FDP_b  (mean over R_mc replicates; an "empirical" FDR estimate)
  Power = S/m1 (m1=1 planted pair per alt replicate) ; estimate = mean over replicates.
  wrong_direction: the planted pair is declared but effect<=0 (not counted in V; reported separately).
  Monte Carlo: Power and P(any) with Clopper-Pearson; FDR_hat with mean +/- z*SE (clipped to [0,1]). Zero events in R_mc replicates is NOT "rate=0".
  These are "empirical performance in the simulation"; no guaranteed FDR/FDP control is claimed.

  python 16_simulation_benchmark.py --cohort DISCOVERY [--quick] [--r-null 500 --r-alt 300 --r-null-missing 300] [--with-wy]
  python 16_simulation_benchmark.py --cohort DISCOVERY --only-multi --multi-m1 10 20 30 --r-multi 100   # multi-pair scenario only; the main outputs are untouched
Sign: effect>0 => mutant AUC lower; planting = subtract delta*SD from the mutants' AUC => effect>0 (recovery: effect>0). n_comutation_associated_false is a descriptive count only (Fisher p<0.05), not proof of confounding.
DATASPLIT: A selects top-K; B is used for testing only. heldout_signed_effect_B_all_testable = all A-selected pairs evaluable in B (no B significance condition); pair-wise values + CI in simulation_datasplit_pairs.csv.
R_mc = completed replicates; rejected replicates and the reason are in simulation_run_status.csv (R_requested in the summary).
Outputs (multi: simulation_multi_{replicates,summary,run_status,planted_log,winners_curse,pair_power}.csv): simulation_run_status.csv, simulation_datasplit_pairs.csv, simulation_replicates.csv, simulation_summary.csv, simulation_planted_log.csv, simulation_winners_curse.csv
"""
import argparse
import time
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, beta as beta_dist, norm
from statsmodels.stats.multitest import multipletests
from common import config_official as cfg, data, screen, stats, selection, adjust

# H20 must have exactly the same size as the main rule; otherwise MAIN_* is no longer a subset of H20 and the method comparison becomes unfair.
assert cfg.SIM_FAMILY_MIN_N == cfg.MAIN_MIN_N_PER_GROUP, (
    "Simulation H20 family must use the same minimum group size as the main analysis "
    f"(SIM_FAMILY_MIN_N={cfg.SIM_FAMILY_MIN_N} vs MAIN_MIN_N_PER_GROUP={cfg.MAIN_MIN_N_PER_GROUP}).")


def grid(a, shape, key):
    g = np.full(shape, np.nan)
    g[a["gi"], a["di"]] = a[key]
    return g


def apply_missing(Dr, G, rng, informative):
    """Missingness only inside the mutation-positive stratum of one gene (random; WT untouched) for a fraction of drugs (in-place). informative: only mutants with AUC above the median are dropped.
    Returns: (gene, affected drugs)."""
    nd = Dr.shape[1]
    J = rng.choice(nd, max(1, int(cfg.SIM_MISS_DRUG_FRAC * nd)), replace=False)
    pool = np.where(G.sum(0) >= 2 * cfg.MIN_MUTATED)[0]
    pool = pool if len(pool) else np.arange(G.shape[1])
    g = int(rng.choice(pool))
    mrows = G[:, g] == 1
    for j in J:
        col = Dr[:, j]
        if informative:
            with np.errstate(invalid="ignore"):
                cand = mrows & (col > np.nanmedian(col))
        else:
            cand = mrows
        Dr[cand & (rng.random(len(col)) < cfg.SIM_MISS_DROP_P), j] = np.nan
    return g, J


def cp_ci(x, level=None):
    """Clopper-Pearson CI for a proportion (x: 0/1 series)."""
    level = cfg.SIM_MC_CI_LEVEL if level is None else level
    k, n = int(np.sum(x)), len(x)
    al = 1 - level
    lo = 0.0 if k == 0 else float(beta_dist.ppf(al / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - al / 2, k + 1, n - k))
    return lo, hi


def mean_ci(x, level=None):
    """mean +/- z*SE (clipped to [0,1]) for FDR_hat = mean FDP."""
    level = cfg.SIM_MC_CI_LEVEL if level is None else level
    x = np.asarray(x, float)
    mu = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan
    z = norm.ppf(1 - (1 - level) / 2)
    return mu, se, max(0.0, mu - z * se), min(1.0, mu + z * se)


def metrics_row(name, family, mk, planted, effect, m_all, m_h20, comut=0, K_used=np.nan, n_mut_planted=np.nan):
    """V,R,S for one method in one replicate. planted: None, one index, or a list of indices of the planted pairs (multi_alt)."""
    pls = [] if planted is None else ([int(planted)] if np.isscalar(planted) else [int(x) for x in planted])
    m1, pset = len(pls), set(pls)
    decl = np.where(mk)[0]
    false = [i for i in decl if i not in pset]
    decl_pl = [i for i in pls if bool(mk[i])]
    right = [i for i in decl_pl if effect[i] > 0]
    den = max(m1, 1)
    cast = (lambda x: int(x)) if m1 <= 1 else (lambda x: x / den)      # m1<=1: same integer (0/1) output as before
    return {"method": name, "family": family, "n_eligible_pairs": m_all, "n_H20": m_h20,
            "n_declared": len(decl), "n_false": len(false), "any_declared": int(len(decl) > 0),
            "any_false": int(len(false) > 0), "fdp": len(false) / max(len(decl), 1),
            "n_comutation_associated_false": comut, "K_used": K_used,
            "hit": cast(len(right)), "planted_declared": cast(len(decl_pl)),
            "wrong_direction_recovery": cast(len(decl_pl) - len(right)),
            "planted_not_recovered": cast(m1 - len(decl_pl)),
            "n_planted": m1, "S_recovered": len(right),
            "planted_n_mut": n_mut_planted}


def _comutated(G, g1, g2):
    a_, b_ = G[:, g1] == 1, G[:, g2] == 1
    orr, p = fisher_exact([[int((a_ & b_).sum()), int((a_ & ~b_).sum())], [int((~a_ & b_).sum()), int((~a_ & ~b_).sum())]])
    return bool(p < 0.05 and orr > 1)


def choose_multi_pairs(a, G, h20, m1, rng, n_clusters, linked):
    """m1 pairs from H20: n_clusters driver genes, each on several distinct drugs. Returns (indices, cluster id) or None if impossible."""
    gi, di = a["gi"], a["di"]
    hidx = np.where(h20)[0]
    per = [m1 // n_clusters + (1 if k < m1 % n_clusters else 0) for k in range(n_clusters)]
    used_d, drivers, chosen, cl = set(), [], [], []
    for k, need in enumerate(per):
        if need == 0:
            continue
        avail = [int(i) for i in hidx if di[i] not in used_d and int(gi[i]) not in drivers]
        cnt = {}
        for i in avail:
            cnt[int(gi[i])] = cnt.get(int(gi[i]), 0) + 1
        elig = [g for g, c in cnt.items() if c >= need]
        if not elig:
            return None
        if linked and drivers:
            lk = [g for g in elig if _comutated(G, drivers[0], g)]
            elig = lk or elig
        es = set(elig)
        pool = [i for i in avail if int(gi[i]) in es]              # weight by the number of H20 pairs of each gene (as in single-pair planting)
        g = int(gi[pool[int(rng.integers(len(pool)))]])
        cand = [i for i in avail if int(gi[i]) == g]
        take = [int(x) for x in rng.choice(cand, need, replace=False)]
        drivers.append(g)
        used_d |= {di[i] for i in take}
        chosen += take
        cl += [k] * need
    return chosen, np.array(cl)


def run_replicate(D, G, genes, drugs, idx, mutdf, rng, delta=None, with_wy=False, miss_mode="none", multi=None, record_ds=True):
    n = D.shape[0]
    Nmin = cfg.SIM_FAMILY_MIN_N
    Dr = D[rng.permutation(n)].copy()
    miss_gene, n_miss_drugs = None, 0
    if miss_mode != "none":
        mg, J = apply_missing(Dr, G, rng, informative=(miss_mode == "informative"))
        miss_gene, n_miss_drugs = genes[mg], len(J)
    a = screen.screen_arrays(Dr, G)
    if len(a["p"]) < 30:
        return ("skip", "lt30_eligible_pairs")
    h20 = (a["n1"] >= Nmin) & (a["n2"] >= Nmin)
    if h20.sum() < 30:
        return ("skip", "lt30_H20_pairs")
    pl, pls, pl_info = None, [], None
    if multi is not None:                       # multi_alt: m1 pairs, co-mutation drivers, distinct drugs; all from H20
        pick = choose_multi_pairs(a, G, h20, multi["m1"], rng, multi["n_clusters"], multi["linked"])
        if pick is None:
            return ("skip", "lt_m1_plantable_pairs")
        pls, cl_id = pick
        dls = np.resize(np.asarray(cfg.SIM_DELTAS_SD, float), len(pls))
        rng.shuffle(dls)
        for k_, dk in zip(pls, dls):
            g, j = a["gi"][k_], a["di"][k_]
            rows = (G[:, g] == 1) & ~np.isnan(Dr[:, j])
            Dr[rows, j] -= dk * np.nanstd(Dr[:, j])
        pl_info = {"cluster": cl_id, "delta": dls}
        a = screen.screen_arrays(Dr, G)
    elif delta is not None:                     # the planted pair comes from H20 only
        pl = int(rng.choice(np.where(h20)[0]))
        pls = [pl]
        g, j = a["gi"][pl], a["di"][pl]
        rows = (G[:, g] == 1) & ~np.isnan(Dr[:, j])
        Dr[rows, j] -= delta * np.nanstd(Dr[:, j])
        a = screen.screen_arrays(Dr, G)         # pair order stays the same (n and NaN unchanged)
    plset = set(pls)
    m, shape = len(a["p"]), (G.shape[1], Dr.shape[1])
    p, rb, gi, di = a["p"], a["rb"], a["gi"], a["di"]
    q, K0 = cfg.SIM_COMPARATOR_Q, cfg.SIM_TOP_K
    op = np.ones(m, bool)                       # operational family: all eligible with n>=MIN_*_WITH_AUC

    def on(sel, fn):
        mk = np.zeros(m, bool)
        ii = np.where(sel)[0]
        if len(ii):
            mk[ii] = fn(ii)
        return mk

    def topk(sel, K):
        return on(sel, lambda ii: np.isin(np.arange(len(ii)), np.argsort(p[ii], kind="stable")[:max(int(K), 0)]))

    def mt(sel, meth):
        return on(sel, lambda ii: multipletests(p[ii], method=meth)[1] < q)

    # ---- data splitting (A: top-K selection, B: BH test + same sign)
    perm = rng.permutation(n)
    A_, B_ = perm[: n // 2], perm[n // 2:]
    aA = screen.screen_arrays(Dr[A_], G[A_], min_drug=cfg.MIN_DRUG_SAMPLES // 2, min_mut_auc=5, min_wt_auc=5)
    aB = screen.screen_arrays(Dr[B_], G[B_], min_drug=cfg.MIN_DRUG_SAMPLES // 2, min_mut_auc=3, min_wt_auc=3)
    pA, rA, pB, rBg = grid(aA, shape, "p"), grid(aA, shape, "rb"), grid(aB, shape, "p"), grid(aB, shape, "rb")
    pa, ra, pb, rbb = pA[gi, di], rA[gi, di], pB[gi, di], rBg[gi, di]
    okA = np.isfinite(pa)

    def datasplit(sel, K, record=False):
        pool = np.where(okA & sel)[0]
        selA = pool[np.argsort(pa[pool], kind="stable")[:K]]
        ds = np.zeros(m, bool)
        recs = []
        wc = {"naive_abs_effect_A": np.nan, "heldout_signed_effect_B_all_testable": np.nan, "heldout_signed_effect_B_declared_only": np.nan,
              "eb_shrunk_abs_effect_A_exploratory": np.nan,
              "n_ds_selected": int(len(selA)), "n_ds_testable": int(np.isfinite(pb[selA]).sum()) if len(selA) else 0}
        if len(selA) and np.isfinite(pb[selA]).any():
            s2 = selA[np.isfinite(pb[selA])]               # all A-selected pairs evaluable in B (no B significance condition)
            qq = multipletests(pb[s2], method="fdr_bh")[1]
            ok2 = (qq < q) & (np.sign(rbb[s2]) == np.sign(ra[s2]))
            ds[s2[ok2]] = True
            fac, _, _ = stats.eb_shrink(stats.signed_z(ra[okA], pa[okA]))
            fmap = np.full(m, np.nan)
            fmap[np.where(okA)[0]] = fac
            sg = np.sign(ra[s2]) * rbb[s2]
            wc.update(naive_abs_effect_A=float(np.mean(np.abs(ra[s2]))), heldout_signed_effect_B_all_testable=float(np.mean(sg)),
                      heldout_signed_effect_B_declared_only=float(np.mean(sg[ok2])) if ok2.any() else np.nan,
                      eb_shrunk_abs_effect_A_exploratory=float(np.mean(np.abs(ra[s2]) * np.nan_to_num(fmap[s2]))))
            if record:                                     # per-pair B effect + bootstrap CI (B is used for testing only, not for selection)
                for k_, i in enumerate(s2):
                    col = Dr[B_, di[i]]
                    okc = ~np.isnan(col)
                    ism = G[B_, gi[i]] == 1
                    ma_, wa_ = col[okc & ism], col[okc & ~ism]
                    lo_, hi_ = stats.bootstrap_ci(ma_, wa_, cfg.SIM_N_BOOTSTRAP, int(rng.integers(1 << 31))) if len(ma_) >= 3 and len(wa_) >= 3 else (np.nan, np.nan)
                    recs.append({"gene": genes[gi[i]], "drug": drugs[di[i]], "is_planted": bool(i in plset),
                                 "effect_A": float(ra[i]), "p_A": float(pa[i]), "effect_B": float(rbb[i]), "p_B": float(pb[i]),
                                 "effect_B_CI_low": lo_, "effect_B_CI_high": hi_, "same_sign_AB": bool(np.sign(rbb[i]) == np.sign(ra[i])),
                                 "declared_by_split": bool(ok2[k_])})
        return ds, wc, recs

    ds_h20, wc, ds_recs = datasplit(h20, K0, record_ds)
    fam = {"H20": {"NOMINAL_STRICT": h20 & (p < cfg.SIM_NOMINAL_P), "BH": mt(h20, "fdr_bh"), "BY": mt(h20, "fdr_by"),
                   "HOLM": mt(h20, "holm"), "DATASPLIT": ds_h20},
           "operational": {"NOMINAL_STRICT": p < cfg.SIM_NOMINAL_P, "TOPK": topk(op, min(K0, m)), "BH": mt(op, "fdr_bh"),
                           "BY": mt(op, "fdr_by"), "HOLM": mt(op, "holm"), "DATASPLIT": datasplit(op, K0)[0]}}
    if with_wy:
        ii = np.where(h20)[0]
        padj, _ = screen.westfall_young_minp(Dr, G, gi[ii], di[ii], p[ii], 50, int(rng.integers(1 << 31)))
        w = np.zeros(m, bool)
        w[ii] = padj < q
        fam["H20"]["WY"] = w

    # ---- pipeline rule (shared code in common/; stability and adjustment with cache)
    df = pd.DataFrame({"gene": genes[gi], "drug": drugs[di], "n_mutated_with_AUC": a["n1"], "n_wt_with_AUC": a["n2"], "p_value": p})
    pos = {(g_, d_): i for i, (g_, d_) in enumerate(zip(df["gene"], df["drug"]))}
    cache = {}

    def ev(i):
        if i not in cache:
            col = Dr[:, di[i]]
            ok = ~np.isnan(col)
            ism = G[:, gi[i]] == 1
            s = selection.stability_row(col[ok & ism], col[ok & ~ism], cfg.SIM_N_BOOTSTRAP, cfg.SIM_N_SPLIT_HALF,
                                        seed=int(rng.integers(1 << 31)))
            cache[i] = {"stable": all(selection.is_stable(s)), "eff": selection.passes_effect_precision(s, a["n1"][i], a["n2"][i]),
                        "lab": None, "col": col}
        return cache[i]

    def adj_label(i):
        e = ev(i)
        if e["lab"] is None:
            y = pd.Series(e["col"], index=idx).dropna()
            e["lab"] = adjust.adjust_pair(genes[gi[i]], y, mutdf)["interpretation"]
        return e["lab"]

    z = lambda: np.zeros(m, bool)
    mN, mNS, mNSA, mNSP, mNSE = z(), z(), z(), z(), z()
    for pr in selection.main_candidates(df):               # n>=MAIN_MIN_N ⇒ ⊂ H20 (SIM_FAMILY_MIN_N = MAIN_MIN_N)
        i = pos[pr]
        mN[i] = True
        if not ev(i)["stable"]:
            continue
        mNS[i] = True
        lab = adj_label(i)
        mNSA[i] = lab in cfg.ADJ_PASS_LABELS
        mNSP[i] = lab in cfg.ADJ_PASS_LABELS_PERMISSIVE
        mNSE[i] = bool(mNSA[i] and ev(i)["eff"])
    fam["H20"].update({"MAIN_N": mN, "MAIN_NS": mNS, "MAIN_NSA": mNSA, "MAIN_NSA_PERMISSIVE": mNSP, "MAIN_EFFECT": mNSE})
    Ks = {"TOPK_MATCHED": int(mNSA.sum()), "TOPK_MATCHED_N": int(mN.sum())}   # K from the declared count of the same replicate
    fam["H20"]["TOPK_MATCHED"], fam["H20"]["TOPK_MATCHED_N"] = topk(h20, Ks["TOPK_MATCHED"]), topk(h20, Ks["TOPK_MATCHED_N"])
    fam["H20"]["DATASPLIT_MATCHED"] = datasplit(h20, Ks["TOPK_MATCHED"])[0]      # K = |MAIN_NSA| of the same replicate (size-matched comparison)

    # sensitivity: BH gate over H20 (q is computed over H20)
    dfh = df[h20].copy()
    dfh["FDR_q"] = multipletests(dfh["p_value"], method="fdr_bh")[1]
    mB, mBS, mBSA, mBSP = z(), z(), z(), z()
    for pr in selection.bh_gate_candidates(dfh):
        i = pos[pr]
        mB[i] = True
        if not ev(i)["stable"]:
            continue
        mBS[i] = True
        lab = adj_label(i)
        mBSA[i] = lab in cfg.ADJ_PASS_LABELS
        mBSP[i] = lab in cfg.ADJ_PASS_LABELS_PERMISSIVE
    fam["H20"].update({"BHGATE": mB, "BHGATE_S": mBS, "BHGATE_SA": mBSA, "BHGATE_SA_PERMISSIVE": mBSP})

    # operational sensitivity: legacy threshold-stable Top-N rule (own family: n>=10/20/30)
    for N in cfg.SENSITIVITY_TOP_N:
        always, _ = selection.threshold_stable(df, top_n=N)
        mT, mTS, mTF = z(), z(), z()
        for pr in always:
            i = pos[pr]
            mT[i] = True
            if ev(i)["stable"]:
                mTS[i] = True
                mTF[i] = adj_label(i) in cfg.ADJ_PASS_LABELS_PERMISSIVE
        if N == cfg.TOP_N_FOR_OVERLAP:
            fam["operational"]["LEGACY_T"], fam["operational"]["LEGACY_TS"] = mT, mTS
        fam["operational"][f"LEGACY_FINAL_TOP{N}"] = mTF

    # ---- metrics
    out = []
    for fname, masks in fam.items():
        for name, mk in masks.items():
            comut = 0
            for i in (np.where(mk)[0] if pls else []):
                if i in plset:
                    continue
                for k_ in pls:
                    if di[i] == di[k_]:
                        gm, cm = G[:, gi[k_]] == 1, G[:, gi[i]] == 1
                        if fisher_exact([[int((gm & cm).sum()), int((gm & ~cm).sum())],
                                         [int((~gm & cm).sum()), int((~gm & ~cm).sum())]])[1] < 0.05:
                            comut += 1
                            break
            out.append(metrics_row(name, fname, mk, pls if multi is not None else pl, rb, m, int(h20.sum()), comut,
                                   Ks.get(name, K0 if name == "TOPK" else np.nan),
                                   int(a["n1"][pl]) if pl is not None else np.nan))
    if multi is None:
        log = {"planted_gene": genes[gi[pl]] if pl is not None else None, "planted_drug": drugs[di[pl]] if pl is not None else None,
               "planted_n_mut": int(a["n1"][pl]) if pl is not None else np.nan, "planted_n_wt": int(a["n2"][pl]) if pl is not None else np.nan,
               "planted_in_H20": bool(h20[pl]) if pl is not None else None, "n_planted_pairs": int(pl is not None)}
    else:
        log = {"planted_gene": ";".join(genes[gi[k_]] for k_ in pls), "planted_drug": ";".join(drugs[di[k_]] for k_ in pls),
               "planted_n_mut": float(np.median(a["n1"][pls])), "planted_n_wt": float(np.median(a["n2"][pls])),
               "planted_in_H20": bool(h20[pls].all()), "n_planted_pairs": len(pls),
               "n_driver_genes": int(len(set(gi[pls]))), "planted_deltas": ";".join(f"{x:g}" for x in pl_info["delta"])}
    log.update({"n_null_pairs_H20": int(h20.sum()) - len(pls), "n_eligible_pairs": m, "n_H20": int(h20.sum()),
                "missingness_gene": miss_gene, "n_missingness_drugs": n_miss_drugs,
                "K_matched_NSA": Ks["TOPK_MATCHED"], "K_matched_N": Ks["TOPK_MATCHED_N"]})
    if multi is not None:                      # effect of each planted pair by method (for power as a function of delta and n_mut)
        log["pair_detail"] = [{"family": "H20", "method": name, "planted_gene": genes[gi[k_]], "planted_drug": drugs[di[k_]],
                               "cluster": int(c_), "delta_sd": float(d_), "n_mut": int(a["n1"][k_]), "declared": bool(mk[k_]),
                               "right_direction": bool(mk[k_] and rb[k_] > 0)}
                              for name, mk in fam["H20"].items() for k_, c_, d_ in zip(pls, pl_info["cluster"], pl_info["delta"])]
    return out, wc, log, ds_recs


SCEN_NOTE = {"global_null": "global null: G fixed (co-mutation preserved), drug rows permuted (gene->AUC link broken)",
             "null_gt_missing": "null + random missingness restricted to the mutation-positive stratum of one gene (depends on genotype only; NOT MCAR overall; gene->AUC link still null)",
             "null_informative_missing": "DATA ARTEFACT: missingness depends on genotype and AUC; declared != intrinsic error of the method",
             "alt": "one planted gene-drug pair with a delta*SD shift (m1=1)",
             "multi_alt": "m1 planted pairs; co-mutation drivers; delta mixed from cfg.SIM_DELTAS_SD; distinct drugs"}


def _is_multi(sc):
    return str(sc).startswith("multi_alt")


def summarize(rep):
    bins = pd.cut(rep["planted_n_mut"], [0, 39, 79, 1e9], labels=["20-39", "40-79", "80+"])
    rows = []
    for (fm, sc, dl, me), g in rep.groupby(["family", "scenario", "delta_sd", "method"], dropna=False):
        for b in ["all", "20-39", "40-79", "80+"]:
            s = g if b == "all" else g[bins.loc[g.index] == b]
            if len(s) == 0:
                continue
            multi = _is_multi(sc)
            alt = sc == "alt" or multi
            fdr, fdr_se, fdr_lo, fdr_hi = mean_ci(s.fdp)
            dec = s[s.n_declared > 0]
            rows.append({"family": fm, "scenario": sc, "delta_sd": dl, "method": me, "planted_n_mut_bin": b, "R_mc": len(s),
                         "scenario_note": SCEN_NOTE.get(sc, SCEN_NOTE["multi_alt"] if _is_multi(sc) else ""),
                         "P_any_declared": s.any_declared.mean(), "P_any_declared_CI_low": cp_ci(s.any_declared)[0],
                         "P_any_declared_CI_high": cp_ci(s.any_declared)[1],
                         "P_any_false": s.any_false.mean(), "P_any_false_CI_low": cp_ci(s.any_false)[0],
                         "P_any_false_CI_high": cp_ci(s.any_false)[1],
                         "FDR_hat": fdr, "FDR_hat_SE": fdr_se, "FDR_hat_CI_low": fdr_lo, "FDR_hat_CI_high": fdr_hi,
                         "mean_FDP_given_declared": dec.fdp.mean() if len(dec) else np.nan,
                         "mean_n_declared": s.n_declared.mean(), "mean_n_false": s.n_false.mean(),
                         "mean_K_used": s.K_used.mean(),
                         "mean_n_comutation_associated_false": s.n_comutation_associated_false.mean(),
                         "power_recovery": s.hit.mean() if alt else np.nan,
                         "power_CI_low": (mean_ci(s.hit)[2] if multi else cp_ci(s.hit)[0]) if alt else np.nan,
                         "power_CI_high": (mean_ci(s.hit)[3] if multi else cp_ci(s.hit)[1]) if alt else np.nan,
                         "n_planted": s.n_planted.mean(), "mean_S_recovered": s.S_recovered.mean() if alt else np.nan,
                         "wrong_direction_rate": s.wrong_direction_recovery.mean() if alt else np.nan,
                         "not_recovered_rate": s.planted_not_recovered.mean() if alt else np.nan})
    return pd.DataFrame(rows)


def _flags_from_discovery(log):
    """Descriptive only (does not affect selection): was the planted pair/drug among the real stable/frozen pairs of Discovery? Answers the "four drugs" ambiguity.
    In multi_alt the columns planted_gene/planted_drug are ';'-separated lists; any = at least one, n_* = counts."""
    st = data.read("DISCOVERY", "stable_pairs.csv", required=False)
    fz = data.read("DISCOVERY", "frozen_candidate_list.csv", required=False)
    sp = set(zip(st.gene, st.drug)) if st is not None and len(st) else set()
    sd = {d for _, d in sp}
    fp = set(zip(fz.gene, fz.drug)) if fz is not None and len(fz) else set()
    fd = {d for _, d in fp}
    sp_l = lambda x: [] if x is None or x != x else str(x).split(";")
    prs = [list(zip(sp_l(g), sp_l(d))) for g, d in zip(log.planted_gene, log.planted_drug)]
    log["planted_pair_is_real_stable"] = [any(x in sp for x in p) for p in prs]
    log["planted_drug_in_real_stable_drugs"] = [any(d in sd for _, d in p) for p in prs]
    log["planted_pair_is_real_frozen"] = [any(x in fp for x in p) for p in prs]
    log["planted_drug_in_real_frozen_drugs"] = [any(d in fd for _, d in p) for p in prs]
    log["n_planted_pairs_real_stable"] = [sum(x in sp for x in p) for p in prs]
    log["n_planted_pairs_real_frozen"] = [sum(x in fp for x in p) for p in prs]
    return log


def _execute(scen, rng, ctx, with_wy):
    """scen: [(name, miss_mode, delta, R, multi_kw|None)] -> (rep, status, wcs, logs, ds_pairs, pair_detail)."""
    D, G, genes, drugs, idx, mutdf = ctx
    rows, wcs, logs, dsp, pdet, status, t0 = [], [], [], [], [], [], time.time()
    for sc, mm, dl, R, mk in scen:
        skipped = {}
        for r in range(R):
            res = run_replicate(D, G, genes, drugs, idx, mutdf, rng, None if sc != "alt" else dl, with_wy, mm,
                                multi=mk, record_ds=mk is None)
            if res[0] == "skip":
                skipped[res[1]] = skipped.get(res[1], 0) + 1
                continue
            out, wc, lg, recs = res
            det = lg.pop("pair_detail", None)
            for o in out:
                rows.append({"scenario": sc, "delta_sd": dl, "rep": r, **o})
            wcs.append({"scenario": sc, "delta_sd": dl, "rep": r, **wc})
            logs.append({"scenario": sc, "delta_sd": dl, "rep": r, **lg})
            dsp += [{"scenario": sc, "delta_sd": dl, "rep": r, **x} for x in recs]
            if det:
                pdet += [{"scenario": sc, "rep": r, **x} for x in det]
        status.append({"scenario": sc, "delta_sd": dl, "R_requested": R, "R_completed": R - sum(skipped.values()),
                       "R_skipped": sum(skipped.values()), "skip_reasons": ";".join(f"{k}:{v}" for k, v in skipped.items())})
        print(f"[{sc} delta={dl}] done, {time.time() - t0:.0f}s, skipped={sum(skipped.values())}/{R}")
    return pd.DataFrame(rows), pd.DataFrame(status), wcs, logs, dsp, pdet


def _wc_table(wcs):
    return pd.DataFrame(wcs).groupby(["scenario", "delta_sd"], dropna=False)[
        ["naive_abs_effect_A", "heldout_signed_effect_B_all_testable", "heldout_signed_effect_B_declared_only",
         "eb_shrunk_abs_effect_A_exploratory", "n_ds_selected", "n_ds_testable"]].mean().reset_index()


def _requested(summ, st):
    rq = {(x.scenario, x.delta_sd if x.delta_sd == x.delta_sd else -1): x.R_requested for x in st.itertuples()}
    summ["R_requested"] = [rq.get((sc, d if d == d else -1)) for sc, d in zip(summ.scenario, summ.delta_sd)]
    return summ


def pair_power_table(pdet):
    """Pair-wise power in multi_alt: by delta and n_mut group; CI = mean +/- z*SE over replicates (pairs within one replicate are not independent)."""
    d = pd.DataFrame(pdet)
    d["n_mut_bin"] = pd.cut(d["n_mut"], [0, 39, 79, 1e9], labels=["20-39", "40-79", "80+"]).astype(str)
    out = []
    for var in ("delta_sd", "n_mut_bin"):
        per = d.groupby(["scenario", "family", "method", var, "rep"])["right_direction"].mean().reset_index()
        for (sc, fm, me, lv), g in per.groupby(["scenario", "family", "method", var]):
            mu, se, lo, hi = mean_ci(g["right_direction"])
            out.append({"scenario": sc, "family": fm, "method": me, "stratum_var": var, "stratum": str(lv), "R_mc": len(g),
                        "power_recovery": mu, "power_SE": se, "power_CI_low": lo, "power_CI_high": hi})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="DISCOVERY", choices=["DISCOVERY", "FULL"])
    ap.add_argument("--r-null", type=int, default=cfg.SIM_R_NULL)
    ap.add_argument("--r-alt", type=int, default=cfg.SIM_R_ALT)
    ap.add_argument("--r-null-missing", type=int, default=cfg.SIM_R_NULL_MISSING)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--with-wy", action="store_true")
    ap.add_argument("--multi-m1", type=int, nargs="*", default=[], help="number of planted pairs, e.g. 10 20 30")
    ap.add_argument("--r-multi", type=int, default=100)
    ap.add_argument("--multi-clusters", type=int, default=3)
    ap.add_argument("--multi-unlinked", action="store_true", help="independent drivers (no selection based on co-mutation)")
    ap.add_argument("--only-multi", action="store_true", help="multi_alt only; the main simulation_*.csv files are left untouched")
    a = ap.parse_args()
    if a.only_multi and not a.multi_m1:
        a.multi_m1 = [10, 20, 30]
    if a.quick:
        a.r_null, a.r_alt, a.r_null_missing, a.r_multi = 15, 10, 8, 5
    mut, drug = data.load_cohort_matrices(a.cohort)
    D, G, genes, drugs, idx = screen.prepare(drug, mut)
    ctx = (D, G, genes, drugs, idx, mut.loc[idx])
    if not a.only_multi:
        rng = np.random.default_rng(cfg.RANDOM_SEED)
        scen = [("global_null", "none", np.nan, a.r_null, None), ("null_gt_missing", "gt", np.nan, a.r_null_missing, None),
                ("null_informative_missing", "informative", np.nan, a.r_null_missing, None)] + \
               [("alt", "none", d, a.r_alt, None) for d in cfg.SIM_DELTAS_SD]
        rep, st, wcs, logs, dsp, _ = _execute(scen, rng, ctx, a.with_wy)
        data.write(rep, a.cohort, "simulation_replicates.csv")
        data.write(st, a.cohort, "simulation_run_status.csv")
        summ = _requested(summarize(rep), st)
        data.write(pd.DataFrame(dsp), a.cohort, "simulation_datasplit_pairs.csv")
        data.write(summ, a.cohort, "simulation_summary.csv")
        data.write(_flags_from_discovery(pd.DataFrame(logs)), a.cohort, "simulation_planted_log.csv")
        wc = _wc_table(wcs)
        data.write(wc, a.cohort, "simulation_winners_curse.csv")
        show = summ[(summ.planted_n_mut_bin == "all") & (summ.family == "H20")]
        for sc in ["global_null", "null_gt_missing", "null_informative_missing"]:
            print(f"\n[H20] {sc} (every declared = false):")
            print(show[show.scenario == sc][["method", "R_mc", "P_any_declared", "P_any_declared_CI_low", "P_any_declared_CI_high",
                                             "FDR_hat", "mean_n_false", "mean_K_used"]].round(3).to_string(index=False))
        print("\n[H20] ALT (planted):")
        print(show[show.scenario == "alt"][["delta_sd", "method", "power_recovery", "power_CI_low", "power_CI_high", "FDR_hat",
                                            "FDR_hat_CI_low", "FDR_hat_CI_high", "wrong_direction_rate", "mean_K_used"]].round(3).to_string(index=False))
        print("\nnaive vs held-out vs EB-shrunk (exploratory; not selection-adjusted):\n", wc.round(3).to_string(index=False))
    if a.multi_m1:
        rngm = np.random.default_rng([int(cfg.RANDOM_SEED), 4242])     # independent of the main scenario runs
        mk = lambda m1: {"m1": m1, "n_clusters": a.multi_clusters, "linked": not a.multi_unlinked}
        scen = [(f"multi_alt_m{m1}", "none", np.nan, a.r_multi, mk(m1)) for m1 in a.multi_m1]
        rep, st, wcs, logs, _, pdet = _execute(scen, rngm, ctx, False)
        data.write(rep, a.cohort, "simulation_multi_replicates.csv")
        data.write(st, a.cohort, "simulation_multi_run_status.csv")
        summ = _requested(summarize(rep), st)
        data.write(summ, a.cohort, "simulation_multi_summary.csv")
        data.write(_flags_from_discovery(pd.DataFrame(logs)), a.cohort, "simulation_multi_planted_log.csv")
        data.write(_wc_table(wcs), a.cohort, "simulation_multi_winners_curse.csv")
        if pdet:
            data.write(pair_power_table(pdet), a.cohort, "simulation_multi_pair_power.csv")
        show = summ[(summ.planted_n_mut_bin == "all") & (summ.family == "H20")]
        print("\n[H20] MULTI_ALT (planted; mixed delta; the empirical FDR_hat is not comparable with real data unless the structure is similar):")
        print(show[["scenario", "method", "R_mc", "n_planted", "power_recovery", "power_CI_low", "power_CI_high", "FDR_hat",
                    "FDR_hat_CI_low", "FDR_hat_CI_high", "mean_n_false", "mean_K_used"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

