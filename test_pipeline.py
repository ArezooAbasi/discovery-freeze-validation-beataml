# -*- coding: utf-8 -*-
"""test_pipeline.py -- unit/consistency tests without real data (synthetic data). Run:  python test_pipeline.py"""
import importlib.util, os, re, sys, tempfile, pathlib, traceback
os.environ["BEATAML_SYNTHETIC"] = "1"
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from common import config_official as cfg, stats, adjust, selection, screen, data


def load_script(fname):
    spec = importlib.util.spec_from_file_location("m_" + re.sub(r"\W", "_", fname), HERE / fname)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_mwu_vs_scipy():
    rng = np.random.default_rng(1)
    for ties in (False, True):
        v = rng.normal(size=60); v = np.round(v, 1) if ties else v
        g = (rng.random(60) < .4).astype(float)
        _, _, _, p, rb = stats.mwu_columns(v, g[:, None])
        ref = mannwhitneyu(v[g == 1], v[g == 0], alternative="two-sided", method="asymptotic")
        assert abs(p[0] - ref.pvalue) < 1e-9 and abs(rb[0] - stats.rank_biserial(v[g == 1], v[g == 0])) < 1e-12
    assert stats.rank_biserial(rng.normal(100, 10, 30), rng.normal(130, 10, 30)) > 0


def test_simes_removed():
    assert not hasattr(stats, "simes_fdp_bound")
    assert not hasattr(cfg, "SIM_FDP_ALPHA") and not hasattr(cfg, "SIM_FDP_GAMMA")


def test_sim_metrics_definitions():
    sim = load_script("16_simulation_benchmark.py")
    mk = np.array([1, 1, 0, 1, 0], bool)
    r = sim.metrics_row("X", "H20", mk, planted=1, effect=np.array([0, .5, 0, 0, 0.]), m_all=5, m_h20=5)
    assert (r["n_declared"], r["n_false"], r["hit"]) == (3, 2, 1) and abs(r["fdp"] - 2 / 3) < 1e-12
    r0 = sim.metrics_row("X", "H20", np.zeros(5, bool), planted=None, effect=np.zeros(5), m_all=5, m_h20=5)
    assert r0["fdp"] == 0 and r0["any_false"] == 0
    lo, hi = sim.cp_ci(np.zeros(500)); assert lo == 0 and 0.005 < hi < 0.01


def test_sim_h20_assert():
    """SIM_FAMILY_MIN_N must equal MAIN_MIN_N_PER_GROUP; a mismatch => AssertionError when loading 16."""
    load_script("16_simulation_benchmark.py")                       # current values: must load without error
    old = cfg.SIM_FAMILY_MIN_N
    try:
        cfg.SIM_FAMILY_MIN_N = cfg.MAIN_MIN_N_PER_GROUP + 1
        try:
            load_script("16_simulation_benchmark.py")
        except AssertionError as e:
            assert "H20" in str(e)
        else:
            raise RuntimeError("assert H20 did not run (is python running with -O?)")
    finally:
        cfg.SIM_FAMILY_MIN_N = old


def test_sim_planted_sign_positive():
    """Planting (lowering the mutant AUC) must give rank-biserial>0; recovery is consistent with effect>0."""
    rng = np.random.default_rng(0)
    v = rng.normal(150, 40, 200); g = (rng.random(200) < .3).astype(float)
    v2 = v.copy(); v2[g == 1] -= 0.9 * v.std()
    _, _, _, p, rb = stats.mwu_columns(v2, g[:, None])
    assert rb[0] > 0 and p[0] < 0.05


def test_sim_quick_status_and_pairs():
    sim = load_script("16_simulation_benchmark.py")
    mut, drug = data.load_cohort_matrices("DISCOVERY")
    D, G, genes, drugs, idx = screen.prepare(drug, mut)
    r = sim.run_replicate(D, G, genes, drugs, idx, mut.loc[idx], np.random.default_rng(1), delta=1.2)
    assert r[0] != "skip"
    out, wc, log, recs = r
    assert {x["method"] for x in out if x["family"] == "H20"} >= {"DATASPLIT_MATCHED", "TOPK_MATCHED", "BH"}
    assert log["planted_in_H20"] and "heldout_signed_effect_B_all_testable" in wc and len(recs) == wc["n_ds_testable"]


def test_classify_and_status_mapping():
    assert adjust.classify(1, .01, -1, .01, -200).startswith("sign_flip")
    assert adjust.classify(1, .01, .4, .01, -60).startswith("attenuated")
    assert adjust.classify(1, .01, .9, .2, -10).startswith("loses")
    assert adjust.classify(1, .01, .9, .02, -10) == "robust_to_adjustment"
    labels = ["robust_to_adjustment", "no_confounder_candidate_found", "insufficient_n_for_adjustment", "unadjusted_model_not_estimable",
              "adjusted_model_not_estimable", "sign_flip_after_adjustment_likely_confounded",
              "attenuated_>=50pct_possible_partial_confounding", "loses_nominal_significance_after_adjustment"]
    exp = ["robust_to_adjustment", "no_confounder_candidate_found", "adjustment_not_assessable", "adjustment_not_assessable",
           "adjustment_not_assessable", "attenuated_or_not_robust", "attenuated_or_not_robust", "attenuated_or_not_robust"]
    assert [adjust.coarse_status(x) for x in labels] == exp
    assert cfg.ADJ_PASS_LABELS == {"robust_to_adjustment"}                       # only the first status passes
    assert cfg.ADJ_PASS_LABELS < cfg.ADJ_PASS_LABELS_PERMISSIVE
    # test "all possible outputs": main pass => status robust
    assert all((x in cfg.ADJ_PASS_LABELS) == (adjust.coarse_status(x) == "robust_to_adjustment") for x in labels)


def test_adj_p_source_switch():
    mut, drug = data.load_cohort_matrices("DISCOVERY")
    y = drug["Midostaurin"].dropna()
    old = cfg.ADJ_P_SOURCE
    try:
        cfg.ADJ_P_SOURCE = "classic"; r1 = adjust.adjust_pair("FLT3", y, mut)
        cfg.ADJ_P_SOURCE = "HC3"; r2 = adjust.adjust_pair("FLT3", y, mut)
    finally:
        cfg.ADJ_P_SOURCE = old
    for k in ("unadjusted_beta", "adjusted_beta", "adjusted_p", "adjusted_p_HC3", "confounders_included"):
        a, b = r1[k], r2[k]
        assert (a == b) or (isinstance(a, float) and np.isnan(a) and np.isnan(b)), k
    assert "adjustment_status" in r1 and r1["adjustment_status"] == adjust.coarse_status(r1["interpretation"])


def test_main_rule_nominal_no_topn_null_friendly():
    rng = np.random.default_rng(3)
    n = 800
    base = pd.DataFrame({"gene": [f"g{i}" for i in range(n)], "drug": "d", "n_mutated_with_AUC": 40, "n_wt_with_AUC": 100})
    # if every p is large: zero candidates (no forced Top-N)
    df = base.assign(p_value=rng.uniform(.3, 1, n)); df["FDR_q"] = stats.fdr(df["p_value"])
    assert selection.main_candidates(df) == [] and selection.bh_gate_candidates(df) == []
    # 7 very small p: exactly 7; not 20; independent of N
    p = rng.uniform(.3, 1, n); p[:7] = 1e-9
    df = base.assign(p_value=p); df["FDR_q"] = stats.fdr(p)
    assert len(selection.main_candidates(df)) == 7
    # n threshold: the same small p but small n => excluded
    assert selection.main_candidates(df.assign(n_mutated_with_AUC=cfg.MAIN_MIN_N_PER_GROUP - 1)) == []
    # BH gate independent of the main rule: p=0.004 (below 0.01) with a high q is in main only
    q = np.full(n, .5); df2 = base.assign(p_value=np.r_[.004, np.full(n - 1, .9)], FDR_q=q)
    assert len(selection.main_candidates(df2)) == 1 and selection.bh_gate_candidates(df2) == []


def test_effect_rule():
    r = {"bootstrap_CI_low": .3, "bootstrap_CI_high": .6}
    assert selection.passes_effect_precision(r, 20, 20)
    assert not selection.passes_effect_precision({"bootstrap_CI_low": -.1, "bootstrap_CI_high": .9}, 20, 20)   # large effect but wide CI / CI covers zero
    assert not selection.passes_effect_precision({"bootstrap_CI_low": .05, "bootstrap_CI_high": .9}, 20, 20)
    assert not selection.passes_effect_precision(r, 10, 20)
    assert selection.conservative_effect({"bootstrap_CI_low": -.9, "bootstrap_CI_high": -.3}) == .3            # the CI end closest to zero
    assert selection.conservative_effect({**r, "observed_effect": -.4}) == 0.0                                 # the CI sign does not match the effect
    assert selection.conservative_effect({**r, "observed_effect": .4}) == .3


def test_stability_definitions_and_denominator():
    rng = np.random.default_rng(5)
    s = selection.stability_row(rng.normal(80, 20, 25), rng.normal(130, 20, 60), B=300, n_splits=100, seed=1)
    assert s["observed_effect"] > 0 and s["n_boot_draws"] == 300 and s["n_split_half_attempted"] == 100
    assert 0 <= s["pct_split_half_match_full"] <= s["pct_split_half_valid"] <= 100       # denominator = all splits
    assert all(selection.is_stable(s))
    s2 = dict(s, observed_effect=-s["observed_effect"])                                  # the CI does not share the sign of the effect => rejected
    assert not selection.is_stable(s2)[0]
    assert not selection.is_stable(dict(s, pct_split_half_match_full=50.0))[1]
    n = selection.stability_row(rng.normal(100, 20, 25), rng.normal(100, 20, 60), B=300, n_splits=100, seed=1)
    assert not all(selection.is_stable(n))                                               # noise: does not pass
    tiny = selection.stability_row(rng.normal(80, 20, 6), rng.normal(130, 20, 60), B=100, n_splits=50, seed=1)
    assert tiny["n_split_half_valid"] == 0 and not selection.is_stable(tiny)[1]          # invalid split => failure


def test_freeze_binding_and_tagging():
    h0 = data.params_sha256()
    old = cfg.NOMINAL_P_MAX
    try:
        cfg.NOMINAL_P_MAX = 0.05
        assert data.params_sha256() != h0                                                # changing a threshold => the hash changes
    finally:
        cfg.NOMINAL_P_MAX = old
    assert data.params_sha256() == h0
    assert "NOMINAL_P_MAX" in cfg.FREEZE_PARAMS and all(hasattr(cfg, k) for k in cfg.FREEZE_PARAMS)


def _run(fname, *argv):
    old = sys.argv
    sys.argv = [fname, *argv]
    try:
        load_script(fname).main()
    finally:
        sys.argv = old


def test_smoke_end_to_end_synthetic():
    """02->03->04->08->08b->13 on synthetic data; freeze_audit holds all gates (including rejected ones); main/sensitivity are separate."""
    keep = {k: getattr(cfg, k) for k in ("OUTPUT_ROOT", "BOOTSTRAP_B", "N_SPLIT_HALF", "N_WY_PERMUTATIONS")}
    with tempfile.TemporaryDirectory() as td:
        cfg.OUTPUT_ROOT = pathlib.Path(td)
        cfg.BOOTSTRAP_B, cfg.N_SPLIT_HALF, cfg.N_WY_PERMUTATIONS = 200, 100, 20
        try:
            mut, drug = data.load_cohort_matrices("DISCOVERY")
            res = screen.screen_matrices(drug, mut)
            res["FDR_q"] = stats.fdr(res["p_value"])
            res["fdr_family"] = f"all evaluable pairs (m={len(res)})"
            data.write(res.sort_values("p_value"), "DISCOVERY", "mutation_drug_associations.csv")
            for f in ("02_threshold_robustness.py", "03_stability_checks.py", "04_export_stable_pairs.py",
                      "08_adjusted_analysis.py", "08b_freeze_candidates.py", "13_master_table.py"):
                _run(f, "--cohort", "DISCOVERY") if f != "08b_freeze_candidates.py" else _run(f)
            D = lambda n: pd.read_csv(data.path("DISCOVERY", n))
            cand, aud, st, fa = D("candidate_pairs.csv"), D("stability_gate_audit.csv"), D("stable_pairs.csv"), D("freeze_audit.csv")
            fz, fs, mt = D("frozen_candidate_list.csv"), D("frozen_sensitivity_list.csv"), D("final_master_association_table.csv")
            assert len(cand) == len(aud) == len(fa) and cand.in_main.any()
            assert set(zip(st.gene, st.drug)) <= set(zip(aud.gene, aud.drug))
            assert not (set(zip(fz.gene, fz.drug)) & set(zip(fs.gene, fs.drug)))
            assert fz.in_main.all() and (fz.p_value < cfg.NOMINAL_P_MAX).all()
            assert fz.adjustment_interpretation.isin(cfg.ADJ_PASS_LABELS).all()
            assert (fa.loc[~fa.G1_in_main, "first_failed_gate"] == "G1_not_nominal_main").all()
            assert (fa.is_frozen_main == (fa.G1_in_main & fa.G2_stable & fa.G3_adjustment_pass)).all()
            assert data.path("DISCOVERY", "freeze_manifest.json").exists()
            assert (mt.frozen_main == mt.tier.eq("frozen_pending_validation")).all()
            assert mt.loc[mt.frozen_sensitivity, "tier"].str.startswith("sensitivity_").all()
        finally:
            for k, v in keep.items():
                setattr(cfg, k, v)


def test_family_consistency_01_vs_screen():
    m01 = load_script("01_association_screen.py")
    mut, drug = data.load_cohort_matrices("DISCOVERY")          # the same intersect that the corrected 01 applies
    qg, qd = m01.eligible_genes_and_drugs(mut, drug)
    rows = {}
    for g in qg[:25]:
        for d in qd:
            base, ma, wa = m01.evaluate_pair(g, d, mut, drug)
            if len(ma) < cfg.MIN_MUTATED_WITH_AUC or len(wa) < cfg.MIN_WT_WITH_AUC:
                continue
            rows[(g, d)] = stats.test_pair(ma, wa)["p_value"]
    sm = screen.screen_matrices(drug, mut)
    sm = sm[sm.gene.isin(qg[:25])]
    assert set(rows) == set(zip(sm.gene, sm.drug))
    assert max(abs(rows[(g, d)] - p) for g, d, p in zip(sm.gene, sm.drug, sm.p_value)) < 1e-8


def test_validation_not_used_for_selection():
    for f in ["02_threshold_robustness.py", "03_stability_checks.py", "04_export_stable_pairs.py", "08_adjusted_analysis.py",
              "08b_freeze_candidates.py", "09_sensitivity_negative_control.py", "10_prespecified_confounding.py", "11_analysis_population.py"]:
        t = (HERE / f).read_text(encoding="utf-8")
        assert not re.search(r"(read|path|load_cohort_matrices)\(\s*[\"']VALIDATION", t), f
    for f in ["common/selection.py", "common/adjust.py", "16_simulation_benchmark.py"]:
        assert "VALIDATION" not in (HERE / f).read_text(encoding="utf-8"), f


def test_output_isolation_and_freeze_guard():
    old = cfg.OUTPUT_ROOT
    with tempfile.TemporaryDirectory() as td:
        cfg.OUTPUT_ROOT = pathlib.Path(td)
        try:
            assert data.tagged("validation_results.csv", "sensitivity") == "validation_results_sensitivity.csv"
            assert data.tagged("run_metadata.json", None) == "run_metadata.json"
            assert data.path("VALIDATION", "validation_results.csv") != data.path("VALIDATION", data.tagged("validation_results.csv", "sensitivity"))
            assert data.path("DISCOVERY", "x.csv") != data.path("VALIDATION", "x.csv")
            assert not data.validation_started()
            data.path("VALIDATION", "validation_results_secondary_fwer.csv").write_text("a\n1\n")
            assert data.validation_started()                      # any validation result => freeze locked
        finally:
            cfg.OUTPUT_ROOT = old


def test_master_main_and_sensitivity_independent():
    m13 = load_script("13_master_table.py")
    k = ["gene", "drug"]
    st = pd.DataFrame({"gene": ["A", "B", "C"], "drug": "d", "in_main": [True, True, False], "in_top20": [True, False, True]})
    assoc = pd.DataFrame({"gene": ["A", "B", "C"], "drug": "d", "p_value": [.001, .002, .003], "FDR_q": [.01, .02, .03], "rank_biserial_effect": [.5] * 3})
    boot = pd.DataFrame({**{c: 0.0 for c in ["bootstrap_median_effect", "bootstrap_CI_low", "bootstrap_CI_high", "pct_split_half_match_full"]}, "gene": ["A", "B", "C"], "drug": "d"})
    adj = pd.DataFrame({"gene": ["A", "B", "C"], "drug": "d", "confounders_included": "", "adjusted_beta": 0.0, "adjusted_p": .1,
                        "interpretation": ["robust_to_adjustment", "no_confounder_candidate_found", "robust_to_adjustment"],
                        "adjustment_status": ["robust_to_adjustment", "no_confounder_candidate_found", "robust_to_adjustment"]})
    sa = pd.DataFrame({"gene": ["A", "B", "C"], "drug": "d", "p_WY": .1, "effect_EB": .2, "local_fdr": .3})
    fz = pd.DataFrame({"gene": ["A"], "drug": "d"}); fs = pd.DataFrame({"gene": ["B", "C"], "drug": "d"})
    m = m13.assemble(st, assoc, boot, adj, sa, fz, fs, None, None, {}).set_index("gene")
    assert list(m.loc[["A", "B", "C"], "adjustment_pass_main"]) == [True, False, True]
    assert list(m.loc[["A", "B", "C"], "adjustment_pass_permissive"]) == [True, True, True]
    assert list(m.loc[["A", "B", "C"], "frozen_main"]) == [True, False, False]
    assert list(m.loc[["A", "B", "C"], "frozen_sensitivity"]) == [False, True, True]
    assert m.loc["A", "tier"] == "frozen_pending_validation" and m.loc["B", "tier"].startswith("sensitivity_")


def test_flt3_composite():
    """Composite FLT3 = WES OR clinical ITD; NaN is not positive; an unknown value => error; only the FLT3 column changes; ITD only for profiled patients."""
    from common import io_official as io
    old = (cfg.FLT3_ITD_POSITIVE_VALUES, cfg.FLT3_ITD_NEGATIVE_VALUES)
    try:
        cfg.FLT3_ITD_POSITIVE_VALUES, cfg.FLT3_ITD_NEGATIVE_VALUES = {"Positive"}, {"negative"}
        cl = pd.DataFrame({"dbgap_subject_id": pd.array(["1", "2", "3", "4", "4", "5", "9"], dtype="string"),
                           "FLT3-ITD": pd.array(["positive", "negative", None, "negative", "POSITIVE ", "negative", "positive"], dtype="string")})
        st = io.flt3_itd_status(cl)
        assert list(st.loc[["1", "2", "3", "4"], "any_pos"]) == [True, False, False, True] and bool(st.loc["4", "discordant"])
        mat = pd.DataFrame({"FLT3": [0, 0, 1, 0, 0], "NPM1": [1, 0, 0, 1, 0]}, index=["1", "2", "3", "4", "5"])   # "9" is ITD+ but not profiled
        out = io.apply_flt3_composite(mat, cl)
        assert list(out["FLT3"]) == [1, 0, 1, 1, 0] and list(out["NPM1"]) == list(mat["NPM1"]) and "9" not in out.index
        assert list(mat["FLT3"]) == [0, 0, 1, 0, 0]                                   # input untouched
        bad = cl.copy(); bad.loc[0, "FLT3-ITD"] = "maybe"
        try:
            io.flt3_itd_status(bad)
        except ValueError as e:
            assert "maybe" in str(e)
        else:
            raise RuntimeError("an unknown value did not raise")
        cfg.FLT3_ITD_POSITIVE_VALUES = set()
        try:
            io.flt3_itd_status(cl)
        except ValueError:
            pass
        else:
            raise RuntimeError("an empty POSITIVE did not raise")
    finally:
        cfg.FLT3_ITD_POSITIVE_VALUES, cfg.FLT3_ITD_NEGATIVE_VALUES = old


def test_flt3_params_in_analysis_definition():
    for k in ("FLT3_COMPOSITE", "FLT3_GENE", "FLT3_ITD_COLUMN", "FLT3_ITD_POSITIVE_VALUES", "FLT3_ITD_NEGATIVE_VALUES"):
        assert k in cfg.ANALYSIS_DEFINITION_PARAMS and k in cfg.FREEZE_PARAMS
    h0 = data.analysis_definition_sha256()
    old = cfg.FLT3_COMPOSITE
    try:
        cfg.FLT3_COMPOSITE = not old
        assert data.analysis_definition_sha256() != h0
    finally:
        cfg.FLT3_COMPOSITE = old


def test_flt3_composite_audit():
    from common import io_official as io
    old = (cfg.FLT3_ITD_POSITIVE_VALUES, cfg.FLT3_ITD_NEGATIVE_VALUES)
    try:
        cfg.FLT3_ITD_POSITIVE_VALUES, cfg.FLT3_ITD_NEGATIVE_VALUES = {"positive"}, {"negative"}
        cl = pd.DataFrame({"dbgap_subject_id": pd.array(["1", "2", "3", "4", "4"], dtype="string"),
                           "FLT3-ITD": pd.array(["positive", "negative", None, "negative", "positive"], dtype="string")})
        wes = pd.DataFrame({"FLT3": [1, 0, 1, 0, 0]}, index=["1", "2", "3", "4", "5"])
        a = io.flt3_composite_audit(wes, cl).set_index("patient_id")
        assert list(a["source"]) == ["both", "neither", "wes_only", "itd_only", "neither"]
        assert list(a["itd_status"]) == ["positive_only", "negative_only", "no_clinical_value", "discordant", "no_clinical_value"]
        assert list(a["composite"]) == [True, False, True, True, False]
        comp = io.apply_flt3_composite(wes, cl)
        assert list(comp["FLT3"].astype(bool)) == list(a["composite"])        # audit == matrix
    finally:
        cfg.FLT3_ITD_POSITIVE_VALUES, cfg.FLT3_ITD_NEGATIVE_VALUES = old


if __name__ == "__main__":
    bad = 0
    for n, f in list(globals().items()):
        if n.startswith("test_") and callable(f):
            try:
                f(); print("PASS", n)
            except Exception:
                bad += 1; print("FAIL", n); traceback.print_exc()
    sys.exit(1 if bad else 0)
