# -*- coding: utf-8 -*-
"""data.py -- loads the patient-level matrices of a cohort + I/O helpers shared by all stages + binds VALIDATION to the freeze."""
import argparse
import ast
import datetime
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
from . import config_official as cfg

_SRC = {}


def is_synthetic():
    return os.environ.get("BEATAML_SYNTHETIC") == "1"


def parse_cohort(desc, choices=("DISCOVERY", "FULL"), default="DISCOVERY"):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--cohort", choices=list(choices), default=default)
    return ap.parse_args().cohort


def load_cohort_matrices(cohort, overrides=None):
    """
    (mut, drug) aligned on the patients that are both profiled and have at least one AUC.
    overrides: dict of cfg settings (e.g. {"DRUG_REPEAT_POLICY": "mean"}) for sensitivity analyses only.
    """
    if is_synthetic():
        from . import synthetic
        mut, drug = synthetic.cohort_matrices(cohort)
    else:
        from . import io_official as io, cohort as co
        manifest = pd.read_csv(cfg.MANIFEST_FILE, dtype={"patient_id": str})
        patients = co.patients_of(manifest, cohort)
        if "src" not in _SRC:
            _SRC["src"] = io.load_sources()
        saved = {k: getattr(cfg, k) for k in (overrides or {})}
        try:
            for k, v in (overrides or {}).items():
                setattr(cfg, k, v)
            mut = io.build_patient_mutation_matrix(_SRC["src"], patients)
            drug = io.build_patient_drug_matrix(_SRC["src"], patients)
        finally:
            for k, v in saved.items():
                setattr(cfg, k, v)
    idx = mut.index.intersection(drug.index)
    return mut.loc[idx].astype(int), drug.loc[idx].astype(float)


def group_aucs(gene, drug, mut, drugdf, exclude=None):
    if gene not in mut.columns or drug not in drugdf.columns:
        return None, None
    y = drugdf[drug].dropna()
    y = y[y.index.isin(mut.index)]
    if exclude:
        y = y[~y.index.isin(set(exclude))]
    m = mut[gene].reindex(y.index) == 1
    return y[m].to_numpy(float), y[~m].to_numpy(float)


def mutated_set(mut, gene):
    return set(mut.index[mut[gene] == 1]) if gene in mut.columns else set()


def path(cohort, name):
    return cfg.cohort_dir(cohort) / name


def read(cohort, name, required=True, hint=""):
    p = path(cohort, name)
    if not p.exists():
        if required:
            raise SystemExit(f"[!] {p} پیدا نشد. {hint}")
        return None
    return pd.read_csv(p)


def write(df, cohort, name):
    p = path(cohort, name)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"[saved] {p}  ({len(df)} rows)")
    return p


def drug_names():
    """Mapping inhibitor -> standard_name from Stage 05 (falls back to the inhibitor itself if absent)."""
    p = cfg.cohort_dir("shared") / "drug_lookup_table.csv"
    if p.exists():
        d = pd.read_csv(p)
        return dict(zip(d["inhibitor"], d["standard_name"]))
    return {}


# ------------------------------------------------------------------ output versioning
def tagged(name, tag=None):
    """validation_results.csv + 'sensitivity' -> validation_results_sensitivity.csv (main and sensitivity never share a file)."""
    if not tag:
        return name
    p = Path(name)
    return f"{p.stem}_{tag}{p.suffix}"


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


# ------------------------------------------------------------------ parameter hashing
def _jsonable(v):
    """Recursive normalisation: set/frozenset -> sorted list, tuple -> list, Path -> str, dict with str keys."""
    if isinstance(v, (set, frozenset)):
        return sorted(_jsonable(x) for x in v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, Path):
        return str(v)
    return v


def _params(names):
    return {k: _jsonable(getattr(cfg, k)) for k in names}


def _hash_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def freeze_params():
    return _params(cfg.FREEZE_PARAMS)


def params_sha256():
    return _hash_obj(freeze_params())


def analysis_definition():
    """Structural "data/analysis definition" parameters (mutation, controls, drug, DNA, confounding checks)."""
    return _params(cfg.ANALYSIS_DEFINITION_PARAMS)


def analysis_definition_sha256():
    return _hash_obj(analysis_definition())


# ------------------------------------------------------------------ code hashing
# Rule: a change in these files after the freeze = hard stop (only a new freeze + a Discovery re-run is allowed).
#   - all common/*.py except config_official.py (its parameters are checked separately through hashes)
#   - stage scripts 00..08c (the stages that build and evaluate the frozen list)
# A change only in comments/docstrings (identical AST) in a hard file => warning with exact record.
# All other files (simulation, reports, ...) => warning.
# This definition lives in data.py itself, so changing it is also a hard stop (there is no way around it).
_HARD_TOP_RE = re.compile(r"^0[0-8][a-z]?_.*\.py$")


def _code_root():
    return Path(cfg.__file__).resolve().parents[1]


def _code_files():
    root = _code_root()
    return sorted(root.glob("*.py")) + sorted((root / "common").glob("*.py"))


def _rel(f):
    return str(f.relative_to(_code_root())).replace("\\", "/")


def is_hard_code_file(rel):
    if rel == "common/config_official.py":
        return False
    if rel.startswith("common/"):
        return True
    return bool(_HARD_TOP_RE.match(rel))


def code_hashes():
    return {_rel(f): sha256_file(f) for f in _code_files()}


def _ast_hash(f):
    """Hash of the program structure without comments and docstrings. If parsing fails => None (i.e. a "hard change")."""
    try:
        tree = ast.parse(Path(f).read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) \
                    and isinstance(b[0].value.value, str):
                node.body = b[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def code_ast_hashes():
    return {_rel(f): _ast_hash(f) for f in _code_files()}


# ------------------------------------------------------------------ VALIDATION lock
def validation_started():
    """The real lock: any marker (validation_started*.json) or any validation_results*.csv exists.
    The marker is created before any computation starts, so even a failed run keeps the freeze locked."""
    d = cfg.OUTPUT_ROOT / "VALIDATION"
    return d.exists() and (any(d.glob("validation_started*.json")) or any(d.glob("validation_results*.csv")))


def begin_validation(frozen_path, cohort, tag, binding):
    """
    Atomic, non-overwritable marker written before every Validation computation.
    If a marker (or result) already exists for this tag => SystemExit. Reset only by manual archiving with an explicit recorded reason.
    """
    d = cfg.cohort_dir("VALIDATION")
    marker = d / tagged("validation_started.json", tag)
    results = d / tagged("validation_results.csv", tag)
    if marker.exists() or results.exists():
        raise SystemExit(f"[!] VALIDATION (tag={tag}) قبلاً شروع شده: {marker if marker.exists() else results}\n"
                         "    اجرای مجدد/بازنویسی مجاز نیست. فقط با آرشیو کامل خروجی‌های Validation و ثبت صریح علت.")
    run_id = uuid.uuid4().hex
    payload = {"run_id": run_id, "cohort": cohort, "tag": tag, "started_at_utc": _utc_now(),
               "frozen_list": str(Path(frozen_path).resolve()), "frozen_list_sha256": binding["frozen_list_sha256"],
               "freeze_manifest_sha256": binding["freeze_manifest_sha256"], "freeze_id": binding.get("freeze_id"),
               "selection_params_sha256": binding["selection_params_sha256"],
               "analysis_definition_sha256": binding["analysis_definition_sha256"],
               "patient_manifest_sha256": sha256_file(cfg.MANIFEST_FILE) if (cfg.MANIFEST_FILE.exists() and not is_synthetic()) else None,
               "code_sha256": code_hashes()}
    text = json.dumps(payload, indent=2, default=str)
    tmp = d / f".{marker.name}.{run_id}.tmp"
    tmp.write_text(text)
    try:
        try:
            os.link(tmp, marker)                 # atomic, no overwrite; raises FileExistsError if it exists
        except FileExistsError:
            raise SystemExit(f"[!] marker همزمان ساخته شد: {marker}")
        except OSError:                          # filesystem without hardlink support
            try:
                with open(marker, "x", encoding="utf-8") as fh:
                    fh.write(text)
            except FileExistsError:
                raise SystemExit(f"[!] marker همزمان ساخته شد: {marker}")
    finally:
        tmp.unlink(missing_ok=True)
    print(f"[lock] {marker} (run_id={run_id})")
    return payload


def complete_validation(marker_payload, results_path):
    """Record the successful end of a run (result hash). A missing file next to the marker means the run did not complete."""
    p = cfg.cohort_dir("VALIDATION") / tagged("validation_completed.json", marker_payload.get("tag"))
    p.write_text(json.dumps({"run_id": marker_payload["run_id"], "completed_at_utc": _utc_now(),
                             "results_file": Path(results_path).name,
                             "results_sha256": sha256_file(results_path)}, indent=2))
    return p


# ------------------------------------------------------------------ binding VALIDATION to the freeze
# Discovery files that are hashed in freeze_manifest.json and re-checked in Validation.
FREEZE_ARTIFACTS = {
    "frozen_candidate_list_sha256": "frozen_candidate_list.csv",
    "frozen_sensitivity_list_sha256": "frozen_sensitivity_list.csv",
    "freeze_audit_sha256": "freeze_audit.csv",
    "associations_sha256": "mutation_drug_associations.csv",
    "stable_pairs_sha256": "stable_pairs.csv",
    "stability_gate_audit_sha256": "stability_gate_audit.csv",
    "adjusted_analysis_results_sha256": "adjusted_analysis_results.csv",
}


def verify_freeze(frozen_path, tag=None):
    """
    Binds the VALIDATION evaluation to Discovery.
    Hard mismatch => SystemExit:
      hashes of the lists and of the Discovery files (including adjusted_analysis_results), hash of the selection parameters (FREEZE_PARAMS),
      analysis_definition_sha256, hash of the patient manifest and of the sources (their absence is an error too; only synthetic mode is exempt),
      a meaningful change of any "hard" code file (is_hard_code_file) or its addition/removal.
    Warning: a change of config_official.py (its parameters are checked separately), a comment/docstring-only change in a hard file, a change of non-analysis files.
    """
    ddir = cfg.cohort_dir("DISCOVERY")
    mp = ddir / "freeze_manifest.json"
    if not mp.exists():
        raise SystemExit("[!] freeze_manifest.json نیست؛ اول 08b_freeze_candidates.py را اجرا کنید.")
    fm = json.loads(mp.read_text())
    key = "frozen_sensitivity_list_sha256" if tag == "sensitivity" else "frozen_candidate_list_sha256"
    expected = ddir / FREEZE_ARTIFACTS[key]
    errs, warns = [], []

    if not Path(frozen_path).exists() or Path(frozen_path).resolve() != expected.resolve():
        errs.append(f"--frozen باید همان فایل تولیدشده‌ی 08b باشد: {expected}")

    for k, fname in FREEZE_ARTIFACTS.items():
        f = ddir / fname
        if fm.get(k) is None:
            errs.append(f"freeze_manifest.json فاقد {k} است (freeze با نسخه‌ی قدیمی 08b ساخته شده؛ دوباره freeze کنید)")
        elif not f.exists():
            errs.append(f"{fname} از زمان freeze حذف شده")
        elif sha256_file(f) != fm[k]:
            errs.append(f"hash {fname} با freeze_manifest.json نمی‌خواند (فایل بعد از freeze عوض شده)")

    if params_sha256() != fm.get("selection_params_sha256"):
        errs.append("پارامترهای انتخاب (FREEZE_PARAMS) با زمان freeze فرق دارند")
    if fm.get("analysis_definition_sha256") is None:
        errs.append("freeze_manifest.json فاقد analysis_definition_sha256 است؛ دوباره freeze کنید")
    elif analysis_definition_sha256() != fm["analysis_definition_sha256"]:
        old, new = fm.get("analysis_definition") or {}, analysis_definition()
        diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
        errs.append(f"تعریف داده/تحلیل (ANALYSIS_DEFINITION_PARAMS) با زمان freeze فرق دارد: {diff or '؟'}")

    if not is_synthetic():
        if not cfg.MANIFEST_FILE.exists():
            errs.append(f"patient manifest نیست: {cfg.MANIFEST_FILE}")
        elif fm.get("patient_manifest_sha256") is None:
            errs.append("freeze_manifest.json فاقد patient_manifest_sha256 است")
        elif sha256_file(cfg.MANIFEST_FILE) != fm["patient_manifest_sha256"]:
            errs.append("patient manifest بعد از freeze عوض شده")
        mm = cfg.MANIFEST_DIR / "manifest_metadata.json"
        if not mm.exists():
            errs.append("manifest_metadata.json is missing")
        else:
            src = json.loads(mm.read_text()).get("source_sha256")
            if src is None or fm.get("source_sha256") is None:
                errs.append("source_sha256 در manifest_metadata.json یا freeze_manifest.json ثبت نشده")
            elif src != fm["source_sha256"]:
                errs.append("Source hashes do not match freeze (hash فایل‌های رسمی منبع با زمان freeze فرق دارد)")

    if sha256_file(cfg.__file__) != fm.get("config_official_sha256"):
        warns.append("فایل config_official.py از زمان freeze تغییر کرده (hash پارامترها و تعریف تحلیل جداگانه چک شد و یکی است؛ تغییر را در Methods توضیح دهید)")

    frozen_code = fm.get("code_sha256") or {}
    frozen_ast = fm.get("code_ast_sha256") or {}
    if not frozen_code:
        errs.append("freeze_manifest.json فاقد code_sha256 است")
    now_code, now_ast = code_hashes(), code_ast_hashes()
    hard_changed, soft_changed, comment_only = [], [], []
    for f in sorted(set(frozen_code) | set(now_code)):
        if frozen_code.get(f) == now_code.get(f):
            continue
        if not is_hard_code_file(f):
            soft_changed.append(f)
        elif (f in frozen_code and f in now_code and frozen_ast.get(f) is not None
              and frozen_ast.get(f) == now_ast.get(f)):
            comment_only.append(f)
        else:
            hard_changed.append(f)
    if hard_changed:
        errs.append(f"فایل‌های کد مؤثر بر تحلیل از زمان freeze تغییر کرده‌اند (فقط freeze جدید + اجرای مجدد Discovery): {hard_changed}")
    if comment_only:
        warns.append(f"فقط کامنت/docstring تغییر کرده (AST یکسان) در: {comment_only}")
    if soft_changed:
        warns.append(f"فایل‌های غیرمؤثر بر انتخاب تغییر کرده‌اند: {soft_changed}")

    for w in warns:
        print("[warn]", w)
    if errs:
        raise SystemExit("[!] اتصال به freeze نقض شد:\n  - " + "\n  - ".join(errs))
    return {"frozen_list_sha256": sha256_file(frozen_path), "selection_params_sha256": params_sha256(),
            "analysis_definition_sha256": analysis_definition_sha256(),
            "freeze_manifest_sha256": sha256_file(mp), "freeze_id": fm.get("freeze_id"),
            "frozen_at_utc": fm.get("frozen_at_utc"), "warnings": warns,
            "code_changed_comment_only": comment_only, "code_changed_non_analysis": soft_changed}
