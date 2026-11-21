# -*- coding: utf-8 -*-
"""
fix_simulation_null_label.py
=========================================================================================
Run once: in the existing simulation outputs, converts scenario == "null" to "global_null"
(without re-running the simulation; no other number changes). Before changing a file it makes a .bak_before_global_null copy.
The rewrite is at text (csv) level, not pandas, so column types/number formats stay untouched.

  python fix_simulation_null_label.py [--cohort DISCOVERY] [--dry-run]
"""
import argparse
import csv
import shutil
from common import data

FILES = ["simulation_replicates.csv", "simulation_run_status.csv", "simulation_summary.csv",
         "simulation_planted_log.csv", "simulation_winners_curse.csv", "simulation_datasplit_pairs.csv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="DISCOVERY", choices=["DISCOVERY", "FULL"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for name in FILES:
        p = data.path(a.cohort, name)
        if not p.exists():
            print(f"[skip] {name}: not found")
            continue
        with open(p, "r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.reader(fh))
        if not rows or "scenario" not in rows[0]:
            print(f"[skip] {name}: no scenario column")
            continue
        k = rows[0].index("scenario")
        n_null = sum(1 for r in rows[1:] if len(r) > k and r[k] == "null")
        n_gn = sum(1 for r in rows[1:] if len(r) > k and r[k] == "global_null")
        if n_null == 0:
            print(f"[ok]   {name}: no 'null' (global_null={n_gn})")
            continue
        print(f"[fix]  {name}: {n_null} row(s) 'null' -> 'global_null'")
        if a.dry_run:
            continue
        bak = p.with_name(p.name + ".bak_before_global_null")
        if not bak.exists():
            shutil.copy2(p, bak)
        for r in rows[1:]:
            if len(r) > k and r[k] == "null":
                r[k] = "global_null"
        with open(p, "w", encoding="utf-8-sig", newline="") as fh:
            csv.writer(fh).writerows(rows)
    print("Done. Every script that reads these files must now use read_csv_safe (safe_io.py).")


if __name__ == "__main__":
    main()
