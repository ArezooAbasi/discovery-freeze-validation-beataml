# -*- coding: utf-8 -*-
"""
safe_io.py -- safe CSV reading for the project outputs.

Why: by default pandas reads strings such as "null", "NA" and "None" as NaN. The simulation scenario "null"
was silently turned into NaN and dropped from tables/figures. This function turns only the empty string into NaN.

  from safe_io import read_csv_safe
  df = read_csv_safe(path)          # instead of pd.read_csv(path)
"""
import pandas as pd


def read_csv_safe(path, **kw):
    kw.setdefault("keep_default_na", False)
    kw.setdefault("na_values", [""])
    kw.setdefault("encoding", "utf-8-sig")
    return pd.read_csv(path, **kw)


def assert_scenarios_present(df, expected=("global_null",), col="scenario"):
    """If an expected scenario is missing from the table or is NaN, stop right there (instead of dropping it silently)."""
    if col not in df.columns:
        raise ValueError(f"column {col} not found")
    if df[col].isna().any():
        raise ValueError(f"{int(df[col].isna().sum())} row(s) with {col}=NaN (probably 'null' was read by pd.read_csv; use read_csv_safe)")
    miss = [s for s in expected if s not in set(df[col])]
    if miss:
        raise ValueError(f"expected scenarios missing from the table: {miss}")
