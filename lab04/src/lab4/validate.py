"""Step 3 - Validation: schema, type, range, null, category, duplicate checks.

Nothing is dropped here. Every violation is reported (counts + row-level CSV).
"""
import logging

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from . import schema as S

log = logging.getLogger("lab4")


def standardize(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list]:
    """Return a COPY with canonical column names and stripped strings.

    OpenML stores the original '?' tokens as NaN; to follow the lab's null policy
    ('?' means missing) we map categorical NaN back to '?'. Raw frame is not touched.
    """
    notes = []
    df = raw.copy(deep=True)
    tgt_raw = cfg["data"]["target_raw_name"]
    if tgt_raw in df.columns and tgt_raw != S.TARGET:
        df = df.rename(columns={tgt_raw: S.TARGET})
        notes.append(f"renamed '{tgt_raw}' -> '{S.TARGET}'")
    df.columns = [str(c).strip() for c in df.columns]
    for c in S.CAT_COLS + [S.TARGET]:
        if c in df.columns:
            s = df[c].astype(object)
            s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
            n_nan = int(s.isna().sum())
            if n_nan and c != S.TARGET:
                notes.append(f"{c}: {n_nan} NaN mapped to '{S.MISSING_TOKEN}'")
                s = s.where(s.notna(), S.MISSING_TOKEN)
            if c == S.TARGET:
                dotted = int(s.astype(str).str.endswith(".").sum())
                if dotted:
                    notes.append(f"income: {dotted} labels had trailing '.' (e.g. '>50K.') -> stripped")
                    s = s.map(lambda v: v.rstrip(".") if isinstance(v, str) else v)
            df[c] = s
    return df, notes


def _is_int_like(series: pd.Series) -> tuple[bool, str]:
    if ptypes.is_integer_dtype(series):
        return True, "integer dtype"
    if ptypes.is_float_dtype(series):
        v = series.dropna()
        if (np.mod(v, 1) == 0).all():
            return True, f"{series.dtype} but all values integer-valued (castable to int)"
        return False, f"{series.dtype} with non-integer values"
    return False, f"non-numeric dtype {series.dtype}"


def validate(df: pd.DataFrame, raw_dtypes: dict) -> tuple[dict, pd.DataFrame]:
    violations = []           # row-level records
    report = {"n_rows": int(len(df)), "n_cols": int(df.shape[1])}

    # 1. Schema: column presence
    missing_cols = [c for c in S.ALL_COLS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in S.ALL_COLS]
    report["schema"] = {"missing_columns": missing_cols, "unexpected_columns": extra_cols,
                        "passed": not missing_cols and not extra_cols}

    # 2. Types
    type_res = {}
    for c, t in S.TYPES.items():
        if c not in df.columns:
            continue
        if t == "int":
            ok, why = _is_int_like(pd.to_numeric(df[c], errors="coerce")) \
                if not ptypes.is_numeric_dtype(df[c]) else _is_int_like(df[c])
            if not ptypes.is_numeric_dtype(df[c]):
                bad = pd.to_numeric(df[c], errors="coerce").isna() & df[c].notna()
                for i in df.index[bad]:
                    violations.append((i, c, df.at[i, c], "type: not numeric"))
                ok = ok and not bad.any()
        else:
            ok = ptypes.is_object_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype) \
                or ptypes.is_string_dtype(df[c])
            why = f"{df[c].dtype}"
        type_res[c] = {"expected": t, "observed_raw_dtype": raw_dtypes.get(c, raw_dtypes.get("class")) if c == S.TARGET else raw_dtypes.get(c),
                       "passed": bool(ok), "note": why}
    report["types"] = type_res

    # 3. Ranges (lab-sheet ranges + informational extras)
    def range_check(rng, label):
        out = {}
        for c, (lo, hi) in rng.items():
            x = pd.to_numeric(df[c], errors="coerce")
            bad = pd.Series(False, index=df.index)
            if lo is not None:
                bad |= x < lo
            if hi is not None:
                bad |= x > hi
            for i in df.index[bad]:
                violations.append((i, c, df.at[i, c], f"{label}: outside [{lo}, {hi}]"))
            out[c] = {"range": [lo, hi], "observed_min": float(x.min()), "observed_max": float(x.max()),
                      "violations": int(bad.sum())}
        return out
    report["ranges"] = range_check(S.RANGES, "range")
    report["extra_ranges"] = range_check(S.EXTRA_RANGES, "dictionary range")

    # 4. Nulls ('?' and NaN both count as missing)
    nulls = {}
    for c in S.ALL_COLS:
        q = int((df[c].astype(object) == S.MISSING_TOKEN).sum())
        n = int(df[c].isna().sum())
        nulls[c] = {"question_mark": q, "nan": n, "total_missing": q + n,
                    "pct": round(100 * (q + n) / len(df), 3)}
        for i in df.index[df[c].isna()]:
            violations.append((i, c, None, "null: NaN (not allowed after standardisation)"))
    rows_any_missing = int(((df[S.CAT_COLS].astype(object) == S.MISSING_TOKEN).any(axis=1)
                            | df[S.ALL_COLS].isna().any(axis=1)).sum())
    report["nulls"] = {"per_column": nulls, "rows_with_any_missing": rows_any_missing,
                       "policy": "'?' = missing; '?' is an allowed category; rows are kept"}

    # 5. Allowed categories
    cats = {}
    for c, allowed in S.ALLOWED.items():
        vals = df[c].astype(object)
        bad = ~vals.isin(allowed) & vals.notna()
        unexpected = sorted(map(str, vals[bad].unique()))
        for i in df.index[bad]:
            violations.append((i, c, df.at[i, c], "category: not in allowed set"))
        cats[c] = {"n_unique": int(vals.nunique()), "unexpected_values": unexpected,
                   "violations": int(bad.sum())}
    report["categories"] = cats

    # 6. Duplicates (reported only)
    report["duplicates"] = {
        "exact_duplicate_rows_incl_label": int(df.duplicated().sum()),
        "duplicate_feature_rows": int(df.duplicated(subset=S.FEATURES).sum()),
    }

    # 7. Label distribution
    vc = df[S.TARGET].value_counts()
    report["label_distribution"] = {str(k): int(v) for k, v in vc.items()}

    # 8. Consistency (education vs education-num)
    exp = df["education"].map(S.EDU_NUM)
    bad = exp.notna() & (pd.to_numeric(df["education-num"], errors="coerce") != exp)
    for i in df.index[bad]:
        violations.append((i, "education-num", df.at[i, "education-num"],
                           f"consistency: expected {exp[i]} for {df.at[i, 'education']}"))
    report["consistency"] = {"education_vs_education_num_mismatch": int(bad.sum())}

    vdf = pd.DataFrame(violations, columns=["row_index", "column", "value", "rule"])
    hard = vdf[~vdf["rule"].str.startswith("dictionary")]
    report["summary"] = {
        "total_violation_records": int(len(vdf)),
        "hard_violations": int(len(hard)),
        "rows_with_violations": int(vdf["row_index"].nunique()) if len(vdf) else 0,
        "rows_dropped_by_validation": 0,
        "passed": bool(len(hard) == 0 and report["schema"]["passed"]
                       and all(v["passed"] for v in type_res.values())),
    }
    log.info("VALIDATE violations=%d rows_with_violations=%d missing_rows=%d",
             len(vdf), report["summary"]["rows_with_violations"], rows_any_missing)
    return report, vdf


def report_to_markdown(rep: dict, notes: list) -> str:
    L = ["# Validation Report — UCI Adult", "",
         f"Rows: **{rep['n_rows']}**, columns: **{rep['n_cols']}**. "
         "Validation never drops rows; it only reports.", ""]
    if notes:
        L += ["## Standardisation notes (on a copy, raw data untouched)", ""]
        L += [f"- {n}" for n in notes] + [""]
    s = rep["schema"]
    L += ["## 1. Schema", "", f"- Missing columns: {s['missing_columns'] or 'none'}",
          f"- Unexpected columns: {s['unexpected_columns'] or 'none'}",
          f"- Result: {'PASS' if s['passed'] else 'FAIL'}", "",
          "## 2. Types", "", "| Column | Expected | Raw dtype | Result | Note |", "|---|---|---|---|---|"]
    for c, v in rep["types"].items():
        L.append(f"| {c} | {v['expected']} | {v['observed_raw_dtype']} | "
                 f"{'PASS' if v['passed'] else 'FAIL'} | {v['note']} |")
    L += ["", "## 3. Ranges (lab sheet)", "", "| Column | Allowed | Observed min | Observed max | Violations |",
          "|---|---|---|---|---|"]
    for c, v in rep["ranges"].items():
        L.append(f"| {c} | {v['range'][0]}–{v['range'][1]} | {v['observed_min']:g} | "
                 f"{v['observed_max']:g} | {v['violations']} |")
    L += ["", "Informational (data dictionary):", ""]
    for c, v in rep["extra_ranges"].items():
        L.append(f"- {c} in [{v['range'][0]}, {v['range'][1]}]: {v['violations']} violations "
                 f"(min {v['observed_min']:g}, max {v['observed_max']:g})")
    L += ["", "## 4. Missing values (null policy: '?' = missing, rows kept)", "",
          "| Column | '?' | NaN | Total | % |", "|---|---|---|---|---|"]
    for c, v in rep["nulls"]["per_column"].items():
        if v["total_missing"]:
            L.append(f"| {c} | {v['question_mark']} | {v['nan']} | {v['total_missing']} | {v['pct']} |")
    L += ["", f"Rows with at least one missing value: **{rep['nulls']['rows_with_any_missing']}**", "",
          "## 5. Allowed categories", "", "| Column | Unique | Unexpected values | Violations |",
          "|---|---|---|---|"]
    for c, v in rep["categories"].items():
        L.append(f"| {c} | {v['n_unique']} | {', '.join(v['unexpected_values']) or '—'} | {v['violations']} |")
    d = rep["duplicates"]
    L += ["", "## 6. Duplicates (reported, not dropped here)", "",
          f"- Exact duplicate rows (features + label): **{d['exact_duplicate_rows_incl_label']}**",
          f"- Duplicate feature rows (ignoring label): **{d['duplicate_feature_rows']}**", "",
          "## 7. Label distribution", ""]
    tot = sum(rep["label_distribution"].values())
    for k, v in rep["label_distribution"].items():
        L.append(f"- {k}: {v} ({100 * v / tot:.2f}%)")
    L += ["", "## 8. Consistency", "",
          f"- education vs education-num mismatches: {rep['consistency']['education_vs_education_num_mismatch']}",
          "", "## Summary", ""]
    for k, v in rep["summary"].items():
        L.append(f"- {k}: {v}")
    L += ["", "Row-level details: `results/validation_violations.csv`."]
    return "\n".join(L) + "\n"
