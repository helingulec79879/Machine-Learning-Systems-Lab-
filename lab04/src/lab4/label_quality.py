"""Step 6 - Label quality: duplicate-conflict audit + 50-row re-check vs data dictionary."""
import logging

import pandas as pd

from . import schema as S

log = logging.getLogger("lab4")


def duplicate_conflicts(df: pd.DataFrame):
    """Groups of identical 14-feature rows that carry BOTH income labels."""
    g = df.groupby(S.FEATURES, dropna=False, observed=True)[S.TARGET]
    nun = g.transform("nunique")
    size = g.transform("size")
    conflict_rows = df[(nun > 1)]
    dup_rows = df[size > 1]
    n_groups = int(conflict_rows.groupby(S.FEATURES, dropna=False, observed=True).ngroups) if len(conflict_rows) else 0
    res = {
        "duplicate_feature_groups": int(dup_rows.groupby(S.FEATURES, dropna=False, observed=True).ngroups) if len(dup_rows) else 0,
        "rows_in_duplicate_feature_groups": int(len(dup_rows)),
        "conflicting_groups": n_groups,
        "rows_in_conflicting_groups": int(len(conflict_rows)),
        "pct_rows_conflicting": round(100 * len(conflict_rows) / len(df), 4),
    }
    out = conflict_rows.sort_values(S.FEATURES).reset_index().rename(columns={"index": "row_index"})
    log.info("LABEL conflicting_groups=%d rows=%d", n_groups, len(conflict_rows))
    return res, out


def row_flags(r) -> list:
    """Rule checks a human would do when reading a row against adult.names."""
    f = []
    for c, (lo, hi) in S.RANGES.items():
        if not (lo <= r[c] <= hi):
            f.append(f"{c} out of range")
    if r["education"] in S.EDU_NUM and S.EDU_NUM[r["education"]] != r["education-num"]:
        f.append("education/education-num mismatch")
    if r["relationship"] == "Husband" and r["sex"] == "Female":
        f.append("Husband but Female")
    if r["relationship"] == "Wife" and r["sex"] == "Male":
        f.append("Wife but Male")
    if r["relationship"] in ("Husband", "Wife") and r["marital-status"] not in (
            "Married-civ-spouse", "Married-AF-spouse"):
        f.append("spouse relationship but not married")
    if r["capital-gain"] > 0 and r["capital-loss"] > 0:
        f.append("both capital-gain and capital-loss > 0")
    if r["capital-gain"] == 99999:
        f.append("capital-gain = 99999 (top-coded sentinel)")
    if r["age"] < 21 and r["education-num"] >= 13:
        f.append("age < 21 with Bachelors or higher (suspicious)")
    if r["workclass"] == "Never-worked" and r["occupation"] != "?":
        f.append("Never-worked but has occupation")
    for c in S.CAT_COLS:
        if r[c] not in S.ALLOWED[c]:
            f.append(f"{c} not in dictionary")
    return f


def manual_recheck(df: pd.DataFrame, n: int, seed: int):
    sample = df.sample(n=n, random_state=seed).copy()
    sample["auto_flags"] = sample.apply(lambda r: "; ".join(row_flags(r)), axis=1)
    sample["missing_fields"] = sample.apply(
        lambda r: ", ".join(c for c in S.CAT_COLS if r[c] == "?"), axis=1)
    sample["manual_verdict"] = ""     # fill by hand: OK / IMPOSSIBLE / SUSPICIOUS
    sample = sample.reset_index().rename(columns={"index": "row_index"})
    # Same rules over the whole dataset, for context
    all_flags = df.apply(row_flags, axis=1)
    counts = {}
    for fl in all_flags:
        for x in fl:
            counts[x] = counts.get(x, 0) + 1
    res = {
        "sample_size": n,
        "random_state": seed,
        "sample_rows_flagged": int((sample["auto_flags"] != "").sum()),
        "sample_rows_with_missing": int((sample["missing_fields"] != "").sum()),
        "sample_flag_counts": pd.Series([x for s in sample["auto_flags"] if s for x in s.split("; ")])
                                .value_counts().to_dict(),
        "full_dataset_flag_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "sample_label_distribution": sample[S.TARGET].value_counts().to_dict(),
    }
    return res, sample


def to_markdown(dup: dict, man: dict) -> str:
    L = ["# Label-Quality Report", "",
         "Adult labels come from census income, so there is no subjective re-labelling; "
         "instead we run a duplicate-conflict audit and a 50-row manual re-check.", "",
         "## Duplicate-conflict audit (14 features identical, label differs)", ""]
    for k, v in dup.items():
        L.append(f"- {k}: **{v}**")
    L += ["", "Details: `results/label_conflicts.csv`.", "",
          f"## Manual re-check of {man['sample_size']} random rows (seed {man['random_state']})", "",
          f"- Rows with at least one rule flag: **{man['sample_rows_flagged']}**",
          f"- Rows with a missing ('?') field: **{man['sample_rows_with_missing']}**",
          f"- Label mix in sample: {man['sample_label_distribution']}", ""]
    if man["sample_flag_counts"]:
        L += ["Flags in sample:", ""] + [f"- {k}: {v}" for k, v in man["sample_flag_counts"].items()] + [""]
    L += ["Same rules on the full dataset:", ""]
    L += [f"- {k}: {v}" for k, v in man["full_dataset_flag_counts"].items()] or ["- none"]
    L += ["", "Rows to review by hand (column `manual_verdict`): `results/manual_check_50.csv`."]
    return "\n".join(L) + "\n"
