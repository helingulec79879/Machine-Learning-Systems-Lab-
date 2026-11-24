"""Step 4 - Split, then clean/transform with everything fitted on TRAIN only."""
import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import schema as S
from .transformers import DERIVED, FeatureEngineer, QuantileClipper

log = logging.getLogger("lab4")


def split(df: pd.DataFrame, cfg: dict):
    """70/30 stratified split on the validated (NOT cleaned) data."""
    sp = cfg["split"]
    train, test = train_test_split(df, test_size=sp["test_size"], random_state=sp["random_state"],
                                   stratify=df[S.TARGET] if sp["stratify"] else None)
    log.info("SPLIT train=%d test=%d", len(train), len(test))
    return train, test


def to_model_frame(part: pd.DataFrame, missing_token="?"):
    """'?' -> NaN, ints numeric, categories as plain object. Returns X, y (y: 1 if >50K)."""
    X = part[S.FEATURES].copy()
    for c in S.CAT_COLS:
        X[c] = X[c].astype(object).where(X[c].astype(object) != missing_token, np.nan)
    for c in S.INT_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    y = (part[S.TARGET].astype(str) == ">50K").astype(int)
    return X, y


def drop_train_duplicates(train: pd.DataFrame):
    before = len(train)
    out = train.drop_duplicates(keep="first")
    log.info("DEDUP train exact duplicates removed=%d (test untouched)", before - len(out))
    return out, before - len(out)


def build_preprocessor(cfg: dict, with_features=True) -> Pipeline:
    c = cfg["cleaning"]
    num_cols = S.INT_COLS + (DERIVED if with_features else [])
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy=c["numeric_imputer"])),
        ("clip", QuantileClipper(c["clip_lower_percentile"], c["clip_upper_percentile"])),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy=c["categorical_imputer"])),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    cols = ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, S.CAT_COLS)],
                             verbose_feature_names_out=True)
    return Pipeline([("features", FeatureEngineer(enabled=with_features)), ("columns", cols)])


def fitted_summary(pre: Pipeline) -> dict:
    """Numbers learned from TRAIN (for the report / leakage evidence)."""
    ct = pre.named_steps["columns"]
    num = ct.named_transformers_["num"]
    cat = ct.named_transformers_["cat"]
    num_cols = ct.transformers_[0][2]
    return {
        "numeric_medians": dict(zip(num_cols, map(float, num.named_steps["impute"].statistics_))),
        "clip_low_p1": dict(zip(num_cols, map(float, num.named_steps["clip"].lower_))),
        "clip_high_p99": dict(zip(num_cols, map(float, num.named_steps["clip"].upper_))),
        "categorical_modes": dict(zip(S.CAT_COLS, map(str, cat.named_steps["impute"].statistics_))),
        "n_output_features": int(len(pre.get_feature_names_out())),
    }
