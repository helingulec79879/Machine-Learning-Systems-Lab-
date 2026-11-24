"""Custom sklearn transformers (importable, so the joblib file can be re-loaded)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DERIVED = ["capital_net", "hours_per_age"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds derived features. Stateless (learns nothing), so it cannot leak test info.

    capital_net   = capital-gain - capital-loss
    hours_per_age = hours-per-week / (age - 16)   (age >= 17, so denominator >= 1)
    """

    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        if self.enabled:
            X["capital_net"] = X["capital-gain"] - X["capital-loss"]
            denom = (X["age"] - 16).where(X["age"] > 16)   # guard against /0 -> NaN -> imputed
            X["hours_per_age"] = X["hours-per-week"] / denom
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(self.feature_names_in_)
        return np.asarray(base + (DERIVED if self.enabled else []), dtype=object)


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each numeric column to [p_low, p_high] percentiles learned on TRAIN only."""

    def __init__(self, lower=1, upper=99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        A = np.asarray(X, dtype=float)
        self.lower_ = np.nanpercentile(A, self.lower, axis=0)
        self.upper_ = np.nanpercentile(A, self.upper, axis=0)
        self.n_features_in_ = A.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        A = np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)
        if hasattr(X, "columns"):
            return pd.DataFrame(A, columns=X.columns, index=X.index)
        return A

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return getattr(self, "feature_names_in_", np.asarray([f"x{i}" for i in range(self.n_features_in_)]))
