"""Random-forest surrogate with uncertainty from the spread across trees."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

from lab.features import DEFAULT_SET, feature_table

DEFAULT_FEATURE_SET = DEFAULT_SET  # chosen by lab/compare_features.py


class Surrogate:
    def __init__(self, n_estimators: int = 100, seed: int = 0, feature_set: str | None = None):
        self.n_estimators = n_estimators
        self.seed = seed
        self.feature_set = feature_set
        self.model: RandomForestRegressor | None = None

    def _X(self, ids: list[str]) -> np.ndarray:
        ft = feature_table(self.feature_set or DEFAULT_FEATURE_SET)
        return ft.loc[ids].to_numpy(dtype=float)

    def fit(self, ids: list[str], log_sigma: list[float]) -> "Surrogate":
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators, min_samples_leaf=2, max_features=0.5,
            random_state=self.seed, n_jobs=1,
        ).fit(self._X(ids), np.asarray(log_sigma, dtype=float))
        return self

    def predict(self, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        assert self.model is not None, "fit first"
        per_tree = np.stack([t.predict(self._X(ids)) for t in self.model.estimators_])
        return per_tree.mean(0), per_tree.std(0) + 1e-6

    def score(self, ids: list[str], threshold: float, beta: float = 1.0) -> pd.DataFrame:
        mu, sd = self.predict(ids)
        return pd.DataFrame(
            {
                "mu": mu,
                "sd": sd,
                "ucb": mu + beta * sd,
                "p_hit": 1 - norm.cdf((threshold - mu) / sd),
            },
            index=pd.Index(ids, name="candidate_id"),
        )
