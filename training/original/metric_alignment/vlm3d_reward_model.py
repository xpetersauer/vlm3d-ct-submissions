"""Small, version-stable reward regressor used by the Task-2 container.

The class intentionally depends only on NumPy.  A joblib pickle created on
RunPod therefore does not depend on a matching scikit-learn minor version in
the Forithmus image.
"""

from __future__ import annotations

import numpy as np


class NumpyRidgeRewardModel:
    def __init__(self, mean, scale, coef, intercept):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = float(intercept)
        self.n_features_in_ = int(self.coef.size)

    def predict(self, values):
        x = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Task-2 reward model expected {self.n_features_in_} features, "
                f"received {x.shape[1]}"
            )
        z = (x - self.mean) / self.scale
        return z @ self.coef + self.intercept


def fit_ridge(values, targets, alpha=1.0):
    x = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("Expected X=[rows, features] and one target per row")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([z, np.ones(len(z))])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[-1, -1] = 0.0
    params = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return NumpyRidgeRewardModel(mean, scale, params[:-1], params[-1])
