import typing as t

import numpy as np


class TSNaive:
    """TODO."""
    def __init__(self):
        """TODO."""
        self.last_obs = None

    def fit(self, _: np.ndarray, y: np.ndarray) -> "_TSNaive":
        """TODO."""
        self.last_obs = y[-1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """TODO."""
        return np.full(fill_value=self.last_obs, shape=X.size)


class TSNaiveDrift:
    """TODO."""
    def __init__(self):
        """TODO."""
        self.slope = None
        self.last_obs = None

    def fit(self, _: np.ndarray, y: np.ndarray) -> "_TSNaiveDrift":
        """TODO."""
        self.last_obs = y[-1]
        self.last_obs_ind = y.size
        self.slope = (y[-1] - y[0]) / (y.size - 1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """TODO."""
        return self.last_obs + (X - self.last_obs_ind) * self.slope


class TSNaiveSeasonal:
    """TODO."""
    def __init__(self, ts_period: t.Optional[int] = None):
        """TODO."""
        self.y = None
        self.ts_period = ts_period

    def fit(self,
            _: np.ndarray,
            y: np.ndarray,
            ts_period: t.Optional[int] = None,
            copy: bool = False) -> "_TSNaiveSeasonal":
        """TODO."""
        if copy or not isinstance(y, np.ndarray):
            self.y = np.copy(y)

        else:
            self.y = y

        if ts_period is not None:
            self.ts_period = ts_period

        if self.ts_period is None:
            raise ValueError("'ts_period' must be given.")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """TODO."""
        shift = (X - self.y.size) // self.ts_period
        inds = X - self.ts_period * (1 + shift)
        return self.y[inds]
