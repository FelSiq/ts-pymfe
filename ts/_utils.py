import typing as t

import sklearn.preprocessing
import numpy as np
import pandas as pd


def apply_on_tiles(ts: np.ndarray, num_tiles: int,
                   func: t.Callable[[np.ndarray],
                                    t.Any], *args, **kwargs) -> np.ndarray:
    """Apply a function on time-series tiles (non-overlapping windows)."""
    res = np.array(
        [
            func(split, *args, **kwargs)  # type: ignore
            for split in np.array_split(ts, num_tiles)
        ],
        dtype=float)

    return res


def get_rolling_window(
    ts: np.ndarray,
    window_size: int,
    center: bool = True,
    ts_scaled: t.Optional[np.ndarray] = None,
) -> pd.core.window.rolling.Rolling:
    """Apply a function on time-series rolling (overlapping) windows.

    If ``center`` is True, then each rolling window is centered at the
    central instance rather than the initial instance.
    """
    if ts_scaled is None:
        ts_scaled = sklearn.preprocessing.StandardScaler().fit_transform(
            ts.reshape(-1, 1)).ravel()

    window_size = min(ts.size, window_size)
    return pd.Series(ts_scaled).rolling(window_size, center=center)


def smape(arr_a: np.ndarray,
          arr_b: np.ndarray,
          percentage: bool = False,
          half_denom: bool = True) -> float:
    """Calculate SMAPE (Symmetric Mean Absolute Percentage Error).

    Parameters
    ----------
    arr_a, arr_b : :obj:`np.ndarray`
        Arrays to calculate the SMAPE from.

    percentage : bool, optional
        If True, multiply the result by 100 (i.e., return the
        percentage value, not the fraction).

    half_denom : bool, optional
        If True, divide the denominator by 2. If False, the
        result if bounded by [0, 100] (if ``percentage`` is
        True), or by [0, 1] otherwise.

    Returns
    -------
    float
        SMAPE estimation between the two given arrays. If
        ``percentage`` is True, then return the estimation in
        the percentage form (in [0, 100] range). Return the
        error in fraction form (in [0, 1] range) otherwise.
    """
    res = np.mean(
        np.abs(arr_a - arr_b) / (1e-9 + np.abs(arr_a) + np.abs(arr_b)))

    if percentage:
        res *= 100

    if not half_denom:
        res *= 2

    return res


def sample_data(ts: np.ndarray,
                lm_sample_frac: float,
                X: t.Optional[np.ndarray] = None) -> np.ndarray:
    """Select ``lm_sample_frac`` percent of data from ``ts``."""
    threshold = int(np.ceil(ts.size * lm_sample_frac))

    if threshold >= ts.size:
        if X is not None:
            return ts, X

        return ts

    if X is not None:
        return ts[:threshold], X[:threshold, :]

    return ts[:threshold]
