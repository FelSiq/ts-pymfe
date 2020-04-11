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
          percentage: bool = False) -> float:
    """Calculate SMAPE (Symmetric Mean Absolute Percentage Error).

    Parameters
    ----------
    arr_a, arr_b : :obj:`np.ndarray`
        Arrays to calculate the SMAPE from.

    percentage : bool, optional
        If True, multiply the result by 100 (i.e., return the
        percentage value, not the fraction).

    Returns
    -------
    float
        SMAPE estimation between the two given arrays. If
        ``percentage`` is True, then return the estimation in
        the percentage form (in [0, 100] range). Return the
        error in fraction form (in [0, 1] range) otherwise.
    """
    res = 2 * np.mean(np.abs(arr_a - arr_b) / (arr_a + arr_b))

    if percentage:
        return 100 * res

    return res


def _get_sample_inds(num_inst: int, lm_sample_frac: float,
                     random_state: t.Optional[int]) -> np.ndarray:
    """Sample indices to calculate subsampling landmarking metafeatures."""
    if random_state is not None:
        np.random.seed(random_state)

    sample_inds = np.random.choice(a=num_inst,
                                   size=int(lm_sample_frac * num_inst),
                                   replace=False)

    return sample_inds


def sample_data(
        ts: np.ndarray,
        lm_sample_frac: float,
        random_state: t.Optional[int] = None,
        sample_inds: t.Optional[np.ndarray] = None,
) -> np.ndarray:
    """Select ``lm_sample_frac`` percent of data from ``ts``."""
    if lm_sample_frac >= 1.0 and sample_inds is None:
        return ts

    if sample_inds is None:
        num_inst = ts.size

        sample_inds = _get_sample_inds(num_inst=num_inst,
                                       lm_sample_frac=lm_sample_frac,
                                       random_state=random_state)

    return ts[sample_inds]
