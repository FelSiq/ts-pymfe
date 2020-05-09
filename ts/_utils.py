import typing as t
import operator

import sklearn.preprocessing
import numpy as np
import pandas as pd


def apply_on_tiles(ts: np.ndarray, num_tiles: int,
                   func: t.Callable[[np.ndarray],
                                    t.Any], *args, **kwargs) -> np.ndarray:
    """Apply a function on time-series tiles (non-overlapping windows)."""
    if num_tiles > 0.5 * ts.size:
        raise ValueError("'num_tiles' ({}) larger than half the "
                         "time-series size ({}).".format(
                             num_tiles, 0.5 * ts.size))

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


def standardize_ts(ts: np.ndarray,
                   ts_scaled: t.Optional[np.ndarray] = None) -> np.ndarray:
    """Standardize (z-score normalization) time-series."""
    if ts_scaled is None:
        return sklearn.preprocessing.StandardScaler().fit_transform(
            ts.reshape(-1, 1)).ravel()

    return ts


def find_plateau_pt(arr: np.ndarray,
                    arr_diff: t.Optional[np.ndarray] = None) -> np.ndarray:
    """Find plateau points in array."""
    if arr_diff is None:
        arr_diff = np.diff(arr)

    arr_diff_2 = np.diff(arr_diff)

    res = np.logical_and(np.isclose(arr_diff_2, 0),
                         np.isclose(arr_diff[:-1], 0))

    return np.hstack((0, res.astype(int), 0))


def find_crit_pt(arr: np.ndarray, type_: str) -> np.ndarray:
    """Find critical points on the given values.

    ``type`` must be in {"min", "max", "plateau", "non-plateau", "any"}.
    """
    if arr.size <= 2:
        raise ValueError("Array too small (size {}). Need at least "
                         "3 elements.".format(arr.size))

    VALID_TYPES = {"min", "max", "plateau", "non-plateau", "any"}

    if type_ not in VALID_TYPES:
        raise ValueError("'type_' must be in {} (got '{}')."
                         "".format(type_, VALID_TYPES))

    # Note: first discrete derivative
    arr_diff_1 = np.diff(arr)

    if type_ == "plateau":
        return find_plateau_pt(arr, arr_diff_1)

    turning_pt = arr_diff_1[1:] * arr_diff_1[:-1] < 0

    if type_ == "non-plateau":
        return np.hstack((0, turning_pt.astype(int), 0))

    if type_ == "any":
        plat = find_plateau_pt(arr, arr_diff_1)
        turning_pt = np.hstack((0, turning_pt.astype(int), 0))
        res = np.logical_or(turning_pt, plat)
        return res

    # Note: second discrete derivative
    arr_diff_2 = np.diff(arr_diff_1)

    rel = operator.lt if type_ == "max" else operator.gt

    interest_pt = rel(arr_diff_2, 0)
    local_m = np.logical_and(turning_pt, interest_pt)

    return np.hstack((0, local_m.astype(int), 0))


def fit_gaussian_mix(
    ts: np.ndarray,
    n_components: int = 2,
    random_state: t.Optional[int] = None,
    gaussian_model: t.Optional[sklearn.mixture.GaussianMixture] = None
) -> sklearn.mixture.GaussianMixture:
    """Fit a Gaussian Mixture model to the time-series data."""
    if gaussian_model is None or gaussian_model.n_components != n_components:
        gaussian_model = sklearn.mixture.GaussianMixture(
            n_components=n_components, random_state=random_state)
        gaussian_model.fit(X=ts.reshape(-1, 1))

    return gaussian_model


def _test() -> None:
    import matplotlib.pyplot as plt
    np.random.seed(16)
    arr = 10 * np.random.random(15)
    arr[5] = arr[6] = arr[7]

    pts_1 = find_crit_pt(arr, "any")
    pts_2 = find_crit_pt(arr, "non-plateau")
    pts_3 = find_crit_pt(arr, "plateau")
    pts_4 = find_crit_pt(arr, "min")
    pts_5 = find_crit_pt(arr, "max")

    print(pts_1.shape)
    print(pts_2.shape)
    print(pts_3.shape)
    print(pts_4.shape)
    print(pts_5.shape)

    time = np.arange(arr.size)
    plt.vlines(x=time, ymin=0, ymax=21, linestyle="dotted", color="gray")
    plt.hlines(y=np.arange(10),
               xmin=0,
               xmax=arr.size - 1,
               linestyle="dotted",
               color="black")
    plt.plot(time, 10 + arr, label="time series")
    plt.scatter(time, 8 + pts_1, label="any")
    plt.scatter(time, 6 + pts_2, label="non-plateau")
    plt.scatter(time, 4 + pts_3, label="plateau")
    plt.scatter(time, 2 + pts_4, label="min")
    plt.scatter(time, 0 + pts_5, label="max")
    plt.ylim((-10, 21.0))
    plt.legend(loc="lower center")
    plt.show()


if __name__ == "__main__":
    _test()
