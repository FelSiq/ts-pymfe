import typing as t
import operator

import sklearn.preprocessing
import sklearn.mixture
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
    window_size: t.Union[float, int],
    center: bool = True,
    ts_scaled: t.Optional[np.ndarray] = None,
) -> pd.core.window.rolling.Rolling:
    """Apply a function on time-series rolling (overlapping) windows.

    If ``center`` is True, then each rolling window is centered at the
    central instance rather than the initial instance.
    """
    if window_size <= 0:
        raise ValueError("'window_size' must be positive (got {})."
                         "".format(window_size))

    if ts_scaled is None:
        ts_scaled = sklearn.preprocessing.StandardScaler().fit_transform(
            ts.reshape(-1, 1)).ravel()

    if 0 < window_size < 1:
        window_size = max(1, int(np.ceil(window_size * ts.size)))

    else:
        window_size = min(ts.size, window_size)

    if window_size % 2 == 0:
        # Note: forcing 'window_size' be a off number in order to the
        # reference value be exactly on the window center (and avoid
        # possibility of bias towards the larger tail)
        window_size -= 1

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
    return_residuals: bool = False,
    gaussian_model: t.Optional[sklearn.mixture.GaussianMixture] = None
) -> t.Union[np.ndarray, sklearn.mixture.GaussianMixture]:
    """Fit a Gaussian Mixture model to the time-series data.

    The fitted model is returned unless ``return_residuals`` is
    True, which in this case the model residuals is returned
    instead.
    """
    if ts.ndim == 1:
        ts = ts.reshape(-1, 1)

    if gaussian_model is None or gaussian_model.n_components != n_components:
        gaussian_model = sklearn.mixture.GaussianMixture(
            n_components=n_components, random_state=random_state)

        if return_residuals:
            return (ts - gaussian_model.fit_predict(X=ts)).ravel()

        gaussian_model.fit(X=ts)

    if return_residuals:
        return (ts - gaussian_model.predict(X=ts)).ravel()

    return gaussian_model


def calc_ioi_stats(ts: np.ndarray,
                   funcs: t.Union[t.Callable[[np.ndarray], float],
                                  t.Iterable[t.Callable[[np.ndarray], float]]],
                   ts_scaled: t.Optional[np.ndarray] = None,
                   step_size: float = 0.05,
                   differentiate: bool = False) -> np.ndarray:
    """TODO.

    https://github.com/benfulcher/hctsa/blob/master/Operations/DN_OutlierInclude.m
    """
    try:
        if len(funcs) == 0:
            raise ValueError("'funcs' is empty.")

    except:
        funcs = [funcs]

    ts_scaled = standardize_ts(ts=ts, ts_scaled=ts_scaled)

    # Note: originally, the step size of the threshold is calculated
    # as step_size * std(ts). However, we are considering just the
    # normalized time-series and, therefore, std(ts_scaled) = 1.
    # This means that the step size is actually just the step_size.
    ts_abs = np.abs(ts_scaled)
    max_abs_ts = np.max(ts_abs)

    res = []  # type: t.List[float]
    threshold = 0.0

    while threshold < max_abs_ts:
        threshold += step_size
        outlier_tsteps = np.flatnonzero(ts_abs >= threshold)

        if (outlier_tsteps.size < 0.02 * ts_scaled.size
                or outlier_tsteps.size <= 1):
            break

        diff_tsteps = np.diff(outlier_tsteps, int(differentiate))

        res.append([func(diff_tsteps) for func in funcs])

    res = np.asarray(res, dtype=float)

    if res.shape[1] == 1:
        return res.ravel()

    return res


def apply_on_ts_samples(ts: np.ndarray,
                        func: t.Callable[[np.ndarray], float],
                        num_samples: int = 128,
                        sample_size_frac: float = 0.2,
                        random_state: t.Optional[int] = None,
                        **kwargs) -> np.ndarray:
    """TODO."""
    if not 0 < sample_size_frac < 1:
        raise ValueError("'sample_size_frac' must be in (0, 1) "
                         "range (got {}).".format(sample_size_frac))

    if random_state is not None:
        np.random.seed(random_state)

    sample_size = int(np.ceil(ts.size * sample_size_frac))
    start_inds = np.random.randint(ts.size - sample_size + 1, size=num_samples)

    res = np.array([
        func(ts[s_ind:s_ind + sample_size], **kwargs) for s_ind in start_inds
    ],
                   dtype=float)

    # Note: the original metafeatures are the mean value of
    # 'result'. However, to enable summarization,
    # here we return all the values.
    return res


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
