"""Utility functions used ubiquitously over this library source code."""
import typing as t
import operator

import sklearn.preprocessing
import sklearn.gaussian_process
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


def process_window_size(ts: np.ndarray, window_size: t.Union[float,
                                                             int]) -> int:
    """Standardize rolling window sizes.

    The following steps are made in this method:
        1. Check if window size is stricly positive (otherwise, raise a
            ValueError exception).
        2. If 0 < window_size < 1, transform it to the corresponding integer
            size, based on the ``ts`` length. Otherwise, set window_size to
            min(len(ts), window_size).
        3. Force window_size be an odd value, to keep the reference window
            value at the center of the window.
    """
    if window_size <= 0:
        raise ValueError("'window_size' must be positive (got {})."
                         "".format(window_size))

    if 0 < window_size < 1:
        window_size = max(1, int(np.ceil(window_size * ts.size)))

    else:
        window_size = min(ts.size, window_size)

    if window_size % 2 == 0:
        # Note: forcing 'window_size' be a off number in order to the
        # reference value be exactly on the window center (and avoid
        # possibility of bias towards the larger tail)
        window_size -= 1

    return int(window_size)


def standardize_ts(ts: np.ndarray,
                   ts_scaled: t.Optional[np.ndarray] = None) -> np.ndarray:
    """Standardize (z-score normalization) time-series."""
    if ts_scaled is None:
        if not isinstance(ts, np.ndarray):
            ts = np.asarray(ts, dtype=float)

        if ts.ndim == 1:
            ts = ts.reshape(-1, 1)

        return sklearn.preprocessing.StandardScaler().fit_transform(ts).ravel()

    return ts


def get_rolling_window(
        ts: np.ndarray,
        window_size: t.Union[float, int],
        center: bool = True,
        scale: bool = True,
        ts_scaled: t.Optional[np.ndarray] = None,
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None,
) -> pd.core.window.rolling.Rolling:
    """Apply a function on time-series rolling (overlapping) windows.

    If ``center`` is True, then each rolling window is centered at the central
    instance rather than the initial instance.

    If ``scaled`` is True, the time-series will be standardized before the
    rolling window is built (or ``ts_scaled`` will be used, if not None).
    """
    window_size = process_window_size(ts=ts, window_size=window_size)

    if ts_rol_win is not None and window_size == ts_rol_win.window:
        return ts_rol_win

    if scale:
        ts = standardize_ts(ts=ts, ts_scaled=ts_scaled)

    return pd.Series(ts).rolling(window_size, center=center)


def sample_data(
        ts: np.ndarray,
        lm_sample_frac: float,
        X: t.Optional[np.ndarray] = None,
) -> t.Union[np.ndarray, t.Tuple[np.ndarray, np.ndarray]]:
    """Select ``lm_sample_frac`` percent of data from ``ts``.

    ``lm_sample_frac`` must be in (0, 1] range.

    The chosen instances are the most recent ones (on the higher indices
    of the ``ts`` array).

    ``X`` is any array of values associated to each ``ts`` observation, and
    will be sampled alongside it (mantaining the index correspondence) if
    given. Tipically, ``X`` is the timestamps of each ``ts`` observation.
    """
    if not 0.0 < lm_sample_frac <= 1.0:
        raise ValueError("'lm_sample_frac' must be in (0, 1] range (got {})."
                         "".format(lm_sample_frac))

    threshold = int(np.ceil(ts.size * lm_sample_frac))

    if threshold >= ts.size:
        if X is not None:
            return ts, X

        return ts

    if X is not None:
        return ts[-threshold:], X[-threshold:]

    return ts[-threshold:]


def find_plateau_pt(arr: np.ndarray,
                    arr_diff: t.Optional[np.ndarray] = None) -> np.ndarray:
    """Find plateau points in array.

    ``arr_diff`` is the first-order differenced ``arr``, which can be
    passed to speed up this computation since this value is needed within
    this function.
    """
    if arr_diff is None:
        arr_diff = np.diff(arr)

    arr_diff_2 = np.diff(arr_diff)

    res = np.logical_and(np.isclose(arr_diff_2, 0),
                         np.isclose(arr_diff[:-1], 0))

    return np.hstack((False, res, False))


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
        return np.hstack((False, turning_pt, False))

    if type_ == "any":
        plat = find_plateau_pt(arr, arr_diff_1)
        turning_pt = np.hstack((False, turning_pt, False))
        res = np.logical_or(turning_pt, plat)
        return res

    # Note: second discrete derivative
    arr_diff_2 = np.diff(arr_diff_1)

    rel = operator.lt if type_ == "max" else operator.gt

    interest_pt = rel(arr_diff_2, 0)
    local_m = np.logical_and(turning_pt, interest_pt)

    return np.hstack((False, local_m, False))


def fit_gaussian_process(
        ts: np.ndarray,
        random_state: t.Optional[int] = None,
        return_residuals: bool = False,
        ts_scaled: t.Optional[np.ndarray] = None,
        gaussian_model: t.Optional[
            sklearn.gaussian_process.GaussianProcessRegressor] = None,
) -> t.Union[np.ndarray, sklearn.gaussian_process.GaussianProcessRegressor]:
    """Fit a Gaussian Process model to the time-series data.

    The fitted model is returned unless ``return_residuals`` is True, which in
    this case the model residuals is returned instead.
    """
    ts_scaled = standardize_ts(ts=ts, ts_scaled=ts_scaled)

    if gaussian_model is None or return_residuals:
        timestamps = np.linspace(0, 1, ts_scaled.size).reshape(-1, 1)

    if gaussian_model is None:
        gaussian_model = sklearn.gaussian_process.GaussianProcessRegressor(
            copy_X_train=False, random_state=random_state)

        gaussian_model.fit(X=timestamps, y=ts_scaled)

    if return_residuals:
        return (ts_scaled - gaussian_model.predict(X=timestamps)).ravel()

    return gaussian_model


def calc_ioe_stats(ts: np.ndarray,
                   funcs: t.Collection[t.Callable[[np.ndarray], float]],
                   ts_scaled: t.Optional[np.ndarray] = None,
                   step_size: float = 0.05,
                   max_it: int = 1024,
                   differentiate: bool = False) -> np.ndarray:
    """Get statistics using the iterative outlier exclusion strategy.

    In the iterative outlier exclusion, a uniformly spaced set of thresholds
    over the time-series range is build in increasing order. For each threshold
    it is calculated a statistic of the diference of the timestamp values of
    instances larger or equal than the current threshold.

    Parameters
    ----------
    ts : :obj:`np.ndarray`
        One-dimensional time-series values.

    funcs : callable or list of callables
        Callables that extract the statistics from the timestamps. Every
        callable must receive a numeric list of values as the first argument
        and must return a single numeric value.

    ts_scaled : :obj:`np.ndarray`, optional
        Standardized time-series values. Used to take advantage of
        precomputations.

    step_size : float, optional (default=0.05)
        Increase of the outlier threshold in each iteration. Must be a number
        strictly positive.

    max_it : int, optional (default=1024)
        Maximum number of iterations.

    differentiate : bool, optional (default=False)
        If True, differentiate the timestamps before calculating each
        statistic. If False, all statistics will be calculated on the
        raw timestamps.

    Returns
    -------
    :obj:`np.ndarray`
        Array where each row corresponds to a distinct statistisc extracted
        from the timestamps, and each column corresponds to each iteration
        of the iterative outlier inclusive process. If ``funcs`` is a single
        function, then the return value will be flattened to a 1-D array.

    References
    ----------
    .. [1] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework
        for Automated Time-Series Phenotyping Using Massive Feature
        Extraction, Cell Systems 5: 527 (2017).
        DOI: 10.1016/j.cels.2017.10.001
    .. [2] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
        time-series analysis: the empirical structure of time series and
        their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013).
        DOI: 10.1098/rsif.2013.0048
    """
    if step_size <= 0:
        raise ValueError("'step_size' must be positive (got {})."
                         "".format(step_size))
    try:
        if len(funcs) == 0:
            raise ValueError("'funcs' is empty.")

    except TypeError:
        funcs = [funcs]  # type: ignore

    ts_scaled = standardize_ts(ts=ts, ts_scaled=ts_scaled)

    ts_abs = np.abs(ts_scaled)
    min_abs_ts, max_abs_ts = np.quantile(ts_abs, (0, 1))

    step_size *= np.std(ts_abs)
    _max_it = min(max_it, int(np.ceil(max_abs_ts / step_size)))
    ioe_stats = np.zeros((_max_it, len(funcs)))
    threshold = min_abs_ts
    it = 0

    while it < _max_it:
        threshold += step_size
        outlier_tsteps = np.flatnonzero(ts_abs >= threshold)

        if (outlier_tsteps.size < 0.02 * ts_scaled.size
                or outlier_tsteps.size <= 1):
            break

        diff_tsteps = np.diff(outlier_tsteps, int(differentiate))

        ioe_stats[it, :] = [func(diff_tsteps) for func in funcs]
        it += 1

    if len(funcs) == 1:
        return ioe_stats.ravel()

    # Note: transposing in order to each statistic be represented by a row,
    # and not a column.
    return ioe_stats.T


def apply_on_samples(ts: np.ndarray,
                     func: t.Callable[..., float],
                     num_samples: int = 128,
                     sample_size_frac: float = 0.2,
                     random_state: t.Optional[int] = None,
                     **kwargs) -> np.ndarray:
    """Apply ``func`` on time-series random samples.

    The samples are ``num_samples`` consecutive observations from the
    time-series, starting from random positions.

    Parameters
    ----------
    ts : :obj:`np.ndarray`
        One-dimensional time-series values.

    func : callable
        Function to apply on each sample. Must receive an list of numeric
        values as argument and return a single numeric valud.

    num_samples : int, optional (default=128)
        Number of samples. If zero or less, a empty array will be returned.

    sample_size_frac : float, optional (default=0.2)
        Size of each sample as a fraction of the size of the original
        time-series. Must be in (0, 1) range.

    random_state : int, optional
        Random seed to ensure reproducibility.

    kwargs:
        Extra arguments to ``func``.

    Returns
    -------
    :obj:`np.ndarray`
        ``func`` values from each time-series sample.
    """
    if num_samples <= 0:
        return np.empty(0)

    if not 0 < sample_size_frac < 1:
        raise ValueError("'sample_size_frac' must be in (0, 1) "
                         "range (got {}).".format(sample_size_frac))

    if random_state is not None:
        np.random.seed(random_state)

    sample_size = int(np.ceil(ts.size * sample_size_frac))
    start_inds = np.random.randint(ts.size - sample_size + 1, size=num_samples)

    res = np.array([
        func(ts[s_ind:s_ind + sample_size], **kwargs) for s_ind in start_inds
    ])

    return res


def discretize(ts: np.ndarray,
               num_bins: int,
               strategy: str = "equal-width",
               dtype: t.Type = int) -> np.ndarray:
    """Discretize a time-series using a histogram.

    Parameters
    ----------
    ts : :obj:`np.ndarray`
        One-dimensional time-series values.

    num_bins : int
        Number of bins in the histogram.

    strategy : {`equal-width`,`equiprobable`}, optional (default="equal-width")
            Strategy used to define the histogram bins. Must be either
            `equal-width` (bins with equal with) or `equiprobable` (bins
            with the same amount of observations within).

    dtype : type, optional (default=int)
        Output type of the discretized time-series.

    Returns
    -------
    :obj:`np.ndarray`
        Discretized time-series with the selected strategy.
    """
    VALID_METHODS = {"equal-width", "equiprobable"}

    if strategy not in VALID_METHODS:
        raise ValueError("'strategy' must be in {} (got {})."
                         "".format(VALID_METHODS, strategy))

    if strategy == "equal-width":
        bins = np.histogram(ts, num_bins)[1][:-1]

    elif strategy == "equiprobable":
        bins = np.quantile(ts, np.linspace(0, 1, num_bins + 1)[:-1])

    ts_disc = np.digitize(ts, bins)

    return ts_disc.astype(dtype)
