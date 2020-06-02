"""Module dedicated to localized statistical time-series meta-features."""
import typing as t

import pandas as pd
import scipy.stats
import numpy as np

import pymfe._utils as _utils
import pymfe._summary as _summary

try:
    import pymfe.stat_tests as stat_tests

except ImportError:
    pass

try:
    import pymfe.autocorr as autocorr

except ImportError:
    pass

try:
    import pymfe.info_theory as info_theory

except ImportError:
    pass


class MFETSLocalStats:
    """Extract time-series meta-features from Local Statistics group."""
    @classmethod
    def precompute_ts_scaled(cls, ts: np.ndarray,
                             **kwargs) -> t.Dict[str, np.ndarray]:
        """Precompute a standardized time series.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        kwargs:
            Additional arguments and previous precomputed items. May
            speed up this precomputation.

        Returns
        -------
        dict
            The following precomputed item is returned:
                * ``ts_scaled`` (:obj:`np.ndarray`): standardized time-series
                    values (z-score).
        """
        precomp_vals = {}  # type: t.Dict[str, np.ndarray]

        if "ts_scaled" not in kwargs:
            precomp_vals["ts_scaled"] = _utils.standardize_ts(ts=ts)

        return precomp_vals

    @classmethod
    def precompute_rolling_window(
            cls,
            ts: np.ndarray,
            window_size: t.Union[int, float] = 0.1,
            **kwargs) -> t.Dict[str, pd.core.window.rolling.Rolling]:
        """Precompute a configured rolling window.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        window_size : int or float, optional (default=0.1)
            Size of the window. Must be strictly positive.
            If int >= 1, this argument defines the window size.
            If 0 < float < 1, this argument defines the fraction of the
            time-series length used as the window size.

        kwargs:
            Additional arguments and previous precomputed items. May
            speed up this precomputation.

        Returns
        -------
        dict
            The following precomputed item is returned:
                * ``ts_tol_win`` (:obj:`pd.core.window.rolling.Rolling`):
                    Configured rolling window object.

            The following item is necessary and, therefore, also precomputed
            if necessary:
                * ``ts_scaled`` (:obj:`np.ndarray`): standardized time-series
                    values (z-score).
        """
        precomp_vals = {}  # type: t.Dict[str, pd.core.window.rolling.Rolling]

        ts_scaled = kwargs.get("ts_scaled")

        if ts_scaled is None:
            precomp_vals.update(cls.precompute_ts_scaled(ts=ts))
            ts_scaled = precomp_vals["ts_scaled"]

        if "ts_tol_win" not in kwargs:
            ts_rol_win = _utils.get_rolling_window(ts=ts,
                                                   window_size=window_size,
                                                   ts_scaled=ts_scaled)

            precomp_vals["ts_tol_win"] = ts_rol_win

        return precomp_vals

    @classmethod
    def _rol_stat_postprocess(cls,
                              rolling_stat: pd.Series,
                              remove_nan: bool = True,
                              diff_order: int = 0,
                              diff_lag: int = 0,
                              abs_value: bool = False) -> np.ndarray:
        """Postprocess a pandas Series rolling window result.

        Arguments
        ---------
        rolling_stat : :obj:`pd.Series`
            Result of a pandas rolling window statistic.

        remove_nan : bool
            If True, remove the `nan` values. Useful mainly due the rolling
            window statistic corners that could not be computed due to not
            having sufficient neighbors.

        diff_order : int
            Order of differentiation. If 0 (or less), the values will not
            be differentied.

        diff_lag : int
            Lag of differentiation. If less than 1, it will assummed lag 1.
            Used only if ``diff_order`` >= 1.

        abs_value : bool
            If True, return the absolute value of the postprocessed array.

        Returns
        -------
        :obj:`np.ndarray`
            Postprocessed rolling statistic array.
        """
        if not isinstance(rolling_stat, pd.Series):
            rolling_stat = pd.Series(rolling_stat)

        if remove_nan:
            rolling_stat = rolling_stat[~np.isnan(rolling_stat)]

        if diff_order > 0:
            # Note: pandas.Series.diff(ts, n) calculate the first order
            # difference shifted by 'n', while the numpy.diff calculate
            # the n-th order difference shifted by a single value.

            if diff_lag > 1:
                for _ in np.arange(diff_order):
                    rolling_stat = rolling_stat.diff(periods=diff_lag)

            else:
                rolling_stat = np.diff(rolling_stat, n=diff_order)

        if abs_value:
            rolling_stat = np.abs(rolling_stat)

        if isinstance(rolling_stat, np.ndarray):
            return rolling_stat

        return rolling_stat.values

    @classmethod
    def _moving_stat_shift(cls,
                           ts: np.ndarray,
                           stat_func: t.Callable[..., np.ndarray],
                           window_size: t.Union[int, float] = 0.1,
                           diff_order: int = 1,
                           diff_lag: int = 1,
                           abs_value: bool = True,
                           remove_nan: bool = True,
                           ts_scaled: t.Optional[np.ndarray] = None,
                           ts_rol_win: t.Optional[
                               pd.core.window.rolling.Rolling] = None,
                           **kwargs) -> np.ndarray:
        """Calculate the n-lagged `m`th-order differenced of moving statistics.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        stat_func : callable
            Function to extract the local statistics.

        window_size : int or float, optional (default=0.1)
            Size of the window. Must be strictly positive.
            If int >= 1, this argument defines the window size.
            If 0 < float < 1, this argument defines the fraction of the
            time-series length used as the window size.

        diff_order : int, optional (default=1)
            Order of differentiation. If this argument get a value of 0 or
            less, then no differentiation will be performed.

        diff_lag : int, optional (default=1)
            Lag of each differentiation (among the moving statistics). If
            a value lower than 1 is given, then it is assumed lag 1.

        abs_value : bool, optional (default=True)
            If True, return the absolute value of the result.

        remove_nan : bool, optional (default=True)
            If True, remove `nan` values from the ``stat_func`` results
            before any post-processing.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        ts_rol_win : :obj:`pd.core.window.rolling.Rolling`, optional
            Configured rolling window. Used to take advantage of
            precomputations.

        kwargs:
            Additional arguments for the ``stat_func`` callable.

        Returns
        -------
        :obj:`np.ndarray`
            Post-processed rolling statistic values.
        """
        rolling_stat = stat_func(ts=ts,
                                 window_size=window_size,
                                 remove_nan=remove_nan,
                                 ts_scaled=ts_scaled,
                                 ts_rol_win=ts_rol_win,
                                 **kwargs)

        rolling_stat_shifts = cls._rol_stat_postprocess(rolling_stat,
                                                        remove_nan=False,
                                                        diff_order=diff_order,
                                                        diff_lag=diff_lag,
                                                        abs_value=abs_value)

        return rolling_stat_shifts

    @classmethod
    def ft_moving_avg(
            cls,
            ts: np.ndarray,
            window_size: t.Union[int, float] = 0.1,
            remove_nan: bool = True,
            ts_scaled: t.Optional[np.ndarray] = None,
            ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None,
    ) -> np.ndarray:
        """Moving average of overlapping windows.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        window_size : int or float, optional (default=0.1)
            Size of the window. Must be strictly positive.
            If int >= 1, this argument defines the window size.
            If 0 < float < 1, this argument defines the fraction of the
            time-series length used as the window size.

        remove_nan : bool, optional (default=True)
            If True, remove `nan` values that may be generated while collecting
            the rolling statistics before any post-processing.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        ts_rol_win : :obj:`pd.core.window.rolling.Rolling`, optional
            Configured rolling window. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Moving average from overlapping windows in time-series values.
        """
        ts_rol_win = _utils.get_rolling_window(ts=ts,
                                               window_size=window_size,
                                               ts_scaled=ts_scaled,
                                               ts_rol_win=ts_rol_win)

        rolling_stat = ts_rol_win.mean()

        return cls._rol_stat_postprocess(rolling_stat, remove_nan=remove_nan)

    @classmethod
    def ft_moving_avg_shift(
            cls,
            ts: np.ndarray,
            window_size: t.Union[int, float] = 0.1,
            diff_order: int = 1,
            diff_lag: int = 1,
            abs_value: bool = True,
            remove_nan: bool = True,
            ts_scaled: t.Optional[np.ndarray] = None,
            ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None,
    ) -> np.ndarray:
        """Absolute differenced moving average of overlapping windows.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        window_size : int or float, optional (default=0.1)
            Size of the window. Must be strictly positive.
            If int >= 1, this argument defines the window size.
            If 0 < float < 1, this argument defines the fraction of the
            time-series length used as the window size.

        diff_order : int, optional (default=1)
            Order of differentiation. If this argument get a value of 0 or
            less, then no differentiation will be performed.

        diff_lag : int, optional (default=1)
            Lag of each differentiation (among the moving statistics). If
            a value lower than 1 is given, then it is assumed lag 1.

        abs_value : bool, optional (default=True)
            If True, return the absolute value of the result.

        remove_nan : bool, optional (default=True)
            If True, remove `nan` values that may be generated while collecting
            the rolling statistics before any post-processing.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        ts_rol_win : :obj:`pd.core.window.rolling.Rolling`, optional
            Configured rolling window. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Post-processed moving average from overlapping windows in
            time-series values.

        References
        ----------
        .. [1] R. J. Hyndman, E. Wang and N. Laptev, "Large-Scale Unusual Time
            Series Detection," 2015 IEEE International Conference on Data
            Mining Workshop (ICDMW), Atlantic City, NJ, 2015, pp. 1616-1619,
            doi: 10.1109/ICDMW.2015.104.
        .. [2] Hyndman, R. J., Wang, E., Kang, Y., & Talagala, T. (2018).
            tsfeatures: Time series feature extraction. R package version 0.1.
        .. [3] Pablo Montero-Manso, George Athanasopoulos, Rob J. Hyndman,
            Thiyanga S. Talagala, FFORMA: Feature-based forecast model
            averaging, International Journal of Forecasting, Volume 36, Issue
            1, 2020, Pages 86-92, ISSN 0169-2070,
            https://doi.org/10.1016/j.ijforecast.2019.02.011.
        """
        rolling_stat_shift = cls._moving_stat_shift(
            ts=ts,
            stat_func=cls.ft_moving_avg,
            window_size=window_size,
            diff_order=diff_order,
            diff_lag=diff_lag,
            abs_value=abs_value,
            remove_nan=remove_nan,
            ts_scaled=ts_scaled,
            ts_rol_win=ts_rol_win)

        return rolling_stat_shift

    @classmethod
    def ft_moving_var(
            cls,
            ts: np.ndarray,
            window_size: t.Union[int, float] = 0.1,
            ddof: int = 1,
            remove_nan: bool = True,
            ts_scaled: t.Optional[np.ndarray] = None,
            ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None,
    ) -> np.ndarray:
        """Moving variance of overlapping windows.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        window_size : int or float, optional (default=0.1)
            Size of the window. Must be strictly positive.
            If int >= 1, this argument defines the window size.
            If 0 < float < 1, this argument defines the fraction of the
            time-series length used as the window size.

        ddof : int, optional (default=1)
            Degrees of freedom for the variance calculation.

        remove_nan : bool, optional (default=True)
            If True, remove `nan` values that may be generated while collecting
            the rolling statistics before any post-processing.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        ts_rol_win : :obj:`pd.core.window.rolling.Rolling`, optional
            Configured rolling window. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Moving variance from overlapping windows in time-series values.

        References
        ----------
        .. [1] R. J. Hyndman, E. Wang and N. Laptev, "Large-Scale Unusual Time
            Series Detection," 2015 IEEE International Conference on Data
            Mining Workshop (ICDMW), Atlantic City, NJ, 2015, pp. 1616-1619,
            doi: 10.1109/ICDMW.2015.104.
        .. [2] Hyndman, R. J., Wang, E., Kang, Y., & Talagala, T. (2018).
            tsfeatures: Time series feature extraction. R package version 0.1.
        .. [3] Pablo Montero-Manso, George Athanasopoulos, Rob J. Hyndman,
            Thiyanga S. Talagala, FFORMA: Feature-based forecast model
            averaging, International Journal of Forecasting, Volume 36, Issue
            1, 2020, Pages 86-92, ISSN 0169-2070,
            https://doi.org/10.1016/j.ijforecast.2019.02.011.
        """
        ts_rol_win = _utils.get_rolling_window(ts=ts,
                                               window_size=window_size,
                                               ts_scaled=ts_scaled,
                                               ts_rol_win=ts_rol_win)

        rolling_stat = ts_rol_win.var(ddof=ddof)

        return cls._rol_stat_postprocess(rolling_stat, remove_nan=remove_nan)

    @classmethod
    def ft_moving_var_shift(
            cls,
            ts: np.ndarray,
            window_size: t.Union[int, float] = 0.1,
            ddof: int = 1,
            diff_order: int = 1,
            diff_lag: int = 1,
            abs_value: bool = True,
            remove_nan: bool = True,
            ts_scaled: t.Optional[np.ndarray] = None,
            ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None,
    ) -> np.ndarray:
        """Absolute differenced moving variance of overlapping windows.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        window_size : int or float, optional (default=0.1)
            Size of the window. Must be strictly positive.
            If int >= 1, this argument defines the window size.
            If 0 < float < 1, this argument defines the fraction of the
            time-series length used as the window size.

        ddof : int, optional (default=1)
            Degrees of freedom for the variance calculation.

        diff_order : int, optional (default=1)
            Order of differentiation. If this argument get a value of 0 or
            less, then no differentiation will be performed.

        diff_lag : int, optional (default=1)
            Lag of each differentiation (among the moving statistics). If
            a value lower than 1 is given, then it is assumed lag 1.

        abs_value : bool, optional (default=True)
            If True, return the absolute value of the result.

        remove_nan : bool, optional (default=True)
            If True, remove `nan` values that may be generated while collecting
            the rolling statistics before any post-processing.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        ts_rol_win : :obj:`pd.core.window.rolling.Rolling`, optional
            Configured rolling window. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Post-processed moving variance from overlapping windows in
            time-series values.

        References
        ----------
        .. [1] R. J. Hyndman, E. Wang and N. Laptev, "Large-Scale Unusual Time
            Series Detection," 2015 IEEE International Conference on Data
            Mining Workshop (ICDMW), Atlantic City, NJ, 2015, pp. 1616-1619,
            doi: 10.1109/ICDMW.2015.104.
        .. [2] Hyndman, R. J., Wang, E., Kang, Y., & Talagala, T. (2018).
            tsfeatures: Time series feature extraction. R package version 0.1.
        .. [3] Pablo Montero-Manso, George Athanasopoulos, Rob J. Hyndman,
            Thiyanga S. Talagala, FFORMA: Feature-based forecast model
            averaging, International Journal of Forecasting, Volume 36, Issue
            1, 2020, Pages 86-92, ISSN 0169-2070,
            https://doi.org/10.1016/j.ijforecast.2019.02.011.
        """
        rolling_stat_shift = cls._moving_stat_shift(
            ts=ts,
            stat_func=cls.ft_moving_var,
            window_size=window_size,
            diff_order=diff_order,
            diff_lag=diff_lag,
            abs_value=abs_value,
            remove_nan=remove_nan,
            ts_scaled=ts_scaled,
            ts_rol_win=ts_rol_win,
            ddof=ddof)

        return rolling_stat_shift

    @classmethod
    def ft_moving_sd(
            cls,
            ts: np.ndarray,
            window_size: t.Union[int, float] = 0.1,
            ddof: int = 1,
            remove_nan: bool = True,
            ts_scaled: t.Optional[np.ndarray] = None,
            ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None,
    ) -> np.ndarray:
        """Moving standard deviation of overlapping windows.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        window_size : int or float, optional (default=0.1)
            Size of the window. Must be strictly positive.
            If int >= 1, this argument defines the window size.
            If 0 < float < 1, this argument defines the fraction of the
            time-series length used as the window size.

        ddof : int, optional (default=1)
            Degrees of freedom for the standard deviation calculation.

        remove_nan : bool, optional (default=True)
            If True, remove `nan` values that may be generated while collecting
            the rolling statistics before any post-processing.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        ts_rol_win : :obj:`pd.core.window.rolling.Rolling`, optional
            Configured rolling window. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Moving standard deviation from overlapping windows in
            time-series values.
        """
        ts_rol_win = _utils.get_rolling_window(ts=ts,
                                               window_size=window_size,
                                               ts_scaled=ts_scaled,
                                               ts_rol_win=ts_rol_win)

        rolling_stat = ts_rol_win.std(ddof=ddof)

        return cls._rol_stat_postprocess(rolling_stat, remove_nan=remove_nan)

    @classmethod
    def ft_moving_sd_shift(
            cls,
            ts: np.ndarray,
            window_size: t.Union[int, float] = 0.1,
            ddof: int = 1,
            diff_order: int = 1,
            diff_lag: int = 1,
            abs_value: bool = True,
            remove_nan: bool = True,
            ts_scaled: t.Optional[np.ndarray] = None,
            ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None,
    ) -> np.ndarray:
        """Abs. diff. moving standard deviation of overlapping windows.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        window_size : int or float, optional (default=0.1)
            Size of the window. Must be strictly positive.
            If int >= 1, this argument defines the window size.
            If 0 < float < 1, this argument defines the fraction of the
            time-series length used as the window size.

        ddof : int, optional (default=1)
            Degrees of freedom for the standard deviation calculation.

        diff_order : int, optional (default=1)
            Order of differentiation. If this argument get a value of 0 or
            less, then no differentiation will be performed.

        diff_lag : int, optional (default=1)
            Lag of each differentiation (among the moving statistics). If
            a value lower than 1 is given, then it is assumed lag 1.

        abs_value : bool, optional (default=True)
            If True, return the absolute value of the result.

        remove_nan : bool, optional (default=True)
            If True, remove `nan` values that may be generated while collecting
            the rolling statistics before any post-processing.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        ts_rol_win : :obj:`pd.core.window.rolling.Rolling`, optional
            Configured rolling window. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Post-processed moving standard deviation from non-overlapping
            windows in time-series values.
        """
        rolling_stat_shift = cls._moving_stat_shift(ts=ts,
                                                    stat_func=cls.ft_moving_sd,
                                                    window_size=window_size,
                                                    diff_order=diff_order,
                                                    diff_lag=diff_lag,
                                                    abs_value=abs_value,
                                                    remove_nan=remove_nan,
                                                    ts_scaled=ts_scaled,
                                                    ts_rol_win=ts_rol_win,
                                                    ddof=ddof)

        return rolling_stat_shift

    @classmethod
    def ft_moving_skewness(
            cls,
            ts: np.ndarray,
            window_size: t.Union[int, float] = 0.1,
            method: int = 3,
            unbiased: bool = False,
            remove_nan: bool = True,
            ts_scaled: t.Optional[np.ndarray] = None,
            ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None,
    ) -> np.ndarray:
        """Moving skewness of overlapping windows.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        window_size : int or float, optional (default=0.1)
            Size of the window. Must be strictly positive.
            If int >= 1, this argument defines the window size.
            If 0 < float < 1, this argument defines the fraction of the
            time-series length used as the window size.

        method : int, optional (default=3)
            Defines the strategy used for estimate data skewness. This argument
            is used fo compatibility with R package `e1071`. The options must
            be one of the following:

            +--------+-----------------------------------------------+
            |Option  | Formula                                       |
            +--------+-----------------------------------------------+
            |1       | Skew_1 = m_3 / m_2**(3/2)                     |
            |        | (default of ``scipy.stats``)                  |
            +--------+-----------------------------------------------+
            |2       | Skew_2 = Skew_1 * sqrt(n(n-1)) / (n-2)        |
            +--------+-----------------------------------------------+
            |3       | Skew_3 = m_3 / s**3 = Skew_1 ((n-1)/n)**(3/2) |
            +--------+-----------------------------------------------+

            Where `n` is the number of instances in ``ts``, `s` is the standard
            deviation of each attribute in ``ts``, and `m_i` is the ith
            statistical momentum of each attribute in ``ts``.

            Note that if the selected method is unable to be calculated due to
            division by zero, then the first method will be used instead.

        unbiased : bool, optional
            If True, then the calculations are corrected for statistical bias.

        remove_nan : bool, optional (default=True)
            If True, remove `nan` values that may be generated while collecting
            the rolling statistics before any post-processing.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        ts_rol_win : :obj:`pd.core.window.rolling.Rolling`, optional
            Configured rolling window. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Moving skewness from overlapping windows in time-series values.
        """
        ts_rol_win = _utils.get_rolling_window(ts=ts,
                                               window_size=window_size,
                                               ts_scaled=ts_scaled,
                                               ts_rol_win=ts_rol_win)

        rolling_stat = ts_rol_win.apply(
            _summary.sum_skewness,
            kwargs=dict(method=method, bias=~unbiased))

        return cls._rol_stat_postprocess(rolling_stat, remove_nan=remove_nan)

    @classmethod
    def ft_moving_skewness_shift(
            cls,
            ts: np.ndarray,
            window_size: t.Union[int, float] = 0.1,
            method: int = 3,
            unbiased: bool = False,
            diff_order: int = 1,
            diff_lag: int = 1,
            abs_value: bool = True,
            remove_nan: bool = True,
            ts_scaled: t.Optional[np.ndarray] = None,
            ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None,
    ) -> np.ndarray:
        """Absolute differenced moving skewness of overlapping windows.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        window_size : int or float, optional (default=0.1)
            Size of the window. Must be strictly positive.
            If int >= 1, this argument defines the window size.
            If 0 < float < 1, this argument defines the fraction of the
            time-series length used as the window size.

        method : int, optional (default=3)
            Defines the strategy used for estimate data skewness. This argument
            is used fo compatibility with R package `e1071`. The options must
            be one of the following:

            +--------+-----------------------------------------------+
            |Option  | Formula                                       |
            +--------+-----------------------------------------------+
            |1       | Skew_1 = m_3 / m_2**(3/2)                     |
            |        | (default of ``scipy.stats``)                  |
            +--------+-----------------------------------------------+
            |2       | Skew_2 = Skew_1 * sqrt(n(n-1)) / (n-2)        |
            +--------+-----------------------------------------------+
            |3       | Skew_3 = m_3 / s**3 = Skew_1 ((n-1)/n)**(3/2) |
            +--------+-----------------------------------------------+

            Where `n` is the number of instances in ``ts``, `s` is the standard
            deviation of each attribute in ``ts``, and `m_i` is the ith
            statistical momentum of each attribute in ``ts``.

            Note that if the selected method is unable to be calculated due to
            division by zero, then the first method will be used instead.

        unbiased : bool, optional
            If True, then the calculations are corrected for statistical bias.

        diff_order : int, optional (default=1)
            Order of differentiation. If this argument get a value of 0 or
            less, then no differentiation will be performed.

        diff_lag : int, optional (default=1)
            Lag of each differentiation (among the moving statistics). If
            a value lower than 1 is given, then it is assumed lag 1.

        abs_value : bool, optional (default=True)
            If True, return the absolute value of the result.

        remove_nan : bool, optional (default=True)
            If True, remove `nan` values that may be generated while collecting
            the rolling statistics before any post-processing.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        ts_rol_win : :obj:`pd.core.window.rolling.Rolling`, optional
            Configured rolling window. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Post-processed moving skewness from overlapping windows in
            time-series values.
        """
        rolling_stat_shift = cls._moving_stat_shift(
            ts=ts,
            stat_func=cls.ft_moving_skewness,
            window_size=window_size,
            diff_order=diff_order,
            diff_lag=diff_lag,
            abs_value=abs_value,
            remove_nan=remove_nan,
            ts_scaled=ts_scaled,
            ts_rol_win=ts_rol_win,
            method=method,
            unbiased=unbiased)

        return rolling_stat_shift

    @classmethod
    def ft_moving_kurtosis(
            cls,
            ts: np.ndarray,
            window_size: t.Union[int, float] = 0.1,
            method: int = 3,
            unbiased: bool = False,
            remove_nan: bool = True,
            ts_scaled: t.Optional[np.ndarray] = None,
            ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None,
    ) -> np.ndarray:
        """Moving kurtosis of overlapping windows.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        window_size : int or float, optional (default=0.1)
            Size of the window. Must be strictly positive.
            If int >= 1, this argument defines the window size.
            If 0 < float < 1, this argument defines the fraction of the
            time-series length used as the window size.

        method : int, optional (default=3)
            Defines the strategy used for estimate data kurtosis. Used for
            total compatibility with R package ``e1071``. This option must be
            one of the following:

            +--------+-----------------------------------------------+
            |Method  | Formula                                       |
            +--------+-----------------------------------------------+
            |1       | Kurt_1 = (m_4 / m_2**2 - 3)                   |
            |        | (default of `scipy.stats` package)            |
            +--------+-----------------------------------------------+
            |2       | Kurt_2 = (((n+1) * Kurt_1 + 6) * (n-1) / f_2),|
            |        | f_2 = ((n-2)*(n-3))                           |
            +--------+-----------------------------------------------+
            |3       | Kurt_3 = (m_4 / s**4 - 3)                     |
            |        |        = ((Kurt_1+3) * (1 - 1/n)**2 - 3)      |
            +--------+-----------------------------------------------+

            Where `n` is the number of instances in ``ts``, `s` is the standard
            deviation of each attribute in ``ts``, and `m_i` is the ith
            statistical momentum of each attribute in ``ts``.

            Note that if the selected method is unable to be calculated due
            to division by zero, then the first method is used instead.

        unbiased : bool, optional
            If True, then the calculations are corrected for statistical bias.

        remove_nan : bool, optional (default=True)
            If True, remove `nan` values that may be generated while collecting
            the rolling statistics before any post-processing.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        ts_rol_win : :obj:`pd.core.window.rolling.Rolling`, optional
            Configured rolling window. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Moving kurtosis from overlapping windows in time-series values.
        """
        ts_rol_win = _utils.get_rolling_window(ts=ts,
                                               window_size=window_size,
                                               ts_scaled=ts_scaled,
                                               ts_rol_win=ts_rol_win)

        rolling_stat = ts_rol_win.apply(
            _summary.sum_kurtosis,
            kwargs=dict(method=method, bias=~unbiased))

        return cls._rol_stat_postprocess(rolling_stat, remove_nan=remove_nan)

    @classmethod
    def ft_moving_kurtosis_shift(
            cls,
            ts: np.ndarray,
            window_size: t.Union[int, float] = 0.1,
            method: int = 3,
            unbiased: bool = False,
            diff_order: int = 1,
            diff_lag: int = 1,
            abs_value: bool = True,
            remove_nan: bool = True,
            ts_scaled: t.Optional[np.ndarray] = None,
            ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None,
    ) -> np.ndarray:
        """Absolute differenced moving kurtosis of overlapping windows.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        window_size : int or float, optional (default=0.1)
            Size of the window. Must be strictly positive.
            If int >= 1, this argument defines the window size.
            If 0 < float < 1, this argument defines the fraction of the
            time-series length used as the window size.

        method : int, optional (default=3)
            Defines the strategy used for estimate data kurtosis. Used for
            total compatibility with R package ``e1071``. This option must be
            one of the following:

            +--------+-----------------------------------------------+
            |Method  | Formula                                       |
            +--------+-----------------------------------------------+
            |1       | Kurt_1 = (m_4 / m_2**2 - 3)                   |
            |        | (default of `scipy.stats` package)            |
            +--------+-----------------------------------------------+
            |2       | Kurt_2 = (((n+1) * Kurt_1 + 6) * (n-1) / f_2),|
            |        | f_2 = ((n-2)*(n-3))                           |
            +--------+-----------------------------------------------+
            |3       | Kurt_3 = (m_4 / s**4 - 3)                     |
            |        |        = ((Kurt_1+3) * (1 - 1/n)**2 - 3)      |
            +--------+-----------------------------------------------+

            Where `n` is the number of instances in ``ts``, `s` is the standard
            deviation of each attribute in ``ts``, and `m_i` is the ith
            statistical momentum of each attribute in ``ts``.

            Note that if the selected method is unable to be calculated due
            to division by zero, then the first method is used instead.

        unbiased : bool, optional
            If True, then the calculations are corrected for statistical bias.

        diff_order : int, optional (default=1)
            Order of differentiation. If this argument get a value of 0 or
            less, then no differentiation will be performed.

        diff_lag : int, optional (default=1)
            Lag of each differentiation (among the moving statistics). If
            a value lower than 1 is given, then it is assumed lag 1.

        abs_value : bool, optional (default=True)
            If True, return the absolute value of the result.

        remove_nan : bool, optional (default=True)
            If True, remove `nan` values that may be generated while collecting
            the rolling statistics before any post-processing.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        ts_rol_win : :obj:`pd.core.window.rolling.Rolling`, optional
            Configured rolling window. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Post-processed moving kurtosis from overlapping windows in
            time-series values.
        """
        rolling_stat_shift = cls._moving_stat_shift(
            ts=ts,
            stat_func=cls.ft_moving_kurtosis,
            window_size=window_size,
            diff_order=diff_order,
            diff_lag=diff_lag,
            abs_value=abs_value,
            remove_nan=remove_nan,
            ts_scaled=ts_scaled,
            ts_rol_win=ts_rol_win,
            method=method,
            unbiased=unbiased)

        return rolling_stat_shift

    @classmethod
    def ft_moving_acf(
            cls,
            ts: np.ndarray,
            window_size: t.Union[int, float] = 0.1,
            unbiased: bool = True,
            remove_nan: bool = True,
            ts_scaled: t.Optional[np.ndarray] = None,
            ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None,
    ) -> np.ndarray:
        """Moving autocorrelation of overlapping windows.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        window_size : int or float, optional (default=0.1)
            Size of the window. Must be strictly positive.
            If int >= 1, this argument defines the window size.
            If 0 < float < 1, this argument defines the fraction of the
            time-series length used as the window size.

        unbiased : bool, optional
            If True, then the calculations are corrected for statistical bias.

        remove_nan : bool, optional (default=True)
            If True, remove `nan` values that may be generated while collecting
            the rolling statistics before any post-processing.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        ts_rol_win : :obj:`pd.core.window.rolling.Rolling`, optional
            Configured rolling window. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Moving autocorrelation from overlapping windows in time-series
            values.
        """
        ts_rol_win = _utils.get_rolling_window(ts=ts,
                                               window_size=window_size,
                                               ts_scaled=ts_scaled,
                                               ts_rol_win=ts_rol_win)

        rolling_stat = ts_rol_win.apply(autocorr.MFETSAutocorr.ft_acf,
                                        kwargs=dict(nlags=1,
                                                    unbiased=unbiased))

        return cls._rol_stat_postprocess(rolling_stat, remove_nan=remove_nan)

    @classmethod
    def ft_moving_acf_shift(
            cls,
            ts: np.ndarray,
            window_size: t.Union[int, float] = 0.1,
            unbiased: bool = True,
            diff_order: int = 1,
            diff_lag: int = 1,
            abs_value: bool = True,
            remove_nan: bool = True,
            ts_scaled: t.Optional[np.ndarray] = None,
            ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None,
    ) -> np.ndarray:
        """Absolute differenced moving autocorrelation of overlapping windows.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        window_size : int or float, optional (default=0.1)
            Size of the window. Must be strictly positive.
            If int >= 1, this argument defines the window size.
            If 0 < float < 1, this argument defines the fraction of the
            time-series length used as the window size.

        unbiased : bool, optional
            If True, then the calculations are corrected for statistical bias.

        diff_order : int, optional (default=1)
            Order of differentiation. If this argument get a value of 0 or
            less, then no differentiation will be performed.

        diff_lag : int, optional (default=1)
            Lag of each differentiation (among the moving statistics). If
            a value lower than 1 is given, then it is assumed lag 1.

        abs_value : bool, optional (default=True)
            If True, return the absolute value of the result.

        remove_nan : bool, optional (default=True)
            If True, remove `nan` values that may be generated while collecting
            the rolling statistics before any post-processing.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        ts_rol_win : :obj:`pd.core.window.rolling.Rolling`, optional
            Configured rolling window. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Post-processed moving autocorrelation from overlapping windows in
            time-series values.
        """
        rolling_stat_shift = cls._moving_stat_shift(
            ts=ts,
            stat_func=cls.ft_moving_acf,
            window_size=window_size,
            diff_order=diff_order,
            diff_lag=diff_lag,
            abs_value=abs_value,
            remove_nan=remove_nan,
            ts_scaled=ts_scaled,
            ts_rol_win=ts_rol_win,
            unbiased=unbiased)

        return rolling_stat_shift

    @classmethod
    def ft_moving_gmean(
            cls,
            ts: np.ndarray,
            window_size: t.Union[int, float] = 0.1,
            remove_nan: bool = True,
            ts_scaled: t.Optional[np.ndarray] = None,
            ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None,
    ) -> np.ndarray:
        """Moving geometric mean of overlapping windows.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        window_size : int or float, optional (default=0.1)
            Size of the window. Must be strictly positive.
            If int >= 1, this argument defines the window size.
            If 0 < float < 1, this argument defines the fraction of the
            time-series length used as the window size.

        remove_nan : bool, optional (default=True)
            If True, remove `nan` values that may be generated while collecting
            the rolling statistics before any post-processing.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        ts_rol_win : :obj:`pd.core.window.rolling.Rolling`, optional
            Configured rolling window. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Moving geometric mean from overlapping windows in time-series
            values.
        """
        ts_rol_win = _utils.get_rolling_window(ts=ts,
                                               window_size=window_size,
                                               ts_scaled=ts_scaled,
                                               ts_rol_win=ts_rol_win)

        rolling_stat = ts_rol_win.apply(scipy.stats.gmean)

        return cls._rol_stat_postprocess(rolling_stat, remove_nan=remove_nan)

    @classmethod
    def ft_moving_gmean_shift(
            cls,
            ts: np.ndarray,
            window_size: t.Union[int, float] = 0.1,
            diff_order: int = 1,
            diff_lag: int = 1,
            abs_value: bool = True,
            remove_nan: bool = True,
            ts_scaled: t.Optional[np.ndarray] = None,
            ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None,
    ) -> np.ndarray:
        """Absolute differenced moving geometric mean of overlapping windows.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        window_size : int or float, optional (default=0.1)
            Size of the window. Must be strictly positive.
            If int >= 1, this argument defines the window size.
            If 0 < float < 1, this argument defines the fraction of the
            time-series length used as the window size.

        diff_order : int, optional (default=1)
            Order of differentiation. If this argument get a value of 0 or
            less, then no differentiation will be performed.

        diff_lag : int, optional (default=1)
            Lag of each differentiation (among the moving statistics). If
            a value lower than 1 is given, then it is assumed lag 1.

        abs_value : bool, optional (default=True)
            If True, return the absolute value of the result.

        remove_nan : bool, optional (default=True)
            If True, remove `nan` values that may be generated while collecting
            the rolling statistics before any post-processing.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        ts_rol_win : :obj:`pd.core.window.rolling.Rolling`, optional
            Configured rolling window. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Post-processed moving geometric mean from overlapping windows in
            time-series values.
        """
        rolling_stat_shift = cls._moving_stat_shift(
            ts=ts,
            stat_func=cls.ft_moving_gmean,
            window_size=window_size,
            diff_order=diff_order,
            diff_lag=diff_lag,
            abs_value=abs_value,
            remove_nan=remove_nan,
            ts_scaled=ts_scaled,
            ts_rol_win=ts_rol_win)

        return rolling_stat_shift

    @classmethod
    def ft_moving_kldiv(
            cls,
            ts: np.ndarray,
            window_size: t.Union[int, float] = 0.1,
            remove_inf: bool = True,
            remove_nan: bool = True,
            ts_scaled: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Moving Kullback-Leibler divergence of overlapping windows.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        window_size : int or float, optional (default=0.1)
            Size of the window. Must be strictly positive.
            If int >= 1, this argument defines the window size.
            If 0 < float < 1, this argument defines the fraction of the
            time-series length used as the window size.

        remove_inf : bool, optional (default=True)
            If True, remove infinities that may be possibly generated during
            the Kullback-Leibler divergence calculation, before any other
            post-processing.

        remove_nan : bool, optional (default=True)
            If True, remove `nan` values that may be generated while collecting
            the rolling statistics before any post-processing.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        ts_rol_win : :obj:`pd.core.window.rolling.Rolling`, optional
            Configured rolling window. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Moving Kullback-Leibler divergence from overlapping windows in
            time-series values.

        References
        ----------
        .. [1] R. J. Hyndman, E. Wang and N. Laptev, "Large-Scale Unusual Time
            Series Detection," 2015 IEEE International Conference on Data
            Mining Workshop (ICDMW), Atlantic City, NJ, 2015, pp. 1616-1619,
            doi: 10.1109/ICDMW.2015.104.
        .. [2] Hyndman, R. J., Wang, E., Kang, Y., & Talagala, T. (2018).
            tsfeatures: Time series feature extraction. R package version 0.1.
        .. [3] Pablo Montero-Manso, George Athanasopoulos, Rob J. Hyndman,
            Thiyanga S. Talagala, FFORMA: Feature-based forecast model
            averaging, International Journal of Forecasting, Volume 36, Issue
            1, 2020, Pages 86-92, ISSN 0169-2070,
            https://doi.org/10.1016/j.ijforecast.2019.02.011.
        """
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        _window_size = _utils.process_window_size(ts=ts_scaled,
                                                  window_size=window_size)

        rolling_stat = np.zeros(ts_scaled.size - _window_size, dtype=float)

        next_bin = np.histogram(ts_scaled[:_window_size], density=True)[0]
        i = 1

        while i < ts_scaled.size - _window_size:
            cur_bin = next_bin
            next_bin = np.histogram(
                ts_scaled[i:i + _window_size], density=True)[0]
            rolling_stat[i - 1] = scipy.stats.entropy(next_bin, cur_bin)
            i += 1

        if remove_inf:
            rolling_stat = rolling_stat[np.isfinite(rolling_stat)]

        return cls._rol_stat_postprocess(rolling_stat, remove_nan=remove_nan)

    @classmethod
    def ft_moving_kldiv_shift(
            cls,
            ts: np.ndarray,
            window_size: t.Union[int, float] = 0.1,
            diff_order: int = 1,
            diff_lag: int = 1,
            abs_value: bool = True,
            remove_inf: bool = True,
            remove_nan: bool = True,
            ts_scaled: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Abs. diff. mov. Kullback-Leibler divergence of overlapping windows.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        window_size : int or float, optional (default=0.1)
            Size of the window. Must be strictly positive.
            If int >= 1, this argument defines the window size.
            If 0 < float < 1, this argument defines the fraction of the
            time-series length used as the window size.

        diff_order : int, optional (default=1)
            Order of differentiation. If this argument get a value of 0 or
            less, then no differentiation will be performed.

        diff_lag : int, optional (default=1)
            Lag of each differentiation (among the moving statistics). If
            a value lower than 1 is given, then it is assumed lag 1.

        abs_value : bool, optional (default=True)
            If True, return the absolute value of the result.

        remove_inf : bool, optional (default=True)
            If True, remove infinities that may be possibly generated during
            the Kullback-Leibler divergence calculation, before any other
            post-processing.

        remove_nan : bool, optional (default=True)
            If True, remove `nan` values that may be generated while collecting
            the rolling statistics before any post-processing.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Post-processed moving Kullback-Leibler divergence from overlapping
            windows in time-series values.

        References
        ----------
        .. [1] R. J. Hyndman, E. Wang and N. Laptev, "Large-Scale Unusual Time
            Series Detection," 2015 IEEE International Conference on Data
            Mining Workshop (ICDMW), Atlantic City, NJ, 2015, pp. 1616-1619,
            doi: 10.1109/ICDMW.2015.104.
        .. [2] Hyndman, R. J., Wang, E., Kang, Y., & Talagala, T. (2018).
            tsfeatures: Time series feature extraction. R package version 0.1.
        .. [3] Pablo Montero-Manso, George Athanasopoulos, Rob J. Hyndman,
            Thiyanga S. Talagala, FFORMA: Feature-based forecast model
            averaging, International Journal of Forecasting, Volume 36, Issue
            1, 2020, Pages 86-92, ISSN 0169-2070,
            https://doi.org/10.1016/j.ijforecast.2019.02.011.
        """
        rolling_stat = cls.ft_moving_kldiv(ts=ts,
                                           window_size=window_size,
                                           remove_nan=remove_nan,
                                           remove_inf=remove_inf,
                                           ts_scaled=ts_scaled)

        rolling_stat_shifts = cls._rol_stat_postprocess(rolling_stat,
                                                        remove_nan=False,
                                                        diff_order=diff_order,
                                                        diff_lag=diff_lag,
                                                        abs_value=abs_value)

        return rolling_stat_shifts

    @classmethod
    def ft_moving_lilliefors(
            cls,
            ts: np.ndarray,
            window_size: t.Union[int, float] = 0.1,
            distribution: str = "norm",
            return_pval: bool = False,
            remove_nan: bool = True,
            ts_scaled: t.Optional[np.ndarray] = None,
            ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None,
    ) -> np.ndarray:
        """Moving Lilliefors test of overlapping windows.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        window_size : int or float, optional (default=0.1)
            Size of the window. Must be strictly positive.
            If int >= 1, this argument defines the window size.
            If 0 < float < 1, this argument defines the fraction of the
            time-series length used as the window size.

        distribution : {`norm`, `exp`}, optional (default="norm")
            Distribution assumed by the Lilliefors test. Must be either
            `norm` (normal/gaussian distribution) or `exp` (exponential
            distribution).

        return_pval : bool, optional (default=False)
            If True, return the Lilliefors test p-value instead of the
            statistic value.

        remove_nan : bool, optional (default=True)
            If True, remove `nan` values that may be generated while collecting
            the rolling statistics before any post-processing.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        ts_rol_win : :obj:`pd.core.window.rolling.Rolling`, optional
            Configured rolling window. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            If `return_pval` is False, moving Lilliefors test from overlapping
            windows in time-series values. If `return_pval` is True, each test
            statistic is replaced by its correspondent p-value.

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
        ts_rol_win = _utils.get_rolling_window(ts=ts,
                                               window_size=window_size,
                                               ts_scaled=ts_scaled,
                                               ts_rol_win=ts_rol_win)

        rolling_stat = ts_rol_win.apply(
            stat_tests.MFETSStatTests.ft_test_lilliefors,
            kwargs=dict(distribution=distribution, return_pval=return_pval))

        return cls._rol_stat_postprocess(rolling_stat, remove_nan=remove_nan)

    @classmethod
    def ft_moving_approx_ent(
            cls,
            ts: np.ndarray,
            window_size: t.Union[int, float] = 0.1,
            embed_dim: int = 2,
            embed_lag: int = 1,
            threshold: float = 0.2,
            metric: str = "chebyshev",
            p: t.Union[int, float] = 2,
            remove_nan: bool = True,
            ts_scaled: t.Optional[np.ndarray] = None,
            ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None,
    ) -> np.ndarray:
        """Moving approximate entropy of overlapping windows.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        window_size : int or float, optional (default=0.1)
            Size of the window. Must be strictly positive.
            If int >= 1, this argument defines the window size.
            If 0 < float < 1, this argument defines the fraction of the
            time-series length used as the window size.

        embed_dim : int, optional (default=2)
            Embedding dimension to calculate the approximate entropy.

        embed_lag : int, optional (default=1)
            Embedding lag to calculate the approximate entropy.

        threshold : float, optional (default=0.2)
            Threshold for the radius nearest neighbors while calculating
            the approximate entropy.

        metric : str, optional (default="chebyshev")
            Metric used in the radius nearest neighbors of the approximate
            entropy. Check `scipy.spatial.distance.pdist` documentation for
            the complete list of available distance metrics.

        p : int or float, optional (default=2)
            Power argument for the Minkowski metric (used only if metric is
            `minkowski`).

        remove_nan : bool, optional (default=True)
            If True, remove `nan` values that may be generated while collecting
            the rolling statistics before any post-processing.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        ts_rol_win : :obj:`pd.core.window.rolling.Rolling`, optional
            Configured rolling window. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            If `return_pval` is False, moving approximate entropy from
            overlapping windows in time-series values. If `return_pval` is
            True, each test statistic is replaced by its correspondent p-value.

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
        ts_rol_win = _utils.get_rolling_window(ts=ts,
                                               window_size=window_size,
                                               ts_scaled=ts_scaled,
                                               ts_rol_win=ts_rol_win)

        rolling_stat = ts_rol_win.apply(
            info_theory.MFETSInfoTheory.ft_approx_entropy,
            kwargs=dict(embed_dim=embed_dim,
                        embed_lag=embed_lag,
                        threshold=threshold,
                        metric=metric,
                        p=p))

        return cls._rol_stat_postprocess(rolling_stat, remove_nan=remove_nan)

    @classmethod
    def ft_lumpiness(cls,
                     ts: np.ndarray,
                     num_tiles: int = 16,
                     ddof: int = 1,
                     ts_scaled: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Lumpiness (the non-overlapping rolling variance) of the time-series.

        Originally, this meta-feature is calculated using the variance of the
        value returned by this method. However, to enable other types of
        summarization, here we return all the tilled statistic values.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        num_tiles : int, optional (default=16)
            Number of non-overlapping windows in the time-series to extract
            the local statistic.

        ddof : int, optional (default=1)
            Degrees of freedom for the local variances.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Non-overlapping rolling variance of time-series.

        References
        ----------
        .. [1] R. J. Hyndman, E. Wang and N. Laptev, "Large-Scale Unusual Time
            Series Detection," 2015 IEEE International Conference on Data
            Mining Workshop (ICDMW), Atlantic City, NJ, 2015, pp. 1616-1619,
            doi: 10.1109/ICDMW.2015.104.
        .. [2] Hyndman, R. J., Wang, E., Kang, Y., & Talagala, T. (2018).
            tsfeatures: Time series feature extraction. R package version 0.1.
        .. [3] Pablo Montero-Manso, George Athanasopoulos, Rob J. Hyndman,
            Thiyanga S. Talagala, FFORMA: Feature-based forecast model
            averaging, International Journal of Forecasting, Volume 36, Issue
            1, 2020, Pages 86-92, ISSN 0169-2070,
            https://doi.org/10.1016/j.ijforecast.2019.02.011.
        """
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        tilled_vars = _utils.apply_on_tiles(ts=ts_scaled,
                                            num_tiles=num_tiles,
                                            func=np.var,
                                            ddof=ddof)

        # Note: the 'lumpiness' is defined as the variance of the
        # tilled variances. However, here, to enable other summarization,
        # we return the full array of tiled variances.
        return tilled_vars

    @classmethod
    def ft_stability(cls,
                     ts: np.ndarray,
                     num_tiles: int = 16,
                     ts_scaled: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Stability (the non-overlapping rolling mean) of the time-series.

        Originally, this meta-feature is calculated using the variance of the
        value returned by this method. However, to enable other types of
        summarization, here we return all the tilled statistic values.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        num_tiles : int, optional (default=16)
            Number of non-overlapping windows in the time-series to extract
            the local statistic.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Non-overlapping rolling mean of time-series.

        References
        ----------
        .. [1] Hyndman, R. J., Wang, E., Kang, Y., & Talagala, T. (2018).
            tsfeatures: Time series feature extraction. R package version 0.1.
        .. [2] Pablo Montero-Manso, George Athanasopoulos, Rob J. Hyndman,
            Thiyanga S. Talagala, FFORMA: Feature-based forecast model
            averaging, International Journal of Forecasting, Volume 36, Issue
            1, 2020, Pages 86-92, ISSN 0169-2070,
            https://doi.org/10.1016/j.ijforecast.2019.02.011.
        """
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        tilled_means = _utils.apply_on_tiles(ts=ts_scaled,
                                             num_tiles=num_tiles,
                                             func=np.mean)

        # Note: the 'stability' is defined as the variance of the
        # tilled means. However, here, to enable other summarization,
        # we return the full array of tiled variances.
        return tilled_means

    @classmethod
    def ft_local_extrema(
            cls,
            ts: np.ndarray,
            num_tiles: int = 16,
            ts_scaled: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Non-overlapping rolling windows local extrema of the time-series.

        The local extrema is the more extreme observation within a window, and
        it could be either the local minimum or local maximum (whichever have
        the largest absolute value).

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        num_tiles : int, optional (default=16)
            Number of non-overlapping windows in the time-series to extract
            the local statistic.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Non-overlapping rolling local extrema of time-series.

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
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        def get_extreme_val(tile: np.ndarray) -> float:
            """Get extreme (maximum in absolute) value of a tile."""
            min_, max_ = np.quantile(tile, (0, 1))
            return max_ if abs(min_) <= max_ else min_

        tilled_extrema = _utils.apply_on_tiles(ts=ts_scaled,
                                               num_tiles=num_tiles,
                                               func=get_extreme_val)

        return tilled_extrema

    @classmethod
    def ft_local_range(cls,
                       ts: np.ndarray,
                       num_tiles: int = 16,
                       ts_scaled: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Non-overlapping rolling windows range of the time-series.

        The local range is the local maximum minus the local minimum.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        num_tiles : int, optional (default=16)
            Number of non-overlapping windows in the time-series to extract
            the local statistic.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Non-overlapping rolling range of time-series.

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
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        tilled_range = _utils.apply_on_tiles(ts=ts_scaled,
                                             num_tiles=num_tiles,
                                             func=np.ptp)

        return tilled_range
