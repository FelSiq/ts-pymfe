import typing as t
import collections

import sklearn.preprocessing
import numpy as np
import pandas as pd
import pymfe.statistical
import nolds
import scipy.stats

import _detrend
import _embed
import _period
import _utils
import _get_data


class MFETSStatistical:
    @classmethod
    def precompute_time_delta_stats_it(
            cls,
            ts: np.ndarray,
            ts_scaled: t.Optional[np.ndarray] = None,
            step_size: float = 0.05,
            **kwargs) -> t.Dict[str, t.Union[float, np.ndarray]]:
        """TODO."""
        precomp_vals = {}  # type: t.Dict[str, t.Union[float, np.ndarray]]

        funcs = collections.OrderedDict((
            ("median", lambda arr: np.median(arr) / ts.size),
            ("mean", lambda arr: np.mean(arr) / ts.size),
            ("std", lambda arr: np.std(arr, ddof=1) / np.sqrt(arr.size + 1)),
        ))

        _formatted_keys = list(map("tdelta_it_{}".format, funcs.keys()))

        if not frozenset(_formatted_keys).issubset(kwargs):
            res = cls._calc_time_deltas_stats_it(ts=ts,
                                                 funcs=funcs.values(),
                                                 ts_scaled=ts_scaled,
                                                 step_size=step_size)

            precomp_vals.update(zip(_formatted_keys, res.T))

        return precomp_vals

    @classmethod
    def _calc_time_deltas_stats_it(cls,
                                   ts: np.ndarray,
                                   funcs: t.Union[t.Callable[[np.ndarray],
                                                             float],
                                                  t.Iterable[t.Callable[
                                                      [np.ndarray], float]]],
                                   ts_scaled: t.Optional[np.ndarray] = None,
                                   step_size: float = 0.05) -> np.ndarray:
        """TODO.

        https://github.com/benfulcher/hctsa/blob/master/Operations/DN_OutlierInclude.m
        """
        try:
            if len(funcs) == 0:
                raise ValueError("'funcs' is empty.")

        except:
            funcs = [funcs]

        if ts_scaled is None:
            ts_scaled = sklearn.preprocessing.StandardScaler().fit_transform(
                ts.reshape(-1, 1)).ravel()

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

            diff_tsteps = np.diff(outlier_tsteps)

            res.append([func(diff_tsteps) for func in funcs])

        res = np.asarray(res, dtype=float)

        return res

    @classmethod
    def ft_trend(cls,
                 ts_residuals: np.ndarray,
                 ts_deseasonalized: np.ndarray,
                 ddof: int = 1) -> float:
        """Ratio of standard deviations of time-series and after detrend.

        Parameters
        ----------
        ts_residuals : :obj:`np.ndarray`
            Residuals (random noise) of an one-dimensional time-series.

        ts_deseasonalized: :obj:`np.ndarray`
            One-dimensional deseasonalized time-series values.

        ddof : float, optional
            Degrees of freedom for standard deviation.

        Returns
        -------
        float
            Ratio of standard deviation of the original time-series
            and the standard deviation of the detrended version.

        References
        ----------
        TODO.
        """
        trend = 1.0 - (np.var(ts_residuals, ddof=ddof) /
                       np.var(ts_deseasonalized, ddof=ddof))

        return min(1.0, max(0.0, trend))

    @classmethod
    def ft_seasonality(cls,
                       ts_residuals: np.ndarray,
                       ts_detrended: np.ndarray,
                       ddof: int = 1) -> float:
        """
        TODO.

        https://pkg.robjhyndman.com/tsfeatures/articles/tsfeatures.html
        """

        seas = 1.0 - (np.var(ts_residuals, ddof=ddof) /
                      np.var(ts_detrended, ddof=ddof))

        return min(1.0, max(0.0, seas))

    @classmethod
    def ft_sd_residuals(cls, ts_residuals: np.ndarray, ddof: int = 1) -> float:
        """Compute the standard deviation of the time-series residuals.
        
        Parameters
        ----------
        ts_residuals : :obj:`np.ndarray`
            Residuals (random noise) of an one-dimensional time-series.

        ddof : float, optional
            Degrees of freedom for standard deviation.

        Returns
        -------
        float
            Detrended time-series standard deviation.

        References
        ----------
        TODO.
        """
        return np.std(ts_residuals, ddof=ddof)

    @classmethod
    def ft_sd_diff(cls,
                   ts: np.ndarray,
                   num_diff: int = 1,
                   ddof: int = 1) -> float:
        """TODO.

        Parameters
        ----------
        ts: :obj:`np.ndarray`
            TODO.

        ddof : float, optional
            Degrees of freedom for standard deviation.

        Returns
        -------
        float
        TODO.

        References
        ----------
        TODO.
        """
        return np.std(np.diff(ts, n=num_diff), ddof=ddof)

    @classmethod
    def ft_skewness_residuals(cls,
                              ts_residuals: np.ndarray,
                              method: int = 3,
                              bias: bool = True) -> float:
        """Compute the skewness of the time-series residuals.

        Parameters
        ----------
        ts_residuals : :obj:`np.ndarray`
            Residuals (random noise) of an one-dimensional time-series.

        method : int, optional
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

            Where `n` is the number of instances in ``N``, `s` is the standard
            deviation of each attribute in ``N``, and `m_i` is the ith
            statistical momentum of each attribute in ``N``.

            Note that if the selected method is unable to be calculated due to
            division by zero, then the first method will be used instead.

        bias : bool, optional
            If False, then the calculations are corrected for statistical bias.

        Returns
        -------
        float
            Detrended time-series skewness.

        References
        ----------
        .. [1] Donald Michie, David J. Spiegelhalter, Charles C. Taylor, and
           John Campbell. Machine Learning, Neural and Statistical
           Classification, volume 37. Ellis Horwood Upper Saddle River, 1994.
        """
        ts_skew = pymfe.statistical.MFEStatistical.ft_skewness(N=ts_residuals,
                                                               method=method,
                                                               bias=bias)

        return ts_skew

    @classmethod
    def ft_skewness_diff(cls,
                         ts: np.ndarray,
                         num_diff: int = 1,
                         method: int = 3,
                         bias: bool = True) -> float:
        """TODO."""
        ts_diff = np.diff(ts, n=num_diff)
        ts_skew = pymfe.statistical.MFEStatistical.ft_skewness(N=ts_diff,
                                                               method=method,
                                                               bias=bias)

        return ts_skew

    @classmethod
    def ft_kurtosis_residuals(cls,
                              ts_residuals: np.ndarray,
                              method: int = 3,
                              bias: bool = True) -> float:
        """Compute the kurtosis of the time-series residuals.

        Parameters
        ----------
        ts_residuals : :obj:`np.ndarray`
            Residuals (random noise) of an one-dimensional time-series.

        method : int, optional
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

            Where `n` is the number of instances in ``N``, `s` is the standard
            deviation of each attribute in ``N``, and `m_i` is the ith
            statistical momentum of each attribute in ``N``.

            Note that if the selected method is unable to be calculated due
            to division by zero, then the first method is used instead.

        bias : bool, optional
            If False, then the calculations are corrected for statistical bias.

        Returns
        -------
        float
            Detrended time-series kurtosis.

        References
        ----------
        .. [1] Donald Michie, David J. Spiegelhalter, Charles C. Taylor, and
           John Campbell. Machine Learning, Neural and Statistical
           Classification, volume 37. Ellis Horwood Upper Saddle River, 1994.
        """
        ts_kurt = pymfe.statistical.MFEStatistical.ft_kurtosis(N=ts_residuals,
                                                               method=method,
                                                               bias=bias)

        return ts_kurt

    @classmethod
    def ft_kurtosis_diff(cls,
                         ts: np.ndarray,
                         num_diff: int = 1,
                         method: int = 3,
                         bias: bool = True) -> float:
        """TODO."""
        ts_diff = np.diff(ts, n=num_diff)
        ts_kurt = pymfe.statistical.MFEStatistical.ft_kurtosis(N=ts_diff,
                                                               method=method,
                                                               bias=bias)

        return ts_kurt

    @classmethod
    def ft_mdiff_moving_mean(
        cls,
        ts: np.ndarray,
        ts_period: int,
        ts_scaled: t.Optional[np.ndarray] = None,
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
        if ts_rol_win is None:
            ts_rol_win = _utils.get_rolling_window(ts=ts,
                                                   window_size=ts_period,
                                                   ts_scaled=ts_scaled)

        return np.abs(ts_rol_win.mean().diff(ts_period))

    @classmethod
    def ft_mdiff_moving_var(
        cls,
        ts: np.ndarray,
        ts_period: int,
        ddof: int = 1,
        ts_scaled: t.Optional[np.ndarray] = None,
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
        if ts_rol_win is None:
            ts_rol_win = _utils.get_rolling_window(ts=ts,
                                                   window_size=ts_period,
                                                   ts_scaled=ts_scaled)

        return np.abs(ts_rol_win.var(ddof=ddof).diff(ts_period))

    @classmethod
    def ft_mdiff_moving_sd(
        cls,
        ts: np.ndarray,
        ts_period: int,
        ddof: int = 1,
        ts_scaled: t.Optional[np.ndarray] = None,
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
        if ts_rol_win is None:
            ts_rol_win = _utils.get_rolling_window(ts=ts,
                                                   window_size=ts_period,
                                                   ts_scaled=ts_scaled)

        return np.abs(ts_rol_win.std(ddof=ddof).diff(ts_period))

    @classmethod
    def ft_mdiff_moving_skewness(
        cls,
        ts: np.ndarray,
        ts_period: int,
        method: int = 3,
        bias: bool = True,
        ts_scaled: t.Optional[np.ndarray] = None,
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
        if ts_rol_win is None:
            ts_rol_win = _utils.get_rolling_window(ts=ts,
                                                   window_size=ts_period,
                                                   ts_scaled=ts_scaled)

        res = np.abs(
            ts_rol_win.apply(pymfe.statistical.MFEStatistical.ft_skewness,
                             kwargs={
                                 "method": method,
                                 "bias": bias
                             }).diff(ts_period))

        return res

    @classmethod
    def ft_mdiff_moving_kurtosis(
        cls,
        ts: np.ndarray,
        ts_period: int,
        method: int = 3,
        bias: bool = True,
        ts_scaled: t.Optional[np.ndarray] = None,
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
        if ts_rol_win is None:
            ts_rol_win = _utils.get_rolling_window(ts=ts,
                                                   window_size=ts_period,
                                                   ts_scaled=ts_scaled)

        res = np.abs(
            ts_rol_win.apply(pymfe.statistical.MFEStatistical.ft_kurtosis,
                             kwargs={
                                 "method": method,
                                 "bias": bias
                             }).diff(ts_period))

        return res

    @classmethod
    def ft_mdiff_moving_gmean(
        cls,
        ts: np.ndarray,
        ts_period: int,
        ts_scaled: t.Optional[np.ndarray] = None,
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
        if ts_rol_win is None:
            ts_rol_win = _utils.get_rolling_window(ts=ts,
                                                   window_size=ts_period,
                                                   ts_scaled=ts_scaled)

        res = np.abs(ts_rol_win.apply(scipy.stats.gmean).diff(ts_period))

        return res

    @classmethod
    def ft_mdiff_moving_kldiv(
            cls,
            ts: np.ndarray,
            ts_period: int,
            ts_scaled: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """TODO."""
        if ts_scaled is None:
            ts_scaled = sklearn.preprocessing.StandardScaler().fit_transform(
                ts.reshape(-1, 1)).ravel()

        kl_divs = np.zeros(ts.size - ts_period, dtype=float)

        next_wind = ts_scaled[:ts_period]
        i = 1

        while i < ts.size - ts_period:
            cur_wind = next_wind
            next_wind = ts_scaled[i:i + ts_period]
            kl_divs[i - 1] = scipy.stats.entropy(cur_wind, next_wind)
            i += 1

        return np.diff(kl_divs[np.isfinite(kl_divs)])

    @classmethod
    def ft_lumpiness(cls,
                     ts: np.ndarray,
                     num_tiles: int = 16,
                     ddof: int = 1,
                     ts_scaled: t.Optional[np.ndarray] = None) -> np.ndarray:
        """TODO."""
        if num_tiles > 0.5 * ts.size:
            raise ValueError("'num_tiles' ({}) larger than half the "
                             "time-series size ({}).".format(
                                 num_tiles, 0.5 * ts.size))

        if ts_scaled is None:
            ts_scaled = sklearn.preprocessing.StandardScaler().fit_transform(
                ts.reshape(-1, 1)).ravel()

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
        """TODO."""
        if num_tiles > 0.5 * ts.size:
            raise ValueError("'num_tiles' ({}) larger than half the "
                             "time-series size ({}).".format(
                                 num_tiles, 0.5 * ts.size))

        if ts_scaled is None:
            ts_scaled = sklearn.preprocessing.StandardScaler().fit_transform(
                ts.reshape(-1, 1)).ravel()

        tilled_means = _utils.apply_on_tiles(ts=ts_scaled,
                                             num_tiles=num_tiles,
                                             func=np.mean)

        # Note: the 'stability' is defined as the variance of the
        # tilled means. However, here, to enable other summarization,
        # we return the full array of tiled variances.
        return tilled_means

    @classmethod
    def ft_spikiness(cls,
                     ts_residuals: np.ndarray,
                     ddof: int = 1) -> np.ndarray:
        """TODO."""
        vars_ = np.array([
            np.var(np.delete(ts_residuals, i), ddof=ddof)
            for i in np.arange(ts_residuals.size)
        ],
                         dtype=float)

        # Note: on the original reference paper, the spikiness is calculated
        # as the variance of the 'vars_'. However, to enable summarization,
        # here we return the full array.
        return vars_

    @classmethod
    def ft_exp_max_lyap(cls,
                        ts: np.ndarray,
                        embed_dim: int,
                        lag: int,
                        method: str = "rosenstein") -> float:
        """TODO."""
        VALID_METHODS = ("eckmann", "rosenstein")

        if method not in VALID_METHODS:
            raise ValueError("'method' ({}) not in {}.".format(
                method, VALID_METHODS))

        if method == "rosenstein":
            return nolds.lyap_r(data=ts, lag=lag, emb_dim=embed_dim)

        return nolds.lyap_e(data=ts, emb_dim=embed_dim)

    @classmethod
    def ft_exp_hurst(cls, ts: np.ndarray) -> float:
        """TODO."""
        return nolds.hurst_rs(data=ts)

    @classmethod
    def ft_dfa(cls,
               ts: np.ndarray,
               order: int = 1,
               return_coeff: bool = True) -> t.Union[np.ndarray, float]:
        """TODO."""
        if return_coeff:
            hurst_coeff = nolds.dfa(ts, order=order)
            return hurst_coeff

        _, (_, fluct, _) = nolds.dfa(ts, order=order, debug_data=True)
        return fluct

    @classmethod
    def ft_opt_boxcox_coef(cls,
                           ts: np.ndarray,
                           epsilon: float = 1.0e-4,
                           num_lambdas: int = 16) -> float:
        """TODO."""
        ts = ts - ts.min() + epsilon
        return scipy.stats.boxcox_normmax(ts, method="mle")


def _test() -> None:
    ts = _get_data.load_data(3)

    ts_period = _period.ts_period(ts)
    ts_trend, ts_season, ts_residuals = _detrend.decompose(ts,
                                                           ts_period=ts_period)
    ts = ts.to_numpy()

    res = MFETSStatistical.ft_opt_boxcox_coef(ts)
    print(res)

    res = MFETSStatistical.ft_sd_diff(ts)
    print("sd diff", res)

    res = MFETSStatistical.ft_skewness_diff(ts)
    print("skewness diff", res)

    res = MFETSStatistical.ft_kurtosis_diff(ts)
    print("kurtosis diff", res)

    res = MFETSStatistical.ft_sd_diff(ts)
    print("sd diff", res)

    res = MFETSStatistical.ft_exp_max_lyap(ts, embed_dim=ts_period, lag=1)
    print("exp max lyap", res)

    res = MFETSStatistical.ft_exp_hurst(ts)
    print("exp hurst", res)

    res = MFETSStatistical.ft_spikiness(ts_residuals)
    print(np.var(res))

    res = MFETSStatistical.ft_lumpiness(ts)
    print("lumpiness", np.var(res))

    res = MFETSStatistical.ft_stability(ts)
    print("stability", np.var(res))

    res = MFETSStatistical.ft_mdiff_moving_mean(ts, ts_period)
    print(np.nanmax(res))

    res = MFETSStatistical.ft_mdiff_moving_var(ts, ts_period)
    print(np.nanmax(res))

    res = MFETSStatistical.ft_mdiff_moving_skewness(ts, ts_period)
    print("skewness diff", np.nanmax(res))

    res = MFETSStatistical.ft_mdiff_moving_kurtosis(ts, ts_period)
    print("kurtosis diff", np.nanmax(res))

    res = MFETSStatistical.ft_mdiff_moving_gmean(ts, ts_period)
    print("gmean diff", np.nanmax(res))

    res = MFETSStatistical.ft_mdiff_moving_sd(ts, ts_period)
    print(np.nanmax(res))

    res = MFETSStatistical.ft_mdiff_moving_kldiv(ts, ts_period)
    print(np.nanmax(res))

    res = MFETSStatistical.ft_skewness_residuals(ts_residuals)
    print(res)

    res = MFETSStatistical.ft_kurtosis_residuals(ts_residuals)
    print(res)

    res = MFETSStatistical.ft_sd_residuals(ts_residuals)
    print(res)

    res = MFETSStatistical.ft_trend(ts_residuals, ts_trend + ts_residuals)
    print(res)

    res = MFETSStatistical.ft_seasonality(ts_residuals,
                                          ts_season + ts_residuals)
    print(res)

    res = MFETSStatistical.ft_dfa(ts)
    print(res)

    exit(1)
    res = MFETSStatistical.precompute_time_delta_stats_it(ts)
    print(res)


if __name__ == "__main__":
    _test()
