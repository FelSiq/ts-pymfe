import typing as t

import pandas as pd
import scipy.stats
import numpy as np
import pymfe.statistical

import autocorr
import _period
import _detrend
import _get_data
import _utils


class MFETSLocalStats:
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

        return rolling_stat.values

    @classmethod
    def ft_moving_avg(
        cls,
        ts: np.ndarray,
        window_size: int,
        remove_nan: bool = True,
        ts_scaled: t.Optional[np.ndarray] = None,
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
        if ts_rol_win is None or (window_size != ts_rol_win.window):
            ts_rol_win = _utils.get_rolling_window(ts=ts,
                                                   window_size=window_size,
                                                   ts_scaled=ts_scaled)

        rolling_stat = ts_rol_win.mean()

        return cls._rol_stat_postprocess(rolling_stat, remove_nan=remove_nan)

    @classmethod
    def ft_moving_var(
        cls,
        ts: np.ndarray,
        window_size: int,
        ddof: int = 1,
        remove_nan: bool = True,
        ts_scaled: t.Optional[np.ndarray] = None,
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
        if ts_rol_win is None or (window_size != ts_rol_win.window):
            ts_rol_win = _utils.get_rolling_window(ts=ts,
                                                   window_size=window_size,
                                                   ts_scaled=ts_scaled)

        rolling_stat = ts_rol_win.var(ddof=ddof)

        return cls._rol_stat_postprocess(rolling_stat, remove_nan=remove_nan)

    @classmethod
    def ft_moving_sd(
        cls,
        ts: np.ndarray,
        window_size: int,
        ddof: int = 1,
        remove_nan: bool = True,
        ts_scaled: t.Optional[np.ndarray] = None,
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
        if ts_rol_win is None or (window_size != ts_rol_win.window):
            ts_rol_win = _utils.get_rolling_window(ts=ts,
                                                   window_size=window_size,
                                                   ts_scaled=ts_scaled)

        rolling_stat = ts_rol_win.std(ddof=ddof)

        return cls._rol_stat_postprocess(rolling_stat, remove_nan=remove_nan)

    @classmethod
    def ft_moving_skewness(
        cls,
        ts: np.ndarray,
        window_size: int,
        method: int = 3,
        bias: bool = True,
        remove_nan: bool = True,
        ts_scaled: t.Optional[np.ndarray] = None,
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
        if ts_rol_win is None or (window_size != ts_rol_win.window):
            ts_rol_win = _utils.get_rolling_window(ts=ts,
                                                   window_size=window_size,
                                                   ts_scaled=ts_scaled)

        rolling_stat = ts_rol_win.apply(
            pymfe.statistical.MFEStatistical.ft_skewness,
            kwargs=dict(method=method, bias=bias))

        return cls._rol_stat_postprocess(rolling_stat, remove_nan=remove_nan)

    @classmethod
    def ft_moving_kurtosis(
        cls,
        ts: np.ndarray,
        window_size: int,
        method: int = 3,
        bias: bool = True,
        remove_nan: bool = True,
        ts_scaled: t.Optional[np.ndarray] = None,
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
        if ts_rol_win is None or (window_size != ts_rol_win.window):
            ts_rol_win = _utils.get_rolling_window(ts=ts,
                                                   window_size=window_size,
                                                   ts_scaled=ts_scaled)

        rolling_stat = ts_rol_win.apply(
            pymfe.statistical.MFEStatistical.ft_kurtosis,
            kwargs=dict(method=method, bias=bias))

        return cls._rol_stat_postprocess(rolling_stat, remove_nan=remove_nan)

    @classmethod
    def ft_moving_acf(
        cls,
        ts: np.ndarray,
        window_size: int,
        unbiased: bool = True,
        remove_nan: bool = True,
        ts_scaled: t.Optional[np.ndarray] = None,
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
        if ts_rol_win is None or (window_size != ts_rol_win.window):
            ts_rol_win = _utils.get_rolling_window(ts=ts,
                                                   window_size=window_size,
                                                   ts_scaled=ts_scaled)

        rolling_stat = ts_rol_win.apply(autocorr.MFETSAutocorr.ft_acf,
                                        kwargs=dict(nlags=1,
                                                    unbiased=unbiased))

        return cls._rol_stat_postprocess(rolling_stat, remove_nan=remove_nan)

    @classmethod
    def ft_moving_gmean(
        cls,
        ts: np.ndarray,
        window_size: int,
        remove_nan: bool = True,
        ts_scaled: t.Optional[np.ndarray] = None,
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
        if ts_rol_win is None or (window_size != ts_rol_win.window):
            ts_rol_win = _utils.get_rolling_window(ts=ts,
                                                   window_size=window_size,
                                                   ts_scaled=ts_scaled)

        rolling_stat = ts_rol_win.apply(scipy.stats.gmean)

        return cls._rol_stat_postprocess(rolling_stat, remove_nan=remove_nan)

    @classmethod
    def ft_moving_kldiv(
            cls,
            ts: np.ndarray,
            window_size: int,
            remove_inf: bool = True,
            remove_nan: bool = True,
            ts_scaled: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """TODO."""
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        rolling_stat = np.zeros(ts.size - window_size, dtype=float)

        next_wind = ts_scaled[:window_size]
        next_bin = np.histogram(next_wind, density=True)[0]
        i = 1

        while i < ts.size - window_size:
            cur_wind, cur_bin = next_wind, next_bin
            next_wind = ts_scaled[i:i + window_size]
            next_bin = np.histogram(next_wind, density=True)[0]
            rolling_stat[i - 1] = scipy.stats.entropy(next_bin, cur_bin)
            i += 1

        if remove_inf:
            rolling_stat = rolling_stat[np.isfinite(rolling_stat)]

        return cls._rol_stat_postprocess(rolling_stat, remove_nan=remove_nan)

    @classmethod
    def ft_lumpiness(cls,
                     ts: np.ndarray,
                     num_tiles: int = 16,
                     ddof: int = 1,
                     ts_scaled: t.Optional[np.ndarray] = None) -> np.ndarray:
        """TODO."""
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
        """TODO."""
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

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
    def ft_local_extrema(
            cls,
            ts: np.ndarray,
            num_tiles: int = 16,
            ts_scaled: t.Optional[np.ndarray] = None) -> np.ndarray:
        """TODO."""
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        def get_extreme_val(tile: np.ndarray) -> float:
            """Get extreme (maximum in absolute) value of a tile."""
            min_, max_ = np.quantile(tile, (0, 1))
            return max_ if abs(min_) <= max_ else min_

        tilled_extrema = _utils.apply_on_tiles(ts=ts_scaled,
                                               num_tiles=num_tiles,
                                               func=get_extreme_val)

        return tilled_extrema


def _test() -> None:
    ts = _get_data.load_data(3)
    ts_period = _period.ts_period(ts)
    ts_trend, ts_season, ts_residuals = _detrend.decompose(ts)
    ts = ts.to_numpy()

    res = MFETSLocalStats.ft_spikiness(ts_residuals)
    print(np.var(res))

    res = MFETSLocalStats.ft_lumpiness(ts)
    print("lumpiness", np.var(res))

    res = MFETSLocalStats.ft_stability(ts)
    print("stability", np.var(res))

    res = MFETSLocalStats.ft_moving_avg(ts, ts_period)
    print(np.nanmax(res))

    res = MFETSLocalStats.ft_moving_var(ts, ts_period)
    print(np.nanmax(res))

    res = MFETSLocalStats.ft_moving_skewness(ts, ts_period)
    print("skewness diff", np.nanmax(res))

    res = MFETSLocalStats.ft_moving_kurtosis(ts, ts_period)
    print("kurtosis diff", np.nanmax(res))

    res = MFETSLocalStats.ft_moving_gmean(ts, ts_period)
    print("gmean diff", res)

    res = MFETSLocalStats.ft_moving_sd(ts, ts_period)
    print(np.nanmax(res))

    res = MFETSLocalStats.ft_moving_acf(ts, ts_period)
    print("acf diff", res)

    res = MFETSLocalStats.ft_moving_kldiv(ts, ts_period)
    print(res)
    print(np.nanmax(res))

    res = MFETSLocalStats.ft_local_extrema(ts)
    print("LocalStats extrema", res)


if __name__ == "__main__":
    _test()
