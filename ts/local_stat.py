import typing as t

import pandas as pd
import scipy.stats
import numpy as np
import pymfe.statistical

import stat_tests
import autocorr
import info_theory
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
        """TODO."""
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
    def ft_moving_avg_shift(
        cls,
        ts: np.ndarray,
        window_size: t.Union[int, float] = 0.1,
        diff_order: int = 1,
        diff_lag: int = 1,
        abs_value: bool = True,
        remove_nan: bool = True,
        ts_scaled: t.Optional[np.ndarray] = None,
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
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
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
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
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
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
    def ft_rand_samp_std(
            cls,
            ts: np.ndarray,
            num_samples: int = 64,
            sample_size_frac: float = 0.1,
            ddof: int = 1,
            random_state: t.Optional[int] = None,
            ts_scaled: t.Optional[np.ndarray] = None) -> np.ndarray:
        """TODO."""
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        sample_std = _utils.apply_on_samples(ts=ts_scaled,
                                             func=np.std,
                                             num_samples=num_samples,
                                             sample_size_frac=sample_size_frac,
                                             random_state=random_state,
                                             ddof=ddof)

        return sample_std

    @classmethod
    def ft_moving_skewness(
        cls,
        ts: np.ndarray,
        window_size: t.Union[int, float] = 0.1,
        method: int = 3,
        unbiased: bool = False,
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
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
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
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
        if ts_rol_win is None or (window_size != ts_rol_win.window):
            ts_rol_win = _utils.get_rolling_window(ts=ts,
                                                   window_size=window_size,
                                                   ts_scaled=ts_scaled)

        rolling_stat = ts_rol_win.apply(
            pymfe.statistical.MFEStatistical.ft_kurtosis,
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
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
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
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
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
    def ft_moving_gmean_shift(
        cls,
        ts: np.ndarray,
        window_size: t.Union[int, float] = 0.1,
        diff_order: int = 1,
        diff_lag: int = 1,
        abs_value: bool = True,
        remove_nan: bool = True,
        ts_scaled: t.Optional[np.ndarray] = None,
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
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
        """TODO."""
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        window_size = _utils.process_window_size(ts=ts_scaled,
                                                 window_size=window_size)

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
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
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
        dist: str = "norm",
        return_pval: bool = False,
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
            stat_tests.MFETSStatTests.ft_test_lilliefors,
            kwargs=dict(dist=dist, return_pval=return_pval))

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
        return_pval: bool = False,
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

    @classmethod
    def ft_local_range(
            cls,
            ts: np.ndarray,
            num_tiles: int = 16,
            ts_scaled: t.Optional[np.ndarray] = None) -> np.ndarray:
        """TODO."""
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        tilled_range = _utils.apply_on_tiles(ts=ts_scaled,
                                               num_tiles=num_tiles,
                                               func=np.ptp)

        return tilled_range


def _test() -> None:
    ts = _get_data.load_data(3)
    ts_period = _period.ts_period(ts)
    ts_trend, ts_season, ts_residuals = _detrend.decompose(ts)
    ts = ts.to_numpy()

    res = MFETSLocalStats.ft_moving_lilliefors(ts)
    print(res)
    exit(1)

    res = MFETSLocalStats.ft_rand_samp_std(ts)
    print(res)
    exit(1)

    res = MFETSLocalStats.ft_moving_approx_ent(ts)
    print(res)

    exit(1)

    res = MFETSLocalStats.ft_moving_lilliefors(ts)
    print(res)

    res = MFETSLocalStats.ft_moving_avg(ts)
    print(res)

    res = MFETSLocalStats.ft_moving_avg_shift(ts)
    print(res)

    res = MFETSLocalStats.ft_moving_var_shift(ts)
    print(res)

    res = MFETSLocalStats.ft_moving_skewness_shift(ts)
    print("skewness diff", np.nanmax(res))

    res = MFETSLocalStats.ft_moving_kurtosis_shift(ts)
    print("kurtosis diff", np.nanmax(res))

    res = MFETSLocalStats.ft_moving_gmean_shift(ts)
    print("gmean diff", res)

    res = MFETSLocalStats.ft_moving_sd_shift(ts)
    print("sd shift", res)

    res = MFETSLocalStats.ft_moving_acf_shift(ts)
    print("acf diff", res)

    res = MFETSLocalStats.ft_moving_kldiv_shift(ts)
    print(res)
    exit(1)

    res = MFETSLocalStats.ft_spikiness(ts_residuals)
    print(np.var(res))

    res = MFETSLocalStats.ft_lumpiness(ts)
    print("lumpiness", np.var(res))

    res = MFETSLocalStats.ft_stability(ts)
    print("stability", np.var(res))

    res = MFETSLocalStats.ft_moving_var(ts)
    print(np.nanmax(res))

    res = MFETSLocalStats.ft_moving_skewness(ts)
    print("skewness diff", np.nanmax(res))

    res = MFETSLocalStats.ft_moving_kurtosis(ts)
    print("kurtosis diff", np.nanmax(res))

    res = MFETSLocalStats.ft_moving_gmean(ts)
    print("gmean diff", res)

    res = MFETSLocalStats.ft_moving_sd(ts)
    print(np.nanmax(res))

    res = MFETSLocalStats.ft_moving_acf(ts)
    print("acf diff", res)

    res = MFETSLocalStats.ft_moving_kldiv(ts)
    print(res)

    res = MFETSLocalStats.ft_local_extrema(ts)
    print("LocalStats extrema", res)


if __name__ == "__main__":
    _test()
