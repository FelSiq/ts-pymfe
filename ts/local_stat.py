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
    def ft_mdiff_moving_acf(
        cls,
        ts: np.ndarray,
        ts_period: int,
        unbiased: bool = True,
        ts_scaled: t.Optional[np.ndarray] = None,
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
        if ts_rol_win is None:
            ts_rol_win = _utils.get_rolling_window(ts=ts,
                                                   window_size=ts_period,
                                                   ts_scaled=ts_scaled)

        res = np.abs(
            ts_rol_win.apply(autocorr.MFETSAutocorr.ft_acf,
                             kwargs={
                                 "unbiased": unbiased,
                                 "nlags": 1
                             }).diff(ts_period))

        return res.values

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
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        kl_divs = np.zeros(ts.size - ts_period, dtype=float)

        next_wind = ts_scaled[:ts_period]
        next_bin = np.histogram(next_wind)[0]
        i = 1

        while i < ts.size - ts_period:
            cur_wind, cur_bin = next_wind, next_bin
            next_wind = ts_scaled[i:i + ts_period]
            next_bin = np.histogram(next_wind)[0]
            kl_divs[i - 1] = scipy.stats.entropy(next_bin, cur_bin)
            i += 1

        return np.diff(kl_divs[np.isfinite(kl_divs)])

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

    res = MFETSLocalStats.ft_mdiff_moving_mean(ts, ts_period)
    print(np.nanmax(res))

    res = MFETSLocalStats.ft_mdiff_moving_var(ts, ts_period)
    print(np.nanmax(res))

    res = MFETSLocalStats.ft_mdiff_moving_skewness(ts, ts_period)
    print("skewness diff", np.nanmax(res))

    res = MFETSLocalStats.ft_mdiff_moving_kurtosis(ts, ts_period)
    print("kurtosis diff", np.nanmax(res))

    res = MFETSLocalStats.ft_mdiff_moving_gmean(ts, ts_period)
    print("gmean diff", res)

    res = MFETSLocalStats.ft_mdiff_moving_sd(ts, ts_period)
    print(np.nanmax(res))

    res = MFETSLocalStats.ft_mdiff_moving_acf(ts, ts_period)
    print("acf diff", res)

    res = MFETSLocalStats.ft_mdiff_moving_kldiv(ts, ts_period)
    print(res)
    print(np.nanmax(res))

    res = MFETSLocalStats.ft_local_extrema(ts)
    print("LocalStats extrema", res)


if __name__ == "__main__":
    _test()
