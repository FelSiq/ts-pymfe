import typing as t

import sklearn.preprocessing
import numpy as np
import pymfe.statistical
import nolds
import scipy.spatial
import scipy.odr
import pandas as pd

import _detrend
import _embed
import _period
import _get_data


class MFETSGeneral:
    @classmethod
    def ft_sd(cls, ts_residuals: np.ndarray, ddof: int = 1) -> float:
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
    def ft_skewness(cls,
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
    def ft_kurtosis(cls,
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
    def ft_length(cls, ts: np.ndarray) -> int:
        """Length of the time-series.

        Parameters
        ----------
        ts: :obj:`np.ndarray`
            One-dimensional time-series values.

        Returns
        -------
        int
            Length of the time-seties.

        References
        ----------
        TODO.
        """
        return ts.size

    @classmethod
    def ft_period(cls,
                  ts: np.ndarray,
                  ts_period: t.Optional[int] = None) -> int:
        """TODO."""
        if ts_period is not None:
            return ts_period

        return _period.ts_period(ts)

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
    def ft_frac_tp(cls,
                   ts: np.ndarray,
                   normalize: bool = False) -> t.Union[int, float]:
        """Fraction of turning points in the time-series.

        A turning point is a time-series point `p_{i}` which both neighbor
        values, p_{i-1} and p_{i+1}, are either lower (p_{i} > p_{i+1} and
        p_{i} > p_{i-1}) or higher (p_{i} < p_{i+1} and p_{i} < p_{i-1}).

        Parameters
        ----------
        ts: :obj:`np.ndarray`
            One-dimensional time-series values.

        normalize : bool, optional
            If False, return the number of turning points instead.

        Returns
        -------
        float or int
            Fraction of turning points in the time-series, if ``normalize``
            is True. Number of turning points otherwise.

        References
        ----------
        TODO.
        """
        diff_arr = np.ediff1d(ts)
        frac_tp = np.sum(diff_arr[1:] * diff_arr[:-1] < 0)

        if normalize:
            frac_tp /= ts.size - 1

        return frac_tp

    @classmethod
    def ft_frac_sc(cls,
                   ts: np.ndarray,
                   ddof: int = 1,
                   normalize: bool = True) -> t.Union[int, float]:
        """Fraction of step change points in the time-series.

        Let p_{t_{a}}^{t_{b}} be the subsequence of observations from the
        timestep t_{a} and t_{b}, both inclusive. A point `p_i` is a
        turning point if and only if

        abs(p_{i} - mean(p_{1}^{i-1})) > 2 * std(p_{1}^{i-1})

        Parameters
        ----------
        ts: :obj:`np.ndarray`
            One-dimensional time-series values.

        ddof : float, optional
            Degrees of freedom for standard deviation.

        normalize : bool, optional
            If False, return the number of step changes instead.

        Returns
        -------
        float or int
            Fraction of step change points in the time-series, if
            ``normalize`` is True. Number of step changes otherwise.

        References
        ----------
        TODO.
        """
        ts_cmeans = np.cumsum(ts) / np.arange(1, ts.size + 1)

        ts_mean_abs_div = np.abs(ts[1:] - ts_cmeans[:-1])

        num_sc = 0

        for i in np.arange(1 + ddof, ts.size):
            num_sc += int(
                ts_mean_abs_div[i - 1] > 2 * np.std(ts[:i], ddof=ddof))

        if normalize:
            num_sc /= ts.size - 1

        return num_sc

    @classmethod
    def ft_pred(cls,
                ts_embedded: np.ndarray,
                param_1: t.Union[int, float] = 3,
                param_2: t.Union[int, float] = 4,
                metric: str = "minkowski",
                p: t.Union[int, float] = 2,
                ddof: int = 1) -> float:
        """https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4736930/"""
        dist_mat = scipy.spatial.distance.pdist(ts_embedded,
                                                metric=metric,
                                                p=p)

        dist_mean = np.mean(dist_mat)
        dist_std = np.std(dist_mat, ddof=ddof)

        dist_mat = scipy.spatial.distance.squareform(dist_mat)

        var_sets = np.zeros(param_2, dtype=float)

        for i in np.arange(param_2):
            threshold = max(
                0.0,
                dist_mean + param_1 * dist_std * (i * 2 / (param_2 - 1) - 1))

            neighbors = (dist_mat <= threshold).astype(int)
            neighbors[np.diag_indices_from(neighbors)] = 0.0

            for neigh_inds in neighbors:
                if np.sum(neigh_inds) > ddof:
                    var_sets[i] += np.var(ts_embedded[neigh_inds, :],
                                          ddof=ddof)

        var_sets /= ts_embedded.shape[0] * np.var(ts_embedded, ddof=ddof)

        return 1.0 / (1.0 + var_sets)

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
    def ft_frac_cp(cls,
                   ts: np.ndarray,
                   normalize: bool = True) -> t.Union[int, float]:
        """TODO."""
        higher_med = ts <= np.median(ts)
        num_cp = np.sum(np.logical_xor(higher_med[1:], higher_med[:-1]))

        if normalize:
            num_cp /= ts.size - 1

        return num_cp

    @classmethod
    def ft_binmean(cls, ts: np.ndarray) -> np.ndarray:
        """TODO."""
        return (ts >= np.mean(ts)).astype(int)

    @classmethod
    def ft_fs_len(cls, ts: np.ndarray, num_bins: int = 10) -> np.ndarray:
        """TODO."""
        ts_disc = np.digitize(ts, np.linspace(0, np.max(ts), num_bins))
        i = 1
        counter = 1
        fs_len = []  # type: t.List[int]

        while i < ts.size:
            if not np.isclose(ts_disc[i], ts_disc[i - 1]):
                fs_len.append(counter)
                counter = 1

            else:
                counter += 1

            i += 1

        return np.asarray(fs_len, dtype=float)

    @staticmethod
    def _apply_on_tiles(ts: np.ndarray, num_tiles: int,
                        func: t.Callable[[np.ndarray], t.Any], *args,
                        **kwargs) -> np.ndarray:
        """TODO."""
        res = np.array([
            func(split, *args, **kwargs)  # type: ignore
            for split in np.array_split(ts, num_tiles)
        ],
                       dtype=float)

        return res

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

        tilled_vars = cls._apply_on_tiles(ts=ts_scaled,
                                          num_tiles=num_tiles,
                                          func=np.var,
                                          **{"ddof": ddof})

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

        tilled_means = cls._apply_on_tiles(ts=ts_scaled,
                                           num_tiles=num_tiles,
                                           func=np.mean)

        # Note: the 'stability' is defined as the variance of the
        # tilled means. However, here, to enable other summarization,
        # we return the full array of tiled variances.
        return tilled_means

    @classmethod
    def ft_linearity(cls, ts_trend: np.ndarray) -> float:
        """TODO."""
        ts_trend_scaled = sklearn.preprocessing.StandardScaler().fit_transform(
            ts_trend.reshape(-1, 1))

        return -1.0

    @staticmethod
    def _get_rolling_window(
        ts: np.ndarray,
        window_size: int,
        ts_scaled: t.Optional[np.ndarray] = None
    ) -> pd.core.window.rolling.Rolling:
        """TODO."""
        if ts_scaled is None:
            ts_scaled = sklearn.preprocessing.StandardScaler().fit_transform(
                ts.reshape(-1, 1)).ravel()

        window_size = min(ts.size, window_size)
        return pd.Series(ts_scaled).rolling(window_size, center=True)

    @classmethod
    def ft_shift_level(
        cls,
        ts: np.ndarray,
        window_size: int = 12,
        ts_scaled: t.Optional[np.ndarray] = None,
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
        if ts_rol_win is None:
            ts_rol_win = cls._get_rolling_window(ts=ts,
                                                 window_size=window_size,
                                                 ts_scaled=ts_scaled)

        return np.abs(ts_rol_win.mean().diff(window_size))

    @classmethod
    def ft_shift_var(
        cls,
        ts: np.ndarray,
        window_size: int = 12,
        ddof: int = 1,
        ts_scaled: t.Optional[np.ndarray] = None,
        ts_rol_win: t.Optional[pd.core.window.rolling.Rolling] = None
    ) -> np.ndarray:
        """TODO."""
        if ts_rol_win is None:
            ts_rol_win = cls._get_rolling_window(ts=ts,
                                                 window_size=window_size,
                                                 ts_scaled=ts_scaled)

        return np.abs(ts_rol_win.var(ddof=ddof).diff(window_size))

    @staticmethod
    def _calc_season_mode_ind(ts_season: np.ndarray, ts_period: int,
                              indfunc: t.Callable[[np.ndarray], float]) -> int:
        """TODO."""
        inds = np.arange(ts_period)

        inds = np.array([
            indfunc(ts_season[i * ts_period + inds])
            for i in np.arange(1, ts_season.size // ts_period)
        ],
                        dtype=int)

        mode_inds, _ = scipy.stats.mode(inds)
        return mode_inds[0] + 1

    @classmethod
    def ft_peak_frac(cls,
                     ts_season: np.ndarray,
                     ts_period: int,
                     normalize: bool = True) -> t.Union[int, float]:
        """TODO."""
        ind_peak = cls._calc_season_mode_ind(ts_season=ts_season,
                                             ts_period=ts_period,
                                             indfunc=np.argmax)

        if normalize:
            ind_peak /= ts_period  # type: ignore

        return ind_peak

    @classmethod
    def ft_trough_frac(cls,
                       ts_season: np.ndarray,
                       ts_period: int,
                       normalize: bool = True) -> t.Union[int, float]:
        """TODO."""
        ind_trough = cls._calc_season_mode_ind(ts_season=ts_season,
                                               ts_period=ts_period,
                                               indfunc=np.argmin)

        if normalize:
            ind_trough /= ts_period  # type: ignore

        return ind_trough

    @classmethod
    def ft_walker_cross_frac(cls,
                             ts: np.ndarray,
                             step_size: float = 0.1,
                             start_point: t.Optional[t.Union[int,
                                                             float]] = None,
                             normalize: bool = True) -> t.Union[int, float]:
        """TODO."""
        if start_point is None:
            start_point = np.mean(ts)

        walker_pos = np.zeros(ts.size, dtype=float)
        walker_pos[0] = start_point

        for i in np.arange(2, ts.size):
            diff = ts[i - 1] - walker_pos[i - 1]
            walker_pos[i] = walker_pos[i - 1] + step_size * diff

        cross_num = np.sum((walker_pos[:-1] - ts[:-1]) *
                           (walker_pos[1:] - ts[1:]) < 0)

        if normalize:
            cross_num /= walker_pos.size - 1

        return cross_num

    @classmethod
    def ft_trev(cls,
                ts: np.ndarray,
                lag: int = 1,
                only_numerator: bool = False) -> float:
        """TODO.

        Normalized nonlinear autocorrelation.

        https://github.com/benfulcher/hctsa/blob/master/Operations/CO_trev.m
        """
        diff = ts[lag:] - ts[:-lag]

        numen = np.mean(np.power(diff, 3))

        if only_numerator:
            return numen

        denom = np.power(np.mean(np.square(diff)), 1.5)
        trev = numen / denom

        return trev


def _test() -> None:
    ts = _get_data.load_data(3)

    ts_period = _period.ts_period(ts)
    ts_trend, ts_season, ts_residuals = _detrend.decompose(ts,
                                                           ts_period=ts_period)
    ts = ts.to_numpy()

    res = MFETSGeneral.ft_skewness(ts_residuals)
    print(res)

    res = MFETSGeneral.ft_length(ts)
    print(res)

    res = MFETSGeneral.ft_kurtosis(ts_residuals)
    print(res)

    res = MFETSGeneral.ft_sd(ts_residuals)
    print(res)

    res = MFETSGeneral.ft_trend(ts_residuals, ts_trend + ts_residuals)
    print(res)

    res = MFETSGeneral.ft_seasonality(ts_residuals, ts_season + ts_residuals)
    print(res)

    res = MFETSGeneral.ft_frac_tp(ts)
    print(res)

    res = MFETSGeneral.ft_frac_sc(ts)
    print(res)

    res = MFETSGeneral.ft_pred(
        _embed.embed_ts(ts, dim=int(np.ceil(np.log10(ts.size)))))
    print(res)

    res = MFETSGeneral.ft_exp_max_lyap(ts,
                                       embed_dim=int(np.ceil(np.log10(
                                           ts.size))),
                                       lag=1)
    print(res)

    res = MFETSGeneral.ft_exp_hurst(ts)
    print(res)

    res = MFETSGeneral.ft_spikiness(ts_residuals)
    print(np.var(res))

    res = MFETSGeneral.ft_lumpiness(ts)
    print("lumpiness", np.var(res))

    res = MFETSGeneral.ft_stability(ts)
    print("stability", np.var(res))

    res = MFETSGeneral.ft_shift_level(ts)
    print(np.nanmax(res))

    res = MFETSGeneral.ft_shift_var(ts)
    print(np.nanmax(res))

    res = MFETSGeneral.ft_frac_cp(ts)
    print(res)

    res = MFETSGeneral.ft_fs_len(ts)
    print(res)

    res = MFETSGeneral.ft_peak_frac(ts, ts_period=12)
    print(res)

    res = MFETSGeneral.ft_trough_frac(ts, ts_period=12)
    print(res)

    res = MFETSGeneral.ft_sd_diff(ts)
    print(res)

    res = MFETSGeneral.ft_skewness_diff(ts)
    print(res)

    res = MFETSGeneral.ft_kurtosis_diff(ts)
    print(res)

    res = MFETSGeneral.ft_walker_cross_frac(ts)
    print(res)

    res = MFETSGeneral.ft_binmean(ts)
    print(res)

    res = MFETSGeneral.ft_period(ts)
    print(res)

    res = MFETSGeneral.ft_sd_diff(ts)
    print(res)

    res = MFETSGeneral.ft_linearity(ts_trend)
    print(res)

    res = MFETSGeneral.ft_trev(ts, only_numerator=True)
    print(res)


if __name__ == "__main__":
    _test()
