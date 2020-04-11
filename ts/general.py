import typing as t

import sklearn.preprocessing
import numpy as np
import scipy.spatial
import scipy.odr
import pandas as pd

import autocorr
import _detrend
import _embed
import _period
import _utils
import _get_data


class MFETSGeneral:
    @staticmethod
    def _calc_season_mode_ind(ts_season: np.ndarray, ts_period: int,
                              indfunc: t.Callable[[np.ndarray], float]) -> int:
        """Calculate a mode index based on the time-series seasonality.

        Used by both ``ft_trough_frac`` and ``ft_peak_frac`` to calculate,
        respectively, the mode of the argmin and argmax for all seasons.
        """
        inds = np.arange(ts_period)

        inds = np.array([
            indfunc(ts_season[i * ts_period + inds])
            for i in np.arange(1, ts_season.size // ts_period)
        ],
                        dtype=int)

        mode_inds, _ = scipy.stats.mode(inds)
        return mode_inds[0] + 1

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
        ts_disc = np.digitize(ts, np.linspace(np.min(ts), np.max(ts),
                                              num_bins))
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

    @classmethod
    def ft_linearity(cls, ts_trend: np.ndarray) -> float:
        """TODO."""
        ts_trend_scaled = sklearn.preprocessing.StandardScaler().fit_transform(
            ts_trend.reshape(-1, 1))

        return -1.0

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
    def ft_sample_entropy(cls,
                          ts: np.ndarray,
                          embed_dim: int = 2,
                          factor: float = 0.2,
                          metric: str = "chebyshev",
                          p: t.Union[int, float] = 2,
                          lag: int = 1,
                          ddof: int = 1) -> float:
        """TODO."""
        def neigh_num(dim: int) -> int:
            embed = _embed.embed_ts(ts, dim=dim, lag=lag)
            dist_mat = scipy.spatial.distance.pdist(embed, metric=metric, p=p)
            return np.sum(dist_mat < threshold)

        threshold = factor * np.std(ts, ddof=ddof)

        sample_entropy = -np.log(
            neigh_num(embed_dim + 1) / neigh_num(embed_dim))

        return sample_entropy

    @classmethod
    def ft_embed_in_sphere(cls,
                           ts: np.ndarray,
                           radius: t.Union[int, float] = 1,
                           embed_dim: int = 2,
                           lag: t.Optional[int] = None,
                           normalize_ts: bool = True,
                           normalize: bool = True,
                           ts_acfs: t.Optional[np.ndarray] = None,
                           max_nlags: t.Optional[int] = None,
                           unbiased: bool = True) -> t.Union[int, float]:
        """TODO."""
        if radius <= 0:
            raise ValueError(
                "'radius' must be positive (got {}).".format(radius))

        if lag is None:
            lag = autocorr.MFETSAutocorr.ft_first_acf_nonpos(
                ts=ts, ts_acfs=ts_acfs, max_nlags=max_nlags, unbiased=unbiased)

        if normalize_ts:
            ts = sklearn.preprocessing.StandardScaler().fit_transform(
                ts.reshape(-1, 1)).ravel()

        # Note: embed is given by x(t) = [x(t-1), x(t-2), ..., x(t-m+1)]^T
        embed = _embed.embed_ts(ts, dim=embed_dim, lag=lag)

        # Note: here we supposed that every embed forms a zero-centered
        # hypersphere using the regular hypersphere equation:
        # sqrt(x_{i}^2 + x^{i-1}^2 + ... + x^{i-m+1}) = Radius
        embed_radius = np.linalg.norm(embed, ord=2, axis=1)

        # Note: we can check if every embed is in the same zero-centered
        # hypersphere because all hypersphere embeds are also zero-centered.
        in_hypersphere = np.sum(embed_radius <= radius)

        if normalize:
            in_hypersphere /= embed_radius.size

        return in_hypersphere

    @classmethod
    def ft_hist_entropy(cls,
                        ts: np.ndarray,
                        num_bins: int = 10,
                        normalize: bool = True) -> np.ndarray:
        """TODO."""
        ts_disc = np.digitize(ts, np.linspace(np.min(ts), np.max(ts),
                                              num_bins))

        entropy = scipy.stats.entropy(ts_disc, base=2)

        if normalize:
            entropy /= np.log2(ts_disc.size)

        return entropy


def _test() -> None:
    ts = _get_data.load_data(3)

    ts_period = _period.ts_period(ts)
    ts_trend, ts_season, ts_residuals = _detrend.decompose(ts,
                                                           ts_period=ts_period)
    ts = ts.to_numpy()

    res = MFETSGeneral.ft_hist_entropy(ts)
    print(res)

    res = MFETSGeneral.ft_embed_in_sphere(ts)
    print(res)

    res = MFETSGeneral.ft_length(ts)
    print(res)

    res = MFETSGeneral.ft_frac_tp(ts)
    print(res)

    res = MFETSGeneral.ft_frac_sc(ts)
    print(res)

    res = MFETSGeneral.ft_pred(
        _embed.embed_ts(ts, dim=int(np.ceil(np.log10(ts.size)))))
    print(res)

    res = MFETSGeneral.ft_frac_cp(ts)
    print(res)

    res = MFETSGeneral.ft_fs_len(ts)
    print(res)

    res = MFETSGeneral.ft_peak_frac(ts, ts_period=12)
    print(res)

    res = MFETSGeneral.ft_trough_frac(ts, ts_period=12)
    print(res)

    res = MFETSGeneral.ft_walker_cross_frac(ts)
    print(res)

    res = MFETSGeneral.ft_binmean(ts)
    print(res)

    res = MFETSGeneral.ft_period(ts)
    print(res)

    res = MFETSGeneral.ft_linearity(ts_trend)
    print(res)

    res = MFETSGeneral.ft_sample_entropy(ts, lag=1)
    print(res)


if __name__ == "__main__":
    _test()
