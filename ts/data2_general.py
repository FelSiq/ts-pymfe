import typing as t

import numpy as np
import pymfe.statistical
import nolds
import scipy.spatial
import scipy.odr

import data1_detrend
import data1_embed
import get_data


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

        return max(0.0, trend)

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

        return max(0.0, seas)

    @classmethod
    def ft_tp_frac(cls, ts: np.ndarray) -> float:
        """Fraction of turning points in the time-series.

        A turning point is a time-series point `p_{i}` which both neighbor
        values, p_{i-1} and p_{i+1}, are either lower (p_{i} > p_{i+1} and
        p_{i} > p_{i-1}) or higher (p_{i} < p_{i+1} and p_{i} < p_{i-1}).

        Parameters
        ----------
        ts: :obj:`np.ndarray`
            One-dimensional time-series values.

        Returns
        -------
        float
            Fraction of turning points in the time-series.

        References
        ----------
        TODO.
        """
        diff_arr = np.ediff1d(ts)
        tp_frac = np.mean(diff_arr[1:] * diff_arr[:-1] < 0)

        return tp_frac

    @classmethod
    def ft_sc_frac(cls, ts: np.ndarray, ddof: int = 1) -> float:
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

        Returns
        -------
        float
            Fraction of step change points in the time-series.

        References
        ----------
        TODO.
        """
        ts_cmeans = np.cumsum(ts) / np.arange(1, ts.size + 1)

        ts_mean_abs_div = np.abs(ts[1:] - ts_cmeans[:-1])

        sc_num = 0

        for i in np.arange(1 + ddof, ts.size):
            sc_num += int(
                ts_mean_abs_div[i - 1] > 2 * np.std(ts[:i], ddof=ddof))

        return sc_num / (ts.size - 1)

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
    def ft_tilled_var(cls,
                      ts: np.ndarray,
                      num_windows: int = 16,
                      ddof: int = 1) -> np.ndarray:
        """TODO."""
        if num_windows > ts.size:
            raise ValueError("'num_windows' ({}) larger than the "
                             "time-series size ({}).".format(
                                 num_windows, ts.size))

        vars_ = np.array([
            np.var(split, ddof=ddof)
            for split in np.array_split(ts, num_windows)
        ],
                         dtype=float)

        # Note: this feature, when summarized with 'mean', becomes the
        # 'Stability' metafeature, and when summarized with 'var' becomes
        # the 'lumpiness' of the time-series.
        return vars_

    @classmethod
    def _fit_ord_quad_model(cls, ts: np.ndarray) -> t.Any:
        """https://docs.scipy.org/doc/scipy/reference/odr.html"""

    @classmethod
    def ft_linearity(cls,
                     ts: np.ndarray,
                     model_ort_quad: t.Optional[t.Any] = None) -> float:
        """TODO."""
        if model_ort_quad is None:
            model_ort_quad = cls._fit_ord_quad_model(ts=ts)

        return


def _test() -> None:
    ts = get_data.load_data(2)
    ts_trend, ts_season, ts_residuals = data1_detrend.decompose(ts)
    ts = ts.to_numpy()
    """
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

    res = MFETSGeneral.ft_tp_frac(ts)
    print(res)

    res = MFETSGeneral.ft_sc_frac(ts)
    print(res)

    res = MFETSGeneral.ft_pred(
        data1_embed.embed_ts(ts, dim=int(np.ceil(np.log10(ts.size)))))
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

    res = MFETSGeneral.ft_tilled_var(ts)
    print(res)
    """


if __name__ == "__main__":
    _test()
