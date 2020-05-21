"""Module dedicated to global time-series meta-features."""
import typing as t
import warnings

import numpy as np
import pymfe.statistical
import nolds
import scipy.stats

import _detrend
import _period
import _utils
import _get_data


class MFETSGlobalStats:
    """Extract time-series meta-features from Global Statistics group."""
    @classmethod
    def ft_ioi_tdelta_mean(
            cls,
            ts: np.ndarray,
            step_size: float = 0.05,
            normalize: bool = True,
            differentiate: bool = False,
            ts_scaled: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Mean change of interval length with iterative outlier inclusion.

        This method calculates, at each iteration, the mean of the differences
        of the timestamps of instances using the iterative outlier inclusion
        strategy.

        In the iterative outlier inclusion, a uniformly spaced set of
        thresholds over the time-series range is build and, for each iteration,
        it is calculated a statistic of the diference of the timestamp values
        of instances larger or equal than the current threshold.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        step_size : float, optional (default=0.05)
            Increase of the outlier threshold in each iteration. Must be a
            number strictly positive.

        normalize : bool, optional (default=True)
            If True, normalize the statistic in the [-1, 1] interval. If
            False, return the raw mean timestamp values.

        differentiate : bool, optional (default=False)
            If True, differentiate the timestamps before calculating each
            statistic. If False, all statistics will be calculated on the
            raw timestamps.

        Returns
        -------
        :obj:`np.ndarray`
            If `differentiate` is False, the mean value of outlier timestamps
            of all iterations of the iterative outlier inclusion process. If
            `differentiate` is True, the mean value of the timestamps interval 
            of outliers for every iteration. Also, if `normalize` is True,
            every value will be normalized to the [-1, 1] range.

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
        tdelta_it_mean = _utils.calc_ioi_stats(ts=ts,
                                               funcs=np.mean,
                                               ts_scaled=ts_scaled,
                                               step_size=step_size,
                                               differentiate=differentiate)

        if normalize:
            tdelta_it_mean = 2 * tdelta_it_mean / ts.size - 1

        return tdelta_it_mean

    @classmethod
    def ft_trend_strenght(cls,
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
        trend = 1.0 - (np.var(ts_residuals, ddof=ddof) /
                       np.var(ts_deseasonalized, ddof=ddof))

        return min(1.0, max(0.0, trend))

    @classmethod
    def ft_season_strenght(cls,
                           ts_residuals: np.ndarray,
                           ts_detrended: np.ndarray,
                           ddof: int = 1) -> float:
        """Ratio of standard deviations of time-series and after deseasoning.

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
            and the standard deviation of the deseasonalized version.

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
    def ft_sd_mdiff(cls,
                    ts: np.ndarray,
                    ts_period: int,
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
        ts_diff = ts[ts_period:] - ts[:-ts_period]
        return np.std(ts_diff, ddof=ddof)

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

            Where `n` is the number of instances in ``ts``, `s` is the standard
            deviation of each attribute in ``ts``, and `m_i` is the ith
            statistical momentum of each attribute in ``ts``.

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
    def ft_skewness_mdiff(cls,
                          ts: np.ndarray,
                          ts_period: int,
                          method: int = 3,
                          bias: bool = True) -> float:
        """TODO."""
        ts_diff = ts[ts_period] - ts[:-ts_period]
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

            Where `n` is the number of instances in ``ts``, `s` is the standard
            deviation of each attribute in ``ts``, and `m_i` is the ith
            statistical momentum of each attribute in ``ts``.

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
    def ft_kurtosis_mdiff(cls,
                          ts: np.ndarray,
                          ts_period: int,
                          method: int = 3,
                          bias: bool = True) -> float:
        """TODO."""
        ts_diff = ts[ts_period:] - ts[:-ts_period]
        ts_kurt = pymfe.statistical.MFEStatistical.ft_kurtosis(N=ts_diff,
                                                               method=method,
                                                               bias=bias)

        return ts_kurt

    @classmethod
    def ft_exp_max_lyap(cls, ts: np.ndarray, embed_dim: int,
                        lag: int) -> float:
        """TODO."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    module="nolds",
                                    category=RuntimeWarning)

            max_lyap_exp = nolds.lyap_r(data=ts, lag=lag, emb_dim=embed_dim)

        return max_lyap_exp

    @classmethod
    def ft_exp_hurst(cls, ts: np.ndarray) -> float:
        """TODO."""
        return nolds.hurst_rs(data=ts)

    @classmethod
    def ft_dfa(cls,
               ts: np.ndarray,
               order: int = 1,
               overlap_windows: bool = True,
               return_coeff: bool = True) -> float:
        """TODO."""
        hurst_coeff = nolds.dfa(ts, order=order, overlap=overlap_windows)
        return hurst_coeff

    @classmethod
    def ft_corr_dim(cls,
                    ts: np.ndarray,
                    emb_dim: int = 1) -> t.Union[np.ndarray, float]:
        """TODO."""
        corr_dim = nolds.corr_dim(ts, emb_dim=emb_dim)
        return corr_dim

    @classmethod
    def ft_opt_boxcox_coef(cls,
                           ts: np.ndarray,
                           epsilon: float = 1.0e-4,
                           num_lambdas: int = 16) -> float:
        """TODO."""
        ts = ts - ts.min() + epsilon
        return scipy.stats.boxcox_normmax(ts, method="mle")

    @classmethod
    def ft_t_mean(cls, ts: np.ndarray, pcut: float = 0.02) -> np.ndarray:
        """Trimmed mean of the time-series values.


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
        return pymfe.statistical.MFEStatistical.ft_t_mean(N=ts, pcut=pcut)


def _test() -> None:
    ts = _get_data.load_data(3)

    ts_period = _period.ts_period(ts)
    ts_trend, ts_season, ts_residuals = _detrend.decompose(ts,
                                                           ts_period=ts_period)
    ts = ts.to_numpy()

    res = MFETSGlobalStats.ft_dfa(ts)
    print(res)

    res = MFETSGlobalStats.ft_corr_dim(ts)
    print("corr_dim", res)
    exit(1)

    res = MFETSGlobalStats.ft_ioi_tdelta_mean(ts)
    print(res)

    res = MFETSGlobalStats.ft_t_mean(ts)
    print("trimmed mean", res)

    res = MFETSGlobalStats.ft_opt_boxcox_coef(ts)
    print(res)

    res = MFETSGlobalStats.ft_sd_diff(ts)
    print("sd diff", res)

    res = MFETSGlobalStats.ft_sd_mdiff(ts, ts_period)
    print("sd mdiff", res)

    res = MFETSGlobalStats.ft_skewness_diff(ts)
    print("skewness diff", res)

    res = MFETSGlobalStats.ft_skewness_mdiff(ts, ts_period)
    print("skewness mdiff", res)

    res = MFETSGlobalStats.ft_kurtosis_diff(ts)
    print("kurtosis diff", res)

    res = MFETSGlobalStats.ft_kurtosis_mdiff(ts, ts_period)
    print("kurtosis mdiff", res)

    res = MFETSGlobalStats.ft_sd_diff(ts)
    print("sd diff", res)

    res = MFETSGlobalStats.ft_exp_max_lyap(ts, embed_dim=ts_period, lag=1)
    print("exp max lyap", res)

    res = MFETSGlobalStats.ft_exp_hurst(ts)
    print("exp hurst", res)

    res = MFETSGlobalStats.ft_skewness_residuals(ts_residuals)
    print(res)

    res = MFETSGlobalStats.ft_kurtosis_residuals(ts_residuals)
    print(res)

    res = MFETSGlobalStats.ft_sd_residuals(ts_residuals)
    print(res)

    res = MFETSGlobalStats.ft_trend_strenght(ts_residuals,
                                             ts_trend + ts_residuals)
    print(res)

    res = MFETSGlobalStats.ft_season_strenght(ts_residuals,
                                              ts_season + ts_residuals)
    print(res)


if __name__ == "__main__":
    _test()
