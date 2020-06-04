"""Module dedicated to global statistics time-series meta-features."""
import typing as t
import warnings

import numpy as np
import nolds
import scipy.stats

import pymfe._period as _period
import pymfe._utils as _utils
import pymfe._summary as _summary


class MFETSGlobalStats:
    """Extract time-series meta-features from Global Statistics group."""
    @classmethod
    def precompute_period(cls, ts: np.ndarray, **kwargs) -> t.Dict[str, int]:
        """Precompute the time-series period.

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
                * ``ts_period`` (:obj:`int`): time-series period.
        """
        precomp_vals = {}  # type: t.Dict[str, int]

        if "ts_period" not in kwargs:
            precomp_vals["ts_period"] = _period.get_ts_period(ts=ts)

        return precomp_vals

    @classmethod
    def ft_ioe_tdelta_mean(
            cls,
            ts: np.ndarray,
            step_size: float = 0.05,
            normalize: bool = True,
            differentiate: bool = False,
            ts_scaled: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Mean change of interval length with iterative outlier exclusion.

        This method calculates, at each iteration, the mean of the differences
        of the timestamps of instances using the iterative outlier exclusion
        strategy.

        In the iterative outlier exclusion, a uniformly spaced set of
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
            of all iterations of the iterative outlier exclusion process. If
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
        tdelta_it_mean = _utils.calc_ioe_stats(ts=ts,
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

        ddof : int, optional (default=1)
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

        ddof : int, optional (default=1)
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

        ddof : int, optional (default=1)
            Degrees of freedom for standard deviation.

        Returns
        -------
        float
            Standard deviation of the time-series residuals.
        """
        return np.std(ts_residuals, ddof=ddof)

    @classmethod
    def ft_sd_diff(cls,
                   ts: np.ndarray,
                   num_diff: int = 1,
                   ddof: int = 1) -> float:
        """Standard deviation of the nth-order differenced time-series.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        num_diff : int, optional (default=1)
            Order of the differentiation.

        ddof : float, optional (default=1)
            Degrees of freedom for standard deviation.

        Returns
        -------
        float
            Standard deviation of the nth-order differenced time-series.
        """
        return np.std(np.diff(ts, n=num_diff), ddof=ddof)

    @classmethod
    def ft_sd_sdiff(cls,
                    ts: np.ndarray,
                    ddof: int = 1,
                    ts_period: t.Optional[int] = None) -> float:
        """Seasonal standard dev.  of the first-order differenced time-series.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        ddof : int, optional (default=1)
            Degrees of freedom for standard deviation.

        ts_period : int, optional
            Time-series period. Used to take advantage of precomputations.

        Returns
        -------
        float
            Standard deviation of the first-order difference of the lagged
            time-series by its own period.
        """
        _ts_period = _period.get_ts_period(ts=ts, ts_period=ts_period)
        ts_sdiff = ts[_ts_period:] - ts[:-_ts_period]
        return np.std(ts_sdiff, ddof=ddof)

    @classmethod
    def ft_skewness_residuals(cls,
                              ts_residuals: np.ndarray,
                              method: int = 3,
                              unbiased: bool = False) -> float:
        """Compute the skewness of the time-series residuals.

        Parameters
        ----------
        ts_residuals : :obj:`np.ndarray`
            Residuals (random noise) of an one-dimensional time-series.

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
        ts_skew = _summary.sum_skewness(values=ts_residuals,
                                        method=method,
                                        bias=not unbiased)

        return float(ts_skew)

    @classmethod
    def ft_skewness_diff(cls,
                         ts: np.ndarray,
                         num_diff: int = 1,
                         method: int = 3,
                         unbiased: bool = False) -> float:
        """Skewness of the nth-order differenced time-series.

        This method calculates the skewness of the nth-order differenced
        time-series (with lag = 1), with `n` being given by `num_diff`.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        num_diff : int, optional (default=1)
            Order of the differentiation.

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

        Returns
        -------
        float
            Skewness of the nth-order differenced time-series

        References
        ----------
        .. [1] Donald Michie, David J. Spiegelhalter, Charles C. Taylor, and
           John Campbell. Machine Learning, Neural and Statistical
           Classification, volume 37. Ellis Horwood Upper Saddle River, 1994.
        """
        ts_diff = np.diff(ts, n=num_diff)
        ts_skew = _summary.sum_skewness(values=ts_diff,
                                        method=method,
                                        bias=not unbiased)

        return float(ts_skew)

    @classmethod
    def ft_skewness_sdiff(cls,
                          ts: np.ndarray,
                          method: int = 3,
                          unbiased: bool = False,
                          ts_period: t.Optional[int] = None) -> float:
        """Seasonal skewness of the first-order differenced time-series.

        This method calculates the skewness of the first-order differenced
        time-series, lagged with its period.

        If the time-series is not seasonal, then its period is assumed to be 1.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

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

        ts_period : int, optional
            Time-series period. Used to take advantage of precomputations.

        Returns
        -------
        float
            Skewness of the first-order difference of the lagged time-series
            by its own period.
        """
        _ts_period = _period.get_ts_period(ts=ts, ts_period=ts_period)
        ts_sdiff = ts[_ts_period:] - ts[:-_ts_period]
        ts_skew = _summary.sum_skewness(values=ts_sdiff,
                                        method=method,
                                        bias=not unbiased)

        return float(ts_skew)

    @classmethod
    def ft_kurtosis_residuals(cls,
                              ts_residuals: np.ndarray,
                              method: int = 3,
                              unbiased: bool = False) -> float:
        """Compute the kurtosis of the time-series residuals.

        Parameters
        ----------
        ts_residuals : :obj:`np.ndarray`
            Residuals (random noise) of an one-dimensional time-series.

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
        ts_kurt = _summary.sum_kurtosis(values=ts_residuals,
                                        method=method,
                                        bias=not unbiased)

        return float(ts_kurt)

    @classmethod
    def ft_kurtosis_diff(cls,
                         ts: np.ndarray,
                         num_diff: int = 1,
                         method: int = 3,
                         unbiased: bool = False) -> float:
        """Kurtosis of the nth-order differenced time-series.

        This method calculates the kurtosis of the nth-order differenced
        time-series (with lag = 1), with `n` being given by `num_diff`.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        num_diff : int, optional (default=1)
            Order of the differentiation.

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

        Returns
        -------
        float
            Kurtosis of the nth-order differenced time-series
        """
        ts_diff = np.diff(ts, n=num_diff)
        ts_kurt = _summary.sum_kurtosis(values=ts_diff,
                                        method=method,
                                        bias=not unbiased)

        return float(ts_kurt)

    @classmethod
    def ft_kurtosis_sdiff(cls,
                          ts: np.ndarray,
                          method: int = 3,
                          unbiased: bool = False,
                          ts_period: t.Optional[int] = None) -> float:
        """Seasonal kurtosis of the first-order differenced time-series.

        This method calculates the kurtosis of the first-order differenced
        time-series, lagged with its period.

        If the time-series is not seasonal, then its period is assumed to be 1.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

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

        ts_period : int, optional
            Time-series period. Used to take advantage of precomputations.

        Returns
        -------
        float
            Kurtosis of the first-order difference of the lagged time-series
            by its own period.
        """
        _ts_period = _period.get_ts_period(ts=ts, ts_period=ts_period)
        ts_sdiff = ts[_ts_period:] - ts[:-_ts_period]
        ts_kurt = _summary.sum_kurtosis(values=ts_sdiff,
                                        method=method,
                                        bias=not unbiased)

        return float(ts_kurt)

    @classmethod
    def ft_exp_max_lyap(cls,
                        ts: np.ndarray,
                        embed_dim: int = 10,
                        lag: t.Optional[int] = None) -> float:
        """Estimation of the maximum Lyapunov coefficient.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        embed_dim : int, optional (default=10)
            Time-series embed dimension.

        lag : int, optional
            Lag of the embed.

        Returns
        -------
        float
            Estimation of the maximum Lyapunov coefficient.

        References
        ----------
        .. [1] H. E. Hurst, The problem of long-term storage in reservoirs,
            International Association of Scientific Hydrology. Bulletin, vol.
            1, no. 3, pp. 13–27, 1956.
        .. [2] H. E. Hurst, A suggested statistical model of some time series
            which occur in nature, Nature, vol. 180, p. 494, 1957.
        .. [3] R. Weron, Estimating long-range dependence: finite sample
            properties and confidence intervals, Physica A: Statistical
            Mechanics and its Applications, vol. 312, no. 1, pp. 285–299,
            2002.
        .. [4] "nolds" Python package: https://pypi.org/project/nolds/
        .. [5] Lemke, Christiane & Gabrys, Bogdan. (2010). Meta-learning for
            time series forecasting and forecast combination. Neurocomputing.
            73. 2006-2016. 10.1016/j.neucom.2009.09.020.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    module="nolds",
                                    category=RuntimeWarning)

            max_lyap_exp = nolds.lyap_r(data=ts, lag=lag, emb_dim=embed_dim)

        return max_lyap_exp

    @classmethod
    def ft_exp_hurst(cls, ts: np.ndarray) -> float:
        """Estimation of the Hurst exponent.

        Check `nolds.hurst_rs` documentation for a clear explanation about
        the underlying function.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        Returns
        -------
        float
            Estimation of the hurst exponent.

        References
        ----------
        .. [1] H. E. Hurst, The problem of long-term storage in reservoirs,
            International Association of Scientific Hydrology. Bulletin, vol.
            1, no. 3, pp. 13–27, 1956.
        .. [2] H. E. Hurst, A suggested statistical model of some time series
            which occur in nature, Nature, vol. 180, p. 494, 1957.
        .. [3] R. Weron, Estimating long-range dependence: finite sample
            properties and confidence intervals, Physica A: Statistical
            Mechanics and its Applications, vol. 312, no. 1, pp. 285–299,
            2002.
        .. [4] "nolds" Python package: https://pypi.org/project/nolds/
        """
        return nolds.hurst_rs(data=ts)

    @classmethod
    def ft_dfa(cls,
               ts: np.ndarray,
               pol_order: int = 1,
               overlap_windows: bool = True) -> float:
        """Calculate the Hurst parameter from Detrended fluctuation analysis.

        Note that the ``Hurst parameter`` is not the same quantity as the
        ``Hurst exponent``. The Hurst parameter `H` is defined as the quantity
        such that the following holds: std(ts, l * n) = l ** H * std(ts, n),
        where `ts` is the time-series, `l` is a constant factor, `n` is some
        window length of `ts`, and std(ts, k) is the standard deviation of
        `ts` within a window of size `k`.

        Check `nolds.dfa` documentation for a clear explanation about the
        underlying function.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        pol_order : int, optional (default=1)
            Order of the detrending polynomial within each window of the
            analysis.

        overlap_windows : bool, optional (default=True)
            If True, overlap the windows used while performing the analysis.

        Returns
        -------
        float
            Hurst parameter.

        References
        ----------
        .. [1] C.-K. Peng, S. V. Buldyrev, S. Havlin, M. Simons,
            H. E. Stanley, and A. L. Goldberger, Mosaic organization of
            DNA nucleotides, Physical Review E, vol. 49, no. 2, 1994.
        .. [2] R. Hardstone, S.-S. Poil, G. Schiavone, R. Jansen,
            V. V. Nikulin, H. D. Mansvelder, and K. Linkenkaer-Hansen,
            Detrended fluctuation analysis: A scale-free view on neuronal
            oscillations, Frontiers in Physiology, vol. 30, 2012.
        .. [3] "nolds" Python package: https://pypi.org/project/nolds/
        """
        hurst_coeff = nolds.dfa(ts, order=pol_order, overlap=overlap_windows)
        return hurst_coeff

    @classmethod
    def ft_corr_dim(cls, ts: np.ndarray, emb_dim: int = 1) -> float:
        """Correlation dimension of the time-series.

        It is used the Grassberger-Procaccia algorithm for the correlation
        dimension estimation.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        emb_dim : int, optional (default=1)
            Embedding dimension to estimate the correlation dimension.

        Returns
        -------
        float
            Estimated correlation dimension.

        References
        ----------
        .. [1] P. Grassberger and I. Procaccia, Characterization of strange
            attractors, Physical review letters, vol. 50, no. 5, p. 346,
            1983.
        .. [2] P. Grassberger and I. Procaccia, Measuring the strangeness of
            strange attractors, Physica D: Nonlinear Phenomena, vol. 9,
            no. 1, pp. 189–208, 1983.
        .. [3] P. Grassberger, Grassberger-Procaccia algorithm,
            Scholarpedia, vol. 2, no. 5, p. 3043.
        .. [4] "nolds" Python package. URL: https://pypi.org/project/nolds/
        """
        try:
            corr_dim = nolds.corr_dim(ts, emb_dim=emb_dim)

        except AssertionError:
            corr_dim = np.nan

        return corr_dim

    @classmethod
    def ft_opt_boxcox_coef(cls,
                           ts: np.ndarray,
                           adjust_data: bool = True) -> float:
        """Estimated optimal box-cox transformation coefficient.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        adjust_data : bool, optional (default=True)
            If True, transform the data to y(t) = ts(t) - min(ts) + 1. This is
            required for non-positive data. If False, estimate the coefficient
            with the original data, possibly failing if the time-series have
            non-positive data.

        Returns
        -------
        float
            Estimated optimal box-cox transformation coefficient.

        References
        ----------
        .. [1] Box, G. E. P. and Cox, D. R. (1964). An analysis of
            transformations, Journal of the Royal Statistical Society, Series
            B, 26, 211-252.
        .. [2] Yanfei Kang, Rob J. Hyndman, Kate Smith-Miles, Visualising
            forecasting algorithm performance using time series instance
            spaces, International Journal of Forecasting, Volume 33, Issue 2,
            2017, Pages 345-358, ISSN 0169-2070,
            https://doi.org/10.1016/j.ijforecast.2016.09.004.
        """
        if adjust_data:
            ts = ts - ts.min() + 1

        return scipy.stats.boxcox_normmax(ts, method="mle")

    @classmethod
    def ft_t_mean(cls, ts: np.ndarray, pcut: float = 0.02) -> float:
        """Trimmed mean of the time-series values.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        pcut : float, optional (default=0.02)
            Proportion of outlier cut. Must be in [0, 0.5) range.

        Returns
        -------
        float
            Trimmed mean of time-series.

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
        return scipy.stats.trim_mean(ts, proportiontocut=pcut)

    @classmethod
    def ft_spikiness(cls,
                     ts_residuals: np.ndarray,
                     ddof: int = 1) -> np.ndarray:
        """Spikiness of the time-series residuals.

        The spikiness of the time-series residuals is the variance of the
        variance with jackknife resampling (leave-one-out) on the residuals.
        Here, in order to enable other times of summarization, we return all
        the `jackknifed` variances.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        ddof : int, optional (default=1)
            Degrees of freedom to calculate the variances.

        Returns
        -------
        :obj:`np.ndarray`
            Spikiness of the time-series residuals.

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
        vars_ = np.array([
            np.var(np.delete(ts_residuals, i), ddof=ddof)
            for i in np.arange(ts_residuals.size)
        ])

        # Note: on the original reference paper, the spikiness is calculated
        # as the variance of the 'vars_'. However, to enable summarization,
        # here we return the full array.
        return vars_
