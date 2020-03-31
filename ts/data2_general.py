import typing as t

import numpy as np
import pymfe.statistical
import statsmodels.stats.stattools

import data1_detrend
import get_data


class MFETSGeneral:
    @classmethod
    def ft_sd(cls, ts_detrended: np.ndarray, ddof: int = 1) -> float:
        """Compute the standard deviation of the detrended time-series.
        
        Parameters
        ----------
        ts_detrended : :obj:`np.ndarray`
            One-dimensional detrended time-series values.

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
        return np.std(ts_detrended, ddof=ddof)

    @classmethod
    def ft_skewness(cls,
                    ts_detrended: np.ndarray,
                    method: int = 3,
                    bias: bool = True) -> float:
        """Compute the skewness of the detrended time-series.

        Parameters
        ----------
        ts_detrended : :obj:`np.ndarray`
            One-dimensional detrended time-series values.

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
        ts_skew = pymfe.statistical.MFEStatistical.ft_skewness(N=ts_detrended,
                                                               method=method,
                                                               bias=bias)

        return ts_skew

    @classmethod
    def ft_kurtosis(cls,
                    ts_detrended: np.ndarray,
                    method: int = 3,
                    bias: bool = True) -> float:
        """Compute the kurtosis of the detrended time-series.

        Parameters
        ----------
        ts_detrended : :obj:`np.ndarray`
            One-dimensional detrended time-series values.

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
        ts_kurt = pymfe.statistical.MFEStatistical.ft_kurtosis(N=ts_detrended,
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
                 ts: np.ndarray,
                 ts_detrended: np.ndarray,
                 ddof: int = 1) -> float:
        """Ratio of standard deviations of time-series and after detrend.

        Parameters
        ----------
        ts: :obj:`np.ndarray`
            One-dimensional time-series values.

        ts_detrended : :obj:`np.ndarray`
            One-dimensional detrended time-series values.

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
        return np.std(ts, ddof=ddof) / np.std(ts_detrended, ddof=ddof)

    @classmethod
    def ft_dw(cls, ts_detrended: np.ndarray) -> float:
        """Durbin-Watson test statistic value.

        This measure is in [0, 4] range.

        Parameters
        ----------
        ts_detrended : :obj:`np.ndarray`
            One-dimensional detrended time-series values.

        Returns
        -------
        float
            Durbin-Watson test statistic for the detrended time-series.

        References
        ----------
        TODO.
        """
        return statsmodels.stats.stattools.durbin_watson(ts_detrended)

    @classmethod
    def ft_tp(cls, ts: np.ndarray) -> float:
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
        diff_sign_arr = np.sign(np.ediff1d(ts))
        tp_frac = np.mean(np.equal(-1, diff_sign_arr[1:] * diff_sign_arr[:-1]))

        return tp_frac

    @classmethod
    def ft_sc(cls, ts: np.ndarray, ddof: int = 1) -> float:
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


def _test() -> None:
    ts = get_data.load_data()
    ts_detrended = data1_detrend.detrend(ts, degrees=1)

    res = MFETSGeneral.ft_skewness(ts_detrended)
    print(res)

    res = MFETSGeneral.ft_kurtosis(ts_detrended)
    print(res)

    res = MFETSGeneral.ft_sd(ts_detrended)
    print(res)

    res = MFETSGeneral.ft_trend(ts, ts_detrended)
    print(res)

    res = MFETSGeneral.ft_dw(ts_detrended)
    print(res)

    res = MFETSGeneral.ft_tp(ts)
    print(res)

    res = MFETSGeneral.ft_sc(ts)
    print(res)


if __name__ == "__main__":
    _test()
