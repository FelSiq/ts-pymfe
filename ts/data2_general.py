import typing as t

import numpy as np
import pymfe.statistical

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
        """
        return pymfe.statistical.MFEStatistical.ft_sd(ddof=ddof)

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
        ts_kurt = np.apply_along_axis(func1d=_summary.sum_kurtosis,
                                      axis=0,
                                      arr=N,
                                      method=method,
                                      bias=bias)

        return ts_kurt


def _test() -> None:
    ts = get_data.load_data()
    ts_detrended = data1_detrend.detrend(ts, degrees=1)
    res = MFETSFreqDomain._calc_power_spec(ts_detrended)
    print(res)

    res = MFETSFreqDomain.ft_ps_max(ts_detrended)
    print(res)


if __name__ == "__main__":
    _test()
