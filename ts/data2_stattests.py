import typing as t

import numpy as np
import arch.unitroot
import statsmodels.stats.stattools
import statsmodels.stats.diagnostic

import data1_period
import data1_detrend
import data1_embed
import get_data


class MFETSStatTests:
    @classmethod
    def ft_test_dw(cls,
                   ts_residuals: np.ndarray,
                   normalize: bool = True) -> float:
        """Durbin-Watson test statistic value.

        The ADF test works as follows:
        Null Hypothesis (NH): there is no serial correlation.
        Alternative Hypothesis (AH): there is serial correlation.

        This test statistic value is in [0, 4] range.

        Parameters
        ----------
        ts_residuals : :obj:`np.ndarray`
            Residuals (random noise) of an one-dimensional time-series.

        normalize : :obj:`bool`, optional
            If True, divide the result by 4 to make the test statistic in
            the [0, 1] range.

        Returns
        -------
        float
            Durbin-Watson test statistic for the time-series residuals.
            If ``normalize`` is True, this test statistic is normalized
            in [0, 1] range.

        References
        ----------
        TODO.
        """
        dw_stat = statsmodels.stats.stattools.durbin_watson(ts_residuals)

        if normalize:
            dw_stat *= 0.25

        return dw_stat

    @classmethod
    def ft_test_lb(cls,
                   ts_residuals: np.ndarray,
                   lags: t.Optional[int] = 5,
                   return_pval: bool = False) -> float:
        """Ljung-Box (LB) test of autocorrelation in residuals.

        The LB test works as follows:
        TODO.

        Parameters
        ----------
        ts_residuals : :obj:`np.ndarray`
            Residuals (random noise) of an one-dimensional time-series.

        TODO.
        Returns
        -------
        TODO.

        References
        ----------
        TODO.
        """
        res = statsmodels.stats.diagnostic.acorr_ljungbox(ts_residuals,
                                                          lags=lags,
                                                          return_df=False)

        stat, pvalue = res

        if return_pval:
            return pvalue

        return stat

    @classmethod
    def ft_test_adf(cls,
                    ts: np.ndarray,
                    max_lags: t.Optional[int] = None,
                    return_pval: bool = False) -> float:
        """Augmented Dickey-Fuller (ADF) test.

        The ADF test works as follows:
        Null Hypothesis (NH): there is a unit root.
        Alternative Hypothesis (AH): there is not a unit root.

        TODO.
        """
        res = statsmodels.tsa.stattools.adfuller(ts, maxlag=max_lags)

        (stat, pvalue), _ = res[:2], res[2:]

        if return_pval:
            return pvalue

        return stat

    @classmethod
    def ft_test_adf_gls(cls,
                        ts: np.ndarray,
                        lag: t.Optional[int] = None,
                        max_lags: t.Optional[int] = None,
                        return_pval: bool = False) -> float:
        """Dickey-Fuller GLS (ADF-GLS) test.

        The ADF-GLS test works as follows:
        Null Hypothesis (NH): there is a unit root.
        Alternative Hypothesis (AH): there is not a unit root.

        TODO.
        """
        res = arch.unitroot.DFGLS(ts, lags=lag, max_lags=max_lags)

        if return_pval:
            return res.pvalue

        return res.stat

    @classmethod
    def ft_test_pp(cls,
                   ts: np.ndarray,
                   lag: t.Optional[int] = None,
                   return_pval: bool = False) -> float:
        """Phillips-Perron (PP) test.

        The PP test works as follows:
        Null Hypothesis (NH): there is a unit root.
        Alternative Hypothesis (AH): there is not a unit root.

        TODO.
        """
        res = arch.unitroot.PhillipsPerron(ts, lags=lag)

        if return_pval:
            return res.pvalue

        return res.stat

    @classmethod
    def ft_test_kpss(cls,
                     ts: np.ndarray,
                     lag: t.Optional[int] = None,
                     return_pval: bool = False) -> float:
        """Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test.

        The KPSS test works as follows:
        Null Hypothesis (NH): the series is weakly stationary.
        Alternative Hypothesis (AH): the series is non-stationary.

        Return
        ------
        TODO.
        """
        res = arch.unitroot.KPSS(ts, lags=lag)

        if return_pval:
            return res.pvalue

        return res.stat

    @classmethod
    def ft_test_za(cls, ts: np.ndarray, return_pval: bool = False) -> float:
        """Zivot-Andrews (ZA) Test.

        The ZA test works as follows:
        Null Hypothesis (NH): the process contains a unit root with a
        single structural break.
        Alternative Hypothesis (AH): The process is trend and break
        stationary.

        Return
        """
        res = arch.unitroot.ZivotAndrews(ts)

        if return_pval:
            return res.pvalue

        return res.stat


def _test() -> None:
    ts = get_data.load_data(3)
    ts_period = data1_period.ts_period(ts)
    ts_trend, ts_season, ts_residuals = data1_detrend.decompose(ts, ts_period=ts_period)
    ts = ts.to_numpy().astype(float)

    res = MFETSStatTests.ft_test_adf(ts, return_pval=False)
    print(res)

    res = MFETSStatTests.ft_test_adf_gls(ts, return_pval=False)
    print(res)

    res = MFETSStatTests.ft_test_kpss(ts, return_pval=False)
    print(res)

    res = MFETSStatTests.ft_test_pp(ts, return_pval=False)
    print(res)

    res = MFETSStatTests.ft_test_dw(ts_residuals)
    print(res)

    res = MFETSStatTests.ft_test_lb(ts_residuals)
    print(res)

    res = MFETSStatTests.ft_test_za(ts, return_pval=False)
    print(res)


if __name__ == "__main__":
    _test()
