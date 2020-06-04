"""Module dedicated to statistical tests time-series meta-features."""
import typing as t

import numpy as np
import arch.unitroot
import statsmodels.stats.stattools
import statsmodels.stats.diagnostic


class MFETSStatTests:
    """Extract time-series meta-features from Statistical Tests group."""
    @staticmethod
    def _extract_arch_module_pval(arch_result: t.Any) -> float:
        """Extract the p-value from a result of the `arch` module."""
        try:
            return arch_result.pvalue

        except (IndexError, AssertionError):
            # Note: catching a weird exceptions from arch module.
            return np.nan

    @classmethod
    def ft_test_dw(cls,
                   ts_residuals: np.ndarray,
                   normalize: bool = True) -> float:
        """Durbin-Watson test statistic value.

        This tests tries to detect autocorrelation of lag 1 in the given
        residuals.

        This test statistic value is in [0, 4] range.

        Parameters
        ----------
        ts_residuals : :obj:`np.ndarray`
            Residuals of an one-dimensional time-series.

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
        .. [1] J. DURBIN, G. S. WATSON, TESTING FOR SERIAL CORRELATION IN
            LEAST SQUARES REGRESSION. I, Biometrika, Volume 37, Issue 3-4,
            December 1950, Pages 409–428,
            https://doi.org/10.1093/biomet/37.3-4.409
        .. [2] J. DURBIN, G. S. WATSON, TESTING FOR SERIAL CORRELATION IN
            LEAST SQUARES REGRESSION. II, Biometrika, Volume 38, Issue 1-2,
            June 1951, Pages 159–178,
            https://doi.org/10.1093/biomet/38.1-2.159
        """
        dw_stat = statsmodels.stats.stattools.durbin_watson(ts_residuals)

        if normalize:
            dw_stat *= 0.25

        return dw_stat

    @classmethod
    def ft_test_lb(cls,
                   ts_residuals: np.ndarray,
                   max_nlags: t.Optional[int] = 16,
                   return_pval: bool = True) -> np.ndarray:
        """Ljung-Box (LB) test of autocorrelation in residuals.

        The test is defined as:

        Null Hypothesis (H0): the data are independently distributed (i.e.
            the correlations in the population from which the sample is taken
            are 0, so that any observed correlations in the data result from
            randomness of the sampling process).

        Alternative Hypothesis (HA): The data is not independently distributed;
            they exhibit serial correlation.

        Parameters
        ----------
        ts_residuals : :obj:`np.ndarray`
            Residuals of an one-dimensional time-series.

        max_nlags : int, optional (default=16)
            Maximum number of lags tested for autocorrelation.

        return_pval : bool, optional (default=True)
            If True, return the test p-value instead of the test statistic.

        Returns
        -------
        :obj:`np.ndarray`
            If `return_pval` is False, return the Ljung-Box test statistic
            for every lag from 1 up to ``max_nlags``.
            If `return_pval` is True, return the p-value associated with the
            obtained test statistic for every lag.

        References
        ----------
        .. [1] G. M. LJUNG, G. E. P. BOX, On a measure of lack of fit in time
            series models, Biometrika, Volume 65, Issue 2, August 1978, Pages
            297–303, https://doi.org/10.1093/biomet/65.2.297
        """
        res = statsmodels.stats.diagnostic.acorr_ljungbox(ts_residuals,
                                                          lags=max_nlags,
                                                          return_df=False)

        stat, pvalue = res

        if return_pval:
            return pvalue

        return stat

    @classmethod
    def ft_test_earch(cls,
                      ts_residuals: np.ndarray,
                      max_nlags: t.Optional[int] = 16,
                      return_pval: bool = True) -> float:
        """Engle's Test for Autoregressive Conditional Heteroscedasticity.

        The Engle's test works as follows:

        Null Hypothesis (H0): the residuals of a ARIMA model are homoscedastic.

        Alternative Hypothesis (HA): the residuals of a ARIMA model are not
            homoscedastic.

        Parameters
        ----------
        ts_residuals : :obj:`np.ndarray`
            Residuals of an one-dimensional time-series.

        max_nlags : int, optional (default=16)
            Maximum number of lags included in the test.

        return_pval : bool, optional (default=True)
            If True, return the test p-value instead of the test statistic.

        Returns
        -------
        float
            If `return_pval` is False, return the Engle's test statistic.
            If `return_pval` is True, return the p-value associated with the
            obtained test statistic.

        References
        ----------
        .. [1] Engle, Robert F. "Autoregressive Conditional Heteroskedasticity
            with Estimates of the Variance of United Kingdom Inflation."
            Econometrica. Vol. 50, 1982, pp. 987–1007.
        """
        stat, pvalue = statsmodels.stats.diagnostic.het_arch(
            ts_residuals, nlags=max_nlags)[:2]

        if return_pval:
            return pvalue

        return stat

    @classmethod
    def ft_test_adf(cls,
                    ts: np.ndarray,
                    max_nlags: t.Optional[int] = 16,
                    return_pval: bool = True) -> float:
        """Augmented Dickey-Fuller (ADF) test.

        The ADF test works as follows:

        Null Hypothesis (H0): there is a unit root.

        Alternative Hypothesis (HA): there is not a unit root.

        A unit root test tests whether a time series variable is non-stationary
        and, therefore, possesses a unit root.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        max_nlags : int, optional (default=16)
            Maximum number of lags to include on the test.

        return_pval : bool, optional (default=True)
            If True, return the test p-value instead of the test statistic.

        Returns
        -------
        float
            If `return_pval` is False, return the ADF test statistic.
            If `return_pval` is True, return the p-value associated with the
            obtained test statistic.

        References
        ----------
        .. [1]  David A. Dickey & Wayne A. Fuller (1979) Distribution of the
            Estimators for Autoregressive Time Series with a Unit Root, Journal
            of the American Statistical Association, 74:366a, 427-431,
            DOI: 10.1080/01621459.1979.10482531
        """
        stat, pvalue = statsmodels.tsa.stattools.adfuller(ts,
                                                          maxlag=max_nlags)[:2]

        if return_pval:
            return pvalue

        return stat

    @classmethod
    def ft_test_adf_gls(cls,
                        ts: np.ndarray,
                        lags: t.Optional[int] = None,
                        max_nlags: t.Optional[int] = 16,
                        return_pval: bool = True) -> float:
        """Dickey-Fuller GLS (ADF-GLS) test.

        The ADF-GLS test works as follows:

        Null Hypothesis (H0): there is a unit root.

        Alternative Hypothesis (HA): there is not a unit root.

        A unit root test tests whether a time series variable is non-stationary
        and, therefore, possesses a unit root.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        lags : int, optional
            Number of lags included in the test.

        max_nlags : int, optional (default=16)
            Maximum number of lags tested. Used only if ``lags`` is None.

        return_pval : bool, optional (default=True)
            If True, return the test p-value instead of the test statistic.

        Returns
        -------
        float
            If `return_pval` is False, return the ADF-GLS test statistic.
            If `return_pval` is True, return the p-value associated with the
            obtained test statistic.

        References
        ----------
        .. [1] Elliott, Graham & Rothenberg, Thomas J & Stock, James H, 1996.
            "Efficient Tests for an Autoregressive Unit Root," Econometrica,
            Econometric Society, vol. 64(4), pages 813-836, July.
            https://ideas.repec.org/a/ecm/emetrp/v64y1996i4p813-36.html
        .. [2] Kevin Sheppard. (2019, March 28). bashtage/arch: Release 4.13
            (Version 4.13). Zenodo. https://zenodo.org/record/593254
        """
        res = arch.unitroot.DFGLS(ts, lags=lags, max_lags=max_nlags)

        if return_pval:
            return cls._extract_arch_module_pval(arch_result=res)

        return res.stat

    @classmethod
    def ft_test_pp(cls,
                   ts: np.ndarray,
                   max_nlags: t.Optional[int] = 16,
                   return_pval: bool = True) -> float:
        """Phillips-Perron (PP) test.

        The PP test works as follows:

        Null Hypothesis (H0): there is a unit root.

        Alternative Hypothesis (HA): there is not a unit root.

        A unit root test tests whether a time series variable is non-stationary
        and, therefore, possesses a unit root.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        max_nlags : int, optional (default=16)
            Maximum number of lags to include on the test.

        return_pval : bool, optional (default=True)
            If True, return the test p-value instead of the test statistic.

        Returns
        -------
        float
            If `return_pval` is False, return the PP test statistic.
            If `return_pval` is True, return the p-value associated with the
            obtained test statistic.

        References
        ----------
        .. [1] Phillips, P. C. B., and P. Perron. 1988. "Testing for a unit
            root in time series regression". Biometrika 75, 335-346.
        .. [2] Kevin Sheppard. (2019, March 28). bashtage/arch: Release 4.13
            (Version 4.13). Zenodo. https://zenodo.org/record/593254
        """
        res = arch.unitroot.PhillipsPerron(ts, lags=max_nlags)

        if return_pval:
            return cls._extract_arch_module_pval(arch_result=res)

        return res.stat

    @classmethod
    def ft_test_kpss(cls,
                     ts: np.ndarray,
                     max_nlags: t.Optional[int] = 16,
                     return_pval: bool = True) -> float:
        """Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test.

        The KPSS test works as follows:

        Null Hypothesis (H0): the series is weakly stationary.

        Alternative Hypothesis (HA): the series is non-stationary.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        max_nlags : int, optional (default=16)
            Maximum number of lags to be inclused on the test.

        return_pval : bool, optional (default=True)
            If True, return the test p-value instead of the test statistic.

        Returns
        -------
        float
            If `return_pval` is False, return the KPSS test statistic.
            If `return_pval` is True, return the p-value associated with the
            obtained test statistic.

        References
        ----------
        .. [1] Denis Kwiatkowski, Peter C.B. Phillips, Peter Schmidt, Yongcheol
            Shin, Testing the null hypothesis of stationarity against the
            alternative of a unit root: How sure are we that economic time
            series have a unit root?, Journal of Econometrics, Volume 54,
            Issues 1–3, 1992, Pages 159-178, ISSN 0304-4076,
            https://doi.org/10.1016/0304-4076(92)90104-Y.
        .. [2] Kevin Sheppard. (2019, March 28). bashtage/arch: Release 4.13
            (Version 4.13). Zenodo. https://zenodo.org/record/593254
        """
        res = arch.unitroot.KPSS(ts, lags=max_nlags)

        if return_pval:
            return cls._extract_arch_module_pval(arch_result=res)

        return res.stat

    @classmethod
    def ft_test_za(cls, ts: np.ndarray, return_pval: bool = True) -> float:
        """Zivot-Andrews (ZA) Test.

        The ZA test works as follows:

        Null Hypothesis (H0): the process contains a unit root with a
        single structural break.

        Alternative Hypothesis (HA): The process is trend and break
        stationary.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        return_pval : bool, optional (default=True)
            If True, return the test p-value instead of the test statistic.

        Returns
        -------
        float
            If `return_pval` is False, return the ZA test statistic.
            If `return_pval` is True, return the p-value associated with the
            obtained test statistic.

        References
        ----------
        .. [1] Zivot, E., and Andrews, D.W.K. (1992). Further evidence on the
            great crash, the oil-price shock, and the unit-root hypothesis.
            Journal of Business & Economic Studies, 10: 251-270.
        .. [2] Kevin Sheppard. (2019, March 28). bashtage/arch: Release 4.13
            (Version 4.13). Zenodo. https://zenodo.org/record/593254
        """
        res = arch.unitroot.ZivotAndrews(ts)

        if return_pval:
            return cls._extract_arch_module_pval(arch_result=res)

        return res.stat

    @classmethod
    def ft_test_lilliefors(cls,
                           ts: np.ndarray,
                           distribution: str = "norm",
                           return_pval: bool = True) -> float:
        """Lilliefors test.

        The Lilliefors test works as follows:

        Null Hypothesis (H0): the data come from some normally distributed
            population.

        Alternative Hypothesis (HA): the data does not come from a normally
            distributed population.

        Note that the `normally distributed` may be replaced by a exponential
        distribution.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        distribution : {`norm`, `exp`}, optional (default="norm")
            Distribution assumed by the Lilliefors test. Must be either
            `norm` (normal/gaussian distribution) or `exp` (exponential
            distribution).

        return_pval : bool, optional (default=True)
            If True, return the test p-value instead of the test statistic.

        Returns
        -------
        float
            If `return_pval` is False, return the Lilliefors test statistic.
            If `return_pval` is True, return the p-value associated with the
            obtained test statistic.

        References
        ----------
        .. [1] Lilliefors, H.W. (1967). On the Kolmogorov-Smirnov Test for
            Normality with Mean and Variance Unknown.
        .. [2] Hubert W. Lilliefors (1969) On the Kolmogorov-Smirnov Test for
            the Exponential Distribution with Mean Unknown, Journal of the
            American Statistical Association, 64:325, 387-389,
            DOI: 10.1080/01621459.1969.10500983
        """
        stat, pvalue = statsmodels.stats.diagnostic.lilliefors(
            ts, dist=distribution, pvalmethod="approx")

        if return_pval:
            return pvalue

        return stat
