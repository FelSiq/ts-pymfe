"""Module dedicated to autocorrelation time-series meta-features."""
import typing as t

import statsmodels.tsa.stattools
import numpy as np
import sklearn.gaussian_process

import pymfe._embed as _embed
import pymfe._utils as _utils
import pymfe._detrend as _detrend

try:
    import pymfe.stat_tests as stat_tests

except ImportError:
    pass


class MFETSAutocorr:
    """Extract time-series meta-features from Autocorr group."""
    @classmethod
    def precompute_detrended_acf(cls,
                                 ts: np.ndarray,
                                 nlags: t.Optional[int] = None,
                                 unbiased: bool = True,
                                 **kwargs) -> t.Dict[str, np.ndarray]:
        """Precompute the detrended autocorrelation function.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        nlags : int, optional
            Number of lags to calculate the autocorrelation function.

        unbiased : bool, optional (default=True)
            If True, the autocorrelation function is corrected for statistical
            bias.

        kwargs:
            Additional arguments and previous precomputed items. May
            speed up this precomputation.

        Returns
        -------
        dict
            The following precomputed item is returned:
                * ``detrended_acfs`` (:obj:`np.ndarray`): the autocorrelation
                    function from the detrended time-series.
        """
        precomp_vals = {}

        if "detrended_acfs" not in kwargs:
            precomp_vals["detrended_acfs"] = cls.ft_acf_detrended(
                ts=ts, nlags=nlags, unbiased=unbiased)

        return precomp_vals

    @classmethod
    def precompute_gaussian_model(cls,
                                  ts: np.ndarray,
                                  random_state: t.Optional[int] = None,
                                  **kwargs) -> t.Dict[str, t.Any]:
        """Precompute a gaussian process model.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        random_state : int, optional
            Random seed to optimize the gaussian process model, to keep
            the results reproducible.

        kwargs:
            Additional arguments and previous precomputed items. May
            speed up this precomputation.

        Returns
        -------
        dict
            The following precomputed item is returned:
                * ``gaussian_model`` (:obj:`GaussianProcessRegressor`):
                    Gaussian process fitted model.
                * ``gaussian_resid`` (:obj:`np.ndarray`): Gaussian process
                    model residuals (diference from the original time-series).

            The following item is necessary and, therefore, also precomputed
            if necessary:
                * ``ts_scaled`` (:obj:`np.ndarray`): standardized time-series
                    values (z-score).
        """
        precomp_vals = {}  # type: t.Dict[str, t.Any]

        ts_scaled = kwargs.get("ts_scaled")

        if ts_scaled is None:
            precomp_vals["ts_scaled"] = _utils.standardize_ts(ts=ts)
            ts_scaled = precomp_vals["ts_scaled"]

        if "gaussian_model" not in kwargs:
            gaussian_model = _utils.fit_gaussian_process(
                ts=ts, ts_scaled=ts_scaled, random_state=random_state)
            precomp_vals["gaussian_model"] = gaussian_model

        gaussian_model = kwargs.get("gaussian_model",
                                    precomp_vals["gaussian_model"])

        if "gaussian_resid" not in kwargs:
            gaussian_resid = _utils.fit_gaussian_process(
                ts=ts,
                ts_scaled=ts_scaled,
                gaussian_model=gaussian_model,
                return_residuals=True)

            precomp_vals["gaussian_resid"] = gaussian_resid

        return precomp_vals

    @classmethod
    def _calc_acf(cls,
                  ts: np.ndarray,
                  nlags: t.Optional[int] = None,
                  unbiased: bool = True,
                  detrend: bool = True,
                  detrended_acfs: t.Optional[np.ndarray] = None,
                  ts_detrended: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Precompute the autocorrelation function.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        nlags : int, optional
            Number of lags to calculate the autocorrelation function.

        unbiased : bool, optional (default=True)
            If True, the autocorrelation function is corrected for statistical
            bias.

        detrend : bool, optional (default=True)
            If True, detrend the time-series using Friedman's Super Smoother
            before calculating the autocorrelation function, or the user
            given detrended time-series from ``ts_detrended`` argument.

        detrended_acfs : :obj:`np.ndarray`, optional
            This method's return value. Used to take advantage of
            precomputations.

        ts_detrended : :obj:`np.ndarray`, optional
            Detrended time-series. Used only if `detrend` is False.

        Returns
        -------
        :obj:`np.ndarray`
            If `detrend` is True, the autocorrelation function up to `nlags`
            lags of the detrended time-series. If `detrend` is False, the
            autocorrelation function up to `nlags` lags of the time-series.
        """
        if detrended_acfs is not None and (nlags is None
                                           or detrended_acfs.size == nlags):
            return detrended_acfs

        if detrend and ts_detrended is None:
            try:
                ts_detrended = _detrend.decompose(ts=ts, ts_period=0)[2]

            except ValueError:
                pass

        if ts_detrended is None:
            ts_detrended = ts

        if nlags is None:
            nlags = ts.size // 2

        acf = statsmodels.tsa.stattools.acf(ts_detrended,
                                            nlags=nlags,
                                            unbiased=unbiased,
                                            fft=True)
        return acf[1:]

    @classmethod
    def _calc_pacf(cls,
                   ts: np.ndarray,
                   nlags: t.Optional[int] = None,
                   method: str = "ols-unbiased",
                   detrend: bool = True,
                   ts_detrended: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Precompute the partial autocorrelation function.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        nlags : int, optional
            Number of lags to calculate the partial autocorrelation function.

        method : str, optional (default="ols-unbiased")
            Method used to estimate the partial autocorrelations. Check the
            `statsmodels.tsa.stattools.pacf` documentation for the complete
            list of the available methods.

        detrend : bool, optional (default=True)
            If True, detrend the time-series using Friedman's Super Smoother
            before calculating the autocorrelation function, or the user
            given detrended time-series from ``ts_detrended`` argument.

        ts_detrended : :obj:`np.ndarray`, optional
            Detrended time-series. Used only if `detrend` is False. If not
            given, the time-series is detrended within this method using
            Friedman's Super Smoother.

        Returns
        -------
        :obj:`np.ndarray`
            If `detrend` is True, the partial autocorrelation function up to
            `nlags` lags of the detrended time-series. If `detrend` is False,
            the autocorrelation function up to `nlags` lags of the time-series.
        """
        if nlags is None:
            nlags = 1 + ts.size // 10

        if detrend and ts_detrended is None:
            try:
                ts_detrended = _detrend.decompose(ts=ts, ts_period=0)[2]

            except ValueError:
                pass

        if ts_detrended is None:
            ts_detrended = ts

        pacf = statsmodels.tsa.stattools.pacf(ts_detrended,
                                              nlags=nlags,
                                              method=method)
        return pacf[1:]

    @classmethod
    def _first_acf_below_threshold(
            cls,
            ts: np.ndarray,
            threshold: float,
            abs_acf_vals: bool = False,
            max_nlags: t.Optional[int] = None,
            unbiased: bool = True,
            detrended_acfs: t.Optional[np.ndarray] = None,
    ) -> t.Union[int, float]:
        """First autocorrelation lag below a given threshold.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        threshold : float
            The threshold to find the first lag below it.

        abs_acf_vals : bool, optional (default=False)
            If True, avaliate the aboslute value of the autocorrelation
            function.

        max_nlags : int, optional
            Number of lags to avaluate the autocorrelation function.

        unbiased : bool, optional (default=True)
            If True, the autocorrelation function is corrected for statistical
            bias.

        detrended_acfs : :obj:`np.ndarray`, optional
            This method's return value. Used to take advantage of
            precomputations.

        Returns
        -------
        int or float
            Lag corresponding to the first autocorrelation function the
            given ``threshold``, if any. Return `np.nan` if no such index is
            found.
        """
        detrended_acfs = cls._calc_acf(ts=ts,
                                       nlags=max_nlags,
                                       unbiased=unbiased,
                                       detrended_acfs=detrended_acfs)

        if abs_acf_vals:
            # Note: in this case, we are testing if
            # -threshold <= acf <= threshold.
            detrended_acfs = np.abs(detrended_acfs)

        nonpos_acfs = np.flatnonzero(detrended_acfs <= threshold)

        try:
            return nonpos_acfs[0] + 1

        except IndexError:
            return np.nan

    @classmethod
    def ft_acf(cls,
               ts: np.ndarray,
               nlags: t.Optional[int] = None,
               unbiased: bool = True) -> np.ndarray:
        """Autocorrelation function of the time-series.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        nlags : int, optional
            Number of lags to calculate the autocorrelation function.

        unbiased : bool, optional (default=True)
            If True, the autocorrelation function is corrected for statistical
            bias.

        Returns
        -------
        :obj:`np.ndarray`
            The autocorrelation function up to `nlags` lags of the time-series.
        """
        return cls._calc_acf(ts=ts,
                             nlags=nlags,
                             unbiased=unbiased,
                             detrend=False)

    @classmethod
    def ft_acf_detrended(
            cls,
            ts: np.ndarray,
            nlags: t.Optional[int] = None,
            unbiased: bool = True,
            ts_detrended: t.Optional[np.ndarray] = None,
            detrended_acfs: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Autocorrelation function of the detrended time-series.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        nlags : int, optional
            Number of lags to calculate the autocorrelation function.

        unbiased : bool, optional (default=True)
            If True, the autocorrelation function is corrected for statistical
            bias.

        ts_detrended : :obj:`np.ndarray`, optional
            Detrended time-series. If not given, the time-series is detrended
            within this method using Friedman's Super Smoother.

        detrended_acfs : :obj:`np.ndarray`, optional
            This method's return value. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            The autocorrelation function up to `nlags` lags of the detrended
            time-series.
        """
        return cls._calc_acf(ts=ts,
                             nlags=nlags,
                             unbiased=unbiased,
                             detrend=True,
                             detrended_acfs=detrended_acfs,
                             ts_detrended=ts_detrended)

    @classmethod
    def ft_acf_diff(cls,
                    ts: np.ndarray,
                    num_diff: int = 1,
                    nlags: t.Optional[int] = None,
                    detrend: bool = True,
                    ts_detrended: t.Optional[np.ndarray] = None,
                    unbiased: bool = True) -> np.ndarray:
        """Autocorrelation function of the differenced time-series.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        num_diff : int, optional (default=1)
            Order of differentiation.

        nlags : int, optional
            Number of lags to calculate the autocorrelation function.

        detrend : bool, optional (default=True)
            If True, detrend the time-series using Friedman's Super Smoother
            before calculating the autocorrelation function, or the user
            given detrended time-series from ``ts_detrended`` argument.

        ts_detrended : :obj:`np.ndarray`, optional
            Detrended time-series. If not given and ``detrend`` is True, the
            time-series is detrended within this method using Friedman's Super
            Smoother.

        unbiased : bool, optional (default=True)
            If True, the autocorrelation function is corrected for statistical
            bias.

        Returns
        -------
        :obj:`np.ndarray`
            The autocorrelation function up to `nlags` lags of the differenced
            time-series.
        """
        return cls._calc_acf(ts=np.diff(ts, n=num_diff),
                             detrend=detrend,
                             nlags=nlags,
                             unbiased=unbiased,
                             ts_detrended=ts_detrended)

    @classmethod
    def ft_pacf(cls,
                ts: np.ndarray,
                nlags: t.Optional[int] = None,
                method: str = "ols-unbiased") -> np.ndarray:
        """Partial autocorrelation function of the time-series.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        nlags : int, optional
            Number of lags to calculate the partial autocorrelation function.

        method : str, optional (default="ols-unbiased")
            Method used to estimate the partial autocorrelations. Check the
            `statsmodels.tsa.stattools.pacf` documentation for the complete
            list of the available methods.

        Returns
        -------
        :obj:`np.ndarray`
            The autocorrelation function up to `nlags` lags of the time-series.
        """
        return cls._calc_pacf(ts=ts, nlags=nlags, method=method, detrend=False)

    @classmethod
    def ft_pacf_detrended(
            cls,
            ts: np.ndarray,
            nlags: t.Optional[int] = None,
            method: str = "ols-unbiased",
            ts_detrended: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Partial autocorrelation function of the detrended time-series.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        nlags : int, optional
            Number of lags to calculate the partial autocorrelation function.

        method : str, optional (default="ols-unbiased")
            Method used to estimate the partial autocorrelations. Check the
            `statsmodels.tsa.stattools.pacf` documentation for the complete
            list of the available methods.

        ts_detrended : :obj:`np.ndarray`, optional
            Detrended time-series. If not given, the time-series is detrended
            within this method using Friedman's Super Smoother.

        Returns
        -------
        :obj:`np.ndarray`
            The partial autocorrelation function up to `nlags` lags of the
            detrended time-series.
        """
        return cls._calc_pacf(ts=ts,
                              nlags=nlags,
                              method=method,
                              detrend=True,
                              ts_detrended=ts_detrended)

    @classmethod
    def ft_pacf_diff(
            cls,
            ts: np.ndarray,
            num_diff: int = 1,
            nlags: t.Optional[int] = None,
            method: str = "ols-unbiased",
            detrend: bool = True,
            ts_detrended: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Partial autocorrelation function of the differenced time-series.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        nlags : int, optional
            Number of lags to calculate the partial autocorrelation function.

        method : str, optional (default="ols-unbiased")
            Method used to estimate the partial autocorrelations. Check the
            `statsmodels.tsa.stattools.pacf` documentation for the complete
            list of the available methods.

        detrend : bool, optional (default=True)
            If True, detrend the time-series using Friedman's Super Smoother
            before calculating the autocorrelation function, or the user
            given detrended time-series from ``ts_detrended`` argument.

        ts_detrended : :obj:`np.ndarray`, optional
            Detrended time-series. Used only if `detrend` is False. If not
            given, the time-series is detrended within this method using
            Friedman's Super Smoother.

        Returns
        -------
        :obj:`np.ndarray`
            If `detrend` is True, the partial autocorrelation function up to
            `nlags` lags of the detrended time-series. If `detrend` is False,
            the autocorrelation function up to `nlags` lags of the time-series.
        """
        return cls._calc_pacf(ts=np.diff(ts, n=num_diff),
                              nlags=nlags,
                              method=method,
                              detrend=detrend,
                              ts_detrended=ts_detrended)

    @classmethod
    def ft_acf_first_nonsig(
            cls,
            ts: np.ndarray,
            max_nlags: t.Optional[int] = None,
            unbiased: bool = True,
            threshold: t.Optional[t.Union[int, float]] = None,
            detrended_acfs: t.Optional[np.ndarray] = None,
    ) -> t.Union[int, float]:
        """First non-significative detrended autocorrelation lag.

        The critical value to determine if a autocorrelation is significative
        is 1.96 / sqrt(len(ts)), but can be changed using the ``threshold``
        parameter.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        max_nlags : int, optional
            Number of lags to avaluate the autocorrelation function.

        unbiased : bool, optional (default=True)
            If True, the autocorrelation function is corrected for statistical
            bias.

        threshold : int or float, default
            The critical value to determine if a autocorrelation value is
            significative or not. This means that any autocorrelation with
            absolute value higher than is considered significative. If None,
            then the threshold used will be 1.96 / sqrt(len(ts)).

        ts_detrended : :obj:`np.ndarray`, optional
            Detrended time-series. Used only if `detrend` is False. If not
            given, the time-series is detrended within this method using
            Friedman's Super Smoother.

        Returns
        -------
        int or float
            Lag corresponding to the first autocorrelation with absolute value
            below the given ``threshold``, if any. Return `np.nan` if no such
            index is found.
        """
        if threshold is None:
            threshold = 1.96 / np.sqrt(ts.size)

        res = cls._first_acf_below_threshold(ts=ts,
                                             threshold=threshold,
                                             abs_acf_vals=True,
                                             max_nlags=max_nlags,
                                             unbiased=unbiased,
                                             detrended_acfs=detrended_acfs)
        return res

    @classmethod
    def ft_acf_first_nonpos(
            cls,
            ts: np.ndarray,
            max_nlags: t.Optional[int] = None,
            unbiased: bool = True,
            detrended_acfs: t.Optional[np.ndarray] = None,
    ) -> t.Union[int, float]:
        """First non-positive detrended autocorrelation lag.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        max_nlags : int, optional
            Number of lags to avaluate the autocorrelation function.

        unbiased : bool, optional (default=True)
            If True, the autocorrelation function is corrected for statistical
            bias.

        detrended_acfs : :obj:`np.ndarray`, optional
            Detrended time-series autocorrelation function with each index
            corresponding to its lag starting from the lag 1.

        Returns
        -------
        int or float
            Lag corresponding to the first autocorrelation below or equal
            zero, if any. Return `np.nan` if no such index is found.
        """
        res = cls._first_acf_below_threshold(ts=ts,
                                             threshold=0,
                                             abs_acf_vals=False,
                                             max_nlags=max_nlags,
                                             unbiased=unbiased,
                                             detrended_acfs=detrended_acfs)
        return res

    @classmethod
    def ft_first_acf_locmin(
            cls,
            ts: np.ndarray,
            max_nlags: t.Optional[int] = None,
            unbiased: bool = True,
            detrended_acfs: t.Optional[np.ndarray] = None,
    ) -> t.Union[int, float]:
        """First local minima detrended autocorrelation lag.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        max_nlags : int, optional
            Number of lags to avaluate the autocorrelation function.

        unbiased : bool, optional (default=True)
            If True, the autocorrelation function is corrected for statistical
            bias.

        detrended_acfs : :obj:`np.ndarray`, optional
            Detrended time-series autocorrelation function with each index
            corresponding to its lag starting from the lag 1.

        Returns
        -------
        int or float
            Lag corresponding to the first autocorrelation below or equal
            zero, if any. Return `np.nan` if no such index is found.
        """
        detrended_acfs = cls._calc_acf(ts=ts,
                                       nlags=max_nlags,
                                       unbiased=unbiased,
                                       detrended_acfs=detrended_acfs)

        acfs_locmin = np.flatnonzero(
            _utils.find_crit_pt(detrended_acfs, type_="min"))

        try:
            return acfs_locmin[0] + 1

        except IndexError:
            return np.nan

    @classmethod
    def ft_trev(cls,
                ts: np.ndarray,
                lag: t.Optional[t.Union[str, int]] = None,
                only_numerator: bool = False,
                max_nlags: t.Optional[int] = None,
                detrended_acfs: t.Optional[np.ndarray] = None,
                detrended_ami: t.Optional[np.ndarray] = None) -> float:
        """Normalized nonlinear autocorrelation Trev statistic.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        lag : int or str, optional
            Lag to calculate the statistic. It must be a strictly positive
            value, None or a string in {`acf`, `acf-nonsig`, `ami`}. In the
            last two type of options, the lag is estimated within this method
            using the given strategy method (or, if None, it is used the
            strategy `acf-nonsig` by default) up to ``max_nlags``.
                1. `acf`: the lag corresponds to the first non-positive value
                    in the autocorrelation function.
                2. `acf-nonsig`: lag corresponds to the first non-significant
                    value in the autocorrelation function (absolute value below
                    the critical value of 1.96 / sqrt(ts.size)).
                3. `ami`: lag corresponds to the first local minimum of the
                    time-series automutual information function.

        only_numerator : bool, optional (default=False)
            If True, return only the numerator from this statistic definition.
            Check `autocorr.MFETSAutocorr.ft_trev` documentation for more
            information.

        max_nlags : int, optional
            If ``lag`` is not a numeric value, than it will be estimated using
            either the time-series autocorrelation or mutual information
            function estimated up to this argument value.

        detrended_acfs : :obj:`np.ndarray`, optional
            Array of time-series autocorrelation function (for distinct ordered
            lags) of the detrended time-series. Used only if ``lag`` is any of
            `acf`, `acf-nonsig` or None.  If this argument is not given and the
            previous condiditon is meet, the autocorrelation function will be
            calculated inside this method up to ``max_nlags``.

        detrended_ami : :obj:`np.ndarray`, optional
            Array of time-series automutual information function (for distinct
            ordered lags). Used only if ``lag`` is `ami`. If not given and the
            previous condiditon is meet, the automutual information function
            will be calculated inside this method up to ``max_nlags``.

        Returns
        ------
        float
            Trev statistic.

        References
        ----------
        .. [1] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework
            for Automated Time-Series Phenotyping Using Massive Feature
            Extraction, Cell Systems 5: 527 (2017).
            DOI: 10.1016/j.cels.2017.10.001
        .. [2] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
            time-series analysis: the empirical structure of time series and
            their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013).
        """
        _lag = _embed.embed_lag(ts=ts,
                                lag=lag,
                                max_nlags=max_nlags,
                                detrended_acfs=detrended_acfs,
                                detrended_ami=detrended_ami)

        diff = ts[_lag:] - ts[:-_lag]

        numen = np.mean(np.power(diff, 3))

        if only_numerator:
            return numen

        denom = np.power(np.mean(np.square(diff)), 1.5)
        trev = numen / denom

        return trev

    @classmethod
    def ft_tc3(cls,
               ts: np.ndarray,
               lag: t.Optional[t.Union[str, int]] = None,
               only_numerator: bool = False,
               max_nlags: t.Optional[int] = None,
               detrended_acfs: t.Optional[np.ndarray] = None,
               detrended_ami: t.Optional[np.ndarray] = None) -> float:
        """Normalized nonlinear autocorrelation Tc3 statistic.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        lag : int or str, optional
            Lag to calculate the statistic. It must be a strictly positive
            value, None or a string in {`acf`, `acf-nonsig`, `ami`}. In the
            last two type of options, the lag is estimated within this method
            using the given strategy method (or, if None, it is used the
            strategy `acf-nonsig` by default) up to ``max_nlags``.
                1. `acf`: the lag corresponds to the first non-positive value
                    in the autocorrelation function.
                2. `acf-nonsig`: lag corresponds to the first non-significant
                    value in the autocorrelation function (absolute value below
                    the critical value of 1.96 / sqrt(ts.size)).
                3. `ami`: lag corresponds to the first local minimum of the
                    time-series automutual information function.

        only_numerator : bool, optional (default=False)
            If True, return only the numerator from this statistic definition.
            Check `autocorr.MFETSAutocorr.ft_tc3` documentation for more
            information.

        max_nlags : int, optional
            If ``lag`` is not a numeric value, than it will be estimated using
            either the time-series autocorrelation or mutual information
            function estimated up to this argument value.

        detrended_acfs : :obj:`np.ndarray`, optional
            Array of time-series autocorrelation function (for distinct ordered
            lags) of the detrended time-series. Used only if ``lag`` is any of
            `acf`, `acf-nonsig` or None.  If this argument is not given and the
            previous condiditon is meet, the autocorrelation function will be
            calculated inside this method up to ``max_nlags``.

        detrended_ami : :obj:`np.ndarray`, optional
            Array of time-series automutual information function (for distinct
            ordered lags). Used only if ``lag`` is `ami`. If not given and the
            previous condiditon is meet, the automutual information function
            will be calculated inside this method up to ``max_nlags``.

        Returns
        ------
        float
            Tc3 statistic.

        References
        ----------
        .. [1] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework
            for Automated Time-Series Phenotyping Using Massive Feature
            Extraction, Cell Systems 5: 527 (2017).
            DOI: 10.1016/j.cels.2017.10.001
        .. [2] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
            time-series analysis: the empirical structure of time series and
            their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013).
        """
        _lag = _embed.embed_lag(ts=ts,
                                lag=lag,
                                max_nlags=max_nlags,
                                detrended_acfs=detrended_acfs,
                                detrended_ami=detrended_ami)

        ts_shift_1 = ts[:-2 * _lag]
        ts_shift_2 = ts[_lag:-_lag]
        ts_shift_3 = ts[2 * _lag:]

        _aux = ts_shift_1 * ts_shift_2
        numen = np.mean(_aux * ts_shift_3)

        if only_numerator:
            return numen

        denom = np.abs(np.mean(_aux))**1.5

        tc3 = numen / denom

        return tc3

    @classmethod
    def ft_gen_autocorr(cls,
                        ts: np.ndarray,
                        alpha: float = 1,
                        beta: float = 1,
                        lag: t.Optional[t.Union[str, int]] = None,
                        max_nlags: t.Optional[int] = None,
                        detrended_acfs: t.Optional[np.ndarray] = None,
                        detrended_ami: t.Optional[np.ndarray] = None) -> float:
        """Generalized autocorrelation of the time-series.

        If alpha = beta, estimates how values of the same order of magnitude
        are related in time. Otherwise, estimates correlations between
        different magnitudes of the time series.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        alpha : float, optional (default=1)
            Non-zero parameter.

        beta : float, optional (default=1)
            Non-zero parameter.

        lag : int or str, optional
            Lag to calculate the statistic. It must be a strictly positive
            value, None or a string in {`acf`, `acf-nonsig`, `ami`}. In the
            last two type of options, the lag is estimated within this method
            using the given strategy method (or, if None, it is used the
            strategy `acf-nonsig` by default) up to ``max_nlags``.
                1. `acf`: the lag corresponds to the first non-positive value
                    in the autocorrelation function.
                2. `acf-nonsig`: lag corresponds to the first non-significant
                    value in the autocorrelation function (absolute value below
                    the critical value of 1.96 / sqrt(ts.size)).
                3. `ami`: lag corresponds to the first local minimum of the
                    time-series automutual information function.

        max_nlags : int, optional
            If ``lag`` is not a numeric value, than it will be estimated using
            either the time-series autocorrelation or mutual information
            function estimated up to this argument value.

        detrended_acfs : :obj:`np.ndarray`, optional
            Array of time-series autocorrelation function (for distinct ordered
            lags) of the detrended time-series. Used only if ``lag`` is any of
            `acf`, `acf-nonsig` or None.  If this argument is not given and the
            previous condiditon is meet, the autocorrelation function will be
            calculated inside this method up to ``max_nlags``.

        detrended_ami : :obj:`np.ndarray`, optional
            Array of time-series automutual information function (for distinct
            ordered lags). Used only if ``lag`` is `ami`. If not given and the
            previous condiditon is meet, the automutual information function
            will be calculated inside this method up to ``max_nlags``.

        Returns
        -------
        float
            Generalized autocorrelation of the time-series.

        References
        ----------
        .. [1] S.M. Duarte Queirós, L.G. Moyano, Yet on statistical properties
            of traded volume: Correlation and mutual information at different
            value magnitudes, Physica A: Statistical Mechanics and its
            Applications, Volume 383, Issue 1, 2007, Pages 10-15, ISSN
            0378-4371, https://doi.org/10.1016/j.physa.2007.04.082.
        .. [2] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework
            for Automated Time-Series Phenotyping Using Massive Feature
            Extraction, Cell Systems 5: 527 (2017).
            DOI: 10.1016/j.cels.2017.10.001
        .. [3] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
            time-series analysis: the empirical structure of time series and
            their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013).
            DOI: 10.1098/rsif.2013.0048
        """
        if np.isclose(alpha, 0.0):
            raise ValueError("'alpha' parameter must be nonzero (got {})."
                             "".format(alpha))

        if np.isclose(beta, 0.0):
            raise ValueError("'beta' parameter must be nonzero (got {})."
                             "".format(beta))

        _lag = _embed.embed_lag(ts=ts,
                                lag=lag,
                                max_nlags=max_nlags,
                                detrended_acfs=detrended_acfs,
                                detrended_ami=detrended_ami)

        ts_abs = np.abs(ts)
        ts_sft_1 = ts_abs[:-_lag]
        ts_sft_2 = ts_abs[_lag:]

        ts_sft_1_a = ts_sft_1**alpha
        ts_sft_2_b = ts_sft_2**beta

        ts_sft_1_a_mean = np.mean(ts_sft_1_a)
        ts_sft_2_b_mean = np.mean(ts_sft_2_b)

        gen_autocorr = (
            np.mean(ts_sft_1_a * ts_sft_2_b) -
            ts_sft_1_a_mean * ts_sft_2_b_mean /
            (np.sqrt(np.mean(ts_sft_1**(2 * alpha)) - ts_sft_1_a_mean**2) *
             np.sqrt(np.mean(ts_sft_2**(2 * beta)) - ts_sft_2_b_mean**2)))

        return gen_autocorr

    @classmethod
    def ft_autocorr_crit_pt(
            cls,
            ts: np.ndarray,
            crit_point_type: str = "non-plateau",
            return_lags: bool = True,
            max_nlags: t.Optional[int] = None,
            unbiased: bool = True,
            detrended_acfs: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Lags corresponding to minima or maxima of autocorrelation function.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        crit_point_type : str, optional (default="non-plateau")
            Critical point type. Must be a value in {`non-plateau`, `plateau`,
            `min`, `max`, `any`}.

        return_lags : bool, optional (default=True)
            If True, return the lags corresponding to the autocorrelation
            function critical points. If False, return a binary array marking
            with `1` the positions corresponding to the critical points, and
            `0` otherwise.

        max_nlags : int, optional
            Number of lags to avaluate the autocorrelation function.

        unbiased : bool, optional (default=True)
            If True, the autocorrelation function is corrected for statistical
            bias.

        detrended_acfs : :obj:`np.ndarray`, optional
            Detrended time-series autocorrelation function with each index
            corresponding to its lag starting from the lag 1.

        Returns
        -------
        :obj:`np.ndarray`
            If `return_lags` is True, return the lags corresponding to the
            autocorrelation function critical points. If `return_lags` is
            False, return a binary array marking with `1` the lag indices
            (starting from lag 1) corresponding to the autocorrelation function
            critical points.

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
        detrended_acfs = cls._calc_acf(ts=ts,
                                       nlags=max_nlags,
                                       unbiased=unbiased,
                                       detrended_acfs=detrended_acfs)

        ac_shape = _utils.find_crit_pt(arr=detrended_acfs,
                                       type_=crit_point_type)

        # Note: in 'hctsa', either the sum or the mean is returned.
        # However, to enable summarization, here we return the whole
        # array.

        if return_lags:
            return np.flatnonzero(ac_shape)

        return ac_shape.astype(int)

    @classmethod
    def ft_gresid_autocorr(
            cls,
            ts: np.ndarray,
            nlags: int = 8,
            unbiased: bool = True,
            random_state: t.Optional[int] = None,
            ts_scaled: t.Optional[np.ndarray] = None,
            gaussian_resid: t.Optional[np.ndarray] = None,
            gaussian_model: t.Optional[
                sklearn.gaussian_process.GaussianProcessRegressor] = None,
    ) -> np.ndarray:
        """Autocorrelation function of the gaussian process model residuals.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        nlags : int, optional (default=8)
            Number of lags evaluated in the autocorrelation function.

        unbiased : bool, optional (default=True)
            If True, the autocorrelation function is corrected for statistical
            bias.

        random_state : int, optional
            Random seed to optimize the gaussian process model, to keep
            the results reproducible.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations. Used only if ``gaussian_resid`` is None.

        gaussian_resid : :obj:`np.ndarray`, optional
            Residuals of a gaussian process. Used to take advantage of
            precomputations.

        gaussian_model : :obj:`GaussianProcessRegressor`, optional
            A fitted model of a gaussian process. Used to take advantage of
            precomputations. Used only if ``gaussian_resid`` is None.

        Returns
        -------
        :obj:`np.ndarray`
            Autocorrelation function of the gaussian process residuals up
            to ``nlags``.

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
        if gaussian_resid is None:
            gaussian_resid = _utils.fit_gaussian_process(
                ts=ts,
                ts_scaled=ts_scaled,
                random_state=random_state,
                gaussian_model=gaussian_model,
                return_residuals=True)

        gaussian_resid_acf = cls._calc_acf(ts=gaussian_resid,
                                           nlags=nlags,
                                           unbiased=unbiased)

        return gaussian_resid_acf

    @classmethod
    def ft_gresid_lbtest(
            cls,
            ts: np.ndarray,
            nlags: int = 8,
            return_pval: bool = True,
            random_state: t.Optional[int] = None,
            ts_scaled: t.Optional[np.ndarray] = None,
            gaussian_resid: t.Optional[np.ndarray] = None,
            gaussian_model: t.Optional[
                sklearn.gaussian_process.GaussianProcessRegressor] = None,
    ) -> np.ndarray:
        """Ljung–Box test in the residuals of a gaussian process model.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        nlags : int, optional (default=8)
            Number of lags evaluated in the Ljung-Box test.

        return_pval : bool, optional (default=True)
            If True, return the p-value of the test instead of the test
            statistic.

        random_state : int, optional
            Random seed to optimize the gaussian process model, to keep
            the results reproducible.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations. Used only if ``gaussian_resid`` is None.

        gaussian_resid : :obj:`np.ndarray`, optional
            Residuals of a gaussian process. Used to take advantage of
            precomputations.

        gaussian_model : :obj:`GaussianProcessRegressor`, optional
            A fitted model of a gaussian process. Used to take advantage of
            precomputations. Used only if ``gaussian_resid`` is None.

        Returns
        -------
        :obj:`np.ndarray`
            If `return_pval` is False, Ljung-Box test statistic for each lag
            of the gaussian process residuals.
            If `return_pval` is True, p-value associated with the Ljung-Box
            test statistic for each lag of the gaussian process residuals.

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
        if gaussian_resid is None:
            gaussian_resid = _utils.fit_gaussian_process(
                ts=ts,
                ts_scaled=ts_scaled,
                random_state=random_state,
                gaussian_model=gaussian_model,
                return_residuals=True)

        gaussian_lb_test = stat_tests.MFETSStatTests.ft_test_lb(
            ts_residuals=gaussian_resid,
            max_nlags=nlags,
            return_pval=return_pval)

        return gaussian_lb_test

    @classmethod
    def ft_autocorr_out_dist(
            cls,
            ts: np.ndarray,
            p: float = 0.8,
            max_nlags: t.Optional[int] = None,
            unbiased: bool = True,
            detrended_acfs: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Distance between the autocorrelation with and without outliers.

        This method calculates the time-series autocorrelation function
        for all observations, and the aucorrelation function of the
        time-series without a subset of the most extreme values (cut at
        the ``p`` quantile of all absolute values). It is returned the
        absolute difference between these two autocorrelations.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        p : float, optional (default=0.8)
            Quantile of cut in the set of the time-series absolute values to
            determine which instances are considered outliers.

        max_nlags : int, optional
            Number of lags to avaluate the autocorrelation function.

        unbiased : bool, optional (default=True)
            If True, the autocorrelation function is corrected for statistical
            bias.

        detrended_acfs : :obj:`np.ndarray`, optional
            Detrended time-series autocorrelation function with each index
            corresponding to its lag starting from the lag 1.

        Returns
        -------
        :obj:`np.ndarray`
            Absolute difference element-wise between each autocorrelation
            with and without outliers.

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
        detrended_acfs = cls._calc_acf(ts=ts,
                                       nlags=max_nlags,
                                       unbiased=unbiased,
                                       detrended_acfs=detrended_acfs)

        ts_abs = np.abs(ts)
        ts_inliners = ts[ts_abs <= np.quantile(ts_abs, p)]

        ts_inliners_acfs = cls._calc_acf(ts=ts_inliners,
                                         nlags=max_nlags,
                                         unbiased=unbiased)

        dist_acfs = np.abs(detrended_acfs[:ts_inliners_acfs.size] -
                           ts_inliners_acfs)

        return dist_acfs
