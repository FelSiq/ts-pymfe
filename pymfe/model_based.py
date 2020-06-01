"""Module dedicated to model-based time-series meta-features."""
import typing as t
import warnings

import sklearn.gaussian_process
import sklearn.preprocessing
import statsmodels.tsa.holtwinters
import statsmodels.tsa.arima_model
import statsmodels.regression
import statsmodels.tools
import numpy as np

import pymfe._utils as _utils
import pymfe._orthopoly as _orthopoly
import pymfe._period as _period


class MFETSModelBased:
    """Extract time-series meta-features from Model-Based group."""
    @classmethod
    def precompute_ts_scaled(cls, ts: np.ndarray,
                             **kwargs) -> t.Dict[str, np.ndarray]:
        """Precompute a standardized time series.

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
                * ``ts_scaled`` (:obj:`np.ndarray`): standardized time-series
                    values (z-score).
        """
        precomp_vals = {}  # type: t.Dict[str, np.ndarray]

        if "ts_scaled" not in kwargs:
            precomp_vals["ts_scaled"] = _utils.standardize_ts(ts=ts)

        return precomp_vals

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
    def precompute_model_ets(
            cls,
            ts: np.ndarray,
            damped: bool = False,
            ts_period: t.Optional[int] = None,
            **kwargs
    ) -> t.Dict[str, t.Any]:
        """Precompute a standardized time series.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        damped : bool, optional (default=False)
            If True, the exponential smoothing models will be have damping
            effects.

        ts_period : int, optional
            Time-series period. Used to take advantage of precomputations.

        kwargs:
            Additional arguments and previous precomputed items. May
            speed up this precomputation.

        Returns
        -------
        dict
            The following precomputed item is returned:
                * ``res_model_des`` (:obj:`HoltWintersResultsWrapper`):
                    Double exponential smoothing model (exponential smoothing
                    model without the seasonal component) with additive trend
                    results.
                * ``res_model_ets`` (:obj:`HoltWintersResultsWrapper`):
                    Triple exponential smoothing model (exponential smoothing
                    model with the seasonal component) with additive components
                    results.

            The following items are necessary and, therefore, also precomputed
            if necessary:
                * ``ts_scaled`` (:obj:`np.ndarray`): standardized time-series
                    values (z-score).
                * ``ts_period`` (:obj:`int`): time-series period.

        References
        ----------
        .. [1] Winters, Peter R. Forecasting Sales by Exponentially Weighted
            Moving Averages, 1960, INFORMS, Linthicum, MD, USA
            https://doi.org/10.1287/mnsc.6.3.324
        .. [2] Charles C. Holt, Forecasting seasonals and trends by
            exponentially weighted moving averages, International Journal of
            Forecasting, Volume 20, Issue 1, 2004, Pages 5-10, ISSN 0169-2070,
            https://doi.org/10.1016/j.ijforecast.2003.09.015.
        """
        precomp_vals = {}  # type: t.Dict[str, t.Any]

        ts_scaled = kwargs.get("ts_scaled")

        if ts_scaled is None:
            precomp_vals.update(cls.precompute_ts_scaled(ts=ts))
            ts_scaled = precomp_vals["ts_scaled"]

        if "res_model_des" not in kwargs:
            model = cls._fit_res_model_des(ts=ts_scaled, damped=damped)
            precomp_vals["res_model_des"] = model

        if ts_period is None:
            precomp_vals.update(cls.precompute_period(ts=ts))
            ts_period = precomp_vals["ts_period"]

        if "res_model_ets" not in kwargs:
            model = cls._fit_res_model_ets(ts=ts_scaled,
                                           ts_period=ts_period,
                                           damped=damped)
            precomp_vals["res_model_ets"] = model

        return precomp_vals

    @classmethod
    def precompute_ioe_std_linear_model(cls,
                                        ts: np.ndarray,
                                        step_size: float = 0.05,
                                        **kwargs) -> t.Dict[str, t.Any]:
        """Precompute linear model of std with iterative outlier exclusion.

        In the iterative outlier exclusion, a uniformly spaced set of
        thresholds over the time-series range is build in increasing order.
        For each threshold it is calculated a statistic of the timestamp
        values of instances larger or equal than the current threshold.

        This precomputed linear model is the standard deviation of the
        timestamps regressed on the thresholds of each iteration.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        step_size : float, optional (default=0.05)
            Increase of the outlier threshold in each iteration. Must be a
            number strictly positive.

        kwargs:
            Additional arguments and previous precomputed items. May
            speed up this precomputation.

        Returns
        -------
        dict
            The following precomputed item is returned:
                * ``res_ioe_std_linreg`` (:obj:`RegressionResults`): Results
                    from the linear regression model of iterative outlier
                    exclusion timestamps standard deviation regression on the
                    thresholds.

            The following item is necessary and, therefore, also precomputed
            if necessary:
                * ``ts_scaled`` (:obj:`np.ndarray`): standardized time-series
                    values (z-score).

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
        precomp_vals = {}  # type: t.Dict[str, t.Any]

        ts_scaled = kwargs.get("ts_scaled")

        if ts_scaled is None:
            precomp_vals.update(cls.precompute_ts_scaled(ts=ts))
            ts_scaled = precomp_vals["ts_scaled"]

        if "res_ioe_std_linreg" not in kwargs:
            lin_reg_res = cls._fit_ioe_std_lin_model(ts=ts,
                                                     ts_scaled=ts_scaled,
                                                     step_size=step_size)
            precomp_vals["res_ioe_std_linreg"] = lin_reg_res

        return precomp_vals

    @staticmethod
    def _fit_ioe_std_lin_model(
            ts: np.ndarray,
            step_size: float = 0.05,
            ts_scaled: t.Optional[np.ndarray] = None,
    ) -> statsmodels.regression.linear_model.RegressionResults:
        """Fit a linear model of IOE standard deviations onto thresholds.

        In the iterative outlier exclusion, a uniformly spaced set of
        thresholds over the time-series range is build in increasing order.
        For each threshold it is calculated a statistic of the timestamp
        values of instances larger or equal than the current threshold.

        This precomputed linear model is the standard deviation of the
        timestamps regressed on the thresholds of each iteration.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        step_size : float, optional (default=0.05)
            Increase of the outlier threshold in each iteration. Must be a
            number strictly positive.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`statsmodels.regression.linear_model.RegressionResults`
            Results from the linear regression model of iterative outlier
            exclusion timestamps standard deviation regression on the
            thresholds.

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
        def ioe_std_func(arr: np.ndarray) -> float:
            """Normalized standard deviation with ddof=1."""
            return np.std(arr, ddof=1) / np.sqrt(arr.size)

        ioe_std = _utils.calc_ioe_stats(ts=ts,
                                        funcs=[ioe_std_func],
                                        ts_scaled=ts_scaled,
                                        step_size=step_size)

        thresholds = statsmodels.tools.add_constant(
            np.arange(ioe_std.size) * step_size)

        lin_reg_res = statsmodels.regression.linear_model.OLS(
            ioe_std, thresholds).fit()

        return lin_reg_res

    @staticmethod
    def _fit_res_model_des(
            ts: np.ndarray,
            damped: bool = False,
            ts_scaled: t.Optional[np.ndarray] = None,
    ) -> statsmodels.tsa.holtwinters.HoltWintersResultsWrapper:
        """Fit a double exponential smoothing model with additive trend.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        damped : bool, optional (default=False)
            Whether or not the exponential smoothing model should include a
            damping component.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`statsmodels.tsa.holtwinters.HoltWintersResultsWrapper`
            Results of a optimized double exponential smoothing model.

        References
        ----------
        .. [1] Holt, C. E. (1957). Forecasting seasonals and trends by
            exponentially weighted averages (O.N.R. Memorandum No. 52).
            Carnegie Institute of Technology, Pittsburgh USA.
            https://doi.org/10.1016/j.ijforecast.2003.09.015
        """
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                module="statsmodels",
                category=statsmodels.tools.sm_exceptions.ConvergenceWarning)

            model = statsmodels.tsa.holtwinters.ExponentialSmoothing(
                endog=ts_scaled, trend="additive", damped=damped,
                seasonal=None).fit()

        return model

    @staticmethod
    def _fit_res_model_ets(
            ts: np.ndarray,
            damped: bool = False,
            grid_search_guess: bool = True,
            ts_period: t.Optional[int] = None,
            ts_scaled: t.Optional[np.ndarray] = None,
    ) -> statsmodels.tsa.holtwinters.HoltWintersResultsWrapper:
        """Fit a triple exponential smoothing model with additive components.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        damped : bool, optional (default=False)
            Whether or not the exponential smoothing model should include a
            damping component.

        grid_search_guess : bool, optional (default=True)
            If True, used grid search (a.k.a. brute force) to search for good
            starting parameters. If False, this method becomes more less
            computationally intensive, but may fail to converge with higher
            chances.

        ts_period : int, optional
            Time-series period.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`statsmodels.tsa.holtwinters.HoltWintersResultsWrapper`
            Results of a optimized triple exponential smoothing model.

        References
        ----------
        .. [1] Winters, Peter R. Forecasting Sales by Exponentially Weighted
            Moving Averages, 1960, INFORMS, Linthicum, MD, USA
            https://doi.org/10.1287/mnsc.6.3.324
        .. [2] Charles C. Holt, Forecasting seasonals and trends by
            exponentially weighted moving averages, International Journal of
            Forecasting, Volume 20, Issue 1, 2004, Pages 5-10, ISSN 0169-2070,
            https://doi.org/10.1016/j.ijforecast.2003.09.015.
        """
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        ts_period = _period.get_ts_period(ts=ts_scaled, ts_period=ts_period)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                module="statsmodels",
                category=statsmodels.tools.sm_exceptions.ConvergenceWarning)

            model = statsmodels.tsa.holtwinters.ExponentialSmoothing(
                endog=ts_scaled,
                trend="additive",
                seasonal="additive",
                damped=damped,
                seasonal_periods=ts_period).fit(use_brute=grid_search_guess)

        return model

    @staticmethod
    def _fit_ortho_pol_reg(
            ts_trend: np.ndarray,
            degree: int = 2
    ) -> statsmodels.regression.linear_model.RegressionResults:
        """Regress the time-series trend on orthogonal polinomials.

        Parameters
        ----------
        ts_trend : :obj:`np.ndarray`
            One-dimensional time-series trend component.

        degree : int, optional (default=2)
            Degree of the highest order polynomial (and, therefore, the number
            of distinct polynomials used).

        Returns
        -------
        :obj:`statsmodels.regression.linear_model.RegressionResults`
            Optimized parameters of the linear model of the time-series trend
            component regressed on the orthogonal polynomials.
        """
        X = _orthopoly.ortho_poly(ts=np.linspace(0, 1, ts_trend.size),
                                  degree=degree,
                                  return_coeffs=False)

        X = statsmodels.tools.add_constant(X)

        ts_trend_scaled = _utils.standardize_ts(ts=ts_trend)

        return statsmodels.regression.linear_model.OLS(ts_trend_scaled,
                                                       X).fit()

    @classmethod
    def ft_des_level(
            cls,
            ts: np.ndarray,
            damped: bool = False,
            ts_scaled: t.Optional[np.ndarray] = None,
            res_model_des: t.Optional[
                statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None,
    ) -> float:
        """Double exponential smoothing model (additive trend) level parameter.

        The `level` parameter is also known as the `alpha` parameter, from the
        traditional Double Exponential Smoothing definition formula. It is the
        `smoothing` factor of the model.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        damped : bool, optional (default=False)
            Whether or not the exponential smoothing model should include a
            damping component.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        res_model_des : :obj:`HoltWintersResultsWrapper`, optional
            Results after fitting a double exponential smoothing model. Used
            to take advantage of precomputations.

        Returns
        -------
        float
            `Level` (or `alpha`) parameter from a double exponential smoothing
            model.

        References
        ----------
        .. [1] Winters, Peter R. Forecasting Sales by Exponentially Weighted
            Moving Averages, 1960, INFORMS, Linthicum, MD, USA
            https://doi.org/10.1287/mnsc.6.3.324
        .. [2] Hyndman, R. J., Wang, E., Kang, Y., & Talagala, T. (2018).
            tsfeatures: Time series feature extraction. R package version 0.1.
        .. [3] Pablo Montero-Manso, George Athanasopoulos, Rob J. Hyndman,
            Thiyanga S. Talagala, FFORMA: Feature-based forecast model
            averaging, International Journal of Forecasting, Volume 36, Issue
            1, 2020, Pages 86-92, ISSN 0169-2070,
            https://doi.org/10.1016/j.ijforecast.2019.02.011.
        """
        if res_model_des is None:
            res_model_des = cls._fit_res_model_des(ts=ts,
                                                   ts_scaled=ts_scaled,
                                                   damped=damped)

        param_level = res_model_des.params["smoothing_level"]

        return param_level

    @classmethod
    def ft_des_slope(
            cls,
            ts: np.ndarray,
            damped: bool = False,
            ts_scaled: t.Optional[np.ndarray] = None,
            res_model_des: t.Optional[
                statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None,
    ) -> float:
        """Double exponential smoothing model (additive trend) slope parameter.

        The `slope` parameter is also known as the `beta` parameter, from the
        traditional Double Exponential Smoothing definition formula. This
        parameter controls the decay of the influence of the trend change into
        the model.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        damped : bool, optional (default=False)
            Whether or not the exponential smoothing model should include a
            damping component.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        res_model_des : :obj:`HoltWintersResultsWrapper`, optional
            Results after fitting a double exponential smoothing model. Used
            to take advantage of precomputations.

        Returns
        -------
        float
            `Slope` (or `beta`) parameter from a double exponential smoothing
            model.

        References
        ----------
        .. [1] Winters, Peter R. Forecasting Sales by Exponentially Weighted
            Moving Averages, 1960, INFORMS, Linthicum, MD, USA
            https://doi.org/10.1287/mnsc.6.3.324
        .. [2] Hyndman, R. J., Wang, E., Kang, Y., & Talagala, T. (2018).
            tsfeatures: Time series feature extraction. R package version 0.1.
        .. [3] Pablo Montero-Manso, George Athanasopoulos, Rob J. Hyndman,
            Thiyanga S. Talagala, FFORMA: Feature-based forecast model
            averaging, International Journal of Forecasting, Volume 36, Issue
            1, 2020, Pages 86-92, ISSN 0169-2070,
            https://doi.org/10.1016/j.ijforecast.2019.02.011.
        """
        if res_model_des is None:
            res_model_des = cls._fit_res_model_des(ts=ts,
                                                   ts_scaled=ts_scaled,
                                                   damped=damped)

        param_slope = res_model_des.params["smoothing_slope"]

        return param_slope

    @classmethod
    def ft_ets_level(
            cls,
            ts: np.ndarray,
            damped: bool = True,
            ts_period: t.Optional[int] = None,
            ts_scaled: t.Optional[np.ndarray] = None,
            res_model_ets: t.Optional[
                statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None,
    ) -> float:
        """ETS (additive components) model level parameter.

        ETS models are also known as `Holt-Winters Exponential Smoothing`.

        The `level` parameter is also known as the `alpha` parameter, from the
        traditional Triple Exponential Smoothing definition formula. It is the
        `smoothing` factor of the model.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        damped : bool, optional (default=False)
            Whether or not the exponential smoothing model should include a
            damping component.

        ts_period : int, optional
            Time-series period.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        res_model_ets : :obj:`HoltWintersResultsWrapper`, optional
            Results after fitting a triple exponential smoothing model. Used
            to take advantage of precomputations.

        Returns
        -------
        float
            `Level` (or `alpha`) parameter from a ETS model.

        References
        ----------
        .. [1] Winters, Peter R. Forecasting Sales by Exponentially Weighted
            Moving Averages, 1960, INFORMS, Linthicum, MD, USA
            https://doi.org/10.1287/mnsc.6.3.324
        .. [2] Charles C. Holt, Forecasting seasonals and trends by
            exponentially weighted moving averages, International Journal of
            Forecasting, Volume 20, Issue 1, 2004, Pages 5-10, ISSN 0169-2070,
            https://doi.org/10.1016/j.ijforecast.2003.09.015.
        .. [3] Hyndman, R. J., Wang, E., Kang, Y., & Talagala, T. (2018).
            tsfeatures: Time series feature extraction. R package version 0.1.
        .. [4] Pablo Montero-Manso, George Athanasopoulos, Rob J. Hyndman,
            Thiyanga S. Talagala, FFORMA: Feature-based forecast model
            averaging, International Journal of Forecasting, Volume 36, Issue
            1, 2020, Pages 86-92, ISSN 0169-2070,
            https://doi.org/10.1016/j.ijforecast.2019.02.011.
        """
        if res_model_ets is None:
            res_model_ets = cls._fit_res_model_ets(ts=ts,
                                                   ts_scaled=ts_scaled,
                                                   ts_period=ts_period,
                                                   damped=damped)

        param_level = res_model_ets.params["smoothing_level"]

        return param_level

    @classmethod
    def ft_ets_slope(
            cls,
            ts: np.ndarray,
            damped: bool = True,
            ts_period: t.Optional[int] = None,
            ts_scaled: t.Optional[np.ndarray] = None,
            res_model_ets: t.Optional[
                statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None,
    ) -> float:
        """ETS (additive components) model slope parameter.

        ETS models are also known as `Holt-Winters Exponential Smoothing`.

        The `slope` parameter is also known as the `beta` parameter, from the
        traditional Triple Exponential Smoothing definition formula. This
        parameter controls the decay of the influence of the trend change into
        the model.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        damped : bool, optional (default=False)
            Whether or not the exponential smoothing model should include a
            damping component.

        ts_period : int, optional
            Time-series period.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        res_model_ets : :obj:`HoltWintersResultsWrapper`, optional
            Results after fitting a triple exponential smoothing model. Used
            to take advantage of precomputations.

        Returns
        -------
        float
            `Slope` (or `beta`) parameter from a ETS model.

        References
        ----------
        .. [1] Winters, Peter R. Forecasting Sales by Exponentially Weighted
            Moving Averages, 1960, INFORMS, Linthicum, MD, USA
            https://doi.org/10.1287/mnsc.6.3.324
        .. [2] Charles C. Holt, Forecasting seasonals and trends by
            exponentially weighted moving averages, International Journal of
            Forecasting, Volume 20, Issue 1, 2004, Pages 5-10, ISSN 0169-2070,
            https://doi.org/10.1016/j.ijforecast.2003.09.015.
        .. [3] Hyndman, R. J., Wang, E., Kang, Y., & Talagala, T. (2018).
            tsfeatures: Time series feature extraction. R package version 0.1.
        .. [4] Pablo Montero-Manso, George Athanasopoulos, Rob J. Hyndman,
            Thiyanga S. Talagala, FFORMA: Feature-based forecast model
            averaging, International Journal of Forecasting, Volume 36, Issue
            1, 2020, Pages 86-92, ISSN 0169-2070,
            https://doi.org/10.1016/j.ijforecast.2019.02.011.
        """
        if res_model_ets is None:
            res_model_ets = cls._fit_res_model_ets(ts=ts,
                                                   ts_scaled=ts_scaled,
                                                   ts_period=ts_period,
                                                   damped=damped)

        param_slope = res_model_ets.params["smoothing_slope"]

        return param_slope

    @classmethod
    def ft_ets_season(
            cls,
            ts: np.ndarray,
            damped: bool = True,
            ts_period: t.Optional[int] = None,
            ts_scaled: t.Optional[np.ndarray] = None,
            res_model_ets: t.Optional[
                statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None,
    ) -> float:
        """ETS (additive components) model seasonal parameter.

        ETS models are also known as `Holt-Winters Exponential Smoothing`.

        The `seasonal` parameter is also known as the `gamma` parameter, from
        the traditional Triple Exponential Smoothing definition formula. It
        controls the influence of the seasonal component into the model.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        damped : bool, optional (default=False)
            Whether or not the exponential smoothing model should include a
            damping component.

        ts_period : int, optional
            Time-series period.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        res_model_ets : :obj:`HoltWintersResultsWrapper`, optional
            Results after fitting a triple exponential smoothing model. Used
            to take advantage of precomputations.

        Returns
        -------
        float
            `Seasonal` (or `gamma`) parameter from a ETS model.

        References
        ----------
        .. [1] Winters, Peter R. Forecasting Sales by Exponentially Weighted
            Moving Averages, 1960, INFORMS, Linthicum, MD, USA
            https://doi.org/10.1287/mnsc.6.3.324
        .. [2] Charles C. Holt, Forecasting seasonals and trends by
            exponentially weighted moving averages, International Journal of
            Forecasting, Volume 20, Issue 1, 2004, Pages 5-10, ISSN 0169-2070,
            https://doi.org/10.1016/j.ijforecast.2003.09.015.
        .. [3] Hyndman, R. J., Wang, E., Kang, Y., & Talagala, T. (2018).
            tsfeatures: Time series feature extraction. R package version 0.1.
        .. [4] Pablo Montero-Manso, George Athanasopoulos, Rob J. Hyndman,
            Thiyanga S. Talagala, FFORMA: Feature-based forecast model
            averaging, International Journal of Forecasting, Volume 36, Issue
            1, 2020, Pages 86-92, ISSN 0169-2070,
            https://doi.org/10.1016/j.ijforecast.2019.02.011.
        """
        if res_model_ets is None:
            res_model_ets = cls._fit_res_model_ets(ts=ts,
                                                   ts_scaled=ts_scaled,
                                                   ts_period=ts_period,
                                                   damped=damped)

        param_season = res_model_ets.params["smoothing_seasonal"]

        return param_season

    @classmethod
    def ft_linearity(
            cls,
            ts_trend: np.ndarray,
            res_model_orthoreg: t.Optional[
                statsmodels.regression.linear_model.RegressionResults] = None,
    ) -> float:
        """Linearity measure from a orthogonal polynomial linear regression.

        The linearity measure is defined as the coefficient associated with
        the first order orthogonal polynomial from a linear regression of the
        time-series trend component onto a pair of orthogonal polynomials of
        order 1 and 2.

        Parameters
        ----------
        ts_trend : :obj:`np.ndarray`
            One-dimensional time-series trend component.

        res_model_orthoreg : :obj:`RegressionResults`, optional
            Linear regression results from the time-series trend component
            regressed onto a pair of orthogonal polynomials of order 1 and
            2. Used to take advantage of precomputations.

        Returns
        -------
        float
            Linearity measure.

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
        if res_model_orthoreg is None:
            res_model_orthoreg = cls._fit_ortho_pol_reg(ts_trend=ts_trend)

        _, linearity, _ = res_model_orthoreg.params

        return linearity

    @classmethod
    def ft_curvature(
            cls,
            ts_trend: np.ndarray,
            res_model_orthoreg: t.Optional[
                statsmodels.regression.linear_model.RegressionResults] = None,
    ) -> float:
        """Curvature measure from a orthogonal polynomial linear regression.

        The curvature measure is defined as the coefficient associated with
        the second order orthogonal polynomial from a linear regression of the
        time-series trend component onto a pair of orthogonal polynomials of
        order 1 and 2.

        Parameters
        ----------
        ts_trend : :obj:`np.ndarray`
            One-dimensional time-series trend component.

        res_model_orthoreg : :obj:`RegressionResults`, optional
            Linear regression results from the time-series trend component
            regressed onto a pair of orthogonal polynomials of order 1 and
            2. Used to take advantage of precomputations.

        Returns
        -------
        float
            Curvature measure.

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
        if res_model_orthoreg is None:
            res_model_orthoreg = cls._fit_ortho_pol_reg(ts_trend=ts_trend)

        _, _, curvature = res_model_orthoreg.params

        return curvature

    @classmethod
    def ft_avg_cycle_period(cls, ts: np.ndarray) -> float:
        r"""Average cycle period from a AR(2) model.

        The average cycle period is defined as:
        $$
            \frac{2 * pi}{\arccos{-4 * phi_1 * (1 - \phi_2) / \phi_2}}
        $$
        If and only if $phi_1^{2} + 4 * phi_2 < 0$, where $\phi_1$ and
        $\phi_2$ are the parameters of the AR model associated with,
        respectively, the $y_{t-1}$ and $y_{t-2}$ previous observations
        from a time series $y$.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        Returns
        -------
        float
            Time-series average cycle period, if any. Return `np.nan` if no
            cycle is detected by the AR(2) model.

        References
        ----------
        .. [1] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on May 23 2020.
            URL to this fomula: https://otexts.com/fpp2/non-seasonal-arima.html
        """
        model_res = statsmodels.tsa.arima_model.ARIMA(ts, order=(2, 0, 0)).fit(
            disp=-1, full_output=False)
        theta_a, theta_b = model_res.arparams

        has_cycle = theta_a**2 + 4 * theta_b < 0

        if not has_cycle:
            return np.nan

        avg_cycle_period = 2 * np.pi / np.arccos(-0.25 * theta_a *
                                                 (1 - theta_b) / theta_b)

        return avg_cycle_period

    @classmethod
    def ft_gaussian_r_sqr(
            cls,
            ts: np.ndarray,
            random_state: t.Optional[int] = None,
            ts_scaled: t.Optional[np.ndarray] = None,
            gaussian_model: t.Optional[
                sklearn.gaussian_process.GaussianProcessRegressor] = None,
    ) -> float:
        """R^2 from a gaussian process model.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        random_state : int, optional
            Random seed to optimize the gaussian process model, to keep
            the results reproducible.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        gaussian_model : :obj:`GaussianProcessRegressor`, optional
            A fitted model of a gaussian process. Used to take advantage of
            precomputations.

        Returns
        -------
        float
            R^2 of a gaussian process model.

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
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        gaussian_model = _utils.fit_gaussian_process(
            ts=ts_scaled,
            random_state=random_state,
            gaussian_model=gaussian_model,
            ts_scaled=ts_scaled)

        X = np.linspace(0, 1, ts_scaled.size).reshape(-1, 1)
        r_squared = gaussian_model.score(X=X, y=ts_scaled)

        return r_squared

    @classmethod
    def ft_ioe_std_slope(
            cls,
            ts: np.ndarray,
            step_size: float = 0.05,
            ts_scaled: t.Optional[np.ndarray] = None,
            res_ioe_std_linreg: t.Optional[
                statsmodels.regression.linear_model.RegressionResults] = None,
    ) -> float:
        """Linear model of IOE standard deviations onto thresholds slope.

        In the iterative outlier exclusion, a uniformly spaced set of
        thresholds over the time-series range is build in increasing order.
        For each threshold it is calculated a statistic of the timestamp
        values of instances larger or equal than the current threshold.

        This method calculates the slope coefficient of a linear model of the
        standard deviation of the timestamps regressed on the thresholds of
        each iteration.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        step_size : float, optional (default=0.05)
            Increase of the outlier threshold in each iteration. Must be a
            number strictly positive.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        res_ioe_std_linreg : :obj:`RegressionResults`, optional
            Results from the linear regression model of iterative outlier
            exclusion timestamps standard deviation regression on the
            thresholds.

        Returns
        -------
        float
            Slope from the linear regression model.

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
        if res_ioe_std_linreg is None:
            res_ioe_std_linreg = cls._fit_ioe_std_lin_model(
                ts=ts, step_size=step_size, ts_scaled=ts_scaled)

        _, slope = res_ioe_std_linreg.params

        return slope

    @classmethod
    def ft_ioe_std_adj_r_sqr(
            cls,
            ts: np.ndarray,
            step_size: float = 0.05,
            ts_scaled: t.Optional[np.ndarray] = None,
            res_ioe_std_linreg: t.Optional[
                statsmodels.regression.linear_model.RegressionResults] = None,
    ) -> float:
        """Linear model of IOE standard deviations onto thresholds Adj. R^2.

        In the iterative outlier exclusion, a uniformly spaced set of
        thresholds over the time-series range is build in increasing order.
        For each threshold it is calculated a statistic of the timestamp
        values of instances larger or equal than the current threshold.

        This method calculates the Adjusted R^2 from the linear model of the
        standard deviation of the timestamps regressed on the thresholds of
        each iteration.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        step_size : float, optional (default=0.05)
            Increase of the outlier threshold in each iteration. Must be a
            number strictly positive.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        res_ioe_std_linreg : :obj:`RegressionResults`, optional
            Results from the linear regression model of iterative outlier
            exclusion timestamps standard deviation regression on the
            thresholds.

        Returns
        -------
        float
            Adjusted R^2 from the linear regression model.

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
        if res_ioe_std_linreg is None:
            res_ioe_std_linreg = cls._fit_ioe_std_lin_model(
                ts=ts, step_size=step_size, ts_scaled=ts_scaled)

        adj_r_sqr = res_ioe_std_linreg.rsquared_adj

        return adj_r_sqr
