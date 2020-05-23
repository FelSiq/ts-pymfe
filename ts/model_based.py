"""Module dedicated to model-based time-series meta-features."""
import typing as t

import sklearn.gaussian_process
import sklearn.preprocessing
import statsmodels.tsa.holtwinters
import statsmodels.tsa.arima_model
import statsmodels.regression
import statsmodels.tools
import numpy as np

import _utils
import _orthopoly
import _period
import _detrend
import _get_data


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
            ts_period = _period.ts_period(ts=ts, ts_period=ts_period)
            precomp_vals["ts_period"] = ts_period

        return precomp_vals

    @classmethod
    def precompute_model_ets(
        cls,
        ts: np.ndarray,
        damped: bool = False,
        ts_period: t.Optional[int] = None,
        **kwargs,
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
                * ``res_model_ets_double`` (:obj:`HoltWintersResultsWrapper`):
                    Double exponential smoothing model (exponential smoothing
                    model without the seasonal component) with additive trend
                    results.
                * ``res_model_ets_triple`` (:obj:`HoltWintersResultsWrapper`):
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

        if "ts_scaled" not in kwargs:
            precomp_vals["ts_scaled"] = _utils.standardize_ts(ts=ts)

        ts_scaled = kwargs.get("ts_scaled", precomp_vals["ts_scaled"])

        if "res_model_ets_double" not in kwargs:
            model = cls._fit_res_model_ets_double(ts=ts_scaled, damped=damped)
            precomp_vals["res_model_ets_double"] = model

        if ts_period is None:
            precomp_vals.update(cls.precompute_period(ts=ts))
            ts_period = precomp_vals["ts_period"]

        if "res_model_ets_triple" not in kwargs:
            model = cls._fit_res_model_ets_triple(ts=ts_scaled,
                                                  ts_period=ts_period,
                                                  damped=damped)
            precomp_vals["res_model_ets_triple"] = model

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
                * ``res_ioe_std_lin_model`` (:obj:`RegressionResults`):
                    Results of the regression model of iterative outlier
                    exclusion timestamps standard deviation regression on
                    the thresholds.

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

        if "ts_scaled" not in kwargs:
            precomp_vals["ts_scaled"] = _utils.standardize_ts(ts=ts)

        ts_scaled = kwargs.get("ts_scaled", precomp_vals["ts_scaled"])

        if "res_ioe_std_lin_model" not in kwargs:
            lin_reg_res = cls._fit_ioe_std_lin_model(ts=ts,
                                                     ts_scaled=ts_scaled,
                                                     step_size=step_size)
            precomp_vals["res_ioe_std_lin_model"] = lin_reg_res

        return precomp_vals

    @staticmethod
    def _fit_ioe_std_lin_model(
        ts: np.ndarray,
        step_size: float = 0.05,
        ts_scaled: t.Optional[np.ndarray] = None
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
            Results of the regression model of iterative outlier exclusion
            timestamps standard deviation regression on the thresholds.

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
        ioe_std_func = lambda arr: np.std(arr, ddof=1) / np.sqrt(arr.size)

        ioe_std = _utils.calc_ioe_stats(ts=ts,
                                        funcs=ioe_std_func,
                                        ts_scaled=ts_scaled,
                                        step_size=step_size)

        thresholds = statsmodels.tools.add_constant(
            np.arange(ioe_std.size) * step_size)

        lin_reg_res = statsmodels.regression.linear_model.OLS(
            ioe_std, thresholds).fit()

        return lin_reg_res

    @staticmethod
    def _fit_res_model_ets_double(
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
        .. [1] Winters, Peter R. Forecasting Sales by Exponentially Weighted
            Moving Averages, 1960, INFORMS, Linthicum, MD, USA
            https://doi.org/10.1287/mnsc.6.3.324
        """
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        model = statsmodels.tsa.holtwinters.ExponentialSmoothing(
            endog=ts_scaled, trend="additive", damped=damped,
            seasonal=None).fit()

        return model

    @staticmethod
    def _fit_res_model_ets_triple(
        ts: np.ndarray,
        damped: bool = False,
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

        ts_period = _period.ts_period(ts=ts_scaled, ts_period=ts_period)

        model = statsmodels.tsa.holtwinters.ExponentialSmoothing(
            endog=ts_scaled,
            trend="additive",
            seasonal="additive",
            damped=damped,
            seasonal_periods=ts_period).fit()

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
    def ft_ets_double_level(
        cls,
        ts: np.ndarray,
        damped: bool = False,
        ts_scaled: t.Optional[np.ndarray] = None,
        res_model_ets_double: t.Optional[
            statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None
    ) -> float:
        """TODO."""
        if res_model_ets_double is None:
            res_model_ets_double = cls._fit_res_model_ets_double(
                ts=ts, ts_scaled=ts_scaled, damped=damped)

        param_level = res_model_ets_double.params["smoothing_level"]

        return param_level

    @classmethod
    def ft_ets_double_slope(
        cls,
        ts: np.ndarray,
        damped: bool = False,
        ts_scaled: t.Optional[np.ndarray] = None,
        res_model_ets_double: t.Optional[
            statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None,
    ) -> float:
        """TODO."""
        if res_model_ets_double is None:
            res_model_ets_double = cls._fit_res_model_ets_double(
                ts=ts, ts_scaled=ts_scaled, damped=damped)

        param_slope = res_model_ets_double.params["smoothing_slope"]

        return param_slope

    @classmethod
    def ft_ets_triple_level(
        cls,
        ts: np.ndarray,
        damped: bool = True,
        ts_period: t.Optional[int] = None,
        ts_scaled: t.Optional[np.ndarray] = None,
        res_model_ets_triple: t.Optional[
            statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None,
    ) -> float:
        """TODO."""
        if res_model_ets_triple is None:
            res_model_ets_triple = cls._fit_res_model_ets_triple(
                ts=ts, ts_scaled=ts_scaled, ts_period=ts_period, damped=damped)

        param_level = res_model_ets_triple.params["smoothing_level"]

        return param_level

    @classmethod
    def ft_ets_triple_slope(
        cls,
        ts: np.ndarray,
        damped: bool = True,
        ts_period: t.Optional[int] = None,
        ts_scaled: t.Optional[np.ndarray] = None,
        res_model_ets_triple: t.Optional[
            statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None,
    ) -> float:
        """TODO."""
        if res_model_ets_triple is None:
            res_model_ets_triple = cls._fit_res_model_ets_triple(
                ts=ts, ts_scaled=ts_scaled, ts_period=ts_period, damped=damped)

        param_slope = res_model_ets_triple.params["smoothing_slope"]

        return param_slope

    @classmethod
    def ft_ets_triple_season(
        cls,
        ts: np.ndarray,
        damped: bool = True,
        ts_period: t.Optional[int] = None,
        ts_scaled: t.Optional[np.ndarray] = None,
        res_model_ets_triple: t.Optional[
            statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None,
    ) -> float:
        """TODO."""
        if res_model_ets_triple is None:
            res_model_ets_triple = cls._fit_res_model_ets_triple(
                ts=ts, ts_scaled=ts_scaled, ts_period=ts_period, damped=damped)

        param_season = res_model_ets_triple.params["smoothing_seasonal"]

        return param_season

    @classmethod
    def ft_linearity(
        cls,
        ts_trend: np.ndarray,
        res_model_orthop_reg: t.Optional[
            statsmodels.regression.linear_model.RegressionResults] = None
    ) -> float:
        """TODO."""
        if res_model_orthop_reg is None:
            res_model_orthop_reg = cls._fit_ortho_pol_reg(ts_trend=ts_trend)

        _, linearity, _ = res_model_orthop_reg.params

        return linearity

    @classmethod
    def ft_curvature(
        cls,
        ts_trend: np.ndarray,
        res_model_orthop_reg: t.Optional[
            statsmodels.regression.linear_model.RegressionResults] = None
    ) -> float:
        """TODO."""
        if res_model_orthop_reg is None:
            res_model_orthop_reg = cls._fit_ortho_pol_reg(ts_trend=ts_trend)

        _, _, curvature = res_model_orthop_reg.params

        return curvature

    @classmethod
    def ft_avg_cycle_period(cls, ts_residuals: np.ndarray) -> float:
        """TODO.

        https://otexts.com/fpp2/non-seasonal-arima.html
        """
        model_res = statsmodels.tsa.arima_model.ARIMA(ts_residuals,
                                                      order=(2, 0, 0)).fit(
                                                          disp=-1,
                                                          full_output=False)
        theta_a, theta_b = model_res.arparams

        has_cycle = theta_a**2 + 4 * theta_b < 0

        if not has_cycle:
            return np.nan

        avg_cycle_period = 2 * np.pi / np.arccos(-0.25 * theta_a *
                                                 (1 - theta_b) / theta_b)

        return avg_cycle_period

    @classmethod
    def ft_gaussian_mle(
        cls,
        ts: np.ndarray,
        random_state: t.Optional[int] = None,
        ts_scaled: t.Optional[np.ndarray] = None,
        gaussian_model: t.Optional[
            sklearn.gaussian_process.GaussianProcessRegressor] = None,
    ) -> float:
        """TODO."""
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
    def ft_ioe_std_curvature(
        cls,
        ts: np.ndarray,
        step_size: float = 0.05,
        ts_scaled: t.Optional[np.ndarray] = None,
        res_ioe_std_lin_model: t.Optional[
            statsmodels.regression.linear_model.RegressionResults] = None
    ) -> float:
        """TODO."""
        if res_ioe_std_lin_model is None:
            res_ioe_std_lin_model = cls._fit_ioe_std_lin_model(
                ts=ts, step_size=step_size,
                ts_scaled=ts_scaled)

        _, curvature = res_ioe_std_lin_model.params

        return curvature

    @classmethod
    def ft_ioe_std_adj_r_sqr(
        cls,
        ts: np.ndarray,
        step_size: float = 0.05,
        ts_scaled: t.Optional[np.ndarray] = None,
        res_ioe_std_lin_model: t.Optional[
            statsmodels.regression.linear_model.RegressionResults] = None
    ) -> float:
        """TODO."""
        if res_ioe_std_lin_model is None:
            res_ioe_std_lin_model = cls._fit_ioe_std_lin_model(
                ts=ts, step_size=step_size,
                ts_scaled=ts_scaled)

        adj_r_sqr = res_ioe_std_lin_model.rsquared_adj

        return adj_r_sqr


def _test() -> None:
    ts = _get_data.load_data(3)

    # Note: add cyclic behaviour to data
    # ts += -100 * (np.random.random(ts.size) < 0.3)
    """
    import matplotlib.pyplot as plt
    plt.plot(ts)
    plt.show()
    """
    ts_period = _period.ts_period(ts)
    ts_trend, ts_season, ts_residuals = _detrend.decompose(ts,
                                                           ts_period=ts_period)
    ts = ts.to_numpy()

    res = MFETSModelBased.ft_ioe_std_adj_r_sqr(ts)
    print(res)

    res = MFETSModelBased.ft_ioe_std_curvature(ts)
    print(res)

    res = MFETSModelBased.ft_gaussian_mle(ts)
    print(res)

    res = MFETSModelBased.ft_avg_cycle_period(ts_residuals)
    print(res)

    res = MFETSModelBased.ft_linearity(ts_trend)
    print("linearity", res)

    res = MFETSModelBased.ft_curvature(ts_trend)
    print("curvature", res)

    res = MFETSModelBased.ft_ets_double_level(ts)
    print(res)

    res = MFETSModelBased.ft_ets_double_slope(ts)
    print(res)

    res = MFETSModelBased.ft_ets_triple_level(ts)
    print(res)

    res = MFETSModelBased.ft_ets_triple_slope(ts)
    print(res)

    res = MFETSModelBased.ft_ets_triple_season(ts)
    print(res)


if __name__ == "__main__":
    _test()
