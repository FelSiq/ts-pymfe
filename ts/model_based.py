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

        if ts_period not in kwargs:
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
                    model without the seasonal component) results.
                * ``res_model_ets_triple`` (:obj:`HoltWintersResultsWrapper`):
                    Triple exponential smoothing model (exponential smoothing
                    model with the seasonal component) results.

            The following items are necessary and, therefore, also precomputed
            if necessary:
                * ``ts_scaled`` (:obj:`np.ndarray`): standardized time-series
                    values (z-score).
                * ``ts_period`` (:obj:`int`): time-series period.
        """
        precomp_vals = {}  # type: t.Dict[str, t.Any]

        if ts_scaled not in kwargs:
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
    def precompute_ioe_std_linear_model(
        cls,
        ts: np.ndarray,
        step_size: float = 0.05,
        ts_scaled: t.Optional[np.ndarray] = None,
        **kwargs
    ) -> t.Dict[str, statsmodels.regression.linear_model.RegressionResults]:
        """TODO."""
        precomp_vals = {}

        if "res_ioe_std_lin_model" not in kwargs:
            ioe_std_func = lambda arr: np.std(arr, ddof=1) / np.sqrt(arr.size)

            ioe_std = _utils.calc_ioe_stats(ts=ts,
                                            funcs=ioe_std_func,
                                            ts_scaled=ts_scaled,
                                            step_size=step_size)

            thresholds = statsmodels.tools.add_constant(
                np.arange(ioe_std.size) * step_size)

            precomp_vals["res_ioe_std_lin_model"] = (
                statsmodels.regression.linear_model.OLS(ioe_std,
                                                        thresholds).fit())

        return precomp_vals

    @staticmethod
    def _fit_res_model_ets_double(
        ts: np.ndarray,
        damped: bool = False,
        ts_scaled: t.Optional[np.ndarray] = None,
    ) -> statsmodels.tsa.holtwinters.HoltWintersResultsWrapper:
        """TODO."""
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        model = statsmodels.tsa.holtwinters.ExponentialSmoothing(
            endog=ts_scaled, trend="additive", damped=damped,
            seasonal=None).fit()

        return model

    @staticmethod
    def _fit_res_model_ets_triple(
        ts: np.ndarray,
        ts_period: t.Optional[int] = None,
        damped: bool = False,
        ts_scaled: t.Optional[np.ndarray] = None,
    ) -> statsmodels.tsa.holtwinters.HoltWintersResultsWrapper:
        """TODO."""
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
        """TODO."""
        X = _orthopoly.ortho_poly(ts=np.linspace(0, 1, ts_trend.size),
                                  degree=degree,
                                  return_coeffs=False)

        X = statsmodels.tools.add_constant(X)

        ts_trend_scaled = _utils.standardize_ts(ts=ts_trend)

        return statsmodels.regression.linear_model.OLS(ts_trend_scaled,
                                                       X).fit()

    @classmethod
    def ft_ets_double_alpha(
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

        alpha = res_model_ets_double.params_formatted["param"][
            "smoothing_level"]

        return alpha

    @classmethod
    def ft_ets_double_beta(
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

        beta = res_model_ets_double.params_formatted["param"][
            "smoothing_slope"]

        return beta

    @classmethod
    def ft_ets_triple_alpha(
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

        alpha = res_model_ets_triple.params_formatted["param"][
            "smoothing_level"]

        return alpha

    @classmethod
    def ft_ets_triple_beta(
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

        beta = res_model_ets_triple.params_formatted["param"][
            "smoothing_slope"]

        return beta

    @classmethod
    def ft_ets_triple_gamma(
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

        gamma = res_model_ets_triple.params_formatted["param"][
            "smoothing_seasonal"]

        return gamma

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
            res_ioe_std_lin_model = cls.precompute_ioe_std_linear_model(
                ts=ts, step_size=step_size,
                ts_scaled=ts_scaled)["res_ioe_std_lin_model"]

        curvature = res_ioe_std_lin_model.params[0]

        return curvature

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
            res_ioe_std_lin_model = cls.precompute_ioe_std_linear_model(
                ts=ts, step_size=step_size,
                ts_scaled=ts_scaled)["res_ioe_std_lin_model"]

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
            res_ioe_std_lin_model = cls.precompute_ioe_std_linear_model(
                ts=ts, step_size=step_size,
                ts_scaled=ts_scaled)["res_ioe_std_lin_model"]

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

    res = MFETSModelBased.ft_ets_double_alpha(ts)
    print(res)

    res = MFETSModelBased.ft_ets_double_beta(ts)
    print(res)

    res = MFETSModelBased.ft_ets_triple_alpha(ts)
    print(res)

    res = MFETSModelBased.ft_ets_triple_beta(ts)
    print(res)

    res = MFETSModelBased.ft_ets_triple_gamma(ts)
    print(res)


if __name__ == "__main__":
    _test()
