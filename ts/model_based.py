import typing as t

import sklearn.mixture
import sklearn.preprocessing
import statsmodels.tsa.holtwinters
import statsmodels.regression
import statsmodels.api
import statsmodels.tsa.arima_model
import numpy as np

import _utils
import _orthopoly
import _period
import _detrend
import _get_data


class MFETSModelBased:
    @classmethod
    def precompute_model_ets(
        cls,
        ts: np.ndarray,
        ts_period: int,
        damped: bool = False,
        **kwargs,
    ) -> t.Dict[str, statsmodels.tsa.holtwinters.HoltWintersResultsWrapper]:
        """TODO."""
        precomp_vals = {}

        ts_scaled = kwargs.get("ts_scaled")

        if ts_scaled is None:
            ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)
            precomp_vals["ts_scaled"] = ts_scaled

        if "res_model_ets_double" not in kwargs:
            model = cls._fit_res_model_ets_double(ts=ts_scaled, damped=damped)
            precomp_vals["res_model_ets_double"] = model

            model = cls._fit_res_model_ets_triple(ts=ts_scaled,
                                                  ts_period=ts_period,
                                                  damped=damped)
            precomp_vals["res_model_ets_triple"] = model

        return precomp_vals

    @classmethod
    def precompute_ioi_std_linear_model(
        cls,
        ts: np.ndarray,
        step_size: float = 0.05,
        ts_scaled: t.Optional[np.ndarray] = None,
        **kwargs
    ) -> t.Dict[str, statsmodels.regression.linear_model.RegressionResults]:
        """TODO."""
        precomp_vals = {}

        if "res_ioi_std_lin_model" not in kwargs:
            ioi_std_func = lambda arr: np.std(arr, ddof=1) / np.sqrt(arr.size)

            ioi_std = _utils.calc_ioi_stats(ts=ts,
                                            funcs=ioi_std_func,
                                            ts_scaled=ts_scaled,
                                            step_size=step_size)
            thresholds = np.arange(ioi_std.size) * step_size

            precomp_vals["res_ioi_std_lin_model"] = (
                statsmodels.regression.linear_model.OLS(ioi_std,
                                                        thresholds).fit())

        return precomp_vals

    @staticmethod
    def _fit_res_model_ets_double(
        ts: np.ndarray,
        damped: bool = False
    ) -> statsmodels.tsa.holtwinters.HoltWintersResultsWrapper:
        """TODO."""
        model = statsmodels.tsa.holtwinters.ExponentialSmoothing(
            endog=ts, trend="additive", damped=damped, seasonal=None).fit()

        return model

    @staticmethod
    def _fit_res_model_ets_triple(
        ts: np.ndarray,
        ts_period: int,
        damped: bool = False,
    ) -> statsmodels.tsa.holtwinters.HoltWintersResultsWrapper:
        """TODO."""
        model = statsmodels.tsa.holtwinters.ExponentialSmoothing(
            endog=ts,
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
        X = _orthopoly.ortho_poly(ts=np.arange(ts_trend.size),
                                  degree=degree,
                                  return_coeffs=False)

        ts_trend_scaled = sklearn.preprocessing.StandardScaler().fit_transform(
            ts_trend.reshape(-1, 1)).ravel()

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
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        if res_model_ets_double is None:
            res_model_ets_double = cls._fit_res_model_ets_double(ts=ts_scaled,
                                                                 damped=damped)

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
        if ts_scaled is None:
            ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        if res_model_ets_double is None:
            res_model_ets_double = cls._fit_res_model_ets_double(ts=ts_scaled,
                                                                 damped=damped)

        beta = res_model_ets_double.params_formatted["param"][
            "smoothing_slope"]

        return beta

    @classmethod
    def ft_ets_triple_alpha(
        cls,
        ts: np.ndarray,
        ts_period: int,
        damped: bool = True,
        ts_scaled: t.Optional[np.ndarray] = None,
        res_model_ets_triple: t.Optional[
            statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None,
    ) -> float:
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        if res_model_ets_triple is None:
            res_model_ets_triple = cls._fit_res_model_ets_triple(
                ts=ts_scaled, ts_period=ts_period, damped=damped)

        alpha = res_model_ets_triple.params_formatted["param"][
            "smoothing_level"]

        return alpha

    @classmethod
    def ft_ets_triple_beta(
        cls,
        ts: np.ndarray,
        ts_period: int,
        damped: bool = True,
        ts_scaled: t.Optional[np.ndarray] = None,
        res_model_ets_triple: t.Optional[
            statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None,
    ) -> float:
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        if res_model_ets_triple is None:
            res_model_ets_triple = cls._fit_res_model_ets_triple(
                ts=ts_scaled, ts_period=ts_period, damped=damped)

        beta = res_model_ets_triple.params_formatted["param"][
            "smoothing_slope"]

        return beta

    @classmethod
    def ft_ets_triple_gamma(
        cls,
        ts: np.ndarray,
        ts_period: int,
        damped: bool = True,
        ts_scaled: t.Optional[np.ndarray] = None,
        res_model_ets_triple: t.Optional[
            statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None,
    ) -> float:
        if ts_scaled is None:
            ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        if res_model_ets_triple is None:
            res_model_ets_triple = cls._fit_res_model_ets_triple(
                ts=ts_scaled, ts_period=ts_period, damped=damped)

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

        linearity, _ = res_model_orthop_reg.params

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

        _, curvature = res_model_orthop_reg.params

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
        n_components: int = 2,
        random_state: t.Optional[int] = None,
        ts_scaled: t.Optional[np.ndarray] = None,
        gaussian_model: t.Optional[sklearn.mixture.GaussianMixture] = None,
    ) -> float:
        """TODO."""
        ts_scaled = _utils.standardize_ts(ts=ts,
                                          ts_scaled=ts_scaled).reshape(-1, 1)

        gaussian_model = _utils.fit_gaussian_mix(ts=ts_scaled,
                                                 n_components=n_components,
                                                 random_state=random_state,
                                                 gaussian_model=gaussian_model)

        gaussian_mle = gaussian_model.score(ts_scaled)

        return gaussian_mle

    @classmethod
    def ft_ioi_std_curvature(
        cls,
        ts: np.ndarray,
        step_size: float = 0.05,
        ts_scaled: t.Optional[np.ndarray] = None,
        res_ioi_std_lin_model: t.Optional[
            statsmodels.regression.linear_model.RegressionResults] = None
    ) -> float:
        """TODO."""
        if res_ioi_std_lin_model is None:
            res_ioi_std_lin_model = cls.precompute_ioi_std_linear_model(
                ts=ts, step_size=step_size,
                ts_scaled=ts_scaled)["res_ioi_std_lin_model"]

        curvature = res_ioi_std_lin_model.params[0]

        return curvature

    @classmethod
    def ft_ioi_std_curvature(
        cls,
        ts: np.ndarray,
        step_size: float = 0.05,
        ts_scaled: t.Optional[np.ndarray] = None,
        res_ioi_std_lin_model: t.Optional[
            statsmodels.regression.linear_model.RegressionResults] = None
    ) -> float:
        """TODO."""
        if res_ioi_std_lin_model is None:
            res_ioi_std_lin_model = cls.precompute_ioi_std_linear_model(
                ts=ts, step_size=step_size,
                ts_scaled=ts_scaled)["res_ioi_std_lin_model"]

        curvature = res_ioi_std_lin_model.params[0]

        return curvature

    @classmethod
    def ft_ioi_std_adj_r_sqr(
        cls,
        ts: np.ndarray,
        step_size: float = 0.05,
        ts_scaled: t.Optional[np.ndarray] = None,
        res_ioi_std_lin_model: t.Optional[
            statsmodels.regression.linear_model.RegressionResults] = None
    ) -> float:
        """TODO."""
        if res_ioi_std_lin_model is None:
            res_ioi_std_lin_model = cls.precompute_ioi_std_linear_model(
                ts=ts, step_size=step_size,
                ts_scaled=ts_scaled)["res_ioi_std_lin_model"]

        adj_r_sqr = res_ioi_std_lin_model.rsquared_adj

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

    res = MFETSModelBased.ft_ioi_std_adj_r_sqr(ts)
    print(res)
    exit(1)

    res = MFETSModelBased.ft_ioi_std_curvature(ts)
    print(res)
    exit(1)

    res = MFETSModelBased.ft_gaussian_mle(ts)
    print(res)
    exit(1)

    res = MFETSModelBased.ft_avg_cycle_period(ts_residuals)
    print(res)
    exit(1)

    res = MFETSModelBased.ft_linearity(ts_trend)
    print(res)

    res = MFETSModelBased.ft_curvature(ts_trend)
    print(res)

    res = MFETSModelBased.ft_ets_double_alpha(ts)
    print(res)

    res = MFETSModelBased.ft_ets_double_beta(ts)
    print(res)

    res = MFETSModelBased.ft_ets_triple_alpha(ts, ts_period)
    print(res)

    res = MFETSModelBased.ft_ets_triple_beta(ts, ts_period)
    print(res)

    res = MFETSModelBased.ft_ets_triple_gamma(ts, ts_period)
    print(res)


if __name__ == "__main__":
    _test()
