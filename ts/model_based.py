import typing as t

import sklearn.preprocessing
import statsmodels.tsa.holtwinters
import statsmodels.regression
import statsmodels.api
import numpy as np
import arch

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
            ts_scaled = cls._scale_ts(ts=ts)
            precomp_vals["ts_scaled"] = ts_scaled

        if "res_model_ets_double" not in kwargs:
            model = cls._fit_res_model_ets_double(ts=ts_scaled, damped=damped)
            precomp_vals["res_model_ets_double"] = model

            model = cls._fit_res_model_ets_triple(ts=ts_scaled,
                                                  ts_period=ts_period,
                                                  damped=damped)
            precomp_vals["res_model_ets_triple"] = model

        return precomp_vals

    @staticmethod
    def _scale_ts(ts: np.ndarray) -> np.ndarray:
        """TODO."""
        ts_scaled = sklearn.preprocessing.StandardScaler().fit_transform(
            ts.reshape(-1, 1)).ravel()

        return ts_scaled

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
        return statsmodels.regression.linear_model.OLS(ts_trend, X).fit()

    @classmethod
    def ft_ets_double_alpha(
        cls,
        ts: np.ndarray,
        damped: bool = False,
        ts_scaled: t.Optional[np.ndarray] = None,
        res_model_ets_double: t.Optional[
            statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None
    ) -> float:
        if ts_scaled is None:
            ts_scaled = cls._scale_ts(ts=ts)

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
            ts_scaled = cls._scale_ts(ts=ts)

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
        if ts_scaled is None:
            ts_scaled = cls._scale_ts(ts=ts)

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
        if ts_scaled is None:
            ts_scaled = cls._scale_ts(ts=ts)

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
            ts_scaled = cls._scale_ts(ts=ts)

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


def _test() -> None:
    ts = _get_data.load_data(3)

    ts_period = _period.ts_period(ts)
    ts_trend, ts_season, ts_residuals = _detrend.decompose(ts,
                                                           ts_period=ts_period)
    ts = ts.to_numpy()

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
