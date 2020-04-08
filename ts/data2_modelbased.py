import typing as t

import sklearn.preprocessing
import statsmodels.tsa.holtwinters
import numpy as np
import arch

import get_data
import data1_period
import data1_detrend


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

        if "model_ets_double" not in kwargs:
            model = cls._fit_model_ets_double(ts=ts_scaled, damped=damped)
            precomp_vals["model_ets_double"] = model

            model = cls._fit_model_ets_triple(ts=ts_scaled, damped=damped)
            precomp_vals["model_ets_triple"] = model

        return precomp_vals

    @staticmethod
    def _scale_ts(ts: np.ndarray) -> np.ndarray:
        """TODO."""
        ts_scaled = sklearn.preprocessing.StandardScaler().fit_transform(
            ts.reshape(-1, 1)).ravel()

        return ts_scaled

    @staticmethod
    def _fit_model_ets_double(
        ts: np.ndarray,
        damped: bool = False
    ) -> statsmodels.tsa.holtwinters.HoltWintersResultsWrapper:
        """TODO."""
        model = statsmodels.tsa.holtwinters.ExponentialSmoothing(
            endog=ts, trend="additive", damped=damped, seasonal=None).fit()

        return model

    @staticmethod
    def _fit_model_ets_triple(
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
    def _fit_model_arch(
            ts: np.ndarray) -> arch.univariate.base.ARCHModelResult:
        model = arch.arch_model(ts).fit(disp="off")
        return model

    @classmethod
    def ft_ets_double_alpha(
        cls,
        ts: np.ndarray,
        damped: bool = False,
        ts_scaled: t.Optional[np.ndarray] = None,
        model_ets_double: t.Optional[
            statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None
    ) -> float:
        if ts_scaled is None:
            ts_scaled = cls._scale_ts(ts=ts)

        if model_ets_double is None:
            model_ets_double = cls._fit_model_ets_double(ts=ts_scaled,
                                                         damped=damped)

        alpha = model_ets_double.params_formatted["param"]["smoothing_level"]

        return alpha

    @classmethod
    def ft_ets_double_beta(
        cls,
        ts: np.ndarray,
        damped: bool = False,
        ts_scaled: t.Optional[np.ndarray] = None,
        model_ets_double: t.Optional[
            statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None,
    ) -> float:
        if ts_scaled is None:
            ts_scaled = cls._scale_ts(ts=ts)

        if model_ets_double is None:
            model_ets_double = cls._fit_model_ets_double(ts=ts_scaled,
                                                         damped=damped)

        beta = model_ets_double.params_formatted["param"]["smoothing_slope"]

        return beta

    @classmethod
    def ft_ets_triple_alpha(
        cls,
        ts: np.ndarray,
        ts_period: int,
        damped: bool = True,
        ts_scaled: t.Optional[np.ndarray] = None,
        model_ets_triple: t.Optional[
            statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None,
    ) -> float:
        if ts_scaled is None:
            ts_scaled = cls._scale_ts(ts=ts)

        if model_ets_triple is None:
            model_ets_triple = cls._fit_model_ets_triple(ts=ts_scaled,
                                                         ts_period=ts_period,
                                                         damped=damped)

        alpha = model_ets_triple.params_formatted["param"]["smoothing_level"]

        return alpha

    @classmethod
    def ft_ets_triple_beta(
        cls,
        ts: np.ndarray,
        ts_period: int,
        damped: bool = True,
        ts_scaled: t.Optional[np.ndarray] = None,
        model_ets_triple: t.Optional[
            statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None,
    ) -> float:
        if ts_scaled is None:
            ts_scaled = cls._scale_ts(ts=ts)

        if model_ets_triple is None:
            model_ets_triple = cls._fit_model_ets_triple(ts=ts_scaled,
                                                         ts_period=ts_period,
                                                         damped=damped)

        beta = model_ets_triple.params_formatted["param"]["smoothing_slope"]

        return beta

    @classmethod
    def ft_ets_triple_gamma(
        cls,
        ts: np.ndarray,
        ts_period: int,
        damped: bool = True,
        ts_scaled: t.Optional[np.ndarray] = None,
        model_ets_triple: t.Optional[
            statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None,
    ) -> float:
        if ts_scaled is None:
            ts_scaled = cls._scale_ts(ts=ts)

        if model_ets_triple is None:
            model_ets_triple = cls._fit_model_ets_triple(ts=ts_scaled,
                                                         ts_period=ts_period,
                                                         damped=damped)

        gamma = model_ets_triple.params_formatted["param"][
            "smoothing_seasonal"]

        return gamma

    @classmethod
    def ft_arch_adj_r_sqr(
        cls,
        ts: np.ndarray,
        ts_scaled: t.Optional[np.ndarray] = None,
        model_arch: t.Optional[arch.univariate.base.ARCHModelResult] = None
    ) -> float:
        """TODO."""
        if ts_scaled is None:
            ts_scaled = cls._scale_ts(ts=ts)

        if model_arch is None:
            model_arch = cls._fit_model_arch(ts=ts_scaled)

        return model_arch.rsquared_adj


def _test() -> None:
    ts = get_data.load_data(3)

    ts_period = data1_period.ts_period(ts)
    ts_trend, ts_season, ts_residuals = data1_detrend.decompose(
        ts, ts_period=ts_period)
    ts = ts.to_numpy()

    """
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
    """

    res = MFETSModelBased.ft_arch_adj_r_sqr(ts_residuals)
    print(res)


if __name__ == "__main__":
    _test()
