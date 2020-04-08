import typing as t

import statsmodels.tsa.holtwinters
import numpy as np

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

        if "model_ets_double" not in kwargs:
            model = cls._fit_model_ets_double(ts=ts, damped=damped)
            precomp_vals["model_ets_double"] = model

            model = cls._fit_model_ets_triple(ts=ts, damped=damped)
            precomp_vals["model_ets_triple"] = model

        return precomp_vals

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

    @classmethod
    def ft_ets_double_alpha(
        cls,
        ts: np.ndarray,
        damped: bool = False,
        model_ets_double: t.Optional[
            statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None
    ) -> float:
        if model_ets_double is None:
            model_ets_double = cls._fit_model_ets_double(ts=ts, damped=damped)

        alpha = model_ets_double.params_formatted["param"]["smoothing_level"]

        return alpha

    @classmethod
    def ft_ets_double_beta(
        cls,
        ts: np.ndarray,
        damped: bool = False,
        model_ets_double: t.Optional[
            statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None,
    ) -> float:
        if model_ets_double is None:
            model_ets_double = cls._fit_model_ets_double(ts=ts, damped=damped)

        beta = model_ets_double.params_formatted["param"]["smoothing_slope"]

        return beta

    @classmethod
    def ft_ets_triple_alpha(
        cls,
        ts: np.ndarray,
        ts_period: int,
        damped: bool = True,
        model_ets_triple: t.Optional[
            statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None,
    ) -> float:
        if model_ets_triple is None:
            model_ets_triple = cls._fit_model_ets_triple(ts=ts,
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
        model_ets_triple: t.Optional[
            statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None,
    ) -> float:
        if model_ets_triple is None:
            model_ets_triple = cls._fit_model_ets_triple(ts=ts,
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
        model_ets_triple: t.Optional[
            statsmodels.tsa.holtwinters.HoltWintersResultsWrapper] = None,
    ) -> float:
        if model_ets_triple is None:
            model_ets_triple = cls._fit_model_ets_triple(ts=ts,
                                                         ts_period=ts_period,
                                                         damped=damped)

        gamma = model_ets_triple.params_formatted["param"][
            "smoothing_seasonal"]

        return gamma


def _test() -> None:
    ts = get_data.load_data(3)

    ts_period = data1_period.ts_period(ts)
    ts_trend, ts_season, ts_residuals = data1_detrend.decompose(
        ts, ts_period=ts_period)
    ts = ts.to_numpy()

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
