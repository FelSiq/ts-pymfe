import typing as t

import statsmodels.tsa.holtwinters
import numpy as np

import get_data
import data1_period
import data1_detrend


class MFETSModelBased:
    @classmethod
    def ft_ets_double(cls, ts: np.ndarray, ts_period: int, damped: bool = False) -> np.ndarray:
        model = statsmodels.tsa.holtwinters.ExponentialSmoothing(
            endog=ts,
            trend="additive",
            damped=damped,
            seasonal=None).fit()

        params = model.params_formatted["param"]

        alpha = params["smoothing_level"]
        beta = params["smoothing_slope"]

        return alpha, beta

    @classmethod
    def ft_ets_triple(cls, ts: np.ndarray, ts_period: int, damped: bool = True) -> np.ndarray:
        model = statsmodels.tsa.holtwinters.ExponentialSmoothing(
            endog=ts,
            trend="additive",
            seasonal="additive",
            damped=damped,
            seasonal_periods=ts_period).fit()

        params = model.params_formatted["param"]

        alpha = params["smoothing_level"]
        beta = params["smoothing_slope"]
        gamma = params["smoothing_seasonal"]

        return alpha, beta, gamma


def _test() -> None:
    ts = get_data.load_data(3)

    ts_period = data1_period.ts_period(ts)
    ts_trend, ts_season, ts_residuals = data1_detrend.decompose(
        ts, ts_period=ts_period)
    ts = ts.to_numpy()
    
    res = MFETSModelBased.ft_ets_double(ts, ts_period=ts_period)
    print(res)

    res = MFETSModelBased.ft_ets_triple(ts, ts_period=ts_period)
    print(res)


if __name__ == "__main__":
    _test()
