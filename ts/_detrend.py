import typing as t

import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.linear_model
import statsmodels.stats.stattools
import statsmodels.tsa.stattools
import statsmodels.tsa.seasonal
import pandas as pd
import supersmoother

import _get_data


def detrend(ts: np.ndarray,
            degrees: t.Union[int, t.Sequence[int]] = (1, 2, 3),
            sig_level: t.Optional[float] = None,
            plot: bool = False,
            verbose: bool = False) -> np.ndarray:
    """Detrend a time series with a polynomial regression for each ``degree``."""
    if plot and sig_level is not None:
        raise ValueError("Can't 'plot' with 'sig_level' given.")

    if isinstance(degrees, int):
        degrees = [degrees]

    t = np.arange(ts.size).reshape(-1, 1)

    res = np.zeros((len(degrees), 2 * ts.size))

    for i, deg in enumerate(degrees):
        pip = sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.PolynomialFeatures(deg),
            sklearn.linear_model.Ridge())

        pip.fit(t, ts)
        trend_pred = pip.predict(t)
        residuals = ts - trend_pred

        if verbose or sig_level is not None:
            adfuller_res = statsmodels.tsa.stattools.adfuller(residuals)
            adfuller_pval = adfuller_res[1]

        if verbose:
            print(f"Durbin-Watson test for degree {deg}:",
                  statsmodels.stats.stattools.durbin_watson(residuals))

            print(f"Augmented Dickey-Fuller test for degree {deg}:",
                  adfuller_res)

        if sig_level is not None and adfuller_pval <= sig_level:
            return residuals, trend_pred

        if plot:
            plt.subplot(2, 2, deg + 1)
            plt.title(f"Detrended w/ degree {deg}")
            plt.plot(t, residuals)

        res[i, :] = np.hstack((residuals, trend_pred))

    if plot:
        plt.subplot(221)
        plt.title("With trend")
        plt.plot(t, ts)

        plt.show()

    if sig_level is not None:
        raise RuntimeError("Can't detrend series. Please choose "
                           "more degrees.")

    if len(degrees) == 1:
        return np.split(res[0, :], 2)

    return np.split(res, 2, axis=1)


def _decompose_ssmoother(
        ts: t.Union[np.ndarray, pd.core.series.Series],
        plot: bool = False) -> t.Tuple[t.Optional[np.ndarray], ...]:
    """TODO."""
    timestamp = np.arange(ts.size)
    model = supersmoother.SuperSmoother().fit(timestamp, ts, presorted=True)
    trend = model.predict(timestamp)
    residual = ts - trend

    return trend, np.zeros(ts.size, dtype=float), residual


def _decompose_stl(ts: t.Union[np.ndarray, pd.core.series.Series],
                   ts_period: t.Optional[int] = None,
                   plot: bool = False) -> t.Tuple[np.ndarray, ...]:
    """Decompose a time-series in trend, seasonality and residuals."""
    res = statsmodels.tsa.seasonal.STL(ts, period=ts_period).fit()

    if plot:
        res.plot()
        plt.show()

    if isinstance(ts, pd.core.series.Series):
        return res.trend.values, res.seasonal.values, res.resid.values

    return res.trend, res.seasonal, res.resid


def decompose(ts: t.Union[np.ndarray, pd.core.series.Series],
              ts_period: t.Optional[int] = None,
              plot: bool = False) -> t.Tuple[t.Optional[np.ndarray], ...]:
    """TODO."""
    if ts_period <= 1:
        return _decompose_ssmoother(ts=ts, plot=plot)

    return _decompose_stl(ts=ts, ts_period=ts_period, plot=plot)


def _test() -> None:
    detrend(_get_data.load_data(), plot=True)


if __name__ == "__main__":
    _test()
