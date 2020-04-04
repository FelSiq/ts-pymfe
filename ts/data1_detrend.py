import typing as t

import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.linear_model
import statsmodels.stats.stattools
import statsmodels.tsa.stattools
import pandas as pd

import get_data


def detrend(y: np.ndarray,
            degrees: t.Union[int, t.Sequence[int]] = (1, 2, 3),
            plot: bool = False) -> np.ndarray:
    """Detrend a time series with a polynomial regression for each ``degree``."""
    if isinstance(degrees, int):
        degrees = [degrees]

    t = np.arange(y.size).reshape(-1, 1)

    res = np.zeros((len(degrees), 2 * y.size))
    
    for i, deg in enumerate(degrees):
        pip = sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.PolynomialFeatures(deg),
            sklearn.linear_model.Ridge())
    
        pip.fit(t, y)
        y_pred = pip.predict(t)
        residuals = y - y_pred
    
        print(
            f"Durbin-Watson test for degree {deg}:", 
            statsmodels.stats.stattools.durbin_watson(residuals))

        print(
            f"Augmented Dickey-Fuller test for degree {deg}:",
            statsmodels.tsa.stattools.adfuller(residuals))

        if plot:
            plt.subplot(2, 2, deg + 1)
            plt.title(f"Detrended w/ degree {deg}")
            plt.plot(t, residuals)

        res[i, :] = np.hstack((residuals, y_pred))

    if plot:
        plt.subplot(221)
        plt.title("With trend")
        plt.plot(t, y)

        plt.show()

    if len(degrees) == 1:
        return np.split(res[0, :], 2)

    return np.split(res, 2, axis=1)


def decompose(ts: np.ndarray, period: t.Optional[int] = None) -> np.ndarray:
    """Decompose a time-series in trend, seasonality and residuals."""
    res = statsmodels.tsa.seasonal.STL(ts, period=period).fit()
    return res.trend, res.seasonal, res.resid


def _test() -> None:
    detrend(get_data.load_data(), plot=True)


if __name__ == "__main__":
    _test()
