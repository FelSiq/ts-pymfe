"""Time-series decomposition and detrending functions."""
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

try:
    import _period

except ImportError:
    pass


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


def _decompose_ssmoother(ts: t.Union[np.ndarray, pd.core.series.Series],
                         plot: bool = False) -> t.Tuple[np.ndarray, ...]:
    """Time-series decomposition using Friedman's Super Smoother.

    The seasonal component returned is an array full of zeros with the same
    length as the given time-series.

    Parameters
    ----------
    ts : :obj:`np.ndarray` or :obj:`pd.core.series.Series`
        One-dimensional time-series values.

    plot : bool, optional (default=False)
        If True, plot the decomposed components.

    Returns
    -------
    tuple of :obj:`np.ndarray`
        A tuple with tree components, in the following order: time-series
        trend component, an array full of zeros (seasonal component), and
        the residual component.

    References
    ----------
    .. [1] Friedman, J. H. 1984, A variable span scatterplot smoother
        Laboratory for Computational Statistics, Stanford University
        Technical Report No. 5. Available at:
        https://www.slac.stanford.edu/pubs/slacpubs/3250/slac-pub-3477.pdf
        Accessed on May 12 2020.
    """
    timestamp = np.arange(ts.size)
    model = supersmoother.SuperSmoother().fit(timestamp, ts, presorted=True)
    trend = model.predict(timestamp)
    residual = ts - trend

    if plot:
        fig, (ax_raw, ax_trend, ax_resid) = plt.subplots(3, 1)

        fig.suptitle("Friedman's Super Smoother results")

        ax_raw.plot(ts)
        ax_raw.set_title("Original time-series")

        ax_trend.plot(trend)
        ax_trend.set_title("Trend component")

        ax_resid.plot(residual)
        ax_resid.set_title("Residual component")

        plt.show()

    return trend, np.zeros(ts.size, dtype=float), residual


def _decompose_stl(ts: t.Union[np.ndarray, pd.core.series.Series],
                   ts_period: t.Optional[int] = None,
                   plot: bool = False) -> t.Tuple[np.ndarray, ...]:
    """Decompose a time-series in STL decomposition.

    STL stands for `Seasonal-Trend Decomposition Procedure Based on Loess`.

    Parameters
    ----------
    ts : :obj:`np.ndarray` or :obj:`pd.core.series.Series`
        One-dimensional time-series values.

    ts_period : int, optional
        Time-series period.

    plot : bool, optional (default=False)
        If True, plot the decomposed components.

    Returns
    -------
    tuple of :obj:`np.ndarray`
        A tuple with tree components, in the following order: time-series
        trend component, the seasonal component, and the residual component.

    References
    ----------
    .. [1] Cleveland, R. B., Cleveland, W. S., McRae, J. E. & Terpenning, I.
        (1990). STL: A Seasonal-Trend Decomposition Procedure Based on
        Loess (with Discussion). Journal of Official Statistics, 6, 3--73.
    """
    res = statsmodels.tsa.seasonal.STL(ts, period=ts_period).fit()

    if plot:
        res.plot()
        plt.show()

    if isinstance(ts, pd.core.series.Series):
        return res.trend.values, res.seasonal.values, res.resid.values

    return res.trend, res.seasonal, res.resid


def decompose(ts: t.Union[np.ndarray, pd.core.series.Series],
              ts_period: t.Optional[int] = None,
              plot: bool = False) -> t.Tuple[np.ndarray, ...]:
    """Decompose a time-series into separated additive components.

    If the time-series is seasonal (period > 1), then it is used the STL
    (Seasonal-Trend Decomposition Procedure Based on Loess) algorithm.
    Otherwise (period <= 1), it is used the Friedman's Super Smoother
    algorithm.

    Parameters
    ----------
    ts : :obj:`np.ndarray` or :obj:`pd.core.series.Series`
        One-dimensional time-series values.

    ts_period : int, optional
        Time-series period. If not given, it is estimated using the minima in
        the autocorrelation function from the detrended time-series using
        Friedman's Super Smoother. If the estimated lag is less or equal 1,
        then it is simply returned the previously decomposed version.
        Otherwise, it is used STL decomposition afterwards in the original
        time-series.

    plot : bool, optional (default=False)
        If True, plot the decomposed components.

    References
    ----------
    .. [1] Friedman, J. H. 1984, A variable span scatterplot smoother
        Laboratory for Computational Statistics, Stanford University
        Technical Report No. 5. Available at:
        https://www.slac.stanford.edu/pubs/slacpubs/3250/slac-pub-3477.pdf
        Accessed on May 12 2020.
    .. [2] Cleveland, R. B., Cleveland, W. S., McRae, J. E. & Terpenning, I.
        (1990). STL: A Seasonal-Trend Decomposition Procedure Based on
        Loess (with Discussion). Journal of Official Statistics, 6, 3--73.
    """
    ssmoother_comps = None

    if ts_period is None:
        ssmoother_comps = _decompose_ssmoother(ts=ts)
        ts_period = _period.ts_period(ts, ts_detrended=ssmoother_comps[0])

    if ts_period <= 1:
        if ssmoother_comps is not None:
            return ssmoother_comps

        return _decompose_ssmoother(ts=ts, plot=plot)

    return _decompose_stl(ts=ts, ts_period=ts_period, plot=plot)


def _test() -> None:
    detrend(_get_data.load_data(3), plot=True)


if __name__ == "__main__":
    _test()
