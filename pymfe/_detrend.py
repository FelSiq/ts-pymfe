"""Time-series decomposition and detrending functions."""
import typing as t

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.stattools
import statsmodels.tsa.stattools
import statsmodels.tsa.seasonal
import pandas as pd
import supersmoother

try:
    import pymfe._period as _period

except ImportError:
    pass


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

    comp_trend = model.predict(timestamp)
    comp_resid = ts - comp_trend
    comp_season = np.zeros(ts.size, dtype=float)

    if plot:
        fig, (ax_raw, ax_trend, ax_resid) = plt.subplots(3, 1)

        fig.suptitle("Friedman's Super Smoother results")

        ax_raw.plot(ts)
        ax_raw.set_title("Original time-series")

        ax_trend.plot(comp_trend)
        ax_trend.set_title("Trend component")

        ax_resid.plot(comp_resid)
        ax_resid.set_title("Residual component")

        plt.show()

    return comp_trend, comp_season, comp_resid


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
    components = statsmodels.tsa.seasonal.STL(ts, period=ts_period).fit()

    if plot:
        components.plot()
        plt.show()

    return components.trend, components.seasonal, components.resid


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
        ts_period = _period.get_ts_period(ts, ts_detrended=ssmoother_comps[2])

    if ts_period <= 1:
        if ssmoother_comps is None:
            ssmoother_comps = _decompose_ssmoother(ts=ts, plot=plot)

        components = ssmoother_comps

    else:
        components = _decompose_stl(ts=ts, ts_period=ts_period, plot=plot)

    comp_trend, comp_season, comp_resid = components

    if isinstance(comp_trend, pd.core.series.Series):
        comp_trend = comp_trend.values

    if isinstance(comp_season, pd.core.series.Series):
        comp_season = comp_season.values

    if isinstance(comp_resid, pd.core.series.Series):
        comp_resid = comp_resid.values

    return comp_trend, comp_season, comp_resid
