"""Calculate the time-series periodicity."""
import typing as t

import numpy as np
import statsmodels.tsa.stattools

try:
    import pymfe._detrend as _detrend

except ImportError:
    pass


def get_ts_period(ts: np.ndarray,
                  ts_detrended: t.Optional[np.ndarray] = None,
                  ts_period: t.Optional[int] = None) -> int:
    """Return the time-series periodicity, if any.

    The time-series is detrended first using the Friedman's Super Smoother
    (if ``ts_detrended`` is None).

    It is calculated the autocorrelation of the time-series up to
    floor(ts.size / 2), using the fast-fourier transform method.

    The time-series period is the argument where the autocorrelation function
    assumed maximal absolute value.
    """
    if ts_period is not None:
        return max(int(ts_period), 1)

    if ts.size <= 1:
        return 1

    if ts_detrended is None:
        ts_detrended = _detrend.decompose(ts=ts, ts_period=0)[2]

    autocorr = statsmodels.tsa.stattools.acf(ts_detrended,
                                             nlags=ts_detrended.size // 2,
                                             fft=True,
                                             unbiased=True)[1:]

    period = np.argmax(np.abs(autocorr)) + 1

    return period
