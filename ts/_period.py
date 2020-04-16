import typing as t

import numpy as np
import statsmodels.tsa.stattools

import _detrend


def ts_period(ts: np.ndarray,
              ts_detrended: t.Optional[np.ndarray] = None) -> int:
    """TODO."""
    if ts_detrended is None:
        ts_detrended = _detrend.decompose(ts=ts, ts_period=0)[2]

    autocorr = statsmodels.tsa.stattools.acf(ts_detrended,
                                             nlags=1 + ts_detrended.size // 2,
                                             fft=True,
                                             unbiased=True)[1:]

    period = np.argmax(autocorr) + 1

    return period


def _test() -> None:
    import get_data
    ts = get_data.load_data(3)
    print(ts_period(ts))


if __name__ == "__main__":
    _test()
