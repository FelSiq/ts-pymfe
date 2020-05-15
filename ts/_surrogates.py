"""TODO."""
import typing as t

import numpy as np
import sklearn.metrics


def iaaft(ts: np.ndarray,
          max_iter: int = 128,
          atol: float = 1e-8,
          rtol: float = 1e-10,
          random_state: t.Optional[np.ndarray] = None) -> np.ndarray:
    """Return iterative amplitude adjusted Fourier transform surrogates.

    Notes
    -----
    Adapted from:
    https://github.com/manu-mannattil/nolitsa/blob/master/nolitsa/surrogates.py
    """
    ampl = np.abs(np.fft.rfft(ts))
    sort = np.sort(ts)
    err_prev, err_cur = -1, atol + 1

    if random_state is not None:
        np.random.seed(random_state)

    ts_fft = np.fft.rfft(np.random.permutation(ts))
    ts_sur = ts

    for _ in np.arange(max_iter):
        s = np.fft.irfft(ampl * ts_fft / np.abs(ts_fft), n=ts.size).real
        ts_sur = sort[np.argsort(np.argsort(s))]
        ts_fft = np.fft.rfft(ts_sur)
        err_cur = sklearn.metrics.mean_squared_error(ampl**2,
                                                     np.abs(ts_fft)**2,
                                                     squared=False)

        if abs(err_cur - err_prev) <= atol + rtol * abs(err_prev):
            break

        err_prev = err_cur

    return ts_sur


def apply_on_surrogates(ts: np.ndarray,
                        surrogate_num: int,
                        func: t.Callable[[np.ndarray], float],
                        max_iter: int = 128,
                        atol: float = 1e-8,
                        rtol: float = 1e-10,
                        random_state: t.Optional[np.ndarray] = None,
                        **kwargs) -> np.ndarray:
    """TODO."""
    stats = np.zeros(surrogate_num, dtype=float)

    for ind in np.arange(surrogate_num):
        if random_state is not None:
            random_state += 1

        ts_surr = iaaft(ts=ts, max_iter=max_iter, random_state=random_state)

        stats[ind] = func(ts_surr, **kwargs)

    return stats


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.random.seed(32)
    ts = np.random.randn(100) + 0.05 * np.arange(100)
    plt.plot(ts, label="time-series")
    plt.plot(iaaft(ts, random_state=16), label="surrogate")
    plt.show()
