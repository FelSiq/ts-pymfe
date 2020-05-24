"""Functions associated to time-series surrogates."""
import typing as t

import numpy as np
import sklearn.metrics


def iaaft(ts: np.ndarray,
          max_iter: int = 128,
          atol: float = 1e-8,
          rtol: float = 1e-10,
          random_state: t.Optional[int] = None) -> np.ndarray:
    """Return iterative amplitude adjusted Fourier transform surrogates.

    Parameters
    ----------
    ts : :obj:`np.ndarray`
        One-dimensional time-series values.

    max_iter : int, optional (default=128)
        Maximum number of iterations allowed before the convergence of the
        IAAFT algorithm process.

    atol : float, optional (default=1e-8)
        Absolute value of tolerance to check convergence. The convergence
        criteria is the difference of the Root Mean Squared Error (RMSE),
        calculated between the current surrogate and the original time-series
        in the frequency domain, between two distinct iterations.

    rtol : float, optional (default=1e-8)
        Relative proportion of tolerance to check convergence.

    random_state : int, optional
        Random seed to ensure reproducibility.

    Returns
    -------
    :obj:`np.ndarray`
        Generated surrogate time-series using IAAFT algorithm.

    References
    ----------
    .. [1] Kugiumtzis, D.: Test your surrogate data before you test for
        nonlinearity, Phys. Rev. E, 60(3), 2808–2816, 1999.
    .. [2] Schreiber, T. and Schmitz, A.: Improved surrogate data for
        nonlinearity tests, Phys. Rev. Lett, 77, 635–638, 1996.
    .. [3] Schreiber, T. and Schmitz, A.: Surrogate time series,
        Physica D,142(3–4), 346–382, 2000.
    .. [4] Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., and Farmer,
        J.D.: Testing for nonlinearity in time series: the method of surrogate
        data, Physica D, 58, 77–94, 1992.
    .. [5] Theiler, J. and Prichard, D.: Constrained-realization Monte-Carlo
        method for hypothesis testing, Physica D, 94(4), 221–235, 1996.
    .. [6] "nolitsa" Python package, Adapted from the source code:
        https://github.com/manu-mannattil/nolitsa/blob/master/nolitsa/surrogates.py
    """
    ampl = np.abs(np.fft.rfft(ts))
    sort = np.sort(ts)
    err_prev, err_cur = -1.0, atol + 1

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
                        func: t.Callable[..., float],
                        max_iter: int = 128,
                        atol: float = 1e-8,
                        rtol: float = 1e-10,
                        random_state: t.Optional[np.ndarray] = None,
                        **kwargs) -> np.ndarray:
    """Extract a statistic from surrogate time-series.

    Parameters
    ----------
    ts : :obj:`np.ndarray`
        One-dimensional time-series values.

    surrogate_num : int, optional (default=32)
        Number of surrogate time-series.

    func : callable
        Callable that extract the desired statistic. This callable must
        receive an array of numeric values as the first argument, and
        return a single numeric value.

    max_iter : int, optional (default=128)
        Maximum number of iterations allowed before the convergence of the
        IAAFT algorithm process.

    atol : float, optional (default=1e-8)
        Absolute value of tolerance to check convergence. The convergence
        criteria is the difference of the Root Mean Squared Error (RMSE),
        calculated between the current surrogate and the original time-series
        in the frequency domain, between two distinct iterations.

    rtol : float, optional (default=1e-8)
        Relative proportion of tolerance to check convergence.

    random_state : int, optional
        Random seed to ensure reproducibility.

    kwargs:
        Extra arguments for the ``func`` callable.

    Returns
    -------
    :obj:`np.ndarray`
        Statistic extracted from distinct generated surrogate time-series using
        IAAFT algorithm.

    References
    ----------
    .. [1] Kugiumtzis, D.: Test your surrogate data before you test for
        nonlinearity, Phys. Rev. E, 60(3), 2808–2816, 1999.
    .. [2] Schreiber, T. and Schmitz, A.: Improved surrogate data for
        nonlinearity tests, Phys. Rev. Lett, 77, 635–638, 1996.
    .. [3] Schreiber, T. and Schmitz, A.: Surrogate time series,
        Physica D,142(3–4), 346–382, 2000.
    .. [4] Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., and Farmer,
        J.D.: Testing for nonlinearity in time series: the method of surrogate
        data, Physica D, 58, 77–94, 1992.
    .. [5] Theiler, J. and Prichard, D.: Constrained-realization Monte-Carlo
        method for hypothesis testing, Physica D, 94(4), 221–235, 1996.
    .. [6] "nolitsa" Python package, Adapted from the source code:
        https://github.com/manu-mannattil/nolitsa/blob/master/nolitsa/surrogates.py
    """
    stats = np.zeros(surrogate_num, dtype=float)

    for ind in np.arange(surrogate_num):
        if random_state is not None:
            # Note: changing the 'random_state' every iteration to generate
            # a distinct surrogate every iteration, but still mantaining the
            # determinism of the procedure.
            random_state += 1

        ts_surr = iaaft(ts=ts,
                        max_iter=max_iter,
                        random_state=random_state,
                        atol=atol,
                        rtol=rtol)

        stats[ind] = func(ts_surr, **kwargs)

    return stats
