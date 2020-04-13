import typing as t

import numpy as np


def embed_ts(ts: np.ndarray,
             dim: int,
             lag: int = 1,
             include_val: bool = False) -> np.ndarray:
    """Embbed a time-series in dimension ``dim``.

    Arguments
    ---------
    ts : :obj:`np.ndarray`, shape: (ts.size,)
        One-dimensional time-series.

    dim : int
        Dimension of the embedding.

    lag : int, optional (default = 1)
        Lag of the time-series.

    include_val : bool, optional (default = False)
        Include the value itself on its own embedding.

    Returns
    -------
    :obj:`np.ndarray`, shape: (ts.size - dim * lag, dim)
        Embbed time-series.
    """
    if dim <= 0:
        raise ValueError("'dim' must be positive (got {}).".format(dim))

    if lag <= 0:
        raise ValueError("'lag' must be positive (got {}).".format(lag))

    if dim * lag >= ts.size:
        raise ValueError("'dim * lag' ({}) must be smaller than "
                         "the time-series length ({}).".format(
                             dim * lag, ts.size))

    ts_emb = np.zeros((ts.size - dim * lag, dim + int(include_val)),
                      dtype=ts.dtype)

    shift_inds = lag * (dim - 1 - np.arange(-int(include_val), dim))

    for i in np.arange(ts_emb.shape[0]):
        ts_emb[i, :] = ts[i + shift_inds]

    return ts_emb


def _test() -> None:
    ts = np.arange(10)
    print(embed_ts(ts, dim=2, lag=1))


if __name__ == "__main__":
    _test()
