import typing as t

import numpy as np


def ts_period(ts: np.ndarray,
              ts_residuals: t.Optional[np.ndarray] = None) -> int:
    """TODO."""
    max_corr = -np.inf
    period = -1

    for cur_period in np.arange(1, 1 + ts.size // 2):
        cf = np.corrcoef(ts[cur_period:], ts[:-cur_period])[0, 1]
        if cf > max_corr:
            max_corr = cf
            period = cur_period

    return period


def _test() -> None:
    import get_data
    ts = get_data.load_data(3)
    print(ts_period(ts))


if __name__ == "__main__":
    _test()
