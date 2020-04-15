import typing as t

import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import scipy.stats
import numpy as np

import _detrend
import _get_data


class MFETSDiversity:
    pass


def _test() -> None:
    ts = _get_data.load_data(2)
    ts_trend, ts_season, ts_residuals = _detrend.decompose(ts)
    ts = ts.to_numpy()


if __name__ == "__main__":
    _test()
