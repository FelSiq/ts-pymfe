"""Scoring module or forecasting models."""
import collections

import numpy as np
import sklearn.metrics


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the MSE (Mean Squared Error)."""
    return sklearn.metrics.mean_squared_error(y_true, y_pred)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the RMSE (Root Mean Squared Error)."""
    return sklearn.metrics.mean_squared_error(y_true, y_pred, squared=False)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the MAE (Mean Absolute Error)."""
    return sklearn.metrics.mean_absolute_error(y_true, y_pred)


def smape(arr_a: np.ndarray,
          arr_b: np.ndarray,
          percentage: bool = False,
          half_denom: bool = True) -> float:
    """Calculate SMAPE (Symmetric Mean Absolute Percentage Error).

    Parameters
    ----------
    arr_a, arr_b : :obj:`np.ndarray`
        Arrays to calculate the SMAPE from.

    percentage : bool, optional
        If True, multiply the result by 100 (i.e., return the
        percentage value, not the fraction).

    half_denom : bool, optional
        If True, divide the denominator by 2. If False, the
        result if bounded by [0, 100] (if ``percentage`` is
        True), or by [0, 1] otherwise.

    Returns
    -------
    float
        SMAPE estimation between the two given arrays. If
        ``percentage`` is True, then return the estimation in
        the percentage form (in [0, 100] range). Return the
        error in fraction form (in [0, 1] range) otherwise.
    """
    res = np.mean(
        np.abs(arr_a - arr_b) / (1e-9 + np.abs(arr_a) + np.abs(arr_b)))

    if percentage:
        res *= 100

    if not half_denom:
        res *= 2

    return res


VALID_SCORING = collections.OrderedDict((
    ("mse", mse),
    ("rmse", rmse),
    ("mae", mae),
    ("smape", smape),
))
