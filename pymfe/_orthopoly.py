"""Orthogonal polynomial in concordance with R `poly` implementation."""
import numpy as np


def ortho_poly(ts: np.ndarray,
               degree: int = 1,
               return_coeffs: bool = True,
               check_degree: bool = True) -> np.ndarray:
    """Fit a orthogonal polynomial with given ``degree``.

    Compatible with the R implementation ``poly``.
    """
    if check_degree and degree >= np.unique(ts).size:
        raise ValueError("'degree' must be less than number "
                         "of unique points.")

    ts_mean = np.mean(ts)
    ts = ts - ts_mean
    ts_pol = np.fliplr(np.vander(x=ts, N=degree + 1))

    mat_q, mat_r = np.linalg.qr(ts_pol)

    raw = mat_q * np.diag(mat_r)
    raw_sqr = np.square(raw)

    sum_raw_sqr = np.sum(raw_sqr, axis=0)

    res = (raw / np.sqrt(sum_raw_sqr))[:, 1:]

    if return_coeffs:
        alpha = (np.sum(raw_sqr * ts.reshape(-1, 1), axis=0) / sum_raw_sqr +
                 ts_mean)[:degree]

        return res, alpha, np.hstack((1, sum_raw_sqr))

    return res
