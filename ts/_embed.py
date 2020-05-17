import typing as t

import numpy as np
import scipy.spatial

import _utils

try:
    import autocorr

except ImportError:
    pass

try:
    import info_theory

except ImportError:
    pass


def embed_ts(ts: np.ndarray,
             dim: int,
             lag: int = 1,
             include_val: bool = True) -> np.ndarray:
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

    if dim * lag > ts.size:
        raise ValueError("'dim * lag' ({}) can't be larger than the "
                         "time-series length ({}).".format(dim * lag, ts.size))

    if include_val:
        dim -= 1

    ts_emb = np.zeros((ts.size - dim * lag, dim + int(include_val)),
                      dtype=ts.dtype)

    shift_inds = lag * (dim - 1 - np.arange(-int(include_val), dim))

    for i in np.arange(ts_emb.shape[0]):
        ts_emb[i, :] = ts[i + shift_inds]

    return ts_emb


def nn(embed: np.ndarray,
       metric: str = "chebyshev",
       p: t.Union[int, float] = 2) -> np.ndarray:
    """TODO."""
    dist_mat = scipy.spatial.distance.cdist(embed, embed, metric=metric, p=p)

    # Note: prevent nearest neighbor be the instance itself, and also
    # be exact equal instances. We follow Cao's recommendation to pick
    # the next nearest neighbor when this happens.
    dist_mat[np.isclose(dist_mat, 0.0)] = np.inf

    nn_inds = np.argmin(dist_mat, axis=1)

    return nn_inds, dist_mat[nn_inds, np.arange(nn_inds.size)]


def embed_dim_fnn(ts: np.ndarray,
                  lag: int,
                  dims: t.Union[int, t.Sequence[int]] = 16,
                  rtol: t.Union[int, float] = 10,
                  atol: t.Union[int, float] = 2,
                  ts_scaled: t.Optional[np.ndarray] = None) -> int:
    """TODO.

    References
    ----------
    .. [1] Determining embedding dimension for phase-space reconstruction using
        a geometrical construction, Kennel, Matthew B. and Brown, Reggie and
        Abarbanel, Henry D. I., Phys. Rev. A, volume 45, 1992, American
        Physical Society.
    """
    if lag <= 0:
        raise ValueError("'lag' must be positive (got {}).".format(lag))

    if np.isscalar(dims):
        dims = np.arange(1, dims + 1)

    ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

    fnn_prop = np.zeros(len(dims), dtype=float)

    ts_std = np.std(ts)

    for ind, dim in enumerate(dims):
        try:
            emb_next = embed_ts(ts=ts_scaled, lag=lag, dim=dim + 1)
            emb_cur = emb_next[:, 1:]

        except ValueError:
            # Note: no need to explore further since all embeds larger than
            # the current dimension will also fail.
            ed[ind:] = np.nan
            ed_star[ind:] = np.nan
            break

        nn_inds, dist_cur = nn(embed=emb_cur)

        emb_next_abs_diff = np.abs(emb_next[:, 0] - emb_next[nn_inds, 0])
        dist_next = np.maximum(dist_cur, emb_next_abs_diff)

        crit_1 = emb_next_abs_diff > rtol * dist_cur
        crit_2 = dist_next > atol * ts_std

        fnn_prop[ind] = np.mean(np.logical_or(crit_1, crit_2))

    return fnn_prop


def embed_dim_cao(ts: np.ndarray,
                  lag: int,
                  dims: t.Union[int, t.Sequence[int]] = 16,
                  ts_scaled: t.Optional[np.ndarray] = None) -> int:
    """TODO.

    References
    ----------
    .. [1] Liangyue Cao, Practical method for determining the minimum
        embedding dimension of a scalar time series, Physica D: Nonlinear
        Phenomena, Volume 110, Issues 1–2, 1997, Pages 43-50,
        ISSN 0167-2789, https://doi.org/10.1016/S0167-2789(97)00118-8.
    """
    if lag <= 0:
        raise ValueError("'lag' must be positive (got {}).".format(lag))

    if np.isscalar(dims):
        dims = np.arange(1, dims + 1)

    ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

    ed, ed_star = np.zeros((2, len(dims)), dtype=float)

    for ind, dim in enumerate(dims):
        try:
            emb_next = embed_ts(ts=ts_scaled, lag=lag, dim=dim + 1)
            emb_cur = emb_next[:, 1:]

        except ValueError:
            # Note: no need to explore further since all embeds larger than
            # the current dimension will also fail.
            ed[ind:] = np.nan
            ed_star[ind:] = np.nan
            break

        nn_inds, dist_cur = nn(embed=emb_cur)

        emb_next_abs_diff = np.abs(emb_next[:, 0] - emb_next[nn_inds, 0])
        # Note: 'chebyshev'/'manhattan'/'L1'/max norm distance of X and Y,
        # both in the embed of (d + 1) dimensions, can be defined in respect
        # to one dimension less:
        # L1(X_{d+1}, Y_{d+1}) = |X_{d+1}, Y_{d+1}|_{inf}
        #   = max(|x_1 - y_1|, ..., |x_{d+1} - y_{d+1}|)
        #   = max(max(|x_1 - y_1|, ..., |x_d - y_d|), |x_{d+1} - y_{d+1}|)
        #   = max(L1(X_{d}, Y_{d}), |x_{d+1} - y_{d+1}|)
        dist_next = np.maximum(dist_cur, emb_next_abs_diff)

        # Note: 'ed' and 'ed_star' refers to, respectively, E_{d} and
        # E^{*}_{d} from the Cao's paper.
        ed[ind] = np.mean(dist_next / dist_cur)
        ed_star[ind] = np.mean(emb_next_abs_diff)

    # Note: the minimum embedding dimension is D such that e1[D]
    # is the first index where e1 stops changing significantly.
    e1 = ed[1:] / ed[:-1]

    # Note: This is the E2(d) Cao's metric. Its purpose is to
    # separate random time-series. For random-generated time-
    # series, e2 will be 1 for any dimension. For deterministic
    # data, however, e2 != 1 for some d.
    e2 = ed_star[1:] / ed_star[:-1]

    return e1, e2


def embed_lag(ts: np.ndarray,
              lag: t.Optional[t.Union[str, int]] = None,
              default_lag: int = 1,
              max_nlags: t.Optional[int] = None,
              ts_acfs: t.Optional[np.ndarray] = None,
              ts_ami: t.Optional[np.ndarray] = None,
              **kwargs) -> int:
    """TODO."""
    VALID_OPTIONS = {
        "ami": info_theory.MFETSInfoTheory.ft_ami_first_critpt,
        "acf": autocorr.MFETSAutocorr.ft_acf_first_nonpos,
        "acf-nonsig": autocorr.MFETSAutocorr.ft_acf_first_nonsig,
    }

    if lag is None:
        lag = "acf"

    if isinstance(lag, str):
        if lag not in VALID_OPTIONS:
            raise ValueError("'lag' must be in {} (got '{}')."
                             "".format(VALID_OPTIONS.keys(), lag))

        if max_nlags is None:
            max_nlags = ts.size // 2

        if lag == "ami":
            kwargs["ts_ami"] = ts_ami

        else:
            kwargs["ts_acfs"] = ts_acfs

        kwargs["max_nlags"] = max_nlags

        lag = VALID_OPTIONS[lag](ts, **kwargs)

        return default_lag if np.isnan(lag) else lag

    if np.isscalar(lag):
        lag = int(lag)

        if lag <= 0:
            raise ValueError("'lag' must be positive (got {}).".format(lag))

        return lag

    raise TypeError("'lag' type must be a scalar, a string or None (got {})."
                    "".format(type(lag)))


def _test() -> None:
    ts = np.arange(10)
    print(embed_ts(ts, dim=2, lag=1))


if __name__ == "__main__":
    _test()
