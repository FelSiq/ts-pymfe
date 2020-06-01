"""Time-series embedding functions."""
import typing as t

import numpy as np
import scipy.spatial

try:
    import pymfe._utils as _utils

except ImportError:
    pass

try:
    import pymfe.autocorr as autocorr

except ImportError:
    pass

try:
    import pymfe.info_theory as info_theory

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

    lag : int, optional (default=1)
        Lag of the time-series.

    include_val : bool, optional (default=False)
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
    """Return the Nearest neighbor of each embedded time-series observation."""
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
    """Estimation of the False Nearest Neighbors proportion for each dimension.

    The False Nearest Neighbors calculates the average number of false nearest
    neighbors of each time-series observations, given a fixed embedding
    dimension.

    A false nearest neighbors are a pair of instances that are farther apart
    in the appropriate embedding dimension, but close together in a smaller
    dimension simply because both are projected in a innapropriate dimension.
    Sure enough, we could have just use a `sufficiently large` embedding
    dimension to remove all possibility of false nearest neighbors. However,
    this strategy may imply in a lack of computational effciency, and all
    statistical concerns that may arise in high dimensional data analysis.
    The idea behind of analysing the proportion of false neighbors is to
    estimate the minimum embedding dimension that makes only true neighbors
    be close together in that given space.

    Thus, it is expected that, given the appropriate embedding dimension, the
    proportion of false neighbors will be close to zero.

    Differently from the reference paper, here we are using the Chebyshev
    distance (or maximum norm distance) rather than the Euclidean distance.

    Parameters
    ----------
    ts : :obj:`np.ndarray`
        One-dimensional time-series values.

    lag : int
        Embedding lag. You may want to check the `embed_lag` function
        documentation for embedding lag estimation. Must be a stricly
        positive value.

    dims : int or sequence of int
        Dimensions to estimate the Cao's `E1` and `E2` statistic values.
        If integer, estimate all dimensions from 1 up to the given number.
        If a sequence of integers, estimate the FNN proportion for all
        given dimensions, and return the corresponding values in the same
        order of the given dimensions.
        All dimensions with non-positive values will receive a `np.nan`.

    rtol : float, optional (default=10)
        Relative tolerance between the relative difference of the distances
        between each observation and its nearest neighbor $D_{d}$ in a given
        dimension $d$, and the distance $D_{d+1}$ of the observation and the
        same nearest neighbor in the next embedding dimension. It is used in
        the first criteria from the reference paper to define which instances
        are false neighbors. The default value (10) is the recommended value
        from the original paper, and it means that nearest neighbors that are
        ten times farther in the next dimension relative to the distance in
        the current dimension are considered false nearest neighbors.

    atol : float, optional (default=2)
        Number of time-series standard deviations that an observation and
        its nearest neighbor must be in the next dimension in order to be
        considered false neighbors. This is the reference paper's second
        criteria.

    ts_scaled : :obj:`np.ndarray`, optional
        Standardized time-series values. Used to take advantage of
        precomputations.

    Returns
    -------
    :obj:`np.ndarray`
        Proportion of false nearest neighbos for each given dimension. It is
        used the union of both criterium to determine whether a pair of
        neighbors are false neighbors in a fixed embedding dimension (i.e.,
        any pair of neighbors considered false in either of the criterium
        alone are considered false).

    References
    ----------
    .. [1] Determining embedding dimension for phase-space reconstruction using
        a geometrical construction, Kennel, Matthew B. and Brown, Reggie and
        Abarbanel, Henry D. I., Phys. Rev. A, volume 45, 1992, American
        Physical Society.
    """
    if lag <= 0:
        raise ValueError("'lag' must be positive (got {}).".format(lag))

    _dims: t.Sequence[int]

    if np.isscalar(dims):
        _dims = np.arange(1, int(dims) + 1)  # type: ignore

    else:
        _dims = np.asarray(dims, dtype=int)

    ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

    fnn_prop = np.zeros(len(_dims), dtype=float)

    # Note: since we are using the standardized time-series, its standard
    # deviation is always 1. However, we keep this variable to make clear
    # the correspondence between the reference paper's formulas and what
    # are programmed here.
    ts_std = 1.0  # = np.std(ts_scaled)

    for ind, dim in enumerate(_dims):
        try:
            emb_next = embed_ts(ts=ts_scaled, lag=lag, dim=dim + 1)
            emb_cur = emb_next[:, 1:]

        except ValueError:
            fnn_prop[ind] = np.nan
            continue

        nn_inds, dist_cur = nn(embed=emb_cur)

        emb_next_abs_diff = np.abs(emb_next[:, 0] - emb_next[nn_inds, 0])
        dist_next = np.maximum(dist_cur, emb_next_abs_diff)

        # Note: in the reference paper, there were three criteria for
        # determining what is a False Nearest Neighbor. The first and second
        # one are, respectively, related to the `crit_1` and `crit_2`
        # variables. The third criteria is the union of the criteria, which
        # means that the observation is considered a False Neighbor if either
        # criteria accuses it as such. Here, we are using the third and
        # therefore the most conservative criteria.
        crit_1 = emb_next_abs_diff > rtol * dist_cur
        crit_2 = dist_next > atol * ts_std

        fnn_prop[ind] = np.mean(np.logical_or(crit_1, crit_2))

    return fnn_prop


def embed_dim_cao(
        ts: np.ndarray,
        lag: int,
        dims: t.Union[int, t.Sequence[int]] = 16,
        ts_scaled: t.Optional[np.ndarray] = None,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Estimate Cao's metrics to estimate time-series embedding dimension.

    The Cao's metrics are two statistics, `E1` and `E2`, used to estimate the
    appropriate embedding metric of a time-series. From the `E1` statistic it
    can be defined the appropriate embedding dimension as the index after the
    saturation of the metric from a set of ordered lags.

    The precise `saturation` concept may be a subjective concept, since this
    metric can show some curious `artifacts` related to specific lags for
    specific time-series, which will need deeper further investigation.

    The `E2` statistics is to detect `false positives` from the `E1` statistic
    since if is used to distinguish between random white noise and a process
    generated from a true, non completely random, underlying process. If the
    time-series is purely random white noise, then all values of `E2` will be
    close to 1. If there exists a dimension with the `E2` metric estimated
    `sufficiently far` from 1, then this series is considered not a white
    random noise.

    Parameters
    ----------
    ts : :obj:`np.ndarray`
        One-dimensional time-series values.

    lag : int
        Embedding lag. You may want to check the `embed_lag` function
        documentation for embedding lag estimation. Must be a stricly
        positive value.

    dims : int or sequence of int
        Dimensions to estimate the Cao's `E1` and `E2` statistic values.
        If integer, estimate all dimensions from 1 up to the given number.
        If a sequence of integers, estimate all Cao's statistics for all
        given dimensions, and return the corresponding values in the same
        order of the given dimensions.
        All dimensions with non-positive values will receive a `np.nan`
        value for both Cao's metric.

    ts_scaled : :obj:`np.ndarray`, optional
        Standardized time-series values. Used to take advantage of
        precomputations.

    Returns
    -------
    tuple of :obj:`np.ndarray`
        `E1` and `E2` Cao's metrics, necessarily in that order, for all
        given dimensions (and with direct index correspondence for the
        given dimensions).

    References
    ----------
    .. [1] Liangyue Cao, Practical method for determining the minimum
        embedding dimension of a scalar time series, Physica D: Nonlinear
        Phenomena, Volume 110, Issues 1–2, 1997, Pages 43-50,
        ISSN 0167-2789, https://doi.org/10.1016/S0167-2789(97)00118-8.
    """
    if lag <= 0:
        raise ValueError("'lag' must be positive (got {}).".format(lag))

    _dims: t.Sequence[int]

    if np.isscalar(dims):
        _dims = np.arange(1, int(dims) + 1)  # type: ignore

    else:
        _dims = np.asarray(dims, dtype=int)

    ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

    ed, ed_star = np.zeros((2, len(_dims)), dtype=float)

    for ind, dim in enumerate(_dims):
        try:
            emb_next = embed_ts(ts=ts_scaled, lag=lag, dim=dim + 1)
            emb_cur = emb_next[:, 1:]

        except ValueError:
            ed[ind] = np.nan
            ed_star[ind] = np.nan
            continue

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
              detrended_acfs: t.Optional[np.ndarray] = None,
              detrended_ami: t.Optional[np.ndarray] = None,
              **kwargs) -> int:
    """Find the appropriate embedding lag using a given criteria.

    Parameters
    ----------
    ts : :obj:`np.ndarray`
        One-dimensional time-series values.

    lag : int or str, optional (default = None)
        If scalar, return its own value casted to integer,

        If string, it must be one value in {`ami`, `acf`, `acf-nonsig`},
        which defines the strategy of defining the appropriate lag of
        the embedding.
            1. `ami`: uses the first minimum lag of the automutual information
                of the time-series.
            2. `acf`: uses the first negative lag of the autocorrelation of the
                time-series.
            3. `acf-nonsig` (default): uses the first non-significant lag of
                the time-series autocorrelation function. The non-significant
                value is defined as the first lag that has the absolute value
                of is autocorrelation below the critical value defined as
                1.96 / sqrt(ts.size).

        If None, the lag will be searched will the 'acf-nonsig'
        criteria.

    max_nlags : int, optional
        If ``lag`` is not a numeric value, than it will be estimated using
        either the time-series autocorrelation or mutual information
        function estimated up to this argument value.

    detrended_acfs : :obj:`np.ndarray`, optional
        Array of time-series autocorrelation function (for distinct ordered
        lags) of the detrended time-series. Used only if ``lag`` is any of
        `acf`, `acf-nonsig` or None.  If this argument is not given and the
        previous condiditon is meet, the autocorrelation function will be
        calculated inside this method up to ``max_nlags``.

    detrended_ami : :obj:`np.ndarray`, optional
        Array of time-series automutual information function (for distinct
        ordered lags). Used only if ``lag`` is `ami`. If not given and the
        previous condiditon is meet, the automutual information function
        will be calculated inside this method up to ``max_nlags``.

    kwargs:
        Extra arguments for the function used to estimate the lag. used
        only if `lag` is not a numeric value.

    Returns
    -------
    int
        Estimated embedding lag.

    Notes
    -----
    This method may be used to estimate `auto-interations` of the time-series
    (such as calculating the autocorrelation function, for instance) aswell.
    """
    VALID_OPTIONS = {
        "ami": info_theory.MFETSInfoTheory.ft_ami_first_critpt,
        "acf": autocorr.MFETSAutocorr.ft_acf_first_nonpos,
        "acf-nonsig": autocorr.MFETSAutocorr.ft_acf_first_nonsig,
    }  # type: t.Dict[str, t.Callable[..., t.Union[float, int]]]

    if lag is None:
        lag = "acf-nonsig"

    if isinstance(lag, str):
        if lag not in VALID_OPTIONS:
            raise ValueError("'lag' must be in {} (got '{}')."
                             "".format(VALID_OPTIONS.keys(), lag))

        if max_nlags is None:
            max_nlags = ts.size // 2

        if lag == "ami":
            kwargs["detrended_ami"] = detrended_ami

        else:
            kwargs["detrended_acfs"] = detrended_acfs

        kwargs["max_nlags"] = max_nlags

        estimated_lag = VALID_OPTIONS[lag](ts, **kwargs)

        return default_lag if np.isnan(estimated_lag) else int(estimated_lag)

    if np.isscalar(lag):
        lag = int(lag)

        if lag <= 0:
            raise ValueError("'lag' must be positive (got {}).".format(lag))

        return lag

    raise TypeError("'lag' type must be a scalar, a string or None (got {})."
                    "".format(type(lag)))
