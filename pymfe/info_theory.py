"""Module dedicated to information theoretic time-series meta-features."""
import typing as t

import numpy as np
import sklearn.linear_model
import scipy.stats

import pymfe._detrend as _detrend
import pymfe._embed as _embed
import pymfe._utils as _utils

try:
    import pymfe.autocorr as autocorr

except ImportError:
    pass


class MFETSInfoTheory:
    """Extract time-series meta-features from Information Theory group."""
    @classmethod
    def precompute_ts_scaled(cls, ts: np.ndarray,
                             **kwargs) -> t.Dict[str, np.ndarray]:
        """Precompute a standardized time series.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        kwargs:
            Additional arguments and previous precomputed items. May
            speed up this precomputation.

        Returns
        -------
        dict
            The following precomputed item is returned:
                * ``ts_scaled`` (:obj:`np.ndarray`): standardized time-series
                    values (z-score).
        """
        precomp_vals = {}  # type: t.Dict[str, np.ndarray]

        if "ts_scaled" not in kwargs:
            precomp_vals["ts_scaled"] = _utils.standardize_ts(ts=ts)

        return precomp_vals

    @classmethod
    def precompute_detrended_ami(
            cls,
            ts: np.ndarray,
            num_bins: int = 64,
            lags: t.Optional[t.Union[int, t.Sequence[int]]] = None,
            return_dist: bool = False,
            max_nlags: t.Optional[int] = None,
            unbiased: bool = True,
            **kwargs) -> t.Dict[str, np.ndarray]:
        """Precompute detrended time-series Automutual Information function.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        num_bins : int, optional (default=64)
            Number of histogram bins to estimate both the probability density
            of each lagged component, and the joint probability distribution,
            which are all necessary to the automutual information computation.

        lags : int or sequence of int, optional
            Lags to calculate the automutual information.
            If int, calculate the automutual information from lag 1 to up to
            the given ``lags`` value.
            If sequence of integers, calculate the automutual information for
            each of the given lags.

        return_dist : bool, optional (default=False)
            If True, return the automutual information distance for every lag,
            defined as:
            $$
                DAMI(ts) = 1 - AMI(ts) / H(ts_A, ts_B)
                         = (H(ts_A) + H(ts_B)) / H(ts_A, ts_B)
            $$

        max_nlags : int, optional
            If ``lags`` is None, then a single lag will be estimated from the
            first negative value of the detrended time-series autocorrelation
            function up to `max_nlags`, if any. Otherwise, lag 1 will be used.
            Used only if ``lags`` is None.

        unbiased : bool, optional (default=True)
            If True, correct the autocorrelation function for statistical
            bias. Used only if ``lags`` is None.

        kwargs:
            Additional arguments and previous precomputed items. May
            speed up this precomputation.

        Returns
        -------
        dict
            The following precomputed item is returned:
                * ``detrended_ami`` (:obj:`np.ndarray`): the automutual
                    information function of the detrended time-series.

            The following item is necessary and, therefore, also precomputed
            if necessary:
                * ``detrended_acfs`` (:obj:`np.ndarray`): the autocorrelation
                    function from the detrended time-series.

        References
        ----------
        .. [1] Fraser AM, Swinney HL. Independent coordinates for strange
            attractors from mutual information. Phys Rev A Gen Phys.
            1986;33(2):1134‐1140. doi:10.1103/physreva.33.1134
        .. [2] Thomas M. Cover and Joy A. Thomas. 1991. Elements of information
            theory. Wiley-Interscience, USA.
        """
        precomp_vals = {}  # type: t.Dict[str, np.ndarray]

        detrended_acfs = kwargs.get("detrended_acfs")

        if lags is None and detrended_acfs is None:
            precomp_vals.update(
                autocorr.MFETSAutocorr.precompute_detrended_acf(
                    ts=ts,
                    nlags=max_nlags,
                    unbiased=unbiased))

            detrended_acfs = precomp_vals["detrended_acfs"]

        if "detrended_ami" not in kwargs:
            precomp_vals["detrended_ami"] = cls.ft_ami_detrended(
                ts=ts,
                lags=lags,
                num_bins=num_bins,
                return_dist=return_dist,
                detrended_acfs=detrended_acfs)

        return precomp_vals

    @classmethod
    def _calc_ami(cls,
                  ts: np.ndarray,
                  lag: int,
                  num_bins: int = 64,
                  return_dist: bool = False) -> float:
        """Calculate the Automutual Information of a time-series for a lag.

        The automutual information AMI is defined as:
        $$
            AMI(ts) = H(ts_A) + H(ts_B) - H(ts_A, ts_B)
        $$
        where `ts` is the time-series, $H$ is the Shannon entropy function, and
        $H(A, B)$ is the Shannon entropy of the joint probability of A and B.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        lag : int
            Lag of the automutual information.

        num_bins : int, optional (default=64)
            Number of histogram bins to estimate both the probability density
            of each lagged component, and the joint probability distribution,
            all necessary to the automutual information calculus.

        return_dist : bool, optional (default=False)
            If True, return the automutual information distance defined as
            $$
                DAMI(ts) = 1 - AMI(ts) / H(ts_A, ts_B)
                         = (H(ts_A) + H(ts_B)) / H(ts_A, ts_B)
            $$

        Returns
        -------
        float
            If `return_dist` is False, return the automutual information of
            the time-series with the given lag. If `return_dist` is True,
            return the distance metric version of the automutual information.

        References
        ----------
        .. [1] Fraser AM, Swinney HL. Independent coordinates for strange
            attractors from mutual information. Phys Rev A Gen Phys.
            1986;33(2):1134‐1140. doi:10.1103/physreva.33.1134
        .. [2] Thomas M. Cover and Joy A. Thomas. 1991. Elements of information
            theory. Wiley-Interscience, USA.
        """
        ts_x = ts[:-lag]
        ts_y = ts[lag:]

        ts_x_prob = np.histogram(ts_x, bins=num_bins, density=True)[0]
        ts_y_prob = np.histogram(ts_y, bins=num_bins, density=True)[0]
        joint_prob = np.histogram2d(ts_x, ts_y, bins=num_bins, density=True)[0]

        ent_ts_x = scipy.stats.entropy(ts_x_prob, base=2)
        ent_ts_y = scipy.stats.entropy(ts_y_prob, base=2)
        ent_joint = scipy.stats.entropy(joint_prob.ravel(), base=2)

        ami = ent_ts_x + ent_ts_y - ent_joint

        if return_dist:
            # Note: this is the same as defining, right from the start,
            # ami = (ent_ts_x + ent_ts_y) / ent_joint
            # However, here all steps are kept to make the code clearer.
            ami = 1 - ami / ent_joint

        return ami

    @classmethod
    def ft_hist_entropy(cls,
                        ts: np.ndarray,
                        num_bins: int = 10,
                        normalize: bool = True) -> float:
        """Shannon's Entropy from a histogram frequencies.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        num_bins : int, optional (default=10)
            Number of histogram bins.

        normalize : bool, optional (default=True)
            If True, normalize the result in the [0, 1] range.

        Returns
        -------
        float
            Histogram Shannon's entropy.

        References
        ----------
        .. [1] Shannon, C.E. (1948), A Mathematical Theory of Communication.
            Bell System Technical Journal, 27: 379-423.
            doi:10.1002/j.1538-7305.1948.tb01338.x
        """
        freqs = np.histogram(ts, bins=num_bins, density=True)[0]

        entropy = scipy.stats.entropy(freqs, base=2)

        if normalize:
            entropy /= np.log2(freqs.size)

        return entropy

    @classmethod
    def ft_hist_ent_out_diff(cls,
                             ts: np.ndarray,
                             num_bins: int = 10,
                             pcut: float = 0.05,
                             normalize: bool = True) -> float:
        """Difference of histogram entropy with and without outliers.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        num_bins : int, optional (default=10)
            Number of histogram bins.

        pcut : float, optional (default=0.05)
            Proportion of outlier cut, in both extremes. Must be a value
            in (0.0, 0.5) range.

        normalize : bool, optional (default=True)
            If True, normalize the result in the [0, 1] range.

        Returns
        -------
        float
            Difference of histogram Shannon's entropy with and without outlier
            observations.

        References
        ----------
        .. [1] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework
            for Automated Time-Series Phenotyping Using Massive Feature
            Extraction, Cell Systems 5: 527 (2017).
            DOI: 10.1016/j.cels.2017.10.001
        .. [2] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
            time-series analysis: the empirical structure of time series and
            their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013).
            DOI: 10.1098/rsif.2013.0048
        """
        if not 0 < pcut < 0.5:
            raise ValueError("'pcut' must be in (0.0, 0.5) (got "
                             "{}).".format(pcut))

        cut_low, cut_high = np.quantile(ts, (pcut, 1 - pcut))
        ts_inliners = ts[np.logical_and(cut_low <= ts, ts <= cut_high)]

        ent_ts = cls.ft_hist_entropy(ts=ts,
                                     num_bins=num_bins,
                                     normalize=normalize)
        ent_ts_inliners = cls.ft_hist_entropy(ts=ts_inliners,
                                              num_bins=num_bins,
                                              normalize=normalize)

        entropy_diff = ent_ts - ent_ts_inliners

        return entropy_diff

    @classmethod
    def ft_ami(cls,
               ts: np.ndarray,
               num_bins: int = 64,
               lags: t.Optional[t.Sequence[int]] = None,
               return_dist: bool = False,
               max_nlags: t.Optional[int] = None,
               detrended_acfs: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Automutual information of the time-series.

        The automutual information AMI is defined as:
        $$
            AMI(ts) = H(ts_A) + H(ts_B) - H(ts_A, ts_B)
        $$
        where `ts` is the time-series, $H$ is the Shannon entropy function, and
        $H(A, B)$ is the Shannon entropy of the joint probability of A and B.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        num_bins : int, optional (default=64)
            Number of histogram bins to estimate both the probability density
            of each lagged component, and the joint probability distribution,
            which are all necessary to the automutual information computation.

        lags : sequence of int, optional
            Lags to calculate the automutual information.

        return_dist : bool, optional (default=False)
            If True, return the automutual information distance for every lag,
            defined as:
            $$
                DAMI(ts) = 1 - AMI(ts) / H(ts_A, ts_B)
                         = (H(ts_A) + H(ts_B)) / H(ts_A, ts_B)
            $$

        max_nlags : int, optional
            If ``lags`` is None, then a single lag will be estimated from the
            first negative value of the detrended time-series autocorrelation
            function up to `max_nlags`, if any. Otherwise, lag 1 will be used.
            Used only if ``detrended_acfs`` is None.

        detrended_acfs : :obj:`np.ndarray`, optional
            Array of time-series autocorrelation function (for distinct ordered
            lags) of the detrended time-series. Used only if ``lag`` is None.
            If this argument is not given and the previous condiditon is meet,
            the autocorrelation function will be calculated inside this method
            up to ``max_nlags``.

        Returns
        -------
        :obj:`np.ndarray`
            If `return_dist` is False, return the automutual information of
            the time-series for all given lags. If `return_dist` is True,
            return the distance metric version of the automutual information
            for all given lags.

        References
        ----------
        .. [1] Fraser AM, Swinney HL. Independent coordinates for strange
            attractors from mutual information. Phys Rev A Gen Phys.
            1986;33(2):1134‐1140. doi:10.1103/physreva.33.1134
        .. [2] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework
            for Automated Time-Series Phenotyping Using Massive Feature
            Extraction, Cell Systems 5: 527 (2017).
            DOI: 10.1016/j.cels.2017.10.001
        .. [3] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
            time-series analysis: the empirical structure of time series and
            their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013).
            DOI: 10.1098/rsif.2013.0048
        .. [4] Thomas M. Cover and Joy A. Thomas. 1991. Elements of information
            theory. Wiley-Interscience, USA.
        """
        # Note: using ts_detrended=ts to avoid detrending.
        non_detrended_ami = cls.ft_ami_detrended(ts=ts,
                                                 num_bins=num_bins,
                                                 lags=lags,
                                                 return_dist=return_dist,
                                                 max_nlags=max_nlags,
                                                 detrended_acfs=detrended_acfs,
                                                 ts_detrended=ts)

        return non_detrended_ami

    @classmethod
    def ft_ami_detrended(
            cls,
            ts: np.ndarray,
            num_bins: int = 64,
            lags: t.Optional[t.Union[int, t.Sequence[int]]] = None,
            return_dist: bool = False,
            max_nlags: t.Optional[int] = None,
            ts_detrended: t.Optional[np.ndarray] = None,
            detrended_acfs: t.Optional[np.ndarray] = None,
            detrended_ami: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Automutual information of the detrended time-series.

        The automutual information AMI is defined as:
        $$
            AMI(ts) = H(ts_A) + H(ts_B) - H(ts_A, ts_B)
        $$
        where `ts` is the time-series, $H$ is the Shannon entropy function, and
        $H(A, B)$ is the Shannon entropy of the joint probability of A and B.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        num_bins : int, optional (default=64)
            Number of histogram bins to estimate both the probability density
            of each lagged component, and the joint probability distribution,
            which are all necessary to the automutual information computation.

        lags : int or sequence of int, optional
            Lags to calculate the automutual information.
            If int, calculate the automutual information from lag 1 to up to
            the given ``lags`` value.
            If sequence of integers, calculate the automutual information for
            each of the given lags.
            If None, estimate the appropriate lag from the autocorrelation
            function (using the lag corresponding to the first non-positive
            value), and estimate the automutual information from lag 1 up to
            the estimated appropriate lag.

        return_dist : bool, optional (default=False)
            If True, return the automutual information distance for every lag,
            defined as:
            $$
                DAMI(ts) = 1 - AMI(ts) / H(ts_A, ts_B)
                         = (H(ts_A) + H(ts_B)) / H(ts_A, ts_B)
            $$

        max_nlags : int, optional
            If ``lags`` is None, then a single lag will be estimated from the
            first negative value of the detrended time-series autocorrelation
            function up to `max_nlags`, if any. Otherwise, lag 1 will be used.
            Used only if ``detrended_acfs`` is None.

        ts_detrended : :obj:`np.ndarray`, optional
            Detrended time-series. If None, the time-series will be detrended
            using Friedman's Super Smoother.

        detrended_acfs : :obj:`np.ndarray`, optional
            Array of time-series autocorrelation function (for distinct ordered
            lags) of the detrended time-series. Used only if ``lag`` is None.
            If this argument is not given and the previous condiditon is meet,
            the autocorrelation function will be calculated inside this method
            up to ``max_nlags``.

        detrended_ami : :obj:`np.ndarray`, optional
            This method's return value. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            If `return_dist` is False, return the automutual information of
            the time-series for all given lags. If `return_dist` is True,
            return the distance metric version of the automutual information
            for all given lags.

        References
        ----------
        .. [1] Fraser AM, Swinney HL. Independent coordinates for strange
            attractors from mutual information. Phys Rev A Gen Phys.
            1986;33(2):1134‐1140. doi:10.1103/physreva.33.1134
        .. [2] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework
            for Automated Time-Series Phenotyping Using Massive Feature
            Extraction, Cell Systems 5: 527 (2017).
            DOI: 10.1016/j.cels.2017.10.001
        .. [3] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
            time-series analysis: the empirical structure of time series and
            their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013).
            DOI: 10.1098/rsif.2013.0048
        .. [4] Thomas M. Cover and Joy A. Thomas. 1991. Elements of information
            theory. Wiley-Interscience, USA.
        """
        if detrended_ami is not None:
            return detrended_ami

        if lags is None:
            lags = _embed.embed_lag(ts=ts,
                                    lag="acf",
                                    max_nlags=max_nlags,
                                    detrended_acfs=detrended_acfs)

        if np.isscalar(lags):
            lags = np.arange(1, 1 + int(lags))  # type: ignore

        if ts_detrended is None:
            ts_detrended = _detrend.decompose(ts=ts, ts_period=0)[2]

        _lags: t.Sequence[int] = np.asarray(lags, dtype=int)
        ami = np.zeros(len(_lags), dtype=float)

        for ind, lag in enumerate(_lags):
            ami[ind] = cls._calc_ami(ts=ts_detrended,
                                     lag=lag,
                                     num_bins=num_bins,
                                     return_dist=return_dist)

        return ami

    @classmethod
    def ft_ami_first_critpt(
            cls,
            ts: np.ndarray,
            num_bins: int = 64,
            max_nlags: t.Optional[int] = None,
            dist_ami: bool = False,
            detrended_ami: t.Optional[np.ndarray] = None
    ) -> t.Union[int, float]:
        """First critical point of the automutual information function.

        If `return_dist` is False, then it is search for the first local
        minima in the automutual information function. If `return_dist` is
        True, then search for the first local maxima in the automutual
        information distance function.

        Check `ft_ami` documentation for more information.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        num_bins : int, optional (default=64)
            Number of histogram bins to estimate both the probability density
            of each lagged component, and the joint probability distribution,
            which are all necessary to the automutual information computation.
            Used only if ``detrended_ami`` is None.

        dist_ami : bool, optional (default=False)
            If True, check for critical points on the automutual information
            distance function. Check `ft_ami` documentation for more
            information.

        max_nlags : int, optional
            Maximum number of lags to estimate the automutual information. Is
            None, then the default value will be min(64, floor(len(ts) / 2)).
            Used only if ``detrended_ami`` is None.

        detrended_ami : :obj:`np.ndarray`, optional
            If `dist_ami` is False, automutual information function from the
            detrended time-series.
            If `dist_ami` is True, automutual information distance function
            from the detrended time-series.
            Used to take advantage of precomputations.

        Returns
        -------
        int or float
            If `dist_ami` False, lag corresponding to the first local minima
            in the automutual information function. If `dist_ami` is True, lag
            corresponding to the first local maxima of the automutual
            information distance function. If no critical point of interest is
            found, this method will return `np.nan`.

        References
        ----------
        .. [1] Fraser AM, Swinney HL. Independent coordinates for strange
            attractors from mutual information. Phys Rev A Gen Phys.
            1986;33(2):1134‐1140. doi:10.1103/physreva.33.1134
        .. [2] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework
            for Automated Time-Series Phenotyping Using Massive Feature
            Extraction, Cell Systems 5: 527 (2017).
            DOI: 10.1016/j.cels.2017.10.001
        .. [3] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
            time-series analysis: the empirical structure of time series and
            their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013).
            DOI: 10.1098/rsif.2013.0048
        .. [4] Thomas M. Cover and Joy A. Thomas. 1991. Elements of information
            theory. Wiley-Interscience, USA.
        """
        if detrended_ami is None:
            if max_nlags is None:
                max_nlags = min(64, ts.size // 2)

            detrended_ami = cls.ft_ami_detrended(ts=ts,
                                                 num_bins=num_bins,
                                                 lags=max_nlags,
                                                 return_dist=dist_ami)

        # Note: if 'return_dist=True', return the first local maximum.
        # If otherwise, return the first local minimum.
        type_ = "max" if dist_ami else "min"

        crit_point = _utils.find_crit_pt(arr=detrended_ami, type_=type_)

        try:
            return np.flatnonzero(crit_point)[0] + 1

        except IndexError:
            return np.nan

    @classmethod
    def ft_ami_curvature(
            cls,
            ts: np.ndarray,
            noise_range: t.Tuple[float, float] = (0, 3),
            noise_inc_num: float = 10,
            lag: t.Optional[t.Union[str, int]] = None,
            random_state: t.Optional[int] = None,
            ts_scaled: t.Optional[np.ndarray] = None,
            max_nlags: t.Optional[int] = None,
            detrended_acfs: t.Optional[np.ndarray] = None,
    ) -> float:
        """Estimate the Automutual information curvature.

        The Automutual information curvature is estimated using iterative noise
        amplification strategy.

        In the iterative noise amplification strategy, a random white noise
        is sampled from a normal distribution (mean 0 and variance 1). Then,
        this same noise is iteratively amplified from a uniformly spaced
        scales in ``noise_range`` range and added to the time-series. The
        automutual information is calculated from the perturbed time-series
        for each noise amplification.

        The automutual information curvature is the angular coefficient of a
        linear regression of the automutual information onto the noise scales.

        The lag used for every iteration is fixed from the start and, if not
        fixed by the user, it is estimated from the autocorrelation function
        by default.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        noise_range : tuple of float, optional (default=(0, 3))
            A tuple of floats in the form (min_scale, max_scale) for the noise
            amplication range.

        noise_inc_num: float, optional (default=10)
            Number of noise amplifications. The parameter ``noise_range`` will
            be split evenly into ``noise_inc_num`` parts.

        lag : int or str, optional
            Lag to calculate the statistic. It must be a strictly positive
            value, None or a string in {`acf`, `acf-nonsig`, `ami`}. In the
            last two type of options, the lag is estimated within this method
            using the given strategy method (or, if None, it is used the
            strategy `acf-nonsig` by default) up to ``max_nlags``.
                1. `acf`: the lag corresponds to the first non-positive value
                    in the autocorrelation function.
                2. `acf-nonsig`: lag corresponds to the first non-significant
                    value in the autocorrelation function (absolute value below
                    the critical value of 1.96 / sqrt(ts.size)).
                3. `ami`: lag corresponds to the first local minimum of the
                    time-series automutual information function.

        random_state : int, optional
            Random seed to ensure reproducibility.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        max_nlags : int, optional
            If ``lag`` is None, then a single lag will be estimated from the
            first negative value of the detrended time-series autocorrelation
            function up to `max_nlags`, if any. Otherwise, lag 1 will be used.
            Used only if ``detrended_acfs`` is None.

        detrended_acfs : :obj:`np.ndarray`, optional
            Array of time-series autocorrelation function (for distinct ordered
            lags) of the detrended time-series. Used only if ``lag`` is None.
            If this argument is not given and the previous condiditon is meet,
            the autocorrelation function will be calculated inside this method
            up to ``max_nlags``.

        Returns
        -------
        float
            Estimated automutual information curvature.

        References
        ----------
        .. [1] Fraser AM, Swinney HL. Independent coordinates for strange
            attractors from mutual information. Phys Rev A Gen Phys.
            1986;33(2):1134‐1140. doi:10.1103/physreva.33.1134
        .. [2] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework
            for Automated Time-Series Phenotyping Using Massive Feature
            Extraction, Cell Systems 5: 527 (2017).
            DOI: 10.1016/j.cels.2017.10.001
        .. [3] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
            time-series analysis: the empirical structure of time series and
            their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013).
            DOI: 10.1098/rsif.2013.0048
        .. [4] Thomas M. Cover and Joy A. Thomas. 1991. Elements of information
            theory. Wiley-Interscience, USA.
        """
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        # Note: casting lag to an array since 'ft_ami_detrended' demands
        # a sequence of lags.
        _lag = np.asarray([
            _embed.embed_lag(ts=ts_scaled,
                             lag=lag,
                             max_nlags=max_nlags,
                             detrended_acfs=detrended_acfs)
        ])

        if random_state is not None:
            np.random.seed(random_state)

        # Note: the noise is fixed from the start, and amplified at each
        # iteration.
        gaussian_noise = np.random.randn(ts_scaled.size)
        noise_std = np.linspace(*noise_range, noise_inc_num)

        ami = np.zeros(noise_inc_num, dtype=float)

        for ind, cur_std in enumerate(noise_std):
            ts_corrupted = ts_scaled + cur_std * gaussian_noise

            ami[ind] = cls.ft_ami_detrended(ts=ts_corrupted,
                                            num_bins=32,
                                            lags=_lag,
                                            return_dist=False)

        model = sklearn.linear_model.LinearRegression().fit(
            X=noise_std.reshape(-1, 1), y=ami)

        curvature = model.coef_[0]

        return curvature

    @classmethod
    def ft_approx_entropy(cls,
                          ts: np.ndarray,
                          embed_dim: int = 2,
                          embed_lag: int = 1,
                          threshold: float = 0.2,
                          metric: str = "chebyshev",
                          p: t.Union[int, float] = 2,
                          ts_scaled: t.Optional[np.ndarray] = None) -> float:
        """Approximate entropy of the time-series.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        embed_dim : int, optional (default=2)
            Embedding dimension.

        embed_dim : int, optional (default=1)
            Embedding lag.

        threshold : float, optional (default=0.2)
            Threshold to consider which observations are next to each other
            after embedding.

        metric : str, optional (default="chebyshev")
            Distance metric to calculate the pairwise distance of the
            observations after each embedding.
            Check `scipy.spatial.distance.cdist` documentation for the complete
            list of available distance metrics.

        p : int or float, optional (default=2)
            Power parameter for the minkowski metric. Used only if metric is
            `minkowski`.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        Returns
        -------
        float
            Estimated approximate entropy.

        References
        ----------
        .. [1] Pincus, S.M., Gladstone, I.M. & Ehrenkranz, R.A. A regularity
            statistic for medical data analysis. J Clin Monitor Comput 7,
            335–345 (1991). https://doi.org/10.1007/BF01619355
        .. [2] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework
            for Automated Time-Series Phenotyping Using Massive Feature
            Extraction, Cell Systems 5: 527 (2017).
            DOI: 10.1016/j.cels.2017.10.001
        .. [3] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
            time-series analysis: the empirical structure of time series and
            their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013).
            DOI: 10.1098/rsif.2013.0048
        """
        def neigh_num(dim: int) -> int:
            """Mean-log-mean of the number of radius neighbors."""
            embed = _embed.embed_ts(ts_scaled, dim=dim, lag=embed_lag)
            dist_mat = scipy.spatial.distance.cdist(embed,
                                                    embed,
                                                    metric=metric,
                                                    p=p)
            return np.mean(np.log(np.mean(dist_mat < threshold, axis=1)))

        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        approx_entropy = neigh_num(embed_dim) - neigh_num(embed_dim + 1)

        return approx_entropy

    @classmethod
    def ft_sample_entropy(cls,
                          ts: np.ndarray,
                          embed_dim: int = 2,
                          embed_lag: int = 1,
                          threshold: float = 0.2,
                          metric: str = "chebyshev",
                          p: t.Union[int, float] = 2,
                          ts_scaled: t.Optional[np.ndarray] = None) -> float:
        """Sample entropy of the time-series.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        embed_dim : int, optional (default=2)
            Embedding dimension.

        embed_dim : int, optional (default=1)
            Embedding lag.

        threshold : float, optional (default=0.2)
            Threshold to consider which observations are next to each other
            after embedding.

        metric : str, optional (default="chebyshev")
            Distance metric to calculate the pairwise distance of the
            observations after each embedding.
            Check `scipy.spatial.distance.cdist` documentation for the complete
            list of available distance metrics.

        p : int or float, optional (default=2)
            Power parameter for the minkowski metric. Used only if metric is
            `minkowski`.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        Returns
        -------
        float
            Estimated sample entropy.

        References
        ----------
        .. [1] Physiological time-series analysis using approximate entropy and
            sample entropy Joshua S. Richman and J. Randall Moorman, American
            Journal of Physiology-Heart and Circulatory Physiology 2000 278:6,
            H2039-H2049
        .. [2] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework
            for Automated Time-Series Phenotyping Using Massive Feature
            Extraction, Cell Systems 5: 527 (2017).
            DOI: 10.1016/j.cels.2017.10.001
        .. [3] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
            time-series analysis: the empirical structure of time series and
            their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013).
            DOI: 10.1098/rsif.2013.0048
        """
        def log_neigh_num(dim: int) -> int:
            """Logarithm of the number of nearest neighbors."""
            embed = _embed.embed_ts(ts_scaled, dim=dim, lag=embed_lag)
            dist_mat = scipy.spatial.distance.pdist(embed, metric=metric, p=p)
            return np.log(np.sum(dist_mat < threshold))

        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        sample_entropy = log_neigh_num(embed_dim) - log_neigh_num(embed_dim +
                                                                  1)

        return sample_entropy

    @classmethod
    def ft_control_entropy(cls,
                           ts: np.ndarray,
                           embed_dim: int = 2,
                           threshold: float = 0.2,
                           metric: str = "chebyshev",
                           p: t.Union[int, float] = 2,
                           embed_lag: int = 1,
                           ts_scaled: t.Optional[np.ndarray] = None) -> float:
        """Control entropy of the time-series.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        embed_dim : int, optional (default=2)
            Embedding dimension.

        embed_dim : int, optional (default=1)
            Embedding lag.

        threshold : float, optional (default=0.2)
            Threshold to consider which observations are next to each other
            after embedding.

        metric : str, optional (default="chebyshev")
            Distance metric to calculate the pairwise distance of the
            observations after each embedding.
            Check `scipy.spatial.distance.cdist` documentation for the complete
            list of available distance metrics.

        p : int or float, optional (default=2)
            Power parameter for the minkowski metric. Used only if metric is
            `minkowski`.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        Returns
        -------
        float
            Estimated control entropy.

        References
        ----------
        .. [1] Erik M. Bollt and Joseph D. Skufca, Control Entropy: a
            complexity measure for nonstationary signals Mathematical
            Biosciences and Engineering Volume 6, Number 1, January 2009
            doi:10.3934/mbe.2009.6.1 pp. 1–25
        .. [2] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework
            for Automated Time-Series Phenotyping Using Massive Feature
            Extraction, Cell Systems 5: 527 (2017).
            DOI: 10.1016/j.cels.2017.10.001
        .. [3] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
            time-series analysis: the empirical structure of time series and
            their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013).
            DOI: 10.1098/rsif.2013.0048
        """
        control_entropy = cls.ft_sample_entropy(ts=np.diff(ts),
                                                embed_dim=embed_dim,
                                                embed_lag=embed_lag,
                                                threshold=threshold,
                                                metric=metric,
                                                p=p,
                                                ts_scaled=ts_scaled)

        return control_entropy

    @classmethod
    def ft_surprise(cls,
                    ts: np.ndarray,
                    num_bins: int = 10,
                    memory_size: t.Union[float, int] = 0.1,
                    num_it: int = 128,
                    method: str = "distribution",
                    diff_order: int = 1,
                    epsilon: float = 1e-8,
                    random_state: t.Optional[int] = None) -> np.ndarray:
        """Surprise factor of a nth-order differenced time-series.

        The surprise measure is an estimation of the negative log-probablity of
        a given random reference observation have its value given a short-term
        memory of `ceil(memory_size * len(ts))` most recent values.

        This method measures the surprise factor associated with each value on
        a nth-order differenced time-series. This analysis may also be
        performed on the original time-series, simply using the 0th-order
        differenced time-series.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        num_bins : int, optional (default=10)
            Number of histogram bins to time-series discretization. The data is
            discretized using an equiprobable histogram (i.e., all bins have
            the same number of instances).

        memory_size : float or int, optional (default=0.1)
            Recent memory size S.
            If 0 < S < 1, the proportion of the time-series length as the
            memory size.
            If S >= 1, length of the memory.
            Must be a value stricly positive and, if ``method`` value is
            `1-transition`, then it must be larger or equal than 2.

        num_it : int, optional (default=128)
            Number of reference observations sampled.

        method : str, optional (default="distribution")
            Defines the probability function to determine how much surprise
            a given value causes given the past values in the recent memory.
            Must be either `distribution` or `1-transition`.
            1. `distribution`: The ``memory_size`` previous values immediately
                before the reference value are compared to it.
            2. `1-transition`: compare the current reference value to the
                values one time unit after the values equal to the previous
                value of the current reference value within the recent memory.
                This means that we are searching for transitions `B -> A`, from
                the previous value `B` to the current value `A`, in the recent
                memory, and calculating the probability of that event happens.
                In other words, we are calculating the conditional probability
                P(y[t] = A | y[t-1] = B).

        diff_order : int, optional (default=1)
            Order of differentiation of the time-series to carry the analysis.
            If `0`, then this analysis will be performed on the original
            time-series value. Must be a non-negative integer.

        epsilon : float, optional (default=1e-8)
            Tiny threshold value to consider probabilities as zero.

        random_state : int, optional
            Random seed to ensure reproducibility.

        Returns
        -------
        :obj:`np.ndarray`
            Surprise (-1.0 * log-probability) for each random reference
            instance.

        References
        ----------
        .. [1] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework
            for Automated Time-Series Phenotyping Using Massive Feature
            Extraction, Cell Systems 5: 527 (2017).
            DOI: 10.1016/j.cels.2017.10.001
        .. [2] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
            time-series analysis: the empirical structure of time series and
            their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013).
            DOI: 10.1098/rsif.2013.0048
        """
        VALID_METHODS = ("distribution", "1-transition")

        if method not in VALID_METHODS:
            raise ValueError("'method' must in {} (got '{}')."
                             "".format(VALID_METHODS, method))

        if memory_size <= 0:
            raise ValueError("'memory_size' must be positive (got "
                             "{}).".format(memory_size))

        if num_it <= 0:
            raise ValueError("'num_it' must be positive (got {})."
                             "".format(num_it))

        def get_reference_inds(inds_num: int, max_ind: int) -> np.ndarray:
            """Get min(ts.size - memory_size, inds_num) random indices.

            The indices are chosen randomly without replacement.
            """
            if max_ind - memory_size > num_it:
                if random_state is not None:
                    np.random.seed(random_state)

                # Note: we can't use indices before the first 'memory_size'
                # indices as reference, since we need a 'memory' of size
                # 'memory_size' (i.e., we need at least 'memory_size' past
                # indices). Therefore, we always skip they.
                return memory_size + np.random.choice(
                    max_ind - memory_size, size=inds_num, replace=False)

            # Note: the requested number of indices is not smaller than
            # the number of available indices. Therefore, just return
            # all available indices.
            return np.arange(memory_size, max_ind)

        if 0 < memory_size < 1:
            memory_size = int(np.ceil(ts.size * memory_size))

        if method == "distribution":

            def prob_func(ref_ind: int, ts_bin: np.ndarray,
                          hor_len: int) -> float:
                """Return probability of the referenced value."""
                return np.mean(ts_bin[ref_ind -
                                      hor_len:ref_ind] == ts_bin[ref_ind])

        else:
            if memory_size <= 1:
                raise ValueError("'memory_size' must be >= 2 with "
                                 "'1-transition' method.")

            def prob_func(ref_ind: int, ts_bin: np.ndarray,
                          hor_len: int) -> float:
                """Return probability of the referenced transition."""
                mem_data = ts_bin[ref_ind - hor_len:ref_ind]
                prev_val = mem_data[-1]
                prev_val_inds = np.flatnonzero(mem_data[:-1] == prev_val)
                equal_vals = mem_data[prev_val_inds + 1] == ts_bin[ref_ind]
                return np.mean(equal_vals) if equal_vals.size else 0.0

        ts_diff = np.diff(ts, n=diff_order)

        # Note: discretize time-series using an equiprobable histogram
        # (i.e. all bins have approximately the same number of instances).
        ts_bin = _utils.discretize(ts=ts_diff,
                                   num_bins=num_bins,
                                   strategy="equiprobable")

        probs = np.zeros(num_it, dtype=float)
        ref_inds = get_reference_inds(inds_num=num_it, max_ind=ts_diff.size)

        for ind, ref_ind in enumerate(ref_inds):
            probs[ind] = prob_func(ref_ind=ref_ind,
                                   ts_bin=ts_bin,
                                   hor_len=int(memory_size))

        probs[probs < epsilon] = 1.0
        surprise = -np.log(probs)

        return surprise

    @classmethod
    def ft_lz_complexity(cls,
                         ts: np.ndarray,
                         num_bins: int = 10,
                         normalize: bool = True) -> float:
        """Lempel-Ziv complexity of the discretized time-series.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        num_bins : int, optional (default=10)
            Number of histogram bins to discretize the time-series. It is
            used a histogram with bins of equal width.

        normalize : bool, optional (default=True)
            If True, normalize the final measure in [0, 1] range.

        Returns
        -------
        float
            If `normalize` is False, Lempel-Ziv complexity estimation for the
            discretized time-series. If `normalize` is True, then the measure
            will be normalized accordingly to the number of histogram bins
            used.

        References
        ----------
        .. [1] A. Lempel and J. Ziv, "On the Complexity of Finite Sequences,"
            in IEEE Transactions on Information Theory, vol. 22, no. 1, pp.
            75-81, January 1976, doi: 10.1109/TIT.1976.1055501.
        .. [2] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework
            for Automated Time-Series Phenotyping Using Massive Feature
            Extraction, Cell Systems 5: 527 (2017).
            DOI: 10.1016/j.cels.2017.10.001
        .. [3] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
            time-series analysis: the empirical structure of time series and
            their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013).
            DOI: 10.1098/rsif.2013.0048
        """
        ts_bin = tuple(
            _utils.discretize(ts=ts,
                              num_bins=num_bins,
                              strategy="equal-width",
                              dtype=int))

        ind_start, ind_end = 0, 1
        substrings = set()
        _len = len(ts_bin)

        while ind_end <= _len:
            substring = ts_bin[ind_start:ind_end]

            if substring not in substrings:
                substrings.add(substring)
                ind_start = ind_end

            ind_end += 1

        lz_comp = len(substrings)

        if normalize:
            lz_comp *= np.log(_len) / (_len * np.log(num_bins))

        return lz_comp
