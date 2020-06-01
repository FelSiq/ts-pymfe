"""Module dedicated to itrandd time-series meta-features."""
import typing as t
import collections

import numpy as np
import statsmodels.tsa

import pymfe._utils as _utils
import pymfe._embed as _embed
import pymfe._surrogates as _surrogates

try:
    import pymfe.autocorr as autocorr

except ImportError:
    pass


class MFETSRandomize:
    """Extract time-series meta-features from Randomize group."""
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
    def precompute_itrand_stats(cls,
                                ts: np.ndarray,
                                strategy: str = "dist-dynamic",
                                prop_rep: t.Union[int, float] = 2,
                                prop_interval: float = 0.1,
                                random_state: t.Optional[int] = None,
                                **kwargs) -> t.Dict[str, np.ndarray]:
        """Precompute global statistics with iterative perturbation method.

        In the iterative perturbation method, a copy of the time-series is
        modified at each iteration. The quantity of observations modified and
        the sample pool from which the new values are drawn depends on the
        ``strategy``selected. Then, statistics are extracted after every `k`
        iterations (given by ceil(ts.size * ``prop_interval``))..

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        strategy : str, optional (default="dist-dynamic")
            The strategy used to perturb the current population. Must be one
            of the following:

                1. `dist-static`: (static distribution) one observation of the
                current population is overwritten by one observation from the
                original time-series.

                2. `dist-dynamic`: (dynamic distribution) one observation of
                the current population is overwritten by another observation
                of the current population.

                3. `permute`: two observations of the current population swaps
                its positions.

        prop_rep : int or float, optional (default=2)
            Number of total iterations proportional to the time-series size.
            This means that this process will iterate for approximately
            ceil(prop_rep * ts.size) iterations. More rigorously, the number
            of iterations also depends on the number of iterations that the
            statistics are extracted, to avoid lose computations.

        prop_interval : float, optional (default=0.1)
            Interval that the statistics are extracted from the current
            population, proportional to the time-series length.

        random_state : int, optional
            Random seed to ensure reproducibility.

        kwargs:
            Additional arguments and previous precomputed items. May
            speed up this precomputation.

        Returns
        -------
        dict
            The following precomputed item is returned:
                * ``itrand_stat_mean`` (:obj:`np.ndarray`): mean extracted
                    from each extraction event in the iterative perturbation
                    method on the time-series.
                * ``itrand_stat_std`` (:obj:`np.ndarray`): std extracted
                    from each extraction event in the iterative perturbation
                    method on the time-series.
                * ``itrand_stat_acf`` (:obj:`np.ndarray`): autocorrelation of
                    lag 1 extracted from each extraction event in the iterative
                    perturbation method on the time-series.

            The following items is necessary and, therefore, also precomputed
            if necessary:
                * ``ts_scaled`` (:obj:`np.ndarray`): standardized time-series
                    values (z-score).

        References
        ----------
        .. [1] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework
            for Automated Time-Series Phenotyping Using Massive Feature
            Extraction, Cell Systems 5: 527 (2017).
            DOI: 10.1016/j.cels.2017.10.001
        .. [2] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
            time-series analysis: the empirical structure of time series and
            their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013).
        """
        precomp_vals = {}  # type: t.Dict[str, np.ndarray]

        ts_scaled = kwargs.get("ts_scaled")

        if ts_scaled is None:
            precomp_vals.update(cls.precompute_ts_scaled(ts=ts))
            ts_scaled = precomp_vals["ts_scaled"]

        stats = collections.OrderedDict(
            (
                ("mean", np.mean),
                ("std", np.std),
                ("acf", lambda arr: statsmodels.tsa.stattools.acf(
                    arr, nlags=1, fft=True)[1]),
            )
        )  # type: collections.OrderedDict[str,t.Callable[[np.ndarray], float]]

        stat_names = list(map("itrand_stat_{}".format, stats.keys()))

        if not set(stat_names).issubset(kwargs):
            stat_vals = cls._itrand_stat(ts=ts,
                                         func_stats=stats.values(),
                                         strategy=strategy,
                                         prop_rep=prop_rep,
                                         prop_interval=prop_interval,
                                         random_state=random_state,
                                         ts_scaled=ts_scaled)

            precomp_vals.update(zip(stat_names, stat_vals))

        return precomp_vals

    @classmethod
    def _itrand_stat(cls,
                     ts: np.ndarray,
                     func_stats: t.Collection[t.Callable[[np.ndarray], float]],
                     strategy: str = "dist-dynamic",
                     prop_rep: t.Union[int, float] = 2,
                     prop_interval: float = 0.1,
                     ts_scaled: t.Optional[np.ndarray] = None,
                     random_state: t.Optional[int] = None) -> np.ndarray:
        """Calculate global statistics with iterative perturbation method.

        In the iterative perturbation method, a copy of the time-series is
        modified at each iteration. The quantity of observations modified and
        the sample pool from which the new values are drawn depends on the
        ``strategy``selected. Then, a statistic is extracted after every `k`
        iterations (given by ceil(ts.size * ``prop_interval``)).

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        func_stats : sequence of callable
            Sequence of callables to extract the statistic values. Each
            callable must receive a list of numeric values as the first
            argument, and return a single numeric value.

        strategy : str, optional (default="dist-dynamic")
            The strategy used to perturb the current population. Must be one
            of the following:

                1. `dist-static`: (static distribution) one observation of the
                current population is overwritten by one observation from the
                original time-series.

                2. `dist-dynamic`: (dynamic distribution) one observation of
                the current population is overwritten by another observation
                of the current population.

                3. `permute`: two observations of the current population swaps
                its positions.

        prop_rep : int or float, optional (default=2)
            Number of total iterations proportional to the time-series size.
            This means that this process will iterate for approximately
            ceil(prop_rep * ts.size) iterations. More rigorously, the number
            of iterations also depends on the number of iterations that the
            statistics are extracted, to avoid lose computations.

        prop_interval : float, optional (default=0.1)
            Interval that the statistics are extracted from the current
            population, proportional to the time-series length.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        random_state : int, optional
            Random seed to ensure reproducibility.

        Returns
        -------
        :obj:`np.ndarray`
            Statistics extracted from the dynamic population. Each row is
            associated to a method from ``func_stats``, and each column is
            one distinct extraction event, ordered temporally by index (i.e.
            lower indices corresponds to populations more similar to the
            starting state, and higher indices to populations more affected
            by the process).

        References
        ----------
        .. [1] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework
            for Automated Time-Series Phenotyping Using Massive Feature
            Extraction, Cell Systems 5: 527 (2017).
            DOI: 10.1016/j.cels.2017.10.001
        .. [2] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
            time-series analysis: the empirical structure of time series and
            their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013).
        """
        if prop_rep <= 0:
            raise ValueError(
                "'prop_rep' must be positive (got {}).".format(prop_rep))

        if prop_interval <= 0:
            raise ValueError(
                "'prop_interval' must be positive (got {}).".format(
                    prop_interval))

        VALID_STRATEGY = ("dist-static", "dist-dynamic", "permute")

        if strategy not in VALID_STRATEGY:
            raise ValueError("'strategy' not in {} (got '{}')."
                             "".format(VALID_STRATEGY, strategy))

        if not hasattr(func_stats, "__len__"):
            func_stats = [func_stats]  # type: ignore

        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        rep_it = int(np.ceil(prop_interval * ts_scaled.size))

        # Note: adding (num_it % rep_it) to avoid lose computation of
        # the remaining iterations that do not produce a statistic.
        num_it = int(np.ceil(prop_rep * ts_scaled.size))
        num_it += (num_it % rep_it)

        res = np.zeros((len(func_stats), 1 + num_it // rep_it))
        ts_rnd = np.copy(ts_scaled)
        ts_src = ts_scaled if strategy == "dist-static" else ts_rnd
        swap = strategy == "permute"
        stat_ind = 0

        if random_state is not None:
            np.random.seed(random_state)

        inds_rnd = np.random.randint(ts_scaled.size, size=(num_it, 2))

        for it, (ind_a, ind_b) in enumerate(inds_rnd):
            if swap:
                ts_rnd[ind_a], ts_src[ind_b] = ts_src[ind_b], ts_rnd[ind_a]

            else:
                ts_rnd[ind_a] = ts_src[ind_b]

            if it % rep_it == 0:
                for ind_f, func in enumerate(func_stats):
                    res[ind_f, stat_ind] = func(ts_rnd)

                stat_ind += 1

        return res if len(func_stats) > 1 else res.ravel()

    @classmethod
    def ft_itrand_mean(
            cls,
            ts: np.ndarray,
            strategy: str = "dist-dynamic",
            prop_rep: t.Union[int, float] = 2,
            prop_interval: float = 0.1,
            ts_scaled: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None,
            itrand_stat_mean: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate time-series mean with iterative perturbation method.

        In the iterative perturbation method, a copy of the time-series is
        modified at each iteration. The quantity of observations modified and
        the sample pool from which the new values are drawn depends on the
        ``strategy``selected. Then, a statistic is extracted after every `k`
        iterations (given by ceil(ts.size * ``prop_interval``)).

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        strategy : str, optional (default="dist-dynamic")
            The strategy used to perturb the current population. Must be one
            of the following:

                1. `dist-static`: (static distribution) one observation of the
                current population is overwritten by one observation from the
                original time-series.

                2. `dist-dynamic`: (dynamic distribution) one observation of
                the current population is overwritten by another observation
                of the current population.

                3. `permute`: two observations of the current population swaps
                its positions.

        prop_rep : int or float, optional (default=2)
            Number of total iterations proportional to the time-series size.
            This means that this process will iterate for approximately
            ceil(prop_rep * ts.size) iterations. More rigorously, the number
            of iterations also depends on the number of iterations that the
            statistics are extracted, to avoid lose computations.

        prop_interval : float, optional (default=0.1)
            Interval that the statistics are extracted from the current
            population, proportional to the time-series length.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        random_state : int, optional
            Random seed to ensure reproducibility.

        itrand_stat_mean : :obj:`np.ndarray`, optional
            The return value of this method. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Mean extracted from the dynamic population. Each index corresponds
            to a extraction event, ordered temporally by index (i.e.  lower
            indices corresponds to populations more similar to the starting
            state, and higher indices to populations more affected by the
            process).

        References
        ----------
        .. [1] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework
            for Automated Time-Series Phenotyping Using Massive Feature
            Extraction, Cell Systems 5: 527 (2017).
            DOI: 10.1016/j.cels.2017.10.001
        .. [2] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
            time-series analysis: the empirical structure of time series and
            their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013).
        """
        if itrand_stat_mean is not None:
            return itrand_stat_mean

        res = cls._itrand_stat(ts=ts,
                               func_stats=np.mean,
                               strategy=strategy,
                               prop_rep=prop_rep,
                               prop_interval=prop_interval,
                               random_state=random_state,
                               ts_scaled=ts_scaled)

        return res

    @classmethod
    def ft_itrand_sd(
            cls,
            ts: np.ndarray,
            strategy: str = "dist-dynamic",
            prop_rep: t.Union[int, float] = 2,
            prop_interval: float = 0.1,
            ts_scaled: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None,
            itrand_stat_std: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Time-series standard deviation with iterative perturbation method.

        In the iterative perturbation method, a copy of the time-series is
        modified at each iteration. The quantity of observations modified and
        the sample pool from which the new values are drawn depends on the
        ``strategy``selected. Then, a statistic is extracted after every `k`
        iterations (given by ceil(ts.size * ``prop_interval``)).

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        strategy : str, optional (default="dist-dynamic")
            The strategy used to perturb the current population. Must be one
            of the following:

                1. `dist-static`: (static distribution) one observation of the
                current population is overwritten by one observation from the
                original time-series.

                2. `dist-dynamic`: (dynamic distribution) one observation of
                the current population is overwritten by another observation
                of the current population.

                3. `permute`: two observations of the current population swaps
                its positions.

        prop_rep : int or float, optional (default=2)
            Number of total iterations proportional to the time-series size.
            This means that this process will iterate for approximately
            ceil(prop_rep * ts.size) iterations. More rigorously, the number
            of iterations also depends on the number of iterations that the
            statistics are extracted, to avoid lose computations.

        prop_interval : float, optional (default=0.1)
            Interval that the statistics are extracted from the current
            population, proportional to the time-series length.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        random_state : int, optional
            Random seed to ensure reproducibility.

        itrand_stat_std : :obj:`np.ndarray`, optional
            The return value of this method. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Standard deviation extracted from the dynamic population. Each
            index corresponds to a extraction event, ordered temporally by
            index (i.e. lower indices corresponds to populations more similar
            to the starting state, and higher indices to populations more
            affected by the process).

        References
        ----------
        .. [1] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework
            for Automated Time-Series Phenotyping Using Massive Feature
            Extraction, Cell Systems 5: 527 (2017).
            DOI: 10.1016/j.cels.2017.10.001
        .. [2] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
            time-series analysis: the empirical structure of time series and
            their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013).
        """
        if itrand_stat_std is not None:
            return itrand_stat_std

        res = cls._itrand_stat(ts=ts,
                               func_stats=np.std,
                               strategy=strategy,
                               prop_rep=prop_rep,
                               prop_interval=prop_interval,
                               random_state=random_state,
                               ts_scaled=ts_scaled)

        return res

    @classmethod
    def ft_itrand_acf(
            cls,
            ts: np.ndarray,
            strategy: str = "dist-dynamic",
            prop_rep: t.Union[int, float] = 2,
            prop_interval: float = 0.1,
            ts_scaled: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None,
            itrand_stat_acf: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Time-series autocorrelation with iterative perturbation method.

        In the iterative perturbation method, a copy of the time-series is
        modified at each iteration. The quantity of observations modified and
        the sample pool from which the new values are drawn depends on the
        ``strategy``selected. Then, a statistic is extracted after every `k`
        iterations (given by ceil(ts.size * ``prop_interval``)).

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        strategy : str, optional (default="dist-dynamic")
            The strategy used to perturb the current population. Must be one
            of the following:

                1. `dist-static`: (static distribution) one observation of the
                current population is overwritten by one observation from the
                original time-series.

                2. `dist-dynamic`: (dynamic distribution) one observation of
                the current population is overwritten by another observation
                of the current population.

                3. `permute`: two observations of the current population swaps
                its positions.

        prop_rep : int or float, optional (default=2)
            Number of total iterations proportional to the time-series size.
            This means that this process will iterate for approximately
            ceil(prop_rep * ts.size) iterations. More rigorously, the number
            of iterations also depends on the number of iterations that the
            statistics are extracted, to avoid lose computations.

        prop_interval : float, optional (default=0.1)
            Interval that the statistics are extracted from the current
            population, proportional to the time-series length.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        random_state : int, optional
            Random seed to ensure reproducibility.

        itrand_stat_acf: :obj:`np.ndarray`, optional
            The return value of this method. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Autocorrelation extracted from the dynamic population. Each
            index corresponds to a extraction event, ordered temporally by
            index (i.e. lower indices corresponds to populations more similar
            to the starting state, and higher indices to populations more
            affected by the process).

        References
        ----------
        .. [1] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework
            for Automated Time-Series Phenotyping Using Massive Feature
            Extraction, Cell Systems 5: 527 (2017).
            DOI: 10.1016/j.cels.2017.10.001
        .. [2] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
            time-series analysis: the empirical structure of time series and
            their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013).
        """
        if itrand_stat_acf is not None:
            return itrand_stat_acf

        def func_acf(arr: np.ndarray) -> float:
            """Autocorrelation of the first lag."""
            return statsmodels.tsa.stattools.acf(arr, nlags=1, fft=True)[1]

        res = cls._itrand_stat(ts=ts,
                               func_stats=[func_acf],
                               strategy=strategy,
                               prop_rep=prop_rep,
                               prop_interval=prop_interval,
                               random_state=random_state,
                               ts_scaled=ts_scaled)

        return res

    @classmethod
    def ft_resample_std(
            cls,
            ts: np.ndarray,
            num_samples: int = 64,
            sample_size_frac: float = 0.1,
            ddof: int = 1,
            random_state: t.Optional[int] = None,
            ts_scaled: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Time-series standard deviation from repeated subsampling.

        A subsample of size L is L consecutive observations from the
        time-series, starting from a random index in [0, len(ts)-L] range.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        num_samples : int, optional (default=64)
            Number of time-series subsamples.

        sample_size_frac : float, optional (default=0.1)
            Size of each subsample proportional to the time-series length.

        ddof : int, optional (default=1)
            Degrees of freedom of the standard deviation.

        random_state : int, optional
            Random seed to ensure reproducibility.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Standard deviations from repeated subsampling.
        """
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        sample_std = _utils.apply_on_samples(ts=ts_scaled,
                                             func=np.std,
                                             num_samples=num_samples,
                                             sample_size_frac=sample_size_frac,
                                             random_state=random_state,
                                             ddof=ddof)

        return sample_std

    @classmethod
    def ft_resample_first_acf_nonpos(
            cls,
            ts: np.ndarray,
            num_samples: int = 128,
            sample_size_frac: float = 0.2,
            max_nlags: t.Optional[int] = None,
            unbiased: bool = True,
            random_state: t.Optional[int] = None) -> np.ndarray:
        """First non-positive autocorrelation lag using repeated subsampling.

        A subsample of size L is L consecutive observations from the
        time-series, starting from a random index in [0, len(ts)-L] range.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        num_samples : int, optional (default=64)
            Number of time-series subsamples.

        sample_size_frac : float, optional (default=0.2)
            Size of each subsample proportional to the time-series length.

        ddof : int, optional (default=1)
            Degrees of freedom of the first non-positive autocorrelation.

        random_state : int, optional
            Random seed to ensure reproducibility.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            First non-positive autocorrelation lags from repeated subsampling.
        """
        sample_acf_nonpos = _utils.apply_on_samples(
            ts=ts,
            func=autocorr.MFETSAutocorr.ft_acf_first_nonpos,
            num_samples=num_samples,
            sample_size_frac=sample_size_frac,
            random_state=random_state,
            max_nlags=max_nlags,
            unbiased=unbiased)

        return sample_acf_nonpos

    @classmethod
    def ft_resample_first_acf_locmin(
            cls,
            ts: np.ndarray,
            num_samples: int = 128,
            sample_size_frac: float = 0.2,
            max_nlags: t.Optional[int] = None,
            unbiased: bool = True,
            random_state: t.Optional[int] = None) -> np.ndarray:
        """First local minima autocorrelation lag using repeated subsampling.

        A subsample of size L is L consecutive observations from the
        time-series, starting from a random index in [0, len(ts)-L] range.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        num_samples : int, optional (default=64)
            Number of time-series subsamples.

        sample_size_frac : float, optional (default=0.2)
            Size of each subsample proportional to the time-series length.

        ddof : int, optional (default=1)
            Degrees of freedom of the first local minima autocorrelation.

        random_state : int, optional
            Random seed to ensure reproducibility.

        ts_scaled : :obj:`np.ndarray`, optional
            Standardized time-series values. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            First local minima autocorrelation lags from repeated subsampling.
        """
        sample_acf_locmin = _utils.apply_on_samples(
            ts=ts,
            func=autocorr.MFETSAutocorr.ft_first_acf_locmin,
            num_samples=num_samples,
            sample_size_frac=sample_size_frac,
            random_state=random_state,
            max_nlags=max_nlags,
            unbiased=unbiased)

        return sample_acf_locmin

    @classmethod
    def ft_surr_trev(
            cls,
            ts: np.ndarray,
            surrogate_num: int = 32,
            max_iter: int = 128,
            relative: bool = True,
            lag: t.Optional[t.Union[str, int]] = None,
            only_numerator: bool = False,
            random_state: t.Optional[int] = None,
            max_nlags: t.Optional[int] = None,
            detrended_acfs: t.Optional[np.ndarray] = None,
            detrended_ami: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Trev statistic extracted from surrogate time-series.

        The surrogate time-series are generated using the IAAFT algorithm.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        surrogate_num : int, optional (default=32)
            Number of surrogate time-series.

        max_iter : int, optional (default=128)
            Maximum number of iterations allowed before the convergence of the
            IAAFT algorithm process.

        relative : bool, optional (default=True)
            If True, the obtained statistics will be normalized by the
            statistic value extracted from the original time-series.

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

        only_numerator : bool, optional (default=False)
            If True, return only the numerator from this statistic definition.
            Check `autocorr.MFETSAutocorr.ft_trev` documentation for more
            information.

        random_state : int, optional
            Random seed to ensure reproducibility.

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

        Returns
        ------
        :obj:`np.ndarray`
            Trev statistic extracted from distinct generated surrogate
            time-series using the IAAFT algorithm.

        References
        ----------
        .. [1] Kugiumtzis, D.: Test your surrogate data before you test for
            nonlinearity, Phys. Rev. E, 60(3), 2808–2816, 1999.
        .. [2] Schreiber, T. and Schmitz, A.: Improved surrogate data for
            nonlinearity tests, Phys. Rev. Lett, 77, 635–638, 1996.
        .. [3] Schreiber, T. and Schmitz, A.: Surrogate time series, Physica
            D,142(3–4), 346–382, 2000.
        .. [4] Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., and
            Farmer, J.D.: Testing for nonlinearity in time series: the method
            of surrogate data, Physica D, 58, 77–94, 1992.
        .. [5] Theiler, J. and Prichard, D.: Constrained-realization
            Monte-Carlo method for hypothesis testing, Physica D, 94(4),
            221–235, 1996.
        .. [6] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework
            for Automated Time-Series Phenotyping Using Massive Feature
            Extraction, Cell Systems 5: 527 (2017).
            DOI: 10.1016/j.cels.2017.10.001
        .. [7] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
            time-series analysis: the empirical structure of time series and
            their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013).
        """
        lag = _embed.embed_lag(ts=ts,
                               lag=lag,
                               max_nlags=max_nlags,
                               detrended_acfs=detrended_acfs,
                               detrended_ami=detrended_ami)

        surr_trev = _surrogates.apply_on_surrogates(
            ts=ts,
            surrogate_num=surrogate_num,
            func=autocorr.MFETSAutocorr.ft_trev,
            max_iter=max_iter,
            random_state=random_state,
            only_numerator=only_numerator,
            lag=lag)

        if relative:
            surr_trev /= autocorr.MFETSAutocorr.ft_trev(
                ts=ts, lag=lag, only_numerator=only_numerator)

        return surr_trev

    @classmethod
    def ft_surr_tc3(
            cls,
            ts: np.ndarray,
            surrogate_num: int = 32,
            max_iter: int = 128,
            relative: bool = True,
            lag: t.Optional[t.Union[str, int]] = None,
            only_numerator: bool = False,
            random_state: t.Optional[int] = None,
            max_nlags: t.Optional[int] = None,
            detrended_acfs: t.Optional[np.ndarray] = None,
            detrended_ami: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Tc3 statistic extracted from surrogate time-series.

        The surrogate time-series are generated using the IAAFT algorithm.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        surrogate_num : int, optional (default=32)
            Number of surrogate time-series.

        max_iter : int, optional (default=128)
            Maximum number of iterations allowed before the convergence of the
            IAAFT algorithm process.

        relative : bool, optional (default=True)
            If True, the obtained statistics will be normalized by the
            statistic value extracted from the original time-series.

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

        only_numerator : bool, optional (default=False)
            If True, return only the numerator from this statistic definition.
            Check `autocorr.MFETSAutocorr.ft_tc3` documentation for more
            information.

        random_state : int, optional
            Random seed to ensure reproducibility.

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

        Returns
        ------
        :obj:`np.ndarray`
            Tc3 statistic extracted from distinct generated surrogate
            time-series using the IAAFT algorithm.

        References
        ----------
        .. [1] Kugiumtzis, D.: Test your surrogate data before you test for
            nonlinearity, Phys. Rev. E, 60(3), 2808–2816, 1999.
        .. [2] Schreiber, T. and Schmitz, A.: Improved surrogate data for
            nonlinearity tests, Phys. Rev. Lett, 77, 635–638, 1996.
        .. [3] Schreiber, T. and Schmitz, A.: Surrogate time series, Physica
            D,142(3–4), 346–382, 2000.
        .. [4] Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., and
            Farmer, J.D.: Testing for nonlinearity in time series: the method
            of surrogate data, Physica D, 58, 77–94, 1992.
        .. [5] Theiler, J. and Prichard, D.: Constrained-realization
            Monte-Carlo method for hypothesis testing, Physica D, 94(4),
            221–235, 1996.
        .. [6] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework
            for Automated Time-Series Phenotyping Using Massive Feature
            Extraction, Cell Systems 5: 527 (2017).
            DOI: 10.1016/j.cels.2017.10.001
        .. [7] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
            time-series analysis: the empirical structure of time series and
            their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013).
        """
        lag = _embed.embed_lag(ts=ts,
                               lag=lag,
                               max_nlags=max_nlags,
                               detrended_acfs=detrended_acfs,
                               detrended_ami=detrended_ami)

        surr_tc3 = _surrogates.apply_on_surrogates(
            ts=ts,
            surrogate_num=surrogate_num,
            func=autocorr.MFETSAutocorr.ft_tc3,
            max_iter=max_iter,
            random_state=random_state,
            only_numerator=only_numerator,
            lag=lag)

        if relative:
            surr_tc3 /= autocorr.MFETSAutocorr.ft_tc3(
                ts=ts, lag=lag, only_numerator=only_numerator)

        return surr_tc3
