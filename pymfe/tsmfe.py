"""Main module for extracting metafeatures from time-series."""
import typing as t
import collections
import shutil
import time

import texttable
import numpy as np

import pymfe._internal as _internal
import pymfe._period as _period
import pymfe._detrend as _detrend

_TypeSeqExt = t.Sequence[t.Tuple[str, t.Callable, t.Tuple[str, ...],
                                 t.Tuple[str, ...]]]
"""Type annotation for a sequence of TypeExtMtdTuple objects."""


class TSMFE:
    """Core class for time-series metafeature extraction.

    Attributes
    ----------
    ts : :obj:`Sequence`
        Independent attributes of the dataset.

    groups : :obj:`tuple` of :obj:`str`
        Tuple object containing fitted meta-feature groups loaded in the model
        at instantiation.

    features : :obj:`tuple` of :obj:`str`
        Contains loaded meta-feature extraction method names available for
        meta-feature extraction, from selected metafeatures groups and features
        listed at instantiation.

        summary : :obj:`tuple` of :obj:`str`
            Tuple object which contains summary functions names for features
            summarization.
    """
    groups_alias = [('default', _internal.DEFAULT_GROUP)]

    def __init__(self,
                 groups: t.Union[str, t.Iterable[str]] = "default",
                 features: t.Union[str, t.Iterable[str]] = "all",
                 summary: t.Union[str, t.Iterable[str]] = ("mean", "sd"),
                 measure_time: t.Optional[str] = None,
                 wildcard: str = "all",
                 score: str = "rmse",
                 num_cv_folds: int = 5,
                 lm_sample_frac: float = 1.0,
                 suppress_warnings: bool = False,
                 random_state: t.Optional[int] = None) -> None:
        """Provides easy access for metafeature extraction from datasets.

        It expected that user first calls ``fit`` method after instantiation
        and then ``extract`` for effectively extract the selected metafeatures.
        Check reference [1]_ for more information.

        Parameters
        ----------
        groups : :obj:`Iterable` of :obj:`str` or :obj:`str`
            A collection or a single metafeature group name representing the
            desired group of metafeatures for extraction. Use the method
            ``valid_groups`` to get a list of all available groups.

            Setting with ``all`` enables all available groups.

            The value provided by the argument ``wildcard`` can be used to
            select all metafeature groups rapidly.

        features : :obj:`Iterable` of :obj:`str` or :obj:`str`, optional
            A collection or a single metafeature name desired for extraction.
            Keep in mind that the extraction only gathers features also in the
            selected ``groups``. Check this class ``features`` attribute to get
            a list of available metafeatures from selected groups, or use the
            method ``valid_metafeatures`` to get a list of all available
            metafeatures filtered by group. Alternatively, you can use the
            method ``metafeature_description`` to get or print a table with
            all metafeatures with its respectives groups and descriptions.

            The value provided by the argument ``wildcard`` can be used to
            select all features from all selected groups rapidly.

        summary : :obj:`Iterable` of :obj:`str` or :obj:`str`, optional
            A collection or a single summary function to summarize a group of
            metafeature measures into a fixed-length group of value, typically
            a single value. The values must be one of the following:

                1. ``mean``: Average of the values.
                2. ``sd``: Standard deviation of the values.
                3. ``count``: Computes the cardinality of the measure. Suitable
                   for variable cardinality.
                4. ``histogram``: Describes the distribution of the measured
                   values. Suitable for high cardinality.
                5. ``iq_range``: Computes the interquartile range of the
                   measured values.
                6. ``kurtosis``: Describes the shape of the measures values
                   distribution.
                7. ``max``: Results in the maximum value of the measure.
                8. ``median``: Results in the central value of the measure.
                9. ``min``: Results in the minimum value of the measure.
                10. ``quantiles``: Results in the minimum, first quartile,
                    median, third quartile and maximum of the measured values.
                11. ``range``: Computes the range of the measured values.
                12. ``skewness``: Describes the shape of the measure values
                    distribution in terms of symmetry.

            You can concatenate `nan` with the desired summary function name
            to use an alternative version of the same summary which ignores
            `nan` values. For instance, `nanmean` is the `mean` summary
            function which ignores all `nan` values, while 'naniq_range`
            is the interquartile range calculated only with valid (non-`nan`)
            values.

            If more than one summary function is selected, then all multivalued
            extracted metafeatures are summarized with each summary function.

            The particular value provided by the argument ``wildcard`` can be
            used to select all summary functions rapidly.

            Use the method ``valid_summary`` to get a list of all available
            summary functions.

        measure_time : :obj:`str`, optional
            Options for measuring the time elapsed during metafeature
            extraction. If this argument value is :obj:`NoneType`, no time
            elapsed is measured. Otherwise, this argument must be a :obj:`str`
            valued as one of the options below:

                1. ``avg``: average time for each metafeature (total time
                   divided by the feature cardinality, i.e., number of features
                   extracted by a single feature-extraction related method),
                   without summarization time.
                2. ``avg_summ``: average time for each metafeature (total time
                   of extraction divided by feature cardinality) including
                   required time for summarization.
                3. ``total``: total time for each metafeature, without
                   summarization time.
                4. ``total_summ``: total time for each metafeature including
                   the required time for summarization.

            The ``cardinality`` of the feature is the number of values
            extracted by a single calculation method.

            For example, ``mean`` feature has cardinality equal to the number
            of numeric features in the dataset, where ``cor`` (from
            ``correlation``) has cardinality equals to (N - 1)/2, where N is
            the number of numeric features in the dataset.

            The cardinality is used to divide the total execution time of that
            method if an option starting with ``avg`` is selected.

            If a summary method has cardinality higher than one (more than one
            value returned after summarization and, thus, creating more than
            one entry in the result lists) like, for example, ``histogram``
            summary method, then the corresponding time of this summary will be
            inserted only in the first correspondent element of the time list.
            The remaining entries are all filled with 0 value, to keep
            consistency between the size of all lists returned and index
            correspondence between they.

        wildcard : :obj:`str`, optional
            Value used as ``select all`` for ``groups``, ``features`` and
            ``summary`` arguments.

        score : :obj:`str`, optional
            Score metric used to extract ``landmarking`` metafeatures.

        num_cv_folds : :obj:`int`, optional
            Number of folds to create a Stratified K-Fold cross
            validation to extract the ``landmarking`` metafeatures.

        lm_sample_frac : :obj:`float`, optional
            Sample proportion used to produce the ``landmarking`` metafeatures.
            This argument must be in 0.5 and 1.0 (both inclusive) interval.

        suppress_warnings : :obj:`bool`, optional
            If True, then ignore all warnings invoked at the instantiation
            time.

        random_state : :obj:`int`, optional
            Random seed used to control random events. Keeps the experiments
            reproducible.

        Notes
        -----
            .. [1] Rivolli et al. "Towards Reproducible Empirical
               Research in Meta-Learning,".
               Rivolli et al. URL: https://arxiv.org/abs/1808.10406

        Examples
        --------

        Load a dataset

        >>> from sklearn.datasets import load_iris
        >>> from pymfe.tsmfe import TSMFE

        >>> data = load_iris()
        >>> y = data.target
        >>> ts = data.data

        Extract all measures

        >>> mfe = TSMFE()
        >>> mfe.fit(ts, y)
        >>> ft = mfe.extract()
        >>> print(ft)

        Extract general, local-stat and information-theoretic measures

        >>> mfe = TSMFE(groups=["general", "local-stat", "info-theory"])
        >>> mfe.fit(ts, y)
        >>> ft = mfe.extract()
        >>> print(ft)

        """
        self.groups = _internal.process_generic_set(
            values=groups, group_name="groups",
            groups_alias=TSMFE.groups_alias,
            wildcard=wildcard)  # type: t.Tuple[str, ...]

        self.groups, self.inserted_group_dep = (
            _internal.solve_group_dependencies(
                groups=self.groups))

        proc_feat = _internal.process_features(
            features=features,
            groups=self.groups,
            suppress_warnings=suppress_warnings,
            wildcard=wildcard,
        )  # type: t.Tuple[t.Tuple[str, ...], _TypeSeqExt, t.Tuple[str, ...]]

        self.features, self._metadata_mtd_ft, self.groups = proc_feat
        del proc_feat

        self.summary, self._metadata_mtd_sm = _internal.process_summary(
            summary,
            wildcard=wildcard)  # type: t.Tuple[t.Tuple[str, ...], _TypeSeqExt]

        self.timeopt = _internal.process_generic_option(
            value=measure_time, group_name="timeopt",
            allow_none=True)  # type: t.Optional[str]

        self.ts = None  # type: t.Optional[np.ndarray]
        self.ts_trend = None  # type: t.Optional[np.ndarray]
        self.ts_season = None  # type: t.Optional[np.ndarray]
        self.ts_residuals = None  # type: t.Optional[np.ndarray]
        self.ts_detrended = None  # type: t.Optional[np.ndarray]
        self.ts_deseasonalized = None  # type: t.Optional[np.ndarray]
        self.ts_period = -1

        self._custom_args_ft = None  # type: t.Optional[t.Dict[str, t.Any]]
        """User-independent arguments for ft. methods (e.g. ``ts``)"""

        self._custom_args_sum = None  # type: t.Optional[t.Dict[str, t.Any]]
        """User-independent arguments for summary functions methods."""

        self._precomp_args_ft = None  # type: t.Optional[t.Dict[str, t.Any]]
        """Precomputed common feature-extraction method arguments."""

        self._postprocess_args_ft = {}  # type: t.Dict[str, t.Any]
        """User-independent arguments for post-processing methods."""

        if random_state is None or isinstance(random_state, int):
            self.random_state = random_state
            np.random.seed(random_state)

        else:
            raise ValueError(
                'Invalid "random_state" argument ({0}). '
                'Expecting None or an integer.'.format(random_state))

        if isinstance(num_cv_folds, int):
            self.num_cv_folds = num_cv_folds

        else:
            raise ValueError('Invalid "num_cv_folds" argument ({0}). '
                             'Expecting an integer.'.format(random_state))

        if isinstance(lm_sample_frac, int):
            lm_sample_frac = float(lm_sample_frac)

        if isinstance(lm_sample_frac, float)\
           and 0.5 <= lm_sample_frac <= 1.0:
            self.lm_sample_frac = lm_sample_frac

        else:
            raise ValueError('Invalid "lm_sample_frac" argument ({0}). '
                             'Expecting an float [0.5, 1].'
                             .format(random_state))

        self.score = _internal.check_score(score, self.groups)

        # """Total time elapsed for precomputations."""
        self.time_precomp = -1.0

        # """Total time elapsed for metafeature extraction."""
        self.time_extract = -1.0

        # """Total time elapsed in total (precomp + extract.)"""
        self.time_total = -1.0

    def _call_summary_methods(
            self,
            feature_values: t.Sequence[_internal.TypeNumeric],
            feature_name: str,
            verbose: int = 0,
            suppress_warnings: bool = False,
            **kwargs
    ) -> t.Tuple[t.List[str], t.List[t.Union[float, t.Sequence]], t.
                 List[float]]:
        """Invoke summary functions loaded in the model on given feature
        values.

        Parameters
        ----------
        feature_values : :obj:`sequence` of numerics
            Sequence containing values from feature-extraction methods.

        feature_name : :obj:`str`
            Name of the feature method used for produce the ``feature_value.``

        verbose : :obj:`int`, optional
            Select the verbosity level of the summarization process.
            If == 1, then print just the ending message, without a line break.
            If >= 2, then messages about the summarization process may be
            printed. Note that there is no relation between this argument and
            warnings (see ``suppress_warnings`` argument below).

        suppress_warnings : :obj:`bool`, optional
            If True, ignore all warnings invoked before and after summary
            method calls. The summary callables may still invoke warnings by
            itself and the user need to ignore them, if possible, via kwargs.

        kwargs:
            User-defined arguments for the summary callables.

        Returns
        -------
        :obj:`tuple`(:obj:`list`, :obj:`list`, :obj:`list`)
            A tuple containing three lists.

            The first field is the identifiers of each summarized value in the
            form ``feature_name.summary_mtd_name`` (i.e., the feature
            extraction name concatenated by the summary method name, separated
            by a dot). If the summary function return more than one value
            (cardinality greater than 1), then each value name have an extra
            concatenated id starting from 0 to differ between values (i.e.
            ``feature_name.summary_mtd_name.id``).

            The second field is the summarized values. Both lists have a 1-1
            correspondence by the index of each element (i.e., the value at
            index ``i`` in the second list has its identifier at the same index
            in the first list and vice-versa).

            The third field is a list with measured time wasted by each summary
            function. If the cardinality of the summary function is greater
            than 1, then the correspondent measured time is kept only in the
            first correspondent field, and the extra fields are filled with 0
            to keep the consistency of the size between all lists.

                Example:
                    ([``attr_ent.mean``, ``attr_ent.sd``], [0.98346, 0.34436])
                    is the return value for the feature `attr_end` summarized
                    by both ``mean`` and ``sd`` (standard deviation), giving
                    the values ``0.98347`` and ``0.34436``, respectively.
        """
        metafeat_vals = []  # type: t.List[t.Union[int, float, t.Sequence]]
        metafeat_names = []  # type: t.List[str]
        metafeat_times = []  # type: t.List[float]

        for cur_metadata in self._metadata_mtd_sm:
            sm_mtd_name, sm_mtd_callable, sm_mtd_args, _ = cur_metadata

            if verbose >= 2:
                print(
                    " {} Summarizing '{}' feature with '{}' summary "
                    "function...".format(
                        _internal.VERBOSE_BLOCK_MID_SYMBOL,
                        feature_name,
                        sm_mtd_name),
                    end=" ")

            sm_mtd_args_pack = _internal.build_mtd_kwargs(
                mtd_name=sm_mtd_name,
                mtd_args=sm_mtd_args,
                mtd_mandatory=set(),
                user_custom_args=kwargs.get(sm_mtd_name),
                inner_custom_args=self._custom_args_sum,
                suppress_warnings=suppress_warnings)

            summarized_val, time_sm = _internal.timeit(
                _internal.summarize, feature_values, sm_mtd_callable,
                sm_mtd_args_pack)

            if not suppress_warnings:
                _internal.check_summary_warnings(
                    value=summarized_val,
                    name_feature=feature_name,
                    name_summary=sm_mtd_name)

            if isinstance(summarized_val, np.ndarray):
                summarized_val = summarized_val.flatten().tolist()

            if (isinstance(summarized_val, collections.Sequence)
                    and not isinstance(summarized_val, str)):
                metafeat_vals += summarized_val
                metafeat_names += [
                    ".".join((feature_name, sm_mtd_name, str(i)))
                    for i in range(len(summarized_val))
                ]
                metafeat_times += ([time_sm] + (
                    (len(summarized_val) - 1) * [0.0]))

            else:
                metafeat_vals.append(summarized_val)
                metafeat_names.append(".".join((feature_name, sm_mtd_name)))
                metafeat_times.append(time_sm)

            if verbose >= 2:
                print("Done.")

        if verbose >= 2:
            print(" {} Done summarizing '{}' feature.".format(
                _internal.VERBOSE_BLOCK_END_SYMBOL,
                feature_name))

        return metafeat_names, metafeat_vals, metafeat_times

    def _call_feature_methods(
            self,
            verbose: int = 0,
            # enable_parallel: bool = False,
            suppress_warnings: bool = False,
            **kwargs) -> t.Tuple[t.List[str],
                                 t.List[t.Union[int, float, t.Sequence]],
                                 t.List[float]]:
        """Invoke feature methods loaded in the model and gather results.

        The returned values are already summarized if needed.

        For more information, check ``extract`` method documentation for
        in-depth information about arguments and return value.
        """
        metafeat_vals = []  # type: t.List[t.Union[int, float, t.Sequence]]
        metafeat_names = []  # type: t.List[str]
        metafeat_times = []  # type: t.List[float]

        skipped_count = 0
        for ind, cur_metadata in enumerate(self._metadata_mtd_ft, 1):
            (ft_mtd_name, ft_mtd_callable,
             ft_mtd_args, ft_mandatory) = cur_metadata

            ft_name_without_prefix = _internal.remove_prefix(
                value=ft_mtd_name, prefix=_internal.MTF_PREFIX)

            try:
                ft_mtd_args_pack = _internal.build_mtd_kwargs(
                    mtd_name=ft_name_without_prefix,
                    mtd_args=ft_mtd_args,
                    mtd_mandatory=ft_mandatory,
                    user_custom_args=kwargs.get(ft_name_without_prefix),
                    inner_custom_args=self._custom_args_ft,
                    precomp_args=self._precomp_args_ft,
                    suppress_warnings=suppress_warnings)

            except RuntimeError:
                # Not all method's mandatory arguments were satisfied.
                # Skip the current method.
                if verbose >= 2:
                    print("\nSkipped '{}' ({} of {}).".format(
                        ft_mtd_name, ind, len(self._metadata_mtd_ft)))

                skipped_count += 1
                continue

            if verbose >= 2:
                print("\nExtracting '{}' feature ({} of {})..."
                      .format(ft_mtd_name, ind, len(self._metadata_mtd_ft)))

            features, time_ft = _internal.timeit(
                _internal.get_feat_value, ft_mtd_name, ft_mtd_args_pack,
                ft_mtd_callable, suppress_warnings)

            ft_has_length = isinstance(features,
                                       (np.ndarray, collections.Sequence))

            if ft_has_length and self._timeopt_type_is_avg():
                time_ft /= len(features)

            if self._metadata_mtd_sm and ft_has_length:
                sm_ret = self._call_summary_methods(
                    feature_values=features,
                    feature_name=ft_name_without_prefix,
                    verbose=verbose,
                    suppress_warnings=suppress_warnings,
                    **kwargs)

                summarized_names, summarized_vals, times_sm = sm_ret

                metafeat_vals += summarized_vals
                metafeat_names += summarized_names
                metafeat_times += self._combine_time(time_ft, times_sm)

            else:
                metafeat_vals.append(features)
                metafeat_names.append(ft_name_without_prefix)
                metafeat_times.append(time_ft)

            if verbose > 0:
                _internal.print_verbose_progress(
                    cur_progress=100 * ind / len(self._metadata_mtd_ft),
                    cur_mtf_name=ft_mtd_name,
                    item_type="feature",
                    verbose=verbose)

        if verbose == 1:
            _t_num_cols, _ = shutil.get_terminal_size()
            print("\r{:<{fill}}".format(
                "Process of metafeature extraction finished.",
                fill=_t_num_cols))

        if verbose >= 2 and skipped_count > 0:
            print("\nNote: skipped a total of {} metafeatures, "
                  "out of {} ({:.2f}%).".format(
                      skipped_count,
                      len(self._metadata_mtd_ft),
                      100 * skipped_count / len(self._metadata_mtd_ft)))

        return metafeat_names, metafeat_vals, metafeat_times

    def _timeopt_type_is_avg(self) -> bool:
        """Checks if user selected time option is an ``average`` type."""
        return (isinstance(self.timeopt, str)
                and self.timeopt.startswith(_internal.TIMEOPT_AVG_PREFIX))

    def _timeopt_include_summary(self) -> bool:
        """Checks if user selected time option includes ``summary`` time."""
        return (isinstance(self.timeopt, str)
                and self.timeopt.endswith(_internal.TIMEOPT_SUMMARY_SUFFIX))

    def _combine_time(self, time_ft: float,
                      times_sm: t.List[float]) -> t.List[float]:
        """Treat time from feature extraction and summarization based in
        ``timeopt``.

        Parameters
        ----------
        time_ft : :obj:`float`
            Time necessary to extract some feature.

        times_sm : :obj:`list` of :obj:`float`
            List of values to summarize the metafeature value with each summary
            function.

        Returns
        -------
        :obj:`list`
            If ``timeopt`` attribute considers ``summary`` time (i.e., selected
            option ends with ``summ``), then these returned list values are the
            combination of times gathered in feature extraction and
            summarization methods. Otherwise, the list values are the value of
            ``time_ft`` copied ``len(times_sm)`` times, to keep consistency
            with the correspondence between the values of all lists returned by
            ``extract`` method.
        """
        total_time = np.array([time_ft] * len(times_sm))

        if self._timeopt_include_summary():
            total_time += times_sm

        # As seen in ``_call_summary_methods`` method documentation, zero-
        # valued elements are created to fill the time list to keep its size
        # consistent with another feature extraction related lists. In this
        # case, here they're kept zero-valued.
        total_time[np.array(times_sm) == 0.0] = 0.0

        return total_time.tolist()

    def fit(self,
            ts: t.Sequence[int],
            ts_period: t.Optional[int] = None,
            rescale: t.Optional[str] = None,
            rescale_args: t.Optional[t.Dict[str, t.Any]] = None,
            precomp_groups: t.Optional[str] = "all",
            wildcard: str = "all",
            suppress_warnings: bool = False,
            verbose: int = 0,
            **kwargs,
            ) -> "TSMFE":
        """Fits dataset into an TSMFE model.

        Parameters
        ----------
        ts : :obj:`Sequence`
            Predictive attributes of the dataset.

        ts_period : int, optional
            Period of the time-series. If not given, it will be estimated
            from the autocorrelation function.

        rescale : :obj:`str`, optional
            If :obj:`NoneType`, the model keeps all numeric data with its
            original values. Otherwise, this argument can assume one of the
            string options below to rescale all numeric values:

                1. ``standard``: set numeric data to zero mean, unit variance.
                   Also known as ``z-score`` normalization. Check the
                   documentation of ``sklearn.preprocessing.StandardScaler``
                   for in-depth information.

                2. `'min-max``: set numeric data to interval [a, b], a < b. It
                   is possible to define values to ``a`` and ``b`` using
                   argument ``rescale_args``. The default values are a = 0.0
                   and b = 1.0. Check ``sklearn.preprocessing.MinMaxScaler``
                   documentation for more information.

                3. ``robust``: rescale data using statistics robust to the
                   presence of outliers. For in-depth information, check
                   documentation of ``sklearn.preprocessing.RobustScaler``.

        rescale_args : :obj:`dict`, optional
            Dictionary containing parameters for rescaling data. Used only if
            ``rescale`` argument is not :obj:`NoneType`. These dictionary keys
            are the parameter names as strings and the values, the
            corresponding parameter value.

        precomp_groups : :obj:`str`, optional
            Defines which metafeature groups common values should be cached to
            share among various meta-feature extraction related methods (e.g.
            ``classes``, or ``covariance``). This argument may speed up
            meta-feature extraction but also consumes more memory, so it may
            not be suitable for huge datasets.

        wildcard : :obj:`str`, optional
            Value used as ``select all`` for ``precomp_groups``.

        suppress_warnings : :obj:`bool`, optional
            If True, ignore all warnings invoked while fitting dataset.

        verbose : :obj:`int`, optional
            Defines the level of verbosity for the fit method. If `1`, then
            print a progress bar related to the precomputations. If `2` or
            higher, then log every step of the fitted data transformations and
            the precomputation steps.

        **kwargs:
            Extra custom arguments to the precomputation methods. Keep in
            mind that those values may even replace internal custom parameters,
            if the name matches. Use this resource carefully.

            Hint: you can check which are the internal custom arguments by
            verifying the values in '._custom_args_ft' attribute after the
            model is fitted.

            This argument format is {'parameter_name': parameter_value}.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the number of rows of ts and y length does not match.
        TypeError
            If ts is neither a :obj:`list` or a :obj:`np.ndarray` object.

        """
        rescale = _internal.process_generic_option(
            value=rescale, group_name="rescale", allow_none=True)

        if verbose >= 2:
            print("Fitting data into model... ", end="")

        self.ts = _internal.check_data(ts)

        if rescale is not None:
            self.ts = _internal.rescale_data(
                data=self.ts.reshape(-1, 1),
                option=rescale,
                args=rescale_args).ravel()

        if verbose >= 2:
            print("Done.")
            print("Getting time-series period... ", end="")

        self.ts_period = _period.get_ts_period(ts=self.ts, ts_period=ts_period)

        if verbose >= 2:
            print("Done (got period {}).".format(ts_period))
            print("Started time-series decomposition... ", end="")

        _ts_components = _detrend.decompose(self.ts, ts_period=self.ts_period)

        self.ts_trend, self.ts_season, self.ts_residuals = _ts_components

        self.ts_detrended = self.ts - self.ts_trend
        self.ts_deseasonalized = self.ts - self.ts_season

        if verbose >= 2:
            print("Done.")

        # Custom arguments for metafeature extraction methods
        self._custom_args_ft = {
            "ts": self.ts,
            "ts_trend": self.ts_trend,
            "ts_season": self.ts_season,
            "ts_residuals": self.ts_residuals,
            "ts_detrended": self.ts_detrended,
            "ts_deseasonalized": self.ts_deseasonalized,
            "ts_period": self.ts_period,
            "num_cv_folds": self.num_cv_folds,
            "lm_sample_frac": self.lm_sample_frac,
            "score": self.score,
            "random_state": self.random_state,
        }

        if verbose >= 2:
            print("Started precomputation process.")

        _time_start = time.time()

        # Custom arguments from preprocessing methods
        self._precomp_args_ft = _internal.process_precomp_groups(
            precomp_groups=precomp_groups,
            groups=self.groups,
            wildcard=wildcard,
            suppress_warnings=suppress_warnings,
            verbose=verbose,
            **{**self._custom_args_ft, **kwargs})

        self.time_precomp = time.time() - _time_start

        if verbose >= 2:
            print("\nFinished precomputation process.",
                  " {} Total time elapsed: {:.8f} seconds".format(
                      _internal.VERBOSE_BLOCK_MID_SYMBOL,
                      self.time_precomp),
                  " {} Got a total of {} precomputed values.".format(
                      _internal.VERBOSE_BLOCK_END_SYMBOL,
                      len(self._precomp_args_ft)),
                  sep="\n")

        # Custom arguments for postprocessing methods
        self._postprocess_args_ft = {
            "inserted_group_dep": self.inserted_group_dep,
        }

        # Custom arguments for summarization methods
        self._custom_args_sum = {
            "ddof": 1,
        }

        return self

    def extract(
            self,
            verbose: int = 0,
            enable_parallel: bool = False,
            suppress_warnings: bool = False,
            **kwargs) -> t.Tuple[t.Sequence, ...]:
        """Extracts metafeatures from the previously fitted dataset.

        Parameters
        ----------
        verbose : :obj:`int`, optional
            Defines the verbosity level related to the metafeature extraction.
            If == 1, show just the current progress, without line breaks.
            If >= 2, print all messages related to the metafeature extraction
            process.

            Note that warning messages are not affected by this option (see
            ``suppress_warnings`` argument below).

        enable_parallel : :obj:`bool`, optional
            If True, then the meta-feature extraction is done with
            multi-processes. Currently, this argument has no effect by now
            (to be implemented).

        suppress_warnings : :obj:`bool`, optional
            If True, do not show warnings about unknown user custom parameters
            for feature extraction and summary methods passed via kwargs. Note
            that both feature extraction and summary methods may still raise
            warnings by itself.

        kwargs:
            Used to pass custom arguments for both feature-extraction and
            summary methods. The expected format is the following:

            {``mtd_name``: {``arg_name``: arg_value, ...}, ...}

            In words, the key values of ``**kwargs`` should be the target
            methods which receives the custom arguments, and each method has
            another dictionary containing customs method argument names as keys
            and their correspondent values, as values. See ``Examples``
            subsection for a clearer explanation.

            For more information see Examples.

        Returns
        -------
        :obj:`tuple`(:obj:`list`, :obj:`list`)
            A tuple containing two lists (if ``measure_time`` is None.)

            The first field is the identifiers of each summarized value in the
            form ``feature_name.summary_mtd_name`` (i.e., the feature
            extraction name concatenated by the summary method name, separated
            by a dot).

            The second field is the summarized values.

            Both lists have a 1-1 correspondence by the index of each element
            (i.e., the value at index ``i`` in the second list has its
            identifier at the same index in the first list and vice-versa).

            Example:
                ([``attr_ent.mean``, ``attr_ent.sd``], [``0.983``, ``0.344``])
                is the return value for the feature ``attr_end`` summarized by
                both ``mean`` and ``sd`` (standard deviation), giving the valu-
                es ``0.983`` and ``0.344``, respectively.

            if ``measure_time`` is given during the model instantiation, a
            third list will be returned with the time spent during the
            calculations for the corresponding (by index) metafeature.

        Raises
        ------
        TypeError
            If calling ``extract`` method before ``fit`` method.

        Examples
        --------
        Using kwargs. Option 1 to pass ft. extraction custom arguments:

        >>> args = {
        >>> 'sd': {'ddof': 2},
        >>> '1NN': {'metric': 'minkowski', 'p': 2},
        >>> 'leaves': {'max_depth': 4},
        >>> }

        >>> model = TSMFE().fit(ts=data)
        >>> result = model.extract(**args)

        Option 2 (note: metafeatures with name starting with numbers are not
        allowed!):

        >>> model = TSMFE().fit(ts=data)
        >>> res = extract(sd={'ddof': 2}, leaves={'max_depth': 4})

        """
        if self.ts is None:
            raise TypeError("Fitted data not found. Call "
                            '"fit" method before "extract".')

        if not isinstance(self.ts, np.ndarray):
            self.ts = _internal.check_data(self.ts)

        if verbose >= 2:
            print("Started the metafeature extraction process.")

        _time_start = time.time()

        results = self._call_feature_methods(
            verbose=verbose,
            enable_parallel=enable_parallel,
            suppress_warnings=suppress_warnings,
            **kwargs)  # type: t.Tuple[t.List, ...]

        _internal.post_processing(
            results=results,
            groups=self.groups,
            suppress_warnings=suppress_warnings,
            **self._postprocess_args_ft,
            **kwargs)

        self.time_extract = time.time() - _time_start
        self.time_total = self.time_extract + self.time_precomp

        if results and results[0]:
            # Sort results by metafeature name
            results = tuple(
                map(list, zip(*sorted(zip(*results),
                                      key=lambda item: item[0]))))

        res_names, res_vals, res_times = results

        if verbose >= 2:
            _ext_t_pct = 100 * self.time_extract / self.time_total
            print(
                "\nMetafeature extraction process done.",
                " {} Time elapsed in total (precomputations + extraction): "
                "{:.8f} seconds.".format(
                    _internal.VERBOSE_BLOCK_MID_SYMBOL, self.time_total),
                " {} Time elapsed for extractions: {:.8f} seconds ({:.2f}% "
                "from the total).".format(
                    _internal.VERBOSE_BLOCK_MID_SYMBOL,
                    self.time_extract,
                    _ext_t_pct),
                " {} Time elapsed for precomputations: {:.8f} seconds "
                "({:.2f}% from the total).".format(
                    _internal.VERBOSE_BLOCK_MID_SYMBOL,
                    self.time_precomp, 100 - _ext_t_pct),
                " {} Total of {} values obtained.".format(
                    _internal.VERBOSE_BLOCK_END_SYMBOL, len(res_vals)),
                sep="\n")

        if self.timeopt:
            return res_names, res_vals, res_times

        return res_names, res_vals

    def _extract_with_bootstrap(self,
                                extractor: "TSMFE",
                                sample_num: int,
                                arguments_fit: t.Dict[str, t.Any],
                                arguments_extract: t.Dict[str, t.Any],
                                verbose: int = 0) -> t.Tuple[np.ndarray, ...]:
        """Extract metafeatures using bootstrapping."""
        if self.ts is None:
            raise TypeError("Fitted data not found. Please call 'fit' "
                            "method first.")

        def _handle_extract_ret(
                res: t.Tuple[np.ndarray, ...],
                args: t.Tuple[t.Sequence, ...],
                it_num: int) -> t.Tuple[np.ndarray, ...]:
            """Handle each .extraction method return value."""
            mtf_names, mtf_vals, mtf_time = res

            if not self.timeopt:
                cur_mtf_names, cur_mtf_vals = args

            else:
                cur_mtf_names, cur_mtf_vals, cur_mtf_time = args

            if mtf_names.size:
                mtf_vals[:, it_num] = cur_mtf_vals

                if self.timeopt:
                    mtf_time += cur_mtf_time

            else:
                mtf_names = np.asarray(cur_mtf_names, dtype=str)
                mtf_vals = np.zeros(
                    (len(cur_mtf_vals), sample_num), dtype=float)
                mtf_vals[:, 0] = cur_mtf_vals

                if self.timeopt:
                    mtf_time = np.asarray(cur_mtf_time, dtype=float)

            return mtf_names, mtf_vals, mtf_time

        res = 3 * (np.array([]),)

        if self.random_state is None:
            # Enforce pseudo-random behaviour to avoid previously set
            # random seeds out of this context
            np.random.seed()

        bootstrap_random_state = (
            self.random_state
            if self.random_state is not None
            else np.random.randint(2 ** 20 - 1))

        for it_num in np.arange(sample_num):
            if verbose > 0:
                print("Extracting from sample dataset {} of {} ({:.2f}%)..."
                      .format(1 + it_num,
                              sample_num,
                              100.0 * (1 + it_num) / sample_num))

            # Note: setting random state to prevent same sample indices due
            # to random states set during fit/extraction
            np.random.seed(bootstrap_random_state)
            bootstrap_random_state += 1

            sample_inds = np.random.randint(
                self.ts.shape[0],
                size=self.ts.shape[0])

            ts_sample = self.ts[sample_inds, :]

            extractor.fit(ts_sample, **arguments_fit)

            res = _handle_extract_ret(
                res=res,
                args=extractor.extract(**arguments_extract),
                it_num=it_num)

            if verbose > 0:
                print("Done extracting from sample dataset {}.\n"
                      .format(1 + it_num))

        return res

    def extract_with_confidence(
            self,
            sample_num: int = 128,
            confidence: t.Union[float, t.Sequence[float]] = 0.95,
            return_avg_val: bool = True,
            arguments_fit: t.Optional[t.Dict[str, t.Any]] = None,
            arguments_extract: t.Optional[t.Dict[str, t.Any]] = None,
            verbose: int = 0,
    ) -> t.Tuple[t.List, ...]:
        """Extract metafeatures with confidence intervals.

        To build the confidence intervals, each metafeature is extracted
        ``sample_num`` times from a distinct dataset built from the
        fitted data using bootstrap.

        All configuration used by this method are from the configuration
        while instantiating the current model.

        Parameters
        ----------
        sample_num : int, optional
            Number of samples from the fitted data using bootstrap. Each
            metafeature will be extracted ``sample_num`` times.

        confidence : float or sequence of floats, optional
            Confidence level of the interval. Must be in (0.0, 1.0) range.
            If a sequence of confidence levels is given, a confidence
            interval will be extracted for all values. Each confidence
            interval will be calculated as [confidence/2, 1 - confidence/2].

        return_avg_vals : bool, optional
            If True, return the average value for both the metafeature
            values and the time elapsed for its extraction (if any
            ``measure_time`` option was chosen.) If False, then all
            extracted metafeature values are returned as a 2D numpy array
            of shape (`metafeature_num`, `sample_num`) (i.e., each row
            represents a distinct metafeature, and each column is the
            value of the corresponding metafeature extracted from a
            distinct sample dataset) and the time elapsed will be the
            sum of all extractions for each metafeature.

        arguments_fit : dict, optional
            Extra arguments for the fit method for each sampled dataset.
            See ``.fit`` method documentation for more information.

        arguments_extract : dict, optional
            Extra arguments for each metafeature extraction procedure.
            See ``.extract`` method documentation for more information.

        verbose : int, optional
            Verbosity level for this method. Please note that the
            verbosity level for both ``.fit`` and ``.extract`` methods
            performed within this method must be controlled separately
            using, respectively, ``arguments_fit`` and ``arguments_extract``
            parameters.

        Returns
        -------
        tuple of :obj:`np.ndarray`
            The same return value format of the ``extract`` method, appended
            with the confidence intervals as a new sequence of values in the
            form (interval_low_1, interval_low_2, ..., interval_high_(n-1),
            interval_high_n) for each corresponding metafeature, and with shape
            (`metafeature_num`, 2 * C), where `C` is the number of confidence
            levels given in ``confidence`` (i.e., the rows represents each
            metafeature and the columns each interval limit). This means that
            all interval lower limits are given first, and all the interval
            upper limits are grouped together afterwards. The sequence order
            of the interval limits follows the same sequence order of the
            confidence levels given in ``confidence``. For instance, if
            `confidence=[0.80, 0.90, 0.99]`, then the confidence intervals
            will be returned in the following order (for all metafeatures):
            (lower_0.80, lower_0.90, lower_0.99, upper_0.80, upper_0.90,
            upper_0.99).

            if ``return_avg_val`` is True, the metafeature values and the
            time elapsed for extraction for each item (if any ``measure_time``
            options was chosen) will be the average value between all
            extractions. Otherwise, all extracted metafeature values will be
            returned as a 2D numpy array (where each columns is from a distinct
            sampled dataset, and each row is a distinct metafeature), and the
            time elapsed will be the sum of all extractions for the
            corresponding metafeature.

        Raises
        ------
        ValueError
            If ``confidence`` is not in (0.0, 1.0) range.

        Notes
        -----
        The model used to fit and extract metafeatures for each sampled
        dataset is instantiated within this method and, therefore, this
        method does not affect the current model (if any) by any means.
        """
        _confidence = np.asarray(confidence, dtype=float)

        if np.any(np.logical_or(_confidence <= 0.0, _confidence >= 1.0)):
            raise ValueError("'_confidence' must be in (0.0, 1.0) range (got "
                             "{}.)".format(_confidence))

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if arguments_fit is None:
            arguments_fit = {}

        if arguments_extract is None:
            arguments_extract = {}

        # Note: the metafeature extraction random seed will be fixed due
        # to the random indices while bootstrapping the fitted data.
        _random_state = self.random_state if self.random_state else 1234

        if verbose > 0:
            print("Started metafeature extract with _confidence interval.")
            print("Random seed:")
            print(" {} For extractor model: {}{}".format(
                _internal.VERBOSE_BLOCK_END_SYMBOL,
                _random_state,
                "" if self.random_state else " (chosen by default)"))

            print(" {} For bootstrapping: {}".format(
                _internal.VERBOSE_BLOCK_END_SYMBOL, self.random_state))

        extractor = TSMFE(
            features=self.features,
            groups=self.groups,
            summary=self.summary,
            measure_time=self.timeopt,
            random_state=_random_state)

        mtf_names, mtf_vals, mtf_time = self._extract_with_bootstrap(
            extractor=extractor,
            sample_num=sample_num,
            verbose=verbose,
            arguments_fit=arguments_fit,
            arguments_extract=arguments_extract)

        if verbose > 0:
            print("Finished metafeature extract with _confidence interval.")
            print("Now getting _confidence intervals...", end=" ")

        _half_sig_level = 0.5 * (1.0 - _confidence)
        quantiles = np.hstack((_half_sig_level, 1.0 - _half_sig_level))
        mtf_conf_int = np.quantile(a=mtf_vals, q=quantiles, axis=1).T

        if verbose > 0:
            print("Done.")

        if return_avg_val:
            mtf_vals = np.nanmean(mtf_vals, axis=1)

        if self.timeopt:
            if return_avg_val:
                mtf_time /= sample_num

            return mtf_names, mtf_vals, mtf_time, mtf_conf_int

        return mtf_names, mtf_vals, mtf_conf_int

    @classmethod
    def valid_groups(cls) -> t.Tuple[str, ...]:
        """Return a tuple of valid metafeature groups.

        Notes
        -----
        The returned ``groups`` are not related to the groups fitted in
        the model in the model instantation. The returned groups are all
        available metafeature groups in the ``Pymfe`` package. Check the
        ``TSMFE`` documentation for deeper information.
        """
        return _internal.VALID_GROUPS

    @classmethod
    def valid_summary(cls) -> t.Tuple[str, ...]:
        """Return a tuple of valid summary functions.

        Notes
        -----
        The returned ``summaries`` are not related to the summaries fitted
        in the model in the model instantation. The returned summaries are
        all available in the ``Pymfe`` package. Check the documentation of
        ``TSMFE`` for deeper information.
        """
        return _internal.VALID_SUMMARY

    @classmethod
    def _check_groups_type(cls,
                           groups: t.Optional[t.Union[str, t.Iterable[str]]]
                           ) -> t.Set[str]:
        """Cast ``groups`` to a tuple of valid metafeature group names."""
        if groups is None:
            return set(_internal.VALID_GROUPS)

        groups = _internal.convert_alias(TSMFE.groups_alias, groups)

        return set(groups)

    @classmethod
    def _filter_groups(cls,
                       groups: t.Set[str]
                       ) -> t.Set[str]:
        """Filter given groups by the available metafeature group names."""
        filtered_group_set = {
            group for group in groups
            if group in _internal.VALID_GROUPS
        }
        return filtered_group_set

    @classmethod
    def valid_metafeatures(
            cls,
            groups: t.Optional[t.Union[str, t.Iterable[str]]] = None,
    ) -> t.Tuple[str, ...]:
        """Return a tuple with all metafeatures related to given ``groups``.

        Parameters
        ----------
        groups : :obj:`Sequence` of :obj:`str` or :obj:`str`, optional:
            Can be a string such value is a name of a specific metafeature
            group (see ``valid_groups`` method for more information) or a
            sequence of metafeature group names. It can be also None, which
            in that case all available metafeature names will be returned.

        Returns
        -------
        :obj:`tuple` of :obj:`str`
            Tuple with all available metafeature names of the given ``groups``.

        Notes
        -----
        The returned ``metafeatures`` are not related to the groups or to the
        metafeatures fitted in the model in the model instantation. All the
        returned metafeatures are available in the ``Pymfe`` package. Check
        the ``TSMFE`` documentation for deeper information.
        """
        groups = TSMFE._check_groups_type(groups)
        groups = TSMFE._filter_groups(groups)

        deps = _internal.check_group_dependencies(groups)

        mtf_names = []  # type: t.List
        for group in groups.union(deps):
            class_ind = _internal.VALID_GROUPS.index(group)

            mtf_names += (
                _internal.get_prefixed_mtds_from_class(
                    class_obj=_internal.VALID_MFECLASSES[class_ind],
                    prefix=_internal.MTF_PREFIX,
                    only_name=True,
                    prefix_removal=True))

        return tuple(mtf_names)

    @classmethod
    def parse_by_group(
            cls,
            groups: t.Union[t.Sequence[str], str],
            extracted_results: t.Tuple[t.Sequence, ...],
    ) -> t.Tuple[t.List, ...]:
        """Parse the result of ``extract`` for given metafeature ``groups``.

        Can be used to easily separate the results of each metafeature
        group.

        Parameters
        ----------
        groups : :obj:`Sequence` of :obj:`str` or :obj:`str`
            Metafeature group names which the results should be parsed
            relative to. Use ``valid_groups`` method to check the available
            metafeature groups.

        extracted_results : :obj:`tuple` of :obj:`t.Sequence`
            Output of ``extract`` method. Should contain all outputed lists
            (metafeature names, values and elapsed time for extraction, if
            present.)

        Returns
        -------
        :obj:`tuple` of :obj:`str`
            Slices of lists of ``extracted_results``, selected based on
            given ``groups``.

        Notes
        -----
        The given ``groups`` are not related to the groups fitted in the
        model in the model instantation. Check ``valid_groups`` method to
        get a list of all available groups from the ``Pymfe`` package.
        Check the ``TSMFE`` documentation for deeper information about all
        these groups.
        """
        selected_indexes = _internal.select_results_by_classes(
            mtf_names=extracted_results[0],
            class_names=groups,
            include_dependencies=True)

        filtered_res = (
            [seq[ind] for ind in selected_indexes]
            for seq in extracted_results
        )

        return tuple(filtered_res)

    @staticmethod
    def _parse_description(docstring: str,
                           include_references: bool = False
                           ) -> t.Tuple[str, str]:
        """Parse the docstring to get initial description and reference.

        Parameters
        ----------
        docstring : str
            An numpy docstring as ``str``.

        include_references : bool
            If True include a column with article reference.

        Returns
        -------
        tuple of str
            The initial docstring description in the first position and the
            reference in the second.

        """
        initial_description = ""  # type: str
        reference_description = ""  # type: str

        # get initial description
        split = docstring.split("\n\n")
        if split:
            initial_description = " ".join(split[0].split())

        # get reference description
        if include_references:
            aux = docstring.split("References\n        ----------\n")
            if len(aux) >= 2:
                split = aux[1].split(".. [")
                if len(split) >= 2:
                    del split[0]
                    for spl in split:
                        reference_description += "[" + " ".join(
                            spl.split()) + "\n"

        return (initial_description, reference_description)

    @classmethod
    def metafeature_description(
            cls,
            groups: t.Optional[t.Union[str, t.Iterable[str]]] = None,
            sort_by_group: bool = False,
            sort_by_mtf: bool = False,
            print_table: bool = True,
            include_references: bool = False
    ) -> t.Optional[t.Tuple[t.List[t.List[str]], str]]:
        """Print a table with groups, metafeatures and description.

        Parameters
        ----------
        groups : sequence of str or str, optional:
            Can be a string such value is a name of a specific metafeature
            group (see ``valid_groups`` method for more information) or a
            sequence of metafeature group names. It can be also None, which
            in that case all available metafeature names will be returned.

        sort_by_group: bool
            Sort table by meta-feature group name.

        sort_by_mtf: bool
            Sort table by meta-feature name.

        print_table : bool
            If True a table will be printed with the description, otherwise the
            table will be send by return.

        print_table : bool
            If True sort the table by metafeature name.

        include_references : bool
            If True include a column with article reference.

        Returns
        -------
        list of list
            A table with the metafeature descriptions or None.

        Notes
        -----
        The returned ``metafeatures`` are not related to the groups or to the
        metafeatures fitted in the model instantation. All the
        returned metafeatures are available in the ``Pymfe`` package. Check
        the ``TSMFE`` documentation for deeper information.
        """

        groups = TSMFE._check_groups_type(groups)
        groups = TSMFE._filter_groups(groups)

        deps = _internal.check_group_dependencies(groups)

        if not isinstance(sort_by_group, bool):
            raise TypeError("The parameter sort_by_group should be bool.")

        if not isinstance(sort_by_mtf, bool):
            raise TypeError("The parameter sort_by_mtf should be bool.")

        if not isinstance(print_table, bool):
            raise TypeError("The parameter print_table should be bool.")

        mtf_names = []  # type: t.List[str]
        mtf_desc = [["Group", "Meta-feature name", "Description"]]
        if include_references:
            mtf_desc[0].append("Reference")

        for group in groups.union(deps):
            class_ind = _internal.VALID_GROUPS.index(group)

            mtf_names = (  # type: ignore
                _internal.get_prefixed_mtds_from_class(  # type: ignore
                    class_obj=_internal.VALID_MFECLASSES[class_ind],
                    prefix=_internal.MTF_PREFIX,
                    only_name=False,
                    prefix_removal=True))

            for name, method in mtf_names:
                ini_desc, ref_desc = TSMFE._parse_description(
                    str(method.__doc__), include_references)
                mtf_desc_line = [group, name, ini_desc]
                mtf_desc.append(mtf_desc_line)

                if include_references:
                    mtf_desc_line.append(ref_desc)

        if sort_by_mtf:
            mtf_desc.sort(key=lambda i: i[1])

        if sort_by_group:
            mtf_desc.sort(key=lambda i: i[0])

        draw = texttable.Texttable().add_rows(mtf_desc).draw()
        if print_table:
            print(draw)
            return None
        return mtf_desc, draw
