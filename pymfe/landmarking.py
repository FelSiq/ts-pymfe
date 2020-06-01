"""Module dedicated to landmarking time-series meta-features."""
import typing as t
import warnings

import numpy as np
import sklearn.model_selection
import sklearn.gaussian_process
import sklearn.base
import statsmodels.tsa.arima_model
import statsmodels.tsa.holtwinters
import statsmodels.tools.sm_exceptions
import statsmodels.base.model

import pymfe._utils as _utils
import pymfe._period as _period
import pymfe._models as _models
import pymfe._embed as _embed

try:
    import pymfe.general as general

except ImportError:
    pass

try:
    import pymfe.autocorr as autocorr

except ImportError:
    pass


class MFETSLandmarking:
    """Extract time-series meta-features from Landmarking group."""
    @classmethod
    def _standard_pipeline_sklearn(
            cls,
            y: np.ndarray,
            model: t.Union[_models.BaseModel, sklearn.base.BaseEstimator],
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            X: t.Optional[np.ndarray] = None,
            args_fit: t.Optional[t.Dict[str, t.Any]] = None,
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
            scale_range: t.Optional[t.Tuple[float, float]] = (0.0, 1.0),
    ) -> np.ndarray:
        """Fit a model using a canonical pipeline with models from sklearn.

        In this pipeline, each instance of the time-series `y[i]` will be
        automatically associated with a value `X[i]` linearly spaced in the
        [0, 1] range if no `X` is given.

        Let `min` and `max` be the values in the first and second element of
        ``scale_range`` argument, respectively.  Therefore, `X[i]` is given
        by:
        $$
            X[i] = min + (max - min) * i / (len(y) - 1)`
        $$

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at
        each iteration of the validation, a distinct single fold is used as
        the test set, and all other previous folds are used as the train set.

        Parameters
        ----------
        y : :obj:`np.ndarray`
            One-dimensional time-series values.

        model : :obj:`_models.BaseModel` or :obj:`sklearn.base.BaseEstimator`
            A sklearn model, or a custom model from the `_models.py` module.
            Must have `.fit(X, y)` and `.predict(X)` methods implemented.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        X : :obj:`np.ndarray`, optional
            Time-stamps for each time-series observation.

        args_fit : dict, optional
            kwargs for fittig the model (`.fit(X, y, **args_fit)`).

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``y``) are used.

        scale_range : tuple of float, optional (default=(0.0, 1.0))
            Range of ``X`` and ``y`` after normalization. If None, then use
            the original scale of ``y``, and `X = [0, 1, 2, ..., len(y) - 1]`.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.
        """
        if tskf is None:
            tskf = sklearn.model_selection.TimeSeriesSplit(
                n_splits=num_cv_folds)

        if args_fit is None:
            args_fit = {}

        # Note: 'X' are the unitless timesteps of the timeseries
        if X is None:
            y = _utils.sample_data(ts=y, lm_sample_frac=lm_sample_frac)
            X_range = scale_range if scale_range is not None else (0, y.size)
            X = np.linspace(*X_range, y.size).reshape(-1, 1)

        else:
            y, X = _utils.sample_data(ts=y, X=X, lm_sample_frac=lm_sample_frac)

        y = y.reshape(-1, 1)
        res = np.zeros(tskf.n_splits, dtype=float)

        if scale_range is not None:
            scaler = sklearn.preprocessing.MinMaxScaler(
                feature_range=scale_range)

        for ind_fold, (inds_train, inds_test) in enumerate(tskf.split(X)):
            X_train, X_test = X[inds_train, :], X[inds_test, :]
            y_train, y_test = y[inds_train], y[inds_test]

            if scale_range is not None:
                y_train = scaler.fit_transform(y_train).ravel()
                y_test = scaler.transform(y_test).ravel()

            try:
                model.fit(X_train, y_train, **args_fit)
                y_pred = model.predict(X_test).ravel()
                res[ind_fold] = score(y_pred, y_test)

            except (TypeError, ValueError):
                res[ind_fold] = np.nan

        return res

    @classmethod
    def _standard_pipeline_statsmodels(
            cls,
            ts: np.ndarray,
            model_callable: statsmodels.base.model.Model,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            args_inst: t.Optional[t.Dict[str, t.Any]] = None,
            args_fit: t.Optional[t.Dict[str, t.Any]] = None,
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
            scale_range: t.Optional[t.Tuple[float, float]] = (0.0, 1.0),
    ) -> np.ndarray:
        """Fit a model using a canonical pipeline with models from statsmodels.

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at
        each iteration of the validation, a distinct single fold is used as
        the test set, and all other previous folds are used as the train set.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        model_callable : :obj:`statsmodels.base.model.Model`
            Callable model from statsmodels package.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        args_inst : dict, optional
            Extra arguments for the model instantiation.

        args_fit : dict, optional
            Extra arguments for the fit method.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        scale_range : tuple of float, optional (default=(0.0, 1.0))
            Range of ``ts`` after normalization. If None, then use the original
            ``ts``.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.
        """
        if args_inst is None:
            args_inst = {}

        if args_fit is None:
            args_fit = {}

        ts = _utils.sample_data(ts=ts, lm_sample_frac=lm_sample_frac)

        if tskf is None:
            tskf = sklearn.model_selection.TimeSeriesSplit(
                n_splits=num_cv_folds)

        ts = ts.reshape(-1, 1)
        res = np.zeros(tskf.n_splits, dtype=float)

        if scale_range is not None:
            scaler = sklearn.preprocessing.MinMaxScaler(
                feature_range=scale_range)

        with warnings.catch_warnings():
            # Note: We are ignoring these warnings because they are related to
            # poor model fits when the time-series does not match the algorithm
            # assumptions. This is expected for a couple of model types, since
            # the same unprocessed time-series is fitted in a large amount of
            # different model types. It is also expected that bad score relates
            # with poor fits, and hence the metafeature will still reflect the
            # true (bad) relationship between the model and the data.
            warnings.filterwarnings(
                "ignore",
                module="statsmodels",
                category=statsmodels.tools.sm_exceptions.ConvergenceWarning)

            warnings.filterwarnings("ignore",
                                    module="statsmodels",
                                    category=RuntimeWarning)

            warnings.filterwarnings("ignore",
                                    module="statsmodels",
                                    category=statsmodels.tools.sm_exceptions.
                                    HessianInversionWarning)

            for ind_fold, (inds_train, inds_test) in enumerate(tskf.split(ts)):
                ts_train, ts_test = ts[inds_train], ts[inds_test]

                if scale_range is not None:
                    ts_train = scaler.fit_transform(ts_train).ravel()
                    ts_test = scaler.transform(ts_test).ravel()

                try:
                    model = model_callable(ts_train,
                                           **args_inst).fit(**args_fit)

                    ts_pred = model.predict(start=ts_train.size,
                                            end=ts_train.size + ts_test.size -
                                            1).ravel()

                    res[ind_fold] = score(ts_pred, ts_test)

                except (TypeError, ValueError):
                    res[ind_fold] = np.nan

        return res

    @classmethod
    def _model_acf_first_nonpos(
            cls,
            ts: np.ndarray,
            perf_ft_method: t.Callable[..., np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
            unbiased: bool = True,
            max_nlags: t.Optional[int] = None,
            **kwargs) -> np.ndarray:
        """First non-positive autocorrelation lag of cross-validation erros.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        perf_ft_method : callable
            Internal cross-validation method.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        unbiased : bool, optional (default=True)
            If True, the autocorrelation function is corrected for statistical
            bias.

        max_nlags : int, optional
            Number of lags to avaluate the autocorrelation function.

        kwargs:
            Additional parameters for ``perf_ft_method``.

        Returns
        -------
        :obj:`np.ndarray`
            Lag of the first non-positive in the autocorrelation function for
            each forward chaining cross-validation fold.
        """
        def score(ts_pred: np.ndarray, ts_test: np.ndarray) -> float:
            """Score function: autocorrelation of the errors."""
            return autocorr.MFETSAutocorr.ft_acf_first_nonpos(
                ts=ts_pred - ts_test, unbiased=unbiased, max_nlags=max_nlags)

        model_acf_first_nonpos = perf_ft_method(ts=ts,
                                                tskf=tskf,
                                                score=score,
                                                num_cv_folds=num_cv_folds,
                                                lm_sample_frac=lm_sample_frac,
                                                **kwargs)

        return model_acf_first_nonpos

    @classmethod
    def ft_model_mean(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        """Cross-validated performance of the global mean forecasting model.

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at
        each iteration of the validation, a distinct single fold is used as
        the test set, and all other previous folds are used as the train set.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.

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
        .. [3] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on 26 May 2020.
        """
        model_callable = statsmodels.tsa.arima_model.ARIMA

        args_inst = {"order": (0, 0, 0)}
        args_fit = {
            "trend": "c",
            "disp": False,
            "transparams": True,
            "maxiter": 1,
        }

        res = cls._standard_pipeline_statsmodels(ts=ts,
                                                 model_callable=model_callable,
                                                 args_inst=args_inst,
                                                 args_fit=args_fit,
                                                 score=score,
                                                 tskf=tskf,
                                                 num_cv_folds=num_cv_folds,
                                                 lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_loc_mean(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            loc_prop: float = 0.25,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        """Cross-validated performance of the local mean forecasting model.

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at
        each iteration of the validation, a distinct single fold is used as
        the test set, and all other previous folds are used as the train set.

        The local mean model uses the mean of the most recent observation
        as the forecasting value for all the next observations.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        loc_prop : float, optional (default=0.25)
            Fraction of the most recent observations to be used as the training
            data. Must be in (0, 1] range.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.

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
        .. [3] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on 26 May 2020.
        """
        model = _models.TSLocalMean(train_prop=loc_prop)

        res = cls._standard_pipeline_sklearn(y=ts,
                                             model=model,
                                             score=score,
                                             tskf=tskf,
                                             num_cv_folds=num_cv_folds,
                                             lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_loc_median(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            loc_prop: float = 0.25,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        """Cross-validated performance of the local median forecasting model.

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at
        each iteration of the validation, a distinct single fold is used as
        the test set, and all other previous folds are used as the train set.

        The local median model uses the median of the most recent observation
        as the forecasting value for all the next observations.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        loc_prop : float, optional (default=0.25)
            Fraction of the most recent observations to be used as the training
            data. Must be in (0, 1] range.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.

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
        .. [3] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on 26 May 2020.
        """
        model = _models.TSLocalMedian(train_prop=loc_prop)

        res = cls._standard_pipeline_sklearn(y=ts,
                                             model=model,
                                             score=score,
                                             tskf=tskf,
                                             num_cv_folds=num_cv_folds,
                                             lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_sine(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            opt_initial_guess: bool = True,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
            random_state: t.Optional[int] = None,
    ) -> np.ndarray:
        """Cross-validated performance of the sine forecasting model.

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at each
        iteration of the validation, a distinct single fold is used as the test
        set, and all other previous folds are used as the train set.

        The sine model tries to optimize parameters such that a function `f` in
        the form `f(x) = a * sin(w * x + b) + c` describes well the time-series
        values.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        opt_initial_guess : bool, optional (default=True)
            If True, make and optimal initial guess. If False, make a faster
            but naive initial guess.

        loc_prop : float, optional (default=0.25)
            Fraction of the most recent observations to be used as the training
            data. Must be in (0, 1] range.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        random_state : int, optional
            Random seed to perform a random naive initial guess. Used only
            if ``opt_initial_guess`` is False.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.

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
        .. [3] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on 26 May 2020.
        """
        model = _models.TSSine(opt_initial_guess=opt_initial_guess,
                               random_state=random_state)

        res = cls._standard_pipeline_sklearn(y=ts,
                                             model=model,
                                             score=score,
                                             tskf=tskf,
                                             num_cv_folds=num_cv_folds,
                                             lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_exp(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        """Cross-validated performance of the exponential forecasting model.

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at each
        iteration of the validation, a distinct single fold is used as the test
        set, and all other previous folds are used as the train set.

        The exponential model tries to optimize parameters such that a function
        `f` in the form `f(x) = a * exp(b * x) + c` describes well the
        time-series values.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        loc_prop : float, optional (default=0.25)
            Fraction of the most recent observations to be used as the training
            data. Must be in (0, 1] range.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.

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
        .. [3] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on 26 May 2020.
        """
        model = _models.TSExp()

        res = cls._standard_pipeline_sklearn(y=ts,
                                             model=model,
                                             score=score,
                                             tskf=tskf,
                                             num_cv_folds=num_cv_folds,
                                             lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_gaussian(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
            random_state: t.Optional[int] = None,
    ) -> np.ndarray:
        """Cross-validated performance of the gaussian process model.

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at each
        iteration of the validation, a distinct single fold is used as the test
        set, and all other previous folds are used as the train set.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        loc_prop : float, optional (default=0.25)
            Fraction of the most recent observations to be used as the training
            data. Must be in (0, 1] range.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        random_state : int, optional
            Random seed, to keep the optimization process deterministic.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.

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
        .. [3] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on 26 May 2020.
        """
        model = sklearn.gaussian_process.GaussianProcessRegressor(
            copy_X_train=False, random_state=random_state)

        res = cls._standard_pipeline_sklearn(y=ts,
                                             model=model,
                                             score=score,
                                             tskf=tskf,
                                             num_cv_folds=num_cv_folds,
                                             lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_linear(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        """Cross-validated performance of the linear model in time domain.

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at each
        iteration of the validation, a distinct single fold is used as the test
        set, and all other previous folds are used as the train set.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.
        """
        model = sklearn.linear_model.LinearRegression()

        res = cls._standard_pipeline_sklearn(y=ts,
                                             model=model,
                                             score=score,
                                             tskf=tskf,
                                             num_cv_folds=num_cv_folds,
                                             lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_linear_embed(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            emb_dim: t.Optional[int] = None,
            lag: t.Optional[t.Union[str, int]] = None,
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
            max_nlags: t.Optional[int] = None,
            detrended_acfs: t.Optional[np.ndarray] = None,
            detrended_ami: t.Optional[np.ndarray] = None,
            ts_scaled: t.Optional[np.ndarray] = None,
            emb_dim_cao_e1: t.Optional[np.ndarray] = None,
            emb_dim_cao_e2: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit a linear model on the embedded time-series.

        The appropriate lag is estimated using either the detrended time-series
        automutual information function or the autocorrelation function.

        The appropriate dimension is estimated using the Cao's algorithm.

        The time-series is detrended using the Friedman's Super Smoorther
        algorithm.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        emb_bim : int, optional
            Embedding dimension. If not given,

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        loc_prop : float, optional (default=0.25)
            Fraction of the most recent observations to be used as the training
            data. Must be in (0, 1] range.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        lag : int or str, optional (default = None)
            If scalar, return its own value casted to integer,

            If string, it must be one value in {`ami`, `acf`, `acf-nonsig`},
            which defines the strategy of defining the appropriate lag of
            the embedding.
                1. `ami`: uses the first minimum lag of the automutual
                    information of the time-series.
                2. `acf`: uses the first negative lag of the autocorrelation of
                    the time-series.
                3. `acf-nonsig` (default): uses the first non-significant lag
                    of the time-series autocorrelation function.
                    The non-significant value is defined as the first lag that
                    has the absolute value of is autocorrelation below the
                    critical value defined as 1.96 / sqrt(ts.size).

            If None, the lag will be searched will the 'acf-nonsig' criteria.

        max_nlags : int, optional
            If ``lag`` is not a numeric value, then it will be estimated using
            either the time-series autocorrelation or mutual information
            function estimated up to this argument value.

        detrended_acfs : :obj:`np.ndarray`, optional
            Array of time-series autocorrelation function (for distinct ordered
            lags) of the detrended time-series. Used only if ``emb_dim`` is
            None and ``lag`` is any of `acf`, `acf-nonsig` or None.  If this
            argument is not given and the previous condiditon is meet, the
            autocorrelation function will be calculated inside this method up
            to ``max_nlags``.

        detrended_ami : :obj:`np.ndarray`, optional
            Array of time-series automutual information function (for distinct
            ordered lags). Used only if ``emb_dim`` is None and ``lag`` is
            `ami`. If not given and the previous condiditon is meet, the
            automutual information function will be calculated inside this
            method up to ``max_nlags``.

        emb_dim_cao_e1 : :obj:`np.ndarray`, optional
            E1 values from the Cao's method. Used to take advantage of
            precomputations. Used only if ``emb_dim`` is None.

        emb_dim_cao_e2 : :obj:`np.ndarray`, optional
            E2 values from the Cao's method. Used to take advantage of
            precomputations. Used only if ``emb_dim`` is None.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.

        References
        ----------
        .. [1] Liangyue Cao, Practical method for determining the minimum
            embedding dimension of a scalar time series, Physica D: Nonlinear
            Phenomena, Volume 110, Issues 1–2, 1997, Pages 43-50,
            ISSN 0167-2789, https://doi.org/10.1016/S0167-2789(97)00118-8.
        .. [2] Friedman, J. H. 1984, A variable span scatterplot smoother
            Laboratory for Computational Statistics, Stanford University
            Technical Report No. 5. Available at:
            https://www.slac.stanford.edu/pubs/slacpubs/3250/slac-pub-3477.pdf
            Accessed on May 12 2020.
        .. [3] Fraser AM, Swinney HL. Independent coordinates for strange
            attractors from mutual information. Phys Rev A Gen Phys.
            1986;33(2):1134‐1140. doi:10.1103/physreva.33.1134
        .. [4] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on 26 May 2020.
        """
        model = sklearn.linear_model.LinearRegression()

        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        lag = _embed.embed_lag(ts=ts_scaled,
                               lag=lag,
                               max_nlags=max_nlags,
                               detrended_acfs=detrended_acfs,
                               detrended_ami=detrended_ami)

        if emb_dim is None:
            emb_dim = general.MFETSGeneral.ft_emb_dim_cao(
                ts=ts,
                lag=lag,
                emb_dim_cao_e1=emb_dim_cao_e1,
                emb_dim_cao_e2=emb_dim_cao_e2)

        X = _embed.embed_ts(ts=ts, dim=emb_dim, lag=lag)

        res = cls._standard_pipeline_sklearn(y=X[:, 0],
                                             X=X[:, 1:],
                                             model=model,
                                             score=score,
                                             tskf=tskf,
                                             num_cv_folds=num_cv_folds,
                                             lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_linear_seasonal(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            ts_period: t.Optional[int] = None,
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        """Cross-validated performance of the seasonal linear model.

        This model is evaluated using a linear regression of the dummy
        variables marking the corresponding value from the previous season
        onto the time-series values.

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at each
        iteration of the validation, a distinct single fold is used as the test
        set, and all other previous folds are used as the train set.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        ts_period : int, optional
            Time-series period. If not given, it will be estimated using
            the minima of the absolute autocorrelation function from lag
            1 up to half the time-series size.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.

        References
        ----------
        .. [1] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on 26 May 2020.
        """
        _ts_period = _period.get_ts_period(ts=ts, ts_period=ts_period)

        if _ts_period <= 1:
            raise ValueError("Time-series is not seasonal (period <= 1).")

        X = np.tile(np.arange(_ts_period),
                    1 + ts.size // _ts_period)[:ts.size, np.newaxis]

        # Note: remove one dummy variable to avoid the 'dummy
        # variable trap'.
        X = sklearn.preprocessing.OneHotEncoder(
            sparse=False, drop="first").fit_transform(X)

        model = sklearn.linear_model.LinearRegression()

        res = cls._standard_pipeline_sklearn(y=ts,
                                             X=X,
                                             model=model,
                                             score=score,
                                             tskf=tskf,
                                             num_cv_folds=num_cv_folds,
                                             lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_naive(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        """Cross-validated performance of the naive model.

        The naive model uses only the last observation of the time-series as
        the forecast value for all future timesteps.

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at each
        iteration of the validation, a distinct single fold is used as the test
        set, and all other previous folds are used as the train set.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.

        References
        ----------
        .. [1] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on 26 May 2020.
        """
        model = _models.TSNaive()

        res = cls._standard_pipeline_sklearn(y=ts,
                                             model=model,
                                             score=score,
                                             tskf=tskf,
                                             num_cv_folds=num_cv_folds,
                                             lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_naive_drift(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        """Cross-validated performance of the naive model with drift.

        The naive model uses the last observation of the time-series as the
        forecast value for all future timesteps, with a drift added based on
        the timestamp. The drift is the angular coefficient from the line that
        crosses the first and the last observation of the time-series (which
        is equivalent to the mean first-order difference of the time-series).

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at each
        iteration of the validation, a distinct single fold is used as the test
        set, and all other previous folds are used as the train set.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.

        References
        ----------
        .. [1] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on 26 May 2020.
        """
        model = _models.TSNaiveDrift()

        res = cls._standard_pipeline_sklearn(y=ts,
                                             model=model,
                                             score=score,
                                             tskf=tskf,
                                             num_cv_folds=num_cv_folds,
                                             lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_naive_seasonal(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            ts_period: t.Optional[int] = None,
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        """Cross-validated performance of the seasonal linear model.

        This model is evaluated using a linear regression of the dummy
        variables marking the corresponding value from the previous season
        onto the time-series values.

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at each
        iteration of the validation, a distinct single fold is used as the test
        set, and all other previous folds are used as the train set.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        ts_period : int, optional
            Time-series period. If not given, it will be estimated using
            the minima of the absolute autocorrelation function from lag
            1 up to half the time-series size.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.

        References
        ----------
        .. [1] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on 26 May 2020.
        """
        _ts_period = _period.get_ts_period(ts=ts, ts_period=ts_period)

        if _ts_period <= 1:
            raise ValueError("Time-series is not seasonal (period <= 1).")

        model = _models.TSNaiveSeasonal(ts_period=_ts_period)

        res = cls._standard_pipeline_sklearn(y=ts,
                                             model=model,
                                             score=score,
                                             tskf=tskf,
                                             num_cv_folds=num_cv_folds,
                                             lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_arima_100_c(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
            solver: str = "lbfgs",
            maxiter: int = 512,
    ) -> np.ndarray:
        """Cross-validated performance of the ARIMA(1,0,0) with constant model.

        ARIMA(1,0,0) is a first-order regressive model.

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at each
        iteration of the validation, a distinct single fold is used as the test
        set, and all other previous folds are used as the train set.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        solver : str, optional (default="lbfgs")
            Solver used to optimize the model.

        maxiter : int, optional (default=512)
            Maximum number of optimization iterations.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.

        References
        ----------
        .. [1] Statistical forecasting: notes on regression and time series
            analysis, Robert Nau, Fuqua School of Business, Duke University.
            URL: https://people.duke.edu/~rnau/411arim.htm#mixed
            Accessed on 26 May 2020.
        .. [2] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on 26 May 2020.
        """
        model_callable = statsmodels.tsa.arima_model.ARIMA

        args_inst = {"order": (1, 0, 0)}
        args_fit = {
            "trend": "c",
            "disp": False,
            "transparams": True,
            "maxiter": maxiter,
            "solver": solver,
        }

        res = cls._standard_pipeline_statsmodels(ts=ts,
                                                 model_callable=model_callable,
                                                 args_inst=args_inst,
                                                 args_fit=args_fit,
                                                 score=score,
                                                 tskf=tskf,
                                                 num_cv_folds=num_cv_folds,
                                                 lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_arima_010_c(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
            solver: str = "lbfgs",
            maxiter: int = 512,
    ) -> np.ndarray:
        """Cross-validated performance of the ARIMA(0,1,0) with constant model.

        ARIMA(0,1,0) is a random walk model.

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at each
        iteration of the validation, a distinct single fold is used as the test
        set, and all other previous folds are used as the train set.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        solver : str, optional (default="lbfgs")
            Solver used to optimize the model.

        maxiter : int, optional (default=512)
            Maximum number of optimization iterations.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.

        References
        ----------
        .. [1] Statistical forecasting: notes on regression and time series
            analysis, Robert Nau, Fuqua School of Business, Duke University.
            URL: https://people.duke.edu/~rnau/411arim.htm#mixed
            Accessed on 26 May 2020.
        .. [2] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on 26 May 2020.
        """
        model_callable = statsmodels.tsa.arima_model.ARIMA

        args_inst = {"order": (0, 1, 0)}
        args_fit = {
            "disp": False,
            "trend": "c",
            "transparams": True,
            "maxiter": maxiter,
            "solver": solver,
        }

        res = cls._standard_pipeline_statsmodels(ts=ts,
                                                 model_callable=model_callable,
                                                 args_inst=args_inst,
                                                 args_fit=args_fit,
                                                 score=score,
                                                 tskf=tskf,
                                                 num_cv_folds=num_cv_folds,
                                                 lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_arima_110_c(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
            solver: str = "lbfgs",
            maxiter: int = 512,
    ) -> np.ndarray:
        """Cross-validated performance of the ARIMA(1,1,0) with constant model.

        ARIMA(1,1,0) is a differenced first-order autoregressive model.

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at each
        iteration of the validation, a distinct single fold is used as the test
        set, and all other previous folds are used as the train set.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        solver : str, optional (default="lbfgs")
            Solver used to optimize the model.

        maxiter : int, optional (default=512)
            Maximum number of optimization iterations.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.

        References
        ----------
        .. [1] Statistical forecasting: notes on regression and time series
            analysis, Robert Nau, Fuqua School of Business, Duke University.
            URL: https://people.duke.edu/~rnau/411arim.htm#mixed
            Accessed on 26 May 2020.
        .. [2] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on 26 May 2020.
        """
        model_callable = statsmodels.tsa.arima_model.ARIMA

        args_inst = {"order": (1, 1, 0)}
        args_fit = {
            "disp": False,
            "trend": "c",
            "transparams": True,
            "maxiter": maxiter,
            "solver": solver,
        }

        res = cls._standard_pipeline_statsmodels(ts=ts,
                                                 model_callable=model_callable,
                                                 args_inst=args_inst,
                                                 args_fit=args_fit,
                                                 score=score,
                                                 tskf=tskf,
                                                 num_cv_folds=num_cv_folds,
                                                 lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_arima_011_nc(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
            solver: str = "lbfgs",
            maxiter: int = 512,
    ) -> np.ndarray:
        """Cross-validated performance of the ARIMA(0,1,1) (no constant) model.

        ARIMA(0,1,1) without constant is a simple exponential smoothing model.

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at each
        iteration of the validation, a distinct single fold is used as the test
        set, and all other previous folds are used as the train set.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        solver : str, optional (default="lbfgs")
            Solver used to optimize the model.

        maxiter : int, optional (default=512)
            Maximum number of optimization iterations.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.

        References
        ----------
        .. [1] Statistical forecasting: notes on regression and time series
            analysis, Robert Nau, Fuqua School of Business, Duke University.
            URL: https://people.duke.edu/~rnau/411arim.htm#mixed
            Accessed on 26 May 2020.
        .. [2] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on 26 May 2020.
        """
        model_callable = statsmodels.tsa.arima_model.ARIMA

        args_inst = {"order": (0, 1, 1)}
        args_fit = {
            "disp": False,
            "trend": "nc",
            "transparams": True,
            "maxiter": maxiter,
            "solver": solver,
        }

        res = cls._standard_pipeline_statsmodels(ts=ts,
                                                 model_callable=model_callable,
                                                 args_inst=args_inst,
                                                 args_fit=args_fit,
                                                 score=score,
                                                 tskf=tskf,
                                                 num_cv_folds=num_cv_folds,
                                                 lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_arima_011_c(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
            solver: str = "lbfgs",
            maxiter: int = 512,
    ) -> np.ndarray:
        """Cross-validated performance of the ARIMA(0,1,1) with constant model.

        ARIMA(0,1,1) with constant is a simple exponential smoothing model with
        growth.

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at each
        iteration of the validation, a distinct single fold is used as the test
        set, and all other previous folds are used as the train set.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        solver : str, optional (default="lbfgs")
            Solver used to optimize the model.

        maxiter : int, optional (default=512)
            Maximum number of optimization iterations.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.

        References
        ----------
        .. [1] Statistical forecasting: notes on regression and time series
            analysis, Robert Nau, Fuqua School of Business, Duke University.
            URL: https://people.duke.edu/~rnau/411arim.htm#mixed
            Accessed on 26 May 2020.
        .. [2] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on 26 May 2020.
        """
        model_callable = statsmodels.tsa.arima_model.ARIMA

        args_inst = {"order": (0, 1, 1)}
        args_fit = {
            "disp": False,
            "trend": "c",
            "transparams": True,
            "maxiter": maxiter,
            "solver": solver,
        }

        res = cls._standard_pipeline_statsmodels(ts=ts,
                                                 model_callable=model_callable,
                                                 args_inst=args_inst,
                                                 args_fit=args_fit,
                                                 score=score,
                                                 tskf=tskf,
                                                 num_cv_folds=num_cv_folds,
                                                 lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_arima_021_c(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
            solver: str = "lbfgs",
            maxiter: int = 512,
    ) -> np.ndarray:
        """Cross-validated performance of the ARIMA(0,2,1) with constant model.

        ARIMA(0,2,1) with constant is a linear exponential smoothing model.

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at each
        iteration of the validation, a distinct single fold is used as the test
        set, and all other previous folds are used as the train set.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        solver : str, optional (default="lbfgs")
            Solver used to optimize the model.

        maxiter : int, optional (default=512)
            Maximum number of optimization iterations.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.

        References
        ----------
        .. [1] Statistical forecasting: notes on regression and time series
            analysis, Robert Nau, Fuqua School of Business, Duke University.
            URL: https://people.duke.edu/~rnau/411arim.htm#mixed
            Accessed on 26 May 2020.
        .. [2] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on 26 May 2020.
        """
        model_callable = statsmodels.tsa.arima_model.ARIMA

        args_inst = {"order": (0, 2, 1)}
        args_fit = {
            "disp": False,
            "trend": "c",
            "transparams": True,
            "maxiter": maxiter,
            "solver": solver,
        }

        res = cls._standard_pipeline_statsmodels(ts=ts,
                                                 model_callable=model_callable,
                                                 args_inst=args_inst,
                                                 args_fit=args_fit,
                                                 score=score,
                                                 tskf=tskf,
                                                 num_cv_folds=num_cv_folds,
                                                 lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_arima_112_nc(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
            solver: str = "lbfgs",
            maxiter: int = 512,
    ) -> np.ndarray:
        """Cross-validated performance of the ARIMA(1,1,2) (no constant) model.

        ARIMA(1,1,2) without constant is a damped-trend linear exponential
        smoothing model.

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at each
        iteration of the validation, a distinct single fold is used as the test
        set, and all other previous folds are used as the train set.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        solver : str, optional (default="lbfgs")
            Solver used to optimize the model.

        maxiter : int, optional (default=512)
            Maximum number of optimization iterations.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.

        References
        ----------
        .. [1] Statistical forecasting: notes on regression and time series
            analysis, Robert Nau, Fuqua School of Business, Duke University.
            URL: https://people.duke.edu/~rnau/411arim.htm#mixed
            Accessed on 26 May 2020.
        .. [2] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on 26 May 2020.
        """
        model_callable = statsmodels.tsa.arima_model.ARIMA

        args_inst = {"order": (1, 1, 2)}
        args_fit = {
            "disp": False,
            "trend": "nc",
            "transparams": True,
            "maxiter": maxiter,
            "solver": solver,
        }

        res = cls._standard_pipeline_statsmodels(ts=ts,
                                                 model_callable=model_callable,
                                                 args_inst=args_inst,
                                                 args_fit=args_fit,
                                                 score=score,
                                                 tskf=tskf,
                                                 num_cv_folds=num_cv_folds,
                                                 lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_ses(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        """Cross-validated performance of a Single Exponential Smoothing model.

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at each
        iteration of the validation, a distinct single fold is used as the test
        set, and all other previous folds are used as the train set.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        solver : str, optional (default="lbfgs")
            Solver used to optimize the model.

        maxiter : int, optional (default=512)
            Maximum number of optimization iterations.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.

        References
        ----------
        .. [1] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on 26 May 2020.
        """
        model_callable = statsmodels.tsa.holtwinters.SimpleExpSmoothing

        res = cls._standard_pipeline_statsmodels(ts=ts,
                                                 model_callable=model_callable,
                                                 score=score,
                                                 tskf=tskf,
                                                 num_cv_folds=num_cv_folds,
                                                 lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_hwes_ada(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            ts_period: t.Optional[int] = None,
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        """Performance of a HW(A_{d},A) model.

        Either components (Trend and Seasonal) are additive, and the model has
        a damping component in the trend component.

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at each
        iteration of the validation, a distinct single fold is used as the test
        set, and all other previous folds are used as the train set.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        ts_period : int, optional
            Time-series period. If not given, it will be estimated using
            the minima of the absolute autocorrelation function from lag
            1 up to half the time-series size.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.

        References
        ----------
        .. [1] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on 26 May 2020.
        .. [2] Holt, C. E. (1957). Forecasting seasonals and trends by
            exponentially weighted averages (O.N.R. Memorandum No. 52).
            Carnegie Institute of Technology, Pittsburgh USA.
            https://doi.org/10.1016/j.ijforecast.2003.09.015
        .. [3] Winters, P. R. (1960). Forecasting sales by exponentially
            weighted moving averages. Management Science, 6, 324–342.
            https://doi.org/10.1287/mnsc.6.3.324
        """
        _ts_period = _period.get_ts_period(ts=ts, ts_period=ts_period)

        if _ts_period <= 1:
            raise ValueError("Time-series is not seasonal (period <= 1).")

        model_callable = statsmodels.tsa.holtwinters.ExponentialSmoothing

        args_inst = {
            "seasonal_periods": _ts_period,
            "trend": "add",
            "seasonal": "add",
            "damped": True,
        }

        args_fit = {
            "use_brute": False,
        }

        res = cls._standard_pipeline_statsmodels(ts=ts,
                                                 model_callable=model_callable,
                                                 score=score,
                                                 args_inst=args_inst,
                                                 args_fit=args_fit,
                                                 tskf=tskf,
                                                 num_cv_folds=num_cv_folds,
                                                 lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_hwes_adm(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            ts_period: t.Optional[int] = None,
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        """Performance of a HW(A_{d},M) model.

        The trend component is additive, and the seasonal component is
        multiplicative. Also, the trend component is damped.

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at each
        iteration of the validation, a distinct single fold is used as the test
        set, and all other previous folds are used as the train set.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        score : callable
            Score function. Must receive two numeric values as the first two
            arguments, and return a single numeric value.

        ts_period : int, optional
            Time-series period. If not given, it will be estimated using
            the minima of the absolute autocorrelation function from lag
            1 up to half the time-series size.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        Returns
        -------
        :obj:`np.ndarray`
            The model performance for each iteration of the cross-validation.

        References
        ----------
        .. [1] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on 26 May 2020.
        .. [2] Holt, C. E. (1957). Forecasting seasonals and trends by
            exponentially weighted averages (O.N.R. Memorandum No. 52).
            Carnegie Institute of Technology, Pittsburgh USA.
            https://doi.org/10.1016/j.ijforecast.2003.09.015
        .. [3] Winters, P. R. (1960). Forecasting sales by exponentially
            weighted moving averages. Management Science, 6, 324–342.
            https://doi.org/10.1287/mnsc.6.3.324
        """
        _ts_period = _period.get_ts_period(ts=ts, ts_period=ts_period)

        if _ts_period <= 1:
            raise ValueError("Time-series is not seasonal (period <= 1).")

        model_callable = statsmodels.tsa.holtwinters.ExponentialSmoothing

        args_inst = {
            "seasonal_periods": _ts_period,
            "trend": "add",
            "seasonal": "mul",
            "damped": True,
        }

        args_fit = {
            "use_brute": False,
        }

        # Note: scaling time-series in [1, 2] interval rather than
        # [0, 1] because for Multiplicative Exponential Models,
        # the time-series values must be strictly positive.
        res = cls._standard_pipeline_statsmodels(ts=ts,
                                                 model_callable=model_callable,
                                                 score=score,
                                                 args_inst=args_inst,
                                                 args_fit=args_fit,
                                                 tskf=tskf,
                                                 num_cv_folds=num_cv_folds,
                                                 lm_sample_frac=lm_sample_frac,
                                                 scale_range=(1, 2))

        return res

    @classmethod
    def ft_model_mean_acf_first_nonpos(
            cls,
            ts: np.ndarray,
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
            unbiased: bool = True,
            max_nlags: t.Optional[int] = None,
    ) -> np.ndarray:
        """First non-positive autocorrelation lags for Mean model errors.

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at each
        iteration of the validation, a distinct single fold is used as the test
        set, and all other previous folds are used as the train set. Then, the
        autocorrelation function is calculated from the test set errors (i.e.
        true values subtracted by the predicted values).

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        unbiased : bool, optional (default=True)
            If True, the autocorrelation function is corrected for statistical
            bias.

        max_nlags : int, optional
            Number of lags to avaluate the autocorrelation function.

        Returns
        -------
        :obj:`np.ndarray`
            First non-positive autocorrelation function lag of the mean model
            errors, for each cross-validation fold.

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
        .. [3] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on 26 May 2020.
        """
        acf_first_nonpos_mean = cls._model_acf_first_nonpos(
            ts=ts,
            perf_ft_method=cls.ft_model_mean,
            tskf=tskf,
            num_cv_folds=num_cv_folds,
            lm_sample_frac=lm_sample_frac,
            unbiased=unbiased,
            max_nlags=max_nlags)

        return acf_first_nonpos_mean

    @classmethod
    def ft_model_linear_acf_first_nonpos(
            cls,
            ts: np.ndarray,
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
            unbiased: bool = True,
            max_nlags: t.Optional[int] = None,
    ) -> np.ndarray:
        """First non-positive autocorrelation lags for Linear model errors.

        The linear model is in the time domain, i.e., the time-series value
        is regressed onto the timestamps.

        The model score is validated using Forward Chaining, i.e., the full
        time-series is split into ``num_cv_folds`` of equal sizes, and at each
        iteration of the validation, a distinct single fold is used as the test
        set, and all other previous folds are used as the train set. Then, the
        autocorrelation function is calculated from the test set errors (i.e.
        true values subtracted by the predicted values).

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        tskf : :obj:`sklearn.model_selection.TimeSeriesSplit`, optional
            Custom Forward Chaining cross-validatior.

        num_cv_folds : int, optional (default=5)
            Number of test folds. Used only if ``tskf`` is None.

        lm_sample_frac : float, optional (default=1.0)
            Fraction of dataset to use. Default is to use the entire dataset.
            Must be a value in (0, 1] range. Only the most recent time-series
            observations (in the highest indices of ``ts``) are used.

        unbiased : bool, optional (default=True)
            If True, the autocorrelation function is corrected for statistical
            bias.

        max_nlags : int, optional
            Number of lags to avaluate the autocorrelation function.

        Returns
        -------
        :obj:`np.ndarray`
            First non-positive autocorrelation function lag of the linear model
            errors, for each cross-validation fold.

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
        .. [3] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting:
            principles and practice, 2nd edition, OTexts: Melbourne,
            Australia. OTexts.com/fpp2. Accessed on 26 May 2020.
        """
        acf_first_nonpos_linear = cls._model_acf_first_nonpos(
            ts=ts,
            perf_ft_method=cls.ft_model_linear,
            tskf=tskf,
            num_cv_folds=num_cv_folds,
            lm_sample_frac=lm_sample_frac,
            unbiased=unbiased,
            max_nlags=max_nlags)

        return acf_first_nonpos_linear
