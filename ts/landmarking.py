"""TODO.

Future refs:
https://people.duke.edu/~rnau/411arim.htm#mixed
https://otexts.com/fpp2/taxonomy.html
"""
import typing as t
import warnings

import numpy as np
import arch
import sklearn.model_selection
import statsmodels.tsa.arima_model
import statsmodels.tsa.holtwinters
import statsmodels.tools.sm_exceptions

import autocorr
import _utils
import _period
import _detrend
import _get_data
import _models
import _embed


class MFETSLandmarking:
    """TODO."""
    @classmethod
    def _standard_pipeline_sklearn(
        cls,
        y: np.ndarray,
        model: t.Any,
        score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
        X: t.Optional[np.ndarray] = None,
        args_fit: t.Optional[t.Dict[str, t.Any]] = None,
        tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
        num_cv_folds: int = 5,
        lm_sample_frac: float = 1.0,
        scale_range: t.Optional[t.Tuple[int, int]] = (0, 1),
    ) -> np.ndarray:
        """TODO."""
        if tskf is None:
            tskf = sklearn.model_selection.TimeSeriesSplit(
                n_splits=num_cv_folds)

        if args_fit is None:
            args_fit = {}

        if X is None:
            # Note: 'X' are the unitless timesteps of the timeseries
            y = _utils.sample_data(ts=y, lm_sample_frac=lm_sample_frac)
            X = np.arange(y.size).reshape(-1, 1)

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

            except TypeError:
                res[ind_fold] = np.nan

        return res

    @classmethod
    def _standard_pipeline_statsmodels(
        cls,
        ts: np.ndarray,
        model_callable: t.Any,
        score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
        args_inst: t.Optional[t.Dict[str, t.Any]] = None,
        args_fit: t.Optional[t.Dict[str, t.Any]] = None,
        tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
        num_cv_folds: int = 5,
        lm_sample_frac: float = 1.0,
        scale_range: t.Optional[t.Tuple[int, int]] = (0, 1),
    ) -> np.ndarray:
        """TODO."""
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
            # Note: We are ignoring these warnings because they are related to poor
            # model fits when the time-series does not match the algorithm
            # assumptions. This is expected for a couple of model types, since
            # the same unprocessed time-series is fitted in a large amount of
            # different model types. It is also expected that bad score relates
            # with poor fits, and hence the metafeature will still reflect the true
            # (bad) relationship between the model and the data.
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

                except ValueError:
                    res[ind_fold] = np.nan

        return res

    @classmethod
    def _standard_pipeline_arch(
            cls,
            ts_residuals: np.ndarray,
            model_callable: t.Any,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            args_inst: t.Optional[t.Dict[str, t.Any]] = None,
            args_fit: t.Optional[t.Dict[str, t.Any]] = None,
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        """TODO."""
        if args_inst is None:
            args_inst = {}

        if args_fit is None:
            args_fit = {}

        ts_residuals = _utils.sample_data(ts=ts_residuals,
                                          lm_sample_frac=lm_sample_frac)

        if tskf is None:
            tskf = sklearn.model_selection.TimeSeriesSplit(
                n_splits=num_cv_folds)

        ts_residuals = ts_residuals.reshape(-1, 1)
        res = np.zeros(tskf.n_splits, dtype=float)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    module="arch",
                                    category=RuntimeWarning)

            for ind_fold, (inds_train,
                           inds_test) in enumerate(tskf.split(ts_residuals)):
                ts_train, ts_test = ts_residuals[inds_train], ts_residuals[
                    inds_test]

                try:
                    model = model_callable(ts_train,
                                           **args_inst).fit(**args_fit)

                    ts_var_pred = model.forecast(
                        horizon=ts_test.size).variance.values[-1, :]

                    res[ind_fold] = score(ts_var_pred, np.var(ts_test))

                except ValueError:
                    res[ind_fold] = np.nan

        return res

    @classmethod
    def _model_first_acf_nonpos(
            cls,
            ts: np.ndarray,
            perf_ft_method: t.Callable[..., np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
            unbiased: bool = True,
            max_nlags: t.Optional[int] = None,
    ) -> np.ndarray:
        """TODO."""
        score = lambda ts_pred, ts_test: (
            autocorr.MFETSAutocorr.ft_first_acf_nonpos(
                ts=ts_pred - ts_test, unbiased=unbiased, max_nlags=max_nlags))

        model_acf_first_nonpos = perf_ft_method(ts=ts,
                                                tskf=tskf,
                                                score=score,
                                                num_cv_folds=num_cv_folds,
                                                lm_sample_frac=lm_sample_frac)

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
        """TODO."""
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
    def ft_model_linear(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        """TODO."""
        model = sklearn.linear_model.LinearRegression()

        res = cls._standard_pipeline_sklearn(y=ts,
                                             model=model,
                                             score=score,
                                             tskf=tskf,
                                             num_cv_folds=num_cv_folds,
                                             lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_linear_embed_lag_1(
            cls,
            ts: np.ndarray,
            ts_period: int,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        """TODO."""
        model = sklearn.linear_model.LinearRegression()

        X = _embed.embed_ts(ts=ts, dim=ts_period, lag=1)

        res = cls._standard_pipeline_sklearn(y=ts[ts_period:],
                                             X=X,
                                             model=model,
                                             score=score,
                                             tskf=tskf,
                                             num_cv_folds=num_cv_folds,
                                             lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_linear_embed_lag_2(
            cls,
            ts: np.ndarray,
            ts_period: int,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        """TODO."""
        model = sklearn.linear_model.LinearRegression()

        X = _embed.embed_ts(ts=ts, dim=ts_period, lag=2)

        res = cls._standard_pipeline_sklearn(y=ts[ts_period:],
                                             X=X,
                                             model=model,
                                             score=score,
                                             tskf=tskf,
                                             num_cv_folds=num_cv_folds,
                                             lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_linear_embed_lag_m(
            cls,
            ts: np.ndarray,
            ts_period: int,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        """TODO."""
        model = sklearn.linear_model.LinearRegression()

        X = _embed.embed_ts(ts=ts, dim=3, lag=ts_period)

        res = cls._standard_pipeline_sklearn(y=ts[ts_period:],
                                             X=X,
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
            ts_period: int,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        """TODO."""
        model = sklearn.linear_model.LinearRegression()

        X = np.tile(np.arange(ts_period), 1 + ts.size // ts_period)[:ts.size,
                                                                    np.newaxis]
        # Note: remove one dummy variable to avoid the 'dummy
        # variable trap'.
        X = sklearn.preprocessing.OneHotEncoder(
            sparse=False).fit_transform(X)[:, 1:]

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
        """TODO."""
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
        """TODO."""
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
            ts_period: int,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        """TODO."""
        model = _models.TSNaiveSeasonal(ts_period=ts_period)

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
        """TODO."""
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
        """TODO."""
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
        """TODO."""
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
        """TODO."""
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
        """TODO."""
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
    def ft_model_arima_022_nc(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
            solver: str = "lbfgs",
            maxiter: int = 512,
    ) -> np.ndarray:
        """TODO."""
        model_callable = statsmodels.tsa.arima_model.ARIMA

        args_inst = {"order": (0, 2, 2)}
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
    def ft_model_arima_112_c(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
            solver: str = "lbfgs",
            maxiter: int = 512,
    ) -> np.ndarray:
        """TODO."""
        model_callable = statsmodels.tsa.arima_model.ARIMA

        args_inst = {"order": (1, 1, 2)}
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
    def ft_model_ses(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        """TODO."""
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
            ts_period: int,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        """TODO."""
        model_callable = statsmodels.tsa.holtwinters.ExponentialSmoothing

        args_inst = {
            "seasonal_periods": ts_period,
            "trend": "add",
            "seasonal": "add",
        }

        res = cls._standard_pipeline_statsmodels(ts=ts,
                                                 model_callable=model_callable,
                                                 score=score,
                                                 args_inst=args_inst,
                                                 tskf=tskf,
                                                 num_cv_folds=num_cv_folds,
                                                 lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_hwes_adm(
            cls,
            ts: np.ndarray,
            ts_period: int,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        """TODO."""
        model_callable = statsmodels.tsa.holtwinters.ExponentialSmoothing

        args_inst = {
            "seasonal_periods": ts_period,
            "trend": "add",
            "seasonal": "mul",
        }

        # Note: scaling time-series in [1, 2] interval rather than
        # [0, 1] because for Multiplicative Exponential Models,
        # the time-series values must be strictly positive.
        res = cls._standard_pipeline_statsmodels(ts=ts,
                                                 model_callable=model_callable,
                                                 score=score,
                                                 args_inst=args_inst,
                                                 tskf=tskf,
                                                 num_cv_folds=num_cv_folds,
                                                 lm_sample_frac=lm_sample_frac,
                                                 scale_range=(1, 2))

        return res

    """
    @classmethod
    def ft_model_arch_1_c(
            cls,
            ts_residuals: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        model_callable = arch.arch_model

        args_inst = {
            "mean": "constant",
            "p": 1,
            "vol": "ARCH",
            "rescale": True,
        }
        args_fit = {
            "update_freq": 0,
            "disp": "off",
            "options": {
                "disp": False
            },
            "show_warning": False,
        }

        res = cls._standard_pipeline_arch(ts_residuals=ts_residuals,
                                          model_callable=model_callable,
                                          score=score,
                                          args_inst=args_inst,
                                          args_fit=args_fit,
                                          tskf=tskf,
                                          num_cv_folds=num_cv_folds,
                                          lm_sample_frac=lm_sample_frac)

        return res

    @classmethod
    def ft_model_garch_11_c(
            cls,
            ts_residuals: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
    ) -> np.ndarray:
        model_callable = arch.arch_model

        args_inst = {
            "mean": "constant",
            "p": 1,
            "q": 1,
            "vol": "GARCH",
            "rescale": True,
        }
        args_fit = {
            "update_freq": 0,
            "disp": "off",
            "options": {
                "disp": False
            },
            "show_warning": False,
        }

        res = cls._standard_pipeline_arch(ts_residuals=ts_residuals,
                                          model_callable=model_callable,
                                          score=score,
                                          args_inst=args_inst,
                                          args_fit=args_fit,
                                          tskf=tskf,
                                          num_cv_folds=num_cv_folds,
                                          lm_sample_frac=lm_sample_frac)

        return res
    """

    @classmethod
    def ft_model_mean_first_acf_nonpos(
            cls,
            ts: np.ndarray,
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
            unbiased: bool = True,
            max_nlags: t.Optional[int] = None,
    ) -> np.ndarray:
        """TODO."""
        acf_first_nonpos_mean = cls._model_first_acf_nonpos(
            ts=ts,
            perf_ft_method=cls.ft_model_mean,
            tskf=tskf,
            num_cv_folds=num_cv_folds,
            lm_sample_frac=lm_sample_frac,
            unbiased=unbiased,
            max_nlags=max_nlags)

        return acf_first_nonpos_mean

    @classmethod
    def ft_model_linear_first_acf_nonpos(
            cls,
            ts: np.ndarray,
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
            unbiased: bool = True,
            max_nlags: t.Optional[int] = None,
    ) -> np.ndarray:
        """TODO."""
        acf_first_nonpos_linear = cls._model_first_acf_nonpos(
            ts=ts,
            perf_ft_method=cls.ft_model_linear,
            tskf=tskf,
            num_cv_folds=num_cv_folds,
            lm_sample_frac=lm_sample_frac,
            unbiased=unbiased,
            max_nlags=max_nlags)

        return acf_first_nonpos_linear


def _test() -> None:
    """Ref for not using SMAPE:

    1. https://otexts.com/fpp2/accuracy.html#fnref2
    2. Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. International Journal of Forecasting, 22, 679–688. https://robjhyndman.com/publications/another-look-at-measures-of-forecast-accuracy/
    """
    ts = _get_data.load_data(3)

    ts_period = _period.ts_period(ts)
    ts_trend, ts_season, ts_residuals = _detrend.decompose(ts,
                                                           ts_period=ts_period)
    ts = ts.to_numpy().astype(float)

    score = lambda *args: sklearn.metrics.mean_squared_error(*args,
                                                             squared=False)

    res = MFETSLandmarking.ft_model_hwes_ada(ts, ts_period, score=score)
    print(4, res)

    res = MFETSLandmarking.ft_model_hwes_adm(ts, ts_period, score=score)
    print(5, res)

    res = MFETSLandmarking.ft_model_linear_embed_lag_1(ts,
                                                       ts_period,
                                                       score=score)
    print(18, res)

    res = MFETSLandmarking.ft_model_linear_embed_lag_2(ts,
                                                       ts_period,
                                                       score=score)
    print(19, res)

    res = MFETSLandmarking.ft_model_linear_embed_lag_m(ts,
                                                       ts_period,
                                                       score=score)
    print(20, res)

    res = MFETSLandmarking.ft_model_linear_seasonal(ts, ts_period, score=score)
    print(17, res)

    res = MFETSLandmarking.ft_model_naive_drift(ts, score=score)
    print(14, res)

    res = MFETSLandmarking.ft_model_naive_seasonal(ts, ts_period, score=score)
    print(15, res)

    res = MFETSLandmarking.ft_model_naive(ts, score=score)
    print(16, res)

    res = MFETSLandmarking.ft_model_mean(ts, score=score)
    print(13, res)

    res = MFETSLandmarking.ft_model_mean_first_acf_nonpos(ts)
    print(14, res)
    """
    res = MFETSLandmarking.ft_model_arch_1_c(ts_residuals, score=score)
    print(1, res)

    res = MFETSLandmarking.ft_model_garch_11_c(ts_residuals,
                                               score=score)
    print(2, res)
    """

    res = MFETSLandmarking.ft_model_ses(ts, score=score)
    print(3, res)

    res = MFETSLandmarking.ft_model_arima_100_c(ts, score=score)
    print(6, res)

    res = MFETSLandmarking.ft_model_arima_010_c(ts, score=score)
    print(7, res)

    res = MFETSLandmarking.ft_model_arima_110_c(ts, score=score)
    print(8, res)

    res = MFETSLandmarking.ft_model_arima_011_nc(ts, score=score)
    print(9, res)

    res = MFETSLandmarking.ft_model_arima_011_c(ts, score=score)
    print(10, res)

    res = MFETSLandmarking.ft_model_arima_022_nc(ts,
                                                 score=score,
                                                 num_cv_folds=5)
    print(11, res)

    res = MFETSLandmarking.ft_model_arima_112_c(ts,
                                                score=score,
                                                num_cv_folds=5)
    print(12, res)

    res = MFETSLandmarking.ft_model_linear(ts, score=score)
    print(15, res)

    res = MFETSLandmarking.ft_model_linear_first_acf_nonpos(ts)
    print(16, res)


if __name__ == "__main__":
    _test()
