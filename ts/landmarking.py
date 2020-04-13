import typing as t
import warnings

import numpy as np
import sklearn.model_selection
import statsmodels.tsa.arima_model
import statsmodels.tools.sm_exceptions

import autocorr
import _utils
import _period
import _detrend
import _get_data


class _TSMeanModel:
    """TODO."""
    def __init__(self):
        self.avg = np.nan

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_TSMeanModel":
        """TODO."""
        self.avg = np.mean(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """TODO."""
        return np.full(X.shape, fill_value=self.avg)


class MFETSLandmarking:
    @classmethod
    def _standard_pipeline_sklearn(
            cls,
            y: np.ndarray,
            model: t.Any,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            args_fit: t.Optional[t.Dict[str, t.Any]] = None,
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
            sample_inds: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None) -> np.ndarray:
        """TODO."""
        if args_fit is None:
            args_fit = {}

        y = _utils.sample_data(ts=y,
                               lm_sample_frac=lm_sample_frac,
                               random_state=random_state,
                               sample_inds=sample_inds)

        if tskf is None:
            tskf = sklearn.model_selection.TimeSeriesSplit(
                n_splits=num_cv_folds)

        res = np.zeros(tskf.n_splits, dtype=float)

        # Note: x are the unitless timesteps of the timeseries
        X = np.arange(y.size).reshape(-1, 1)

        try:
            for ind_fold, (inds_train, inds_test) in enumerate(tskf.split(X)):
                X_train, X_test = X[inds_train, :], X[inds_test, :]
                y_train, y_test = y[inds_train], y[inds_test]

                model.fit(X_train, y_train, **args_fit)
                y_pred = model.predict(X_test).ravel()
                res[ind_fold] = score(y_pred, y_test)

        except ValueError:
            res[:] = np.nan

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
            sample_inds: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None) -> np.ndarray:
        """TODO."""
        if args_inst is None:
            args_inst = {}

        if args_fit is None:
            args_fit = {}

        ts = _utils.sample_data(ts=ts,
                                lm_sample_frac=lm_sample_frac,
                                random_state=random_state,
                                sample_inds=sample_inds)

        if tskf is None:
            tskf = sklearn.model_selection.TimeSeriesSplit(
                n_splits=num_cv_folds)

        res = np.zeros(tskf.n_splits, dtype=float)

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
                                    category=statsmodels.tools.sm_exceptions.
                                    HessianInversionWarning)

            try:
                for ind_fold, (inds_train,
                               inds_test) in enumerate(tskf.split(ts)):
                    ts_train, ts_test = ts[inds_train], ts[inds_test]

                    model = model_callable(ts_train,
                                           **args_inst).fit(**args_fit)
                    ts_pred = model.predict(start=ts_train.size,
                                            end=ts_train.size + ts_test.size -
                                            1,
                                            typ="levels")
                    res[ind_fold] = score(ts_pred, ts_test)

            except ValueError:
                res[:] = np.nan

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
            sample_inds: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None) -> np.ndarray:
        """TODO."""
        score = lambda ts_pred, ts_test: (
            autocorr.MFETSAutocorr.ft_first_acf_nonpos(
                ts=ts_pred - ts_test, unbiased=unbiased, max_nlags=max_nlags))

        model_acf_first_nonpos = perf_ft_method(ts=ts,
                                                tskf=tskf,
                                                score=score,
                                                num_cv_folds=num_cv_folds,
                                                lm_sample_frac=lm_sample_frac,
                                                sample_inds=sample_inds,
                                                random_state=random_state)

        return model_acf_first_nonpos

    @classmethod
    def ft_model_mean(cls,
                      ts: np.ndarray,
                      score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
                      tskf: t.Optional[
                          sklearn.model_selection.TimeSeriesSplit] = None,
                      num_cv_folds: int = 5,
                      lm_sample_frac: float = 1.0,
                      sample_inds: t.Optional[np.ndarray] = None,
                      random_state: t.Optional[int] = None) -> np.ndarray:
        """TODO."""
        res = cls._standard_pipeline_sklearn(y=ts,
                                             model=_TSMeanModel(),
                                             score=score,
                                             tskf=tskf,
                                             num_cv_folds=num_cv_folds,
                                             lm_sample_frac=lm_sample_frac,
                                             sample_inds=sample_inds,
                                             random_state=random_state)

        return res

    @classmethod
    def ft_model_linear(cls,
                        ts: np.ndarray,
                        score: t.Callable[[np.ndarray, np.ndarray],
                                          np.ndarray],
                        tskf: t.Optional[
                            sklearn.model_selection.TimeSeriesSplit] = None,
                        num_cv_folds: int = 5,
                        lm_sample_frac: float = 1.0,
                        sample_inds: t.Optional[np.ndarray] = None,
                        random_state: t.Optional[int] = None) -> np.ndarray:
        """TODO."""
        model = sklearn.linear_model.LinearRegression()

        res = cls._standard_pipeline_sklearn(y=ts,
                                             model=model,
                                             score=score,
                                             tskf=tskf,
                                             num_cv_folds=num_cv_folds,
                                             lm_sample_frac=lm_sample_frac,
                                             sample_inds=sample_inds,
                                             random_state=random_state)

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
            sample_inds: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None) -> np.ndarray:
        """TODO."""
        model = statsmodels.tsa.arima_model.ARIMA
        args_inst = {"order": (1, 0, 0)}
        args_fit = {
            "trend": "c",
            "disp": False,
            "transparams": False,
            "maxiter": maxiter,
            "solver": solver
        }

        res = cls._standard_pipeline_statsmodels(ts=ts,
                                                 model_callable=model,
                                                 args_inst=args_inst,
                                                 args_fit=args_fit,
                                                 score=score,
                                                 tskf=tskf,
                                                 num_cv_folds=num_cv_folds,
                                                 lm_sample_frac=lm_sample_frac,
                                                 sample_inds=sample_inds,
                                                 random_state=random_state)

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
            sample_inds: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None) -> np.ndarray:
        """TODO."""
        model = statsmodels.tsa.arima_model.ARIMA
        args_inst = {"order": (0, 1, 0)}
        args_fit = {
            "disp": False,
            "trend": "c",
            "transparams": False,
            "maxiter": maxiter,
            "solver": solver
        }

        res = cls._standard_pipeline_statsmodels(ts=ts,
                                                 model_callable=model,
                                                 args_inst=args_inst,
                                                 args_fit=args_fit,
                                                 score=score,
                                                 tskf=tskf,
                                                 num_cv_folds=num_cv_folds,
                                                 lm_sample_frac=lm_sample_frac,
                                                 sample_inds=sample_inds,
                                                 random_state=random_state)

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
            sample_inds: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None) -> np.ndarray:
        """TODO."""
        model = statsmodels.tsa.arima_model.ARIMA
        args_inst = {"order": (1, 1, 0)}
        args_fit = {
            "disp": False,
            "trend": "c",
            "transparams": False,
            "maxiter": maxiter,
            "solver": solver
        }

        res = cls._standard_pipeline_statsmodels(ts=ts,
                                                 model_callable=model,
                                                 args_inst=args_inst,
                                                 args_fit=args_fit,
                                                 score=score,
                                                 tskf=tskf,
                                                 num_cv_folds=num_cv_folds,
                                                 lm_sample_frac=lm_sample_frac,
                                                 sample_inds=sample_inds,
                                                 random_state=random_state)

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
            sample_inds: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None) -> np.ndarray:
        """TODO."""
        model = statsmodels.tsa.arima_model.ARIMA
        args_inst = {"order": (0, 1, 1)}
        args_fit = {
            "disp": False,
            "trend": "nc",
            "transparams": False,
            "maxiter": maxiter,
            "solver": solver
        }

        res = cls._standard_pipeline_statsmodels(ts=ts,
                                                 model_callable=model,
                                                 args_inst=args_inst,
                                                 args_fit=args_fit,
                                                 score=score,
                                                 tskf=tskf,
                                                 num_cv_folds=num_cv_folds,
                                                 lm_sample_frac=lm_sample_frac,
                                                 sample_inds=sample_inds,
                                                 random_state=random_state)

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
            sample_inds: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None) -> np.ndarray:
        """TODO."""
        model = statsmodels.tsa.arima_model.ARIMA
        args_inst = {"order": (0, 1, 1)}
        args_fit = {
            "disp": False,
            "trend": "c",
            "transparams": False,
            "maxiter": maxiter,
            "solver": solver
        }

        res = cls._standard_pipeline_statsmodels(ts=ts,
                                                 model_callable=model,
                                                 args_inst=args_inst,
                                                 args_fit=args_fit,
                                                 score=score,
                                                 tskf=tskf,
                                                 num_cv_folds=num_cv_folds,
                                                 lm_sample_frac=lm_sample_frac,
                                                 sample_inds=sample_inds,
                                                 random_state=random_state)

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
            sample_inds: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None) -> np.ndarray:
        """TODO."""
        model = statsmodels.tsa.arima_model.ARIMA
        args_inst = {"order": (0, 2, 2)}
        args_fit = {
            "disp": False,
            "trend": "nc",
            "transparams": False,
            "maxiter": maxiter,
            "solver": solver
        }

        res = cls._standard_pipeline_statsmodels(ts=ts,
                                                 model_callable=model,
                                                 args_inst=args_inst,
                                                 args_fit=args_fit,
                                                 score=score,
                                                 tskf=tskf,
                                                 num_cv_folds=num_cv_folds,
                                                 lm_sample_frac=lm_sample_frac,
                                                 sample_inds=sample_inds,
                                                 random_state=random_state)

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
            sample_inds: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None) -> np.ndarray:
        """TODO."""
        model = statsmodels.tsa.arima_model.ARIMA
        args_inst = {"order": (1, 1, 2)}
        args_fit = {
            "disp": False,
            "trend": "c",
            "transparams": False,
            "maxiter": maxiter,
            "solver": solver
        }

        res = cls._standard_pipeline_statsmodels(ts=ts,
                                                 model_callable=model,
                                                 args_inst=args_inst,
                                                 args_fit=args_fit,
                                                 score=score,
                                                 tskf=tskf,
                                                 num_cv_folds=num_cv_folds,
                                                 lm_sample_frac=lm_sample_frac,
                                                 sample_inds=sample_inds,
                                                 random_state=random_state)

        return res

    @classmethod
    def ft_model_mean_first_acf_nonpos(
            cls,
            ts: np.ndarray,
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 5,
            lm_sample_frac: float = 1.0,
            unbiased: bool = True,
            max_nlags: t.Optional[int] = None,
            sample_inds: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None) -> np.ndarray:
        """TODO."""
        acf_first_nonpos_mean = cls._model_first_acf_nonpos(
            ts=ts,
            perf_ft_method=cls.ft_model_mean,
            tskf=tskf,
            num_cv_folds=num_cv_folds,
            lm_sample_frac=lm_sample_frac,
            sample_inds=sample_inds,
            random_state=random_state,
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
            sample_inds: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None) -> np.ndarray:
        """TODO."""
        acf_first_nonpos_linear = cls._model_first_acf_nonpos(
            ts=ts,
            perf_ft_method=cls.ft_model_linear,
            tskf=tskf,
            num_cv_folds=num_cv_folds,
            lm_sample_frac=lm_sample_frac,
            sample_inds=sample_inds,
            random_state=random_state,
            unbiased=unbiased,
            max_nlags=max_nlags)

        return acf_first_nonpos_linear


def _test() -> None:
    ts = _get_data.load_data(3)

    ts_period = _period.ts_period(ts)
    ts_trend, ts_season, ts_residuals = _detrend.decompose(ts,
                                                           ts_period=ts_period)
    ts = ts.to_numpy().astype(float)

    res = MFETSLandmarking.ft_model_arima_100_c(ts, score=_utils.smape)
    print(res)

    res = MFETSLandmarking.ft_model_arima_010_c(ts, score=_utils.smape)
    print(res)

    res = MFETSLandmarking.ft_model_arima_110_c(ts, score=_utils.smape)
    print(res)

    res = MFETSLandmarking.ft_model_arima_011_nc(ts, score=_utils.smape)
    print(res)

    res = MFETSLandmarking.ft_model_arima_011_c(ts, score=_utils.smape)
    print(res)

    res = MFETSLandmarking.ft_model_arima_022_nc(ts,
                                                 score=_utils.smape,
                                                 num_cv_folds=5)
    print(res)

    res = MFETSLandmarking.ft_model_arima_112_c(ts,
                                                score=_utils.smape,
                                                num_cv_folds=5)
    print(res)

    res = MFETSLandmarking.ft_model_mean(ts, score=_utils.smape)
    print(res)

    res = MFETSLandmarking.ft_model_mean_first_acf_nonpos(ts)
    print(res)

    res = MFETSLandmarking.ft_model_linear(ts, score=_utils.smape)
    print(res)

    res = MFETSLandmarking.ft_model_linear_first_acf_nonpos(ts)
    print(res)


if __name__ == "__main__":
    _test()
