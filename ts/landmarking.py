import typing as t

import numpy as np
import sklearn.model_selection

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
    def _standard_pipeline(cls,
                           y: np.ndarray,
                           model: t.Any,
                           score: t.Callable[[np.ndarray, np.ndarray],
                                             np.ndarray],
                           tskf: t.Optional[
                               sklearn.model_selection.TimeSeriesSplit] = None,
                           num_cv_folds: int = 10,
                           lm_sample_frac: float = 1.0,
                           sample_inds: t.Optional[np.ndarray] = None,
                           random_state: t.Optional[int] = None) -> np.ndarray:
        """TODO."""
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

        for ind_fold, (inds_train, inds_test) in enumerate(tskf.split(X)):
            X_train, X_test = X[inds_train, :], X[inds_test, :]
            y_train, y_test = y[inds_train], y[inds_test]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test).ravel()
            res[ind_fold] = score(y_pred, y_test)

        return res

    @classmethod
    def _model_first_acf_nonpos(
            cls,
            ts: np.ndarray,
            perf_ft_method: t.Callable[..., np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 10,
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
                      num_cv_folds: int = 10,
                      lm_sample_frac: float = 1.0,
                      sample_inds: t.Optional[np.ndarray] = None,
                      random_state: t.Optional[int] = None) -> np.ndarray:
        """TODO."""
        res = cls._standard_pipeline(y=ts,
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
                        num_cv_folds: int = 10,
                        lm_sample_frac: float = 1.0,
                        sample_inds: t.Optional[np.ndarray] = None,
                        random_state: t.Optional[int] = None) -> np.ndarray:
        """TODO."""
        model = sklearn.linear_model.LinearRegression()

        res = cls._standard_pipeline(y=ts,
                                     model=model,
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
            num_cv_folds: int = 10,
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
            random_state=random_state)

        return acf_first_nonpos_mean

    @classmethod
    def ft_model_linear_first_acf_nonpos(
            cls,
            ts: np.ndarray,
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 10,
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
            random_state=random_state)

        return acf_first_nonpos_linear


def _test() -> None:
    ts = _get_data.load_data(3)

    ts_period = _period.ts_period(ts)
    ts_trend, ts_season, ts_residuals = _detrend.decompose(ts,
                                                           ts_period=ts_period)
    ts = ts.to_numpy()

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
