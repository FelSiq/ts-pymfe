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
    def fit(self, ts_train: np.ndarray) -> "_TSMeanModel":
        """TODO."""
        self.avg = np.mean(ts_train)
        return self

    def predict(self, ts_test: np.ndarray) -> np.ndarray:
        """TODO."""
        return np.full(ts_test.shape, fill_value=self.avg)


class MFETSLandmarking:
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
        ts = _utils.sample_data(ts=ts,
                                lm_sample_frac=lm_sample_frac,
                                random_state=random_state,
                                sample_inds=sample_inds)

        if tskf is None:
            tskf = sklearn.model_selection.TimeSeriesSplit(
                n_splits=num_cv_folds)

        model = _TSMeanModel()
        res = np.zeros(tskf.n_splits, dtype=float)

        for ind_fold, (inds_train, inds_test) in enumerate(tskf.split(ts)):
            model.fit(ts[inds_train])
            ts_pred = model.predict(ts[inds_test])
            res[ind_fold] = score(ts_pred, ts[inds_test])

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
        score = lambda ts_pred, ts_test: (
            autocorr.MFETSAutocorr.ft_first_acf_nonpos(
                ts=ts_pred - ts_test, unbiased=unbiased, max_nlags=max_nlags))

        mean_acf_first_nonpos = cls.ft_model_mean(
            ts=ts,
            tskf=tskf,
            score=score,
            num_cv_folds=num_cv_folds,
            lm_sample_frac=lm_sample_frac,
            sample_inds=sample_inds,
            random_state=random_state)

        return mean_acf_first_nonpos


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


if __name__ == "__main__":
    _test()
