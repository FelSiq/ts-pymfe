import typing as t

import numpy as np
import sklearn.model_selection

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
    def _get_sample_inds(cls, num_inst: int, lm_sample_frac: float,
                         random_state: t.Optional[int]) -> np.ndarray:
        """Sample indices to calculate subsampling landmarking metafeatures."""
        if random_state is not None:
            np.random.seed(random_state)

        sample_inds = np.random.choice(
            a=num_inst, size=int(lm_sample_frac * num_inst), replace=False)

        return sample_inds

    @classmethod
    def _sample_data(
            cls,
            ts: np.ndarray,
            lm_sample_frac: float,
            random_state: t.Optional[int] = None,
            sample_inds: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Select ``lm_sample_frac`` percent of data from ``ts``."""
        if lm_sample_frac >= 1.0 and sample_inds is None:
            return ts

        if sample_inds is None:
            num_inst = ts.size

            sample_inds = cls._get_sample_inds(
                num_inst=num_inst,
                lm_sample_frac=lm_sample_frac,
                random_state=random_state)

        return ts[sample_inds]

    @classmethod
    def ft_mean(
            cls,
            ts: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            tskf: t.Optional[sklearn.model_selection.TimeSeriesSplit] = None,
            num_cv_folds: int = 10,
            lm_sample_frac: float = 1.0,
            sample_inds: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None) -> np.ndarray:
        """TODO."""
        ts = cls._sample_data(
            ts=ts,
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
            res[ind_fold] = score(ts[inds_test], ts_pred)

        return res


def _test() -> None:
    ts = _get_data.load_data(3)

    ts_period = _period.ts_period(ts)
    ts_trend, ts_season, ts_residuals = _detrend.decompose(ts,
                                                           ts_period=ts_period)
    ts = ts.to_numpy()

    res = MFETSLandmarking.ft_mean(ts, score=sklearn.metrics.mean_absolute_error)
    print(res)


if __name__ == "__main__":
    _test()
