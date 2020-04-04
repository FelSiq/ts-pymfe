import typing as t

import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import scipy.stats
import numpy as np

import get_data
import data1_detrend


class MFETSDiversity:
    @staticmethod
    def calc_smape(arr_a: np.ndarray, arr_b: np.ndarray) -> float:
        return 200 * np.mean(np.abs(arr_a - arr_b) / (arr_a + arr_b))

    @classmethod
    def ft_opt_boxcox_coef(cls,
                           ts_deseasonalized: np.ndarray,
                           test_frac: float = 0.25,
                           num_lambdas: int = 16) -> float:
        """TODO.

        https://reader.elsevier.com/reader/sd/pii/S0169207016301030?token=9034548AA9BF5735897DD269867F9BAEF8BEA6226E2849CAA27A580DAE5C23849811A30FAFC6FE7D17FC7427D7385EF6
        """
        mse = np.zeros(num_lambdas, dtype=float)
        lambdas_ = np.linspace(0, 1, num_lambdas)

        timesteps_train, timesteps_test, ts_train, ts_test = (
            sklearn.model_selection.train_test_split(
                np.arange(ts_deseasonalized.size).reshape(-1, 1),
                ts_deseasonalized - np.min(ts_deseasonalized) + 1,
                test_size=test_frac,
                shuffle=False))

        for ind, lambda_ in enumerate(lambdas_):
            ts_transf = scipy.stats.boxcox(ts_train, lmbda=lambda_)
            model = sklearn.linear_model.Ridge(
                copy_X=False).fit(timesteps_train, ts_transf)
            ts_pred = model.predict(timesteps_test)
            mse[ind] = sklearn.metrics.mean_squared_error(ts_test, ts_pred)

        return lambdas_[np.argmin(mse)]


def _test() -> None:
    ts = get_data.load_data(2)
    ts_trend, ts_season, ts_residuals = data1_detrend.decompose(ts)
    ts = ts.to_numpy()

    res = MFETSDiversity.ft_opt_boxcox_coef(ts_trend + ts_residuals)
    print(res)


if __name__ == "__main__":
    _test()
