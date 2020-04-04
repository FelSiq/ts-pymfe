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
                           ts: np.ndarray,
                           num_lambdas: int = 16) -> float:
        """TODO.

        https://reader.elsevier.com/reader/sd/pii/S0169207016301030?token=9034548AA9BF5735897DD269867F9BAEF8BEA6226E2849CAA27A580DAE5C23849811A30FAFC6FE7D17FC7427D7385EF6
        """
        ts = ts - ts.min() + 0.1
        return scipy.stats.boxcox_normmax(ts, method="mle")


def _test() -> None:
    ts = get_data.load_data(2)
    ts_trend, ts_season, ts_residuals = data1_detrend.decompose(ts)
    ts = ts.to_numpy()

    res = MFETSDiversity.ft_opt_boxcox_coef(ts)
    print(res)


if __name__ == "__main__":
    _test()
