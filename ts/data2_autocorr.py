import typing as t

import statsmodels.tsa.stattools
import numpy as np

import get_data
import data1_detrend


class MFETSAutoCorr:
    @classmethod
    def _calc_acf(cls,
                  data: np.ndarray,
                  nlags: int = 5,
                  unbiased: bool = True) -> np.ndarray:
        """TODO."""
        acf = statsmodels.tsa.stattools.acf(data,
                                            nlags=nlags,
                                            unbiased=unbiased,
                                            fft=True)
        return acf[1:]

    @classmethod
    def _calc_pacf(cls,
                   data: np.ndarray,
                   nlags: int = 5,
                   method: str = "ols-unbiased") -> np.ndarray:
        """TODO."""
        pacf = statsmodels.tsa.stattools.pacf(data, nlags=nlags, method=method)
        return pacf[1:]

    @classmethod
    def ft_pacf(cls,
                ts: np.ndarray,
                nlags: int = 5,
                method: str = "ols-unbiased") -> np.ndarray:
        """TODO."""
        return cls._calc_pacf(data=ts, nlags=nlags, method=method)

    @classmethod
    def ft_pacf_residuals(cls,
                          ts_residuals: np.ndarray,
                          nlags: int = 5,
                          method: str = "ols-unbiased") -> np.ndarray:
        """TODO."""
        return cls._calc_pacf(data=ts_residuals, nlags=nlags, method=method)

    @classmethod
    def ft_pacf_detrended(cls,
                          ts: np.ndarray,
                          ts_trend: np.ndarray,
                          nlags: int = 5,
                          method: str = "ols-unbiased") -> np.ndarray:
        """TODO."""
        return cls._calc_pacf(data=ts - ts_trend, nlags=nlags, method=method)

    @classmethod
    def ft_pacf_deseasonalized(cls,
                               ts: np.ndarray,
                               ts_season: np.ndarray,
                               nlags: int = 5,
                               method: str = "ols-unbiased") -> np.ndarray:
        """TODO."""
        return cls._calc_pacf(data=ts - ts_season, nlags=nlags, method=method)

    @classmethod
    def ft_acf(cls,
               ts: np.ndarray,
               nlags: int = 5,
               unbiased: bool = True) -> np.ndarray:
        """TODO."""
        return cls._calc_acf(data=ts, nlags=nlags, unbiased=unbiased)

    @classmethod
    def ft_acf_residuals(cls,
                         ts_residuals: np.ndarray,
                         nlags: int = 5,
                         unbiased: bool = True) -> np.ndarray:
        """TODO."""
        return cls._calc_acf(data=ts_residuals, nlags=nlags, unbiased=unbiased)

    @classmethod
    def ft_acf_detrended(cls,
                         ts: np.ndarray,
                         ts_trend: np.ndarray,
                         nlags: int = 5,
                         unbiased: bool = True) -> np.ndarray:
        """TODO."""
        return cls._calc_acf(data=ts - ts_trend,
                             nlags=nlags,
                             unbiased=unbiased)

    @classmethod
    def ft_acf_deseasonalized(cls,
                              ts: np.ndarray,
                              ts_season: np.ndarray,
                              nlags: int = 5,
                              unbiased: bool = True) -> np.ndarray:
        """TODO."""
        return cls._calc_acf(data=ts - ts_season,
                             nlags=nlags,
                             unbiased=unbiased)


def _test() -> None:
    ts = get_data.load_data(2)
    ts_trend, ts_season, ts_residuals = data1_detrend.decompose(ts)
    ts = ts.to_numpy()

    res = MFETSAutoCorr.ft_acf(ts)
    print(res)

    res = MFETSAutoCorr.ft_pacf(ts)
    print(res)

    res = MFETSAutoCorr.ft_acf_residuals(ts_residuals)
    print(res)

    res = MFETSAutoCorr.ft_pacf_residuals(ts_residuals)
    print(res)

    res = MFETSAutoCorr.ft_acf_detrended(ts, ts_trend)
    print(res)

    res = MFETSAutoCorr.ft_pacf_detrended(ts, ts_trend)
    print(res)

    res = MFETSAutoCorr.ft_acf_deseasonalized(ts, ts_season)
    print(res)

    res = MFETSAutoCorr.ft_pacf_deseasonalized(ts, ts_season)
    print(res)


if __name__ == "__main__":
    _test()
