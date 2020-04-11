import typing as t

import statsmodels.tsa.stattools
import numpy as np

import _detrend
import _period
import _get_data


class MFETSAutocorr:
    @classmethod
    def _calc_acf(cls,
                  data: np.ndarray,
                  nlags: t.Optional[int] = 5,
                  unbiased: bool = True) -> np.ndarray:
        """TODO."""
        if nlags is None:
            nlags = data.size // 2

        acf = statsmodels.tsa.stattools.acf(data,
                                            nlags=nlags,
                                            unbiased=unbiased,
                                            fft=True)
        return acf[1:]

    @classmethod
    def _calc_pacf(cls,
                   data: np.ndarray,
                   nlags: t.Optional[int] = 5,
                   method: str = "ols-unbiased") -> np.ndarray:
        """TODO."""
        if nlags is None:
            nlags = data.size // 2

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
    def ft_pacf_diff(cls,
                     ts: np.ndarray,
                     num_diff: int = 1,
                     nlags: int = 5,
                     method: str = "ols-unbiased") -> np.ndarray:
        """TODO."""
        return cls._calc_pacf(data=np.diff(ts, n=num_diff),
                              nlags=nlags,
                              method=method)

    @classmethod
    def ft_pacf_residuals(cls,
                          ts_residuals: np.ndarray,
                          nlags: int = 5,
                          method: str = "ols-unbiased") -> np.ndarray:
        """TODO."""
        return cls._calc_pacf(data=ts_residuals, nlags=nlags, method=method)

    @classmethod
    def ft_pacf_trend(cls,
                      ts_trend: np.ndarray,
                      nlags: int = 5,
                      method: str = "ols-unbiased") -> np.ndarray:
        """TODO."""
        return cls._calc_pacf(data=ts_trend, nlags=nlags, method=method)

    @classmethod
    def ft_pacf_seasonality(cls,
                            ts_season: np.ndarray,
                            nlags: int = 5,
                            method: str = "ols-unbiased") -> np.ndarray:
        """TODO."""
        return cls._calc_pacf(data=ts_season, nlags=nlags, method=method)

    @classmethod
    def ft_pacf_detrended(cls,
                          ts_detrended: np.ndarray,
                          nlags: int = 5,
                          method: str = "ols-unbiased") -> np.ndarray:
        """TODO."""
        return cls._calc_pacf(data=ts_detrended, nlags=nlags, method=method)

    @classmethod
    def ft_pacf_deseasonalized(cls,
                               ts_deseasonalized: np.ndarray,
                               nlags: int = 5,
                               method: str = "ols-unbiased") -> np.ndarray:
        """TODO."""
        return cls._calc_pacf(data=ts_deseasonalized,
                              nlags=nlags,
                              method=method)

    @classmethod
    def ft_acf(cls,
               ts: np.ndarray,
               nlags: int = 5,
               unbiased: bool = True,
               ts_acfs: t.Optional[np.ndarray] = None) -> np.ndarray:
        """TODO."""
        if ts_acfs is not None and ts_acfs.size == nlags:
            return ts_acfs

        return cls._calc_acf(data=ts, nlags=nlags, unbiased=unbiased)

    @classmethod
    def ft_acf_diff(cls,
                    ts: np.ndarray,
                    num_diff: int = 1,
                    nlags: int = 5,
                    unbiased: bool = True) -> np.ndarray:
        """TODO."""
        return cls._calc_acf(data=np.diff(ts, n=num_diff),
                             nlags=nlags,
                             unbiased=unbiased)

    @classmethod
    def ft_acf_residuals(cls,
                         ts_residuals: np.ndarray,
                         nlags: int = 5,
                         unbiased: bool = True) -> np.ndarray:
        """TODO."""
        return cls._calc_acf(data=ts_residuals, nlags=nlags, unbiased=unbiased)

    @classmethod
    def ft_acf_trend(cls,
                     ts_trend: np.ndarray,
                     nlags: int = 5,
                     unbiased: bool = True) -> np.ndarray:
        """TODO."""
        return cls._calc_acf(data=ts_trend, nlags=nlags, unbiased=unbiased)

    @classmethod
    def ft_acf_seasonality(cls,
                           ts_season: np.ndarray,
                           nlags: int = 5,
                           unbiased: bool = True) -> np.ndarray:
        """TODO."""
        return cls._calc_acf(data=ts_season, nlags=nlags, unbiased=unbiased)

    @classmethod
    def ft_acf_detrended(cls,
                         ts_detrended: np.ndarray,
                         nlags: int = 5,
                         unbiased: bool = True) -> np.ndarray:
        """TODO."""
        return cls._calc_acf(data=ts_detrended, nlags=nlags, unbiased=unbiased)

    @classmethod
    def ft_acf_deseasonalized(cls,
                              ts_deseasonalized: np.ndarray,
                              nlags: int = 5,
                              unbiased: bool = True) -> np.ndarray:
        """TODO."""
        return cls._calc_acf(data=ts_deseasonalized,
                             nlags=nlags,
                             unbiased=unbiased)

    @classmethod
    def ft_first_acf_nonpos(
            cls,
            ts: np.ndarray,
            max_nlags: t.Optional[int] = None,
            unbiased: bool = True,
            ts_acfs: t.Optional[np.ndarray] = None) -> t.Union[int, float]:
        """TODO."""
        if ts_acfs is None:
            ts_acfs = cls._calc_acf(data=ts,
                                    nlags=max_nlags,
                                    unbiased=unbiased)

        nonpos_acfs = np.flatnonzero(ts_acfs <= 0)

        try:
            return nonpos_acfs[0] + 1

        except IndexError:
            return np.nan

    @classmethod
    def ft_first_acf_locmin(
            cls,
            ts: np.ndarray,
            max_nlags: t.Optional[int] = None,
            unbiased: bool = True,
            ts_acfs: t.Optional[np.ndarray] = None) -> t.Union[int, float]:
        """TODO."""
        if ts_acfs is None:
            ts_acfs = cls._calc_acf(data=ts,
                                    nlags=max_nlags,
                                    unbiased=unbiased)

        if ts_acfs.size <= 2:
            return np.nan

        acfs_diff = np.diff(ts_acfs)
        acfs_locmin = np.flatnonzero(
            np.logical_and(0 < acfs_diff[1:], 0 > acfs_diff[:-1]))

        try:
            return acfs_locmin[0] + 2

        except IndexError:
            return np.nan

    @classmethod
    def _apply_on_ts_samples(
            cls,
            ts: np.ndarray,
            func: t.Callable[[np.ndarray], float],
            num_samples: int = 128,
            sample_size_frac: float = 0.2,
            max_nlags: t.Optional[int] = None,
            unbiased: bool = True,
            random_state: t.Optional[int] = None) -> np.ndarray:
        """TODO."""
        if not 0 < sample_size_frac < 1:
            raise ValueError("'sample_size_frac' must be in (0, 1) "
                             "range (got {}).".format(sample_size_frac))

        if random_state is not None:
            np.random.seed(random_state)

        sample_size = int(np.ceil(ts.size * sample_size_frac))
        start_inds = np.random.randint(ts.size - sample_size + 1,
                                       size=num_samples)

        res = np.array([
            cls.ft_first_acf_nonpos(ts=ts[s_ind:s_ind + sample_size],
                                    max_nlags=max_nlags,
                                    unbiased=unbiased) for s_ind in start_inds
        ],
                       dtype=float)

        # Note: the original metafeatures are the mean value of
        # 'result'. However, to enable summarization,
        # here we return all the values.
        return res

    @classmethod
    def ft_sfirst_acf_nonpos(
            cls,
            ts: np.ndarray,
            num_samples: int = 128,
            sample_size_frac: float = 0.2,
            max_nlags: t.Optional[int] = None,
            unbiased: bool = True,
            random_state: t.Optional[int] = None) -> np.ndarray:
        """TODO."""
        sample_acf_nonpos = cls._apply_on_ts_samples(
            ts=ts,
            func=cls.ft_first_acf_nonpos,
            num_samples=num_samples,
            sample_size_frac=sample_size_frac,
            random_state=random_state,
            max_nlags=max_nlags,
            unbiased=unbiased)

        return sample_acf_nonpos

    @classmethod
    def ft_sfirst_acf_locmin(
            cls,
            ts: np.ndarray,
            num_samples: int = 128,
            sample_size_frac: float = 0.2,
            max_nlags: t.Optional[int] = None,
            unbiased: bool = True,
            random_state: t.Optional[int] = None) -> np.ndarray:
        """TODO."""
        sample_acf_locmin = cls._apply_on_ts_samples(
            ts=ts,
            func=cls.ft_first_acf_locmin,
            num_samples=num_samples,
            sample_size_frac=sample_size_frac,
            random_state=random_state,
            max_nlags=max_nlags,
            unbiased=unbiased)

        return sample_acf_locmin

    @classmethod
    def ft_trev(cls,
                ts: np.ndarray,
                lag: int = 1,
                only_numerator: bool = False) -> float:
        """TODO.

        Normalized nonlinear autocorrelation.

        https://github.com/benfulcher/hctsa/blob/master/Operations/CO_trev.m
        """
        diff = ts[lag:] - ts[:-lag]

        numen = np.mean(np.power(diff, 3))

        if only_numerator:
            return numen

        denom = np.power(np.mean(np.square(diff)), 1.5)
        trev = numen / denom

        return trev


def _test() -> None:
    ts = _get_data.load_data(3)
    ts_period = _period.ts_period(ts)
    ts_trend, ts_season, ts_residuals = _detrend.decompose(ts,
                                                           ts_period=ts_period)
    ts = ts.to_numpy()

    res = MFETSAutocorr.ft_sfirst_acf_locmin(ts)
    print(res)

    res = MFETSAutocorr.ft_sfirst_acf_nonpos(ts)
    print(res)

    res = MFETSAutocorr.ft_first_acf_locmin(ts)
    print(res)

    res = MFETSAutocorr.ft_first_acf_nonpos(ts)
    print(res)

    res = MFETSAutocorr.ft_acf(ts)
    print(res)

    res = MFETSAutocorr.ft_pacf(ts)
    print(res)

    res = MFETSAutocorr.ft_acf_trend(ts_trend)
    print(res)

    res = MFETSAutocorr.ft_pacf_trend(ts_trend)
    print(res)

    res = MFETSAutocorr.ft_acf_residuals(ts_residuals)
    print(res)

    res = MFETSAutocorr.ft_pacf_residuals(ts_residuals)
    print(res)

    res = MFETSAutocorr.ft_acf_seasonality(ts_season)
    print(res)

    res = MFETSAutocorr.ft_pacf_seasonality(ts_season)
    print(res)

    res = MFETSAutocorr.ft_acf_detrended(ts - ts_trend)
    print(res)

    res = MFETSAutocorr.ft_pacf_detrended(ts - ts_trend)
    print(res)

    res = MFETSAutocorr.ft_acf_deseasonalized(ts - ts_season)
    print(res)

    res = MFETSAutocorr.ft_pacf_deseasonalized(ts - ts_season)
    print(res)

    res = MFETSAutocorr.ft_acf_diff(ts)
    print(res)

    res = MFETSAutocorr.ft_pacf_diff(ts)
    print(res)

    res = MFETSAutocorr.ft_trev(ts, only_numerator=True)
    print(res)


if __name__ == "__main__":
    _test()
