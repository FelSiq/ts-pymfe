import typing as t

import statsmodels.tsa.stattools
import numpy as np
import sklearn.mixture

import _utils
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

        acfs_locmin = np.flatnonzero(_utils.find_crit_pt(ts_acfs, type_="min"))

        try:
            return acfs_locmin[0] + 1

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

    @classmethod
    def ft_tc3(cls,
               ts: np.ndarray,
               lag: t.Optional[int] = None,
               only_numerator: bool = True,
               unbiased: bool = True,
               max_nlags: t.Optional[int] = None) -> float:
        """TODO."""
        if lag is None:
            lag = MFETSAutocorr.ft_first_acf_nonpos(ts=ts,
                                                    unbiased=unbiased,
                                                    max_nlags=max_nlags)

        ts_shift_1 = ts[:-2 * lag]
        ts_shift_2 = ts[lag:-lag]
        ts_shift_3 = ts[2 * lag:]

        _aux = ts_shift_1 * ts_shift_2
        numen = np.mean(_aux * ts_shift_3)

        if only_numerator:
            return numen

        denom = np.abs(np.mean(_aux))**1.5

        tc3 = numen / denom

        return tc3

    @classmethod
    def ft_gen_autocorr(cls,
                        ts: np.ndarray,
                        alpha: float = 1,
                        beta: float = 1,
                        lag: t.Optional[int] = None,
                        unbiased: bool = True,
                        max_nlags: t.Optional[int] = None) -> float:
        """TODO.

        References
        ----------
        .. [1] S.M. Duarte Queirós, L.G. Moyano, Yet on statistical properties
            of traded volume: Correlation and mutual information at different
            value magnitudes, Physica A: Statistical Mechanics and its
            Applications, Volume 383, Issue 1, 2007, Pages 10-15, ISSN
            0378-4371, https://doi.org/10.1016/j.physa.2007.04.082.

        .. [2] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework
            for Automated Time-Series Phenotyping Using Massive Feature
            Extraction, Cell Systems 5: 527 (2017).
            DOI: 10.1016/j.cels.2017.10.001

        .. [3] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
            time-series analysis: the empirical structure of time series and
            their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013).
            DOI: 10.1098/rsif.2013.0048
        """
        if lag is None:
            lag = MFETSAutocorr.ft_first_acf_nonpos(ts=ts,
                                                    unbiased=unbiased,
                                                    max_nlags=max_nlags)

        ts_abs = np.abs(ts)
        ts_sft_1 = ts_abs[:-lag]
        ts_sft_2 = ts_abs[lag:]

        ts_sft_1_a = ts_sft_1**alpha
        ts_sft_2_b = ts_sft_2**beta

        ts_sft_1_a_mean = np.mean(ts_sft_1_a)
        ts_sft_2_b_mean = np.mean(ts_sft_2_b)

        gen_autocorr = (
            np.mean(ts_sft_1_a * ts_sft_2_b) -
            ts_sft_1_a_mean * ts_sft_2_b_mean /
            (np.sqrt(np.mean(ts_sft_1**(2 * alpha)) - ts_sft_1_a_mean**2) *
             np.sqrt(np.mean(ts_sft_2**(2 * beta)) - ts_sft_2_b_mean**2)))

        return gen_autocorr

    @classmethod
    def ft_autocorr_crit_pt(
            cls,
            ts: np.ndarray,
            crit_point_type: str = "non-plateau",
            unbiased: bool = True,
            max_nlags: t.Optional[int] = None,
            ts_acfs: t.Optional[np.ndarray] = None) -> t.Union[int, float]:
        """TODO."""
        if ts_acfs is None:
            ts_acfs = cls._calc_acf(data=ts,
                                    nlags=max_nlags,
                                    unbiased=unbiased)

        ac_shape = _utils.find_crit_pt(arr=ts_acfs, type_=crit_point_type)

        # Note: in 'hctsa', either the sum or the mean is returned.
        # However, to enable summarization, here we return the whole
        # array.
        return ac_shape

    @classmethod
    def ft_autocorr_gaussian_resid(
        cls,
        ts: np.ndarray,
        n_components: int = 2,
        nlags: int = 2,
        unbiased: bool = True,
        random_state: t.Optional[int] = None,
        gaussian_model: t.Optional[sklearn.mixture.GaussianMixture] = None
    ) -> np.ndarray:
        """TODO."""
        gaussian_model = _utils.fit_gaussian_mix(ts=ts,
                                                 n_components=n_components,
                                                 random_state=random_state,
                                                 gaussian_model=gaussian_model)
        ts_preds = gaussian_model.predict(ts.reshape(-1, 1))
        ts_resid = ts - ts_preds
        gaussian_resid_acf = cls._calc_acf(data=ts_resid,
                                           nlags=nlags,
                                           unbiased=unbiased)

        return gaussian_resid_acf


def _test() -> None:
    ts = _get_data.load_data(3)
    ts_period = _period.ts_period(ts)
    ts_trend, ts_season, ts_residuals = _detrend.decompose(ts,
                                                           ts_period=ts_period)
    ts = ts.to_numpy()

    res = MFETSAutocorr.ft_autocorr_gaussian_resid(ts)
    print(res)
    exit(1)

    res = MFETSAutocorr.ft_autocorr_crit_pt(ts)
    print(res)
    exit(1)

    res = MFETSAutocorr.ft_gen_autocorr(ts)
    print(res)

    res = MFETSAutocorr.ft_tc3(ts, only_numerator=False)
    print(res)

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
