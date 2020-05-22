import typing as t

import statsmodels.tsa.stattools
import numpy as np
import sklearn.mixture

import stat_tests
import _embed
import _utils
import _detrend
import _period
import _get_data


class MFETSAutocorr:
    @classmethod
    def precompute_acf(cls,
                       ts: np.ndarray,
                       nlags: t.Optional[int] = None,
                       unbiased: bool = True,
                       **kwargs) -> t.Dict[str, np.ndarray]:
        """TODO."""
        precomp_vals = {}

        if "detrended_acfs" not in kwargs:
            precomp_vals["detrended_acfs"] = cls.ft_acf_detrended(
                ts=ts, nlags=nlags, unbiased=unbiased)

        return precomp_vals

    @classmethod
    def _calc_acf(cls,
                  ts: np.ndarray,
                  nlags: t.Optional[int] = None,
                  unbiased: bool = True,
                  detrend: bool = True,
                  detrended_acfs: t.Optional[np.ndarray] = None,
                  ts_detrended: t.Optional[np.ndarray] = None) -> np.ndarray:
        """TODO."""
        if detrended_acfs is not None and (nlags is None
                                           or tc_acfs.size == nlags):
            return detrended_acfs

        try:
            if detrend and ts_detrended is None:
                ts_detrended = _detrend.decompose(ts=ts, ts_period=0)[2]

        except ValueError:
            ts_detrended = ts

        if nlags is None:
            nlags = ts.size // 2

        acf = statsmodels.tsa.stattools.acf(ts,
                                            nlags=nlags,
                                            unbiased=unbiased,
                                            fft=True)
        return acf[1:]

    @classmethod
    def _calc_pacf(cls,
                   ts: np.ndarray,
                   nlags: t.Optional[int] = None,
                   method: str = "ols-unbiased",
                   detrend: bool = True,
                   ts_detrended: t.Optional[np.ndarray] = None) -> np.ndarray:
        """TODO."""
        if nlags is None:
            nlags = ts.size // 2

        try:
            if detrend and ts_detrended is None:
                ts_detrended = _detrend.decompose(ts=ts, ts_period=0)[2]

        except ValueError:
            ts_detrended = ts

        pacf = statsmodels.tsa.stattools.pacf(ts, nlags=nlags, method=method)
        return pacf[1:]

    @classmethod
    def _first_acf_below_threshold(
            cls,
            ts: np.ndarray,
            threshold: float,
            abs_acf_vals: bool = False,
            max_nlags: t.Optional[int] = None,
            unbiased: bool = True,
            detrended_acfs: t.Optional[np.ndarray] = None
    ) -> t.Union[int, float]:
        """TODO."""
        detrended_acfs = cls._calc_acf(ts=ts,
                                       nlags=max_nlags,
                                       unbiased=unbiased,
                                       detrended_acfs=detrended_acfs)

        if abs_acf_vals:
            # Note: in this case, we are testing if
            # -threshold <= acf <= threshold.
            detrended_acfs = np.abs(detrended_acfs)

        nonpos_acfs = np.flatnonzero(detrended_acfs <= threshold)

        try:
            return nonpos_acfs[0] + 1

        except IndexError:
            return np.nan

    @classmethod
    def ft_acf(cls,
               ts: np.ndarray,
               nlags: t.Optional[int] = None,
               unbiased: bool = True) -> np.ndarray:
        """TODO."""
        return cls._calc_acf(ts=ts,
                             nlags=nlags,
                             unbiased=unbiased,
                             detrend=False)

    @classmethod
    def ft_acf_detrended(
            cls,
            ts: np.ndarray,
            nlags: t.Optional[int] = None,
            unbiased: bool = True,
            ts_detrended: t.Optional[np.ndarray] = None,
            detrended_acfs: t.Optional[np.ndarray] = None) -> np.ndarray:
        """TODO."""
        return cls._calc_acf(ts=ts,
                             nlags=nlags,
                             unbiased=unbiased,
                             detrend=True,
                             detrended_acfs=detrended_acfs,
                             ts_detrended=ts_detrended)

    @classmethod
    def ft_acf_diff(cls,
                    ts: np.ndarray,
                    num_diff: int = 1,
                    nlags: t.Optional[int] = None,
                    detrend: bool = True,
                    ts_detrended: t.Optional[np.ndarray] = None,
                    unbiased: bool = True) -> np.ndarray:
        """TODO."""
        return cls._calc_acf(ts=np.diff(ts, n=num_diff),
                             detrend=detrend,
                             nlags=nlags,
                             unbiased=unbiased,
                             ts_detrended=ts_detrended)

    @classmethod
    def ft_pacf(cls,
                ts: np.ndarray,
                nlags: t.Optional[int] = None,
                method: str = "ols-unbiased") -> np.ndarray:
        """TODO."""
        return cls._calc_pacf(ts=ts, nlags=nlags, method=method, detrend=False)

    @classmethod
    def ft_pacf_detrended(
            cls,
            ts: np.ndarray,
            nlags: t.Optional[int] = None,
            method: str = "ols-unbiased",
            ts_detrended: t.Optional[np.ndarray] = None) -> np.ndarray:
        """TODO."""
        return cls._calc_pacf(ts=ts,
                              nlags=nlags,
                              method=method,
                              detrend=True,
                              ts_detrended=ts_detrended)

    @classmethod
    def ft_pacf_diff(
            cls,
            ts: np.ndarray,
            num_diff: int = 1,
            nlags: t.Optional[int] = None,
            method: str = "ols-unbiased",
            detrend: bool = True,
            ts_detrended: t.Optional[np.ndarray] = None) -> np.ndarray:
        """TODO."""
        return cls._calc_pacf(ts=np.diff(ts, n=num_diff),
                              nlags=nlags,
                              method=method,
                              detrend=detrend,
                              ts_detrended=ts_detrended)

    @classmethod
    def ft_acf_first_nonsig(
            cls,
            ts: np.ndarray,
            max_nlags: t.Optional[int] = None,
            unbiased: bool = True,
            detrended_acfs: t.Optional[np.ndarray] = None
    ) -> t.Union[int, float]:
        """TODO."""
        threshold = 1.96 / np.sqrt(ts.size)

        res = cls._first_acf_below_threshold(ts=ts,
                                             threshold=threshold,
                                             abs_acf_vals=True,
                                             max_nlags=max_nlags,
                                             unbiased=unbiased,
                                             detrended_acfs=detrended_acfs)
        return res

    @classmethod
    def ft_acf_first_nonpos(
            cls,
            ts: np.ndarray,
            max_nlags: t.Optional[int] = None,
            unbiased: bool = True,
            detrended_acfs: t.Optional[np.ndarray] = None
    ) -> t.Union[int, float]:
        """TODO."""
        res = cls._first_acf_below_threshold(ts=ts,
                                             threshold=0,
                                             abs_acf_vals=False,
                                             max_nlags=max_nlags,
                                             unbiased=unbiased,
                                             detrended_acfs=detrended_acfs)
        return res

    @classmethod
    def ft_first_acf_locmin(
            cls,
            ts: np.ndarray,
            max_nlags: t.Optional[int] = None,
            unbiased: bool = True,
            detrended_acfs: t.Optional[np.ndarray] = None
    ) -> t.Union[int, float]:
        """TODO."""
        detrended_acfs = cls._calc_acf(ts=ts,
                                       nlags=max_nlags,
                                       unbiased=unbiased,
                                       detrended_acfs=detrended_acfs)

        acfs_locmin = np.flatnonzero(
            _utils.find_crit_pt(detrended_acfs, type_="min"))

        try:
            return acfs_locmin[0] + 1

        except IndexError:
            return np.nan

    @classmethod
    def ft_trev(cls,
                ts: np.ndarray,
                lag: t.Optional[t.Union[str, int]] = None,
                only_numerator: bool = False,
                max_nlags: t.Optional[int] = None,
                detrended_acfs: t.Optional[np.ndarray] = None,
                detrended_ami: t.Optional[np.ndarray] = None) -> float:
        """TODO.

        Normalized nonlinear autocorrelation.

        https://github.com/benfulcher/hctsa/blob/master/Operations/CO_trev.m
        """
        lag = _embed.embed_lag(ts=ts,
                               lag=lag,
                               max_nlags=max_nlags,
                               detrended_acfs=detrended_acfs,
                               detrended_ami=detrended_ami)

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
               lag: t.Optional[t.Union[str, int]] = None,
               only_numerator: bool = False,
               max_nlags: t.Optional[int] = None,
               detrended_acfs: t.Optional[np.ndarray] = None,
               detrended_ami: t.Optional[np.ndarray] = None) -> float:
        """TODO."""
        lag = _embed.embed_lag(ts=ts,
                               lag=lag,
                               max_nlags=max_nlags,
                               detrended_acfs=detrended_acfs,
                               detrended_ami=detrended_ami)

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
                        lag: t.Optional[t.Union[str, int]] = None,
                        max_nlags: t.Optional[int] = None,
                        detrended_acfs: t.Optional[np.ndarray] = None,
                        detrended_ami: t.Optional[np.ndarray] = None) -> float:
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
        lag = _embed.embed_lag(ts=ts,
                               lag=lag,
                               max_nlags=max_nlags,
                               detrended_acfs=detrended_acfs,
                               detrended_ami=detrended_ami)

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
            detrended_acfs: t.Optional[np.ndarray] = None
    ) -> t.Union[int, float]:
        """TODO."""
        detrended_acfs = cls._calc_acf(ts=ts,
                                       nlags=max_nlags,
                                       unbiased=unbiased,
                                       detrended_acfs=detrended_acfs)

        ac_shape = _utils.find_crit_pt(arr=detrended_acfs,
                                       type_=crit_point_type)

        # Note: in 'hctsa', either the sum or the mean is returned.
        # However, to enable summarization, here we return the whole
        # array.
        return ac_shape

    @classmethod
    def ft_autocorr_gaussian_resid(
        cls,
        ts: np.ndarray,
        n_components: int = 2,
        nlags: int = 8,
        unbiased: bool = True,
        random_state: t.Optional[int] = None,
        ts_scaled: t.Optional[np.ndarray] = None,
        gaussian_resid: t.Optional[np.ndarray] = None,
        gaussian_model: t.Optional[sklearn.mixture.GaussianMixture] = None,
    ) -> np.ndarray:
        """TODO."""
        if gaussian_resid is None:
            ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

            gaussian_resid = _utils.fit_gaussian_mix(
                ts=ts_scaled,
                n_components=n_components,
                random_state=random_state,
                gaussian_model=gaussian_model,
                return_residuals=True)

        gaussian_resid_acf = cls._calc_acf(ts=gaussian_resid,
                                           nlags=nlags,
                                           unbiased=unbiased)

        return gaussian_resid_acf

    @classmethod
    def ft_test_gaussian_resid(
        cls,
        ts: np.ndarray,
        n_components: int = 2,
        nlags: int = 8,
        return_pval: bool = True,
        random_state: t.Optional[int] = None,
        ts_scaled: t.Optional[np.ndarray] = None,
        gaussian_resid: t.Optional[np.ndarray] = None,
        gaussian_model: t.Optional[sklearn.mixture.GaussianMixture] = None,
    ) -> np.ndarray:
        """TODO."""
        if gaussian_resid is None:
            ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

            gaussian_resid = _utils.fit_gaussian_mix(
                ts=ts_scaled,
                n_components=n_components,
                random_state=random_state,
                gaussian_model=gaussian_model,
                return_residuals=True)

        gaussian_lb_test = stat_tests.MFETSStatTests.ft_test_lb(
            ts_residuals=gaussian_resid,
            max_lags=nlags,
            return_pval=return_pval)

        return gaussian_lb_test

    @classmethod
    def ft_autocorr_out_dist(
            cls,
            ts: np.ndarray,
            p: float = 0.8,
            max_nlags: t.Optional[int] = None,
            unbiased: bool = True,
            detrended_acfs: t.Optional[np.ndarray] = None) -> np.ndarray:
        """TODO."""
        detrended_acfs = cls._calc_acf(ts=ts,
                                       nlags=max_nlags,
                                       unbiased=unbiased,
                                       detrended_acfs=detrended_acfs)

        ts_abs = np.abs(ts)
        ts_inliners = ts[ts_abs <= np.quantile(ts_abs, p)]

        ts_inliners_acfs = cls._calc_acf(ts=ts_inliners,
                                         nlags=max_nlags,
                                         unbiased=unbiased)

        dist_acfs = np.abs(detrended_acfs[:ts_inliners_acfs.size] -
                           ts_inliners_acfs)

        return dist_acfs


def _test() -> None:
    ts = _get_data.load_data(3)
    ts_period = _period.ts_period(ts)
    ts_trend, ts_season, ts_residuals = _detrend.decompose(ts,
                                                           ts_period=ts_period)
    ts = ts.to_numpy()

    res = MFETSAutocorr.ft_trev(ts, only_numerator=False)
    print(res)

    res = MFETSAutocorr.ft_acf_first_nonsig(ts)
    print(res)

    res = MFETSAutocorr.ft_acf_first_nonpos(ts)
    print(res)

    res = MFETSAutocorr.ft_tc3(ts, only_numerator=False)
    print(res)

    res = MFETSAutocorr.ft_autocorr_out_dist(ts)
    print(res)

    res = MFETSAutocorr.ft_first_acf_locmin(ts)
    print(res)

    res = MFETSAutocorr.ft_test_gaussian_resid(ts, random_state=16)
    print(res)

    res = MFETSAutocorr.ft_autocorr_gaussian_resid(ts, random_state=16)
    print(res)

    res = MFETSAutocorr.ft_autocorr_crit_pt(ts)
    print(res)

    res = MFETSAutocorr.ft_gen_autocorr(ts)
    print(res)

    res = MFETSAutocorr.ft_acf_first_nonpos(ts)
    print(res)

    res = MFETSAutocorr.ft_acf_detrended(ts)
    print(res)

    res = MFETSAutocorr.ft_pacf(ts)
    print(res)

    res = MFETSAutocorr.ft_acf_diff(ts)
    print(res)

    res = MFETSAutocorr.ft_pacf_diff(ts)
    print(res)


if __name__ == "__main__":
    _test()
