import typing as t

import numpy as np
import scipy.stats
import pandas as pd

import autocorr
import _detrend
import _embed
import _period
import _utils
import _get_data


class MFETSInfoTheory:
    @classmethod
    def precompute_auto_mut_info(cls,
                                 ts: np.ndarray,
                                 num_bins: int = 64,
                                 max_nlags: t.Optional[int] = None,
                                 return_dist: bool = True,
                                 unbiased: bool = True,
                                 **kwargs) -> t.Dict[str, np.ndarray]:
        """TODO."""
        precomp_vals = {}

        ts_acfs = None

        if max_nlags is None:
            ts_acfs = kwargs.get("ts_acfs")

            if ts_acfs is None:
                precomp_vals.update(
                    autocorr.precompute_acf(ts=ts,
                                            nlags=max_nlags,
                                            unbiased=unbiased))
                ts_acfs = kwargs["ts_acfs"]

        if "auto_mut_info" in kwargs:
            precomp_vals["auto_mut_info"] = cls.ft_auto_mut_info(
                ts=ts,
                num_bins=num_bins,
                max_nlags=max_nlags,
                return_dist=return_dist,
                unbiased=unbiased,
                ts_acfs=ts_acfs)

        return precomp_vals

    @classmethod
    def ft_hist_entropy(cls,
                        ts: np.ndarray,
                        num_bins: int = 10,
                        normalize: bool = True) -> np.ndarray:
        """TODO."""
        freqs = np.histogram(ts)[0]

        entropy = scipy.stats.entropy(freqs, base=2)

        if normalize:
            entropy /= np.log2(freqs.size)

        return entropy

    @classmethod
    def ft_first_crit_pt_ami(
        cls,
        ts: np.ndarray,
        num_bins: int = 64,
        max_nlags: int = 64,
        return_dist: bool = True,
        auto_mut_info: t.Optional[np.ndarray] = None
    ) -> t.Union[int, float]:
        """TODO."""
        if auto_mut_info is None:
            auto_mut_info = cls.ft_auto_mut_info(ts=ts,
                                                 num_bins=num_bins,
                                                 max_nlags=max_nlags,
                                                 return_dist=return_dist)

        # Note: if 'return_dist=True', return the first local maximum.
        # If otherwise, return the first local minimum.
        type_ = "max" if return_dist else "min"

        crit_point = _utils.find_crit_pt(arr=auto_mut_info, type_=type_)

        try:
            return np.flatnonzero(crit_point)[0] + 1

        except IndexError:
            return np.nan

    @classmethod
    def ft_auto_mut_info(
            cls,
            ts: np.ndarray,
            num_bins: int = 64,
            max_nlags: t.Optional[int] = None,
            return_dist: bool = True,
            unbiased: bool = True,
            ts_acfs: t.Optional[np.ndarray] = None,
            auto_mut_info: t.Optional[np.ndarray] = None) -> float:
        """TODO."""
        if auto_mut_info is not None:
            return auto_mut_info

        if max_nlags is None:
            _aux = autocorr.MFETSAutocorr.ft_first_acf_nonpos(
                ts=ts, unbiased=unbiased, ts_acfs=ts_acfs)
            max_nlags = 16 if np.isnan(_aux) else _aux

        max_ind = ts.size - max_nlags

        ts_slice = ts[:max_ind]
        ts_bin = np.histogram(ts_slice, bins=num_bins)[0]
        ent_ts = scipy.stats.entropy(ts_bin, base=2)

        auto_info = np.zeros(max_nlags, dtype=float)

        for lag in np.arange(1, 1 + max_nlags):
            ts_lagged = ts[lag:max_ind + lag]
            ts_bin_lagged = np.histogram(ts_lagged, bins=num_bins)[0]
            joint_prob = np.histogram2d(ts_slice, ts_lagged, bins=num_bins)[0]

            ent_ts_lagged = scipy.stats.entropy(ts_bin_lagged, base=2)
            ent_joint = scipy.stats.entropy(joint_prob.ravel(), base=2)

            auto_info[lag - 1] = ent_ts + ent_ts_lagged - ent_joint

            if return_dist:
                # Note: this is the same as defining, right from the start,
                # auto_info[lag - 1] = (ent_ts + ent_ts_lagged) / ent_joint
                # However, here all steps are kept to make the code clearer.
                auto_info[lag - 1] = 1 - auto_info[lag - 1] / ent_joint

        return auto_info

    def ft_ami_curvature(
            cls,
            ts: np.ndarray,
    ) -> float:
        """TODO."""


def _test() -> None:
    ts = _get_data.load_data(2)

    ts_period = _period.ts_period(ts)
    ts_trend, ts_season, ts_residuals = _detrend.decompose(ts,
                                                           ts_period=ts_period)
    ts = ts.to_numpy()
    print("TS period:", ts_period)

    res = MFETSInfoTheory.ft_first_crit_pt_ami(ts, return_dist=True)
    print(res)

    res = MFETSInfoTheory.ft_auto_mut_info(ts, return_dist=True)
    print(res)

    res = MFETSInfoTheory.ft_hist_entropy(ts)
    print(res)


if __name__ == "__main__":
    _test()
