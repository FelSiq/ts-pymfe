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
                                 max_lags: int = 64,
                                 return_dist: bool = True,
                                 **kwargs) -> t.Dict[str, np.ndarray]:
        """TODO."""
        precomp_vals = {}

        if "auto_mut_info" in kwargs:
            precomp_vals["auto_mut_info"] = cls.ft_auto_info(
                ts=ts,
                num_bins=num_bins,
                max_lags=max_lags,
                return_dist=return_dist)

        return precomp_vals

    @classmethod
    def ft_hist_entropy(cls,
                        ts: np.ndarray,
                        num_bins: int = 10,
                        normalize: bool = True) -> np.ndarray:
        """TODO."""
        ts_disc = np.digitize(ts, np.linspace(np.min(ts), np.max(ts),
                                              num_bins))

        entropy = scipy.stats.entropy(ts_disc, base=2)

        if normalize:
            entropy /= np.log2(ts_disc.size)

        return entropy

    @classmethod
    def ft_first_crit_pt_ami(
        cls,
        ts: np.ndarray,
        num_bins: int = 64,
        max_lags: int = 64,
        return_dist: bool = True,
        auto_mut_info: t.Optional[np.ndarray] = None
    ) -> t.Union[int, float]:
        """TODO."""
        if auto_mut_info is None:
            auto_mut_info = cls.ft_auto_mut_info(ts=ts,
                                                 num_bins=num_bins,
                                                 max_lags=max_lags,
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
            max_lags: int = 64,
            return_dist: bool = True,
            auto_mut_info: t.Optional[np.ndarray] = None) -> float:
        """TODO."""
        if auto_mut_info is not None:
            return auto_mut_info

        max_ind = ts.size - max_lags

        ts_slice = ts[:max_ind]
        ts_bin = np.histogram(ts_slice, bins=num_bins)[0]
        ent_ts = scipy.stats.entropy(ts_bin, base=2)

        auto_info = np.zeros(max_lags, dtype=float)

        for lag in np.arange(1, 1 + max_lags):
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
    exit(1)

    res = MFETSInfoTheory.ft_hist_entropy(ts)
    print(res)


if __name__ == "__main__":
    _test()
