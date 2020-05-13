import typing as t

import numpy as np
import sklearn.linear_model
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
    def _auto_mut_info(cls,
                       ts: np.ndarray,
                       lag: int,
                       num_bins: int = 64,
                       return_dist: bool = False) -> float:
        """TODO."""
        ts_x = ts[:-lag]
        ts_y = ts[lag:]

        ts_x_bin = np.histogram(ts_x, bins=num_bins)[0]
        ts_y_bin = np.histogram(ts_y, bins=num_bins)[0]
        joint_prob = np.histogram2d(ts_x, ts_y, bins=num_bins, density=True)[0]

        ent_ts_x = scipy.stats.entropy(ts_x_bin, base=2)
        ent_ts_y = scipy.stats.entropy(ts_y_bin, base=2)
        ent_joint = scipy.stats.entropy(joint_prob.ravel(), base=2)

        auto_info = ent_ts_x + ent_ts_y - ent_joint

        if return_dist:
            # Note: this is the same as defining, right from the start,
            # auto_info = (ent_ts_x + ent_ts_y) / ent_joint
            # However, here all steps are kept to make the code clearer.
            auto_info = 1 - auto_info / ent_joint

        return auto_info

    @classmethod
    def ft_hist_entropy(cls,
                        ts: np.ndarray,
                        num_bins: int = 10,
                        normalize: bool = True) -> float:
        """TODO."""
        freqs = np.histogram(ts, density=True)[0]

        entropy = scipy.stats.entropy(freqs, base=2)

        if normalize:
            entropy /= np.log2(freqs.size)

        return entropy

    @classmethod
    def ft_hist_ent_out_diff(cls,
                             ts: np.ndarray,
                             num_bins: int = 10,
                             pcut: float = 0.05,
                             normalize: bool = True) -> float:
        """TODO."""
        if not 0 < pcut < 0.5:
            raise ValueError("'pcut' must be in (0.0, 0.5) (got "
                             "{}).".format(pcut))

        cut_low, cut_high = np.quantile(ts, (pcut, 1 - pcut))
        ts_inliners = ts[np.logical_and(cut_low <= ts, ts <= cut_high)]

        ent_ts = cls.ft_hist_entropy(ts=ts,
                                     num_bins=num_bins,
                                     normalize=normalize)
        ent_ts_inliners = cls.ft_hist_entropy(ts=ts_inliners,
                                              num_bins=num_bins,
                                              normalize=normalize)

        entropy_diff = ent_ts - ent_ts_inliners

        return entropy_diff

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
                                                 lags=max_nlags,
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
            lags: t.Optional[t.Sequence[int]] = None,
            return_dist: bool = True,
            unbiased: bool = True,
            ts_acfs: t.Optional[np.ndarray] = None,
            auto_mut_info: t.Optional[np.ndarray] = None) -> np.ndarray:
        """TODO."""
        if auto_mut_info is not None:
            return auto_mut_info

        if lags is None:
            _aux = autocorr.MFETSAutocorr.ft_first_acf_nonpos(
                ts=ts, unbiased=unbiased, ts_acfs=ts_acfs)
            lags = np.asarray([1 if np.isnan(_aux) else _aux])

        elif np.isscalar(lags):
            lags = np.arange(1, 1 + lags)

        auto_info = np.zeros(lags.size, dtype=float)

        for ind, lag in enumerate(lags):
            auto_info[ind] = cls._auto_mut_info(ts=ts,
                                                lag=lag,
                                                num_bins=num_bins,
                                                return_dist=return_dist)

        return auto_info

    @classmethod
    def ft_ami_curvature(
        cls,
        ts: np.ndarray,
        noise_range: t.Tuple[float, float] = (0, 3),
        noise_inc_num: float = 10,
        random_state: t.Optional[int] = None,
        ts_scaled: t.Optional[np.ndarray] = None,
    ) -> float:
        """TODO."""
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        if random_state is not None:
            np.random.seed(random_state)

        gaussian_noise = np.random.randn(ts_scaled.size)
        noise_std = np.linspace(*noise_range, noise_inc_num)

        ami = np.zeros(noise_inc_num, dtype=float)

        for ind, cur_std in enumerate(noise_std):
            ts_corrupted = ts_scaled + cur_std * gaussian_noise

            ami[ind] = cls.ft_auto_mut_info(ts=ts_corrupted,
                                            unbiased=True,
                                            num_bins=32,
                                            return_dist=False)

        model = sklearn.linear_model.LinearRegression().fit(
            X=noise_std.reshape(-1, 1), y=ami)

        curvature = model.coef_[0]

        return curvature

    @classmethod
    def ft_sample_entropy(cls,
                          ts: np.ndarray,
                          embed_dim: int = 2,
                          factor: float = 0.2,
                          metric: str = "chebyshev",
                          p: t.Union[int, float] = 2,
                          lag: int = 1,
                          ddof: int = 1) -> float:
        """TODO."""
        def neigh_num(dim: int) -> int:
            embed = _embed.embed_ts(ts, dim=dim, lag=lag)
            dist_mat = scipy.spatial.distance.pdist(embed, metric=metric, p=p)
            return np.sum(dist_mat < threshold)

        threshold = factor * np.std(ts, ddof=ddof)

        sample_entropy = -np.log(
            neigh_num(embed_dim + 1) / neigh_num(embed_dim))

        return sample_entropy

    @classmethod
    def ft_control_entropy(cls,
                           ts: np.ndarray,
                           embed_dim: int = 2,
                           factor: float = 0.2,
                           metric: str = "chebyshev",
                           p: t.Union[int, float] = 2,
                           lag: int = 1,
                           ddof: int = 1) -> float:
        """TODO."""
        control_entropy = cls.ft_sample_entropy(
           ts=np.diff(ts),
           embed_dim=embed_dim,
           factor=factor,
           metric=metric,
           p=p,
           lag=lag,
           ddof=ddof)

        return control_entropy


def _test() -> None:
    ts = _get_data.load_data(2)

    ts_period = _period.ts_period(ts)
    ts_trend, ts_season, ts_residuals = _detrend.decompose(ts,
                                                           ts_period=ts_period)
    ts = ts.to_numpy()
    print("TS period:", ts_period)

    res = MFETSInfoTheory.ft_sample_entropy(ts, lag=1)
    print(res)

    res = MFETSInfoTheory.ft_control_entropy(ts, lag=1)
    print(res)

    res = MFETSInfoTheory.ft_ami_curvature(ts, random_state=16)
    print(res)

    res = MFETSInfoTheory.ft_first_crit_pt_ami(ts, return_dist=True)
    print(res)

    res = MFETSInfoTheory.ft_auto_mut_info(ts, return_dist=True)
    print(res)

    res = MFETSInfoTheory.ft_hist_entropy(ts)
    print(res)

    res = MFETSInfoTheory.ft_hist_ent_out_diff(ts)
    print(res)


if __name__ == "__main__":
    _test()
