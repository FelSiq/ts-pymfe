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
    def precompute_ts_ami(cls,
                          ts: np.ndarray,
                          num_bins: int = 64,
                          max_nlags: t.Optional[int] = None,
                          return_dist: bool = False,
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

        if "ts_ami" in kwargs:
            precomp_vals["ts_ami"] = cls.ft_ami(ts=ts,
                                                num_bins=num_bins,
                                                max_nlags=max_nlags,
                                                return_dist=return_dist,
                                                unbiased=unbiased,
                                                ts_acfs=ts_acfs)

        return precomp_vals

    @classmethod
    def _ts_ami(cls,
                ts: np.ndarray,
                lag: int,
                num_bins: int = 64,
                return_dist: bool = False) -> float:
        """TODO."""
        ts_x = ts[:-lag]
        ts_y = ts[lag:]

        ts_x_prob = np.histogram(ts_x, bins=num_bins, density=True)[0]
        ts_y_prob = np.histogram(ts_y, bins=num_bins, density=True)[0]
        joint_prob = np.histogram2d(ts_x, ts_y, bins=num_bins, density=True)[0]

        ent_ts_x = scipy.stats.entropy(ts_x_prob, base=2)
        ent_ts_y = scipy.stats.entropy(ts_y_prob, base=2)
        ent_joint = scipy.stats.entropy(joint_prob.ravel(), base=2)

        ami = ent_ts_x + ent_ts_y - ent_joint

        if return_dist:
            # Note: this is the same as defining, right from the start,
            # ami = (ent_ts_x + ent_ts_y) / ent_joint
            # However, here all steps are kept to make the code clearer.
            ami = 1 - ami / ent_joint

        return ami

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
    def ft_ami(cls,
               ts: np.ndarray,
               num_bins: int = 64,
               lags: t.Optional[t.Sequence[int]] = None,
               return_dist: bool = False,
               max_nlags: t.Optional[int] = None,
               ts_acfs: t.Optional[np.ndarray] = None,
               ts_ami: t.Optional[np.ndarray] = None) -> np.ndarray:
        """TODO."""
        if ts_ami is not None:
            return ts_ami

        if lags is None:
            lags = _embed.embed_lag(ts=ts,
                                    lag="acf",
                                    max_nlags=max_nlags,
                                    ts_acfs=ts_acfs)
            lags = np.asarray([lags])

        elif np.isscalar(lags):
            lags = np.arange(1, 1 + lags)

        ami = np.zeros(lags.size, dtype=float)

        for ind, lag in enumerate(lags):
            ami[ind] = cls._ts_ami(ts=ts,
                                   lag=lag,
                                   num_bins=num_bins,
                                   return_dist=return_dist)

        return ami

    @classmethod
    def ft_ami_first_critpt(
            cls,
            ts: np.ndarray,
            num_bins: int = 64,
            max_nlags: t.Optional[int] = None,
            return_dist: bool = False,
            ts_ami: t.Optional[np.ndarray] = None) -> t.Union[int, float]:
        """TODO."""
        if max_nlags is None:
            max_nlags = max(64, ts.size // 2)

        if ts_ami is None:
            ts_ami = cls.ft_ami(ts=ts,
                                num_bins=num_bins,
                                lags=max_nlags,
                                return_dist=return_dist)

        # Note: if 'return_dist=True', return the first local maximum.
        # If otherwise, return the first local minimum.
        type_ = "max" if return_dist else "min"

        crit_point = _utils.find_crit_pt(arr=ts_ami, type_=type_)

        try:
            return np.flatnonzero(crit_point)[0] + 1

        except IndexError:
            return np.nan

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

            ami[ind] = cls.ft_ami(ts=ts_corrupted,
                                  num_bins=32,
                                  return_dist=False)

        model = sklearn.linear_model.LinearRegression().fit(
            X=noise_std.reshape(-1, 1), y=ami)

        curvature = model.coef_[0]

        return curvature

    @classmethod
    def ft_approx_entropy(cls,
                          ts: np.ndarray,
                          embed_dim: int = 2,
                          embed_lag: int = 1,
                          threshold: float = 0.2,
                          metric: str = "chebyshev",
                          p: t.Union[int, float] = 2,
                          ts_scaled: t.Optional[np.ndarray] = None) -> float:
        """TODO."""
        def neigh_num(dim: int) -> int:
            embed = _embed.embed_ts(ts_scaled, dim=dim, lag=embed_lag)
            dist_mat = scipy.spatial.distance.cdist(embed,
                                                    embed,
                                                    metric=metric,
                                                    p=p)
            return np.mean(np.log(np.mean(dist_mat < threshold, axis=1)))

        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        approx_entropy = neigh_num(embed_dim) - neigh_num(embed_dim + 1)

        return approx_entropy

    @classmethod
    def ft_sample_entropy(cls,
                          ts: np.ndarray,
                          embed_dim: int = 2,
                          embed_lag: int = 1,
                          threshold: float = 0.2,
                          metric: str = "chebyshev",
                          p: t.Union[int, float] = 2,
                          ts_scaled: t.Optional[np.ndarray] = None) -> float:
        """TODO."""
        def log_neigh_num(dim: int) -> int:
            embed = _embed.embed_ts(ts_scaled, dim=dim, lag=embed_lag)
            dist_mat = scipy.spatial.distance.pdist(embed, metric=metric, p=p)
            return np.log(np.sum(dist_mat < threshold))

        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        sample_entropy = log_neigh_num(embed_dim) - log_neigh_num(embed_dim +
                                                                  1)

        return sample_entropy

    @classmethod
    def ft_control_entropy(cls,
                           ts: np.ndarray,
                           embed_dim: int = 2,
                           threshold: float = 0.2,
                           metric: str = "chebyshev",
                           p: t.Union[int, float] = 2,
                           embed_lag: int = 1,
                           ts_scaled: t.Optional[np.ndarray] = None) -> float:
        """TODO."""
        control_entropy = cls.ft_sample_entropy(ts=np.diff(ts),
                                                embed_dim=embed_dim,
                                                embed_lag=embed_lag,
                                                threshold=threshold,
                                                metric=metric,
                                                p=p,
                                                ts_scaled=ts_scaled)

        return control_entropy

    @classmethod
    def ft_surprise(cls,
                    ts: np.ndarray,
                    num_bins: int = 10,
                    memory_size: t.Union[float, int] = 0.1,
                    num_it: int = 128,
                    method: str = "distribution",
                    epsilon: float = 1.0e-8,
                    random_state: t.Optional[int] = None) -> np.ndarray:
        """TODO."""
        VALID_METHODS = ("distribution", "1-transition")

        if method not in VALID_METHODS:
            raise ValueError("'method' must in {} (got '{}')."
                             "".format(VALID_METHODS, method))

        if memory_size <= 0:
            raise ValueError("'memory_size' must be positive (got "
                             "{}).".format(memory_size))

        if num_it <= 0:
            raise ValueError("'num_it' must be positive (got {})."
                             "".format(num_it))

        def get_reference_inds(inds_num: int) -> np.ndarray:
            """Get min(ts.size - memory_size, inds_num) random reference indices."""
            if ts.size - memory_size > num_it:
                if random_state is not None:
                    np.random.seed(random_state)

                # Note: we can't use indices before the first 'memory_size'
                # indices as reference, since we need a 'memory' of size
                # 'memory_size' (i.e., we need at least 'memory_size' past
                # indices). Therefore, we always skip they.
                return memory_size + np.random.choice(
                    ts.size - memory_size, size=inds_num, replace=False)

            # Note: the requested number of indices is not smaller than
            # the number of available indices. Therefore, just return
            # all available indices.
            return np.arange(memory_size, ts.size)

        if 0 < memory_size < 1:
            memory_size = int(np.ceil(ts.size * memory_size))

        if method == "distribution":

            def prob_func(ref_ind: int, ts_bin: np.ndarray):
                return np.mean(ts_bin[ref_ind -
                                      memory_size:ref_ind] == ts_bin[ref_ind])

        else:

            def prob_func(ref_ind: int, ts_bin: np.ndarray):
                mem_data = ts_bin[ref_ind - memory_size:ref_ind]
                prev_val = mem_data[-1]
                prev_val_inds = np.flatnonzero(mem_data[:-1] == prev_val)
                return np.mean(mem_data[prev_val_inds + 1] == ts_bin[ref_ind])

        # Note: not necessarily we need to to this on the differenced
        # time-series. This should be optional in the future.
        ts_diff = np.diff(ts)

        # Note: discretize time-series using an equiprobable histogram
        # (i.e. all bins have approximately the same number of instances).
        ts_bin = _utils.discretize(ts=ts,
                                   num_bins=num_bins,
                                   strategy="equiprobable")

        probs = np.zeros(num_it, dtype=float)

        for ind, ref_ind in enumerate(get_reference_inds(inds_num=num_it)):
            probs[ind] = prob_func(ref_ind=ref_ind, ts_bin=ts_bin)

        probs[probs < epsilon] = 1.0
        surprise = -np.log(probs)

        return surprise

    def ft_lz_complexity(ts: np.ndarray,
                         num_bins: int = 10,
                         normalize: bool = True) -> float:
        """TODO."""
        ts_bin = tuple(
            _utils.discretize(ts=ts,
                              num_bins=num_bins,
                              strategy="equal-width",
                              dtype=int))

        ind_start, ind_end = 0, 1
        substrings = set()
        _len = len(ts_bin)

        while ind_end <= _len:
            substring = ts_bin[ind_start:ind_end]

            if substring not in substrings:
                substrings.add(substring)
                ind_start = ind_end

            ind_end += 1

        lz_comp = len(substrings)

        if normalize:
            lz_comp *= np.log(_len) / (_len * np.log(num_bins))

        return lz_comp


def _test() -> None:
    ts = _get_data.load_data(2)

    ts_period = _period.ts_period(ts)
    ts_trend, ts_season, ts_residuals = _detrend.decompose(ts,
                                                           ts_period=ts_period)
    ts = ts.to_numpy()
    print("TS period:", ts_period)

    res = MFETSInfoTheory.ft_lz_complexity(ts)
    print(res)

    res = MFETSInfoTheory.ft_sample_entropy(ts)
    print(res)

    res = MFETSInfoTheory.ft_approx_entropy(ts)
    print(res)

    res = MFETSInfoTheory.ft_control_entropy(ts)
    print(res)

    res = MFETSInfoTheory.ft_surprise(ts,
                                      random_state=16,
                                      method="1-transition")
    print(res)

    res = MFETSInfoTheory.ft_ami_curvature(ts, random_state=16)
    print(res)

    res = MFETSInfoTheory.ft_ami_first_critpt(ts, return_dist=True)
    print(res)

    res = MFETSInfoTheory.ft_ami(ts, return_dist=True)
    print(res)

    res = MFETSInfoTheory.ft_hist_entropy(ts)
    print(res)

    res = MFETSInfoTheory.ft_hist_ent_out_diff(ts)
    print(res)


if __name__ == "__main__":
    _test()
