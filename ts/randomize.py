import typing as t
import collections

import numpy as np
import scipy.signal
import scipy.stats
import statsmodels.tsa

import _utils
import _detrend
import _period
import _get_data


class MFETSRandomize:
    @classmethod
    def precompute_randomize_stats(cls,
                                   ts: np.ndarray,
                                   strategy: str = "dist-dynamic",
                                   prop_rep: t.Union[int, float] = 2,
                                   prop_interval: float = 0.1,
                                   ts_scaled: t.Optional[np.ndarray] = None,
                                   random_state: t.Optional[int] = None,
                                   **kwargs) -> t.Dict[str, np.ndarray]:
        """TODO."""
        precomp_vals = {}  # type: t.Dict[str, np.ndarray]

        stats = collections.OrderedDict((
            ("mean", np.mean),
            ("std", np.std),
            ("acf", lambda arr: statsmodels.tsa.stattools.acf(
                arr, nlags=1, fft=True)[1]),
        ))

        stat_names = list(map("rand_stat_{}".format, stats.keys()))

        if not set(stat_names).issubset(kwargs):
            stat_vals = cls._randomize_stat(ts=ts,
                                            func_stats=stats.values(),
                                            strategy=strategy,
                                            prop_rep=prop_rep,
                                            prop_interval=prop_interval,
                                            random_state=random_state,
                                            ts_scaled=ts_scaled)

            precomp_vals.update(
                {name: val
                 for name, val in zip(stat_names, stat_vals)})

        return precomp_vals

    @classmethod
    def _randomize_stat(cls,
                        ts: np.ndarray,
                        func_stats: t.Sequence[t.Callable[[np.ndarray],
                                                          float]],
                        strategy: str = "dist-dynamic",
                        prop_rep: t.Union[int, float] = 2,
                        prop_interval: float = 0.1,
                        ts_scaled: t.Optional[np.ndarray] = None,
                        random_state: t.Optional[int] = None) -> np.ndarray:
        """TODO."""
        if prop_rep <= 0:
            raise ValueError(
                "'prop_rep' must be positive (got {}).".format(prop_rep))

        if prop_interval <= 0:
            raise ValueError(
                "'prop_interval' must be positive (got {}).".format(
                    prop_interval))

        VALID_STRATEGY = ("dist-static", "dist-dynamic", "permute")

        if strategy not in VALID_STRATEGY:
            raise ValueError("'strategy' not in {} (got '{}')."
                             "".format(VALID_STRATEGY, strategy))

        if not hasattr(func_stats, "__len__"):
            func_stats = [func_stats]

        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        rep_it = int(np.ceil(prop_interval * ts_scaled.size))

        # Note: adding (num_it % rep_it) to avoid lose computation of
        # the remaining iterations that do not produce a statistic.
        num_it = int(np.ceil(prop_rep * ts_scaled.size))
        num_it += (num_it % rep_it)

        res = np.zeros((len(func_stats), 1 + num_it // rep_it))
        ts_rnd = np.copy(ts_scaled)
        ts_src = ts_scaled if strategy == "dist-static" else ts_rnd
        swap = strategy == "permute"
        stat_ind = 0

        if random_state is not None:
            np.random.seed(random_state)

        inds_rnd = np.random.randint(ts_scaled.size, size=(num_it, 2))

        for it, (ind_a, ind_b) in enumerate(inds_rnd):
            if swap:
                ts_rnd[ind_a], ts_src[ind_b] = ts_src[ind_b], ts_rnd[ind_a]

            else:
                ts_rnd[ind_a] = ts_src[ind_b]

            if it % rep_it == 0:
                for ind_f, func in enumerate(func_stats):
                    res[ind_f, stat_ind] = func(ts_rnd)

                stat_ind += 1

        return res if len(func_stats) > 1 else res.ravel()

    @classmethod
    def ft_randomize_mean(
            cls,
            ts: np.ndarray,
            strategy: str = "dist-dynamic",
            prop_rep: t.Union[int, float] = 2,
            prop_interval: float = 0.1,
            ts_scaled: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None,
            rand_stat_mean: t.Optional[np.ndarray] = None) -> np.ndarray:
        """TODO."""
        if rand_stat_mean is not None:
            return rand_stat_mean

        res = cls._randomize_stat(ts=ts,
                                  func_stats=np.mean,
                                  strategy=strategy,
                                  prop_rep=prop_rep,
                                  prop_interval=prop_interval,
                                  random_state=random_state,
                                  ts_scaled=ts_scaled)

        return res

    @classmethod
    def ft_randomize_sd(
            cls,
            ts: np.ndarray,
            strategy: str = "dist-dynamic",
            prop_rep: t.Union[int, float] = 2,
            prop_interval: float = 0.1,
            ts_scaled: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None,
            rand_stat_std: t.Optional[np.ndarray] = None) -> np.ndarray:
        """TODO."""
        if rand_stat_std is not None:
            return rand_stat_std

        res = cls._randomize_stat(ts=ts,
                                  func_stats=np.std,
                                  strategy=strategy,
                                  prop_rep=prop_rep,
                                  prop_interval=prop_interval,
                                  random_state=random_state,
                                  ts_scaled=ts_scaled)

        return res

    @classmethod
    def ft_randomize_acf(
            cls,
            ts: np.ndarray,
            strategy: str = "dist-dynamic",
            prop_rep: t.Union[int, float] = 2,
            prop_interval: float = 0.1,
            ts_scaled: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None,
            rand_stat_acf: t.Optional[np.ndarray] = None) -> np.ndarray:
        """TODO."""
        if rand_stat_acf is not None:
            return rand_stat_acf

        func_acf = lambda arr: statsmodels.tsa.stattools.acf(
            arr, nlags=1, fft=True)[1]

        res = cls._randomize_stat(ts=ts,
                                  func_stats=func_acf,
                                  strategy=strategy,
                                  prop_rep=prop_rep,
                                  prop_interval=prop_interval,
                                  random_state=random_state,
                                  ts_scaled=ts_scaled)

        return res

    @classmethod
    def ft_rand_samp_std(
            cls,
            ts: np.ndarray,
            num_samples: int = 64,
            sample_size_frac: float = 0.1,
            ddof: int = 1,
            random_state: t.Optional[int] = None,
            ts_scaled: t.Optional[np.ndarray] = None) -> np.ndarray:
        """TODO."""
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        sample_std = _utils.apply_on_samples(ts=ts_scaled,
                                             func=np.std,
                                             num_samples=num_samples,
                                             sample_size_frac=sample_size_frac,
                                             random_state=random_state,
                                             ddof=ddof)

        return sample_std


def _test() -> None:
    ts = _get_data.load_data(3)
    ts_period = _period.ts_period(ts=ts)
    ts_trend, ts_season, ts_residuals = _detrend.decompose(ts,
                                                           ts_period=ts_period)

    res = MFETSRandomize.precompute_randomize_stats(ts, random_state=16)
    print(res)

    res = MFETSRandomize.ft_randomize_mean(ts, random_state=16)
    print(res)

    res = MFETSRandomize.ft_randomize_sd(ts, random_state=16)
    print(res)

    res = MFETSRandomize.ft_randomize_acf(ts, random_state=16)
    print(res)

    res = MFETSRandomize.ft_rand_samp_std(ts, random_state=16)
    print(res)


if __name__ == "__main__":
    _test()
