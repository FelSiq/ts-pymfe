import typing as t

import sklearn.preprocessing
import numpy as np
import scipy.spatial
import scipy.odr
import pandas as pd

import autocorr
import _detrend
import _embed
import _period
import _utils
import _get_data


class MFETSGeneral:
    @staticmethod
    def _calc_season_mode_ind(ts_season: np.ndarray, ts_period: int,
                              indfunc: t.Callable[[np.ndarray], float]) -> int:
        """Calculate a mode index based on the time-series seasonality.

        Used by both ``ft_trough_frac`` and ``ft_peak_frac`` to calculate,
        respectively, the mode of the argmin and argmax for all seasons.
        """
        inds = np.arange(ts_period)

        inds = np.array([
            indfunc(ts_season[i * ts_period + inds])
            for i in np.arange(1, ts_season.size // ts_period)
        ],
                        dtype=int)

        mode_inds, _ = scipy.stats.mode(inds)
        return mode_inds[0] + 1

    @classmethod
    def ft_length(cls, ts: np.ndarray) -> int:
        """Length of the time-series.

        Parameters
        ----------
        ts: :obj:`np.ndarray`
            One-dimensional time-series values.

        Returns
        -------
        int
            Length of the time-seties.

        References
        ----------
        TODO.
        """
        return ts.size

    @classmethod
    def ft_period(cls,
                  ts: np.ndarray,
                  ts_period: t.Optional[int] = None) -> int:
        """TODO."""
        if ts_period is not None:
            return ts_period

        return _period.ts_period(ts)

    @classmethod
    def ft_turning_points(cls, ts: np.ndarray) -> np.ndarray:
        """Turning points in the time-series.

        A turning point is a time-series point `p_{i}` which both neighbor
        values, p_{i-1} and p_{i+1}, are either lower (p_{i} > p_{i+1} and
        p_{i} > p_{i-1}) or higher (p_{i} < p_{i+1} and p_{i} < p_{i-1}).

        Parameters
        ----------
        ts: :obj:`np.ndarray`
            One-dimensional time-series values.

        Returns
        -------
        :obj:`np.ndarray`
            Binary array marking where is the turning points in the
            time-series.

        References
        ----------
        TODO.
        """
        return _utils.find_crit_pt(ts, type_="non-plateau")

    @classmethod
    def ft_turning_points_trend(cls, ts_trend: np.ndarray) -> np.ndarray:
        """Turning points in the time-series trend.

        A turning point is a time-series point `p_{i}` which both neighbor
        values, p_{i-1} and p_{i+1}, are either lower (p_{i} > p_{i+1} and
        p_{i} > p_{i-1}) or higher (p_{i} < p_{i+1} and p_{i} < p_{i-1}).

        Parameters
        ----------
        ts_trend: :obj:`np.ndarray`
            One-dimensional time-series trend values.

        Returns
        -------
        :obj:`np.ndarray`
            Binary array marking where is the turning points in the
            time-series trend.

        References
        ----------
        TODO.
        """
        return cls.ft_turning_points(ts=ts_trend)

    @classmethod
    def ft_step_changes(cls, ts: np.ndarray, ddof: int = 1) -> np.ndarray:
        """Step change points in the time-series.

        Let p_{t_{a}}^{t_{b}} be the subsequence of observations from the
        timestep t_{a} and t_{b}, both inclusive. A point `p_i` is a
        step change if and only if:

        abs(p_{i} - mean(p_{1}^{i-1})) > 2 * std(p_{1}^{i-1})

        Parameters
        ----------
        ts: :obj:`np.ndarray`
            One-dimensional time-series values.

        ddof : float, optional
            Degrees of freedom for standard deviation.

        Returns
        -------
        :obj:`np.ndarray`
            Binary array marking where is the step change points in the
            time-series.

        References
        ----------
        TODO.
        """
        ts_cmeans = np.cumsum(ts) / np.arange(1, ts.size + 1)

        ts_mean_abs_div = np.abs(ts[1:] - ts_cmeans[:-1])

        step_changes = np.array([
            int(ts_mean_abs_div[i - 1] > 2 * np.std(ts[:i], ddof=ddof))
            for i in np.arange(1 + ddof, ts.size)
        ],
                                dtype=int)

        return step_changes

    @classmethod
    def ft_step_changes_trend(cls,
                              ts_trend: np.ndarray,
                              ddof: int = 1) -> np.ndarray:
        """Step change points in the time-series trend.

        Let p_{t_{a}}^{t_{b}} be the subsequence of observations from the
        timestep t_{a} and t_{b}, both inclusive. A point `p_i` is a
        step change if and only if:

        abs(p_{i} - mean(p_{1}^{i-1})) > 2 * std(p_{1}^{i-1})

        Parameters
        ----------
        ts_trend: :obj:`np.ndarray`
            One-dimensional time-series trend values.

        ddof : float, optional
            Degrees of freedom for standard deviation.

        Returns
        -------
        :obj:`np.ndarray`
            Binary array marking where is the step change points in the
            time-series trend.

        References
        ----------
        TODO.
        """
        step_changes = cls.ft_step_changes(ts=ts_trend, ddof=ddof)
        return step_changes

    @classmethod
    def ft_pred(cls,
                ts: np.ndarray,
                embed_dim: int = 2,
                param_1: t.Union[int, float] = 3,
                param_2: t.Union[int, float] = 4,
                metric: str = "minkowski",
                p: t.Union[int, float] = 2,
                ddof: int = 1,
                ts_scaled: t.Optional[np.ndarray] = None) -> float:
        """https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4736930/"""
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)
        ts_embed = _embed.embed_ts(ts_scaled, dim=embed_dim)

        dist_mat = scipy.spatial.distance.pdist(ts_embed, metric=metric, p=p)

        dist_mean = np.mean(dist_mat)
        dist_std = np.std(dist_mat, ddof=ddof)

        dist_mat = scipy.spatial.distance.squareform(dist_mat)

        var_sets = np.zeros(param_2, dtype=float)

        for i in np.arange(param_2):
            threshold = max(
                0.0,
                dist_mean + param_1 * dist_std * (i * 2 / (param_2 - 1) - 1))

            neighbors = (dist_mat <= threshold).astype(int)
            neighbors[np.diag_indices_from(neighbors)] = 0.0

            for neigh_inds in neighbors:
                if np.sum(neigh_inds) > ddof:
                    var_sets[i] += np.var(ts_embed[neigh_inds, :], ddof=ddof)

        var_sets /= ts_embed.shape[0]

        return 1.0 / (1.0 + var_sets)

    @classmethod
    def ft_frac_cp(cls,
                   ts: np.ndarray,
                   normalize: bool = True) -> t.Union[int, float]:
        """TODO."""
        higher_med = ts <= np.median(ts)
        num_cp = np.sum(np.logical_xor(higher_med[1:], higher_med[:-1]))

        if normalize:
            num_cp /= ts.size - 1

        return num_cp

    @classmethod
    def ft_binmean(cls, ts: np.ndarray) -> np.ndarray:
        """TODO."""
        return (ts >= np.mean(ts)).astype(int)

    @classmethod
    def ft_fs_len(cls, ts: np.ndarray, num_bins: int = 10) -> np.ndarray:
        """TODO."""
        ts_disc = np.digitize(ts, np.linspace(np.min(ts), np.max(ts),
                                              num_bins))
        i = 1
        counter = 1
        fs_len = []  # type: t.List[int]

        while i < ts.size:
            if not np.isclose(ts_disc[i], ts_disc[i - 1]):
                fs_len.append(counter)
                counter = 1

            else:
                counter += 1

            i += 1

        return np.asarray(fs_len, dtype=float)

    @classmethod
    def ft_peak_frac(cls,
                     ts_season: np.ndarray,
                     ts_period: int,
                     normalize: bool = True) -> t.Union[int, float]:
        """TODO."""
        if ts_period <= 1:
            return np.nan

        ind_peak = cls._calc_season_mode_ind(ts_season=ts_season,
                                             ts_period=ts_period,
                                             indfunc=np.argmax)

        if normalize:
            ind_peak /= ts_period  # type: ignore

        return ind_peak

    @classmethod
    def ft_trough_frac(cls,
                       ts_season: np.ndarray,
                       ts_period: int,
                       normalize: bool = True) -> t.Union[int, float]:
        """TODO."""
        if ts_period <= 1:
            return np.nan

        ind_trough = cls._calc_season_mode_ind(ts_season=ts_season,
                                               ts_period=ts_period,
                                               indfunc=np.argmin)

        if normalize:
            ind_trough /= ts_period  # type: ignore

        return ind_trough

    @classmethod
    def _ts_walker(
            cls,
            ts: np.ndarray,
            step_size: float = 0.1,
            start_point: t.Optional[t.Union[int, float]] = None) -> np.ndarray:
        """TODO."""
        if start_point is None:
            # Note: actually, it is used mean(ts) as starting point. However,
            # we are using ts_scaled and, therefore, mean(ts_scaled) = 0.
            start_point = 0.0

        walker_path = np.zeros(ts.size, dtype=float)
        walker_path[0] = start_point

        for i in np.arange(1, ts.size):
            diff = ts[i - 1] - walker_path[i - 1]
            walker_path[i] = walker_path[i - 1] + step_size * diff

        return walker_path

    @classmethod
    def ft_walker_path(cls,
                       ts: np.ndarray,
                       step_size: float = 0.1,
                       start_point: t.Optional[t.Union[int, float]] = None,
                       relative_dist: bool = True,
                       walker_path: t.Optional[np.ndarray] = None,
                       ts_scaled: t.Optional[np.ndarray] = None) -> np.ndarray:
        """TODO."""
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        if walker_path is None:
            walker_path = cls._ts_walker(ts=ts_scaled,
                                         step_size=step_size,
                                         start_point=start_point)

        if relative_dist:
            return np.abs(walker_path - ts_scaled)

        return walker_path

    @classmethod
    def ft_walker_cross_frac(
            cls,
            ts: np.ndarray,
            step_size: float = 0.1,
            start_point: t.Optional[t.Union[int, float]] = None,
            normalize: bool = True,
            walker_path: t.Optional[np.ndarray] = None,
            ts_scaled: t.Optional[np.ndarray] = None) -> t.Union[int, float]:
        """TODO."""
        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        if walker_path is None:
            walker_path = cls._ts_walker(ts=ts_scaled,
                                         step_size=step_size,
                                         start_point=start_point)

        cross_num = np.sum((walker_path[:-1] - ts_scaled[:-1]) *
                           (walker_path[1:] - ts_scaled[1:]) < 0)

        if normalize:
            cross_num /= walker_path.size - 1

        return cross_num

    @classmethod
    def ft_moving_threshold(cls,
                            ts: np.ndarray,
                            rate_absorption: float = 0.1,
                            rate_decay: float = 0.1,
                            ts_scaled: t.Optional[np.ndarray] = None,
                            relative: bool = False) -> np.ndarray:
        """TODO."""
        if not 0 < rate_decay < 1:
            raise ValueError("'rate_decay' must be in (0, 1) (got {})."
                             "".format(rate_decay))

        if not 0 < rate_absorption < 1:
            raise ValueError("'rate_absorption' must be in (0, 1) (got"
                             " {}).".format(rate_absorption))

        ts_scaled = np.abs(_utils.standardize_ts(ts=ts, ts_scaled=ts_scaled))

        # Note: threshold[0] = std(ts_scaled) = 1.0.
        threshold = np.ones(1 + ts.size, dtype=float)

        _ra = 1 + rate_absorption
        _rd = 1 - rate_decay

        for ind in np.arange(ts_scaled.size):
            if ts_scaled[ind] > threshold[ind]:
                # Absorb from the time series (absolute) values
                threshold[1 + ind] = _ra * ts_scaled[ind]
            else:
                # Decay the threshold
                threshold[1 + ind] = _rd * threshold[ind]

        if relative:
            # Note: ignore the first initial threshold
            return threshold[1:] - ts_scaled

        return threshold

    @classmethod
    def ft_embed_in_sphere(
            cls,
            ts: np.ndarray,
            radius: t.Union[int, float] = 1,
            embed_dim: int = 2,
            lag: t.Optional[int] = None,
            normalize: bool = True,
            ts_acfs: t.Optional[np.ndarray] = None,
            max_nlags: t.Optional[int] = None,
            unbiased: bool = True,
            ts_scaled: t.Optional[np.ndarray] = None) -> t.Union[int, float]:
        """TODO."""
        if radius <= 0:
            raise ValueError(
                "'radius' must be positive (got {}).".format(radius))

        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        if lag is None:
            lag = autocorr.MFETSAutocorr.ft_first_acf_nonpos(
                ts=ts_scaled,
                ts_acfs=ts_acfs,
                max_nlags=max_nlags,
                unbiased=unbiased)

        # Note: embed is given by x(t) = [x(t-1), x(t-2), ..., x(t-m+1)]^T
        embed = _embed.embed_ts(ts_scaled, dim=embed_dim, lag=lag)

        # Note: here we supposed that every embed forms a zero-centered
        # hypersphere using the regular hypersphere equation:
        # sqrt(x_{i}^2 + x^{i-1}^2 + ... + x^{i-m+1}) = Radius
        embed_radius = np.linalg.norm(embed, ord=2, axis=1)

        # Note: we can check if every embed is in the same zero-centered
        # hypersphere because all hypersphere embeds are also zero-centered.
        in_hypersphere = np.sum(embed_radius <= radius)

        if normalize:
            in_hypersphere /= embed_radius.size

        return in_hypersphere

    @classmethod
    def ft_force_potential(
            cls,
            ts: np.ndarray,
            potential: str = "sine",
            params: t.Optional[t.Tuple[float, float, float]] = None,
            start_point: t.Optional[t.Tuple[float, float]] = None,
            ts_scaled: t.Optional[np.ndarray] = None) -> np.ndarray:
        """TODO."""
        DEF_PARAM = dict(
            sine=((1, 1, 0.1), lambda x: np.sin(x / alpha) / alpha),
            dblwell=((2, 0.1, 0.1), lambda x: alpha**2 * x - x**3),
        )

        if potential not in DEF_PARAM:
            raise ValueError("'potential' must be in {} (got '{}')."
                             "".format(DEF_PARAM.keys(), potential))

        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        alpha, fric, dt = DEF_PARAM[potential][0] if params is None else params
        f_force = DEF_PARAM[potential][1]

        pos, vel = np.zeros((2, ts_scaled.size), dtype=float)

        # Note: it is actually used (mean(ts), 0.0) as default start
        # point, but mean(ts_scaled) = 0.
        pos[0], vel[0] = (0.0, 0.0) if start_point is None else start_point

        for t_prev in np.arange(ts_scaled.size - 1):
            aux = f_force(pos[t_prev]) + ts_scaled[t_prev] - fric * vel[t_prev]
            pos[t_prev + 1] = pos[t_prev] + dt * vel[t_prev] + dt**2 * aux
            vel[t_prev + 1] = vel[t_prev] + dt * aux

        if not np.isfinite(pos[-1]):
            raise ValueError("Potential trajectory diverged.")

        return pos

    @classmethod
    def ft_stick_angles(
            cls,
            ts: np.ndarray,
            ts_scaled: t.Optional[np.ndarray] = None) -> np.ndarray:
        """TODO."""
        def calc_angles(inds: np.ndarray) -> np.ndarray:
            """TODO."""
            return np.arctan(np.diff(ts_scaled[inds]) / np.diff(inds))

        ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

        aux = ts_scaled >= 0
        ts_ang_pos = calc_angles(inds=np.flatnonzero(aux))
        ts_ang_neg = calc_angles(inds=np.flatnonzero(~aux))

        angles = np.hstack((ts_ang_pos, ts_ang_neg))

        return angles


def _test() -> None:
    import matplotlib.pyplot as plt
    ts = _get_data.load_data(3)

    ts_period = _period.ts_period(ts)
    ts_trend, ts_season, ts_residuals = _detrend.decompose(ts,
                                                           ts_period=ts_period)
    ts = ts.to_numpy()
    print("TS period:", ts_period)

    res = MFETSGeneral.ft_stick_angles(ts)
    print(res)
    exit(1)

    import matplotlib.pyplot as plt
    res_a = MFETSGeneral.ft_force_potential(ts, potential="sine")
    res_b = MFETSGeneral.ft_force_potential(ts, potential="dblwell")
    time = np.arange(ts.size)
    plt.plot(time, _utils.standardize_ts(ts), label="series")
    plt.plot(time, res_a, color="red", label="sine")
    plt.plot(time, res_b, color="purple", label="dbwell")
    plt.legend()
    plt.show()
    exit(1)

    res = MFETSGeneral.ft_walker_cross_frac(ts)
    print(res)

    res = MFETSGeneral.ft_walker_path(ts)
    print(res)
    exit(1)

    res = MFETSGeneral.ft_embed_in_sphere(ts)
    print(res)
    exit(1)

    res = MFETSGeneral.ft_pred(ts)
    print(res)
    exit(1)

    res = MFETSGeneral.ft_moving_threshold(ts, relative=True)
    print(res)

    res = MFETSGeneral.ft_turning_points(ts)
    print(np.mean(res))

    res = MFETSGeneral.ft_step_changes(ts)
    print(np.mean(res))

    res = MFETSGeneral.ft_turning_points_trend(ts_trend)
    print(np.mean(res))

    res = MFETSGeneral.ft_step_changes_trend(ts_trend)
    print(np.mean(res))

    res = MFETSGeneral.ft_length(ts)
    print(res)

    res = MFETSGeneral.ft_frac_cp(ts)
    print(res)

    res = MFETSGeneral.ft_fs_len(ts)
    print(res)

    res = MFETSGeneral.ft_binmean(ts)
    print(res)

    res = MFETSGeneral.ft_period(ts)
    print(res)

    res = MFETSGeneral.ft_peak_frac(ts_season, ts_period=ts_period)
    print(res)

    res = MFETSGeneral.ft_trough_frac(ts_season, ts_period=ts_period)
    print(res)


if __name__ == "__main__":
    _test()
