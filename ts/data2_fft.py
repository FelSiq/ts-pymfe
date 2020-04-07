import typing as t

import numpy as np
import scipy.signal
import scipy.stats

import data1_detrend
import get_data


class MFETSFreqDomain:
    @classmethod
    def _calc_power_spec(cls, ts_residuals: np.ndarray) -> np.ndarray:
        """Calculate the positive side power spectrum of a fourier signal."""
        _, ps = scipy.signal.periodogram(ts_residuals,
                                         detrend=None,
                                         scaling="spectrum",
                                         return_onesided=True)
        return ps

    @classmethod
    def ft_ps_max(
            cls,
            ts_residuals: np.ndarray,
            power_spec: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Maximal power spectrum frequency of the given time-series.
        
        Parameters
        ----------
        ts_residuals : :obj:`np.ndarray`
            Residuals (random noise) of an one-dimensional time-series.

        power_spec : :obj:`np.ndarray`, optional
            Power spectrum of ``ts_residuals``. Used to take advantage of
            precomputations.

        Returns
        -------
        float
            Largest power spectrum frequency of the given time-series.
        """
        if power_spec is None:
            power_spec = cls._calc_power_spec(ts_residuals=ts_residuals)

        return np.max(power_spec)

    @classmethod
    def ft_ps_freqs(
            cls,
            ts_residuals: np.ndarray,
            freq_num: int = 3,
            power_spec: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Largest power spectrum frequencies of the given time-series.
        
        Parameters
        ----------
        ts_residuals : :obj:`np.ndarray`
            Residuals (random noise) of an one-dimensional time-series.

        freq_num : int, optional
            Number of largest frequencies to be returned.

        power_spec : :obj:`np.ndarray`, optional
            Power spectrum of ``ts_residuals``. Used to take advantage of
            precomputations.

        Returns
        -------
        float
            Largest power spectrum frequencies of the given time-series.
        """
        if freq_num <= 0:
            raise ValueError("'freq_num' must be positive.")

        if power_spec is None:
            power_spec = cls._calc_power_spec(ts_residuals=ts_residuals)

        power_spec = np.sort(power_spec)

        return power_spec[-freq_num:]

    @classmethod
    def ft_ps_peaks(
            cls,
            ts_residuals: np.ndarray,
            factor: float = 0.6,
            power_spec: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Number of significative power spectrum frequencies.

        Given a set of power spectrum frequencies `p`, a power spectrum
        frequency `f_i` is considered significative if and only if
        `f_i` >= factor * max(p), where `factor` is a user-defined
        parameter.
        
        Parameters
        ----------
        ts_residuals : :obj:`np.ndarray`
            Residuals (random noise) of an one-dimensional time-series.

        factor : float, optional
            Multiplicative factor of the power spectrum maximum value to
            used to create the threshold to define which power spectrum
            frequencies are significative.

        power_spec : :obj:`np.ndarray`, optional
            Power spectrum of ``ts_residuals``. Used to take advantage of
            precomputations.

        Returns
        -------
        float
            Maximal power spectrum frequency of the given time-series.
        """
        if not 0 < factor < 1:
            raise ValueError("'factor' must be in (0, 1) range.")

        if power_spec is None:
            power_spec = cls._calc_power_spec(ts_residuals=ts_residuals)

        max_ps = cls.ft_ps_max(ts_residuals=ts_residuals,
                               power_spec=power_spec)

        return np.sum(power_spec >= factor * max_ps)

    @classmethod
    def ft_ps_entropy(
            cls,
            ts_residuals: np.ndarray,
            normalize: bool = True,
            power_spec: t.Optional[np.ndarray] = None,
    ) -> float:
        """Spectral entropy.

        The spectral entropy is the entropy if the normalized power
        spectrum of the detrended time-series. Technically, it is the
        entropy of the spectral density, which is the power spectrum
        normalized by the length of the time-series. However, this
        constant factor of normalization does not affect the entropy
        value.

        TODO.
        """
        if power_spec is None:
            power_spec = cls._calc_power_spec(ts_residuals=ts_residuals)

        # Note: no need to calculate the power spectrum density 'd':
        # d = power_spec / ts_residuals.size
        # since a constant factor does not affect the entropy value.
        ps_ent = scipy.stats.entropy(power_spec / np.sum(power_spec))

        if normalize:
            ps_ent /= np.log2(ts_residuals.size)

        return ps_ent


def _test() -> None:
    ts = get_data.load_data(3)
    ts_trend, ts_season, ts_residuals = data1_detrend.decompose(ts, period=12)

    res = MFETSFreqDomain._calc_power_spec(ts_residuals)
    print(res)

    res = MFETSFreqDomain.ft_ps_max(ts_residuals)
    print(res)

    res = MFETSFreqDomain.ft_ps_freqs(ts_residuals)
    print(res)

    res = MFETSFreqDomain.ft_ps_peaks(ts_residuals)
    print(res)

    res = MFETSFreqDomain.ft_ps_entropy(ts_residuals)
    print(res)


if __name__ == "__main__":
    _test()
