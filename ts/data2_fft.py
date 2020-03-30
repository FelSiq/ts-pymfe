import typing as t

import numpy as np

import data1_detrend
import get_data


class MFETSFreqDomain:
    @classmethod
    def _calc_power_spec(cls,
                             ts_detrended: np.ndarray,
                             ft_sig: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate the positive side power spectrum of a fourier signal."""
        if ft_sig is None:
            ft_sig = np.fft.rfft(ts_detrended)
    
        return np.square(np.abs(ft_sig))
    
    @classmethod
    def ft_ps_max(cls,
                  ts_detrended: np.ndarray,
                  power_spec: t.Optional[np.ndarray] = None,
                  ft_sig: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Maximal power spectrum frequency of the given signal.
        
        Parameters
        ----------
        ts_detrended : :obj:`np.ndarray`
            One-dimensional detrended time-series values.

        power_spec : :obj:`np.ndarray`, optional
            Power spectrum of ``ts_detrended``. Used to take advantage of
            precomputations.

        ft_sig : :obj:`np.ndarray`, optional
            One-dimensional discrete fourier transform of ``ts_detrended``.
            Used only if ``power_spec`` is None. Used to take advantage
            of precomputations.

        Returns
        -------
        float
            Maximal power spectrum frequency of the given signal.
        """
        if power_spec is None:
            power_spec = cls._calc_power_spec(
                ts_detrended=ts_detrended, ft_sig=ft_sig)

        return np.max(power_spec)

    @classmethod
    def ft_ps_freqs(cls,
                    ts_detrended: np.ndarray,
                    freq_num: int = 3,
                    power_spec: t.Optional[np.ndarray] = None,
                    ft_sig: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Largest power spectrum frequencies of the given signal.
        
        Parameters
        ----------
        ts_detrended : :obj:`np.ndarray`
            One-dimensional detrended time-series values.

        freq_num : int, optional
            Number of largest frequencies to be returned.

        power_spec : :obj:`np.ndarray`, optional
            Power spectrum of ``ts_detrended``. Used to take advantage of
            precomputations.

        ft_sig : :obj:`np.ndarray`, optional
            One-dimensional discrete fourier transform of ``ts_detrended``. Used
            only if ``power_spec`` is None. Used to take advantage
            of precomputations.

        Returns
        -------
        float
            Power spectrum frequency of the given signal.
        """
        if freq_num <= 0:
            raise ValueError("'freq_num' must be positive.")

        if power_spec is None:
            power_spec = cls._calc_power_spec(
                ts_detrended=ts_detrended, ft_sig=ft_sig)

        power_spec = np.sort(power_spec)

        return power_spec[-freq_num:]

    @classmethod
    def ft_ps_peaks(
            cls,
            ts_detrended: np.ndarray,
            factor: float = 0.6,
            power_spec: t.Optional[np.ndarray] = None,
            ft_sig: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Number of significative power spectrum frequencies.

        Given a set of power spectrum frequencies `p`, a power spectrum
        frequency `f_i` is considered significative if and only if
        `f_i` >= factor * max(p), where `factor` is a user-defined
        parameter.
        
        Parameters
        ----------
        ts_detrended : :obj:`np.ndarray`
            One-dimensional detrended time-series values.

        factor : float, optional
            Multiplicative factor of the power spectrum maximum value to
            used to create the threshold to define which power spectrum
            frequencies are significative.

        power_spec : :obj:`np.ndarray`, optional
            Power spectrum of ``ts_detrended``. Used to take advantage of
            precomputations.

        ft_sig : :obj:`np.ndarray`, optional
            One-dimensional discrete fourier transform of ``ts_detrended``.
            Used only if ``power_spec`` is None. Used to take advantage
            of precomputations.

        Returns
        -------
        float
            Maximal power spectrum frequency of the given signal.
        """
        if not 0 < factor < 1:
            raise ValueError("'factor' must be in (0, 1) range.")

        if power_spec is None:
            power_spec = cls._calc_power_spec(
                ts_detrended=ts_detrended, ft_sig=ft_sig)

        max_ps = cls.ft_ps_max(ts_detrended=ts_detrended,
                               power_spec=power_spec)

        return np.sum(power_spec >= factor * max_ps)


def _test() -> None:
    ts = get_data.load_data()
    ts_detrended = data1_detrend.detrend(ts, degrees=1)
    res = MFETSFreqDomain._calc_power_spec(ts_detrended)
    print(res)

    res = MFETSFreqDomain.ft_ps_max(ts_detrended)
    print(res)

    res = MFETSFreqDomain.ft_ps_freqs(ts_detrended)
    print(res)

    res = MFETSFreqDomain.ft_ps_peaks(ts_detrended)
    print(res)


if __name__ == "__main__":
    _test()
