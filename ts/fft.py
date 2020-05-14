import typing as t

import numpy as np
import scipy.signal
import scipy.stats

import _detrend
import _period
import _get_data


class MFETSFreqDomain:
    @classmethod
    def _calc_ps_residuals(cls,
                           ts_residuals: np.ndarray,
                           window: str = "hamming") -> np.ndarray:
        """Calculate the positive side power spectrum of a fourier signal."""
        _, ps = scipy.signal.periodogram(ts_residuals,
                                         detrend=None,
                                         window=window,
                                         scaling="spectrum",
                                         return_onesided=True)
        return ps

    @classmethod
    def ft_ps_residuals(
        cls,
        ts_residuals: np.ndarray,
        ps_residuals: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Power spectrum frequency of the given time-series residuals.
        
        Parameters
        ----------
        ts_residuals : :obj:`np.ndarray`
            Residuals (random noise) of an one-dimensional time-series.

        ps_residuals : :obj:`np.ndarray`, optional
            Power spectrum of ``ts_residuals``. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Power spectrum frequency of the given time-series.
        """
        if ps_residuals is None:
            ps_residuals = cls._calc_ps_residuals(ts_residuals=ts_residuals)

        # Note: in the reference paper, it is used just the maximal value
        # of the power spectrum. However, to enable summarization, here
        # we return the whole power spectrum.
        return ps_residuals

    @classmethod
    def ft_ps_freqs(
        cls,
        ts_residuals: np.ndarray,
        freq_num: t.Union[int, float] = 0.05,
        ps_residuals: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Largest power spectrum frequencies of the given time-series.
        
        Parameters
        ----------
        ts_residuals : :obj:`np.ndarray`
            Residuals (random noise) of an one-dimensional time-series.

        freq_num : int or float, optional
            If int >= 1, number of largest frequencies to be returned.
            If float in (0, 1), fraction of largest frequencies to be returned.

        ps_residuals : :obj:`np.ndarray`, optional
            Power spectrum of ``ts_residuals``. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Largest power spectrum frequencies of the given time-series.
        """
        if freq_num <= 0:
            raise ValueError("'freq_num' must be positive.")

        if freq_num < 1:
            freq_num = np.ceil(freq_num * ts_residuals.size)

        freq_num = int(freq_num)

        if ps_residuals is None:
            ps_residuals = cls._calc_ps_residuals(ts_residuals=ts_residuals)

        ps_highest = np.sort(ps_residuals)[-freq_num:]

        return ps_highest

    @classmethod
    def ft_ps_peaks(
        cls,
        ts_residuals: np.ndarray,
        factor: float = 0.6,
        ps_residuals: t.Optional[np.ndarray] = None,
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

        ps_residuals : :obj:`np.ndarray`, optional
            Power spectrum of ``ts_residuals``. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Binary array marking which time-series residuals power spectrum
            frequencies are larger than `factor * max(ps_residuals)`.
        """
        if not 0 < factor < 1:
            raise ValueError("'factor' must be in (0, 1) range.")

        if ps_residuals is None:
            ps_residuals = cls._calc_ps_residuals(ts_residuals=ts_residuals)

        max_ps = np.max(ps_residuals)

        ps_peaks = (ps_residuals >= factor * max_ps).astype(int)

        # Note: in the reference paper, only the 'ps_peaks' sum is
        # returned. However, to enable summarization, here we return
        # the whole array.
        return ps_peaks

    @classmethod
    def ft_ps_entropy(
        cls,
        ts_residuals: np.ndarray,
        normalize: bool = True,
        ps_residuals: t.Optional[np.ndarray] = None,
        base: int = 2,
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
        if ps_residuals is None:
            ps_residuals = cls._calc_ps_residuals(ts_residuals=ts_residuals)

        # Note: no need to calculate the power spectrum density 'd':
        # d = ps_residuals / ts_residuals.size
        # since a constant factor does not affect the entropy value.
        ps_ent = scipy.stats.entropy(ps_residuals / np.sum(ps_residuals),
                                     base=base)

        if normalize:
            ps_ent /= np.log(ts_residuals.size) / np.log(base)

        return ps_ent


def _test() -> None:
    ts = _get_data.load_data(3)
    ts_period = _period.ts_period(ts=ts)
    ts_trend, ts_season, ts_residuals = _detrend.decompose(ts,
                                                           ts_period=ts_period)

    res = MFETSFreqDomain.ft_ps_residuals(ts_residuals)
    print(res)

    res = MFETSFreqDomain.ft_ps_freqs(ts_residuals)
    print(res)

    res = MFETSFreqDomain.ft_ps_peaks(ts_residuals)
    print(res)

    res = MFETSFreqDomain.ft_ps_entropy(ts_residuals)
    print(res)


if __name__ == "__main__":
    _test()
