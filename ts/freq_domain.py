"""Module dedicated to frequency domain time-series meta-features."""
import typing as t

import numpy as np
import scipy.signal
import scipy.stats


class MFETSFreqDomain:
    """Extract time-series meta-features from Frequency Domain group."""
    @classmethod
    def precompute_ps_residuals(cls, ts_residuals: np.ndarray,
                                **kwargs) -> t.Dict[str, np.ndarray]:
        """Precompute the power spectrum frequency of the residuals.

        Parameters
        ----------
        ts_residuals : :obj:`np.ndarray`
            Residuals of an one-dimensional time-series.

        kwargs:
            Additional arguments and previous precomputed items. May
            speed up this precomputation.

        Returns
        -------
        dict
            The following precomputed item is returned:
                * ``ps_residuals`` (:obj:`np.ndarray`): power spectrum
                    frequency of the time-series residuals.

        References
        ----------
        .. [1] P. Welch, "The use of the fast Fourier transform for the
            estimation of power spectra: A method based on time averaging
            over short, modified periodograms", IEEE Trans. Audio
            Electroacoust. vol. 15, pp. 70-73, 1967.
        """
        precomp_vals = {}  # type: t.Dict[str, np.ndarray]

        if "ps_residuals" not in kwargs:
            ps_residuals = cls._calc_ps_residuals(ts_residuals=ts_residuals)
            precomp_vals["ps_residuals"] = ps_residuals

        return precomp_vals

    @classmethod
    def _calc_ps_residuals(
            cls,
            ts_residuals: np.ndarray,
            window: str = "hamming",
            scaling: str = "spectrum",
            return_freqs: bool = False,
    ) -> t.Union[np.ndarray, t.Tuple[np.ndarray, np.ndarray]]:
        """Calculate the positive side power spectrum of a fourier signal.

        References
        ----------
        .. [1] P. Welch, "The use of the fast Fourier transform for the
            estimation of power spectra: A method based on time averaging
            over short, modified periodograms", IEEE Trans. Audio
            Electroacoust. vol. 15, pp. 70-73, 1967.
        """
        freqs, ps = scipy.signal.periodogram(ts_residuals,
                                             detrend=None,
                                             window=window,
                                             scaling=scaling,
                                             return_onesided=True)

        if return_freqs:
            return freqs, ps

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
            Residuals of an one-dimensional time-series.

        ps_residuals : :obj:`np.ndarray`, optional
            Power spectrum of ``ts_residuals``. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Power spectrum frequency of the given time-series.

        References
        ----------
        .. [1] P. Welch, "The use of the fast Fourier transform for the
            estimation of power spectra: A method based on time averaging
            over short, modified periodograms", IEEE Trans. Audio
            Electroacoust. vol. 15, pp. 70-73, 1967.
        .. [2] Lemke, Christiane & Gabrys, Bogdan. (2010). Meta-learning for
            time series forecasting and forecast combination. Neurocomputing.
            73. 2006-2016. 10.1016/j.neucom.2009.09.020.
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
            Residuals of an one-dimensional time-series.

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

        References
        ----------
        .. [1] P. Welch, "The use of the fast Fourier transform for the
            estimation of power spectra: A method based on time averaging
            over short, modified periodograms", IEEE Trans. Audio
            Electroacoust. vol. 15, pp. 70-73, 1967.
        .. [2] Lemke, Christiane & Gabrys, Bogdan. (2010). Meta-learning for
            time series forecasting and forecast combination. Neurocomputing.
            73. 2006-2016. 10.1016/j.neucom.2009.09.020.
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
            Residuals of an one-dimensional time-series.

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

        References
        ----------
        .. [1] P. Welch, "The use of the fast Fourier transform for the
            estimation of power spectra: A method based on time averaging
            over short, modified periodograms", IEEE Trans. Audio
            Electroacoust. vol. 15, pp. 70-73, 1967.
        .. [2] Lemke, Christiane & Gabrys, Bogdan. (2010). Meta-learning for
            time series forecasting and forecast combination. Neurocomputing.
            73. 2006-2016. 10.1016/j.neucom.2009.09.020.
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
            base: int = 2,
            ps_residuals: t.Optional[np.ndarray] = None,
    ) -> float:
        """Spectral entropy of time-series residuals.

        The spectral entropy is the entropy if the normalized power spectrum
        of the detrended time-series. Technically, it is the entropy of the
        spectral density, which is the power spectrum normalized by the length
        of the time-series. However, this constant factor of normalization does
        not affect the entropy value.

        Parameters
        ----------
        ts_residuals : :obj:`np.ndarray`
            Residuals of an one-dimensional time-series.

        normalize : bool, optional (default=True)
            If True, the return value will be normalized in [0, 1].

        base : int, optional (default=2)
            Base of the entropy measure.

        ps_residuals : :obj:`np.ndarray`, optional
            Power spectrum of ``ts_residuals``. Used to take advantage of
            precomputations.

        Returns
        -------
        float
            Spectral entropy of time-series residuals.

        References
        ----------
        .. [1] P. Welch, "The use of the fast Fourier transform for the
            estimation of power spectra: A method based on time averaging
            over short, modified periodograms", IEEE Trans. Audio
            Electroacoust. vol. 15, pp. 70-73, 1967.
        .. [2] Pan, Y. N., J. Chen, and X. L. Li. "Spectral Entropy: A
            Complementary Index for Rolling Element Bearing Performance
            Degradation Assessment." Proceedings of the Institution of
            Mechanical Engineers, Part C: Journal of Mechanical Engineering
            Science. Vol. 223, Issue 5, 2009, pp. 1223–1231.
        .. [3] Shen, J., J. Hung, and L. Lee. "Robust Entropy-Based Endpoint
            Detection for Speech Recognition in Noisy Environments."
            ICSLP. Vol. 98, November 1998.
        .. [4] Vakkuri, A., A. Yli‐Hankala, P. Talja, S. Mustola, H.
            Tolvanen‐Laakso, T. Sampson, and H. Viertiö‐Oja. "Time‐Frequency
            Balanced Spectral Entropy as a Measure of Anesthetic Drug Effect
            in Central Nervous System during Sevoflurane, Propofol, and
            Thiopental Anesthesia." Acta Anaesthesiologica Scandinavica.
            Vol. 48, Number 2, 2004, pp. 145–153.
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

    @classmethod
    def ft_low_freq_power(cls,
                          ts_residuals: np.ndarray,
                          threshold: float = 0.04) -> float:
        """Proportion of low frequency in power spectrum using Hann window.

        Parameters
        ----------
        ts_residuals : :obj:`np.ndarray`
            Residuals of an one-dimensional time-series.

        threshold : float, optional (default=0.04)
            The threshold used to define how low is a `low frequency`.

        Returns
        -------
        float
            Proportion of low frequency in power spectrum of the time-series
            residuals, using a Hann window.

        References
        ----------
        .. [1] "Heart rate variability: Standards of measurement, physiological
            interpretation, and clinical use", M. Malik et al., Eur. Heart J.
            17(3) 354 (1996)
        .. [2] B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework
            for Automated Time-Series Phenotyping Using Massive Feature
            Extraction, Cell Systems 5: 527 (2017).
            DOI: 10.1016/j.cels.2017.10.001
        .. [3] B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative
            time-series analysis: the empirical structure of time series and
            their methods", J. Roy. Soc. Interface 10(83) 20130048 (2013).
            DOI: 10.1098/rsif.2013.0048
        """
        freqs, hann_ps = cls._calc_ps_residuals(ts_residuals=ts_residuals,
                                                window="hann",
                                                return_freqs=True)

        # Note: scale frequencies to the range [0, pi] (originally in [0, 0.5])
        # in order to be consistent with the Ben Fulcher code used as reference
        freqs *= 2 * np.pi

        # Note: no need to multiply both numerator and denominator by
        # freqs[1] - freqs[0], because these factor will cancel each other.
        low_freq_prop = np.sum(hann_ps[freqs <= threshold]) / np.sum(hann_ps)

        return low_freq_prop
