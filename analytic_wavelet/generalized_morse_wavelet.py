import inspect
import numpy as np
import scipy.special
from scipy.fftpack import ifft
from scipy.special import eval_genlaguerre
import warnings

from .transform import rotate

__all__ = [
    'GeneralizedMorseWavelet']


class GeneralizedMorseWavelet:

    def __init__(self, gamma, beta, is_bandpass_normalized=True):
        # use broadcasting rules to make gamma and beta the same shape
        self._gamma = np.zeros_like(beta) + gamma
        self._beta = np.zeros_like(gamma) + beta
        self._is_bandpass_normalized = is_bandpass_normalized

    @staticmethod
    def replace(instance, **kwargs):
        """
        Returns a copy of a GeneralizedMorseWavelet with its parameters modified according to kwargs
        Args:
            instance: The instance of the GeneralizedMorseWavelet to copy
            **kwargs: Which parameters to replace
        Returns:
            A new instance of GeneralizedMorseWavelet
        Examples:
            morse_b = GeneralizedMorseWavelet.replace(morse_a, beta=beta + 1)
        """
        property_names = [n for n, v in inspect.getmembers(type(instance), lambda m: isinstance(m, property))]
        init_kwargs = inspect.getfullargspec(type(instance).__init__).args
        replaced = dict()
        for k in init_kwargs[1:]:
            if k in kwargs:
                replaced[k] = kwargs[k]
            elif k in property_names:
                replaced[k] = getattr(instance, k)
        return type(instance)(**replaced)

    @property
    def gamma(self):
        return self._gamma

    @property
    def beta(self):
        return self._beta

    @property
    def is_bandpass_normalized(self):
        return self._is_bandpass_normalized

    def time_domain_width(self):
        """
        This quantity divided by pi is the number of oscillations at the peak frequency which fit within
        the central wavelet window as measured by the standard deviation of the demodulated wavelet
        """
        return np.sqrt(self.beta * self.gamma)

    def footprint(self, scale_frequency, standard_widths=4):
        """
        Gives the number of timepoints which account for most of the wavelet's energy. At standard_widths=4, should
        be approximately 95% of the wavelet's energy
        Args:
            scale_frequency: The scale to return the footprint for
            standard_widths: How many standard_widths to use when calculating the footprint.
                Similar to standard deviations.

        Returns:
            num_timepoints: How many timepoints are in the footprint.
        """
        # see Appendix B of Lilly 2017
        # "Element Analysis: a wavelet based method for analyzing time-localized events in noisy time-series"
        # 2 * sqrt(2) * time_domain_width / scale_frequency is rough 4 standard deviations
        footprint = np.sqrt(2) * standard_widths / 2 * self.time_domain_width() / scale_frequency
        return np.ceil(footprint).astype(int)

    def demodulated_skewness_imag(self):
        # This value is a real number even though the skewness is purely imaginary.
        # Like calling np.imag(demodulated_skewness)
        # To get the actual skewness multiply by 1j
        return (self.gamma - 3) / self.time_domain_width()

    def demodulated_kurtosis(self):
        return 3 - np.power(self.demodulated_skewness_imag(), 2) - (2 / np.power(self.time_domain_width(), 2))

    def peak_frequency(self):
        """
        The frequency at which the wavelet magnitude is maximized, energy is maximized at the same frequency
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            return np.where(
                self.beta == 0,
                np.power(np.log(2), 1 / self.gamma),
                np.exp((1 / self.gamma) * (np.log(self.beta) - np.log(self.gamma))))

    def energy_frequency(self):
        """
        The mean frequency of the wavelet energy
        """
        return (1 / np.power(2, 1 / self.gamma)) * (
                scipy.special.gamma((2 * self.beta + 2) / self.gamma)
                / scipy.special.gamma((2 * self.beta + 1) / self.gamma))

    def instantaneous_frequency(self):
        """
        The instantaneous frequency evaluated at the wavelet center
        """
        return scipy.special.gamma((self.beta + 2) / self.gamma) / scipy.special.gamma((self.beta + 1) / self.gamma)

    def curvature_instantaneous_frequency(self):
        """
        The curvature of instantaneous frequency evaluated at the wavelet center
        """
        _, frequency_cumulants = self.frequency_domain_cumulants(3)
        return frequency_cumulants[3] / np.sqrt(np.power(frequency_cumulants[2], 3))

    def heisenberg_box(self):
        beta = np.where(self.beta == 1, 1 + 1e-10, self.beta)
        beta = np.where(beta < 0.5, np.nan, beta)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            log_sigma_freq_1 = ((2 / self.gamma) * np.log(self.gamma / (2 * beta))
                                + scipy.special.gammaln((2 * beta + 1 + 2) / self.gamma)
                                - scipy.special.gammaln((2 * beta + 1) / self.gamma))
            log_sigma_freq_2 = ((2 / self.gamma) * np.log(self.gamma / (2 * beta))
                                + 2 * scipy.special.gammaln((2 * beta + 2) / self.gamma)
                                - 2 * scipy.special.gammaln((2 * beta + 1) / self.gamma))

            sigma_freq = np.sqrt(np.exp(log_sigma_freq_1) - np.exp(log_sigma_freq_2))

            def _log_a(g, b):
                return (b / g) * (1 + np.log(g) - np.log(b))

            ra = (2 * _log_a(self.gamma, beta)
                  - 2 * _log_a(self.gamma, beta - 1)
                  + _log_a(self.gamma, 2 * (beta - 1))
                  - _log_a(self.gamma, 2 * beta))
            rb = (2 * _log_a(self.gamma, beta)
                  - 2 * _log_a(self.gamma, beta - 1 + self.gamma)
                  + _log_a(self.gamma, 2 * (beta - 1 + self.gamma))
                  - _log_a(self.gamma, 2 * beta))
            rc = (2 * _log_a(self.gamma, beta)
                  - 2 * _log_a(self.gamma, beta - 1 + self.gamma / 2)
                  + _log_a(self.gamma, 2 * (beta - 1 + self.gamma / 2))
                  - _log_a(self.gamma, 2 * beta))

            log_sigma_2a = (ra
                            + (2 / self.gamma) * np.log((beta / self.gamma))
                            + 2 * np.log(beta)
                            + scipy.special.gammaln((2 * (beta - 1) + 1) / self.gamma)
                            - scipy.special.gammaln((2 * beta + 1) / self.gamma))
            log_sigma_2b = (rb
                            + (2 / self.gamma) * np.log((beta / self.gamma))
                            + 2 * np.log(self.gamma)
                            + scipy.special.gammaln((2 * (beta - 1 + self.gamma) + 1) / self.gamma)
                            - scipy.special.gammaln((2 * beta + 1) / self.gamma))
            log_sigma_2c = (rc
                            + (2 / self.gamma) * np.log(beta / self.gamma)
                            + np.log(2) + np.log(beta) + np.log(self.gamma)
                            + scipy.special.gammaln((2 * (beta - 1 + self.gamma / 2) + 1) / self.gamma)
                            - scipy.special.gammaln((2 * beta + 1) / self.gamma))

            sigma_time = np.sqrt(np.exp(log_sigma_2a) + np.exp(log_sigma_2b) - np.exp(log_sigma_2c))
            sigma_time = np.where(np.isnan(beta), np.nan, sigma_time)
            sigma_time = np.real(sigma_time)
            sigma_freq = np.real(sigma_freq)

            area = sigma_time * sigma_freq

        return area, sigma_time, sigma_freq

    # def area_of_concentration(self, c):
    #     r = ((2 * self.beta) + 1) / self.gamma
    #     return (np.pi * (c - 1) * scipy.special.gamma(r + 1 - (1 / self.gamma))
    #             * scipy.special.gamma(r + (1 / self.gamma)) / (self.gamma * scipy.special.gamma(r) ** 2))

    def log_spaced_frequencies(
            self,
            num_timepoints=None,
            high=None,
            low=None,
            nyquist_overlap=None,
            endpoint_overlap=None,
            density=4):
        """
        Returns log-spaced frequencies which can be used as input to make_wavelet. Frequencies are returned
        in descending order.
        Args:
            num_timepoints: The largest window size to consider. If not specified, low must be provided. If
                endpoint_overlap is specified, num_timepoints must also be specified
            high: An explicit high frequency cutoff. If both high and nyquist_overlap are provided, then the cutoff
                frequency is set to the minimum of high and the cutoff determined by the nyquist_overlap parameter.
                If neither high or nyquist_overlap is provided, then the function behaves as though
                nyquist_overlap=0.1 and high=np.pi
            low: An explicit low frequency cutoff. If both low and endpoint_overlap are provided, then the cutoff
                frequency is set to the maximum of low and cutoff determined by the endpoint_overlap parameter.
                If neither low or endpoint_overlap is provided, then the function behaves as though
                endpoint_overlap=5 and low=num_timepoints.
            nyquist_overlap: gives the ratio in [0, 1] of a frequency-domain wavelet at the Nyquist
                frequency to its peak value. The cutoff is the highest frequency which has a ratio no larger than
                this value. If both high and nyquist_overlap are provided, then the cutoff
                frequency is set to the minimum of high and the cutoff determined by the nyquist_overlap parameter.
                If neither high or nyquist_overlap is provided, then the function behaves as though
                nyquist_overlap=0.1 and high=np.pi
            endpoint_overlap: If provided, the lowest frequency wavelet will reach endpoint_overlap times its central
                window width at the ends of the time-window. If both low and endpoint_overlap are provided, then the
                cutoff frequency is set to the maximum of low and cutoff determined by the endpoint_overlap parameter.
                If neither low or endpoint_overlap is provided, then the function behaves as though
                endpoint_overlap=5 and low=num_timepoints.
            density: Controls the amount of overlap in the frequency domain. When density == 1, the peak of one
                wavelet is located at the half-power points of the adjacent wavelet. density == 4 (the default) means
                that four other wavelets will occur between the peak of one wavelet and its half-power point.

        Returns:
            frequencies: An array of log-spaced frequencies in the range determined according to the
                parameters.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            if high is None and nyquist_overlap is None:
                high = np.pi
                nyquist_overlap = 0.1
            if low is None and endpoint_overlap is None:
                if num_timepoints is None:
                    raise ValueError('When low is not provided, num_timepoints must be specified.')
                low = num_timepoints
                endpoint_overlap = 5
            if endpoint_overlap is not None and num_timepoints is None:
                raise ValueError('When endpoint_overlap is set, num_timepoints must be specified.')

            r = 1 + (1 / (density * self.time_domain_width()))

            if nyquist_overlap is not None:
                nyquist_high = self._high_frequency_cutoff(nyquist_overlap)
                if high is None:
                    high = nyquist_high
                else:
                    high = np.where(nyquist_high < high, nyquist_high, high)
            if endpoint_overlap is not None:
                endpoint_low = self._low_frequency_cutoff(r, num_timepoints)
                if low is None:
                    low = endpoint_low
                else:
                    low = np.where(endpoint_low > low, endpoint_low, low)

            n = np.floor(np.log(high / low) / np.log(r)).astype(int)

            if not np.ndim(n) == 0:
                indices = np.reshape(np.arange(np.max(n) + 1), (1,) * len(n.shape) + (np.max(n) + 1,))
                indices = np.where(indices < np.expand_dims(n, -1), indices, np.nan)
                return np.expand_dims(high, 1) / np.power(r, indices)
            return high / np.power(r, np.arange(n + 1))

    def _high_frequency_cutoff(self, eta):
        # helper for log_spaced_frequencies
        omega_high = np.reshape(np.linspace(0, np.pi, 10000), (1,) * len(self.gamma.shape) + (-1,))
        peak_frequency = self.peak_frequency()
        if not np.ndim(peak_frequency) == 0:
            peak_frequency = np.expand_dims(peak_frequency, -1)
        # self.gamma.shape + (10000,)
        omega = peak_frequency * np.pi / omega_high
        ln_psi_1 = (self.beta / self.gamma) * np.log((np.exp(1) * self.gamma) / self.beta)
        beta, gamma = self.beta, self.gamma
        if not np.ndim(self.beta) == 0:
            beta, gamma = np.expand_dims(self.beta, -1), np.expand_dims(self.gamma, -1)
        ln_psi_2 = beta * np.log(omega) - np.power(omega, gamma)
        ln_psi = ln_psi_1 + ln_psi_2
        indices = np.tile(
            np.reshape(np.arange(ln_psi.shape[-1]), (1,) * (len(ln_psi.shape) - 1) + (ln_psi.shape[-1],)),
            ln_psi.shape[:-1] + (1,))
        indices = np.where(np.log(eta) - ln_psi <= 0, indices, np.nan)
        indices = np.nanmin(indices, axis=-1).astype(int)
        f = np.reshape(omega_high[np.reshape(indices, -1)], indices.shape)
        if np.ndim(gamma) == 0:
            return f.item()
        return f

    def _low_frequency_cutoff(self, r, num_timepoints):
        # helper for log_spaced_frequencies
        return (2 * np.sqrt(2) * self.time_domain_width() * r) / num_timepoints

    def amplitude(self, orthogonal_family_order=1):
        if self.is_bandpass_normalized:
            if not np.ndim(self.beta) == 0:
                om = self.peak_frequency()
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    result = 2 / np.exp(self.beta * np.log(om) - np.power(om, self.gamma))
                result[self.beta == 0] = 2
            elif self.beta == 0:
                result = 2
            else:
                om = self.peak_frequency()
                result = 2 / np.exp(self.beta * np.log(om) - np.power(om, self.gamma))
        else:
            r = (2 * self.beta + 1) / self.gamma
            result = np.power(
                2 * np.pi * self.gamma * np.power(2, r)
                * np.exp(
                    scipy.special.gammaln(orthogonal_family_order)
                    - scipy.special.gammaln(orthogonal_family_order + r - 1)),
                1 / 2)
        return result

    @staticmethod
    def _moments_to_cumulants(moments):
        result = np.full(moments.shape, np.nan)
        result[0] = np.log(moments[0])
        for idx in range(1, len(moments)):
            coefficients = np.zeros_like(result[0])
            for k in range(1, idx):
                coefficients += scipy.special.binom(idx - 1, k - 1) * result[k] * (moments[idx - k] / moments[0])
            result[idx] = (moments[idx] / moments[0]) - coefficients
        return result

    def frequency_domain_moment(self, order):
        return (self.amplitude()
                * (1 / (2 * np.pi * self.gamma) * scipy.special.gamma((self.beta + order + 1) / self.gamma)))

    def energy_moment(self, order):
        return ((2 / np.power(2, (order + 1) / self.gamma))
                * GeneralizedMorseWavelet.replace(self, beta=2*self.beta).frequency_domain_moment(order))

    def frequency_domain_cumulants(self, max_order):
        moments = None
        for order in range(max_order + 1):
            moment = self.frequency_domain_moment(order)
            if moments is None:
                moments = np.full((max_order + 1,) + moment.shape, np.nan)
            moments[order] = moment
        return moments, GeneralizedMorseWavelet._moments_to_cumulants(moments)

    def energy_cumulants(self, max_order):
        moments = None
        for order in range(max_order + 1):
            moment = self.energy_moment(order)
            if moments is None:
                moments = np.full((max_order + 1,) + moment.shape, np.nan)
            moments[order] = moment
        return moments, GeneralizedMorseWavelet._moments_to_cumulants(moments)

    def make_wavelet(self, num_timepoints, scale_frequencies, num_orthogonal_family_members=1):

        scale_frequencies = np.asarray(scale_frequencies)

        if len(scale_frequencies.shape) > 1:
            raise ValueError('scale frequencies must be at most 1d')

        if np.ndim(scale_frequencies) == 0:
            scale_frequencies = np.expand_dims(scale_frequencies, 0)

        fo = np.reshape(self.peak_frequency(), self.gamma.shape + (1, 1))
        gamma = np.reshape(self.gamma, self.gamma.shape + (1, 1))
        beta = np.reshape(self.beta, self.beta.shape + (1, 1))

        # ([1, ..., 1,] freq, 1)
        scale_frequencies_reshaped = np.reshape(scale_frequencies, (1,) * (len(fo.shape) - 2) + (-1, 1))
        # fo.shape + (freq, 1)
        fact = np.abs(scale_frequencies_reshaped) / fo
        omega = 2 * np.pi * np.linspace(0, 1 - (1 / num_timepoints), num_timepoints)
        # fo.shape + (freq, time)
        omega = np.reshape(omega, (1,) * (len(fact.shape) - 1) + (-1,)) / fact

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            if self.is_bandpass_normalized:
                psi_zero = np.where(
                    beta == 0,
                    2 * np.exp(-omega ** gamma),
                    2 * np.exp(-beta * np.log(fo) + fo ** gamma + beta * np.log(omega) - omega ** gamma))
            else:
                psi_zero = np.where(
                    beta == 0,
                    np.exp(-omega ** gamma),
                    np.exp(beta * np.log(omega) - omega ** gamma))

        # psi_zero shape is fo.shape + (freq, time)
        psi_zero[..., 0] = 1 / 2 * psi_zero[..., 0]
        psi_zero = np.where(np.isnan(psi_zero), 0, psi_zero)

        psi_f = self._first_family(fact, num_timepoints, num_orthogonal_family_members, omega, psi_zero)
        psi_f = np.where(np.isinf(psi_f), 0, psi_f)
        if len(psi_f.shape) > len(omega.shape):
            # add the family-order axis
            omega = np.expand_dims(omega, -3)
            fact = np.expand_dims(fact, -3)

        psi = ifft(psi_f * rotate(omega * (num_timepoints + 1) / 2 * fact))

        indicator_neg_sf = scale_frequencies < 0

        psi[..., indicator_neg_sf, :] = np.conj(psi[..., indicator_neg_sf, :])
        psi_f[..., indicator_neg_sf, 1:] = np.flip(psi_f[..., indicator_neg_sf, 1:], axis=-1)

        return psi, psi_f

    def taylor_expansion_time_domain_wavelet(self, time, scale_frequency, use_cumulant=False, num_terms=None):
        """
        Computes the value of the wavelet function (of the lowest order orthogonal family member)
        in the time domain at the specified time using a Taylor series expansion. See equations (30) and (31)
        in Lilly and Olhede, 2009. Higher Order Properties of Analytic Wavelets
        Args:
            time: The time point (or time points) for which to compute the wavelet. Arbitrary shape.
            scale_frequency: Which scale(s) to compute the wavelet for. Arbitrary shape.
            use_cumulant: If True, the cumulants are used to compute the log of the wavelet, and it is exponentiated
                after. If False (the default), the moments are used to compute the wavelet directly
            num_terms: How many terms of the Taylor expansion to use. Defaults to 100

        Returns:
            psi, the time domain wavelet evaluated at the points specified by time and scale_frequency.
                shape = scale_frequency.shape + time.shape
        """

        if not np.ndim(self.beta) == 0:
            raise ValueError('This function is only supported for scalar GeneralizedMorseWavelet')

        s = np.ones_like(scale_frequency) if self.beta == 0 else self.peak_frequency() / scale_frequency
        s = np.asarray(s)
        time = np.asarray(time)

        s_shape = s.shape
        time_shape = time.shape

        time = np.reshape(time, (1, -1))
        s = np.reshape(s, (-1, 1))

        if use_cumulant:
            if num_terms is None:
                num_terms = 100  # comment says this is 10 in the cumulant case, but code seems to set to 100
            _, terms = self.frequency_domain_cumulants(num_terms - 1)
        else:
            if num_terms is None:
                num_terms = 100
            terms = [self.frequency_domain_moment(t) for t in range(num_terms)]
        assert(len(terms) == num_terms)
        psi = None
        for order, term in enumerate(terms):
            term_psi = (((1j * (time / s)) ** order) / np.math.factorial(order)) * term
            if psi is None:
                psi = np.full((len(terms),) + term_psi.shape, np.nan, term_psi.dtype)
            # factorial can return an object array type for large orders
            # so we need to cast this back to complex128 after dividing by the factorial
            psi[order] = term_psi.astype(psi.dtype)
        psi = np.nansum(psi, axis=0)
        if use_cumulant:
            psi = np.exp(psi)
        psi = (1 / s) * psi
        psi0 = (1 / s) * self.frequency_domain_moment(0)

        # To remove effects of the Taylor series expansion leading to incorrectly
        # high values far from the wavelet center, any coefficients exceeding the
        # central maximum value by more than five percent are set to NaNs.
        result = np.where(np.abs(psi) > np.abs(psi0) * 1.05, np.nan, psi)
        return np.reshape(result, s_shape + time_shape)

    def _first_family(self, fact, num_timepoints, num_orthogonal_family_members, omega, psi_zero):
        beta = np.reshape(self.beta, self.beta.shape + (1, 1))
        gamma = np.reshape(self.gamma, self.gamma.shape + (1, 1))
        r = (2 * beta + 1) / gamma
        c = r - 1

        # ([1, ..., 1,] freq, time)
        laguerre = np.zeros_like(omega)
        indices = np.arange(int(np.ceil(num_timepoints / 2)))
        # (wavelet, ..., freq, time)
        psi_f = np.zeros(((num_orthogonal_family_members,) + psi_zero.shape), dtype=psi_zero.dtype)

        for k in range(num_orthogonal_family_members):
            if self.is_bandpass_normalized:
                coeff = np.where(
                    beta == 0,
                    1,
                    np.sqrt(
                        np.exp(scipy.special.gammaln(r) + scipy.special.gammaln(k + 1) - scipy.special.gammaln(k + r))))
            else:
                # (freq, time)
                coeff = np.sqrt(1 / fact) * self.amplitude(k + 1)
            laguerre[..., indices] = eval_genlaguerre(k, c, 2 * np.power(omega[..., indices], gamma))
            psi_f[k] = coeff * psi_zero * laguerre
        # either eliminate the family-order axis or move the family-order axis to -3
        if num_orthogonal_family_members == 1:
            psi_f = np.squeeze(psi_f, 0)
        else:
            psi_f = np.moveaxis(psi_f, 0, -3)
        # (..., [wavelet,], freq, time)
        return psi_f

    def _noise_covariance_entry(self, alpha, time_shift, scale, scale_ratio):
        # covariance between the wavelet transform of noise and itself at another time and scale

        r_tilde = (1 + scale_ratio ** self.gamma) ** (1 / self.gamma)

        noise_morse = GeneralizedMorseWavelet.replace(
            self, beta=2 * self.beta - 2 * alpha, is_bandpass_normalized=True)

        fact1 = (self.amplitude() ** 2 / noise_morse.amplitude())

        if self.is_bandpass_normalized:
            fact2 = (((scale_ratio ** self.beta) * (scale ** (2 * alpha - 1)))
                     / (r_tilde ** (2 * self.beta - 2 * alpha + 1)))
        else:
            fact2 = (((scale_ratio ** (self.beta + 1 / 2)) * (scale ** (2 * alpha)))
                     / (r_tilde ** (2 * self.beta - 2 * alpha + 1)))

        assert (np.ndim(time_shift) == 0)
        assert (scale.ndim < 2)

        psi = noise_morse.taylor_expansion_time_domain_wavelet(
            time_shift / (scale * r_tilde), noise_morse.peak_frequency())

        return fact1 * fact2 * np.conj(psi)

    def _noise_covariance(self, alpha, scale_ratio, s):
        # noise covariance of this point and the 4 adjacent points in the time/scale plane
        # this is the covariance structure given by Eq. (4.16) in Lilly 2017 assuming that we
        # group the points as:
        # [(t, s), (t + 1, s), (t - 1, s), (t, rs), (t, s / r)]
        sigma_0_0 = self._noise_covariance_entry(alpha, 0, s, 1)
        sigma = np.full(s.shape + (5, 5), np.nan, dtype=sigma_0_0.dtype)
        sigma[..., 0, 0] = sigma_0_0
        sigma[..., 0, 1] = self._noise_covariance_entry(alpha, 1, s, 1)
        sigma[..., 0, 2] = self._noise_covariance_entry(alpha, -1, s, 1)
        sigma[..., 0, 3] = self._noise_covariance_entry(alpha, 0, s, scale_ratio)
        sigma[..., 0, 4] = self._noise_covariance_entry(alpha, 0, s, 1 / scale_ratio)
        sigma[..., 1, 1] = sigma_0_0
        sigma[..., 1, 2] = self._noise_covariance_entry(alpha, -2, s, 1)
        sigma[..., 1, 3] = self._noise_covariance_entry(alpha, -1, s, scale_ratio)
        sigma[..., 1, 4] = self._noise_covariance_entry(alpha, -1, s, 1 / scale_ratio)
        sigma[..., 2, 2] = sigma_0_0
        sigma[..., 2, 3] = self._noise_covariance_entry(alpha, 1, s, scale_ratio)
        sigma[..., 2, 4] = self._noise_covariance_entry(alpha, 1, s, 1 / scale_ratio)
        sigma[..., 3, 3] = self._noise_covariance_entry(alpha, 0, scale_ratio * s, 1)
        sigma[..., 3, 4] = self._noise_covariance_entry(alpha, 0, scale_ratio * s, 1 / scale_ratio ** 2)
        sigma[..., 4, 4] = self._noise_covariance_entry(alpha, 0, s / scale_ratio, 1)

        for i in range(sigma.shape[-2]):
            for j in range(i):
                sigma[..., i, j] = np.conj(sigma[..., j, i])

        moment = self.energy_moment(-2 * alpha)
        if self.is_bandpass_normalized:
            sigma = sigma / np.reshape(moment * s ** (2 * alpha - 1), s.shape + (1, 1))
        else:
            sigma = sigma / np.reshape(moment * s ** (2 * alpha), s.shape + (1, 1))

        return sigma

    @staticmethod
    def _get_noise_maxima_values(cholesky_lower, num_monte_carlo_realizations):
        noise_shape = cholesky_lower.shape[:-1] + (num_monte_carlo_realizations,)
        noise = (np.random.randn(*noise_shape) + 1j * np.random.randn(*noise_shape)) / np.sqrt(2)
        noise = np.abs(np.matmul(cholesky_lower, noise))
        # remember which samples have the unshifted and unscaled noise as the maximum
        indices_maxima = np.nonzero(np.argmax(noise, axis=-2) == 0)
        # now that we know which samples to keep, remove the translated values
        noise = noise[..., 0, :]
        # return the coordinates of the maxima and the maxima values
        return indices_maxima, noise[indices_maxima]

    def distribution_of_maxima_of_transformed_noise(
            self,
            spectral_slope,
            scale_ratio,
            scale_frequency=None,
            num_monte_carlo_realizations=1000,
            max_batch_size=1e7,
            **histogram_kwargs):
        """
        Returns a histogram of the modulus of the maxima of noise transformed by the analytic wavelet transform using
        the specified GeneralizedMorseWavelet. The transform is not actually run on the noise. Rather the method
        described in Lilly 2017 which uses the analytically determined covariance structure of noise after the
        transform is used. This function should be run on only a single scale-frequency. All scale-frequencies will
        give the same results, as shown in Lilly 2017.
        Args:
            spectral_slope: The slope of the noise. (alpha in Lilly 2017). 0 gives white noise. 1 gives red noise.
            scale_ratio: This is the ratio of a scale to the next smaller scale in a planned run of element analysis.
            scale_frequency: Which scale frequency to compute the distribution for. A scalar value. If None, the
                morse wavelet peak_frequency is used
            num_monte_carlo_realizations: How many random samples to use for estimating the distribution.
            max_batch_size: No more than this many monte-carlo-realizations will be computed simultaneously. This
                keeps us from running out of memory
            **histogram_kwargs: Arguments to np.histogram. Note that the 'weights' argument is not allowed here
                and a ValueError will be raised if it is used.
        Returns:
            hist: The binned moduli of the maxima as returned by np.histogram.
            bin_edges: The edges of the bins of hist, similar to np.histogram. Note that bin_edges may vary from
                scale_frequency to scale_frequency. If scale_frequencies is not scalar, then this will be a list with
                bin_edges[i] corresponding to scale_frequencies[i]
        """

        if not np.ndim(self.beta) == 0:
            raise ValueError('This function is only supported on scalar instances of GeneralizedMorseWavelet')
        if not np.ndim(spectral_slope) == 0:
            raise ValueError('spectral_slope must be scalar')
        if scale_frequency is None:
            scale_frequency = self.peak_frequency()
        if not np.ndim(scale_frequency) == 0:
            raise ValueError('scale_frequency must be scalar')

        scale = self.peak_frequency() / scale_frequency

        if 'weights' in histogram_kwargs:
            raise ValueError('weights is disallowed in histogram_kwargs')

        sigma = self._noise_covariance(spectral_slope, scale_ratio, scale)
        lower = np.linalg.cholesky(sigma)

        if num_monte_carlo_realizations > max_batch_size:
            # divide up evenly
            num_batches = int(np.ceil(num_monte_carlo_realizations / max_batch_size))
            batch_size = num_monte_carlo_realizations // num_batches
            maxima_values = list()
            for idx in range(num_batches):
                b = num_monte_carlo_realizations - batch_size * idx if idx == (num_batches - 1) else batch_size
                _, v = GeneralizedMorseWavelet._get_noise_maxima_values(lower, b)
                maxima_values.extend(v)
            maxima_values = np.array(maxima_values)
        else:
            _, maxima_values = GeneralizedMorseWavelet._get_noise_maxima_values(lower, num_monte_carlo_realizations)

        return np.histogram(maxima_values, **histogram_kwargs)
