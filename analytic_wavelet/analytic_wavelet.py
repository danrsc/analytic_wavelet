import inspect
import numpy as np
import scipy.special
from scipy.fftpack import ifft, fft
from scipy.interpolate import interp1d
from scipy.special import eval_genlaguerre
import warnings


__all__ = [
    'GeneralizedMorseWavelet',
    'rotate',
    'masked_detrend',
    'make_unpad_slices',
    'unpad',
    'to_frequency_domain_wavelet',
    'analytic_wavelet_transform',
    'transform_maxima',
    'quadratic_interpolate',
    'linear_interpolate']


class GeneralizedMorseWavelet:

    def __init__(self, gamma, beta, is_bandpass_normalized=True):
        # use broadcasting rules to make gamma and beta the same shape
        self._gamma = np.zeros_like(beta) + gamma
        self._beta = np.zeros_like(gamma) + beta
        self._is_bandpass_normalized = is_bandpass_normalized

    @staticmethod
    def replace(instance, **kwargs) -> 'GeneralizedMorseWavelet':
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
        # see Appendix B of Lilly 2017
        # "Element Analysis: a wavelet based method for analyzing time-localized events in noisy time-series"
        # 2 * sqrt(2) * time_domain_width / scale_frequency is rough 4 standard deviations
        footprint = np.sqrt(2) * standard_widths / 2 * self.time_domain_width() / scale_frequency
        return np.ceil(footprint).astype(int)

    def demodulated_skewness_imag(self):
        """
        This value is a real number even though the skewness is purely imaginary.
        Like calling np.imag(demodulated_skewness)
        To get the actual skewness multiply by 1j
        """
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
        # use broadcasting rules to guarantee the shapes work
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
                window width at the ends of the time-window. If both low and endpoint_overlap are provided, then the cutoff
                frequency is set to the maximum of low and cutoff determined by the endpoint_overlap parameter.
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

            if not np.isscalar(n):
                indices = np.reshape(np.arange(np.max(n) + 1), (1,) * len(n.shape) + (np.max(n) + 1,))
                indices = np.where(indices < np.expand_dims(n, -1), indices, np.nan)
                return np.expand_dims(high, 1) / np.power(r, indices)
            return high / np.power(r, np.arange(n + 1))

    def _high_frequency_cutoff(self, eta):
        omega_high = np.reshape(np.linspace(0, np.pi, 10000), (1,) * len(self.gamma.shape) + (-1,))
        peak_frequency = self.peak_frequency()
        if not np.isscalar(peak_frequency):
            peak_frequency = np.expand_dims(peak_frequency, -1)
        # self.gamma.shape + (10000,)
        omega = peak_frequency * np.pi / omega_high
        ln_psi_1 = (self.beta / self.gamma) * np.log((np.exp(1) * self.gamma) / self.beta)
        beta, gamma = self.beta, self.gamma
        if not np.isscalar(self.beta):
            beta, gamma = np.expand_dims(self.beta, -1), np.expand_dims(self.gamma, -1)
        ln_psi_2 = beta * np.log(omega) - np.power(omega, gamma)
        ln_psi = ln_psi_1 + ln_psi_2
        indices = np.tile(
            np.reshape(np.arange(ln_psi.shape[-1]), (1,) * (len(ln_psi.shape) - 1) + (ln_psi.shape[-1],)),
            ln_psi.shape[:-1] + (1,))
        indices = np.where(np.log(eta) - ln_psi <= 0, indices, np.nan)
        indices = np.nanmin(indices, axis=-1).astype(int)
        f = np.reshape(omega_high[np.reshape(indices, -1)], indices.shape)
        if np.isscalar(gamma):
            return f.item()
        return f

    def _low_frequency_cutoff(self, r, num_timepoints):
        return (2 * np.sqrt(2) * self.time_domain_width() * r) / num_timepoints

    def amplitude(self, orthogonal_family_order=1):
        if self.is_bandpass_normalized:
            if not np.isscalar(self.beta):
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

        if np.isscalar(scale_frequencies):
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

    def zeta(self, beta_2):
        """
        The maximum of the Morse wavelet transform of another wavelet
        """
        if self.is_bandpass_normalized:
            numerator = (self.beta / (beta_2 + 1)) ** (self.beta / self.gamma)
            denominator = ((self.beta / (beta_2 + 1)) + 1) ** ((self.beta + beta_2 + 1) / self.gamma)
        else:
            numerator = ((self.beta + 1 / 2) / (beta_2 + 1 / 2)) ** ((self.beta + 1 / 2) / self.gamma)
            denominator = ((self.beta + 1 / 2) / (beta_2 + 1 / 2) + 1) ** ((self.beta + beta_2 + 1) / self.gamma)
        mu_morse = GeneralizedMorseWavelet.replace(self, beta=beta_2)
        return ((1 / (2 * np.pi * self.gamma))
                * self.amplitude()
                * mu_morse.amplitude()
                * scipy.special.gamma((self.beta + beta_2 + 1) / self.gamma)
                * (numerator / denominator))

    def maxima_parameters(self, maxima_values, maxima_scale_frequencies, event_beta):
        """
        Estimates the parameters of the events corresponding to the maxima
        Args:
            maxima_values: output from transform_maxima
            maxima_scale_frequencies: output from transform_maxima
            event_beta: The beta parameter of the event wavelets

        Returns:
            coefficients (complex), scales, scale_frequencies
        """
        mu_morse = GeneralizedMorseWavelet.replace(self, beta=event_beta)
        if self.is_bandpass_normalized:
            f_hat = (maxima_scale_frequencies
                     * (mu_morse.peak_frequency() / self.peak_frequency())
                     * (self.beta / (event_beta + 1)) ** (1 / self.gamma))
        else:
            f_hat = (maxima_scale_frequencies
                     * (mu_morse.peak_frequency() / self.peak_frequency())
                     * ((self.beta + 1 / 2) / (event_beta + 1 / 2)) ** (1 / self.gamma))
        c_hat = 2 * maxima_values / self.zeta(event_beta)
        return c_hat, mu_morse.peak_frequency() / f_hat, f_hat


def _rotate_main_angle(x):
    x = np.mod(x, 2 * np.pi)
    return np.where(x > np.pi, x - 2 * np.pi, x)


def rotate(x):
    x = _rotate_main_angle(x)
    edge_cases = np.full_like(1j * x, np.nan)
    edge_cases[x == np.pi / 2] = 1j
    edge_cases[x == -np.pi / 2] = -1j
    edge_cases[np.logical_or(x == np.pi, x == -np.pi)] = -1
    edge_cases[np.logical_or(x == 0, np.logical_or(x == 2 * np.pi, x == -2 * np.pi))] = 1
    return np.where(np.isnan(edge_cases), np.exp(1j * x), edge_cases)


def to_frequency_domain_wavelet(time_domain_wavelet, num_timepoints=None):
    """
    Converts a time-domain wavelet to a frequency domain wavelet
    Args:
        time_domain_wavelet: The wavelet to convert with shape (..., time)
        num_timepoints: If specified, the time domain wavelet is extended with 0 padding on both sides.
            If smaller than time_domain_wavelet.shape[-1], currently an error is raised

    Returns:
        The frequency domain wavelet
    """
    if num_timepoints is not None:
        if time_domain_wavelet.shape[-1] < num_timepoints:
            padding = (num_timepoints - time_domain_wavelet.shape[-1]) // 2
            if padding * 2 + time_domain_wavelet.shape[-1] < num_timepoints:
                padding = (padding, padding + 1)
            else:
                padding = (padding, padding)
            pad_width = (0, 0) * (len(time_domain_wavelet.shape) - 1) + padding
            time_domain_wavelet = np.pad(time_domain_wavelet, pad_width, mode='constant')
        elif time_domain_wavelet.shape[-1] > num_timepoints:
            time_domain_wavelet = time_domain_wavelet[..., :num_timepoints]
            raise ValueError('unexpected')  # original code multiplies by nan, not sure what is going on here

    psi_f = fft(time_domain_wavelet)
    omega = 2 * np.pi * np.linspace(0, 1 - (1 / psi_f.shape[-1]), psi_f.shape[-1])
    omega = np.reshape(omega, (1,) * (len(psi_f.shape) - 1) + (omega.shape[0],))
    if psi_f.shape[-1] // 2 * 2 == psi_f.shape[-1]:  # is even
        psi_f = psi_f * rotate(-omega * (psi_f.shape[-1] + 1) / 2) * np.sign(np.pi - omega)
    else:
        psi_f = psi_f * rotate(-omega * (psi_f.shape[-1] + 1) / 2)
    return psi_f


def _awt(x, psi_f, is_time_domain_wavelet_real):

    # x is (batch, time)
    # psi_f is (..., scale, time)

    # unitary transform normalization ?
    if not np.isrealobj(x):
        x = x / np.sqrt(2)

    psi_f = np.conj(psi_f)
    # -> (batch, k * scale_frequencies, time), where k is a stand in for additional dimensions
    result = np.expand_dims(fft(np.where(np.isnan(x), 0, x)), 1) * np.reshape(psi_f, (1, -1, psi_f.shape[-1]))
    # -> (batch, ..., scale_frequencies, time)
    result = np.reshape(result, (x.shape[0],) + psi_f.shape)
    x = np.reshape(x, (x.shape[0],) + (1,) * (len(result.shape) - 2) + (x.shape[1],))
    result = ifft(result)
    if np.isrealobj(x) and is_time_domain_wavelet_real and not np.isrealobj(result):
        result = np.real(result)
    if not np.any(np.isfinite(result)):
        if not np.isrealobj(result):
            result = np.inf * (1 + 1j) * np.ones_like(result)
        else:
            result = np.inf * np.ones_like(result)
    if not np.isrealobj(result):
        result = np.where(np.isnan(x), np.nan * (1 + 1j), result)
    else:
        result = np.where(np.isnan(x), np.nan, result)
    return result


def analytic_wavelet_transform(x, frequency_domain_wavelet, is_time_domain_wavelet_real, unpad_slices=None):
    """
    Computes the transform of x using frequency_domain_wavelet, a wavelet defined in the frequency_domain.
    Args:
        x: An array with shape (..., time)
        frequency_domain_wavelet: An array of wavelets in the frequency domain of shape (..., time). The first N-1
            dimensions are for multiple scales, wavelet families, etc. and need not match the shape of the first M-1
            dimensions of x.
        is_time_domain_wavelet_real: A boolean indicating whether the time domain form of frequency_domain_wavelet
            is real or complex. When True, and x is real the result is coerced to real
        unpad_slices: The result of calling make_unpad_slices(x.ndim, pad_width) with the pad_width that was used to
            pad x. If specified, padding will be stripped from w before it is returned.
            This is equivalent to
                w = analytic_wavelet_transform(x, ...)
                w = w[unpad_slices[:-1] + (slice(None),) * (w.ndim - x.ndim) + unpad_slices[-1]]
    Returns:
        w: Wavelet coefficients with shape x.shape[:-1] + frequency_domain_wavelet.shape[:-1] + (time,)
    """
    x = np.asarray(x)
    frequency_domain_wavelet = np.asarray(frequency_domain_wavelet)

    if x.shape[-1] != frequency_domain_wavelet.shape[-1]:
        raise ValueError(
            'x and frequency_domain_wavelet must match on the time axis. '
            'x.shape: {}, frequency_domain_wavelet.shape: {}'.format(x.shape, frequency_domain_wavelet.shape))

    if unpad_slices is not None:
        if not isinstance(unpad_slices, (tuple, list)):
            unpad_slices = (unpad_slices,)
        else:
            unpad_slices = tuple(unpad_slices)
        if len(unpad_slices) != x.ndim:
            raise ValueError('Mismatched dimensions. Got {} unpad_slices, but x has {} axes'.format(
                len(unpad_slices), x.ndim))

    x_shape = x.shape
    x = np.reshape(x, (-1, x.shape[-1]))

    indicator_good = np.sum(np.isfinite(x), axis=-1) > 1
    if np.sum(indicator_good) == 0:
        # make a dummy result of the correct size
        result = _awt(np.zeros((1, x.shape[-1]), dtype=x.dtype), frequency_domain_wavelet, is_time_domain_wavelet_real)
        result = np.inf * (1 + 1j) * np.tile(result, (x.shape[0],) + (1,) * result.shape[1:])
    else:
        x = x[indicator_good]
        result_good = _awt(x, frequency_domain_wavelet, is_time_domain_wavelet_real)
        result = np.full((x.shape[0],) + result_good.shape[1:], np.inf, dtype=result_good.dtype)
        result[indicator_good] = result_good
    result = np.reshape(result, x_shape[:-1] + result.shape[1:])
    if unpad_slices is not None:
        s = list(unpad_slices[:-1])
        for _ in range(len(unpad_slices), result.ndim):
            s.append(slice(None))
        s.append(unpad_slices[-1])
        s = tuple(s)
        assert(len(s) == result.ndim)
        result = result[s]
    return result


def transform_maxima(scale_frequencies, w, min_amplitude=None, freq_axis=-2, time_axis=-1):
    w0 = w
    w += np.random.randn(w.shape) * np.finfo(w.dtype).eps
    w = np.abs(w)
    indicator = np.logical_not(w == 0)
    indicator = np.logical_and(indicator, w > np.roll(w, 1, axis=time_axis))
    indicator = np.logical_and(indicator, w > np.roll(w, -1, axis=time_axis))
    indicator = np.logical_and(indicator, w > np.roll(w, 1, axis=freq_axis))
    indicator = np.logical_and(indicator, w > np.roll(w, -1, axis=freq_axis))
    edges = np.array([0, -1])
    shape = [1] * len(w.shape)
    shape[time_axis] = 2
    np.put_along_axis(indicator, np.reshape(edges, shape), False, time_axis)
    shape[time_axis] = 1
    shape[freq_axis] = 2
    np.put_along_axis(indicator, np.reshape(edges, shape), False, freq_axis)
    if min_amplitude is not None:
        indicator = np.logical_and(indicator, w >= min_amplitude)
    indices = np.nonzero(indicator)
    freq_indices = indices[freq_axis]
    indices_less_1 = indices[:freq_axis] + (freq_indices - 1) + indices[freq_axis + 1:]
    indices_plus_1 = indices[:freq_axis] + (freq_indices + 1) + indices[freq_axis + 1:]
    _, freq_hat = quadratic_interpolate(
        freq_indices - 1, freq_indices, freq_indices + 1,
        np.abs(w0[indices_less_1]), np.abs(w0[indices]), np.abs(w0[indices_plus_1]))
    interpolated = quadratic_interpolate(
        freq_indices - 1, freq_indices, freq_indices + 1,
        w0[indices_less_1], w0[indices], w0[indices_plus_1], freq_hat)
    scale_frequencies_interp = interp1d(np.arange(len(scale_frequencies)), scale_frequencies)
    return indices, interpolated, scale_frequencies_interp(freq_hat)


def quadratic_interpolate(t1, t2, t3, x1, x2, x3, t=None):
    numerator = x1 * (t2 - t3) + x2 * (t3 - t1) + x3 * (t1 - t2)
    denominator = (t1 - t2) * (t1 - t3) * (t2 - t3)
    a = numerator / denominator

    numerator = x1 * (t2 ** 2 - t3 ** 2) + x2 * (t3 ** 2 - t1 ** 2) + x3 * (t1 ** 2 - t2 ** 2)
    b = -numerator / denominator

    numerator = x1 * t2 * t3 * (t2 - t3) + x2 * t3 * t1 * (t3 - t1) + x3 * t1 * t2 * (t1 - t2)
    c = numerator / denominator

    return_t = t is None
    if t is None:
        t = -b / 2 * a

    x = a * t ** 2 + b * t + c
    if return_t:
        return x, t
    return x


def linear_interpolate(t1, t2, x1, x2, t=None):
    a = (x2 - x1) / (t2 - t1)
    b = x1 - a * t1
    if t is None:
        return -b / a
    return a * t + b


# unfortunately, scipy.signal.detrend does not handle masking
def _masked_detrend(arr, axis=-1):

    to_fit = np.ma.masked_invalid(np.moveaxis(arr, axis, 0))
    x = np.arange(len(to_fit))

    shape = to_fit.shape
    to_fit = np.reshape(to_fit, (to_fit.shape[0], -1))

    # some columns might not have any data
    indicator_can_fit = np.logical_not(np.all(to_fit.mask, axis=0))

    p = np.ma.polyfit(x, to_fit[:, indicator_can_fit], deg=1)

    filled_p = np.zeros((p.shape[0], len(indicator_can_fit)), p.dtype)
    filled_p[:, indicator_can_fit] = p
    p = filled_p

    #      (1, num_columns)            (num_rows, 1)
    lines = np.reshape(p[0], (1, -1)) * np.reshape(np.arange(len(to_fit)), (-1, 1)) + np.reshape(p[1], (1, -1))
    lines = np.moveaxis(np.reshape(lines, shape), 0, axis)
    return arr - lines


def masked_detrend(arr, axis=-1):
    if not np.isrealobj(arr):
        return _masked_detrend(np.real(arr), axis=axis) + 1j * _masked_detrend(np.imag(arr), axis=axis)
    return _masked_detrend(arr, axis=axis)


def _as_pairs(x, ndim, as_index=False):
    # copied from https://github.com/numpy/numpy/blob/v1.17.0/numpy/lib/arraypad.py
    """
    Broadcast `x` to an array with the shape (`ndim`, 2).
    A helper function for `pad` that prepares and validates arguments like
    `pad_width` for iteration in pairs.
    Parameters
    ----------
    x : {None, scalar, array-like}
        The object to broadcast to the shape (`ndim`, 2).
    ndim : int
        Number of pairs the broadcasted `x` will have.
    as_index : bool, optional
        If `x` is not None, try to round each element of `x` to an integer
        (dtype `np.intp`) and ensure every element is positive.
    Returns
    -------
    pairs : nested iterables, shape (`ndim`, 2)
        The broadcasted version of `x`.
    Raises
    ------
    ValueError
        If `as_index` is True and `x` contains negative elements.
        Or if `x` is not broadcastable to the shape (`ndim`, 2).
    """
    if x is None:
        # Pass through None as a special case, otherwise np.round(x) fails
        # with an AttributeError
        return ((None, None),) * ndim

    x = np.array(x)
    if as_index:
        x = np.round(x).astype(np.intp, copy=False)

    if x.ndim < 3:
        # Optimization: Possibly use faster paths for cases where `x` has
        # only 1 or 2 elements. `np.broadcast_to` could handle these as well
        # but is currently slower

        if x.size == 1:
            # x was supplied as a single value
            x = x.ravel()  # Ensure x[0] works for x.ndim == 0, 1, 2
            if as_index and x < 0:
                raise ValueError("index can't contain negative values")
            return ((x[0], x[0]),) * ndim

        if x.size == 2 and x.shape != (2, 1):
            # x was supplied with a single value for each side
            # but except case when each dimension has a single value
            # which should be broadcasted to a pair,
            # e.g. [[1], [2]] -> [[1, 1], [2, 2]] not [[1, 2], [1, 2]]
            x = x.ravel()  # Ensure x[0], x[1] works
            if as_index and (x[0] < 0 or x[1] < 0):
                raise ValueError("index can't contain negative values")
            return ((x[0], x[1]),) * ndim

    if as_index and x.min() < 0:
        raise ValueError("index can't contain negative values")

    # Converting the array with `tolist` seems to improve performance
    # when iterating and indexing the result (see usage in `pad`)
    return np.broadcast_to(x, (ndim, 2)).tolist()


def make_unpad_slices(ndim, pad_width):
    slices = list()
    for begin_pad, end_pad in _as_pairs(pad_width, ndim, as_index=True):
        if end_pad is not None:
            if end_pad == 0:
                end_pad = None
            else:
                end_pad = -end_pad
        slices.append(slice(begin_pad, end_pad))
    return tuple(slices)


def unpad(pad_width, *args):
    if len(args) == 0:
        raise ValueError('Expected at least one array to unpad')
    unpad_slices = make_unpad_slices(args[0].ndim, pad_width)
    if any(not np.array_equal(a.shape, args[0].shape) for a in args):
        raise ValueError('All arrays passed to unpad must have the same shape')
    if len(args) == 1:
        return args[0][unpad_slices]
    return tuple(arr[unpad_slices] for arr in args)
