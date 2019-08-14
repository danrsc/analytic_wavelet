import inspect
import numpy as np
import scipy.special
from scipy.fftpack import ifft, fft
from scipy.interpolate import interp1d
import warnings


__all__ = [
    'GeneralizedMorseWavelet',
    'rotate',
    'analytic_wavelet_transform',
    'transform_maxima',
    'quadratic_interpolate',
    'linear_interpolate']


class GeneralizedMorseWavelet:

    def __init__(self, gamma, beta, is_bandpass_normalized=True):
        self._gamma = gamma
        self._beta = beta
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
        if not np.isscalar(self.beta):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                result = np.exp((1 / self.gamma) * (np.log(self.beta) - np.log(self.gamma)))
            return np.where(self.beta == 0, np.power(np.log(2), 1 / self.gamma), result)
        elif self.beta == 0:
            return np.power(np.log(2), 1 / self.gamma)
        else:
            return np.exp((1 / self.gamma) * (np.log(self.beta) - np.log(self.gamma)))

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

    def log_spaced_frequencies(self, num_timepoints, high=None, low=None, density=4):
        if high is None:
            high = (0.1, np.pi)
        if low is None:
            low = (5, num_timepoints)
        if not np.isscalar(high):
            high = min(high[1], self._high_frequency_cutoff(high[0]))
        if not np.isscalar(low):
            min_low = low[2] if len(low) > 2 else 0
            low = max(min_low, self._low_frequency_cutoff(low[0], low[1]))
        r = 1 + (1 / density * self.time_domain_width())
        n = np.floor(np.log(high / low) / np.log(r))
        return high * np.ones(n + 1) / np.power(r, np.arange(n))

    def _high_frequency_cutoff(self, eta):
        omega_high = np.reshape(np.linspace(0, np.pi, 10000), (1,) * len(self.gamma.shape) + (-1,))
        peak_frequency = self.peak_frequency()
        if not np.isscalar(peak_frequency):
            peak_frequency = np.expand_dims(peak_frequency, -1)
        # shape_gamma + (10000,)
        omega = peak_frequency * np.pi / omega_high
        ln_psi_1 = (self.beta / self.gamma) * np.log((np.exp(1) * self.gamma) / self.beta)
        beta, gamma = self.beta, self.gamma
        if not np.isscalar(self.beta):
            beta, gamma = np.expand_dims(self.beta, -1), np.expand_dims(self.beta, -1)
        ln_psi_2 = beta * np.log(omega) - np.power(omega, gamma)
        ln_psi = ln_psi_1 + ln_psi_2
        indices = np.tile(
            np.reshape(
                np.arange(ln_psi.shape[-1]), (1,) * ln_psi.shape[:-1] + ln_psi.shape[-1]), ln_psi.shape[:-1] + (1,))
        indices = np.where(np.log(eta) - ln_psi < 0, indices, np.nan)
        indices = np.nanmin(indices, axis=-1)
        f = np.reshape(omega_high[np.reshape(indices, -1)], indices.shape)
        if np.isscalar(self.gamma):
            return f.item()
        return f

    def _low_frequency_cutoff(self, r, num_timepoints):
        return (2 * np.sqrt(2) * self.time_domain_width() * r) / num_timepoints

    def amplitude(self, k=1):
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
                * np.exp(scipy.special.gammaln(k) - scipy.special.gammaln(k + r - 1)),
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
        fo = self.peak_frequency()
        # (freq, 1)
        fact = np.expand_dims(np.abs(scale_frequencies) / fo, 1)
        # omega is (freq, time)
        omega = np.expand_dims(2 * np.pi * np.linspace(0, 1 - (1 / num_timepoints), num_timepoints), 0) / fact
        if self.is_bandpass_normalized:
            if self.beta == 0:
                psi_zero = np.exp(-np.power(omega, self.gamma))
            else:
                psi_zero = np.exp(self.beta * np.log(omega) - np.power(omega, self.gamma))
        else:
            if self.beta == 0:
                psi_zero = 2 * np.exp(-np.power(omega, self.gamma))
            else:
                psi_zero = 2 * np.exp(
                    -self.beta * np.log(fo)
                    + np.power(fo, self.gamma)
                    + np.power(self.beta, np.log(omega) - np.power(omega, self.gamma)))
        # psi_zero is (freq, time)
        psi_zero[:, 0] = 1 / 2 * psi_zero[:, 0]
        psi_zero = np.where(np.isnan(psi_zero), 0, psi_zero)
        psi_f = self._first_family(fact, num_timepoints, num_orthogonal_family_members, omega, psi_zero)
        psi_f = np.where(np.isinf(psi_f), 0, psi_f)
        omega = np.expand_dims(omega, 0)
        fact = np.expand_dims(fact, 0)
        psi = ifft(psi_f * rotate(omega * (num_timepoints + 1) / 2 * fact))

        indicator_neg_sf = scale_frequencies < 0

        psi[:, indicator_neg_sf, :] = np.conj(psi[:, indicator_neg_sf, :])
        psi_f[:, indicator_neg_sf, 1:] = np.flip(psi_f[:, indicator_neg_sf, 1:], axis=2)

        if num_orthogonal_family_members == 1:
            psi = np.squeeze(psi, axis=0)
            psi_f = np.squeeze(psi_f, axis=0)

        return psi, psi_f

    def _first_family(self, fact, num_timepoints, num_orthogonal_family_members, omega, psi_zero):
        r = (2 * self.beta + 1) / self.gamma
        c = r - 1
        # (freq, time)
        ell = np.zeros_like(omega)
        indices = np.arange(int(np.ceil(num_timepoints / 2)))
        # (wavelet, freq, time)
        psi_f = np.zeros((num_orthogonal_family_members, psi_zero.shape[0], psi_zero.shape[1]), dtype=psi_zero.dtype)
        for k in range(num_orthogonal_family_members):
            if self.is_bandpass_normalized:
                if self.beta == 0:
                    coeff = 1
                else:
                    coeff = np.sqrt(
                        np.exp(scipy.special.gammaln(r) + scipy.special.gammaln(k + 1) - scipy.special.gammaln(k + r)))
            else:
                # (freq, time)
                coeff = np.sqrt(1 / fact) * self.amplitude(k + 1)
            ell[indices] = GeneralizedMorseWavelet._laguerre(2 * np.power(omega[:, indices], self.gamma), k, c)
            psi_f[k] = coeff * psi_zero * ell
        return psi_f

    @staticmethod
    def _laguerre(x, k, c):  # probably could replace by scipy.special.genlaguerre
        y = np.zeros_like(x)
        for m in range(k + 1):
            fact = np.exp(
                scipy.special.gammaln(k + c + 1) - scipy.special.gammaln(c + m + 1) - scipy.special.gammaln(k - m + 1))
            y += np.power(-1, m) * fact * np.power(x, m) / scipy.special.gamma(m + 1)
        return y

    def zeta(self, mu):
        """
        The maximum of the Morse wavelet transform of another wavelet
        """
        if self.is_bandpass_normalized:
            numerator = (self.beta / (mu + 1)) ** (self.beta / self.gamma)
            denominator = ((self.beta / (mu + 1)) + 1) ** ((self.beta + mu + 1) / self.gamma)
        else:
            numerator = ((self.beta + 1 / 2) / (mu + 1 / 2)) ** ((self.beta + 1 / 2) / self.gamma)
            denominator = ((self.beta + 1 / 2) / (mu + 1 / 2) + 1) ** ((self.beta + mu + 1) / self.gamma)
        mu_morse = GeneralizedMorseWavelet.replace(self, beta=mu)
        return ((1 / (2 * np.pi * self.gamma))
                * self.amplitude()
                * mu_morse.amplitude()
                * scipy.special.gamma((self.beta + mu + 1) / self.gamma)
                * (numerator / denominator))

    def maxima_parameters(self, maxima_values, maxima_scale_frequencies, mu):
        """
        Estimates the parameters of the events corresponding to the maxima
        Args:
            maxima_values: output from transform_maxima
            maxima_scale_frequencies: output from transform_maxima
            mu: ?

        Returns:
            coefficients (complex), scales, scale_frequencies
        """
        mu_morse = GeneralizedMorseWavelet.replace(self, beta=mu)
        if self.is_bandpass_normalized:
            f_hat = (maxima_scale_frequencies
                     * (mu_morse.peak_frequency() / self.peak_frequency())
                     * (self.beta / (mu + 1)) ** (1 / self.gamma))
        else:
            f_hat = (maxima_scale_frequencies
                     * (mu_morse.peak_frequency() / self.peak_frequency())
                     * ((self.beta + 1 / 2) / (mu + 1 / 2)) ** (1 / self.gamma))
        c_hat = 2 * maxima_values / self.zeta(mu)
        return c_hat, mu_morse.peak_frequency() / f_hat, f_hat


def _rotate_main_angle(x):
    x = np.mod(x, 2 * np.pi)
    return np.where(x > np.pi, x - 2 * np.pi, x)


def rotate(x):
    x = _rotate_main_angle(x)
    edge_cases = np.full_like(x, np.nan)
    edge_cases[x == np.pi / 2] = 1j
    edge_cases[x == -np.pi / 2] = -1j
    edge_cases[np.logical_or(x == np.pi, x == -np.pi)] = -1
    edge_cases[np.logical_or(x == 0, np.logical_or(x == 2 * np.pi, x == -2 * np.pi))] = 1
    return np.where(np.isnan(edge_cases), np.exp(1j * x), edge_cases)


def frequency_domain_from_time_domain(num_timepoints, time_domain_wavelet):
    if time_domain_wavelet.shape[-1] < num_timepoints:
        w = np.zeros(time_domain_wavelet.shape[:-1] + (num_timepoints,), time_domain_wavelet.dtype)
        indices = np.arange(time_domain_wavelet.shape[-1]) + (num_timepoints - time_domain_wavelet.shape[-1]) // 2
        w[..., indices] = time_domain_wavelet
        time_domain_wavelet = w
    elif time_domain_wavelet.shape[-1] > num_timepoints:
        time_domain_wavelet = time_domain_wavelet[..., :num_timepoints.shape[1]]
        raise ValueError('unexpected')  # original code multiplies by nan

    psi_f = fft(time_domain_wavelet)
    omega = 2 * np.pi * np.linspace(0, 1 - (1 / num_timepoints), num_timepoints)
    omega = np.reshape(omega, (1,) * (len(time_domain_wavelet.shape) - 1) + (omega.shape[0],))
    if num_timepoints // 2 * 2 == num_timepoints:  # is even
        psi_f = psi_f * rotate(-omega * (num_timepoints + 1) / 2) * np.sign(np.pi - omega)
    else:
        psi_f = psi_f * rotate(-omega * (num_timepoints + 1) / 2)
    return psi_f


def _awt(x, psi_f, is_time_domain_wavelet_real):
    # unitary transform normalization
    if not np.isreal(x):
        x = x / np.sqrt(2)

    psi_f = np.conj(psi_f)
    # -> (batch, k * scale_frequencies, time)
    result = np.expand_dims(fft(x), 1) * np.reshape(psi_f, (1, -1, psi_f.shape[-1]))
    # -> (batch, k, scale_frequencies, time) or (batch, scale_frequencies, time)
    result = np.reshape(result, (x.shape[0],) + psi_f.shape)
    result = ifft(result)
    if np.isreal(x) and is_time_domain_wavelet_real and not np.isreal(result):
        result = np.real(result)
    if not np.any(np.isfinite(result)):
        if not np.isreal(result):
            result = np.inf * (1 + 1j) * np.ones_like(result)
        else:
            result = np.inf * np.ones_like(result)
    if not np.isreal(result):
        result = np.where(np.isnan(x), np.nan * (1 + 1j), result)
    else:
        result = np.where(np.isnan(x), np.nan, result)
    return result


def analytic_wavelet_transform(x, frequency_domain_wavelet, is_time_domain_wavelet_real):
    x = np.asarray(x)
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
        result = np.inf * (1 + 1j) * np.zeros((x.shape[0],) + result_good.shape[1:], dtype=result_good.dtype)
        result[indicator_good] = result_good
    result = np.reshape(result, x_shape[:-1] + result.shape[1:])
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
