import numpy as np
import scipy.special
from scipy.interpolate import interp1d, interp2d

from .interpolate import quadratic_interpolate
from .generalized_morse_wavelet import GeneralizedMorseWavelet
from .transform import analytic_wavelet_transform
from .polygon import Polygon  # unfortunately no built-in for this


__all__ = [
    'ElementMorse',
    'maxima_of_transform',
    'distribution_of_maxima_of_transformed_noise',
    'AmplitudePValueInterp1d',
    'AmplitudePValueInterp2d']


# Primary paper:
# Lilly 2017
# "Element Analysis: a wavelet based method for analyzing time-localized events in noisy time-series"


class ElementMorse(GeneralizedMorseWavelet):

    def __init__(self, analyzing_gamma, analyzing_beta, element_beta, is_bandpass_normalized=True):
        super().__init__(analyzing_gamma, analyzing_beta, is_bandpass_normalized)
        self._element_morse = GeneralizedMorseWavelet(analyzing_gamma, element_beta, is_bandpass_normalized)

    @property
    def element_morse(self):
        return self._element_morse

    @property
    def element_beta(self):
        return self._element_morse.beta

    def event_parameters(self, event_coefficients, event_scale_frequencies):
        """
        Estimates the parameters of the events corresponding to the maxima
        Args:
            event_coefficients: Coefficient values, output from maxima_of_transform
            event_scale_frequencies: output from maxima_of_transform

        Returns:
            coefficients (complex), scales, scale_frequencies
        """
        if self.is_bandpass_normalized:
            f_hat = (event_scale_frequencies
                     * (self.element_morse.peak_frequency() / self.peak_frequency())
                     * (self.beta / (self.element_beta + 1)) ** (1 / self.gamma))
        else:
            f_hat = (event_scale_frequencies
                     * (self.element_morse.peak_frequency() / self.peak_frequency())
                     * ((self.beta + 1 / 2) / (self.element_beta + 1 / 2)) ** (1 / self.gamma))
        c_hat = 2 * event_coefficients / self._maximum_analyzing_transform_of_element()
        return c_hat, self.element_morse.peak_frequency() / f_hat, f_hat

    def _normalized_scale_maximum(self):
        # \tilde{s}_{\beta, \mu, \gamma}^{max} in Lilly 2017. See Eq (3.9)
        if self.is_bandpass_normalized:
            return (self.beta / (self.element_beta + 1)) ** (1 / self.gamma)
        else:
            return ((self.beta + 1 / 2) / (self.element_beta + 1 / 2)) ** (1 / self.gamma)

    def _scale_weighting(self):
        # \vartheta_{\beta, \mu, \gamma} in Lilly 2017. See Eq (3.12)
        if self.is_bandpass_normalized:
            return ((self._normalized_scale_maximum() ** self.beta)
                    / ((self._normalized_scale_maximum() ** self.gamma + 1)
                       ** ((self.beta + self.element_beta + 1) / self.gamma)))
        else:
            return ((self._normalized_scale_maximum() ** (self.beta + 1 / 2))
                    / ((self._normalized_scale_maximum() ** self.gamma + 1)
                       ** ((self.beta + self.element_beta + 1) / self.gamma)))

    def _maximum_analyzing_transform_of_element(self):
        # \zeta_{\beta, \mu, \gamma}^{max} in Lilly 2017. See Eq. (3.11)
        return ((1 / (2 * np.pi * self.gamma))
                * self.amplitude()
                * self.element_morse.amplitude()
                * scipy.special.gamma((self.beta + self.element_beta + 1) / self.gamma)
                * self._scale_weighting())

    def region_of_influence(self, peak_fraction, event_scale, num_samples=1000):
        """
        Finds the region of influence for an event on the wavelet coefficients for
        an event at a given scale analyzed using the parameters in this instance of ElementMorse (i.e.
        gamma, beta, and element_beta)
        Args:
            peak_fraction: Specifies which contour to return. The returned contour is the contour
                where the wavelet transform modulus has fallen off to this fraction of its peak value
            event_scale: The scale of the event for which we are computing the influence
            num_samples: How many samples of the contour should be returned.

        Returns:
            scale: A 1d array of num_samples giving the scale coordinates of the contour of the region
            time: A 1d array of num_samples giving the time coordinates of the contour of the region
        """
        # peak_fraction is \lambda in Lilly 2017. See p. 19 section 4(d) Regions of influence
        # event_scale is \rho

        # we could support arbitrary shapes here, returning an array of
        # self.beta.shape + self.peak_fraction.shape + self.event_scale.shape
        # keeping it scalar for now
        if not np.isscalar(self.beta):
            raise ValueError('This function is only supported on scalar ElementMorse')
        if not np.isscalar(peak_fraction):
            raise ValueError('This function only allows scalar values for peak_fraction')
        if not np.isscalar(event_scale):
            raise ValueError('This function only allows scalar values for event_scale')

        morse_sum_beta = GeneralizedMorseWavelet.replace(self, beta=self.beta + self.element_beta)
        _, cumulants = morse_sum_beta.frequency_domain_cumulants(2)
        scale_weighting = self._scale_weighting()
        if self.is_bandpass_normalized:
            scale_low = (peak_fraction * scale_weighting) ** (1 / self.beta)
            scale_high = (1 / (peak_fraction * scale_weighting)) ** (1 / (self.element_beta + 1))
            scale_0 = np.logspace(np.log10(scale_low), np.log10(scale_high), num_samples)
            x1 = self.beta * np.log(scale_0)
        else:
            scale_low = (peak_fraction * scale_weighting) ** (1 / (self.beta + 1 / 2))
            scale_high = (1 / (peak_fraction * scale_weighting)) ** (1 / (self.element_beta + 1 / 2))
            scale_0 = np.logspace(np.log10(scale_low), np.log10(scale_high), num_samples)
            x1 = (self.beta + 1 / 2) * np.log(scale_0)

        fact = np.sqrt(2) * (((scale_0 ** self.gamma + 1) ** (1 / self.gamma)) / np.sqrt(cumulants[2]))
        x2 = ((self.beta + self.element_beta + 1) / self.gamma) * np.log(scale_0 ** self.gamma + 1)
        ti = (fact * np.sqrt(-np.log(peak_fraction) - np.log(scale_weighting) + x1 - x2))
        scale_i = scale_0
        indicator_real = np.real(ti)
        ti = ti[indicator_real]
        scale_i = scale_i[indicator_real]
        scale_i = np.concatenate([scale_i[..., 0:1], scale_i, np.flip(scale_i, axis=-1)], axis=-1)
        ti = np.concatenate([ti[..., 0:1], -ti, np.flip(ti, axis=-1)], axis=-1)
        f_ti = interp1d(np.arange(ti.shape[-1]) / (ti.shape[-1] - 1), ti)
        t = f_ti(np.arange(num_samples) / (num_samples - 1))
        f_scale_i = interp1d(np.arange(scale_i.shape[-1] / (scale_i.shape[-1] - 1)), scale_i)
        scale = f_scale_i(np.arange(num_samples) / (num_samples - 1))

        rho = self.element_morse.peak_frequency() / event_scale
        t = t * rho
        omega = self.peak_frequency() / (scale * rho)
        return omega, t

    def isolated_maxima(
            self,
            indices_maxima,
            maxima_coefficients,
            maxima_scale_frequencies,
            influence_region,
            freq_axis=-2,
            time_axis=-1):
        """
        Identifies maxima which are isolated. A maximum is isolated if no maximum of larger amplitude exists
        within its region of influence.
        Args:
            indices_maxima: The coordinates of the maxima, as output by maxima_of_transform
            maxima_coefficients: The wavelet coefficients of the maxima, as output by maxima_of_transform
            maxima_scale_frequencies: The scale frequencies of the maxima, as output by maxima of transform
            influence_region: A tuple of (freq_indices, time_indices) which describes coordinates of a contour of a
                region of influence for the morse wavelet and the assumed event-scale. Typically computed by a call
                to region_of_influence.
            freq_axis: Which axis in the coordinates is the frequency axis
            time_axis: Which axis in the coordinates is the time axis

        Returns:
            indicator_isolated: A 1-d array of boolean values, the same length as maxima_coefficients which
                has True for isolated maxima and False otherwise.
            influence_regions: A tuple of (freq_region, time_region). freq_region is shape
                (num_maxima, num_contour_points), and time_region is the same shape. These describe the contours
                of the region of influence for each maximum (isolated or not). If, e.g., time is on the x-axis
                and frequency is on the y-axis in a plot, we can plot the first maximum's region of influence by
                using fill(influence_regions[1][0], influence_regions[0][0]). The first index gives us either the
                time or freq coordinates, and the second index choose which maximum.
        """
        # sort in order of descending magnitude
        indices_sort = np.argsort(-np.abs(maxima_coefficients))
        indices_maxima = tuple(ind[indices_sort] for ind in indices_maxima)
        maxima_coefficients = maxima_coefficients[indices_sort]
        maxima_scale_frequencies = maxima_scale_frequencies[indices_sort]
        c, rho, f_rho = self.event_parameters(maxima_coefficients, maxima_scale_frequencies)

        freq_region, time_region = influence_region

        # the polygons for each maxima, shape (maxima, region_points)
        time_region = np.expand_dims(time_region, 0) / np.expand_dims(indices_maxima[freq_axis] / f_rho, 1)
        time_region = time_region + np.expand_dims(indices_maxima[time_axis], 1)
        freq_region = np.expand_dims(freq_region, 0) * np.expand_dims(indices_maxima[freq_axis] * f_rho, 1)

        indicator_isolated = np.full(len(time_region), True)
        for idx, (event_region_freq_coordinates, event_region_time_coordinates) in enumerate(zip(
                freq_region, time_region)):
            if idx == 0:
                continue
            polygon = Polygon(event_region_time_coordinates, event_region_freq_coordinates)
            # are there any larger coefficients in the region of influence of this event?
            indicator_isolated[idx] = not np.any(polygon.is_inside(
                indices_maxima[time_axis][:idx], indices_maxima[freq_axis][:idx]) >= 0)

        indices_sort = np.arange(len(indices_sort))[indices_sort]
        indicator_isolated = indicator_isolated[indices_sort]
        time_region = time_region[indices_sort]
        freq_region = freq_region[indices_sort]

        return indicator_isolated, (freq_region, time_region)


def maxima_of_transform(x, scale_frequencies, min_amplitude=None, freq_axis=-2, time_axis=-1):
    """
    Finds local maxima in the wavelet coefficients jointly over both the time and frequency axes (i.e. each
        returned maximum is a maximum in both the time and frequency axes). Also uses quadratic interpolation
        in the frequency axis to better estimate the coefficient values and frequencies
    Args:
        x: The wavelet coefficients. Arbitrary shape, but having a frequency axis and time axis.
        scale_frequencies: The scale frequencies with which the wavelet transform was computed
        min_amplitude: A threshold on the minimum amplitude for a maximum.
        freq_axis: Which axis in x is the frequency axis
        time_axis: Which axis in x is the time axis

    Returns:
        maxima_indices: A tuple giving the coordinates of the maxima, suitable for indexing. Each item in the tuple
            is a 1-d array where the length of the array is the number of maxima
        interpolated_coefficients: 1-d array of the interpolated values of the wavelet coefficients at the maxima.
        interpolated_scale_frequencies: A 1-d array of the interpolated values of the scale frequencies at the maxima.
    """
    w0 = x
    x += np.random.randn(x.shape) * np.finfo(x.dtype).eps
    x = np.abs(x)

    # local maxima on time and freq axes
    indicator = np.logical_not(x == 0)
    indicator = np.logical_and(indicator, x > np.roll(x, 1, axis=time_axis))
    indicator = np.logical_and(indicator, x > np.roll(x, -1, axis=time_axis))
    indicator = np.logical_and(indicator, x > np.roll(x, 1, axis=freq_axis))
    indicator = np.logical_and(indicator, x > np.roll(x, -1, axis=freq_axis))

    slices = [slice(None)] * len(x.shape)

    # remove start/end points on time axis
    slices[time_axis] = 0
    indicator[tuple(slices)] = False
    slices[time_axis] = -1
    indicator[tuple(slices)] = False
    slices[time_axis] = slice(None)

    # remove start/end points on freq axis
    slices[freq_axis] = 0
    indicator[tuple(slices)] = False
    slices[freq_axis] = -1
    indicator[tuple(slices)] = False

    if min_amplitude is not None:
        indicator = np.logical_and(indicator, x >= min_amplitude)
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


def _noise_covariance(morse: GeneralizedMorseWavelet, alpha, time_shift, scale, scale_ratio):
    # covariance between the wavelet transform of noise and itself at another time and scale
    r_tilde = (1 + scale_ratio ** morse.gamma) ** (1 / morse.gamma)
    # %Note:  use the input normalization in the numerator, and the *amplitude*
    # %normalization in the denominator.  All the latter does is cancel the
    # %coefficient coming from MORSEXPAND, which assumes the amplitude normalization

    noise_morse = GeneralizedMorseWavelet.replace(morse, beta=2 * morse.beta - 2 * alpha)

    fact1 = (morse.amplitude() ** 2 / noise_morse.amplitude())
    if morse.is_bandpass_normalized:
        fact2 = (((scale_ratio ** morse.beta) * (scale ** (2 * alpha - 1)))
                 / (r_tilde ** (2 * morse.beta - 2 * alpha + 1)))
    else:
        fact2 = (((scale_ratio ** (morse.beta + 1 / 2)) * (scale ** (2 * alpha)))
                 / (r_tilde ** (2 * morse.beta - 2 * alpha + 1)))

    peak = noise_morse.peak_frequency()
    assert(np.isscalar(time_shift))
    assert(scale.ndim < 2)

    scale_r_tilde = scale * r_tilde

    if scale_r_tilde.ndim == 1:
        psi = list(
            [noise_morse.taylor_expansion_time_domain_wavelet(time_shift / item, peak) for item in scale_r_tilde])
        psi = np.array(psi)
    else:
        psi = noise_morse.taylor_expansion_time_domain_wavelet(time_shift / scale_r_tilde, peak)

    return fact1 * fact2 * np.conj(psi)


def distribution_of_maxima_of_transformed_noise(
        morse: GeneralizedMorseWavelet,
        spectral_slope,
        scale_frequencies,
        num_monte_carlo_realizations=1000,
        scale_frequency_ratio=None,
        should_extrapolate=False,
        should_make_p_value_func=False,
        **histogram_kwargs):
    """
    Returns a histogram of the amplitudes of the maxima of noise transformed by the analytic wavelet transform using
    the specified GeneralizedMorseWavelet. The transform is not actually run on the noise. Rather the method described
    in Lilly 2017 which uses the analytically determined covariance structure of noise after the transform is used.
    Args:
        morse: Which morse wavelet to estimate the distribution for. This must be a scalar instantiation of
            GeneralizedMorseWavelet
        spectral_slope: The slope of the noise. (alpha in Lilly 2017). 0 gives white noise. 1 gives red noise.
        scale_frequencies: Which scale frequencies to compute the distribution for. Either a scalar or a 1d array.
        num_monte_carlo_realizations: How many random samples to use for estimating the distribution.
        scale_frequency_ratio: If None, then the scale_frequencies parameter is interpreted as a vector of ordered
            frequencies which differ from one to the next by a constant ratio. In that case, if either scale_frequencies
            is scalar or this ratio is found to not be constant, a ValueError is raised. If provided, then the
            scale_frequencies are interpreted as an unordered set of scale_frequencies with no relationship between
            each other. Then this ratio gives how the scales would be stepped in a hypothetical future run of
            element analysis
        should_extrapolate: When True, only the simulation is only run for the maximum scale frequency, and the results
            of that run are adjusted analytically for the other scales. When False, the simulation is run for every
            scale frequency. Defaults to False.
        should_make_p_value_func: When True, returns a function which will estimate the p-value for new amplitudes.
            If scale_frequencies is scalar, the function takes amplitude as its only argument and uses 1-d
            interpolation to estimate a p-value. If scale_frequencies is 1-d, the function takes amplitude and
            scale_frequency as arguments, and uses 2-d interpolation to estimate a p-value. If this is set to True,
            then the returned histogram will be a normalized, equivalent to setting density=True in np.histogram.
            Defaults to False
        **histogram_kwargs: Arguments to np.histogram. Note that the 'weights' argument is not allowed here
            and a ValueError will be raised if it is used.
    Returns:
        hist: The binned amplitudes of the maxima similar to np.histogram. If scale_frequencies is not scalar, then
            this will be a list with hist[i] corresponding to scale_frequencies[i]. Depending on histogram_kwargs
            and other factors, the number of bins may not be consistent across scale_frequencies
        bin_edges: The edges of the bins of hist, similar to np.histogram. Note that bin_edges may vary from
            scale_frequency to scale_frequency. If scale_frequencies is not scalar, then this will be a list with
            bin_edges[i] corresponding to scale_frequencies[i]
        p_value_func: A function which takes either 1 or 2 arguments and returns p-values. Only returned when
            should_make_p_value_func is True. See should_make_p_value_func
    """

    if not np.isscalar(morse.beta):
        raise ValueError('This function is only supported on scalar instances of GeneralizedMorseWavelet')
    if not np.isscalar(spectral_slope):
        raise ValueError('spectral_slope must be scalar')
    scale_frequencies = np.asarray(scale_frequencies)
    if scale_frequencies.ndim > 1:
        raise ValueError('scale_frequencies must be at most 1d')
    if scale_frequency_ratio is None:
        if np.isscalar(scale_frequencies):
            raise ValueError('When scale_frequency_ratio is None, multiple scale_frequencies must be given')
        scale_frequency_ratios = scale_frequencies / np.roll(scale_frequencies, -1)
        scale_frequency_ratios = scale_frequency_ratios[1:]
        if not np.allclose(scale_frequency_ratios, scale_frequency_ratios[0]):
            raise ValueError('When scale_frequency_ratio is None, the ratio must be constant'
                             ' between adjacent scale_frequencies')
        scale_frequency_ratio = scale_frequency_ratios[0]
    original_scale_frequencies = scale_frequencies
    if should_extrapolate:
        scale_frequencies = np.max(scale_frequencies)
    s = morse.peak_frequency() / scale_frequencies

    if should_make_p_value_func is True:
        if 'normed' in histogram_kwargs:
            raise ValueError('make_p_value_func requires density=True. Please remove normed')
        if 'density' in histogram_kwargs:
            if not histogram_kwargs['density']:
                raise ValueError('make_p_value_func requires a density=True')
        else:
            histogram_kwargs['density'] = True

    # normed is the deprecated version of density
    is_density_hist = 'density' in histogram_kwargs or 'normed' in histogram_kwargs
    if 'weights' in histogram_kwargs:
        raise ValueError('weights is disallowed in histogram_kwargs')

    # noise covariance of this point and the 4 adjacent points in the time/scale plane
    # this is the covariance structure given by Eq. (4.16) in Lilly 2017 assuming that we
    # group the points as:
    # [(t, s), (t + 1, s), (t - 1, s), (t, rs), (t, s / r)]
    sigma_0_0 = _noise_covariance(morse, spectral_slope, 0, s, 1)
    sigma = np.full(s.shape + (5, 5), np.nan, dtype=sigma_0_0.dtype)
    sigma[..., 0, 0] = sigma_0_0
    sigma[..., 0, 1] = _noise_covariance(morse, spectral_slope, 1, s, 1)
    sigma[..., 0, 2] = _noise_covariance(morse, spectral_slope, -1, s, 1)
    sigma[..., 0, 3] = _noise_covariance(morse, spectral_slope, 0, s, scale_frequency_ratio)
    sigma[..., 0, 4] = _noise_covariance(morse, spectral_slope, 0, s, 1 / scale_frequency_ratio)
    sigma[..., 1, 1] = _noise_covariance(morse, spectral_slope, 0, s, 1)
    sigma[..., 1, 2] = _noise_covariance(morse, spectral_slope, -2, s, 1)
    sigma[..., 1, 3] = _noise_covariance(morse, spectral_slope, -1, s, scale_frequency_ratio)
    sigma[..., 1, 4] = _noise_covariance(morse, spectral_slope, -1, s, 1 / scale_frequency_ratio)
    sigma[..., 2, 2] = _noise_covariance(morse, spectral_slope, 0, s, 1)
    sigma[..., 2, 3] = _noise_covariance(morse, spectral_slope, 1, s, scale_frequency_ratio)
    sigma[..., 2, 4] = _noise_covariance(morse, spectral_slope, 1, s, 1 / scale_frequency_ratio)
    sigma[..., 3, 3] = _noise_covariance(morse, spectral_slope, 0, scale_frequency_ratio * s, 1)
    sigma[..., 3, 4] = _noise_covariance(
        morse, spectral_slope, 0, scale_frequency_ratio * s, 1 / scale_frequency_ratio ** 2)
    sigma[..., 4, 4] = _noise_covariance(morse, spectral_slope, 0, s / scale_frequency_ratio, 1)

    for i in range(sigma.shape[-2]):
        for j in range(i):
            sigma[..., i, j] = np.conj(sigma[..., j, i])

    lower = np.linalg.cholesky(sigma)

    # may need to split this into batches for memory ...
    noise_shape = (5, num_monte_carlo_realizations)
    if not np.isscalar(scale_frequencies):
        noise_shape = scale_frequencies.shape + noise_shape
    noise = (np.random.randn(*noise_shape) + 1j * np.random.randn(*noise_shape)) / np.sqrt(2)

    noise = np.abs(lower * noise)

    # the unshifted and unscaled noise is the maximum
    indicator_maxima = np.argmax(noise, axis=-2) == 0

    if not np.isscalar(scale_frequencies):
        result_hist = list()
        result_bins = list()
        for n, m in zip(noise, indicator_maxima):
            hist, bin_edges = np.histogram(n[m], **histogram_kwargs)
            if not is_density_hist:
                hist = hist / num_monte_carlo_realizations
            result_hist.append(hist)
            result_bins.append(bin_edges)
    else:
        result_hist, result_bins = np.histogram(noise[indicator_maxima], **histogram_kwargs)
        if not is_density_hist:
            result_hist = result_hist / num_monte_carlo_realizations

    if should_extrapolate:
        h = list()
        b = list()
        for orig in original_scale_frequencies:
            h.append(result_hist * scale_frequencies / orig)
            b.append(np.copy(result_bins))
        result_hist = h
        result_bins = b

    if should_make_p_value_func:
        if np.isscalar(scale_frequencies):
            p = 1 - np.cumsum(result_hist)
            assert(np.isclose(p[-1], 0))
            bin_centers = np.diff(result_bins) / 2 + result_bins[:-1]
            p_value_func = AmplitudePValueInterp1d(bin_centers, p)
        else:
            p = list()
            bin_centers = list()
            flat_scale = list()
            for h, b, s in zip(result_hist, result_bins, scale_frequencies):
                current_p = 1 - np.cumsum(h)
                assert (np.isclose(current_p[-1], 0))
                p.extend(current_p)
                bin_centers.extend(np.diff(b) / 2 + b[:-1])
                flat_scale.extend([s] * len(current_p))
            p_value_func = AmplitudePValueInterp2d(np.array(bin_centers), np.array(flat_scale), np.array(p))

        return result_hist, result_bins, p_value_func

    return result_hist, result_bins


class AmplitudePValueInterp1d:

    def __init__(self, amplitudes, p_values):
        """
        Wrapper around interp1d which returns 1 when the amplitude is below the lower bound and 0 when the amplitude is
        above the upper bound
        Args:
            amplitudes: The domain for known p_values to interpolate over
            p_values: The p_values to interpolate
        """
        self._min_amplitude = np.min(amplitudes)
        self._interp = interp1d(amplitudes, p_values, bounds_error=False, fill_value=0)

    def __call__(self, amplitude):
        return np.clip(np.where(amplitude < self._min_amplitude, 1, self._interp(amplitude)), 0, 1)


class AmplitudePValueInterp2d:

    def __init__(self, amplitudes, y, p_values):
        """
        A wrapper around interp2d which returns 1 when the amplitude is below the lower bound and 0 when the amplitude
        is above the upper bound.
        Args:
            amplitudes: The amplitude coordinates of the training data
            y: The y-coordinates of the training data (typically the scale-frequencies)
            p_values: The p-values of the training data
        """
        # find the maximum of the minimum amplitudes - we will use this for determining when an out-of-bounds point
        # is too low or too high
        self._min_amplitude = None
        self._min_y = np.min(y)
        self._max_y = np.max(y)
        for u in np.unique(y):
            indicator_u = y == u
            min_u_amplitude = np.min(amplitudes[indicator_u])
            if self._min_amplitude is None or min_u_amplitude > self._min_amplitude:
                self._min_amplitude = min_u_amplitude
        self._interp = interp2d(amplitudes, y, p_values, bounds_error=False)

    def __call__(self, amplitude, y):
        result = self._interp(amplitude, y)
        # out-of-bounds in amplitude, but in bounds in y
        bad_amplitude = np.logical_and(np.isnan(result), np.logical_and(y >= self._min_y, y <= self._max_y))
        result = np.where(np.logical_and(bad_amplitude, amplitude < self._min_amplitude), 1, result)
        return np.clip(np.where(
            np.logical_and(bad_amplitude, amplitude > self._min_amplitude), 0, result), 0, 1)
