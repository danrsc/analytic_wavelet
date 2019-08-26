import inspect
import numpy as np
import scipy.special
from scipy.interpolate import interp1d, interp2d

from .interpolate import quadratic_interpolate
from .generalized_morse_wavelet import GeneralizedMorseWavelet
# from .polygon import Polygon  # unfortunately no built-in for this
from matplotlib.patches import Polygon


__all__ = [
    'ElementAnalysisMorse',
    'maxima_of_transform',
    'distribution_of_maxima_of_transformed_noise',
    'ModulusPValueInterp1d',
    'ModulusPValueInterp2d']


# Primary paper:
# Lilly 2017
# "Element Analysis: a wavelet based method for analyzing time-localized events in noisy time-series"


class ElementAnalysisMorse:

    def __init__(self, gamma, analyzing_beta, element_beta, is_bandpass_normalized=True):
        self._analyzing_morse = GeneralizedMorseWavelet(gamma, analyzing_beta, is_bandpass_normalized)
        self._element_morse = GeneralizedMorseWavelet(gamma, element_beta, is_bandpass_normalized)

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
    def analyzing_morse(self):
        return self._analyzing_morse

    @property
    def element_morse(self):
        return self._element_morse

    @property
    def gamma(self):
        return self._analyzing_morse.gamma

    @property
    def analyzing_beta(self):
        return self._analyzing_morse.beta

    @property
    def element_beta(self):
        return self._element_morse.beta

    @property
    def is_bandpass_normalized(self):
        return self._analyzing_morse.is_bandpass_normalized

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
                     * (self.element_morse.peak_frequency() / self.analyzing_morse.peak_frequency())
                     * (self.analyzing_beta / (self.element_beta + 1)) ** (1 / self.gamma))
        else:
            f_hat = (event_scale_frequencies
                     * (self.element_morse.peak_frequency() / self.analyzing_morse.peak_frequency())
                     * ((self.analyzing_beta + 1 / 2) / (self.element_beta + 1 / 2)) ** (1 / self.gamma))
        c_hat = 2 * event_coefficients / self._maximum_analyzing_transform_of_element()
        return c_hat, self.element_morse.peak_frequency() / f_hat, f_hat

    def _normalized_scale_maximum(self):
        # \tilde{s}_{\beta, \mu, \gamma}^{max} in Lilly 2017. See Eq (3.9)
        if self.is_bandpass_normalized:
            return (self.analyzing_beta / (self.element_beta + 1)) ** (1 / self.gamma)
        else:
            return ((self.analyzing_beta + 1 / 2) / (self.element_beta + 1 / 2)) ** (1 / self.gamma)

    def _scale_weighting(self):
        # \vartheta_{\beta, \mu, \gamma} in Lilly 2017. See Eq (3.12)
        if self.is_bandpass_normalized:
            return ((self._normalized_scale_maximum() ** self.analyzing_beta)
                    / ((self._normalized_scale_maximum() ** self.gamma + 1)
                       ** ((self.analyzing_beta + self.element_beta + 1) / self.gamma)))
        else:
            return ((self._normalized_scale_maximum() ** (self.analyzing_beta + 1 / 2))
                    / ((self._normalized_scale_maximum() ** self.gamma + 1)
                       ** ((self.analyzing_beta + self.element_beta + 1) / self.gamma)))

    def _maximum_analyzing_transform_of_element(self):
        # \zeta_{\beta, \mu, \gamma}^{max} in Lilly 2017. See Eq. (3.11)
        return ((1 / (2 * np.pi * self.gamma))
                * self.analyzing_morse.amplitude()
                * self.element_morse.amplitude()
                * scipy.special.gamma((self.analyzing_beta + self.element_beta + 1) / self.gamma)
                * self._scale_weighting())

    def region_of_influence(self, event_scale, event_time, peak_fraction=0.5, num_samples=1000):
        """
        Finds the region of influence on the wavelet coefficients for events at a given scale analyzed using the
        which have been analyzed using this instance of ElementAnalysisMorse (i.e. using gamma, analyzing_beta, and
        element_beta)
        Args:
            event_scale: The scale of each event for which we are computing the influence region. Typically
                estimated using event_parameters(...) before calling this function
            event_time: The time coordinate of each event for which we are computing the influence region. Typically
                output from maxima_of_transform
            peak_fraction: Specifies which contour to return. The returned contour is the contour
                where the wavelet transform modulus has fallen off to this fraction of its peak value
            num_samples: How many samples of the contour should be returned.

        Returns:
            coordinates: If event_scale is scalar, then a 2d array of shape (2, num_samples) containing the
                frequency coordinates in coordinates[0] and time coordinates in coordinates[1]. If event_scale
                is not scalar, then the returned array is event_scale.shape + (2, num_samples)
        """
        # peak_fraction is \lambda in Lilly 2017. See p. 19 section 4(d) Regions of influence
        # event_scale is \rho

        # we could support arbitrary shapes here, returning an array of
        # self.beta.shape + self.peak_fraction.shape + self.event_scale.shape + (2, num_samples)
        # keeping morse parameters and peak_fraction scalar for now
        if not np.isscalar(self.analyzing_beta):
            raise ValueError('This function is only supported on scalar ElementAnalysisMorse')
        if not np.isscalar(peak_fraction):
            raise ValueError('This function only allows scalar values for peak_fraction')

        morse_sum_beta = GeneralizedMorseWavelet.replace(
            self.analyzing_morse, beta=self.analyzing_beta + self.element_beta)
        _, cumulants = morse_sum_beta.frequency_domain_cumulants(2)
        scale_weighting = self._scale_weighting()
        if self.is_bandpass_normalized:
            scale_low = (peak_fraction * scale_weighting) ** (1 / self.analyzing_beta)
            scale_high = (1 / (peak_fraction * scale_weighting)) ** (1 / (self.element_beta + 1))
            scale_0 = np.logspace(np.log10(scale_low), np.log10(scale_high), num_samples)
            x1 = self.analyzing_beta * np.log(scale_0)
        else:
            scale_low = (peak_fraction * scale_weighting) ** (1 / (self.analyzing_beta + 1 / 2))
            scale_high = (1 / (peak_fraction * scale_weighting)) ** (1 / (self.element_beta + 1 / 2))
            scale_0 = np.logspace(np.log10(scale_low), np.log10(scale_high), num_samples)
            x1 = (self.analyzing_beta + 1 / 2) * np.log(scale_0)

        fact = np.sqrt(2) * (((scale_0 ** self.gamma + 1) ** (1 / self.gamma)) / np.sqrt(cumulants[2]))
        x2 = ((self.analyzing_beta + self.element_beta + 1) / self.gamma) * np.log(scale_0 ** self.gamma + 1)
        ti = (fact * np.sqrt(-np.log(peak_fraction) - np.log(scale_weighting) + x1 - x2 + 1j * 0))
        scale_i = scale_0
        indicator_real = np.isreal(ti)
        ti = np.real(ti[indicator_real])
        scale_i = scale_i[indicator_real]
        scale_i = np.concatenate([scale_i[..., 0:1], scale_i, np.flip(scale_i, axis=-1)], axis=-1)
        ti = np.concatenate([ti[..., 0:1], -ti, np.flip(ti, axis=-1)], axis=-1)
        f_ti = interp1d(np.arange(ti.shape[-1]) / (ti.shape[-1] - 1), ti)
        t = f_ti(np.arange(num_samples) / (num_samples - 1))
        f_scale_i = interp1d(np.arange(scale_i.shape[-1]) / (scale_i.shape[-1] - 1), scale_i)
        scale = f_scale_i(np.arange(num_samples) / (num_samples - 1))

        rho = self.element_morse.peak_frequency() / event_scale
        t = np.expand_dims(t, 0) * np.expand_dims(rho, -1) + np.expand_dims(event_time, -1)
        omega = self.analyzing_morse.peak_frequency() / scale
        omega = np.expand_dims(omega, 0) / np.expand_dims(rho, -1)

        return np.concatenate([np.expand_dims(omega, -2), np.expand_dims(t, -2)], axis=-2)

    def isolated_maxima(
            self,
            indices_maxima,
            maxima_coefficients,
            maxima_scale_frequencies,
            influence_region_peak_fraction=0.5,
            influence_region_num_samples=1000,
            time_axis=-1):
        """
        Identifies maxima which are isolated. A maximum is isolated if no maximum of larger modulus exists
        within its region of influence.
        Args:
            indices_maxima: The coordinates of the maxima, as output by maxima_of_transform
            maxima_coefficients: The wavelet coefficients of the maxima, as output by maxima_of_transform
            maxima_scale_frequencies: The scale frequencies of the maxima, as output by maxima of transform
            influence_region_peak_fraction: Specifies which contour of an event's influence region to use to determine
                isolation. The contour used to define the region is the contour where the wavelet transform modulus
                has fallen off to this fraction of its peak value
            influence_region_num_samples: How many samples of the contour around the region of influence should be
                used to describe the region as a polygon.
            time_axis: Which axis in the coordinates is the time axis

        Returns:
            indicator_isolated: A 1-d array of boolean values, the same length as maxima_coefficients which
                has True for isolated maxima and False otherwise.
            influence_regions: A 3d array of influence regions for each event with shape
                (len(indicator_isolated), 2, influence_region_num_samples). influence_regions[i] gives the coordinates
                of the contour around event i. influence_regions[i][0] is the frequency coordinates and
                influence_regions[i][1] is the time coordinates.
        """
        # sort in order of descending magnitude
        indices_sort = np.argsort(-np.abs(maxima_coefficients))
        indices_maxima = tuple(ind[indices_sort] for ind in indices_maxima)
        maxima_coefficients = maxima_coefficients[indices_sort]
        maxima_scale_frequencies = maxima_scale_frequencies[indices_sort]
        c, rho, f_rho = self.event_parameters(maxima_coefficients, maxima_scale_frequencies)

        # the polygons for each maxima, shape (maxima, 2, region_points)
        influence_regions = self.region_of_influence(
            f_rho, indices_maxima[time_axis],
            peak_fraction=influence_region_peak_fraction, num_samples=influence_region_num_samples)

        # Polygon wants the coordinates as (region_points, 2)
        influence_regions = np.moveaxis(influence_regions, -2, -1)

        points = np.concatenate([
            np.expand_dims(f_rho, 1),
            np.expand_dims(indices_maxima[time_axis], 1)], axis=1)

        indicator_isolated = np.full(len(influence_regions), True)

        for idx, event_region_coordinates in enumerate(influence_regions):
            if idx == 0:
                continue
            # polygon = Polygon(event_region_coordinates[1], event_region_coordinates[0])
            # are there any larger coefficients in the region of influence of this event?

            # indicator_isolated[idx] = not np.any(polygon.is_inside(
            #     indices_maxima[time_axis][:idx], indices_maxima[freq_axis][:idx]) <= 0)

            polygon = Polygon(event_region_coordinates)
            indicator_isolated[idx] = not np.any(polygon.contains_points(points[:idx]))

        # restore to (num_maxima, 2, num_polygon_points)
        influence_regions = np.moveaxis(influence_regions, -1, -2)

        indices_sort = np.argsort(indices_sort)
        indicator_isolated = indicator_isolated[indices_sort]
        influence_regions = influence_regions[indices_sort]

        return indicator_isolated, influence_regions


def maxima_of_transform(x, scale_frequencies, min_modulus=None, freq_axis=-2, time_axis=-1):
    """
    Finds local maxima in the modulus of the wavelet coefficients jointly over both the time and frequency axes
        (i.e. each returned maximum is a maximum in both the time and frequency axes). Also uses quadratic interpolation
        in the frequency axis to better estimate the coefficient values and frequencies
    Args:
        x: The wavelet coefficients. Arbitrary shape, but having a frequency axis and time axis.
        scale_frequencies: The scale frequencies with which the wavelet transform was computed
        min_modulus: A threshold on the minimum value of the modulus for a maximum.
        freq_axis: Which axis in x is the frequency axis
        time_axis: Which axis in x is the time axis

    Returns:
        maxima_indices: A tuple giving the coordinates of the maxima, suitable for indexing. Each item in the tuple
            is a 1-d array where the length of the array is the number of maxima
        interpolated_coefficients: 1-d array of the interpolated values of the wavelet coefficients at the maxima.
        interpolated_scale_frequencies: A 1-d array of the interpolated values of the scale frequencies at the maxima.
    """
    w0 = x
    x += np.random.randn(*x.shape) * np.finfo(x.dtype).eps
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

    if min_modulus is not None:
        indicator = np.logical_and(indicator, x >= min_modulus)
    indices = np.nonzero(indicator)
    freq_indices = indices[freq_axis]
    indices_less_1 = indices[:freq_axis] + (freq_indices - 1,) + indices[freq_axis + 1:]
    indices_plus_1 = indices[:freq_axis] + (freq_indices + 1,) + indices[freq_axis + 1:]
    _, freq_hat = quadratic_interpolate(
        freq_indices - 1, freq_indices, freq_indices + 1,
        np.abs(w0[indices_less_1]), np.abs(w0[indices]), np.abs(w0[indices_plus_1]))
    interpolated = quadratic_interpolate(
        freq_indices - 1, freq_indices, freq_indices + 1,
        w0[indices_less_1], w0[indices], w0[indices_plus_1], freq_hat)
    scale_frequencies_interp = interp1d(np.arange(len(scale_frequencies)), scale_frequencies)
    return indices, interpolated, scale_frequencies_interp(freq_hat)


def _noise_covariance_entry(morse: GeneralizedMorseWavelet, alpha, time_shift, scale, scale_ratio):
    # covariance between the wavelet transform of noise and itself at another time and scale

    r_tilde = (1 + scale_ratio ** morse.gamma) ** (1 / morse.gamma)

    noise_morse = GeneralizedMorseWavelet.replace(
        morse, beta=2 * morse.beta - 2 * alpha, is_bandpass_normalized=True)

    fact1 = (morse.amplitude() ** 2 / noise_morse.amplitude())

    if morse.is_bandpass_normalized:
        fact2 = (((scale_ratio ** morse.beta) * (scale ** (2 * alpha - 1)))
                 / (r_tilde ** (2 * morse.beta - 2 * alpha + 1)))
    else:
        fact2 = (((scale_ratio ** (morse.beta + 1 / 2)) * (scale ** (2 * alpha)))
                 / (r_tilde ** (2 * morse.beta - 2 * alpha + 1)))

    assert(np.isscalar(time_shift))
    assert(scale.ndim < 2)

    psi = noise_morse.taylor_expansion_time_domain_wavelet(
        time_shift / (scale * r_tilde), noise_morse.peak_frequency())

    return fact1 * fact2 * np.conj(psi)


def _noise_covariance(morse, alpha, scale_ratio, s):
    # noise covariance of this point and the 4 adjacent points in the time/scale plane
    # this is the covariance structure given by Eq. (4.16) in Lilly 2017 assuming that we
    # group the points as:
    # [(t, s), (t + 1, s), (t - 1, s), (t, rs), (t, s / r)]
    sigma_0_0 = _noise_covariance_entry(morse, alpha, 0, s, 1)
    sigma = np.full(s.shape + (5, 5), np.nan, dtype=sigma_0_0.dtype)
    sigma[..., 0, 0] = sigma_0_0
    sigma[..., 0, 1] = _noise_covariance_entry(morse, alpha, 1, s, 1)
    sigma[..., 0, 2] = _noise_covariance_entry(morse, alpha, -1, s, 1)
    sigma[..., 0, 3] = _noise_covariance_entry(morse, alpha, 0, s, scale_ratio)
    sigma[..., 0, 4] = _noise_covariance_entry(morse, alpha, 0, s, 1 / scale_ratio)
    sigma[..., 1, 1] = sigma_0_0
    sigma[..., 1, 2] = _noise_covariance_entry(morse, alpha, -2, s, 1)
    sigma[..., 1, 3] = _noise_covariance_entry(morse, alpha, -1, s, scale_ratio)
    sigma[..., 1, 4] = _noise_covariance_entry(morse, alpha, -1, s, 1 / scale_ratio)
    sigma[..., 2, 2] = sigma_0_0
    sigma[..., 2, 3] = _noise_covariance_entry(morse, alpha, 1, s, scale_ratio)
    sigma[..., 2, 4] = _noise_covariance_entry(morse, alpha, 1, s, 1 / scale_ratio)
    sigma[..., 3, 3] = _noise_covariance_entry(morse, alpha, 0, scale_ratio * s, 1)
    sigma[..., 3, 4] = _noise_covariance_entry(
        morse, alpha, 0, scale_ratio * s, 1 / scale_ratio ** 2)
    sigma[..., 4, 4] = _noise_covariance_entry(morse, alpha, 0, s / scale_ratio, 1)

    for i in range(sigma.shape[-2]):
        for j in range(i):
            sigma[..., i, j] = np.conj(sigma[..., j, i])

    moment = morse.energy_moment(-2 * alpha)
    if morse.is_bandpass_normalized:
        sigma = sigma / np.reshape(moment * s ** (2 * alpha - 1), s.shape + (1, 1))
    else:
        sigma = sigma / np.reshape(moment * s ** (2 * alpha), s.shape + (1, 1))

    return sigma


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
        morse: GeneralizedMorseWavelet,
        spectral_slope,
        scale_frequencies,
        num_monte_carlo_realizations=1000,
        scale_ratio=None,
        should_extrapolate=False,
        make_p_value_func=False,
        **histogram_kwargs):
    """
    Returns a histogram of the modulus of the maxima of noise transformed by the analytic wavelet transform using
    the specified GeneralizedMorseWavelet. The transform is not actually run on the noise. Rather the method described
    in Lilly 2017 which uses the analytically determined covariance structure of noise after the transform is used.
    Args:
        morse: Which morse wavelet to estimate the distribution for. This must be a scalar instantiation of
            GeneralizedMorseWavelet
        spectral_slope: The slope of the noise. (alpha in Lilly 2017). 0 gives white noise. 1 gives red noise.
        scale_frequencies: Which scale frequencies to compute the distribution for. Either a scalar or a 1d array.
        num_monte_carlo_realizations: How many random samples to use for estimating the distribution.
        scale_ratio: This is the ratio of a scale to the next smaller scale in a planned run of element analysis.
            If set to None, then the scale_frequencies parameter is interpreted as a vector of ordered
            frequencies which differ from one to the next by a constant ratio. In that case, if either scale_frequencies
            is scalar or this ratio is found to not be constant, a ValueError is raised. If set to a value, then the
            scale_frequencies are interpreted as an unordered set of scale_frequencies with no relationship between
            each other.
        should_extrapolate: When True, only the simulation is only run for the maximum scale frequency, and the results
            of that run are adjusted analytically for the other scales. When False, the simulation is run for every
            scale frequency. Defaults to False.
        make_p_value_func: When True, returns a function which will estimate the p-value for new moduli.
            If scale_frequencies is scalar, the function takes moduli as its only argument and uses 1-d
            interpolation to estimate a p-value. If scale_frequencies is 1-d, the function takes moduli and
            scale_frequency as arguments, and uses 2-d interpolation to estimate a p-value. If this is set to True,
            then the returned histogram will be a normalized, equivalent to setting density=True in np.histogram.
            Defaults to False
        **histogram_kwargs: Arguments to np.histogram. Note that the 'weights' argument is not allowed here
            and a ValueError will be raised if it is used.
    Returns:
        hist: The binned moduli of the maxima similar to np.histogram. If scale_frequencies is not scalar, then
            this will be a list with hist[i] corresponding to scale_frequencies[i]. Depending on histogram_kwargs
            and other factors, the number of bins may not be consistent across scale_frequencies
        bin_edges: The edges of the bins of hist, similar to np.histogram. Note that bin_edges may vary from
            scale_frequency to scale_frequency. If scale_frequencies is not scalar, then this will be a list with
            bin_edges[i] corresponding to scale_frequencies[i]
        p_value_func: A function which takes either 1 or 2 arguments and returns p-values. Only returned when
            make_p_value_func is True. See make_p_value_func
    """

    if not np.isscalar(morse.beta):
        raise ValueError('This function is only supported on scalar instances of GeneralizedMorseWavelet')
    if not np.isscalar(spectral_slope):
        raise ValueError('spectral_slope must be scalar')
    scale_frequencies = np.asarray(scale_frequencies)
    if scale_frequencies.ndim > 1:
        raise ValueError('scale_frequencies must be at most 1d')
    if scale_ratio is None:
        if np.isscalar(scale_frequencies):
            raise ValueError('When scale_frequency_ratio is None, multiple scale_frequencies must be given')
        scale_frequency_ratios = scale_frequencies[:-1] / scale_frequencies[1:]
        if not np.allclose(scale_frequency_ratios, scale_frequency_ratios[0]):
            raise ValueError('When scale_frequency_ratio is None, the ratio must be constant'
                             ' between adjacent scale_frequencies. Ratios: {}'.format(scale_frequency_ratios))
        scale_ratio = scale_frequency_ratios[0]
    original_scale_frequencies = scale_frequencies
    if should_extrapolate:
        scale_frequencies = np.max(scale_frequencies)

    scale = morse.peak_frequency() / scale_frequencies

    if make_p_value_func is True:
        if 'normed' in histogram_kwargs:
            raise ValueError('make_p_value_func requires density=True. Please remove normed')
        if 'density' in histogram_kwargs:
            if not histogram_kwargs['density']:
                raise ValueError('make_p_value_func requires density=True')
        else:
            histogram_kwargs['density'] = True

    if 'weights' in histogram_kwargs:
        raise ValueError('weights is disallowed in histogram_kwargs')

    sigma = _noise_covariance(morse, spectral_slope, scale_ratio, scale)

    try:
        lower = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        if scale.ndim == 0:
            raise
        failure_scales = list()
        for i, item in enumerate(sigma):
            try:
                np.linalg.cholesky(item)
            except np.linalg.LinAlgError:
                failure_scales.append((i, scale_frequencies[i]))
        print('At large scales / low scale-frequencies, the noise covariance matrix may not be positive-definite due'
              ' to numerical issues. Consider using should_extrapolate=True or removing low scale-frequencies. '
              'Failed scales (index, scale) tuples: {}'.format(failure_scales))
        raise

    if not np.isscalar(scale_frequencies):
        # _get_noise_maxima_values can run on the whole thing at once if there is enough memory,
        # but we do one at a time to keep the memory footprint smaller
        result_hist = list()
        result_bins = list()
        for scale_lower in lower:
            _, maxima_values = _get_noise_maxima_values(scale_lower, num_monte_carlo_realizations)
            hist, bin_edges = np.histogram(maxima_values, **histogram_kwargs)
            print(np.sum(hist))
            print(np.cumsum(hist))
            result_hist.append(hist)
            result_bins.append(bin_edges)
    else:
        _, maxima_values = _get_noise_maxima_values(lower, num_monte_carlo_realizations)
        result_hist, result_bins = np.histogram(maxima_values, **histogram_kwargs)

    if should_extrapolate:
        h = list()
        b = list()
        for orig in original_scale_frequencies:
            h.append(result_hist * scale_frequencies / orig)
            b.append(np.copy(result_bins))
        result_hist = h
        result_bins = b

    if make_p_value_func:
        if np.isscalar(scale_frequencies):
            p = 1 - np.cumsum(result_hist)
            assert(np.isclose(p[-1], 0))
            bin_centers = np.diff(result_bins) / 2 + result_bins[:-1]
            p_value_func = ModulusPValueInterp1d(bin_centers, p)
        else:
            p = list()
            bin_centers = list()
            flat_scale = list()
            for h, b, scale in zip(result_hist, result_bins, scale_frequencies):
                current_p = 1 - np.cumsum(h)
                print('last p', current_p[-1])
                assert (np.isclose(current_p[-1], 0))
                p.extend(current_p)
                bin_centers.extend(np.diff(b) / 2 + b[:-1])
                flat_scale.extend([scale] * len(current_p))
            p_value_func = ModulusPValueInterp2d(np.array(bin_centers), np.array(flat_scale), np.array(p))

        return result_hist, result_bins, p_value_func

    return result_hist, result_bins


class ModulusPValueInterp1d:

    @staticmethod
    def from_histogram(histogram, bin_edges):
        sum_hist = np.sum(histogram)
        if not np.isclose(sum_hist, 1):  # convert to density
            db = np.array(np.diff(bin_edges), float)
            histogram = histogram / db / np.sum(histogram)
        p = 1 - np.cumsum(histogram)
        assert (np.isclose(p[-1], 0))
        bin_centers = np.diff(bin_edges) / 2 + bin_edges[:-1]
        return ModulusPValueInterp1d(bin_centers, p)

    def __init__(self, modulus, p_value):
        """
        Wrapper around interp1d which returns 1 when the modulus is below the lower bound and 0 when the modulus is
        above the upper bound
        Args:
            modulus: The domain for known p_values to interpolate over
            p_value: The p_values to interpolate
        """
        self._min_modulus = np.min(modulus)
        self._interp = interp1d(modulus, p_value, bounds_error=False, fill_value=0)

    def __call__(self, modulus):
        return np.clip(np.where(modulus < self._min_modulus, 1, self._interp(modulus)), 0, 1)


class ModulusPValueInterp2d:

    @staticmethod
    def from_histograms(histograms, bin_edges_per_histogram, y):
        if not isinstance(histograms, (list, tuple)) or not isinstance(bin_edges_per_histogram, (list, tuple)):
            raise ValueError('Expected histograms and bin_edges_per_histogram to be either lists or tuples')
        if len(histograms) != len(bin_edges_per_histogram):
            raise ValueError('Mismatched lengths between histograms ({}) and bin_edges_per_histogram ({})'.format(
                len(histograms), len(bin_edges_per_histogram)))
        if len(histograms) != len(y):
            raise ValueError('Mismatched lengths between histograms ({}) and y ({})'.format(
                len(histograms), len(y)))

        p = list()
        bin_centers = list()
        flat_y = list()
        for histogram, bin_edges, current_y in zip(histograms, bin_edges_per_histogram, y):
            sum_hist = np.sum(histogram)
            if not np.isclose(sum_hist, 1):  # convert to density
                db = np.array(np.diff(bin_edges), float)
                histogram = histogram / db / np.sum(histogram)
            current_p = 1 - np.cumsum(histogram)
            assert (np.isclose(current_p[-1], 0))
            p.extend(current_p)
            bin_centers.extend(np.diff(bin_edges) / 2 + bin_edges[:-1])
            flat_y.extend([current_y] * len(current_p))
        return ModulusPValueInterp2d(np.array(bin_centers), np.array(flat_y), np.array(p))

    def __init__(self, modulus, y, p_value):
        """
        A wrapper around interp2d which returns 1 when the modulus is below the lower bound and 0 when the modulus
        is above the upper bound.
        Args:
            modulus: The modulus coordinates of the training data
            y: The y-coordinates of the training data (typically the scale-frequencies)
            p_value: The p-values of the training data
        """
        # find the maximum of the minimum moduli - we will use this for determining when an out-of-bounds point
        # is too low or too high
        self._min_modulus = None
        self._min_y = np.min(y)
        self._max_y = np.max(y)
        for u in np.unique(y):
            indicator_u = y == u
            min_u_modulus = np.min(modulus[indicator_u])
            if self._min_modulus is None or min_u_modulus > self._min_modulus:
                self._min_modulus = min_u_modulus
        self._interp = interp2d(modulus, y, p_value, bounds_error=False)

    def __call__(self, modulus, y):
        result = self._interp(modulus, y)
        # out-of-bounds in modulus, but in bounds in y
        bad_modulus = np.logical_and(np.isnan(result), np.logical_and(y >= self._min_y, y <= self._max_y))
        result = np.where(np.logical_and(bad_modulus, modulus < self._min_modulus), 1, result)
        return np.clip(np.where(
            np.logical_and(bad_modulus, modulus > self._min_modulus), 0, result), 0, 1)
