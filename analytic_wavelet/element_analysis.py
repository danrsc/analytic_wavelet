import inspect
import numpy as np
import scipy.special
from scipy.interpolate import interp1d

from .interpolate import quadratic_interpolate
from .generalized_morse_wavelet import GeneralizedMorseWavelet
from matplotlib.patches import Polygon


__all__ = ['ElementAnalysisMorse', 'maxima_of_transform', 'MaximaPValueInterp1d']


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
        if not np.ndim(self.analyzing_beta) == 0:
            raise ValueError('This function is only supported on scalar ElementAnalysisMorse')
        if not np.ndim(peak_fraction) == 0:
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


class MaximaPValueInterp1d:

    @staticmethod
    def from_histogram(histogram, bin_edges):
        """
        Create an instance of MaximaPValueInterp1d from the histogram, typically from the output of
        GeneralizedMorseWavelet.distribution_of_maxima_of_transformed_noise. Note that bin_edges should
        be normalized by the root-wavelet-spectrum of the noise, as in Eq. (4.11) in Lilly 2017. If these
        are the outputs of GeneralizedMorseWavelet.distribution_of_maxima_of_transformed_noise, the bin_edges
        are already normalized.
        Args:
            histogram: The values of each histogram bin
            bin_edges: The bin-edges for each bin

        Returns:
            An instance of MaximaPValueInterp1d.
        """
        cdf = np.cumsum(histogram)
        cdf = cdf / cdf[-1]
        p = 1 - cdf
        bin_centers = np.diff(bin_edges) / 2 + bin_edges[:-1]
        return MaximaPValueInterp1d(bin_centers, p)

    def __init__(self, normalized_maxima, p_value):
        """
        Wrapper around interp1d which returns 1 when the modulus is below the lower bound and 0 when the modulus is
        above the upper bound
        Args:
            normalized_maxima: The maxima we will be interpolating over. These should be normalized as in Eq. (4.11)
                in Lilly 2017. Typically the normalization is done by:
                    # Estimate the root-wavelet-spectrum of noise by assuming the highest-frequency wavelet transform
                    # only captures noise
                    w = analytic_wavelet_transform(...)
                    sigma_noise = np.sqrt(np.mean(np.square(np.abs(w[np.argmax(omega)]))))
                    # for white noise, sigma_scale_i ** 2 / sigma_scale_j ** 2 = omega_scale_i / omega_scale_j

                    # thus we have:
                    sigma_noise = np.sqrt(np.mean(np.square(np.abs(w[np.argmax(omega)]))))
                    w_tilde = w / sigma_noise * np.sqrt(omega / np.max(omega))
            p_value: The p_values of each normalized_maxima
        """
        self._min_value = np.min(normalized_maxima)
        self._interp = interp1d(normalized_maxima, p_value, bounds_error=False, fill_value=0)

    def __call__(self, normalized_maxima):
        """
        Wrapper around interp1d which returns 1 when normalized_maxima is below the lower bound of the maxima being
        interpolated over and 0 when normalized_maxima is above the upper bound of the maxima being interpolated over.
        Args:
            normalized_maxima: The maxima we will be computing p-values for. These should be normalized as in Eq. (4.11)
                in Lilly 2017. Typically the normalization is done by:
                    # Estimate the root-wavelet-spectrum of noise by assuming the highest-frequency wavelet transform
                    # only captures noise
                    w = analytic_wavelet_transform(...)
                    sigma_noise = np.sqrt(np.mean(np.square(np.abs(w[np.argmax(omega)]))))
                    # for white noise, sigma_scale_i ** 2 / sigma_scale_j ** 2 = omega_scale_i / omega_scale_j

                    # thus we have:
                    sigma_noise = np.sqrt(np.mean(np.square(np.abs(w[np.argmax(omega)]))))
                    w_tilde = w / sigma_noise * np.sqrt(omega / np.max(omega))
        Returns:
            p_value: The p_values for normalized_maxima
        """
        return np.clip(np.where(normalized_maxima < self._min_value, 1, self._interp(normalized_maxima)), 0, 1)
