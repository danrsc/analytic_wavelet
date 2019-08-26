import numpy as np
from scipy.interpolate import interp1d, interp2d

from .generalized_morse_wavelet import GeneralizedMorseWavelet


__all__ = ['distribution_of_maxima_of_transformed_noise']


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
