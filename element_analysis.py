import numpy as np
from scipy.interpolate import interp1d


from .analytic_wavelet import GeneralizedMorseWavelet, analytic_wavelet_transform


__all__ = ['transform_maxima', 'element_parameters']


def element_parameters(x, gamma, beta):
    morse = GeneralizedMorseWavelet(gamma, beta)
    fs = morse.log_spaced_frequencies(x.shape[-1])
    _, psi_f = morse.make_wavelet(x.shape[-1], fs)
    x = analytic_wavelet_transform(x, psi_f, False)
    maxima_indices, maxima_values, maxima_fs = transform_maxima(fs, x)


def transform_maxima(fs, w, min_amplitude=None, freq_axis=-2, time_axis=-1):
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
    _, freq_hat = _quadratic_interpolate(
        freq_indices - 1, freq_indices, freq_indices + 1,
        np.abs(w0[indices_less_1]), np.abs(w0[indices]), np.abs(w0[indices_plus_1]))
    interpolated = _quadratic_interpolate(
        freq_indices - 1, freq_indices, freq_indices + 1,
        w0[indices_less_1], w0[indices], w0[indices_plus_1], freq_hat)
    f = interp1d(np.arange(len(fs)), fs)
    return indices, interpolated, f(freq_hat)


def _quadratic_interpolate(t1, t2, t3, x1, x2, x3, t=None):
    numerator = x1 * (t2 - t3) + x2 * (t3 - t1) + x3 * (t1 - t2)
    denominator = (t1 - t2) * (t1 - t3) * (t2 - t3)
    a = numerator / denominator

    numerator = x1 * (t2 ** 2 - t3 ** 2) + x2 * (t3 ** 2 - t1 ** 2) + x3 * (t1 ** 2 - t2 ** 2)
    b = -numerator / denominator

    numerator = x1 * t2 *t3 * (t2 - t3) + x2 * t3 * t1 * (t3 - t1) + x3 * t1 *t2 * (t1 - t2)
    c = numerator / denominator

    return_t = t is None
    if t is None:
        t = -b / 2 * a

    x = a * t ** 2 + b * t + c
    if return_t:
        return x, t
    return x
