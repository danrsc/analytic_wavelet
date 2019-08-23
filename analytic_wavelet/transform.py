import numpy as np
from scipy.fftpack import fft, ifft


__all__ = ['rotate', 'to_frequency_domain_wavelet', 'analytic_wavelet_transform', 'analytic_transform']


def _rotate_main_angle(x):
    x = np.mod(x, 2 * np.pi)
    return np.where(x > np.pi, x - 2 * np.pi, x)


def rotate(x):
    # Do we need this with numpy? Not sure if the same edge cases apply as in MATLAB
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


def analytic_transform(x, is_output_frequency=False):
    """
    Gives the analytic part of a signal
    Args:
        x: An array of shape (..., time)
        is_output_frequency: If True, the frequency domain signal is returned.
            Otherwise the time domain signal is returned.
    Returns:
        An array of the same shape as x
    """
    z = fft(np.where(np.isnan(x), 0, x))
    if np.isrealobj(x):
        z = 2 * z
    if x.shape[-1] // 2 * 2 == x.shape[-1]:  # even
        indices = np.arange(x.shape[-1] // 2 + 1, z.shape[-1])
        np.put_along_axis(z, np.reshape(indices, (1,) * (len(z.shape) - 1) + (-1,)), 0, -1)
        np.put_along_axis(
            z,
            np.reshape(np.array([x.shape[-1] - 1]), (1,) * (len(z.shape) - 1) + (-1,)),
            np.take(z, x.shape[-1] // 2, -1),
            -1)
    else:
        indices = np.arange((x.shape[-1] + 1) // 2, z.shape[-1])
        np.put_along_axis(z, np.reshape(indices, (1,) * (len(z.shape) - 1) + (-1,)), 0, -1)
    if not is_output_frequency:
        z = ifft(z)
        if z.shape[-1] != x.shape[-1]:
            z = np.take(z, np.arange(0, x.shape[-1]), -1)
    return z
