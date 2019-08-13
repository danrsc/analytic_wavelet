import numpy as np
from scipy.special import binom


__all__ = [
    'first_central_diff',
    'amplitude',
    'instantaneous_frequency',
    'instantaneous_bandwidth',
    'instantaneous_curvature',
    'instantaneous_moments']


def _expand_from_1d(arr, ndim, axis):
    return np.reshape(np.asarray(arr), (1,) * axis + (len(arr),) + (1,) * (ndim - axis - 1))


def _central_diff_part(dest, indices, src, axis):
    np.put_along_axis(
        dest,
        # shift the indices up by 1 to leave 0 open
        _expand_from_1d(indices[:-1] + 1, dest.ndim, axis),
        np.diff(np.take(src, indices, axis)) / 2,
        axis)


def first_central_diff(x, padding='endpoint', axis=-1):
    i1 = np.arange(0, x.shape[axis], 2)
    i2 = i1 + 1
    if i2[-1] == x.shape[axis]:
        i2 = i2[:-1]
    y = np.zeros_like(x)
    _central_diff_part(y, i1, x, axis)
    _central_diff_part(y, i2, x, axis)
    if padding == 'endpoint':
        dx_start = np.take(x, 1, axis) - np.take(x, 0, axis)
        dx_end = np.take(x, -1, axis) - np.take(x, -2, axis)
    elif padding == 'nan':
        dx_start = np.nan
        dx_end = np.nan
    elif padding == 'periodic':
        dx_start = (np.take(x, 1, axis) - np.take(x, 0, axis)) / 2 + (np.take(x, 0, axis) - np.take(x, -1, axis)) / 2
        dx_end = (np.take(x, -1, axis) - np.take(x, -2, axis)) / 2 + (np.take(x, 0, axis) - np.take(x, -1, axis)) / 2
    else:
        raise ValueError('Uknown padding type: {}'.format(padding))
    np.put_along_axis(y, _expand_from_1d([0], y.ndim, axis), dx_start, axis)
    np.put_along_axis(y, _expand_from_1d([y.shape[axis] - 1], y.ndim, axis), dx_end, axis)
    return y


def _bell_polynomial(*x):
    m = [1]
    for n in range(1, len(x) + 1):
        m.append(0)
        for p in range(n):
            m[n] = m[n] + binom(n - 1, p) * x[n - p - 1] * m[p]
    m = m[1:]
    return m


def amplitude(x, variable_axis=None):
    result = np.abs(x)
    if variable_axis is not None:
        return np.sqrt(np.nanmean(np.square(result), axis=variable_axis))
    return result


def instantaneous_frequency(x, dt=1, diff_padding='endpoint', variable_axis=None):
    result = first_central_diff(np.unwrap(np.angle(x), axis=-1), padding=diff_padding) / dt
    if variable_axis is not None:
        weights = np.square(x)
        result = np.nansum(result * weights, axis=variable_axis) / np.nansum(weights, axis=variable_axis)
    return result


def instantaneous_bandwidth(x, dt=1, diff_padding='endpoint', variable_axis=None):
    result = first_central_diff(np.log(np.abs(x)), padding=diff_padding) / dt
    if variable_axis is not None:
        frequency = instantaneous_frequency(x, dt=dt, diff_padding=diff_padding)
        multi_frequency = instantaneous_frequency(x, dt=dt, diff_padding=diff_padding, variable_axis=variable_axis)
        multi_frequency = np.expand_dims(multi_frequency, variable_axis)
        weights = np.square(x)
        result = (np.nansum(np.abs(result + 1j * (frequency - multi_frequency)) ** 2 * weights, axis=variable_axis)
                  / np.nansum(weights, axis=variable_axis))
    return result


def instantaneous_curvature(x, dt=1, diff_padding='endpoint', variable_axis=None):
    moments = instantaneous_moments(x, max_order=3, dt=dt, diff_padding=diff_padding)
    if variable_axis is not None:
        _, frequency, bandwidth, curvature = moments
        multi_frequency = instantaneous_frequency(x, dt=dt, diff_padding=diff_padding, variable_axis=variable_axis)
        np.expand_dims(multi_frequency, variable_axis)
        weights = np.square(x)
        temp = np.abs(
            curvature + 2 * 1j * bandwidth * (frequency - multi_frequency) - (frequency - multi_frequency) ** 2) ** 2
        return np.sqrt(np.nansum(temp * weights, variable_axis) / np.nansum(weights))
    return moments[3]


def instantaneous_moments(x, max_order=0, dt=1, diff_padding='endpoint', variable_axis=None):
    if max_order < 0:
        raise ValueError('max_order < 0: {}'.format(max_order))
    result = [amplitude(x)]
    if max_order > 0:
        result.append(instantaneous_frequency(x, dt=dt, diff_padding=diff_padding, variable_axis=variable_axis))
    if max_order > 1:
        result.append(instantaneous_bandwidth(x, dt=dt, diff_padding=diff_padding, variable_axis=variable_axis))
    if max_order > 2:
        if variable_axis is not None:
            result.append(instantaneous_curvature(x, dt=dt, diff_padding=diff_padding, variable_axis=variable_axis))
            if max_order > 3:
                raise ValueError('max_order cannot be greater than 3 for multivariate moments')
            return result
        eta_diff = first_central_diff(result[1] - 1j * result[2], padding=diff_padding) / dt
        poly_args = [result[2]]
        for current_order in range(2, max_order + 1):
            poly_args.append(1j * eta_diff)
            eta_diff = first_central_diff(eta_diff, padding=diff_padding) / dt
        assert(len(poly_args) == max_order)
        result.extend(_bell_polynomial(*poly_args))
        assert(len(result) == max_order + 1)
    return result
