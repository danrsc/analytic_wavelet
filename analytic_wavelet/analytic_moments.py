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
    if axis < 0:
        axis = ndim + axis
    return np.reshape(np.asarray(arr), (1,) * axis + (len(arr),) + (1,) * (ndim - axis - 1))


# like np.average, but ignores nan
def _nanaverage(arr, axis=None, weights=None, returned=False, keepdims=False):
    summed_weights = np.nansum(weights, axis=axis, keepdims=keepdims)
    result = np.nansum(arr * weights, axis=axis, keepdims=keepdims) / summed_weights
    if returned:
        return result, summed_weights
    return result


def first_central_diff(x, padding='endpoint', axis=-1):
    edge_order = 1 if (padding == 'endpoint' or padding == 'nan') else 2 if padding == 'periodic' else None
    if edge_order is None:
        raise ValueError('Unknown padding type: {}'.format(padding))
    x = np.gradient(x, edge_order=edge_order, axis=axis)
    if padding == 'nan':
        np.put_along_axis(x, _expand_from_1d([0], x.ndim, axis), np.nan, axis)
        np.put_along_axis(x, _expand_from_1d([x.shape[axis] - 1], x.ndim, axis), np.nan, axis)
    return x


def _bell_polynomial(*x):
    m = [1]
    for n in range(1, len(x) + 1):
        m.append(0)
        for p in range(n):
            m[n] = m[n] + binom(n - 1, p) * x[n - p - 1] * m[p]
    m = m[1:]
    return m


def amplitude(x, variable_axis=None, keepdims=False):
    """
    The amplitude of the analytic signal x. If variable-axis is specified, x is treated as multivariate with its
        components along variable_axis.
    Args:
        x: An array with shape (..., time)
        variable_axis: If specified, this axis is treated as the components of a multivariate x
        keepdims: If True, the variable_axis will be retained (and of size 1). Ignored if variable_axis is None.
    Returns:
        The amplitude of the signal. If variable_axis is specified and keepdims is False, the result will be
            shape x.shape[:variable_axis] + x.shape[variable_axis + 1:]
    """
    result = np.abs(x)
    if variable_axis is not None:
        return np.sqrt(np.nanmean(np.abs(result) ** 2, axis=variable_axis, keepdims=keepdims))
    return result


def instantaneous_frequency(x, dt=1, diff_padding='endpoint', variable_axis=None, keepdims=False):
    """
    The instantaneous frequency of the analytic signal x. If variable-axis is specified, x is treated as
        multivariate with its components along variable_axis.
    Args:
        x: An array with shape (..., time)
        dt: The difference between steps on the time axis
        diff_padding: How edges are handled when computing the gradient. Choices are:
            endpoint: At the edges, the gradient is computed between adjacent time steps instead of skipping time steps.
            periodic: At the edges, the gradient is computed as though time is periodic
            nan: The edges are set to nan
        variable_axis: If specified, this axis is treated as the components of a multivariate x
        keepdims: If True, the variable_axis will be retained (and of size 1). Ignored if variable_axis is None.
    Returns:
        The instantaneous frequency of the signal. If variable_axis is specified and keepdims is False,
            the result will be shape x.shape[:variable_axis] + x.shape[variable_axis + 1:]
    """
    result = first_central_diff(np.unwrap(np.angle(x), axis=-1), padding=diff_padding) / dt
    if variable_axis is not None:
        return _nanaverage(result, axis=variable_axis, weights=np.square(np.abs(x)), keepdims=keepdims)
    return result


def instantaneous_bandwidth(x, dt=1, diff_padding='endpoint', variable_axis=None, keepdims=False):
    """
    The instantaneous bandwidth of the analytic signal x. If variable-axis is specified, x is treated as
        multivariate with its components along variable_axis.
    Args:
        x: An array with shape (..., time)
        dt: The difference between steps on the time axis
        diff_padding: How edges are handled when computing the gradient. Choices are:
            endpoint: At the edges, the gradient is computed between adjacent time steps instead of skipping time steps.
            periodic: At the edges, the gradient is computed as though time is periodic
            nan: The edges are set to nan
        variable_axis: If specified, this axis is treated as the components of a multivariate x
        keepdims: If True, the variable_axis will be retained (and of size 1). Ignored if variable_axis is None.
    Returns:
        The instantaneous bandwidth of the signal. If variable_axis is specified and keepdims is False,
            the result will be shape x.shape[:variable_axis] + x.shape[variable_axis + 1:]
    """
    result = first_central_diff(np.log(np.abs(x)), padding=diff_padding) / dt
    if variable_axis is not None:
        frequency = instantaneous_frequency(x, dt=dt, diff_padding=diff_padding)
        multi_frequency = instantaneous_frequency(
            x, dt=dt, diff_padding=diff_padding, variable_axis=variable_axis, keepdims=True)
        result = np.abs(result + 1j * (frequency - multi_frequency)) ** 2
        return np.sqrt(_nanaverage(result, axis=variable_axis, weights=np.square(np.abs(x)), keepdims=keepdims))
    return result


def instantaneous_curvature(x, dt=1, diff_padding='endpoint', variable_axis=None, keepdims=False):
    """
    The instantaneous curvature of the analytic signal x. If variable-axis is specified, x is treated as
        multivariate with its components along variable_axis.
    Args:
        x: An array with shape (..., time)
        dt: The difference between steps on the time axis
        diff_padding: How edges are handled when computing the gradient. Choices are:
            endpoint: At the edges, the gradient is computed between adjacent time steps instead of skipping time steps.
            periodic: At the edges, the gradient is computed as though time is periodic
            nan: The edges are set to nan
        variable_axis: If specified, this axis is treated as the components of a multivariate x
        keepdims: If True, the variable_axis will be retained (and of size 1). Ignored if variable_axis is None.
    Returns:
        The instantaneous curvature of the signal. If variable_axis is specified and keepdims is False,
            the result will be shape x.shape[:variable_axis] + x.shape[variable_axis + 1:]
    """
    moments = instantaneous_moments(x, max_order=3, dt=dt, diff_padding=diff_padding)
    if variable_axis is not None:
        _, frequency, bandwidth, curvature = moments
        multi_frequency = instantaneous_frequency(
            x, dt=dt, diff_padding=diff_padding, variable_axis=variable_axis, keepdims=True)
        temp = curvature + 2 * 1j * bandwidth * (frequency - multi_frequency) - (frequency - multi_frequency) ** 2
        return np.sqrt(
            _nanaverage(np.abs(temp) ** 2, axis=variable_axis, weights=np.square(np.abs(x)), keepdims=keepdims))
    return moments[3]


def instantaneous_moments(x, max_order=0, dt=1, diff_padding='endpoint', variable_axis=None, keepdims=False):
    """
    The instantaneous moments of the analytic signal x up to and including the max_order moment. If variable-axis
        is specified, x is treated as multivariate with its components along variable_axis. max_order must be no
        more than 3 if variable_axis is given (other moments are not defined). The first 4 moments (up to max order 3)
        are the same as calling amplitude, instantaneous_frequency, instantaneous_bandwidth, and
        instantaneous_curvature respectively.
    Args:
        x: An array with shape (..., time)
        max_order: The moments up to and including max_order are returned. Order 0 is the amplitude, order 1 is the
            instantaneous_frequency, order 2 is the instantaneous_bandwidth, and order 3 is the instantaneous_curvature.
            Additional moments can also be computed if variable_axis is None, but max_order must be no more than 3 when
            variable_axis is given.
        dt: The difference between steps on the time axis
        diff_padding: How edges are handled when computing the gradient. Choices are:
            endpoint: At the edges, the gradient is computed between adjacent time steps instead of skipping time steps.
            periodic: At the edges, the gradient is computed as though time is periodic
            nan: The edges are set to nan
        variable_axis: If specified, this axis is treated as the components of a multivariate x. max_order must be no
            more than 3 if variable_axis is given.
        keepdims: If True, the variable_axis will be retained (and of size 1). Ignored if variable_axis is None.
    Returns:
        The first (max_order + 1) instantaneous moments of the signal. If variable_axis is specified and keepdims is
            False, each moment will be shape x.shape[:variable_axis] + x.shape[variable_axis + 1:]
    """

    if max_order < 0:
        raise ValueError('max_order < 0: {}'.format(max_order))
    result = [amplitude(x, variable_axis=variable_axis, keepdims=keepdims)]
    if max_order > 0:
        result.append(instantaneous_frequency(
            x, dt=dt, diff_padding=diff_padding, variable_axis=variable_axis, keepdims=keepdims))
    if max_order > 1:
        result.append(instantaneous_bandwidth(
            x, dt=dt, diff_padding=diff_padding, variable_axis=variable_axis, keepdims=keepdims))
    if max_order > 2:
        if variable_axis is not None:
            result.append(instantaneous_curvature(
                x, dt=dt, diff_padding=diff_padding, variable_axis=variable_axis, keepdims=keepdims))
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
