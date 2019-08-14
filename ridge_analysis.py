import numpy as np
from .analytic_wavelet_transform_moments import instantaneous_frequency, amplitude, first_central_diff
from .analytic_wavelet import rotate, quadratic_interpolate, linear_interpolate, GeneralizedMorseWavelet

__all__ = ['ridge_walk']


def ridge_walk(
        x,
        scale_frequencies,
        dt=1,
        ridge_kind='amplitude',
        morse_wavelet=None,
        min_wavelet_lengths_in_ridge=None,
        trim_half_wavelet_lengths=None,
        frequency_max=None,
        frequency_min=None,
        mask=None,
        alpha=1/4,
        scale_axis=-2,
        time_axis=-1,
        variable_axis=None):
    """
    Finds ridges in x, the output of analytic_wavelet_transform. If variable_axis is not specified, treats any
    extra axes of x as a batch. If variable_axis is specified, then the ridge is computed jointly over the variables
    in that axis, and other axes are treated as a batch.
    Args:
        x: An array of shape (..., scale, time). If variable_axis is not specified, treats any
            extra axes of x as a batch. If variable_axis is specified, then the ridge is computed jointly over the
            variables in that axis, and other axes are treated as a batch.
        scale_frequencies: The scale frequencies with which the transform was computed
        dt: Sample rate used to compute instantaneous frequency along the ridge
        ridge_kind: Whether to use 'amplitude' or 'phase' ridges
        morse_wavelet: The wavelet used to compute the analytic wavelet transform. Only used in combination with
            min_wavelet_lengths_in_ridge and trim_half_wavelet_lengths.
        min_wavelet_lengths_in_ridge: A ridge with less than this many wavelet lengths is ignored
        trim_half_wavelet_lengths: Trims this many half wavelets from each edge since these are contaminated by edge
            effects. A value of 1 is recommended, but if provided requires morse_wavelet.
        frequency_max: Ridge points greater than this frequency will not be considered. Can be an array broadcasting to
            the shape of x but excluding the scale, time, and (if exists) variable axes
        frequency_min: Ridge points lower than this frequency will not be considered. Can be an array broadcasting to
            the shape of x but excluding the scale, time, and (if exists) variable axes
        mask: A boolean array which broadcasts to the shape of x, without the variable axis. If a time-scale point
            is False in the mask it will not be considered for the ridge. This enables the use of auxiliary
            information such as noise estimates at particular time-scale locations
        alpha: Controls aggressiveness of chaining across scales. alpha == (d_omega / omega) where omega is the
            transform frequency and d_omega is the difference between the frequency predicted for the next point
            based on the transform at a 'tail', and the actual frequency at prospective 'heads'. This mostly does not
            need to change, but for strongly chirping or weakly chirping noisy signals better performance may be
            obtained by adjusting it
        variable_axis: If specified, then the ridge is computed jointly over this axis

    Returns:
        ridge: The transform values along the ridge.
        indices: A tuple of indices such that ridge = x[indices]
        instantaneous_frequencies: Instantaneous frequencies along the ridge
        instantaneous_bandwidth: Instantaneous bandwidth along the ridge
        normalized_instantaneous_curvature: Normalized instantaneous curvature along the ridge as a measure of error.
            Should be << 1 if the ridge estimate is good. Only returned if morse_wavelet is provided
    """

    if variable_axis is not None:
        x = np.moveaxis(x, (variable_axis, scale_axis, time_axis), (-3, -2, -1))
        orig_shape = x.shape
        x = np.reshape(x, (-1,) + x.shape[-3:])
        if mask is not None:
            mask_scale_axis = scale_axis if scale_axis < variable_axis else scale_axis - 1
            mask_time_axis = time_axis if time_axis < variable_axis else time_axis - 1
            mask = np.moveaxis(mask, (mask_scale_axis, mask_time_axis), (-2, -1))
            mask = np.reshape(mask, (-1,) + mask.shape[-2:])
    else:
        x = np.moveaxis(x, (scale_axis, time_axis), (-2, -1))
        orig_shape = x.shape
        x = np.reshape(x, (-1,) + x.shape[-2:])
        if mask is not None:
            mask = np.moveaxis(mask, (scale_axis, time_axis), (-2, -1))
            mask = np.reshape(mask, (-1,) + mask.shape[-2:])

    indicator_ridge, ridge_quantity, x, instantaneous_frequencies = _indicator_ridge(
        x, scale_frequencies, ridge_kind, 0, frequency_min, frequency_max, mask)

    min_periods = None
    if min_wavelet_lengths_in_ridge is not None:
        if morse_wavelet is None:
            raise ValueError('If min_wavelet_lengths in ridge is given, morse_wavelet must be specified')
        min_periods = 2 * morse_wavelet.time_domain_width() / np.pi

    _chain_ridge_points(
        x, scale_frequencies, indicator_ridge, ridge_quantity, instantaneous_frequencies, min_periods, alpha, mask)


def _indicator_ridge(x, scale_frequencies, ridge_kind, min_amplitude, frequency_min, frequency_max, mask):
    variable_axis = None

    frequency = instantaneous_frequency(x, variable_axis=variable_axis)

    if len(x.shape) == 4:
        variable_axis = 1
    if ridge_kind == 'amplitude':
        ridge_quantity = amplitude(x, variable_axis=variable_axis)
    elif ridge_kind == 'phase':
        ridge_quantity = frequency - np.reshape(scale_frequencies, (1, len(scale_frequencies), 1))
    else:
        raise ValueError('Unknown ridge_kind: {}'.format(ridge_kind))

    if variable_axis is not None:
        phase_average = np.sum(np.abs(x) * x, axis=variable_axis) / np.sum(np.abs(x)**2, axis=variable_axis)
        x = np.sqrt(np.sum(np.abs(x)**2, axis=variable_axis)) * rotate(np.angle(phase_average))

    if ridge_kind == 'amplitude':
        result = np.logical_and(
            ridge_quantity >= np.roll(ridge_quantity, 1, axis=-2),
            ridge_quantity >= np.roll(ridge_quantity, -1, axis=-2))
    elif ridge_kind == 'phase':
        rqf = np.roll(ridge_quantity, 1, axis=-2)
        rqb = np.roll(ridge_quantity, -1, axis=-2)
        # d/ds < 0
        result = np.logical_or(np.logical_and(rqb < 0, rqf >= 0), np.logical_and(rqb <= 0, rqf > 0))
        del rqf
        del rqb
    else:
        raise ValueError('Unknown ridge_kind: {}'.format(ridge_kind))

    abs_ridge_quantity = np.abs(ridge_quantity)

    # remove minima
    result = np.logical_and(result, np.logical_not(
        np.logical_and(
            np.logical_and(result, np.roll(result, -1, axis=-2)),
            abs_ridge_quantity > np.roll(abs_ridge_quantity, -1, axis=-2))))
    result = np.logical_and(result, np.logical_not(
        np.logical_and(
            np.logical_and(result, np.roll(result, 1, axis=-2)),
            abs_ridge_quantity > np.roll(abs_ridge_quantity, 1, axis=-2))))

    result = np.logical_and(result, np.logical_not(np.isnan(x)))
    result = np.logical_and(result, np.abs(x) >= min_amplitude)

    # remove edges
    shape = [1] * len(result.shape)
    shape[-1] = 2
    np.put_along_axis(result, np.reshape(np.array([0, result.shape[-2]]), shape), False, -2)

    if frequency_min is not None:
        if not np.isscalar(frequency_min):
            frequency_min = np.reshape(frequency_min, (result.shape[0],) + (1,) * (len(result.shape) - 1))
        result = np.logical_and(result, frequency > frequency_min)
    if frequency_max is not None:
        if not np.isscalar(frequency_max):
            frequency_max = np.reshape(frequency_max, (result.shape[0],) + (1,) * (len(result.shape) - 1))
        result = np.logical_and(result, frequency < frequency_max)

    if mask is not None:
        result = np.logical_and(result, mask)

    return result, ridge_quantity, x, frequency


def _chain_ridge_points(
        x, scale_frequencies, indicator_points, ridge_quantity, instantaneous_frequencies, min_periods, alpha, mask):

    batch_indices, scale_indices, time_indices = np.nonzero(indicator_points)
    df_dt = first_central_diff(instantaneous_frequencies, axis=-1)
    instantaneous_frequencies, df_dt, x = _ridge_interpolate(
        [instantaneous_frequencies, df_dt, x], (batch_indices, scale_indices, time_indices), ridge_quantity)
    ridge_sf = scale_frequencies[scale_indices]
    fr_next = instantaneous_frequencies + df_dt
    fr_prev = instantaneous_frequencies - df_dt

    # compress the scale axis to only enough for the total number of simultaneous ridge points
    ridge_indices = np.cumsum(indicator_points, axis=1)
    max_simultaneous_ridge_points = np.max(ridge_indices) + 1

    # this converts ridge_indices into a 1d array which is now suitable for replacing
    # scale_indices
    ridge_indices = ridge_indices[(batch_indices, scale_indices, time_indices)]

    instantaneous_frequencies_ridge = _compress(
        instantaneous_frequencies,
        max_simultaneous_ridge_points, batch_indices, time_indices, scale_indices, ridge_indices)
    fr_next_ridge = _compress(
        fr_next,
        max_simultaneous_ridge_points, batch_indices, time_indices, scale_indices, ridge_indices)
    fr_prev_ridge = _compress(
        fr_prev,
        max_simultaneous_ridge_points, batch_indices, time_indices, scale_indices, ridge_indices)

    # shift on time axis, then compare to predictions (expand_dims effectively takes the transpose)
    df1 = np.expand_dims(np.roll(instantaneous_frequencies_ridge, -1, 1), 3) - np.expand_dims(fr_next_ridge, 2)
    df1 = df1 / np.expand_dims(instantaneous_frequencies_ridge, 3)

    df2 = np.expand_dims(np.roll(fr_prev_ridge, -1, 1), 3) - np.expand_dims(instantaneous_frequencies_ridge, 2)
    df2 = df2 / np.expand_dims(instantaneous_frequencies_ridge, 3)

    df = (np.abs(df1) + np.abs(df2)) / 2

    df[df > alpha] = np.nan

    next_ridge_indices = np.nanargmin(df, axis=3)[(batch_indices, time_indices, ridge_indices)]
    next_ridge_indices = np.where(np.isnan(next_ridge_indices), -1, next_ridge_indices).astype(int)

    ridge_ids = np.full(df.shape, -1, dtype=int)
    ridge_ids[(batch_indices, time_indices, ridge_indices)] = np.arange(len(batch_indices))
    indicator_valid_next = next_ridge_indices >= 0
    valid_next_batch_indices = batch_indices[indicator_valid_next]
    valid_next_time_indices = time_indices[indicator_valid_next]
    valid_next_ridge_indices = ridge_indices[indicator_valid_next]
    valid_next_next_indices = next_ridge_indices[indicator_valid_next]
    last_ridge_ids = None
    # propagate the ids forward along links
    while True:
        if last_ridge_ids is not None and np.array_equal(last_ridge_ids, ridge_ids):
            break
        last_ridge_ids = np.copy(ridge_ids)
        ridge_ids[(valid_next_batch_indices, valid_next_time_indices, valid_next_next_indices)] = \
            ridge_ids[(valid_next_batch_indices, valid_next_time_indices, valid_next_ridge_indices)]


def _compress(x, max_simultaneous_ridge_points, batch_indices, time_indices, scale_indices, ridge_indices):
    c = np.full((x.shape[0], x.shape[2], max_simultaneous_ridge_points), np.nan)
    c[(batch_indices, time_indices, ridge_indices)] = x[(batch_indices, scale_indices, time_indices)]
    return c


def _ridge_interpolate(x_list, indices, ridge_quantity, morse_wavelet=None, mu=None):

    is_single_x = not isinstance(x_list, (list, tuple))
    if is_single_x:
        x_list = [x_list]

    batch_indices, scale_indices, time_indices = indices

    dr = ridge_quantity[(batch_indices, scale_indices, time_indices)]
    dr_plus = ridge_quantity[(batch_indices, scale_indices + 1, time_indices)]
    dr_minus = ridge_quantity[(batch_indices, scale_indices - 1, time_indices)]

    _, interpolated_scales = quadratic_interpolate(
        scale_indices - 1, scale_indices, scale_indices + 1,
        np.abs(dr_minus) ** 2, np.abs(dr) ** 2, np.abs(dr_plus) ** 2)

    indicator_interpolation_fail = np.logical_not(np.logical_and(
        scale_indices + 1 > interpolated_scales, interpolated_scales > scale_indices - 1))

    interpolated_scales[indicator_interpolation_fail] = linear_interpolate(
        scale_indices[indicator_interpolation_fail] - 1, scale_indices[indicator_interpolation_fail] + 1,
        dr_minus[indicator_interpolation_fail], dr_plus[indicator_interpolation_fail])

    if morse_wavelet is not None and mu is not None:

        fact = (GeneralizedMorseWavelet.replace(morse_wavelet, beta=morse_wavelet.beta - mu).peak_frequency()
                / morse_wavelet.peak_frequency())
        freq = scale_indices / fact
        freq_plus = (scale_indices + 1) / fact
        freq_minus = (scale_indices - 1) / fact

        indicator_top = np.ceil(freq_plus) >= ridge_quantity.shape[1]
        freq[indicator_top] = ridge_quantity.shape[1] - 2
        freq_plus[indicator_top] = ridge_quantity.shape[1] - 1
        freq_minus[indicator_top] = ridge_quantity.shape[1] - 3

        indicator_bottom = np.floor(freq_minus) < 0
        freq[indicator_bottom] = 1
        freq_plus[indicator_bottom] = 2
        freq_minus[indicator_bottom] = 0

        x_ridge = list()
        x_ridge_plus = list()
        x_ridge_minus = list()
        for x in x_list:
            x_ridge.append(linear_interpolate(
                np.floor(freq), np.ceil(freq),
                x[(batch_indices, np.floor(freq).astype(int), time_indices)],
                x[(batch_indices, np.ceil(freq).astype(int), time_indices)]))
            x_ridge.append(linear_interpolate(
                np.floor(freq_plus), np.ceil(freq_plus),
                x[(batch_indices, np.floor(freq_plus).astype(int), time_indices)],
                x[(batch_indices, np.ceil(freq_plus).astype(int), time_indices)]))
            x_ridge_minus.append(linear_interpolate(
                np.floor(freq_minus), np.ceil(freq_minus),
                x[(batch_indices, np.floor(freq_minus).astype(int), time_indices)],
                x[(batch_indices, np.ceil(freq_minus).astype(int), time_indices)]))

    else:

        if morse_wavelet is not None or mu is not None:
            raise ValueError('If either morse_wavelet or mu is given, then both must be given')

        x_ridge = list()
        x_ridge_plus = list()
        x_ridge_minus = list()
        for x in x_list:
            x_ridge.append(x[(batch_indices, scale_indices, time_indices)])
            x_ridge_plus.append(x[(batch_indices, scale_indices + 1, time_indices)])
            x_ridge_minus.append(x[(batch_indices, scale_indices - 1, time_indices)])

    interpolated_x = list()
    for xrm, xr, xrp in zip(x_ridge_minus, x_ridge, x_ridge_plus):
        interpolated_x.append(quadratic_interpolate(
            scale_indices - 1, scale_indices, scale_indices + 1, xrm, xr, xrp, interpolated_scales))

        interpolated_x[-1][indicator_interpolation_fail] = linear_interpolate(
            scale_indices[indicator_interpolation_fail] - 1,
            scale_indices[indicator_interpolation_fail] + 1,
            xrm[indicator_interpolation_fail],
            xrp[indicator_interpolation_fail],
            interpolated_scales[indicator_interpolation_fail])

    if is_single_x:
        interpolated_x = interpolated_x[0]

    return interpolated_x


def _ridge_len(frequencies, dt=1, ridge_ids=None):
    if ridge_ids is None:
        ridge_ids = np.cumsum(np.isnan(np.roll(frequencies, -1, -1)), -1)

    result = np.full_like(frequencies, np.nan)
