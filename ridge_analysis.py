from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from .analytic_wavelet_transform_moments import instantaneous_frequency, amplitude, first_central_diff
from .analytic_wavelet import rotate, quadratic_interpolate, linear_interpolate, GeneralizedMorseWavelet


__all__ = ['ridges', 'ridge_collapse', 'dense_ridge', 'period_indices']


def period_indices(instantaneous_frequencies, spacing=1, dt=1, time_axis=-1):
    """
    Returns an array of indices where the indices are computed such that
        indices[..., i + 1, ...] - indices[..., i, ...] >= spacing * instantaneous_period[..., i, ...] for all i.
    The returned indices do not contain the first spacing * period and last spacing * period indices.
    Here i is indexing the time_axis. Since the number of valid indices may differ across axes other than the
    time_axis, the returned indices will be padded with values that are at least as large as
    instantaneous_frequencies.shape[time_axis].
    Args:
        instantaneous_frequencies: The array from which to compute the indices
        spacing: How many periods between indices
        dt: Sample rate used to compute instantaneous frequency
        time_axis: Which axis in the data is the time axis
    Returns:
        indices: An array of shape
        instantaneous_frequencies.shape[:time_axis] + max_indices + instantaneous_frequencies.shape[(time_axis + 1):]
    """
    skips = np.ceil(spacing * ((2 * np.pi) / instantaneous_frequencies * dt)).astype(int)
    indices = [np.take(skips, 0, time_axis)]
    indices[-1][indices[-1] >= instantaneous_frequencies.shape[time_axis] - 1] = -1
    while np.any(indices[-1] >= 0):
        indices.append(indices[-1] + np.take(skips, indices[-1], time_axis))
        indices[-1][np.logical_or(indices[-2] < 0, indices[-1] >= instantaneous_frequencies.shape[time_axis] - 1)] = -1
    # the last indices are all -1, so remove them
    indices = np.concatenate(indices[:-1], time_axis)
    # replace negative values with values that will cause index out of bounds if these are used carelessly
    return np.where(indices < 0, instantaneous_frequencies.shape[time_axis], indices)


def dense_ridge(x, original_shape, ridge_indices, fill_value=np.nan):
    """
    Converts the sparse 1d representation output by ridges to a dense array
    Args:
        x: The 1d array of ridge values to convert to a dense representation
        original_shape: The shape of the wavelet coefficients input to ridges
        ridge_indices: The coordinates of the wavelet coefficients output from ridges
        fill_value: What value to use in the array for coordinates which are not on the ridge

    Returns:
        dense_x: An array of shape original_shape with ridge values at the appropriate coordinates
    """
    result = np.full(original_shape, fill_value, x.dtype)
    result[ridge_indices] = x
    return result


def ridge_compress(x_list, original_shape, ridge_indices, ridge_ids, scale_axis=-2, fill_value=np.nan):
    """
    Conceptually, this maps each item in x_list to a dense representation, and then replaces the scale
        axis with a ridge id axis, so that each ridge lives on a single component of the ridge axis
        and ridge_ids map to indices in tha ridge axis.
    Args:
        x_list: The values on the ridge to collapse. If a list, then the first item in the list must be
            the wavelet coefficients. If an array, then the array is the wavelet coefficients. All values should
            be 1d arrays output by ridges
        original_shape: The shape of the wavelet coefficients array passed in to ridges
        ridge_indices: The coordinates of the wavelet coefficients, as output by ridges
        ridge_ids: The ids of the ridges, as output by ridges
        scale_axis: Which axis in the coordinates is the scale axis
        fill_value: What value to use in the result for coordinates which are not on the ridge
    Returns:
        compressed_dense_x: The compressed dense representation of x with shape:
                original_shape[:scale_axis] + num_ridges + original_shape[scale_axis + 1:]
            If x_list is a list, then the result will be a list with each item having that shape.
    """
    is_singleton = not isinstance(x_list, (list, tuple))
    if is_singleton:
        x_list = [x_list]
    if len(ridge_indices) != len(original_shape):
        raise ValueError('Shape mismatch between ridge_indices (len {}) and original_shape (len {})'.format(
            len(ridge_indices), len(original_shape)))
    ridge_indices = ridge_indices[:scale_axis] + ridge_indices[scale_axis + 1:]
    original_shape = original_shape[:scale_axis] + original_shape[scale_axis + 1:]
    dense_ridges = list()
    for _ in x_list:
        dense_ridges.append(list())
    for ridge_id in np.unique(ridge_ids):
        indicator_ridge_id = ridge_ids == ridge_id
        ridge_id_indices = tuple(ri[indicator_ridge_id] for ri in ridge_indices)
        for idx, x in enumerate(x_list):
            dense_ridges[idx].append(np.expand_dims(
                dense_ridge(x[indicator_ridge_id], original_shape, ridge_id_indices, fill_value=fill_value),
                scale_axis))
    for idx in range(len(dense_ridges)):
        dense_ridges[idx] = np.concatenate(dense_ridges[idx], axis=scale_axis)
    if is_singleton:
        return dense_ridges[0]
    return dense_ridges


def ridge_collapse(x_list, original_shape, ridge_indices, scale_axis=-2):
    """
    Collapses the multiple ridges output by ridges into a single ridge by combining together ridges which exist
        simultaneously.
    Args:
        x_list: The values on the ridge to collapse. If a list, then the first item in the list must be
            the wavelet coefficients. If an array, then the array is the wavelet coefficients. All values should
            be 1d arrays output by ridges
        original_shape: The shape of the wavelet coefficients array passed in to ridges
        ridge_indices: The coordinates of the wavelet coefficients, as output by ridges
        scale_axis: Which axis in the coordinates is the scale axis
    Returns:
        x / x_list: The collapsed ridge values
        ridge_indices: The coordinates of the collapsed ridge values (does not contain scale axis coordinates)
    """
    is_singleton = not isinstance(x_list, (list, tuple))
    if is_singleton:
        x_list = [x_list]
    power = None
    result = list()
    indicator_ridge = np.full(original_shape, False)
    indicator_ridge[ridge_indices] = True
    indicator_ridge = np.max(indicator_ridge, axis=scale_axis)
    new_indices = np.nonzero(indicator_ridge)
    del indicator_ridge
    for idx, item in enumerate(x_list):
        item = dense_ridge(item, original_shape, ridge_indices, fill_value=0)
        if idx == 0:
            if len(x_list) > 1:
                power = np.square(np.abs(item))
            c = np.sum(item, axis=scale_axis)
        else:
            c = np.sum(item * power, axis=scale_axis) / np.sum(power, axis=scale_axis)
        result.append(c[new_indices])
    if is_singleton:
        result = result[0]
    return result, new_indices


@dataclass
class RidgesResult:
    """
    Result returned from a call to ridges
    Fields:
        ridge_values: A 1d array of interpolated values of x for all points on a ridge
        indices: A tuple of indices giving coordinates of ridge points in x. If z is an array of zeros
            with the same shape as x, then z[indices] = ridge_values would have interpolated values of
            x wherever x has a ridge point and zero everywhere else.
        ridge_ids: Same shape as ridge_values, giving the id of the ridge for each ridge value
        instantaneous_frequencies: Same shape as ridge_values. Instantaneous frequencies along the ridge
        instantaneous_bandwidth: Same shape as ridge_values. Instantaneous bandwidth along the ridge
        instantaneous_curvature: Same shape as ridge_values. Instantaneous curvature along the ridge
        total_error: Same shape as ridge_values. A measure of deviation from a multivariate
            (or univariate) oscillation. Should be << 1 if the ridge estimate is good.
            Only returned if morse_wavelet is provided

            See equation 62 in
            Lilly and Olhede (2012), Analysis of Modulated Multivariate
            Oscillations. IEEE Trans. Sig. Proc., 60 (2), 600--612.
    """
    ridge_values: np.ndarray
    indices: Tuple[np.ndarray, ...]
    ridge_ids: np.ndarray
    instantaneous_frequencies: np.ndarray
    instantaneous_bandwidth: np.ndarray
    instantaneous_curvature: np.ndarray
    total_error: Optional[np.ndarray] = None


def ridges(
        x,
        scale_frequencies,
        dt=1,
        ridge_kind='amplitude',
        morse_wavelet=None,
        min_wavelet_lengths_in_ridge=None,
        trim_wavelet_lengths=None,
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
        trim_wavelet_lengths: Trims this many wavelet lengths from the edges of each ridge since these are
            contaminated by edge effects. A value of 1 is recommended, but if provided requires morse_wavelet.
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
        scale_axis: Which axis of x is the scale axis
        time_axis: Which axis of x is the time axis
        variable_axis: If specified, then the ridge is computed jointly over this axis

    Returns:
        An instance of RidgesResult with the fields:
            ridge_values: A 1d array of interpolated values of x for all points on a ridge
            indices: A tuple of indices giving coordinates of ridge points in x. If z is an array of zeros
                with the same shape as x, then z[indices] = ridge_values would have interpolated values of
                x wherever x has a ridge point and zero everywhere else.
            ridge_ids: Same shape as ridge_values, giving the id of the ridge for each ridge value
            instantaneous_frequencies: Same shape as ridge_values. Instantaneous frequencies along the ridge
            instantaneous_bandwidth: Same shape as ridge_values. Instantaneous bandwidth along the ridge
            instantaneous_curvature: Same shape as ridge_values. Instantaneous curvature along the ridge
            total_error: Same shape as ridge_values. A measure of deviation from a multivariate
                (or univariate) oscillation. Should be << 1 if the ridge estimate is good.
                Only returned if morse_wavelet is provided

                See equation 62 in
                Lilly and Olhede (2012), Analysis of Modulated Multivariate
                Oscillations. IEEE Trans. Sig. Proc., 60 (2), 600--612.
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

    # we need to keep this in the multivariate case. _indicator_ridge converts x to 3d (which we want
    # for _assign_ridge_ids, but we also need to keep the 4d version around for interpolation below)
    indicator_ridge, ridge_quantity, instantaneous_frequencies = _indicator_ridge(
        x, scale_frequencies, ridge_kind, 0, frequency_min, frequency_max, mask)

    ridge_ids = _assign_ridge_ids(indicator_ridge, ridge_quantity, instantaneous_frequencies, alpha)

    # not sure why this is necessary...points not in the mask are already eliminated earlier,
    # so I'm not sure how a disallowed point can get into a ridge, but this is in the original
    # code and breaks up ridges that span disallowed points into multiple ridges with different
    # ids
    if mask is not None:
        _mask_ridges(ridge_ids, mask)

    min_periods = None
    if min_wavelet_lengths_in_ridge is not None:
        if morse_wavelet is None:
            raise ValueError('If min_wavelet_lengths_in_ridge is given, morse_wavelet must be specified')
        min_periods = min_wavelet_lengths_in_ridge * 2 * morse_wavelet.time_domain_width() / np.pi

    trim_periods = None
    if trim_wavelet_lengths is not None:
        if morse_wavelet is None:
            raise ValueError('If trim_wavelet_lengths is given, morse_wavelet must be specified')
        trim_periods = trim_wavelet_lengths * morse_wavelet.time_domain_width() / np.pi

    # now clean up the ridge ids according to the parameters, interpolate, and compute bias parameters
    unique_ridge_ids, ridge_id_count = np.unique(ridge_ids, return_counts=True)
    compressed_id = np.max(unique_ridge_ids) + 1
    for ridge_id, ridge_count in zip(unique_ridge_ids, ridge_id_count):
        if ridge_id < 0:
            continue

        indicator_ridge_id = ridge_ids == ridge_id
        # remove singleton ridge ids
        if ridge_count < 2:
            ridge_ids[indicator_ridge_id] = -1
            continue

        if min_periods is not None:
            ridge_batch_indices, ridge_scale_indices, ridge_time_indices = np.nonzero(indicator_ridge_id)
            ridge_len = np.sum(scale_frequencies[ridge_scale_indices] / (2 * np.pi) * dt)
            if ridge_len < min_periods:
                ridge_ids[indicator_ridge_id] = -1

        if trim_wavelet_lengths is not None:
            ridge_batch_indices, ridge_scale_indices, ridge_time_indices = np.nonzero(indicator_ridge_id)
            time_sort = np.argsort(ridge_time_indices)
            age = np.cumsum(scale_frequencies[ridge_scale_indices[time_sort]] / (2 * np.pi) * dt)
            age = age[np.argsort(time_sort)]
            indicator_trim = np.logical_or(age <= trim_periods, age >= np.max(age) - trim_periods)
            trim_batch_indices = ridge_batch_indices[indicator_trim]
            trim_scale_indices = ridge_scale_indices[indicator_trim]
            trim_time_indices = ridge_time_indices[indicator_trim]
            ridge_ids[(trim_batch_indices, trim_scale_indices, trim_time_indices)] = -1
            indicator_ridge_id = ridge_ids == ridge_id

        # reassign ids so that they will be contiguous
        ridge_ids[indicator_ridge_id] = compressed_id
        compressed_id += 1

    # shift the ids to start at 0
    ridge_ids[ridge_ids >= 0] = ridge_ids[ridge_ids >= 0] - (np.max(unique_ridge_ids) + 1)

    instantaneous_frequencies = instantaneous_frequencies / dt
    x1 = first_central_diff(x, padding='endpoint') / dt
    x2 = first_central_diff(x1, padding='nan') / dt
    x2[..., 0] = x2[..., 1]
    x2[..., -1] = x2[..., -2]

    if len(x.shape) == 4:
        # put variable axis last for interpolation
        x = np.moveaxis(x, 1, -1)
        x1 = np.moveaxis(x1, 1, -1)
        x2 = np.moveaxis(x2, 1, -1)

    ridge_indices = np.nonzero(ridge_ids >= 0)

    x, x1, x2, instantaneous_frequencies = _ridge_interpolate(
        [x, x1, x2, instantaneous_frequencies], ridge_indices, ridge_quantity)

    if len(x.shape) == 2:
        instantaneous_frequencies = np.expand_dims(instantaneous_frequencies, 1)
        l2 = np.sqrt(np.sum(np.square(np.abs(x)), axis=1, keepdims=True))
    else:
        l2 = np.sqrt(np.square(np.abs(x)))

    bandwidth = (x1 - 1j * instantaneous_frequencies * x) / l2
    curvature = (x2 - 2 * 1j * instantaneous_frequencies * x1 - instantaneous_frequencies ** 2 * x) / l2

    result = [ridge_ids, x, instantaneous_frequencies, bandwidth, curvature]

    if morse_wavelet is not None:
        curvature_l2 = np.sqrt(np.sum(np.square(np.abs(curvature)), axis=2, keepdims=True)) \
            if len(curvature.shape) == 2 else np.sqrt(np.square(np.abs(curvature)))
        total_err = (1 / 2 * np.square(np.abs(morse_wavelet.time_domain_width() / instantaneous_frequencies))
                     * curvature_l2)
        result = result + [total_err]

    expanded_shape = ridge_ids.shape if len(x.shape) == 1 else ridge_ids.shape + (x.shape[1],)
    expanded_ridge_indices = None
    for idx in range(len(result)):
        if idx == 0:
            expanded = np.full(expanded_shape, -1, dtype=int)  # ridge_ids
        else:
            expanded = np.full(expanded_shape, np.nan, dtype=result[idx].dtype)
        if len(expanded_shape) == 4:
            if len(result[idx].shape) == 3:
                result[idx] = np.expand_dims(result[idx], 3)
            if result[idx].shape[3] == 1:
                result[idx] = np.tile(result[idx], (1, 1, 1, expanded.shape[3]))
        expanded[ridge_indices] = result[idx]
        if len(expanded.shape) == 4:
            # move the variable axis back to axis=1
            expanded = np.moveaxis(expanded, -1, 1)
            # reshape back to input shape
            expanded = np.reshape(expanded, orig_shape)
            # restore axes
            expanded = np.moveaxis(expanded, (-3, -2, -1), (variable_axis, scale_axis, time_axis))
        else:
            # reshape back to input shape
            expanded = np.reshape(expanded, orig_shape)
            # restore axes
            expanded = np.moveaxis(expanded, (-2, -1), (scale_axis, time_axis))
        if idx == 0:  # ridge_ids
            expanded_ridge_indices = np.nonzero(expanded >= 0)
        result[idx] = expanded[expanded_ridge_indices]

    return RidgesResult(
        ridge_values=result[1],
        indices=expanded_ridge_indices,
        ridge_ids=result[0],
        instantaneous_frequencies=result[2],
        instantaneous_bandwidth=result[3],
        instantaneous_curvature=result[4],
        total_error=result[5] if len(result) > 5 else None)


def _mask_ridges(ridge_ids, mask):
    unique_ridge_ids = np.unique(ridge_ids)
    next_ridge_id = np.max(unique_ridge_ids) + 1
    disallowed = np.logical_not(mask)
    time_indices = np.reshape(np.arange(ridge_ids.shape[-1]), (1, 1, ridge_ids.shape[-1]))
    for ridge_id in unique_ridge_ids:
        indicator_ridge = ridge_ids == ridge_id
        breaks = np.logical_and(indicator_ridge, disallowed)
        if np.sum(breaks) > 0:
            # remove the disallowed points from the ridge
            ridge_ids[breaks] = -1
            indicator_ridge = ridge_ids == ridge_id

            # find the time indices of the disallowed point
            break_times = np.flatnonzero(np.max(np.max(breaks, axis=0), axis=0))
            indicator_skip = np.diff(break_times) > 1
            # remove all the times that are redundant (contiguous breaks)
            break_times = np.concatenate([break_times[:1], break_times[1:][indicator_skip]])

            # assign new ids to the points between the breaks
            for t in break_times:
                indicator_greater_time = np.logical_and(indicator_ridge, time_indices > t)
                ridge_ids[indicator_greater_time] = next_ridge_id
                next_ridge_id += 1


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

    # remove edges on frequency axis
    result[:, 0, :] = False
    result[:, -1, :] = False

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

    return result, ridge_quantity, frequency


def _assign_ridge_ids(indicator_points, ridge_quantity, instantaneous_frequencies, alpha):

    batch_indices, scale_indices, time_indices = np.nonzero(indicator_points)
    df_dt = first_central_diff(instantaneous_frequencies, axis=-1)
    instantaneous_frequencies, df_dt = _ridge_interpolate(
        [instantaneous_frequencies, df_dt], (batch_indices, scale_indices, time_indices), ridge_quantity,
        keep_shapes=True)
    fr_next = instantaneous_frequencies + df_dt
    fr_prev = instantaneous_frequencies - df_dt

    # compress the scale axis to only enough for the total number of simultaneous ridge points
    ridge_indices = np.cumsum(indicator_points, axis=1)
    max_simultaneous_ridge_points = np.max(ridge_indices) + 1

    # this converts ridge_indices into a 1d array which is now suitable for replacing
    # scale_indices
    ridge_indices = ridge_indices[(batch_indices, scale_indices, time_indices)]

    # note: this compression also moves the ridge axis to axis=2 since that's what we mostly work with
    instantaneous_frequencies_ridge = _compress_scale_axis_to_ridge_axis(
        instantaneous_frequencies,
        max_simultaneous_ridge_points, batch_indices, time_indices, scale_indices, ridge_indices)
    fr_next_ridge = _compress_scale_axis_to_ridge_axis(
        fr_next,
        max_simultaneous_ridge_points, batch_indices, time_indices, scale_indices, ridge_indices)
    fr_prev_ridge = _compress_scale_axis_to_ridge_axis(
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
    indicator_valid_next = np.logical_and(next_ridge_indices >= 0, time_indices + 1 < ridge_ids.shape[1])
    valid_next_batch_indices = batch_indices[indicator_valid_next]
    valid_next_time_indices = time_indices[indicator_valid_next]
    valid_next_ridge_indices = ridge_indices[indicator_valid_next]
    valid_next_next_indices = next_ridge_indices[indicator_valid_next]
    # propagate the ids forward along links
    while True:
        if np.all(
                ridge_ids[(valid_next_batch_indices, valid_next_time_indices + 1, valid_next_next_indices)] ==
                ridge_ids[(valid_next_batch_indices, valid_next_time_indices, valid_next_ridge_indices)]):
            break
        ridge_ids[(valid_next_batch_indices, valid_next_time_indices + 1, valid_next_next_indices)] = \
            ridge_ids[(valid_next_batch_indices, valid_next_time_indices, valid_next_ridge_indices)]

    # return to the original scale axis
    result = np.full(indicator_points.shape, -1, dtype=int)
    result[(batch_indices, time_indices, scale_indices)] = ridge_ids[(batch_indices, time_indices, ridge_indices)]
    return result


def _compress_scale_axis_to_ridge_axis(
        x, max_simultaneous_ridge_points, batch_indices, time_indices, scale_indices, ridge_indices):
    c = np.full((x.shape[0], x.shape[2], max_simultaneous_ridge_points), np.nan)
    c[(batch_indices, time_indices, ridge_indices)] = x[(batch_indices, scale_indices, time_indices)]
    return c


def _ridge_interpolate(x_list, indices, ridge_quantity, morse_wavelet=None, mu=None, keep_shapes=False):

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
    for xrm, xr, xrp, x in zip(x_ridge_minus, x_ridge, x_ridge_plus, x_list):

        s = np.reshape(scale_indices, (-1,) + (1,) * (len(xr.shape) - 1))

        interpolated_x.append(quadratic_interpolate(
            s - 1, s, s + 1, xrm, xr, xrp, interpolated_scales))

        interpolated_x[-1][indicator_interpolation_fail] = linear_interpolate(
            s[indicator_interpolation_fail] - 1,
            s[indicator_interpolation_fail] + 1,
            xrm[indicator_interpolation_fail],
            xrp[indicator_interpolation_fail],
            interpolated_scales[indicator_interpolation_fail])

        if keep_shapes:
            z = np.full_like(x, np.nan)
            z[(batch_indices, scale_indices, time_indices)] = interpolated_x[-1]
            interpolated_x[-1] = z

    if is_single_x:
        interpolated_x = interpolated_x[0]

    return interpolated_x
