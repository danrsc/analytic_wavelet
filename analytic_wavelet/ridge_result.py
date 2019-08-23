import warnings
import numpy as np
from .transform import rotate

__all__ = ['RidgeFields', 'RidgeRepresentation', 'RidgeResult']


def _dense_ridge(x, original_shape, ridge_indices, fill_value=np.nan):
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


def _ridge_compress(x_list, original_shape, ridge_indices, ridge_ids, scale_axis=-2, fill_value=np.nan):
    """
    Conceptually, this maps each item in x_list to a dense representation, and then replaces the scale
        axis with a ridge id axis, so that each ridge lives on a single component of the ridge axis
        and ridge_ids map to indices in tha ridge axis.
    Args:
        x_list: The values on the ridge to collapse. Either a 1d array, or a list of 1d arrays. If a list,
            then each item in the list is converted.
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

    multiple_original_shapes = isinstance(original_shape[0], (list, tuple))
    if not multiple_original_shapes:
        original_shape = [original_shape] * len(x_list)
    if len(original_shape) != len(x_list):
        raise ValueError('If multiple original shapes are provided, there must be one for each item in x')
    for idx in range(1, len(original_shape)):
        if not np.array_equal(original_shape[idx][:len(ridge_indices)], original_shape[0][:len(ridge_indices)]):
            raise ValueError('Shapes must agree up to the number of ridge_indices: {}, {}'.format(
                original_shape[idx], original_shape[0]))
        if len(original_shape[0]) != len(original_shape[idx]):
            raise ValueError('All shapes must have the same number of dimensions: {}, {}'.format(
                original_shape[idx], original_shape[0]))

    if scale_axis < 0:
        scale_axis += len(original_shape[0])
        assert(0 <= scale_axis < len(original_shape[0]))
    if len(ridge_indices) < scale_axis:
        raise ValueError('Shape mismatch between ridge_indices (len {}) and provided scale_axis ({})'.format(
            len(ridge_indices), scale_axis))

    for idx in range(len(original_shape)):
        if len(original_shape[idx]) < scale_axis:
            raise ValueError('Shape mismatch between original_shape (len {}) and provided scale_axis ({})'.format(
                len(original_shape[idx]), scale_axis))
        original_shape[idx] = original_shape[idx][:scale_axis] + original_shape[idx][(scale_axis + 1):]
    ridge_indices = ridge_indices[:scale_axis] + ridge_indices[(scale_axis + 1):]
    dense_ridges = list()
    for _ in x_list:
        dense_ridges.append(list())
    if ridge_ids.ndim > 1:
        if ridge_ids.ndim > 2 or ridge_ids.shape[1] != 1:
            raise ValueError('Expected ridge_ids to be 1d')
        ridge_ids = np.squeeze(ridge_ids, axis=1)
    for ridge_id in np.unique(ridge_ids):
        indicator_ridge_id = ridge_ids == ridge_id
        ridge_id_indices = tuple(ri[indicator_ridge_id] for ri in ridge_indices)
        for idx, x in enumerate(x_list):
            dense_ridges[idx].append(np.expand_dims(
                _dense_ridge(x[indicator_ridge_id], original_shape[idx], ridge_id_indices, fill_value=fill_value),
                scale_axis))
    for idx in range(len(dense_ridges)):
        dense_ridges[idx] = np.concatenate(dense_ridges[idx], axis=scale_axis)
    if is_singleton:
        return dense_ridges[0]
    return dense_ridges


class RidgeRepresentation:
    points = 'points'
    compressed = 'compressed'
    dense = 'dense'

    def __init__(self):
        raise RuntimeError('Constant class. do not instantiate')


class RidgeFields:
    ridge_values = 'ridge_values'
    instantaneous_frequency = 'instantaneous_frequency'
    instantaneous_bandwidth = 'instantaneous_bandwidth'
    instantaneous_curvature = 'instantaneous_curvature'
    total_error = 'total_error'
    ridge_ids = 'ridge_ids'

    def __init__(self):
        raise RuntimeError('Constant class. do not instantiate')


class RidgeResult:

    def __init__(
            self,
            original_shape,
            ridge_values,
            indices,
            ridge_ids,
            instantaneous_frequency,
            instantaneous_bandwidth,
            instantaneous_curvature,
            total_error,
            variable_axis,
            scale_axis):
        """
        You should not call this constructor yourself. Call ridges to create an instances of RidgeResult
        Args:
            original_shape: The shape of the data on which ridge analysis was run
            ridge_values: A 1d array of interpolated values of x for all points on a ridge
            indices: A tuple of indices giving coordinates of ridge points in x. If z is an array of zeros
                with the same shape as x, then z[indices] = ridge_values would have interpolated values of
                x wherever x has a ridge point and zero everywhere else.
            ridge_ids: Same shape as ridge_values, giving the id of the ridge for each ridge value
            instantaneous_frequency: Same shape as ridge_values. Instantaneous frequencies along the ridge
            instantaneous_bandwidth: Same shape as ridge_values. Instantaneous bandwidth along the ridge
            instantaneous_curvature: Same shape as ridge_values. Instantaneous curvature along the ridge
            total_error: Same shape as ridge_values.
            variable_axis: Which axis contains the components of multivariate values (or None for univariate)
            scale_axis: Which axis contains the scales in the original shape
        """
        self._ridge_values = ridge_values
        self._indices = indices
        self._ridge_ids = ridge_ids
        self._instantaneous_frequency = instantaneous_frequency
        self._instantaneous_bandwidth = instantaneous_bandwidth
        self._instantaneous_curvature = instantaneous_curvature
        self._total_error = total_error

        # internally, the variable axis is the last axis. This means that when we use point/collapsed (sparse)
        # representation the values are actually the multivariate points, which is much more intuitive for the caller.
        # When we convert to a dense/compressed representation we do so by first converting
        # to a dense representation as though the variable axis were last and then moving that axis.
        # Therefore, we adjust the shape and the scale axis here to move the variable axis last for calls
        # to helper functions
        self._true_variable_axis = variable_axis
        self._modified_scale_axis = scale_axis
        self._modified_original_shape = original_shape
        if self._true_variable_axis is not None:
            # handle negative variable axis index
            if self._true_variable_axis < 0:
                self._true_variable_axis += len(self._modified_original_shape)
                assert (0 <= self._true_variable_axis < len(self._modified_original_shape))
            if self._modified_scale_axis < 0:
                self._modified_scale_axis += len(self._modified_original_shape)
                assert (0 <= self._modified_scale_axis < len(self._modified_original_shape))
            # the scale axis is after the variable axis, so when we move the variable axis we decrement it
            if self._modified_scale_axis > self._true_variable_axis:
                self._modified_scale_axis -= 1
            self._modified_original_shape = (
                self._modified_original_shape[:self._true_variable_axis] +
                self._modified_original_shape[self._true_variable_axis+1:] +
                (self._modified_original_shape[self._true_variable_axis],))

    @property
    def indices(self):
        return self._indices

    def ridge_values(self, representation=RidgeRepresentation.points, fill_value=np.nan):
        """
        Interpolated values of the wavelet coefficients for all points on a ridge
        Args:
            representation: A RidgeRepresentation specifying how ridge values should be returned:
                points: Return a 1d (univariate) or 2d (mulitivariate) array where each item on the first axis is
                    a point on a ridge
                dense: Return the ridge_values in a dense array having the same shape as the data on which ridge
                    analysis was originally run. Elements of the array not on a ridge will use fill_value as their
                    values.
                compressed: Similar to dense, but where the scale axis is replaced by a ridge axis. In
                    this representation, the ridge_ids index the array along the ridge axis, so that ridge_id 7
                    is in position 7 on the ridge axis in the compressed array.
            fill_value: The value to use for elements of the array not on the ridge in a dense or compressed
                representation. Ignored for representation == 'points'

        Returns:
            The ridge_values (interpolated wavelet coefficients) according to the specified representation.
        """
        return self._convert_representation(self._ridge_values, representation, fill_value)

    def instantaneous_frequency(self, representation=RidgeRepresentation.points, fill_value=np.nan):
        """
        The instantaneous frequency for all points on a ridge
        Args:
            representation: A RidgeRepresentation specifying how ridge values should be returned:
                points: Return a 1d (univariate) or 2d (mulitivariate) array where each item on the first axis is
                    a point on a ridge
                dense: Return the ridge_values in a dense array having the same shape as the data on which ridge
                    analysis was originally run. Elements of the array not on a ridge will use fill_value as their
                    values.
                compressed: Similar to dense, but where the scale axis is replaced by a ridge axis. In
                    this representation, the ridge_ids index the array along the ridge axis, so that ridge_id 7
                    is in position 7 on the ridge axis in the compressed array.
            fill_value: The value to use for elements of the array not on the ridge in a dense or compressed
                representation. Ignored for representation == 'points'

        Returns:
            The instantaneous frequency according to the specified representation.
        """
        return self._convert_representation(self._instantaneous_frequency, representation, fill_value)

    def instantaneous_bandwidth(self, representation=RidgeRepresentation.points, fill_value=np.nan):
        """
        The instantaneous bandwidth for all points on a ridge. A measure of deviation from a multivariate
        (or univariate) oscillation. See equation 17 in
        Lilly and Olhede (2012), Analysis of Modulated Multivariate
        Oscillations. IEEE Trans. Sig. Proc., 60 (2), 600--612.
        Args:
            representation: A RidgeRepresentation specifying how ridge values should be returned:
                points: Return a 1d (univariate) or 2d (mulitivariate) array where each item on the first axis is
                    a point on a ridge
                dense: Return the ridge_values in a dense array having the same shape as the data on which ridge
                    analysis was originally run. Elements of the array not on a ridge will use fill_value as their
                    values.
                compressed: Similar to dense, but where the scale axis is replaced by a ridge axis. In
                    this representation, the ridge_ids index the array along the ridge axis, so that ridge_id 7
                    is in position 7 on the ridge axis in the compressed array.
            fill_value: The value to use for elements of the array not on the ridge in a dense or compressed
                representation. Ignored for representation == 'points'

        Returns:
            The instantaneous bandwidth according to the specified representation.
        """
        return self._convert_representation(self._instantaneous_bandwidth, representation, fill_value)

    def instantaneous_curvature(self, representation=RidgeRepresentation.points, fill_value=np.nan):
        """
        The instantaneous curvature for all points on a ridge. A measure of deviation from a multivariate
        (or univariate) oscillation. See equation 18 in
        Lilly and Olhede (2012), Analysis of Modulated Multivariate
        Oscillations. IEEE Trans. Sig. Proc., 60 (2), 600--612.
        Args:
            representation: A RidgeRepresentation specifying how ridge values should be returned:
                points: Return a 1d (univariate) or 2d (mulitivariate) array where each item on the first axis is
                    a point on a ridge
                dense: Return the ridge_values in a dense array having the same shape as the data on which ridge
                    analysis was originally run. Elements of the array not on a ridge will use fill_value as their
                    values.
                compressed: Similar to dense, but where the scale axis is replaced by a ridge axis. In
                    this representation, the ridge_ids index the array along the ridge axis, so that ridge_id 7
                    is in position 7 on the ridge axis in the compressed array.
            fill_value: The value to use for elements of the array not on the ridge in a dense or compressed
                representation. Ignored for representation == 'points'

        Returns:
            The instantaneous curvature according to the specified representation.
        """
        return self._convert_representation(self._instantaneous_bandwidth, representation, fill_value)

    def total_error(self, representation=RidgeRepresentation.points, fill_value=np.nan):
        """
        The total_error for all points on a ridge. A measure of deviation from a multivariate
        (or univariate) oscillation. Should be << 1 if the ridge estimate is good.
        Only returned if morse_wavelet is provided

        See equation 62 in
        Lilly and Olhede (2012), Analysis of Modulated Multivariate
        Oscillations. IEEE Trans. Sig. Proc., 60 (2), 600--612.
        Args:
            representation: A RidgeRepresentation specifying how ridge values should be returned:
                points: Return a 1d (univariate) or 2d (mulitivariate) array where each item on the first axis is
                    a point on a ridge
                dense: Return the ridge_values in a dense array having the same shape as the data on which ridge
                    analysis was originally run. Elements of the array not on a ridge will use fill_value as their
                    values.
                compressed: Similar to dense, but where the scale axis is replaced by a ridge axis. In
                    this representation, the ridge_ids index the array along the ridge axis, so that ridge_id 7
                    is in position 7 on the ridge axis in the compressed array.
            fill_value: The value to use for elements of the array not on the ridge in a dense or compressed
                representation. Ignored for representation == 'points'

        Returns:
            The total error according to the specified representation.
        """
        if self._total_error is None:
            return None
        return self._convert_representation(self._total_error, representation, fill_value)

    def ridge_ids(self, representation=RidgeRepresentation.points, fill_value=-1):
        """
        The ridge ids for each point in the ridge
        Args:
            representation: A RidgeRepresentation specifying how ridge values should be returned:
                points: Return a 1d (univariate) or 2d (mulitivariate) array where each item on the first axis is
                    a point on a ridge
                dense: Return the ridge_values in a dense array having the same shape as the data on which ridge
                    analysis was originally run. Elements of the array not on a ridge will use fill_value as their
                    values.
                compressed: Similar to dense, but where the scale axis is replaced by a ridge axis. In
                    this representation, the ridge_ids index the array along the ridge axis, so that ridge_id 7
                    is in position 7 on the ridge axis in the compressed array.
            fill_value: The value to use for elements of the array not on the ridge in a dense or compressed
                representation. Ignored for representation == 'points'

        Returns:
            The ridge ids according to the specified representation.
        """
        return self._convert_representation(self._ridge_ids, representation, fill_value)

    def get_values(self, fields, representation=RidgeRepresentation.points, fill_value=np.nan):
        """
        Get the values for multiple fields in a single function call.
        Args:
            fields: Either a single RidgeField or a list of RidgeFields
            representation: The representation to use, from RidgeRepresentation
            fill_value: The fill_value to use for the dense and compressed representations

        Returns:
            If fields is a list, returns a list of arrays which are the results. Otherwise a single array.
        """
        is_single = not isinstance(fields, (list, tuple))
        if is_single:
            fields = [fields]

        available_values = {
            RidgeFields.ridge_values: self._ridge_values,
            RidgeFields.instantaneous_frequency: self._instantaneous_frequency,
            RidgeFields.instantaneous_bandwidth: self._instantaneous_bandwidth,
            RidgeFields.instantaneous_curvature: self._instantaneous_curvature,
            RidgeFields.total_error: self._total_error,
            RidgeFields.ridge_ids: self._ridge_ids
        }

        requested_values = dict()
        for field in fields:
            if field not in available_values:
                raise ValueError('Unknown field: {}'.format(field))
            if available_values[field] is not None:
                requested_values[field] = available_values[field]

        x = [requested_values[field] for field in requested_values]
        x = self._convert_representation(x, representation, fill_value)
        assert(len(x) == len(requested_values))
        requested_values = dict(zip(requested_values, x))
        result = [requested_values[field] if field in requested_values else None for field in fields]
        if is_single:
            return result[0]
        return result

    def _adjust_dense_shape(self, x):
        if self._true_variable_axis is not None:
            return [np.moveaxis(item, -1, self._true_variable_axis) for item in x]
        return x

    def _dense_shape(self, x):
        if self._true_variable_axis is None:
            return self._modified_original_shape
        # some values broadcast over the multivariate components
        if isinstance(x, (list, tuple)):
            return [self._modified_original_shape[:-1] + (item.shape[-1],) for item in x]
        return self._modified_original_shape[:-1] + (x.shape[-1],)

    def _convert_representation(self, x, representation, fill_value):
        is_single = not isinstance(x, (tuple, list))
        if is_single:
            x = [x]
        if representation == RidgeRepresentation.points:
            if self._true_variable_axis is not None:
                x = [np.squeeze(item, axis=1) if item.shape[1] == 1 else item for item in x]
        elif representation == RidgeRepresentation.compressed:
            x = self._adjust_dense_shape(_ridge_compress(
                x, self._dense_shape(x), self._indices, self._ridge_ids, self._modified_scale_axis,
                fill_value))
        elif representation == RidgeRepresentation.dense:
            x = self._adjust_dense_shape([_dense_ridge(
                item, self._dense_shape(item), self._indices, fill_value) for item in x])
        else:
            raise ValueError('Unrecognized representation: {}'.format(representation))
        if is_single:
            return x[0]
        return x

    def collapse(self):
        """
        Collapses the multiple ridges output by ridges into a single ridge by combining together ridges which exist
            simultaneously using power-weighted averaging.
        Returns:
            A new RidgeResult instance with collapsed ridges
        """
        x_list = [
            self._ridge_values,
            self._instantaneous_frequency,
            self._instantaneous_bandwidth,
            self._instantaneous_curvature,
        ]

        if self._total_error is not None:
            x_list.append(self._total_error)

        power = None
        result = list()
        if self._true_variable_axis is not None:
            indicator_ridge = np.full(self._modified_original_shape[:-1], False)
        else:
            indicator_ridge = np.full(self._modified_original_shape, False)
        indicator_ridge[self._indices] = True
        indicator_ridge = np.any(indicator_ridge, axis=self._modified_scale_axis, keepdims=True)
        new_indices = np.nonzero(indicator_ridge)
        del indicator_ridge
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', category=RuntimeWarning)
            for idx, item in enumerate(x_list):
                item = _dense_ridge(item, self._dense_shape(item), self._indices, fill_value=0)
                if idx == 0:
                    if self._true_variable_axis is not None:
                        phase_avg = (np.nansum(np.abs(item) * item, axis=-1, keepdims=True)
                                     / np.nansum(np.abs(item) ** 2, axis=-1, keepdims=True))
                        joint = (np.sqrt(np.nansum(np.abs(item) ** 2, axis=-1, keepdims=True))
                                 * rotate(np.angle(phase_avg)))
                        power = np.square(np.abs(joint))
                    else:
                        power = np.square(np.abs(item))
                    c = np.nansum(item, axis=self._modified_scale_axis, keepdims=True)
                else:
                    c = (np.nansum(item * power, axis=self._modified_scale_axis, keepdims=True)
                         / np.nansum(power, axis=self._modified_scale_axis, keepdims=True))
                result.append(c[new_indices])

        original_shape = self._modified_original_shape
        scale_axis = self._modified_scale_axis
        if self._true_variable_axis is not None:
            # restore original shape and scale axis to pass into constructor
            original_shape = (
                    original_shape[:self._true_variable_axis]
                    + (original_shape[-1],)
                    + original_shape[self._true_variable_axis:-1])
            if scale_axis + 1 > self._true_variable_axis:
                scale_axis += 1

        return RidgeResult(
            original_shape,
            result[0],
            new_indices,
            np.zeros((len(result[0]),) + self._ridge_ids.shape[1:], dtype=int),  # new ridge ids are all 0
            result[1],
            result[2],
            result[3],
            result[4] if self._total_error is not None else None,
            self._true_variable_axis,
            scale_axis)
