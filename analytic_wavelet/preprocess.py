import numpy as np


__all__ = ['masked_detrend', 'make_unpad_slices', 'unpad']


def _masked_detrend(arr, axis=-1):

    to_fit = np.ma.masked_invalid(np.moveaxis(arr, axis, 0))
    x = np.arange(len(to_fit))

    shape = to_fit.shape
    to_fit = np.reshape(to_fit, (to_fit.shape[0], -1))

    # some columns might not have any data
    indicator_can_fit = np.logical_not(np.all(to_fit.mask, axis=0))

    p = np.ma.polyfit(x, to_fit[:, indicator_can_fit], deg=1)

    filled_p = np.zeros((p.shape[0], len(indicator_can_fit)), p.dtype)
    filled_p[:, indicator_can_fit] = p
    p = filled_p

    #      (1, num_columns)            (num_rows, 1)
    lines = np.reshape(p[0], (1, -1)) * np.reshape(np.arange(len(to_fit)), (-1, 1)) + np.reshape(p[1], (1, -1))
    lines = np.moveaxis(np.reshape(lines, shape), 0, axis)
    return arr - lines


# unfortunately, scipy.signal.detrend does not handle masking
def masked_detrend(arr, axis=-1):
    if not np.isrealobj(arr):
        return _masked_detrend(np.real(arr), axis=axis) + 1j * _masked_detrend(np.imag(arr), axis=axis)
    return _masked_detrend(arr, axis=axis)


def _as_pairs(x, ndim, as_index=False):
    # copied from https://github.com/numpy/numpy/blob/v1.17.0/numpy/lib/arraypad.py
    """
    Broadcast `x` to an array with the shape (`ndim`, 2).
    A helper function for `pad` that prepares and validates arguments like
    `pad_width` for iteration in pairs.
    Parameters
    ----------
    x : {None, scalar, array-like}
        The object to broadcast to the shape (`ndim`, 2).
    ndim : int
        Number of pairs the broadcasted `x` will have.
    as_index : bool, optional
        If `x` is not None, try to round each element of `x` to an integer
        (dtype `np.intp`) and ensure every element is positive.
    Returns
    -------
    pairs : nested iterables, shape (`ndim`, 2)
        The broadcasted version of `x`.
    Raises
    ------
    ValueError
        If `as_index` is True and `x` contains negative elements.
        Or if `x` is not broadcastable to the shape (`ndim`, 2).
    """
    if x is None:
        # Pass through None as a special case, otherwise np.round(x) fails
        # with an AttributeError
        return ((None, None),) * ndim

    x = np.array(x)
    if as_index:
        x = np.round(x).astype(np.intp, copy=False)

    if x.ndim < 3:
        # Optimization: Possibly use faster paths for cases where `x` has
        # only 1 or 2 elements. `np.broadcast_to` could handle these as well
        # but is currently slower

        if x.size == 1:
            # x was supplied as a single value
            x = x.ravel()  # Ensure x[0] works for x.ndim == 0, 1, 2
            if as_index and x < 0:
                raise ValueError("index can't contain negative values")
            return ((x[0], x[0]),) * ndim

        if x.size == 2 and x.shape != (2, 1):
            # x was supplied with a single value for each side
            # but except case when each dimension has a single value
            # which should be broadcasted to a pair,
            # e.g. [[1], [2]] -> [[1, 1], [2, 2]] not [[1, 2], [1, 2]]
            x = x.ravel()  # Ensure x[0], x[1] works
            if as_index and (x[0] < 0 or x[1] < 0):
                raise ValueError("index can't contain negative values")
            return ((x[0], x[1]),) * ndim

    if as_index and x.min() < 0:
        raise ValueError("index can't contain negative values")

    # Converting the array with `tolist` seems to improve performance
    # when iterating and indexing the result (see usage in `pad`)
    return np.broadcast_to(x, (ndim, 2)).tolist()


def make_unpad_slices(ndim, pad_width):
    slices = list()
    for begin_pad, end_pad in _as_pairs(pad_width, ndim, as_index=True):
        if end_pad is not None:
            if end_pad == 0:
                end_pad = None
            else:
                end_pad = -end_pad
        slices.append(slice(begin_pad, end_pad))
    return tuple(slices)


def unpad(pad_width, *args):
    if len(args) == 0:
        raise ValueError('Expected at least one array to unpad')
    unpad_slices = make_unpad_slices(args[0].ndim, pad_width)
    if any(not np.array_equal(a.shape, args[0].shape) for a in args):
        raise ValueError('All arrays passed to unpad must have the same shape')
    if len(args) == 1:
        return args[0][unpad_slices]
    return tuple(arr[unpad_slices] for arr in args)
