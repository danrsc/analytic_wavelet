from collections import OrderedDict
import numpy as np
from scipy.io import loadmat


__all__ = ['load_test_data']


def _convert_void_to_dict(v):
    result = OrderedDict()
    for field in v.dtype.fields:
        if v[field].shape != (1, 1):
            raise ValueError('Expected each value to have shape (1, 1): {}: {}'.format(field, v[field].shape))
        value = v[field][0, 0]
        if value.dtype.kind in ('S', 'U'):  # string
            value = str(value[0])
        elif value.dtype.kind == 'V':  # structure in MATLAB
            value = _convert_void_to_dict(value)
        elif value.dtype.kind == 'O':  # cell array in MATLAB, load as list
            if value.shape[1] != 1:
                raise ValueError('Expected each object kind to be 1d')
            value = np.squeeze(value, axis=1)
            value = list(value)
        result[field] = value
    return result


def load_test_data(path):
    """
    Loads matlab files from the test_data directory into reasonable data structures. Note that this may not
    work perfectly in general for any MATLAB file. We make some assumptions here (for example that dtype.kind == 'O'
    is always 1d and comes from a MATLAB cell array) which make our result nicer to use but which may not hold
    in general for any MATLAB file.
    Args:
        path: The path to the data file to load

    Returns:
        A dictionary where each value in the dictionary could be either a numpy array, a list of numpy arrays, or
            a dictionary which recursively has values of the same kinds.
    """
    d = loadmat(path)
    keys = [k for k in d if not (k.startswith('__') and k.endswith('__'))]
    if len(keys) > 1:
        raise ValueError('Too many keys. Expected just one: {}'.format(keys))
    return _convert_void_to_dict(d[keys[0]])
