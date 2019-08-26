from collections import OrderedDict
import numpy as np
from scipy.io import loadmat


__all__ = ['load_test_data', 'latlon2xy', 'latlon2uv', 'sphere_dist']


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


# these functions are ocean specific, but used in the test code
def _cosd(x):
    return np.cos(np.radians(x))


def _sind(x):
    return np.sin(np.radians(x))


def latlon2xy(lat, lon, lat0, lon0):
    x = np.full_like(lat, np.nan)
    y = np.full_like(lon, np.nan)
    # broadcast
    lat0 = np.zeros_like(lat) + lat0
    lon0 = np.zeros_like(lon) + lon0
    cos_lat = _cosd(lat)
    cos_lon = _cosd(lon)
    cos_lat0 = _cosd(lat0)
    sin_lat0 = _sind(lat0)
    sin_lat = _sind(lat)
    indicator = cos_lat * cos_lon * cos_lat0 + sin_lat0 * sin_lat > 0
    radius_earth = 6371
    x[indicator] = radius_earth * cos_lat[indicator] * _sind(lon[indicator])
    y[indicator] = (-radius_earth * cos_lat[indicator] * sin_lat0[indicator] * cos_lon[indicator]
                    + radius_earth * cos_lat0[indicator] * sin_lat[indicator])
    return x, y


def sphere_dist(lat1, lon1, lat2, lon2):
    if (np.any(np.logical_and(np.isfinite(lat1), np.abs(lat1) > 90))
            or np.any(np.logical_and(np.isfinite(lat2), np.abs(lat2)) > 90)):
        raise ValueError('invalid latitude')
    a1 = np.square(np.abs(_sind((lat2 - lat1) / 2)))
    a2 = _cosd(lat1) * _cosd(lat2) * np.square(np.abs(_sind((lon2 - lon1) / 2)))
    a = (np.square(np.abs(_sind((lat2 - lat1) / 2)))
         + _cosd(lat1) * _cosd(lat2) * np.square(np.abs(_sind((lon2 - lon1) / 2))))
    a = np.minimum(a, 1)
    radius_earth = 6371
    return radius_earth * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _latlon2uv_helper(num, lat, lon, flip=False):
    if flip:
        num = np.flip(num)
        lat = np.flip(lat)
        lon = np.flip(lon)

    lon = np.degrees(np.unwrap(np.radians(lon)))
    dt = np.diff(num * 24 * 3600)
    lat2 = np.roll(lat, 1)
    lon2 = np.roll(lon, 1)
    dr = sphere_dist(lat, lon, lat2, lon2)
    x = _sind(lon2 - lon) * _cosd(lat2)
    y = _cosd(lat) * _sind(lat2) - _sind(lat) * _cosd(lat2) * _cosd(lon2 - lon)
    gamma = np.arctan2(y, x)
    dt = np.concatenate([dt, np.array([dt[-1]])])
    dr[-1] = dr[-2]
    gamma[-1] = gamma[-2]

    if flip:
        dt = np.flip(dt)
        dr = np.flip(dr)
        gamma = np.flip(gamma)

    c = 100 * 1000
    u = c * dr / dt * np.cos(gamma)
    v = c * dr / dt * np.sin(gamma)
    indicator = np.logical_or(np.isnan(u), np.isnan(v))
    u[indicator] = np.nan
    v[indicator] = np.nan
    return u, v


def latlon2uv(num, lat, lon):
    u1, v1 = _latlon2uv_helper(num, lat, lon, flip=False)
    u2, v2 = _latlon2uv_helper(num, lat, lon, flip=True)

    u2[0] = u1[0]
    v2[0] = v1[0]
    u1[-1] = u2[-1]
    v1[-1] = v2[-1]

    return 1 / 2 * (u1 + u2) + 1j * 1 / 2 * (v1 + v2)
