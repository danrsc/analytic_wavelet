import numpy as np

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
        mask: A boolean array which broadcasts to the shape of x. If a time-scale point evaluates is False in the mask
            it will not be considered for the ridge. This enables the use of auxiliary information such as noise
            estimates at particular time-scale locations
        alpha: Controls aggressiveness of chaining across scales. alpha == (d_omega / omega) where omega is the
            transform frequency and d_omega is the difference between the frequency predicted for the next point
            based on the transform at a 'tail', and the actual frequency at prospective 'heads'. This mostly does not
            need to change, but for strongly chirping or weakly chirping noisy signals better performance may be
            obtained by adjusting it
        variable_axis: If specified, then the ridge is computed jointly over this axis

    Returns:
        ridge: The transform values along the ridge
        indices: A tuple of indices such that ridge = x[indices]
        instantaneous_frequencies: Instantaneous frequencies along the ridge
        instantaneous_bandwidth: Instantaneous bandwidth along the ridge
        normalized_instantaneous_curvature: Normalized instantaneous curvature along the ridge as a measure of error.
            Should be << 1 if the ridge estimate is good. Only returned if morse_wavelet is provided
    """

    min_length = None
    if min_wavelet_lengths_in_ridge is not None:
        if morse_wavelet is None:
            raise ValueError('If min_wavelet_lengths in ridge is given, morse_wavelet must be specified')
        min_length = 2 * morse_wavelet.time_domain_width() / np.pi

