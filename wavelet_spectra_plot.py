import numpy as np

__all__ = ['wavelet_contourf', 'time_series_plot']


def wavelet_contourf(ax, time, frequencies, wavelet_coefficients, power=1, **contourf_args):
    time = np.asarray(time)
    frequencies = np.asarray(frequencies)
    wavelet_coefficients = np.asarray(wavelet_coefficients)
    if time.ndim != 1:
        raise ValueError('Expected 1d array for time')
    if frequencies.ndim != 1:
        raise ValueError('Expected 1d array for frequencies')
    if wavelet_coefficients.ndim != 2:
        raise ValueError('Expected 2d array for wavelet_coefficients')
    if wavelet_coefficients.shape != (len(frequencies), len(time)):
        raise ValueError('Expected wavelet_coefficients to have shape ({}, {}). Actual shape: {}'.format(
            len(frequencies), len(time), wavelet_coefficients.shape))
    if not np.isrealobj(wavelet_coefficients) or power != 1:
        wavelet_coefficients = np.abs(wavelet_coefficients)
    if power != 1:
        wavelet_coefficients = np.power(wavelet_coefficients, power)
    time, frequencies = np.meshgrid(time, frequencies)
    return ax.contourf(time, frequencies, wavelet_coefficients, **contourf_args)


def time_series_plot(ax, time, time_series):
    if not np.isrealobj(time_series):
        u = np.real(time_series)
        v = np.imag(time_series)
        ax.plot(time, u)
        ax.plot(time, v)
    else:
        ax.plot(time, time_series)
