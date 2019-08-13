import numpy as np


from .analytic_wavelet import GeneralizedMorseWavelet, analytic_wavelet_transform, transform_maxima


__all__ = ['element_parameters']


def element_parameters(x, gamma, beta, mu):
    morse = GeneralizedMorseWavelet(gamma, beta)
    scale_frequencies = morse.log_spaced_frequencies(x.shape[-1])
    _, psi_f = morse.make_wavelet(x.shape[-1], scale_frequencies)
    x = analytic_wavelet_transform(x, psi_f, False)
    maxima_indices, maxima_values, maxima_scale_frequencies = transform_maxima(scale_frequencies, x)


