# analytic_wavelet_transform
import numpy as np
from analytic_wavelet import analytic_wavelet_transform, GeneralizedMorseWavelet, unpad, rotate, \
    to_frequency_domain_wavelet, make_unpad_slices, masked_detrend, load_test_data, analytic_transform

npg2006 = load_test_data('test_data/npg2006.mat')
cx = np.squeeze(npg2006['cx'], axis=1)
cv = np.squeeze(npg2006['cv'], axis=1)

# mismatched length
fs = 2 * np.pi / np.logspace(np.log10(10),np.log10(100),50)
print(fs)
try:
    psi, psi_f = GeneralizedMorseWavelet(2, 4).make_wavelet(len(cx) + 10, fs)
    analytic_wavelet_transform(np.real(cx), psi_f, np.isrealobj(psi))
    assert(False)
except ValueError:
    pass

# complex
psi, psi_f = GeneralizedMorseWavelet(2, 4).make_wavelet(len(cx), fs)
print(psi)
wx = analytic_wavelet_transform(np.real(cx), psi_f, np.isrealobj(psi))
wy = analytic_wavelet_transform(np.imag(cx), psi_f, np.isrealobj(psi))
wp = analytic_wavelet_transform(cx, psi_f, np.isrealobj(psi))
wn = analytic_wavelet_transform(np.conj(cx), psi_f, np.isrealobj(psi))
tmat = np.array([[1, 1j], [1, -1j]]) / np.sqrt(2)
z = np.matmul(tmat, np.concatenate([np.expand_dims(wx, 1), np.expand_dims(wy, 1)], axis=1))
wp2, wn2 = z[:, 0, :], z[:, 1, :]
assert(np.allclose(wp2, wp, equal_nan=True) and np.allclose(wn2, wn, equal_nan=True))

