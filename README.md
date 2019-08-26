# analytic_wavelet
A translation of J.M. Lilly's code for ridge and element analysis using generalized Morse wavelets into python.

The original jLab repository can be found here: https://github.com/jonathanlilly/jLab

Lilly, J. M. (2019),  jLab: A data analysis package for Matlab, 
        v. 1.6.6, http://www.jmlilly.net/jmlsoft.html.

Note that this repository does not re-implement all of the code in jLab, only the parts that I was interested in. It is also not
a straight port. I have restructured the code to make the APIs more descriptive and to be more python / numpy friendly. I have
also replaced custom functions with built-in numpy and scipy functions where it was obvious to me that I could and I have changed the memory layout to be more efficient in Python (since numpy is row-major and MATLAB is column-major). That means the time axis is usually the last axis in my code but the first axis in jLab. Further simplifications to use more built-in numpy and scipy code can probably be done.

Rough mappings from functions in jLab to functions/methods in analytic_wavelet. Not all APIs map exactly

| jLab         | analytic_wavelet |
| ------------ | ---------------- |
| anatrans | analytic_transform |
| instmom |  <ul><li>amplitude</li><li>instantaneous_frequency</li><li>instantaneous_curvature</li><li>instantaneous_moments</li></ul> |
| isomax  | ElementAnalysisMorse.isolated_maxima |
| lininterp | linear_interpolate |
| maxprops | ElementAnalysisMorse.event_parameters |
| morseafun | GeneralizedMorseWavelet.amplitude |
| morsebox | GeneralizedMorseWavelet.heisenberg_box |
| morsefreq | <ul><li>GeneralizedMorseWavelet.peak_frequency</li><li>GeneralizedMorseWavelet.energy_frequency</li><li>GeneralizedMorseWavelet.instantaneous_frequency</li><li>GeneralizedMorseWavelet.curvature_instantaneous_frequency</li></ul> |
| morsemom | <ul><li>GeneralizedMorseWavelet.frequency_domain_moment</li><li>GeneralizedMorseWavelet.energy_moment</li><li>GeneralizedMorseWavelet.frequency_domain_cumulants</li><li>GeneralizedMorseWavelet.energy_cumulants</li></ul> |
| morseprops | <ul><li>GeneralizedMorseWavelet.time_domain_width</li><li>GeneralizedMorseWavelet.demodulated_skewness_imag</li><li>GeneralizedMorseWavelet.demodulated_kurtosis</li></ul> |
| morseregion | ElementAnalysisMorse.region_of_influence (localization region part of morseregion not implemented) |
| morsespace | GeneralizedMorseWavelet.log_spaced_frequencies |
| morsewave | GeneralizedMorseWavelet.make_wavelet |
| morsexpand | GeneralizedMorseWavelet.taylor_expansion_time_domain_wavelet |
| periodindex | period_indices |
| quadinterp | quadratic_interpolate |
| ridgewalk | ridges |
| ridgemap | <ul><li>RidgeResult.ridge_values</li><li>RidgeResult.instantaneous_frequency</li><li>RidgeResult.instantaneous_bandwidth</li><li>RidgeResult.instantaneous_curvature</li><li>RidgeResult.total_error</li><li>RidgeResult.get_values</li><li>RidgeResult.ridge_ids</li><li>RidgeResult.collapse</li></ul> |
| rot | rotate |
| transmax | maxima_of_transform |
| transmaxdist | distribution_of_maxima_of_transformed_noise |
| wavespecplot | <ul><li>wavelet_contourf</li><li>time_series_plot</li></ul> |
| wavetrans | <ul><li>analytic_wavelet_transform</li><li>to_frequency_domain_wavelet</li><li>masked_detrend</li><li>make_unpad_slices</li><li>unpad</li></ul> |
