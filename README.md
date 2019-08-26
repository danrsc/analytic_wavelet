# analytic_wavelet
A translation of J.M. Lilly's code for ridge and element analysis using generalized Morse wavelets into python.

The original jLab repository can be found here: https://github.com/jonathanlilly/jLab

Lilly, J. M. (2019),  jLab: A data analysis package for Matlab, 
        v. 1.6.6, http://www.jmlilly.net/jmlsoft.html.

Note that this repository does not re-implement all of the code in jLab, only the parts that I was interested in. It is also not
a straight port. I have restructured the code to make the APIs more descriptive and to be more python / numpy friendly. I have
also replaced custom functions with built-in numpy and scipy functions where it was obvious to me that I could and I have changed the memory layout to be more efficient in Python (since numpy is row-major and MATLAB is column-major). That means the time axis is usually the last axis in my code but the first axis in jLab. Further simplifications to use more built-in numpy and scipy code can probably be done.

Rough mappings from functions in jLab to functions/methods in analytic_wavelet. Not all APIs map exactly

jLab                    analytic_wavelet
instmom                 ( amplitude, instantaneous_frequency, instantaneous_curvature, instantaneous_moments )
isomax                  ElementAnalysisMorse.isolated_maxima
lininterp               linear_interpolate
maxprops                ElementAnalysisMorse.event_parameters
morseafun               GeneralizedMorseWavelet.amplitude
morsebox                GeneralizedMorseWavelet.heisenberg_box
morsefreq               ( GeneralizedMorseWavelet.peak_frequency, 
                          GeneralizedMorseWavelet.energy_frequency, 
                          GeneralizedMorseWavelet.instantaneous_frequency,
                          GeneralizedMorseWavelet.curvature_instantaneous_frequency )
morsemom                ( GeneralizedMorseWavelet.frequency_domain_moment,
                          GeneralizedMorseWavelet.energy_moment,
                          GeneralizedMorseWavelet.frequency_domain_cumulants,
                          GeneralizedMorseWavelet.energy_cumulants ) 
morseprops              ( GeneralizedMorseWavelet.time_domain_width,
                          GeneralizedMorseWavelet.demodulated_skewness_imag,
                          GeneralizedMorseWavelet.demodulated_kurtosis )
morseregion             ElementAnalysisMorse.region_of_influence (localization region part of morseregion not implemented)
morsespace              GeneralizedMorseWavelet.log_spaced_frequencies
morsewave               GeneralizedMorseWavelet.make_wavelet
morsexpand              GeneralizedMorseWavelet.taylor_expansion_time_domain_wavelet
periodindex             period_indices
quadinterp              quadratic_interpolate
ridgewalk               ridges
ridgemap                ( RidgeResult.ridge_values, 
                          RidgeResult.instantaneous_frequency, 
                          RidgeResult.instantaneous_bandwidth,
                          RidgeResult.instantaneous_curvature,
                          RidgeResult.total_error,
                          RidgeResult.get_values,
                          RidgeResult.ridge_ids,
                          RidgeResult.collapse )
rot                     rotate
transmax                maxima_of_transform
transmaxdist            distribution_of_maxima_of_transformed_noise
wavespecplot            ( wavelet_contourf, time_series_plot )
wavetrans               analytic_wavelet_transform
