# analytic_wavelet
A translation of J.M. Lilly's code for ridge and element analysis using generalized Morse wavelets into python.

The original jLab repository can be found here: https://github.com/jonathanlilly/jLab

Lilly, J. M. (2019),  jLab: A data analysis package for Matlab, 
        v. 1.6.6, http://www.jmlilly.net/jmlsoft.html.

Note that this repository does not re-implement all of the code in jLab, only the parts that I was interested in. It is also not
a straight port. I have restructured the code to make the APIs more descriptive and to be more python / numpy friendly. I have
also replaced custom functions with built-in numpy and scipy functions where it was obvious to me that I could and I have changed the memory layout to be more efficient in Python (since numpy is row-major and MATLAB is column-major). That means the time axis is usually the last axis in my code but the first axis in jLab. Further simplifications to use more built-in numpy and scipy code can probably be done.
