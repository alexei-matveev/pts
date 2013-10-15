"""
Discrete Cosine Transform

              N-1
    y[k] = 2* sum x[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
              n=0

Inverse Discrete Cosine Transform

               N-1
    x[k] = 1/N sum w[n]*y[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
               n=0

    w(0) = 1/2
    w(n) = 1 for n>0

Examples
--------

    >>> import numpy as np
    >>> x = np.arange(5.0)
    >>> np.abs(x-idct(dct(x))) < 1e-14
    array([ True,  True,  True,  True,  True], dtype=bool)
    >>> np.abs(x-dct(idct(x))) < 1e-14
    array([ True,  True,  True,  True,  True], dtype=bool)

FIXME: Get rid  of the normalization, the user  code should not assume
anymore that icdt(dct(x)) == x. Because this is how SciPy does it.

References
----------

http://en.wikipedia.org/wiki/Discrete_cosine_transform
http://users.ece.utexas.edu/~bevans/courses/ee381k/lectures/
"""

# Must be available in SciPy 0.8, see
# http://projects.scipy.org/scipy/ticket/733
from scipy.fftpack import dct as DCT, idct as IDCT

__all__ = ["dct", "idct"]

def dct (x):
    return DCT (x)

def idct (x):
    return IDCT (x) * (1.0 / (2 * len (x))) 

# python dct.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
