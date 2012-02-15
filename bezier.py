#!/usr/bin/env python
"""
"""

__all__ = ["Bezier"]

from numpy import asarray, shape, ones, empty
from npz import outer
from func import Func

class Bezier(Func):
    """
        >>> from numpy import array

        >>> b = Bezier()

        >>> p = array([[10.,  5., 3.],
        ...            [ 3., 10., 5.]])

        >>> t = [0.0, 0.5, 1.0]
        >>> b(t, p)
        array([[ 10.  ,   3.  ],
               [  5.75,   7.  ],
               [  3.  ,   5.  ]])

        >>> b.fprime(t, p)[0]
        array([[-10.,  14.],
               [ -7.,   2.],
               [ -4., -10.]])

        >>> b.fprime(t, p)[1]
        array([[[ 1.  ,  0.  ,  0.  ],
                [ 1.  ,  0.  ,  0.  ]],
        <BLANKLINE>
               [[ 0.25,  0.5 ,  0.25],
                [ 0.25,  0.5 ,  0.25]],
        <BLANKLINE>
               [[ 0.  ,  0.  ,  1.  ],
                [ 0.  ,  0.  ,  1.  ]]])
    """

    def __init__(self):
        pass

    def f(self, t, p): # t: (j), p: (i, k)

        # f: (j, i, k)
        return casteljau(t, p)

    def fprime(self, t, p): # t: (j), p: (i, k)

        # degree:
        n = shape(p)[-1] - 1 # (k) == (n+1,)

        # ft: (j, i)
        ft = n * casteljau(t, p[..., 1:] - p[..., :-1])

        # bp: (j, k)
        b = bernstein(t, n)

        onei = ones(shape(p[..., 0])) # (i)

        # fp: (j, i, k)
        fp = empty(shape(t) + shape(p))

        for k in range(n + 1):
            fp[..., k] = outer(b[..., k], onei)

        return ft, fp

def fac(n):
    """
        >>> map(fac, [0, 1, 2, 3, 4, 10])
        [1.0, 1.0, 2.0, 6.0, 24.0, 3628800.0]
    """
    f = 1.
    for k in xrange(2, n + 1): f *= k
    return f

def binom(n, k):
    """
        >>> [ [binom(n, k) for k in range(n+1)] for n in range(4)]
        [[1.0], [1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 3.0, 3.0, 1.0]]
    """
    return fac(n) / fac(k) / fac(n-k)

def bernstein(t, n):
    """

    t: (j), n: (),  bernstein(t, p): (j, n)

        >>> from numpy import array

        >>> t = [0.0, 0.25, 0.5, 0.75, 1.0]

        >>> bernstein(t, 3)
        array([[ 1.      ,  0.      ,  0.      ,  0.      ],
               [ 0.421875,  0.421875,  0.140625,  0.015625],
               [ 0.125   ,  0.375   ,  0.375   ,  0.125   ],
               [ 0.015625,  0.140625,  0.421875,  0.421875],
               [ 0.      ,  0.      ,  0.      ,  1.      ]])

        >>> from numpy import sum

        >>> sum(bernstein(t, 3), axis=1)
        array([ 1.,  1.,  1.,  1.,  1.])
    """

    t = asarray(t)

    x = empty(shape(t) + (n + 1,))
    y = empty(shape(t) + (n + 1,))
    b = empty(shape(t) + (n + 1,))

    # powers of t and (1 - t):
    x[..., 0] = 1.
    y[..., 0] = 1.
    for k in xrange(n):
        x[..., k+1] = x[..., k] * t
        y[..., k+1] = y[..., k] * (1 - t)

    # polynomials:
    for k in xrange(n + 1):
        b[..., k] = binom(n, k) * x[..., k] * y[..., n - k]

    return b

def casteljau(t, p):
    """de Casteljau's algorythm for evaluating Bernstein forms,
    e.g. for the third order:

                 3               2          2              3
    C (t) = (1-t) * P  +  3t(1-t) * P  +  3t(1-t) * P  +  t * P
     3               0               1               2         3

    where [P0, P1, P2, P3] are taken from ps[:]

    Should also work for array valued (e.g. 2D) points.
    Here an array of shape (2, 3) parametrizing a Bezier
    line in 2D by three parameters:

        >>> from numpy import array

        >>> p = array([[10.,  5., 3.],
        ...            [ 3., 10., 5.]])

        >>> casteljau(0.5, p)
        array([ 5.75,  7.  ])

    Compare with this:

        >>> casteljau(0.5, [10., 5., 3.])
        array(5.75)
        >>> casteljau(0.5, [3., 10., 5.])
        array(7.0)

        >>> casteljau([0.0, 0.5, 1.0], p)
        array([[ 10.  ,   3.  ],
               [  5.75,   7.  ],
               [  3.  ,   5.  ]])

    """

    t = asarray(t)
    p = asarray(p)

    # polynomial order:
    n = shape(p)[-1] - 1

    # a copy of ps, will be modifying this:
    b = outer(ones(shape(t)), p) # \beta^0

    x = outer(1. - t, ones(shape(p[..., 0])))
    y = outer(     t, ones(shape(p[..., 0])))

    for j in xrange(n):
        for i in xrange(n - j):
            b[..., i] = x * b[..., i] + y * b[..., i+1]

    return b[..., 0] # \beta^{n}_0

# python bezier.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
