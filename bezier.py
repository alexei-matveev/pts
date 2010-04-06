#!/usr/bin/python
"""
"""

__all__ = []

from numpy import array, asarray, shape, ones
from npz import outer
from func import Func

class Bezier(Func):
    def __init__(self):
        pass

    def taylor(self, t, p):

        f = casteljau(t, p)

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
