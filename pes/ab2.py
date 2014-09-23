#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
"""
"""

from numpy import dot, sqrt, exp, zeros, shape, asarray
from pts.func import Func
from pts.rc import _distance, _angle

class AB2 (Func):
    """
    >>> from numpy import pi, array, max, abs, all
    >>> e = AB2 ((1.25, 8.0), (pi / 2 + 0.5, 0.125))
    >>> x = array ([[0., 0., 0.,],
    ...             [0., 0., 1.5],
    ...             [0., -1.25, 0.]])
    >>> e (x)
    0.265625
    >>> from pts.func import NumDiff
    >>> e1 = NumDiff (e)
    >>> max (abs (e.fprime (x) - e1.fprime (x))) < 1.0e-12
    True

    >>> e = AB2 ((1.25, 8.0), (pi, 0.125))
    >>> x = array ([[0., 0., 0.,],
    ...             [0., 0., 1.25],
    ...             [0., 0., -1.25]])
    >>> e (x)
    >>> all (e.fprime (x) == 0.0)
    True
    """
    def __init__ (self, stretching, bending, three=[0, 1, 2]):

        # Save force field parameters and atom indices:
        self.stretching = stretching
        self.bending = bending
        self.three = three

    def taylor (self, x):
        x = asarray (x)

        # Pick coordinates of three atoms
        y = x[self.three]

        e = 0.0
        eprime = zeros (shape (y))

        # Bond length and force constant:
        r0, k = self.stretching

        r, rprime = _distance ([y[0], y[1]])
        e += 0.5 * k * (r - r0)**2
        eprime[0] += k * (r - r0) * rprime[0]
        eprime[1] += k * (r - r0) * rprime[1]

        r, rprime = _distance ([y[0], y[2]])
        e += 0.5 * k * (r - r0)**2
        eprime[0] += k * (r - r0) * rprime[0]
        eprime[2] += k * (r - r0) * rprime[1]

        # Angle and force constant:
        a0, k = self.bending

        # FIXME: _angle() computes 0-1-2 angle, we want 1-0-2 here:
        a, aprime = _angle (y[[1, 0, 2]])
        # print ("a=", a, "a0=", a0)
        e += 0.5 * k * (a - a0)**2
        eprime[0] += k * (a - a0) * aprime[1]
        eprime[1] += k * (a - a0) * aprime[0]
        eprime[2] += k * (a - a0) * aprime[2]

        g = zeros (shape (x))
        a, b, c = self.three
        g[a] = eprime[0]
        g[b] = eprime[1]
        g[c] = eprime[2]

        return e, g

# python ab2.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
