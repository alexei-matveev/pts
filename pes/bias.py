#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
"""
"""

from numpy import zeros, shape, asarray
from pts.func import Func
from pts.rc import _distance

class Bias (Func):
    """
    >>> from numpy import pi, array, max, abs, all
    >>> ea = Bias (1.25, 8.0, [0, 1])
    >>> eb = Bias (1.25, 8.0, [0, 2])
    >>> x = array ([[0., 0., 0.,],
    ...             [0., 0., 1.5],
    ...             [0., -1.0, 0.]])
    >>> e = ea + eb
    >>> e (x)
    0.5
    >>> from pts.func import NumDiff
    >>> e1 = NumDiff (e)
    >>> max (abs (e.fprime (x) - e1.fprime (x))) < 1.0e-12
    True
    """
    def __init__ (self, r0, k, two=[]):

        # Save force field parameters and atom indices:
        self.r0 = r0
        self.k = k
        self.two = two

    def taylor (self, x):
        x = asarray (x)

        # Pick coordinates of two atoms
        y = x[self.two]
        a, b = self.two

        # Bond length and force constant:
        r0, k = self.r0, self.k

        r, rprime = _distance ([y[0], y[1]])
        # print ("r=", r, "a=", a, "b=", b, "xa=", y[0], "xb=", y[1])
        e = 0.5 * k * (r - r0)**2
        fa = k * (r - r0) * rprime[0]
        fb = k * (r - r0) * rprime[1]

        g = zeros (shape (x))
        g[a] = fa
        g[b] = fb

        return e, g

# python ab2.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
