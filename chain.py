#!/usr/bin/python
"""
We will base the chain of states on LJ Ar4 cluster
as an exampel here:

    >>> from ase import Atoms
    >>> ar4 = Atoms("Ar4")

This builds a Func of Ar4 geometry using the
Lennard-Jones potential (the default):

    >>> from qfunc import QFunc
    >>> pes = QFunc(ar4)

This is a function of many geometries returning the
sum of their energies:

    >>> cha = Chain(pes)

Build a series of three geometries:

    >>> x = array([[  1.,  1.,  1. ],
    ...            [ -1., -1.,  1. ],
    ...            [  1., -1., -1. ],
    ...            [ -1.,  1., -1. ]])

    >>> xs = [ t * x for t in [0.3, 0.4, 0.5]]

Total energy of the chain:

    >>> cha(xs)
    99.360086640297595

Unconstrained minimization brings all instances to the
minimum:

    >>> from fopt import minimize
    >>> xms, e3, _ = minimize(cha, xs)

Three times -6.0 is the total energy after optimization:

    >>> from numpy import round
    >>> round(e3, 7)
    -18.0

One equilibrium:

    >>> w=0.39685026
    >>> A = array([[ w,  w,  w],
    ...            [-w, -w,  w],
    ...            [ w, -w, -w],
    ...            [-w,  w, -w]])

Another equilibrium (first two exchanged):

    >>> B = array([[-w, -w,  w],
    ...            [ w,  w,  w],
    ...            [ w, -w, -w],
    ...            [-w,  w, -w]])

Halfway between A and B (first two z-rotated by 90 deg):

    >>> C = array([[-w,  w,  w],
    ...            [ w, -w,  w],
    ...            [ w, -w, -w],
    ...            [-w,  w, -w]])

Three-membered chain:

    >>> xs = array([A, C, B])

Spline path through A-C-B:

    >>> from path import Path
    >>> p = Path(xs)

Make a chain with 5 images:

    >>> from numpy import linspace
    >>> x5 = [p(t) for t in linspace(0., 1., 5)]

Initial energies of the chain and the average:

    >>> es0 = array([pes(x) for x in x5])

    >>> round(es0, 4)
    array([ -6.    ,  30.2702,  92.9904,  30.2702,  -6.    ])

    >>> round(cha(x5) / 5., 4)
    28.3062

The images are not equally spaced:

    >>> spc = Spacing()

    >>> spc(x5)
    array([-0.15749013,  0.        ,  0.15749013])

Optimize the chain and enforce equal spacing:

    >>> xm, fm, stats = smin(cha, x5, spc, maxiter=100)

The optimized energies of the images:

    >>> es1 = array([pes(x) for x in xm])

    >>> round(es1, 4)
    array([-6.    , -6.    , -5.0734, -6.    , -6.    ])

Note that the actuall transition happens in the middle
of the chain. The two first and the two last images
differ (almost) exclusively by rotation in 3D.

The average energy is very close to -6:

    >>> round(cha(xm) / 5., 4)
    -5.8147000000000002

And the spacing is enforced:

    >>> round(spc(xm), 10)
    array([ 0.,  0., -0.])

You can visualize the path by executing:

#   >>> from aof.tools.jmol import jmol_view_path
#   >>> jmol_view_path(xm, syms=["Ar"]*4, refine=5)
"""

__all__ = ["smin", "Chain", "Spacing"]

from func import Func
from numpy import array, asarray, zeros, shape, sum, max, abs
from fopt import cmin, _flatten

def smin(ce, x, cs, **kwargs):
    """Minimize ce(x[1:-1]) with constrains cs(x[:]) = 0

    ce  --      chain energy function
    x   --      initial chain geometry, including the terminal points
    cs  --      chain spacing funciton, should be zeroes in course of
                optimization
    kwargs --   additional keyword arguments, passed as is to fopt.cmin()
    """

    # we are going to modify moving beads here:
    y = asarray(x).copy()

    # define the constrain function of n-2 geoms:
    def cg(z):
        y[1:-1] = z
        c, A = cs.taylor(y)
        #rint "c, A=", c, A
        #rint "c, A[:, 1:-1]=", c, A[:, 1:-1]
        return c, A[:, 1:-1]

    # geometies of moving beads:
    z = y[1:-1]

    # functions of flat arguments:
    fg = _flatten(ce.taylor, z)
    cg = _flatten(cg, z)

    # target values of constrains, request zeroes even if
    # initial path is not equispaced:
    c0 = zeros(len(z))

    zm, fm, stats = cmin(fg, z.flatten(), cg, c0=c0, **kwargs)

    # restore original shape:
    zm.shape = shape(z)
    y[1:-1] = zm

    return y, fm, stats

class Chain(Func):
    """
    """
    def __init__(self, f):
        self.__f = f

    def taylor(self, xs):

        # get a list of (f, fprime) pairs:
        fgs = [ self.__f.taylor(x) for x in xs ]
        # ^^^ place to parallelize!

        # separate values and gradients:
        fs, gs = zip(*fgs)

        return sum(fs), array(gs)

class Norm2(Func):
    """
    For an array of 2 geometries x, return a measure of their
    difference:

                         2
       f(x) = | x  -  x |
                 1     0

    and an array of their derivatives wrt x and x
                                           0     1
    An example:

        >>> n2 = Norm2()

        >>> x = array([(1., 1.,), (2., 1.), (2., 2.), (3.,2.)])

    Differences between adjacent geoms:

        >>> n2(x[0:2])
        1.0
        >>> n2(x[1:3])
        1.0
        >>> n2(x[2:4])
        1.0

    Note that x[0:2] stays for *two* adjacent geometries!

        >>> from func import NumDiff
        >>> N2 = NumDiff(n2)
        >>> max(abs(n2.fprime(x[1:3]) - N2.fprime(x[1:3]))) < 1.e-10
        True
    """

    def taylor(self, x):

        assert len(x) == 2

        # compute the difference of two geometries:
        d = x[1] - x[0]

        # the value:
        f = sum(d**2)

        # the derivative:
        fprime = zeros(shape(x))
        fprime[1] = + 2. * d
        fprime[0] = - 2. * d

        return f, fprime

class Spacing(Func):
    """
    For an array of n geometries x, return the n-2 differences:

       c(x) = (d(x   , x ) - d(x , x  )) / 2, i = 1, ... n-2
        i         i+1   i       i   i-1

    and a (sparse) array of their derivatives wrt x.

    An example:

        >>> cg = Spacing()
        >>> x = array([(1., 1.,), (2.01, 1.), (1.99, 2.), (3.,2.)])
        >>> cg(x)
        array([-0.00985,  0.00985])

        >>> from func import NumDiff
        >>> cg1 = NumDiff(cg)
        >>> max(abs(cg.fprime(x) - cg1.fprime(x))) < 1.e-10
        True
    """
    def __init__(self, dst=Norm2()):
        # mesure of the distance between two geoms:
        self.__dst = dst

    def taylor(self, x):

        # abbreviation for the (differentiable) distance function:
        dst = self.__dst.taylor

        n = len(x)

        # compute the differences of distances:
        c = []
        cprime = []
        for i in range(1, n - 1):

            # [i] and [i+1] one gets from [i:i+2]:
            dp, dpg = dst(x[i:i+2])

            # [i-1] and [i] one gets from [i-1:i+1]:
            dm, dmg = dst(x[i-1:i+1])

            # constrain value:
            c.append((dp - dm) / 2.0)

            # constrain derivative:
            sparse = zeros(shape(x))
            sparse[i+1] =  dpg[1]           / 2.
            sparse[i]   = (dpg[0] - dmg[1]) / 2.
            sparse[i-1] =         - dmg[0]  / 2.

            cprime.append(sparse)

        return array(c), array(cprime)

# python chain.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
