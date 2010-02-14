#!/usr/bin/python
"""
We will base reaction coordinate definiiton on Ar4 cluster
as an example here:

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

Planar structure halfway between A and B
(first two z-rotated by 90 deg):

    >>> C = array([[-w,  w,  w],
    ...            [ w, -w,  w],
    ...            [ w, -w, -w],
    ...            [-w,  w, -w]])

The "volume" function for three geometries:

    >>> v = Volume()

    >>> round(v(A), 7)
    -1.0
    >>> round(v(B), 7)
    1.0
    >>> round(v(C), 7)
    0.0

Three-membered chain:

    >>> xs = array([A, C, B])

Spline path through A-C-B:

    >>> from path import Path
    >>> p = Path(xs)

Make a chain with 5 images:

    >>> from numpy import linspace
    >>> x5 = [p(t) for t in linspace(0., 1., 5)]

Define LJ-PES:

    >>> from ase import Atoms
    >>> from qfunc import QFunc

    >>> pes = QFunc(Atoms("Ar4"))

Chain energy function to minimize:

    >>> from chain import Chain
    >>> cha = Chain(pes)

Base the spacing of the images along the path on differences
of the "volume" as a reaction coordinate:

    >>> from chain import Spacing, RCDiff
    >>> spc = Spacing(RCDiff(Volume()))

Optimize the chain and enforce equal spacing:

    >>> from chain import smin
    >>> xm, fm, stats = smin(cha, x5, spc)

The optimized energies of the images:

    >>> es1 = array([pes(x) for x in xm])

    >>> from numpy import round
    >>> round(es1, 4)
    array([-6.    , -4.533 , -4.4806, -4.533 , -6.    ])

Note that the transition is distributed over more
than one interval. However the TS approximation is
square planar:

    >>> round(xm[2], 4)
    array([[-0.3934,  0.3934,  0.5563],
           [ 0.3934, -0.3934,  0.5563],
           [ 0.3934, -0.3934, -0.5563],
           [-0.3934,  0.3934, -0.5563]])

an therefore relatively high in energy. Is there a reason for it?

The equal spacing is enforced:

    >>> round(spc(xm), 10)
    array([ 0., -0., -0.])

You can visualize the path by executing:

#   >>> from aof.tools.jmol import jmol_view_path
#   >>> jmol_view_path(xm, syms=["Ar"]*4, refine=5)

The Volume function can be used to model the dihedral angle
as a reaction coordinate. You may want to specify the
four indices in such case, the order does matter,
of course:

    >>> v1 = Volume((0,1,3,2))
    >>> round(v1(A), 7), round(v(A), 7)
    (1.0, -1.0)
"""

__all__ = ["Volume"]

from func import Func
from numpy import array, zeros, shape, cross, dot, max, abs

class Volume(Func):
    """For an array of 4 vectors x, return a measure of their
    (signed) volume:

       v(x) = [ ( x  -  x ) x ( x  - x ) ] * ( x  - x )
                   1     0       2    1         3    2

    Here "x" and "*" stay for cross- and dot-products respectively.

    An example:

        >>> v = Volume()

        >>> x = array([(0., 0., 0.), (2., 0., 0.), (0., 2., 0.), (0., 0., 2.)])

    Volume of the cube with side 2:

        >>> v(x)
        8.0

    Verify derivatives:

        >>> from func import NumDiff
        >>> v1 = NumDiff(v)
        >>> max(abs(v.fprime(x) - v1.fprime(x))) < 1.e-10
        True
    """
    def __init__(self, four=[0, 1, 2, 3]):
        # indices of four points in 3D to use:
        self.__four = four

    def taylor(self, x):

        # indices of four points in 3D to use:
        i0, i1, i2, i3 = self.__four

        a = x[i1] - x[i0]
        b = x[i2] - x[i1]
        c = x[i3] - x[i2]

        # the value:
        f = dot(cross(a, b), c)

        # the derivatives wrt a, b, and c:
        fc = cross(a, b)
        fb = cross(c, a)
        fa = cross(b, c)

        # final derivatives:
        fprime = zeros(shape(x))
        fprime[i0] =    - fa
        fprime[i1] = fa - fb
        fprime[i2] = fb - fc
        fprime[i3] = fc

        return f, fprime

# python rc.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
