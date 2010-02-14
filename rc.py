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

__all__ = ["Volume", "Distance", "Angle", "Dihedral"]

from func import Func
from numpy import array, zeros, shape, cross, dot, max, abs
from numpy import sqrt, cos, sin, arccos

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

class Distance(Func):
    """Cartesian distance between two points

    An example:

        >>> d = Distance()

        >>> x = array([(3., 0., 0.), (0., 4., 0.)])

        >>> d(x)
        5.0

    Verify derivatives:

        >>> from func import NumDiff
        >>> max(abs(d.fprime(x) - NumDiff(d).fprime(x))) < 1.e-10
        True
    """
    def __init__(self, two=[0, 1]):
        # indices of two points to use:
        self.__two = two

    def taylor(self, x):

        # indices of four points in 3D to use:
        i0, i1 = self.__two

        d = x[i1] - x[i0]

        # the value:
        f = sqrt(dot(d, d))

        # the derivatives wrt d:
        fd = d / f

        # final derivatives:
        fprime = zeros(shape(x))
        fprime[i0] = - fd
        fprime[i1] = + fd

        return f, fprime

class Angle(Func):
    """Angle between three points

    An example:

        >>> a = Angle()

        >>> x = array([(3., 0., 0.), (0., 0., 0.), (0., 4., 0.)])

        >>> from math import pi
        >>> a(x) / pi * 180.
        90.0

    Verify derivatives:

        >>> from func import NumDiff
        >>> max(abs(a.fprime(x) - NumDiff(a).fprime(x))) < 1.e-10
        True
    """
    def __init__(self, three=[0, 1, 2]):
        # indices of three points to use:
        self.__three = three

    def taylor(self, x):

        # indices of four points in 3D to use:
        i0, i1, i2 = self.__three

        a = x[i0] - x[i1]
        b = x[i2] - x[i1]

        la = sqrt(dot(a, a))
        lb = sqrt(dot(b, b))

        a /= la
        b /= lb

        # cosine:
        cs = dot(a, b)

        # it happens:
        if cs > 1.:
            assert cs - 1. < 1.e-10
            cs = 1.

        if cs < -1.:
            assert -1. - cs < 1.e-10
            cs = -1.

        # the value:
        f = arccos(cs)

        # sine:
        si = sin(f)

        # the derivatives wrt a, b:
        fa = (cs * a - b) / (la * si)
        fb = (cs * b - a) / (lb * si)
        # FIXME: small angles?

        # final derivatives:
        fprime = zeros(shape(x))
        fprime[i0] = + fa
        fprime[i1] = - fa - fb
        fprime[i2] = + fb

        return f, fprime

class Dihedral(Func):
    """Dihedral angle formed by four points

    An example:

        >>> h = Dihedral()

        >>> xp = array([(0., 0., 0.), (2., 0., 0.), (2., 2., 0.), (2., 2., +2.)])
        >>> xm = array([(0., 0., 0.), (2., 0., 0.), (2., 2., 0.), (2., 2., -2.)])

        >>> from math import pi
        >>> h(xp) / pi * 180.
        90.0
        >>> h(xm) / pi * 180.
        -90.0

    Verify derivatives:

        >>> from func import NumDiff
        >>> h1 = NumDiff(h)

        >>> def check(x):
        ...     if max(abs(h.fprime(x) - h1.fprime(x))) > 1.e-8:
        ...         print "derivatives fail for x = "
        ...         print x
        ...         print "numerical:"
        ...         print h1.fprime(x)
        ...         print "analytical:"
        ...         print h.fprime(x)

        >>> check(xp)
        >>> check(xm)

        >>> xp = array([(0., 0., 0.), (2., 0., 0.), (0., 2., 0.), (0., 0., +2.)])
        >>> xm = array([(0., 0., 0.), (2., 0., 0.), (0., 2., 0.), (0., 0., -2.)])

        >>> h(xp) / pi * 180.
        54.735610317245339
        >>> h(xm) / pi * 180.
        -54.735610317245339

        >>> check(xp)
        >>> check(xm)

        >>> xp = array([(0., 0., 0.), (2., 0., 0.), (0., 2., 0.), (4., 4., +2.)])
        >>> xm = array([(0., 0., 0.), (2., 0., 0.), (0., 2., 0.), (4., 4., -2.)])

        >>> h(xp) / pi * 180.
        154.76059817932108
        >>> h(xm) / pi * 180.
        -154.76059817932108

        >>> check(xp)
        >>> check(xm)

        >>> xp = array([(0., 0., 0.), (2., 0., 0.), (2., 2., 0.), (0., 2., 0.)])
        >>> xm = array([(0., 0., 0.), (2., 0., 0.), (2., 2., 0.), (4., 2., 0.)])

        >>> h(xp) / pi * 180.
        0.0
        >>> h(xm) / pi * 180.
        180.0

        >>> check(xp)

    Numerical differentiation fails for dihedral angle 180:

#       >>> check(xm)

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

        # one plane normal, A=A(a,b):
        A = cross(a, b)
        LA = sqrt(dot(A, A))
        A /= LA

        # another plane normal, B=B(b,c):
        B = cross(b, c)
        LB = sqrt(dot(B, B))
        B /= LB

        # cosine:
        cs = dot(A, B)

        # it happens:
        if cs > 1.:
            assert cs - 1. < 1.e-10
            cs = 1.

        if cs < -1.:
            assert -1. - cs < 1.e-10
            cs = -1.

        # angle between two planes, f=f(A,B):
        f = arccos(cs)

        #if True:
        #if False:
        if abs(sin(f)) > 0.1: # division by sin(f)

            # sine:
            si = sin(f)

            # derivatives wrt A, B, note: cos(f) = dot(A, B):
            fA = (cs * A - B) / (LA * si)
            fB = (cs * B - A) / (LB * si)
            # FIXME: small angles?

            # derivatives wrt a, b, c:
            fa = cross(b, fA)
            fb = cross(fA, a) + cross(c, fB)
            fc = cross(fB, b)

        else: # division by cos(f):

            # 90-deg rotated A, C=C(a,b):
            C = cross(b, cross(a, b)) # = a * (b,b) - b * (a,b)
            LC = sqrt(dot(C, C))
            C /= LC

            # sine:
            si = dot(C, B)

            #ssert abs(dot(C, B) - si) < 1.0e-10

            # derivatives wrt B, C, note: sin(f) = dot(C, B)
            fB = (C - si * B) / (LB * cs)
            fC = (B - si * C) / (LC * cs)

            # derivatives wrt a, b, c:
            fa = dot(b, b) * fC - b * dot(b, fC)
            fb = 2. * b * dot(a, fC) - a * dot(b, fC) - fC * dot(a, b) \
               + cross(c, fB)
            fc = cross(fB, b)

            # FXIME: this magic I cannot understand:
            if dot(a, cross(b, c)) < 0:
                fa, fb, fc = -fa, -fb, -fc

        # see if 0-1-2-3-skew is (anti)parallel to the base:
        if dot(a, cross(b, c)) < 0:
            f, fa, fb, fc = -f, -fa, -fb, -fc

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
