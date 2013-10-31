#!/usr/bin/env python
"""
We will base reaction coordinate definiiton on Ar4 cluster
as an example here:

One equilibrium:

    >>> from numpy import array, max, abs

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

    >>> max(abs(spc(xm))) < 1.e-8
    True

You can visualize the path by executing:

#   >>> from pts.tools.jmol import jmol_view_path
#   >>> jmol_view_path(xm, syms=["Ar"]*4, refine=5)

The Volume function can be used to model the dihedral angle
as a reaction coordinate. You may want to specify the
four indices in such case, the order does matter,
of course:

    >>> v1 = Volume([0,1,3,2])
    >>> round(v1(A), 7), round(v(A), 7)
    (1.0, -1.0)

Here we use dihedral angle for image spacing:

    >>> dih = Dihedral()

Optimize the chain and enforce equal spacing:

    >>> xm, fm, stats = smin(cha, x5, Spacing(RCDiff(dih)))

The optimized energies of the images:

    >>> es2 = array(map(pes, xm))

    >>> print es2
    [-6.         -5.10464043 -4.48062016 -5.10464043 -6.        ]

Different energy profile, same planar TS approximation:

    >>> print round(xm[2], 4)
    [[-0.3934  0.3934  0.5563]
     [ 0.3934 -0.3934  0.5563]
     [ 0.3934 -0.3934 -0.5563]
     [-0.3934  0.3934 -0.5563]]

The equal spacing is enforced:

    >>> from numpy import pi
    >>> print round(array(map(dih, xm)) * 180. / pi, 2)
    [-70.53 -35.26   0.    35.26  70.53]

"""

__all__ = ["center", "axes", "axis", \
           "volume", "distance", "angle", "dihedral", \
           "Volume", "Distance", "Angle", "Dihedral", \
           "Center"]

from func import Func
from numpy import zeros, eye, shape, cross, dot
from numpy import sqrt, sin, arccos
from numpy import hstack, vstack
from numpy import array, asarray
from numpy.linalg import svd
from numpy import outer


def center (x):
    x = asarray (x)
    return sum (x) / len (x)


def axes (x):
    """
    Returns local axes v[i], 0 <= i < 3 as row vectors.

        >>> x = [[ 1.,  1.,  1.],
        ...      [ 1., -1., -1.],
        ...      [-1., -1.,  1.],
        ...      [-1.,  1., -1.],
        ...      [ 2.,  2.,  2.]]

        >>> v = axes (x)
        >>> v[0]
        array([-0.57735027, -0.57735027, -0.57735027])

    The other two axes are degenerate, but mutually orthogonal:

        >>> abs (dot (v[0], v[1])) < 1e-16
        True
        >>> abs (dot (v[1], v[2])) < 1e-16
        True
        >>> abs (dot (v[2], v[0])) < 1e-16
        True
    """
    x = asarray (x)

    w = x - center (x)

    # A 3 x 3 "inertia" tensor, a positive definite matrix:
    a = zeros ((3, 3))
    for v in w:
        a += outer (v, v)
    a /= len (w)

    #
    # SVD solves for  M = U * S *  V^T with (m x m) matrix  U, (n x n)
    # matrix  V and  (m x  n) "diagonal  matrix" S.   Diagonal  "s" as
    # returned by an SVD is a 1D-array, though:
    #
    u, s, vt = svd (a)

    # The first eigenvalue is the largest, but maybe the convention is
    # going to change. This is assumed in axis() below though:
    assert s[0] >= s[1] >= s[2] >= 0.0

    # Row vectors vt[i], 0 <= i < 3, are the axes (not the columns!):
    return vt


def axis (x):
    """
    Returns "main" axis approximating x by a linear object:

        >>> x = [[0., 0., 0.],
        ...      [1., 1., 1.],
        ...      [2., 2., 2.]]
        >>> axis (x)
        array([-0.57735027, -0.57735027, -0.57735027])
    """
    v = axes (x)

    return v[0]


class Linear(Func):
    """
    Trivial case of a  "reaction coordinate" --- linear combinaiton of
    primary variables as such:

        >>> x = asarray([1., 2., 3.])
        >>> f = Linear([1.0, 10., 100.0])
        >>> f(x)
        321.0
        >>> f.fprime(x)
        array([   1.,   10.,  100.])
    """
    def __init__(self, m):
        """
        The matrix "m"  should have the same shape  as the argument of
        the Linear Func.
        """
        self.__m = array(m)

    def f(self, x):
        x = asarray(x)

        assert shape(x) == shape(self.__m)

        return sum(x * self.__m)

    def fprime(self, x):
        return self.__m.copy()


class Center (Func):
    """
    An n x 3 -> 3 Func() computing a center of a species.  So far each
    of the three x, y-, and  z resulting values from a Center() Func()
    are just linear combinations of respective inputs.

        >>> c = Center()
        >>> w = 0.39685026
        >>> x = array([[ w,  w,  w],
        ...            [-w, -w,  w],
        ...            [ w, -w, -w],
        ...            [-w,  w, -w]])

        >>> c (x)
        array([ 0.,  0.,  0.])

        >>> from func import NumDiff
        >>> from numpy import max, abs
        >>> c1 = NumDiff (c)
        >>> max (abs (c.fprime (x) - c1.fprime (x))) < 1.0e-12
        True
    """
    def __init__ (self):
        # FIXME: allow for weighted center
        pass

    def taylor (self, x):
        c = sum (x) / len (x)
        cx = zeros (shape (c) + shape (x))
        for i in range (len (x)):
            cx[:, i, :] = eye (3) / len (x)
        return c, cx


class Volume(Func):
    """For an array of 4 vectors x, return a measure of their
    (signed) volume:

       v(x) = [ ( x  -  x ) x ( x  - x ) ] * ( x  - x )
                   1     0       2    1         3    2

    Here "x" and "*" stay for cross- and dot-products respectively.

    An example:

        >>> from numpy import array, max, abs

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
        four = self.__four

        # final derivatives:
        fprime = zeros(shape(x))

        f, fprime[four] = _volume(x[four])

        return f, fprime

# one instance of Volume(Func):
volume = Volume()

def _volume(x):

    a = x[1] - x[0]
    b = x[2] - x[1]
    c = x[3] - x[2]

    # the value:
    f = dot(cross(a, b), c)

    # the derivatives wrt a, b, and c:
    fc = cross(a, b)
    fb = cross(c, a)
    fa = cross(b, c)

    # final derivatives:
    fprime = zeros(shape(x))
    fprime[0] =    - fa
    fprime[1] = fa - fb
    fprime[2] = fb - fc
    fprime[3] = fc

    return f, fprime


class Distance(Func):
    """Cartesian distance between two points

    An example:

        >>> from numpy import array, max, abs

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

        # indices of two points to use:
        two = self.__two

        # final derivatives:
        fprime = zeros(shape(x))

        f, fprime[two] = _distance(x[two])

        return f, fprime

# one instance of Distance(Func):
distance = Distance()

def _distance(x):

    d = x[1] - x[0]

    # the value:
    f = sqrt(dot(d, d))

    # the derivatives wrt d:
    fd = d / f

    # final derivatives:
    fprime = zeros(shape(x))
    fprime[0] = - fd
    fprime[1] = + fd

    return f, fprime

class Angle(Func):
    """Angle between three points

    An example:

        >>> from numpy import array, max, abs

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
        three = self.__three

        # final derivatives:
        fprime = zeros(shape(x))

        f, fprime[three] = _angle(x[three])

        return f, fprime

# one instance of Angle(Func):
angle = Angle()

def _angle(x):

    a = x[0] - x[1]
    b = x[2] - x[1]

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
    fprime[0] = + fa
    fprime[1] = - fa - fb
    fprime[2] = + fb

    return f, fprime

class Dihedral(Func):
    """Dihedral angle formed by four points

    An example:

        >>> from numpy import array, max, abs

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
        ...         print "h(x) =", h(x) / pi * 180.
        ...         print "numerical:"
        ...         print h1.fprime(x)
        ...         print "analytical:"
        ...         print h.fprime(x)

#       ...     else:
#       ...         print "h(x) =", h(x) / pi * 180., "ok"

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
        four = self.__four

        # final derivatives stored here:
        fprime = zeros(shape(x))

        f, fprime[four] = _dihedral(x[four])

        return f, fprime

# one instance of Dihedral(Func):
dihedral = Dihedral()

def _dihedral(x):
    # code uses the stable recipie for derivatives
    # see, e.g. http://bcr.musc.edu/manuals/MODELLER6v0/manual/node180.html

    # indices of four points in 3D to use:
    i, j, k, l = (0, 1, 2, 3)

    a = x[j] - x[i]
    b = x[j] - x[k] # intended
    c = x[l] - x[k]

    # one plane normal:
    M = cross(a, b)
    LM = sqrt(dot(M, M))
    M /= LM

    # another plane normal:
    N = cross(b, c)
    LN = sqrt(dot(N, N))
    N /= LN

    # cosine:
    cs = dot(M, N)

    # it happens:
    if cs > 1.:
        assert cs - 1. < 1.e-10
        cs = 1.

    if cs < -1.:
        assert -1. - cs < 1.e-10
        cs = -1.

    # angle between two planes:
    f = arccos(cs)

    # numerically stable code for derivatives:
    if True:
        # base length:
        lb = sqrt(dot(b, b))

        # weights:
        wa = dot(a, b) / lb**2
        wc = dot(c, b) / lb**2

        fprime = zeros(shape(x))
        fprime[i] = + M * lb / LM
        fprime[l] = - N * lb / LN
        fprime[j] = (wa - 1.) * fprime[i] - wc * fprime[l]
        fprime[k] = (wc - 1.) * fprime[l] - wa * fprime[i]

    # see if 0-1-2-3-skew is (anti)parallel to the base:
    if dot(a, cross(b, c)) > 0:
        f = -f
        # NO!: fprime = -fprime

    return f, fprime

class Difference(Func):
    """Difference of two Funcs (say distances or other internal coordinates)

    An example:

        >>> from numpy import array, max, abs

        >>> x = array([(3., 0., 0.), (0., 4., 0.), (0., 4., 2.)])

        >>> d1 = Distance([0, 1])
        >>> d2 = Distance([1, 2])
        >>> dd = Difference(d1, d2)

        >>> d1(x), d2(x)
        (5.0, 2.0)

        >>> dd(x)
        3.0
    """
    def __init__(self, F1, F2):
        # save two funcs:
        self.__F12 = F1, F2

    def taylor(self, x):

        # aliases:
        F1, F2 = self.__F12

        f1, f1prime = F1.taylor(x)
        f2, f2prime = F2.taylor(x)

        return f1 - f2, f1prime - f2prime

class Array(Func):
    """
    """
    def __init__(self, *fs):
        # save two funcs:
        self.__fs = fs

    def taylor(self, x):

        # aliases:
        fs = self.__fs

        vals = [ f.taylor(x) for f in fs ]

        vs, gs = zip(*vals)

        return hstack(vs), vstack(gs)

# python rc.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
