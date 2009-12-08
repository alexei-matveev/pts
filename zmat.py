"""
This module provides conversion from z-matrix coordiantes
to cartesian and back.

Construct ZMat from a tuple representaiton of atomic
connectivities:

    >>> rep = [(None, None, None), (0, None, None), (0, 1, None)]
    >>> zm = ZMat(rep)

The above may be abbreviated as using Python notations
for empty tuple, 1-tuple, and a 2-tuple:

    >>> zm = ZMat([(), (0,), (0, 1)])

Values of internal coordinates have to be provided separately:

    >>> h2o = (0.96, 0.96, 104.5 * pi / 180.0)
    >>> zm(h2o)
    array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
           [  9.60000000e-01,   0.00000000e+00,   0.00000000e+00],
           [ -2.40364804e-01,   5.69106676e-17,  -9.29421735e-01]])

The entries may come in any order, but cross-referencing should
be correct:

    >>> rep = [(2, 1, None), (2, None, None), (None, None, None)]
    >>> zm = ZMat(rep)

The order of internal variables is "left to right":

    >>> h2o = (0.96, 104.5 * pi / 180.0, 0.96)
    >>> zm(h2o)
    array([[ -2.40364804e-01,   5.69106676e-17,  -9.29421735e-01],
           [  9.60000000e-01,   0.00000000e+00,   0.00000000e+00],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00]])


The |pinv| (pseudo-inverse) method of the ZMat() given the cartesian coordinates
returns the internals according to the definition of connectivities
encoded in ZMat().

Compare |internals| and zm^-1( zm(internals) ):

    >>> h2o = array(h2o)
    >>> zm.pinv(zm(h2o)) - h2o
    array([ 0.,  0.,  0.])

The "pseudo" in the pseudoinverse is to remind you that
the cartesian to internal is not one-to-one:

    >>> xyz = zm(h2o)

Both |xyz| and translated |xyz| correspond to the same set
of internal coordinates:

    >>> zm.pinv(xyz) - zm.pinv(xyz + array((1., 2., 3.)))
    array([ 0.,  0.,  0.])

The same holds for overall rotations.

This CH4 example uses dihedral angles:

    C
    H 1 ch
    H 1 ch 2 hch
    H 1 ch 2 hch 3 hchh
    H 1 ch 2 hch 3 -hchh

    ch     1.09
    hch  109.5
    hchh 120.

Connectivities:

    >>> z4 = ZMat([(), (0,), (0, 1), (0, 1, 2), (0, 1, 2)])

Parameters:

    >>> ch, hch, hchh = 1.09, 109.5 / 180. * pi, 120. / 180. * pi

Internal coordinates:

    >>> ch4 = (ch, ch, hch, ch, hch, hchh, ch, hch, -hchh)
    >>> ch4 = array(ch4)

Cartesian geometry:

    >>> z4(ch4)
    array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
           [  1.09000000e+00,   0.00000000e+00,   0.00000000e+00],
           [ -3.63849477e-01,   6.29149572e-17,  -1.02747923e+00],
           [ -3.63849477e-01,  -8.89823111e-01,   5.13739613e-01],
           [ -3.63849477e-01,   8.89823111e-01,   5.13739613e-01]])

Test consistency with the inverse transformation:

    >>> z4.pinv(z4(ch4)) - ch4
    array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])

"Breathing" mode derivative (estimate by num. diff.):

    >>> d1 = (z4(ch4 * 1.001) - z4(ch4)) / 0.001
    >>> d1
    array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
           [  1.09000000e+00,   0.00000000e+00,   0.00000000e+00],
           [ -2.32879885e+00,   2.01785263e-17,  -3.29540342e-01],
           [ -2.32879885e+00,   7.92879953e-01,   2.02788058e+00],
           [ -2.32879885e+00,  -7.92879953e-01,   2.02788058e+00]])

"Breathing" mode derivative (estimate using zm.fprime):

    >>> d2 = dot(z4.fprime(ch4), ch4)
    >>> d2
    array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
           [  1.09000000e+00,   0.00000000e+00,   0.00000000e+00],
           [ -2.32750153e+00,   2.03360906e-17,  -3.32113563e-01],
           [ -2.32750153e+00,   7.88354946e-01,   2.02969795e+00],
           [ -2.32750153e+00,  -7.88354946e-01,   2.02969795e+00]])

    >>> from numpy import max, abs, round
    >>> max(abs(d2-d1))
    0.004525006862505121

(these are not real breathing modes as we scale also angles).
"""

from math import pi
from numpy import sin, cos, cross, dot, sqrt, arccos
from numpy import array, empty
# from vector import Vector as V, dot, cross
# from bmath import sin, cos, sqrt
from func import NumDiff

class ZMError(Exception):
    pass

class ZMat(NumDiff):
    def __init__(self, zm):

        #
        # Each entry in ZM definition is a 3-tuple (a, b, c)
        # defining x-a-b-c chain of atoms.
        #
        NumDiff.__init__(self, self.f, h=0.001)

        def t3(t):
            "Returns a tuple of length at least 3, missing entries set to None"
            tup = tuple(t)
            # make it at least length 3, append enough None:
            tup += (None,) * (3 - len(tup))
            return tup

        # convert to tuples, append enough |None|s in case thay are missing:
        zm = [ t3(z) for z in zm ]

        # save ZM representation with indices of internal coordinates:
        i = 0
        self.__zm = []
        for a, b, c in zm:
            if c is not None:
                # regular ("4th and beyond") entry x-a-b-c:
                idst, iang, idih = i+0, i+1, i+2
                i += 3
            elif b is not None:
                # special ("third") entry x-a-b:
                idst, iang, idih = i+0, i+1, None
                i += 2
            elif a is not None:
                # special ("second") entry x-a:
                idst, iang, idih = i+0, None, None
                i += 1
            else:
                # special ("first") entry x:
                idst, iang, idih = None, None, None

            self.__zm.append((a, b, c, idst, iang, idih))

        # number of internal variables:
        self.__dim = i

        #
        # Each entry in the ZM is a 6-tuple:
        #
        # (a, b, c, idst, iang, idih)
        #
        # with the last three fields being the left-to-right
        # running index of internal coordinate.
        #

    def f(self, v):
        "Use the input array |v| as values for internal coordinates and return cartesians"

        # use ZM representation and values for internal coords
        # to compute cartesians:
        return self.__z2c(self.__zm, v)

    # For the time without separate implementation
    # inherit from NumDiff:
#   def fprime(self, v):
#       # either num-diff or anything better goes here:
#       raise NotImplemented

    def __z2c(self, atoms, vars):
        """Generates cartesian coordinates from z-matrix and the current set of
        internal coordinates. Based on code in OpenBabel."""

        # cache atomic positions, keys are the indices:
        cache = dict()

        def pos(x):
            "Return atomic position, compute if necessary and memoize"

            if x in cache:
                # catch infinite recursion:
                if cache[x] is None:
                    raise ZMError("cycle")
                else:
                    return cache[x]
            else:
                # prevent infinite recursion, put some nonsense:
                cache[x] = None
                # for actual computation see "pos1" below:
                try:
                    p = pos1(x)
                except ZMError, e:
                    raise ZMError("pos1 of", x, e.args)
                cache[x] = p
                return p

        def pos1(x):
            "Compute atomic position, using memoized funciton pos()"

            # pick the ZM entry from array:
            a, b, c, idst, iang, idih = atoms[x]

            # default origin:
            if a is None: return array((0.0, 0.0, 0.0))

            # sanity:
            if a == x: raise ZMError("same x&a")

            # position of a, and x-a distance:
            avec = pos(a)
            distance = vars[idst]

            # default X-axis:
            if b is None: return array((distance, 0.0, 0.0)) # FXIME: X-axis

            # sanity:
            if b == a: raise ZMError("same x&b")
            if b == x: raise ZMError("same x&b")

            # position of b, and x-a-b angle:
            bvec = pos(b)
            angle = vars[iang]

            # position of c, and x-a-b-c dihedral angle:
            if c is None:
                # default plane here:
                cvec = array((0.0, 1.0, 0.0))
                dihedral = pi / 2.0
            else:
                cvec = pos(c)
                dihedral = vars[idih]

            # sanity:
            if c == b: raise ZMError("same b&c")
            if c == a: raise ZMError("same a&c")
            if c == x: raise ZMError("same x&c")

            # normalize vector:
            def normalise(v):
                n = sqrt(dot(v, v))
                # numpy will just return NaNs:
                if n == 0.0: raise ZMError("divide by zero")
                return v / n

            v1 = avec - bvec
            v2 = avec - cvec

            n = cross(v1,v2)
            nn = cross(v1,n)

            n = normalise(n)
            nn = normalise(nn)

            n *= -sin(dihedral)
            nn *= cos(dihedral)
            v3 = n + nn
            v3 = normalise(v3)
            v3 *= distance * sin(angle)
            v1 = normalise(v1)
            v1 *= distance * cos(angle)
            p = avec + v3 - v1

            return p

        # compute all positions:
        xyz = []
        for atom in range(len(atoms)):
            xyz.append(pos(atom))

        # convert to vector/numpy:
        xyz = array(xyz)
        return xyz

    def pinv(self, atoms):
        "Pseudoinverse of ZMat, returns internal coordinates"

        def dst(x, a):
            "Distance x-a"
            ax = atoms[x] - atoms[a]
            return sqrt(dot(ax, ax))

        def ang(x, a, b):
            "Angle x-a-b"
            ax = atoms[x] - atoms[a]
            ab = atoms[b] - atoms[a]

            ax /= sqrt(dot(ax, ax))
            ab /= sqrt(dot(ab, ab))

            return arccos(dot(ax, ab))

        def dih(x, a, b):
            "Dihedral angle x-a-b-c"
            xa = atoms[a] - atoms[x]
            ab = atoms[b] - atoms[a]
            bc = atoms[c] - atoms[b]

            xab = cross(xa, ab)
            xab /= sqrt(dot(xab, xab))

            abc = cross(ab, bc)
            abc /= sqrt(dot(abc, abc))

            return arccos(dot(xab, abc))

        vars = empty(self.__dim) # array
        x = 0
        for a, b, c, idst, iang, idih in self.__zm:
            if a is not None:
                vars[idst] = dst(x, a)
            if b is not None:
                vars[iang] = ang(x, a, b)
            if c is not None:
                vars[idih] = dih(x, a, b, c)
            x += 1

        return vars


# Testing the examples in __doc__strings, execute
# "python gxmatrix.py", eventualy with "-v" option appended:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# You need to add "set modeline" and eventually "set modelines=5"
# to your ~/.vimrc for this to take effect.
# Dont (accidentally) delete these lines! Unless you do it intentionally ...
# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
