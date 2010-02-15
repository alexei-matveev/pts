"""
This module provides conversion from z-matrix coordiantes
to cartesian and back.

Construct ZMat from a tuple representaiton of atomic
connectivities:

    >>> rep = [(None, None, None), (0, None, None), (0, 1, None)]
    >>> zm = ZMat(rep)

The above may be abbreviated using Python notations
for empty tuple, 1-tuple, and a 2-tuple as:

    >>> zm = ZMat([(), (0,), (0, 1)])

Values of internal coordinates have to be provided separately:

    >>> h2o = (0.96, 0.96, 104.5 * pi / 180.0)
    >>> from numpy import round
    >>> print round(zm(h2o), 7)
    [[ 0.         0.         0.       ]
     [ 0.96       0.         0.       ]
     [-0.2403648  0.        -0.9294217]]

The entries may come in any order, but cross-referencing should
be correct:

    >>> rep = [(2, 1, None), (2, None, None), (None, None, None)]
    >>> zm = ZMat(rep)

The order of internal variables is "left to right":

    >>> h2o = (0.96, 104.5 * pi / 180.0, 0.96)
    >>> print round(zm(h2o), 7)
    [[-0.2403648  0.        -0.9294217]
     [ 0.96       0.         0.       ]
     [ 0.         0.         0.       ]]

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

    >>> print round(z4(ch4), 7)
    [[ 0.         0.         0.       ]
     [ 1.09       0.         0.       ]
     [-0.3638495  0.        -1.0274792]
     [-0.3638495 -0.8898231  0.5137396]
     [-0.3638495  0.8898231  0.5137396]]

Test consistency with the inverse transformation:

    >>> z4.pinv(z4(ch4)) - ch4
    array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])

"Breathing" mode derivative (estimate by num. diff.):

    >>> d1 = (z4(ch4 * 1.001) - z4(ch4)) / 0.001
    >>> print round(d1, 7)
    [[ 0.         0.         0.       ]
     [ 1.09       0.         0.       ]
     [-2.3287989  0.        -0.3295403]
     [-2.3287989  0.79288    2.0278806]
     [-2.3287989 -0.79288    2.0278806]]

"Breathing" mode derivative (estimate using zm.fprime):

    >>> d2 = dot(z4.fprime(ch4), ch4)
    >>> print round(d2, 7)
    [[ 0.         0.         0.       ]
     [ 1.09       0.         0.       ]
     [-2.3275015  0.        -0.3321136]
     [-2.3275015  0.7883549  2.0296979]
     [-2.3275015 -0.7883549  2.0296979]]

    >>> from numpy import max, abs
    >>> print round(max(abs(d2-d1)), 4)
    0.0045

(these are not real breathing modes as we scale also angles).

A surface model with three atoms at these positions:

    >>> slab = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]

A ZMat() with one atom attached to the surface,
referencing the indices of atoms in the "environment":

    >>> zm = ZMat([(1, 2, 3)], fixed=slab)

"Spherical" coordinates for that atom:

    >>> r = (2.5, pi/2, -pi/2)

Evaluation of the zmatrix at |r| will return four
positions, including those of the "surface" atoms:

    >>> round(zm(r), 12)
    array([[ 0. ,  0. ,  2.5],
           [ 0. ,  0. ,  0. ],
           [ 1. ,  0. ,  0. ],
           [ 0. ,  1. ,  0. ]])

    >>> zm.pinv(zm(r))
    array([ 2.5       ,  1.57079633, -1.57079633])

    >>> round(zm.fprime(r), 12)
    array([[[ 0. , -2.5,  0. ],
            [ 0. ,  0. ,  2.5],
            [ 1. ,  0. ,  0. ]],
    <BLANKLINE>
           [[ 0. ,  0. ,  0. ],
            [ 0. ,  0. ,  0. ],
            [ 0. ,  0. ,  0. ]],
    <BLANKLINE>
           [[ 0. ,  0. ,  0. ],
            [ 0. ,  0. ,  0. ],
            [ 0. ,  0. ,  0. ]],
    <BLANKLINE>
           [[ 0. ,  0. ,  0. ],
            [ 0. ,  0. ,  0. ],
            [ 0. ,  0. ,  0. ]]])
"""

from numpy import pi, sin, cos, cross, dot, sqrt, arccos
from numpy import array, asarray, empty
# from vector import Vector as V, dot, cross
# from bmath import sin, cos, sqrt
from func import NumDiff
from rc import distance, angle, dihedral

class ZMError(Exception):
    pass

class ZMat(NumDiff):
    def __init__(self, zm, fixed=None):
        """The first argument |zm| is a representation of connectivities
        used to define internal coordinates.

        |fixed| is an (N x 3)-array (or nested lists) defining cartesian
        coordinates of the fixed "environment" for the part of the system defined
        by |zm|. It is appended to the output of |.f| method as-is and may be
        used to treat the fixed subsystem (e.g. surface model). It should
        be ok to reference the indices of atoms of "environment" in the z-matrix
        definition |zm|.
        """

        #
        # Each entry in ZM definition is a 3-tuple (a, b, c)
        # defining x-a-b-c chain of atoms.
        #
        # not required: NumDiff.__init__(self, self.f, h=0.001)

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
        # internal z-matrix representation:
        self.__zm = []
        # kind of internal cooridnate (dst, ang, dih):
        self.kinds = []
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
            if idst is not None: self.kinds.append("dst")
            if iang is not None: self.kinds.append("ang")
            if idih is not None: self.kinds.append("dih")

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

        #
        # Save the fixed "environment" to be appended to ZMat output:
        #
        if fixed is not None:
            self.__fixed = asarray(fixed)
        else:
            # 0x3 array:
            self.__fixed = array([]).reshape(0, 3)


    def f(self, v):
        "Use the input array |v| as values for internal coordinates and return cartesians"

        # use ZM representation and values for internal coords
        # to compute cartesians:
        return self.__z2c(v)

    # For the time without separate implementation
    # inherit from NumDiff:
#   def fprime(self, v):
#       # either num-diff or anything better goes here:
#       raise NotImplemented

    def __z2c(self, vars):
        """Generates cartesian coordinates from z-matrix and the current set of
        internal coordinates. Based on code in OpenBabel."""

        # number of atoms in z-part
        na = len(self.__zm)

        # number of atoms in fixed environemnt:
        ne = len(self.__fixed)

        # flags to indicate valid atomic positions, keys are the indices:
        cached = [0] * na + [1] * ne
        # 0: undefined, 1: defined, -1: computation in progress

        # (N x 3)-array with junk values:
        xyz = empty((na + ne, 3))

        # fill the preset positions of the "environment" atoms:
        xyz[na:,:] = self.__fixed

        # undefined values set to NaNs (not used anywhere):
        xyz[:na,:] = None

        def pos(x):
            "Return atomic position, compute if necessary and memoize"

            if cached[x] == -1:
                # catch infinite recursion:
                raise ZMError("cycle")

            if cached[x]:
                # return cached value:
                return xyz[x]
            else:
                # prevent infinite recursion, indicate computation in progress:
                cached[x] = -1

                # for actual computation see "pos1" below:
                try:
                    p = pos1(x)
                except Exception, e:
                    raise ZMError("pos1 of", x, e.args)

                # save position of atom x into cache array:
                xyz[x] = p

                # set the flag for valid positions of atom x:
                cached[x] = 1

                return p

        def pos1(x):
            "Compute atomic position, using memoized funciton pos()"

            # pick the ZM entry from array:
            a, b, c, idst, iang, idih = self.__zm[x]

            # print "z-entry =", a, b, c, idst, iang, idih

            # default values for internal coordinates:
            dst = 0.0
            ang = 0.0
            dih = 0.0

            #
            # Default values for anchor points (see also how reper()
            # is constructed from these). These settings ensure that
            # by default position start in x-direction, with bending
            # moving them down in z-direction as in legacy implemetations.
            #
            A = array((0.,  0.,  0.))
            B = array((1.,  0.,  0.))
            C = array((0.,  0., -1.))

            if a is not None:
                # sanity:
                if a == x: raise ZMError("same x&a")

                # position of a, and x-a distance:
                A = pos(a)
                dst = vars[idst]

            if b is not None:
                # sanity:
                if b == a: raise ZMError("same x&b")
                if b == x: raise ZMError("same x&b")

                # position of b, and x-a-b angle:
                B = pos(b)
                ang = vars[iang]

            if c is not None:
                # sanity:
                if c == b: raise ZMError("same b&c")
                if c == a: raise ZMError("same a&c")
                if c == x: raise ZMError("same x&c")

                C = pos(c)
                dih = vars[idih]

            # spherical to cartesian transformation here:
            v = r3(dst, ang, dih) # = r3(r, theta, phi)

            #
            # Orthogonal basis using the three anchor points:
            #
            i, j, k = reper(A, B, C)

            # The default settings for the anchor points (see above)
            # together with the (custom) implementation of the reper()
            # function will lead to:
            #
            #    i = [ 0.  1.  0.]
            #    j = [ 0.  0. -1.]
            #    k = [ 1.  0.  0.]
            #
            # For general values of (A, B, C) one has
            #
            #    k ~ AB, j ~ [AC x k], i = [k x j]
            #
            # FIXME: This appears to be a left-reper, e.g [i x j] = -k,
            #        is there a better way to achive compatibility
            #        with legacy code?
            #

            X = A + v[0] * i + v[1] * j + v[2] * k

            return X

        # force evaluation of all positions:
        for x in range(na + ne):
            # calling pos(x) will set xyz[x] and xyz[y] for all y's
            # that are required to compute pos(x).
            p = pos(x)
            q = xyz[x]
            if (p != q).any():
                raise ZMError("computed and cached positions differ")

        return xyz

    def pinv(self, atoms):
        "Pseudoinverse of ZMat, returns internal coordinates"

        vars = empty(self.__dim) # array
        x = 0
        for a, b, c, idst, iang, idih in self.__zm:
            #
            # Note: distance/angle/dihedral from rc.py return
            # a tuple of a value and derivative, so far
            # only the value is used. Also these funcitons
            # expect the 3D coordiantes of involved atoms
            # in a single array. We provide them by list-indexing
            # into the array "atoms".
            #
            if a is not None:
                vars[idst] = distance(atoms[[x, a]])[0]
            if b is not None:
                vars[iang] = angle(atoms[[x, a, b]])[0]
            if c is not None:
                vars[idih] = dihedral(atoms[[x, a, b, c]])[0]
            x += 1

        return vars

def unit(v):
    "Normalize a vector"
    n = sqrt(dot(v, v))
    # numpy will just return NaNs:
    if n == 0.0: raise ZMError("divide by zero")
    return v / n

def reper(A, B, C):
    """Returns orthogonal basis [i, j, k] where
    "k" is parallel to AB
    "i" is in ABC plane and
    "j" is orthogonal to that plane

    Example:

        >>> A = array((0., 0., 0.))
        >>> B = array((1., 0., 0.))
        >>> C = array((0., 1., 0.))
        >>> print reper(A, B, C)
        [[-0.  1.  0.]
         [ 0.  0. -1.]
         [ 1.  0.  0.]]
    """

    # two vectors in ABC-plane:
    ab = B - A
    ac = C - A

    #
    # Unit vectors for local coordiante system:
    #

    # unit vector in AB-direciton:
    k = unit(ab)

    # orthogonal to ABC plane:
    j = unit(cross(ac, k))

    # in ABC-plane, orthogonal to AB:
    # i = unit(cross(j, k))
    i = unit(cross(k, j))
    # This appears to be a left-reper compatible with legacy code!

    return array([i, j, k])

def r3(r, theta, phi):
    """Spherical to cartesian transformation.

        >>> from numpy import round

        >>> print r3(8., 0., 0.)
        [ 0.  0.  8.]

        >>> print round(r3(8., pi/2., 0.), 4)
        [ 8.  0.  0.]

        >>> print round(r3(8., pi/2., pi/2.), 4)
        [ 0.  8.  0.]
    """

    z = r * cos(theta)
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)

    return array([x, y, z])


# "python zmap.py", eventualy with "-v" option appended:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
