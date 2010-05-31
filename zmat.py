"""
This module provides conversion from z-matrix coordiantes
to cartesian and back.

        Construct ZMat from a tuple representaiton of atomic

    >>> from func import NumDiff

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

The derivative matrix is given by fprime:

    >>> print zm.fprime(h2o)
    [[[-0.25038    -0.92942173  0.        ]
      [ 0.          0.          0.        ]
      [-0.96814764  0.2403648   0.        ]]
    <BLANKLINE>
     [[ 0.          0.          1.        ]
      [ 0.          0.          0.        ]
      [ 0.          0.          0.        ]]
    <BLANKLINE>
     [[ 0.          0.          0.        ]
      [ 0.          0.          0.        ]
      [ 0.          0.          0.        ]]]

    >>> zm1 = NumDiff(zm)
    >>> max(abs(zm.fprime(h2o) - zm1.fprime(h2o))) < 1.e-10
    True

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

The matrix containing the derivatives of the Cartesian coordinates
with regard to the internal ones:
    >>> z41 = NumDiff(z4)
    >>> max(abs(z4.fprime(ch4) - z41.fprime(ch4))) < 1.e-10
    True

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
            [ 1. ,  0. , -0. ]],
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

Looking at the derivative matrix for a lager system,
thus there will be more than one atom of the others
be connected:
The zmatrix correspond to one of a C2H3Pt6 system, like in the hydrogen
shift reaction CH2CH -> CH3C on a Pt(1 1 1) surface, but the values set
are not reasonable.

    >>> bigone = ZMat([(), (0,), (0, 1,), (1, 0, 2), (3, 1, 0),
    ...           (4, 1, 0), (4, 3, 1), (6, 4, 3), (6, 4, 3),
    ...           (8, 6, 4), (8, 6, 4)])

For testing purpose set all xvalues to 1:
    >>> xval = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ...    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ...            1.0, 1.0, 1.0, 1.0, 1.0)

    >>> bigone1 = NumDiff(bigone)
    >>> max(abs(bigone.fprime(xval) - bigone1.fprime(xval))) < 1.e-10
    True

Now set dihedrals to pi or 0, to observe system at this limits:
    >>> xval = (1.0, 1.0, 1.0, 1.0, 1.0, pi, 1.0, 1.0, 0.0, 1.0, 1.0,
    ...    1.0, 1.0, 1.0, 0.0, 1.0, 1.0, pi, 1.0, 1.0, pi, 1.0,
    ...            1.0, 0.0, 1.0, 1.0, 1.0)
    >>> max(abs(bigone.fprime(xval) - bigone1.fprime(xval))) < 1.e-10
    True

Now take also the rotation and translation of the whole system into account:
For any given set of internal variables, together with a rotation and a
translation vector, the Cartesian coordinates and their derivatives are given

Set up a starting system (CH4)
    >>> Zraw = [(), (0,), (0, 1), (0, 1, 2), (0, 1, 2)]
    >>> ch, hch, hchh = 1.09, 109.5 / 180. * pi, 120. / 180. * pi
    >>> ch4 = (ch, ch, hch, ch, hch, hchh, ch, hch, -hchh)
    >>> ch4 = array(ch4)

Some arbitrary vectors for rotation and translation:
    >>> random = array((0.1, 0.8, 5.))
    >>> move = array((1.0, 2.0, 1.0))

Initialising the internal to cartesian converter
    >>> zmat3 = ZMatrix3(Zraw)

Results from it (with all derivatives)
    >>> derint, derrot, dertr = zmat3.fprime(ch4, random, move)

Initalise a function for NumDiff which only takes the internal variables as input
    >>> class only_first_der():
    ...   def __init__(self, Zr, v1, v2):
    ...     self.zmat = ZMatrix3(Zr)
    ...     self.v1 = v1
    ...     self.v2 = v2
    ...   def give(self, inter):
    ...     zm, __ = self.zmat.taylor(inter, self.v1, self.v2)
    ...     return zm


Compare the results for the derivative after the internal coordinates
    >>> d1 = only_first_der(Zraw, random, move)
    >>> derv = NumDiff(d1.give)

    >>> print (derv.fprime(ch4) - derint < 1e-10).all()
    True

The same for the derivatives after the rotation vector
    >>> class only_second_der():
    ...   def __init__(self, Zr, ints, v2):
    ...     self.zmat2 = ZMatrix3(Zr)
    ...     self.ints = ints
    ...     self.v2 = v2
    ...   def give(self, v):
    ...     zm = self.zmat2.f(self.ints, v, self.v2)
    ...     return zm

    >>> d2 = only_second_der(Zraw, ch4, move)
    >>> derv2 = NumDiff(d2.give)

    >>> print (derv2.fprime(random) - derrot < 1e-10).all()
    True

And for the derivatives of the translation vector
    >>> class only_third_der():
    ...   def __init__(self, Zr, ints, v1):
    ...     self.zmat2 = ZMatrix3(Zr)
    ...     self.ints = ints
    ...     self.v1 = v1
    ...   def give(self, v):
    ...     zm = self.zmat2.f(self.ints, self.v1, v)
    ...     return zm

    >>> d3 = only_third_der(Zraw, ch4, random)
    >>> derv3 = NumDiff(d3.give)

    >>> print (derv3.fprime(move) - dertr < 1e-10).all()
    True
"""

from numpy import pi, sin, cos, cross, dot, sqrt, arccos
from numpy import array, asarray, empty, max, abs
from numpy import eye, zeros, outer, hstack, vstack
from numpy import any
# from vector import Vector as V, dot, cross
# from bmath import sin, cos, sqrt
from func import Func
from rc import distance, angle, dihedral
from quat import _rotmat

class ZMError(Exception):
    pass

class ZMat(Func):
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


    def taylor(self, vars):
        """Use the input array |vars| as values for internal coordinates
        and return cartesians. Based on code in OpenBabel.
        """

        # use ZM representation and values for internal coords
        # to compute cartesians:

        # number of atoms in z-part
        na = len(self.__zm)

        # number of atoms in fixed environemnt:
        ne = len(self.__fixed)

        # number of internal coordinates:
        nvar = len(vars)

        # flags to indicate valid atomic positions, keys are the indices:
        cached = [0] * na + [1] * ne
        # 0: undefined, 1: defined, -1: computation in progress

        # (N x 3)-array with junk values, output for coordinates:
        Z = empty((na + ne, 3))

        # (N x 3 x N_int)-array initial values for the derivatives:
        ZPRIME = zeros((na + ne, 3, nvar))

        # fill the preset positions of the "environment" atoms:
        Z[na:, :] = self.__fixed

        # undefined values set to NaNs (not used anywhere):
        Z[:na, :] = None

        def pos(x):
            """Return atomic position and its derivatives as a tuple,
            return cached results or compute, if necessary, and memoize
            """

            if cached[x] == -1:
                # catch infinite recursion:
                raise ZMError("cycle")

            if cached[x]:
                # return cached value:
                return Z[x], ZPRIME[x, :, :]
            else:
                # prevent infinite recursion, indicate computation in progress:
                cached[x] = -1

                # for actual computation see "pos1" below:
                try:
                    p, pprime  = pos1(x)
                except Exception, e:
                    raise ZMError("pos1 of", x, e.args)

                # save position of atom x and its derivative into cache array:
                Z[x] = p
                ZPRIME[x, :, :] = pprime

                # set the flag for valid positions of atom x:
                cached[x] = 1

                return p, pprime

        def pos1(x):
            """Compute atomic position and its derivative,
             using memoized funciton pos()
             """

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

            # the wanted derivatives
            Xprime = zeros((3, nvar))
            Aprime = zeros((3, nvar))
            Bprime = zeros((3, nvar))
            Cprime = zeros((3, nvar))

            if a is not None:
                # sanity:
                if a == x: raise ZMError("same x&a")

                # position of a, and x-a distance:
                A, Aprime = pos(a)
                dst = vars[idst]

            if b is not None:
                # sanity:
                if b == a: raise ZMError("same x&b")
                if b == x: raise ZMError("same x&b")

                # position of b, and x-a-b angle:
                B, Bprime = pos(b)
                ang = vars[iang]

            if c is not None:
                # sanity:
                if c == b: raise ZMError("same b&c")
                if c == a: raise ZMError("same a&c")
                if c == x: raise ZMError("same x&c")

                C, Cprime = pos(c)
                dih = vars[idih]

            # spherical to cartesian transformation here:
            r, rprime  = r3.taylor((dst, ang, dih)) # = r3(r, theta, phi)

            #
            # Orthogonal basis using the three anchor points:
            #
            U = B - A
            Uprime = Bprime - Aprime

            V = C - A
            Vprime = Cprime - Aprime

#           # FIXME: U and V are not normalized, so this check doesnt catch
#           #        the case when they are collinear, the case when B == C should
#           #        be catched earlier:
#           if not abs(U - V).any() > 10e-10:
#               raise ZMError("bad choice for a= %i b= %i and c= %i" % (a, b, c))

            ijk, dijk = reper.taylor([U, V])

            # The default settings for the anchor points (see above)
            # together with the (custom) implementation of the reper()
            # function will lead to:
            #
            #    i, j, k = reper([B - A, C - A])
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
            # ATTENTION: the structure of the anchor points is used below
            # when calculating the derivatives, which will be inherited
            # from the points defining the anchor. For any change it has
            # to be checked, if there also has to be adapted something

            # result for the positions
            X = A + dot(r, ijk) # r[0] * i + r[1] * j + r[2] * k

            #
            # For the derivatives there has something more to be done:
            # for the derivatives with the new internal coordinates
            # consider, that the first three atoms don't have the full set of them
            #
            #    dX / dY = dot( dr / dY, IJK) + ...
            #

            if idst is not None:
                Xprime[:, idst] = dot(rprime[:, 0], ijk)

            if iang is not None:
                Xprime[:, iang] = dot(rprime[:, 1], ijk)

            if idih is not None:
                Xprime[:, idih] = dot(rprime[:, 2], ijk)

            #
            # For all the other internal coordinates Y it will be an indirect dependence of X
            # via the anchor points A, B, C or rather via the coordinate system IJK built of
            # them:
            #
            #   dX / dY = ... + dA / dY
            #                 + dot(r, dIJK / dU * dU / dY)
            #                 + dot(r, dIJK / dV * dV / dY)
            #

            # since the derivatives of IJK also contribute continue with dX / dU and dX / dV:
            Xuv = r[0] * dijk[0, ...] + r[1] * dijk[1, ...] + r[2] * dijk[2, ...]
            # dot() would not work because of additional array axes.

            # complete the chain rule:
            Xprime += Aprime + dot(Xuv[:, 0, :], Uprime) + dot(Xuv[:, 1, :], Vprime)

            return X, Xprime

        # force evaluation of all positions:
        for x in range(na + ne):
            # calling pos(x) will set Z[x] and Z[y] for all y's
            # that are required to compute pos(x).
            # The same holds for the derivatives in ZPRIME[...]
            r, rprime = pos(x)

            if any(r != Z[x]) or any(rprime != ZPRIME[x]):
                raise ZMError("computed and cached positions differ")

        return Z, ZPRIME

    def pinv(self, atoms):
        "Pseudoinverse of ZMat, returns internal coordinates"

        vars = empty(self.__dim) # array
        x = 0
        for a, b, c, idst, iang, idih in self.__zm:
            #
            # Note: distance/angle/dihedral from rc.py
            # expect the 3D coordiantes of involved atoms
            # in a single array. We provide them by list-indexing
            # into the array "atoms".
            #
            if a is not None:
                vars[idst] = distance(atoms[[x, a]])
            if b is not None:
                vars[iang] = angle(atoms[[x, a, b]])
            if c is not None:
                vars[idih] = dihedral(atoms[[x, a, b, c]])
            x += 1

        return vars


class _ZMatrix(Func):
    def __init__(self, zmat):
        self.zmat = zmat

    def taylor(self, vars, v_rot, v_trans):
        #Cartesian coordinates (and their derivatives) for the unrotated and
        # untranslated system
        Xna, Xprimena = self.zmat.taylor(vars)
        # transform rotation vector in a rotationmatrix (for a single atom)
        rot, rotprime = _rotmat(v_rot)

        X = Xna.copy()
        Xprime = Xprimena.copy()

        # the shape of the derivatives Xprime should tell anything about the
        # systemsize
        n1, n2, n3 = Xprime.shape
        # n1 = number atoms, n2 =3, n3 = number internal coordinates
        # There are each three variables for trans. and rot.
        # and every cartesian variable has an effect on them
        allrotpr = zeros((n1,n2,3))
        alltranspr = zeros((n1,n2,3))
        # cycle over all Cartesian (three) atom positions
        for i, xna in enumerate(Xna):
            # change the positions according to rotation and translation
            X[i] = dot(rot, xna) + v_trans
            # update the derivatives
            # trans: dc_ij/dt_k = 1 if j==t, else 0
            alltranspr[i] = eye(3)
            for j in range(n3):
                # dX/di = rot * dX_unrotated/di
                Xprime[i,:,j] = dot(rot, Xprime[i,:,j])
                # dX/dv_r = X * dr/dv
                allrotpr[i] = dot(xna, rotprime)

        return X, (Xprime, allrotpr, alltranspr)

    def f(self, vars, v_rot, v_trans):
         C, __ = self.taylor( vars, v_rot, v_trans)
         return C

    def fprime(self, vars, v_rot, v_trans):
         __, (Cprime, Crotprime, Ctransprime) = self.taylor( vars, v_rot, v_trans)
         return (Cprime, Crotprime, Ctransprime)


class ZMatrix3(_ZMatrix):
    def __init__(self, zm, fixed = None):
        self.zmat = ZMat(zm, fixed = fixed)

def unit(v):
    "Normalize a vector"
    n = sqrt(dot(v, v))
    # numpy will just return NaNs:
    if n == 0.0: raise ZMError("divide by zero")
    return v / n

def M(x):
    "M_ij = delta_ij - x_i * x_j / x**2"
    n = unit(x)
    return eye(3) - outer(n, n)

def E(x):
    """E_ij = epsilon_ijk * x_k (sum over k)

        >>> print E([1, 2, 3])
        [[ 0.  3. -2.]
         [-3.  0.  1.]
         [ 2. -1.  0.]]
    """
    e = zeros((3,3))

    e[0, 1] = x[2]
    e[1, 2] = x[0]
    e[2, 0] = x[1]

    e[1, 0] = - e[0, 1]
    e[2, 1] = - e[1, 2]
    e[0, 2] = - e[2, 0]
    return e

class Reper(Func):
    """Returns orthogonal basis [i, j, k] where
    "k" is parallel to U
    "i" is in UV plane and
    "j" is orthogonal to that plane

    FIXME: there must be a better way to do this ...

    Example:
        >>> from func import NumDiff

        >>> r = Reper()
        >>> u = array((1., 0., 0.))
        >>> v = array((0., 1., 0.))
        >>> print r([u, v])
        [[-0.  1.  0.]
         [ 0.  0. -1.]
         [ 1.  0.  0.]]

        >>> u = array((1.1, 0.3, 0.7))
        >>> v = array((0.5, 1.9, 0.8))
        >>> uv = [u, v]

        >>> r1 = NumDiff(r)
        >>> max(abs(r.fprime(uv) - r1.fprime(uv))) < 1e-10
        True
    """

    def taylor(self, args):

        u, v = args

        lu = sqrt(dot(u, u))
        lv = sqrt(dot(v, v))

        if lu == 0.0: raise ZMError("divide by zero")
        if lv == 0.0: raise ZMError("divide by zero")

        #
        # Unit vectors for local coordiante system:
        #

        # unit vector in U-direciton:
        k = u / lu

        # dk/du:
        ku = M(k) / lu

        # dk/dv:
        kv = zeros((3,3))

        # orthogonal to UV plane:
        w = cross(v, u)
        lw = sqrt(dot(w, w))

        j = w / lw

        # dj/dw
        jw = M(j) / lw

        # dj/du:
        jv = dot(jw, E(u))

        # dj/dv:
        ju = dot(jw, E(-v))

        # in UV-plane, orthogonal to U:
        i = cross(k, j) # FIXME: not cross(j, k)!

        # di/du = di/dk * dk/du + di/dj * dj/du:
        iu = dot(E(j), ku) + dot(E(-k), ju)

        # di/du =                 di/dj * dj/du:
        iv =                 dot(E(-k), jv)

        f = array([i, j, k])

        # convention: fprime[i, k] = df_i / dx_k
        iprime = hstack((iu, iv))
        jprime = hstack((ju, jv))
        kprime = hstack((ku, kv))

        fprime = vstack((iprime,\
                         jprime,\
                         kprime))

        return f, fprime.reshape((3, 3, 2, 3))

# instance of Reper(Func):
reper = Reper()

class R3(Func):
    """Spherical to cartesian transformation.

        >>> from func import NumDiff

        >>> r3 = R3()

        >>> vz = (8., 0., 0.)
        >>> vx = (8., pi/2., 0.)
        >>> vy = (8., pi/2., pi/2.)

        >>> print r3(vz)
        [ 0.  0.  8.]

        >>> from numpy import round

        >>> print round(r3(vx), 4)
        [ 8.  0.  0.]

        >>> print round(r3(vy), 4)
        [ 0.  8.  0.]

        >>> r4 = NumDiff(r3)
        >>> max(abs(r3.fprime(vz) - r4.fprime(vz))) < 1e-10
        True
        >>> max(abs(r3.fprime(vx) - r4.fprime(vx))) < 1e-10
        True
        >>> max(abs(r3.fprime(vy) - r4.fprime(vy))) < 1e-10
        True
    """

    def f(self, args):

        r, theta, phi = args

        z = r * cos(theta)
        x = r * sin(theta) * cos(phi)
        y = r * sin(theta) * sin(phi)

        return array([x, y, z])

    def fprime(self, args):

        r, theta, phi = args

        ct, st =  cos(theta), sin(theta)
        cp, sp =  cos(phi),   sin(phi)

        z = ct
        x = st * cp
        y = st * sp

        fr = array([x, y, z])

        z = - st
        x = + ct * cp
        y = + ct * sp

        ft = array([x, y, z]) * r

        z = 0.0
        x = - st * sp
        y = + st * cp

        fp = array([x, y, z]) * r

        # convention: fprime[i, k] = df_i / dx_k
        return array([fr, ft, fp]).transpose()

# one instance of R3(Func):
r3 = R3()


# "python zmap.py", eventualy with "-v" option appended:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
