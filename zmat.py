"""
This module provides conversion from z-matrix coordiantes to cartesian
and back.

        Construct ZMat from a tuple representaiton of atomic

    >>> from func import NumDiff

connectivities:

    >>> rep = [(None, None, None), (0, None, None), (0, 1, None)]
    >>> zm = ZMat(rep)

The above may  be abbreviated using Python notations  for empty tuple,
1-tuple, and a 2-tuple as:

    >>> zm = ZMat([(), (0,), (0, 1)])

Values of internal coordinates have to be provided separately:

    >>> from numpy import pi

    >>> h2o = (0.96, 0.96, 104.5 * pi / 180.0)
    >>> from numpy import round
    >>> print round(zm(h2o), 7)
    [[ 0.         0.         0.       ]
     [ 0.96       0.         0.       ]
     [-0.2403648  0.        -0.9294217]]

The entries  may come  in any order,  but cross-referencing  should be
correct:

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

The |pinv|  (pseudo-inverse) method of the ZMat()  given the cartesian
coordinates  returns  the internals  according  to  the definition  of
connectivities encoded in ZMat().

Compare |internals| and zm^-1( zm(internals) ):

    >>> h2o = array(h2o)
    >>> zm.pinv(zm(h2o)) - h2o
    array([ 0.,  0.,  0.])

The "pseudo" in the pseudoinverse  is to remind you that the cartesian
to internal is not one-to-one:

    >>> xyz = zm(h2o)

Both |xyz| and translated |xyz| correspond to the same set of internal
coordinates:

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

The  matrix containing  the derivatives  of the  Cartesian coordinates
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

A  ZMat() with  one  atom  attached to  the  surface, referencing  the
indices of atoms in the "environment":

    >>> zm = ZMat([(1, 2, 3)], fixed=slab)

"Spherical" coordinates for that atom:

    >>> r = (2.5, pi/2, -pi/2)

Evaluation of the zmatrix at |r| will return four positions, including
those of the "surface" atoms:

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

Looking at the  derivative matrix for a lager  system, thus there will
be  more  than one  atom  of the  others  be  connected.  The  zmatrix
correspond  to one of  a C2H3Pt6  system, like  in the  hydrogen shift
reaction CH2CH -> CH3C on a Pt(1  1 1) surface, but the values set are
not reasonable.

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
"""

from numpy import dot
from numpy import array, asarray, empty, max, abs
from numpy import eye, zeros
from numpy import any, shape
from numpy import vstack, hstack
# from vector import Vector as V, dot, cross
# from bmath import sin, cos, sqrt
from func import Func
from rc import distance, angle, dihedral, center
from quat import _rotmat, cart2vec, cart2veclin
from quat import reper, r3

class ZMError(Exception):
    pass

class ZMat(Func):
    def __init__(self, zm, fixed=None, base=0):
        """
        The first argument |zm|  is a representation of connectivities
        used to define internal coordinates.

        |fixed|  is  an  (N  x  3)-array (or  nested  lists)  defining
        cartesian coordinates of the  fixed "environment" for the part
        of the system defined by |zm|. It is appended to the output of
        |.f| method as-is and may be used to treat the fixed subsystem
        (e.g. surface model). It should be ok to reference the indices
        of atoms of "environment" in the z-matrix definition |zm|.
        """

        #
        # Each entry in ZM definition  is a 3-tuple (a, b, c) defining
        # x-a-b-c chain of atoms.
        #

        def t3(t):
            "Returns a tuple of length at least 3, missing entries set to None"
            tup = tuple(t)
            # make it at least length 3, append enough None:
            tup += (None,) * (3 - len(tup))
            return tup

        # Convert to  tuples, append enough  |None|s in case  thay are
        # missing:
        zm = [t3(z) for z in zm]

        # Base-1  indices are  assumed  for ZMat  (..., base=1)  which
        # might be more intuitive for humans:
        def rebase (t):
            return tuple (abs (x) - base if x is not None else x for x in t)

        zm = [rebase (z) for z in zm]

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
        # with the  last three fields being  the left-to-right running
        # index of internal coordinate.
        #

        #
        # Save the fixed "environment" to be appended to ZMat output:
        #
        if fixed is not None:
            self.__fixed = asarray(fixed)
        else:
            # 0x3 array:
            self.__fixed = array([]).reshape(0, 3)


    def taylor (self, vars):
        """
        Use the input array  |vars| as values for internal coordinates
        and return cartesians. Based on code in OpenBabel.
        """

        #
        # To avoid cryptic errors when one provides cartesians instead
        # of internals here:
        #
        assert shape (vars) == (self.__dim, ), "Wrong shape of internal coordinates!"

        # Use  ZM representation  and  values for  internal coords  to
        # compute cartesians:

        # number of atoms in z-part
        na = len(self.__zm)

        # number of atoms in fixed environemnt:
        ne = len(self.__fixed)

        # number of internal coordinates:
        nvar = len(vars)

        # flags  to  indicate valid  atomic  positions,  keys are  the
        # indices:
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
            """
            Return  atomic position  and its  derivatives as  a tuple,
            return  cached  results  or  compute,  if  necessary,  and
            memoize
            """

            if cached[x] == -1:
                # catch infinite recursion:
                raise ZMError("cycle")

            if cached[x]:
                # return cached value:
                return Z[x], ZPRIME[x, :, :]
            else:
                # Prevent infinite  recursion, indicate computation in
                # progress:
                cached[x] = -1

                # for actual computation see "pos1" below:
                try:
                    p, pprime  = pos1(x)
                except Exception, e:
                    raise ZMError("pos1 of", x, e.args)

                # save  position of  atom  x and  its derivative  into
                # cache array:
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
            # Default values  for anchor points (see  also how reper()
            # is constructed  from these). These  settings ensure that
            # by default  position start in  x-direction, with bending
            # moving   them   down  in   z-direction   as  in   legacy
            # implemetations.
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
            uv = empty((2, 3))
            uv[0, :] = B - A
            Uprime = Bprime - Aprime

            uv[1, :] = C - A
            Vprime = Cprime - Aprime

          # # FIXME: U  and V are  not normalized, so this  check doesnt
          # #        catch the  case when  they are collinear,  the case
          # #        when B == C should be catched earlier:
          # if not abs(U - V).any() > 10e-10:
          #     raise ZMError("bad choice for a= %i b= %i and c= %i" % (a, b, c))

            ijk, dijk = reper.taylor(uv)

            # The default  settings for the anchor  points (see above)
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
            # FIXME: This  appears to be a  left-reper, e.g [i  x j] =
            #        -k, is there a better way to achive compatibility
            #        with legacy code?
            #
            # ATTENTION: the  structure of  the anchor points  is used
            # below  when calculating the  derivatives, which  will be
            # inherited from  the points defining the  anchor. For any
            # change it  has to  be checked, if  there also has  to be
            # adapted something

            # result for the positions
            X = A + dot(r, ijk) # r[0] * i + r[1] * j + r[2] * k

            #
            # For the derivatives there has something more to be done:
            # for  the derivatives with  the new  internal coordinates
            # consider, that the first three atoms don't have the full
            # set of them
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
            # For all the  other internal coordinates Y it  will be an
            # indirect dependence of  X via the anchor points  A, B, C
            # or rather via the coordinate system IJK built of them:
            #
            #   dX / dY = ... + dA / dY
            #                 + dot(r, dIJK / dU * dU / dY)
            #                 + dot(r, dIJK / dV * dV / dY)
            #

            # Since  the derivatives of  IJK also  contribute continue
            # with dX / dU and dX / dV:
            Xuv = r[0] * dijk[0, ...] + r[1] * dijk[1, ...] + r[2] * dijk[2, ...]
            # dot() would not work because of additional array axes.

            # complete the chain rule:
            Xprime += Aprime + dot(Xuv[:, 0, :], Uprime) + dot(Xuv[:, 1, :], Vprime)

            return X, Xprime

        # force evaluation of all positions:
        for x in range(na + ne):
            # calling pos(x) will  set Z[x] and Z[y] for  all y's that
            # are required to compute  pos(x).  The same holds for the
            # derivatives in ZPRIME[...]
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
            # Note: distance/angle/dihedral  from rc.py expect  the 3D
            # coordiantes  of involved  atoms  in a  single array.  We
            # provide them by list-indexing into the array "atoms".
            #
            if a is not None:
                vars[idst] = distance(atoms[[x, a]])
            if b is not None:
                vars[iang] = angle(atoms[[x, a, b]])
            if c is not None:
                vars[idih] = dihedral(atoms[[x, a, b, c]])
            x += 1

        return vars


class Fixed (Func):
    """
    A const func  of "zero arguments" --- rather  of an array argument
    of empty shape, empty (0):

    >>> x = [[0.,  0., 0.],
    ...      [1.,  0., 0.],
    ...      [0.,  1., 0.]]
    >>> x = array (x)

    >>> f = Fixed (x)
    >>> q = empty (0)
    >>> y = f (q)
    >>> max (abs (y - x)) == 0.0
    True
    >>> shape (f.fprime (q)) == shape (y) + shape (q)
    True
    """

    def __init__ (self, x):
        x = array (x)
        self.__x = x

    def taylor (self, q):
        y = array (self.__x)
        yq = zeros (shape (y) + shape (q))
        return y, yq

    def pinv (self, y):
        return empty (0)


class Rigid (Func):
    """
    A Func()  of a  1D array with  6 elements, that  returns cartesian
    coordiantes as a (n, 3)-shaped array translated and rotated around
    the center.

    >>> x = [[0.,  0., 0.],
    ...      [1.,  0., 0.],
    ...      [0.,  1., 0.]]
    >>> x = array (x)

    >>> f = Rigid (x)
    >>> max (abs (x - f (zeros (6)))) < 1.0e-16
    True

    >>> from numpy import pi
    >>> q = array ([0., 10., 0., pi / 2., 0., 0.])
    >>> f (q)
    array([[  0.        ,  10.33333333,  -0.33333333],
           [  1.        ,  10.33333333,  -0.33333333],
           [  0.        ,  10.33333333,   0.66666667]])

    >>> from pts.func import NumDiff
    >>> f1 = NumDiff (f)
    >>> max (abs (f.fprime (q) - f1.fprime (q))) < 1.0e-10
    True
    >>> max (abs (q - f.pinv (f (q)))) < 1.0e-16
    True
    """

    def __init__ (self, x):
        x = array (x)
        self.__x = x
        # Same procedure as in pinv() method:
        self.__c = center (x)

    def taylor (self, q):
        # aliases:
        x = self.__x
        c = self.__c

        q = asarray (q)
        T = q[0:3]
        W = q[3:6]

        # Transform rotation vector in a rotation matrix:
        R, RW = _rotmat (W)

        # Loop over all Cartesian positions
        y = empty (shape (x))
        yq = empty (shape (y) + shape (q))
        for i in range (len (x)): # == number of atoms
            y[i] = c + dot (R, x[i] - c) + T

            # Derivatives:
            yq[i, :, :3] = eye (3)
            for j in range (len (W)): # == 3
                yq[i, :, j + 3] = dot (RW[..., j], x[i] - c)

        return y, yq

    def pinv (self, y):
        # aliases:
        x = self.__x
        c = self.__c

        # Same procedure as for c:
        c1 = center (y)

        # Translation vector:
        T = c1 - c

        # FIXME: see the logic in RT.pinv():
        if len (x) < 3:
            raise ZMError ("need three centers")

        # Rotation vector:
        W = cart2vec (x, y)

        q = empty (6)
        q[:3] = T
        q[3:] = W

        return q


class ManyBody (Func):
    """
    >>> x = [[0.,  0., 0.],
    ...      [1.,  0., 0.],
    ...      [0.,  1., 0.]]
    >>> x = array (x)

    >>> g = Rigid (x)
    >>> f = ManyBody (g, g)
    >>> q = zeros (12)
    >>> max (abs (f (q) - vstack ([x, x]))) < 1.0e-10
    True

    >>> q = zeros (12) + 0.125
    >>> from pts.func import NumDiff
    >>> f1 = NumDiff (f)
    >>> max (abs (f.fprime (q) - f1.fprime (q))) < 1.0e-10
    True
    >>> max (abs (q - f.pinv (f (q)))) < 1.0e-10
    True

    >>> f = ManyBody (Fixed (x), Rigid (x))
    >>> f1 = NumDiff (f)
    >>> max (abs (f.fprime (q) - f1.fprime (q))) < 1.0e-10
    True
    """
    def __init__ (self, *fs):
        self.__fs = fs

    def taylor (self, x):
        fs = self.__fs

        dofs = [0 if isinstance (f, Fixed) else 6 for f in fs]

        qs = [None] * len (fs)
        k = 0
        for i, n in enumerate (dofs):
            qs[i] = x[k: k + n]
            k += n

        results = [f.taylor (q) for f, q in zip (fs, qs)]

        y = vstack (f for f, _ in results)
        yq = (fq for _, fq in results)

        yx = zeros (shape (y) + shape (x))
        k = 0
        j = 0
        for i, fq in enumerate (yq):
            n, _, dof = shape (fq)
            yx[k: k + n, :, j: j + dof] = fq
            k += n
            j += dof

        return y, yx

    def pinv (self, y):
        fs = self.__fs

        # FIXME:  hm, how  do  I get  number  of atoms  each  f in  fs
        # returns? At  the moment this evaluates  all of them  at 0 to
        # learn about the shape of the results:
        dofs = [0 if isinstance (f, Fixed) else 6 for f in fs]
        qs = [zeros (n) for n in dofs]
        shapes = [shape (f (q)) for f, q in zip (fs, qs)]

        ys = [None] * len (fs)
        k = 0
        for i, shp in enumerate (shapes):
            n, _, = shp
            ys[i] = y[k: k + n, :]
            k += n

        qs = [f.pinv (y) for f, y in zip (fs, ys)]

        return hstack (qs)


class RT(Func):
    """
    A  wrapper  for   functions  that  return  cartesian  coordinates,
    e.g. ZMat().  An RT() is a  Func() object that, in addition to the
    ZMat() coordinates V, takes (skew)vector parameters, W and T, of a
    rotation and  a translation and applies  the Rotate&Translate (RT)
    transformation.

    Set up a starting system (CH4)

        >>> from numpy import pi
        >>> z = [(), (0,), (0, 1), (0, 1, 2), (0, 1, 2)]
        >>> ch, hch, hchh = 1.09, 109.5 / 180. * pi, 120. / 180. * pi

    Internal Z-matrix variables:

        >>> V = (ch, ch, hch, ch, hch, hchh, ch, hch, -hchh)

    Some arbitrary vectors for rotation and translation:

        >>> W = (0.1, 0.8, 5.0)
        >>> T = (1.0, 2.0, 1.0)

        >>> U = (ch, ch, hch, ch, hch, hchh, ch, hch, -hchh, 0.1, 0.8, 5.0, 1.0, 2.0, 1.0)

    Initialising the internal to cartesian converter

        >>> Z = RT(ZMat(z))

        >>> ZM = ZMatrix3(z)

    Cartesian coordiantes and three types of derivatives:

        >>> X = Z(V, W, T)
        >>> XV, XW, XT = Z.fprime(V, W, T)

        >>> X2 = ZM(U)

        >>> X2.shape = (-1, 3)
        >>> max(abs(X - X2)) < 1.e-10
        True

        >>> V2, W2, T2 = Z.pinv(X)
        >>> X3 = Z(V2, W2, T2)

        # not verifiable V = V2, W = W2 because of periodicity
        >>> max(abs(X - X3)) < 1.e-10
        True


        >>> from func import NumDiff, Partial

    NumDiff operates  only with functions of a  single variable, these
    are three partial functions of V, W, and T:

        >>> ZV = Partial(Z, 0, W, T)
        >>> ZW = Partial(Z, 1, V, T)
        >>> ZT = Partial(Z, 2, V, W)

        >>> max(abs(XV - NumDiff(ZV).fprime(V))) < 1.e-10
        True
        >>> max(abs(XW - NumDiff(ZW).fprime(W))) < 1.e-10
        True
        >>> max(abs(XT - NumDiff(ZT).fprime(T))) < 1.e-10
        True

        >>> ZMD = ZM.fprime(U)
        >>> max(abs(ZMD - NumDiff(ZM).fprime(U))) < 1.e-10
        True
    """

    def __init__(self, f):
        # a Func()  of a 1D array, that  returns cartesian coordiantes
        # as a (n, 3)-shaped array:
        self.__f = f

    def taylor(self, V, W, T):
        # alias:
        f = self.__f

        # Cartesian  coordinates  (and   their  derivatives)  for  the
        # unrotated and untranslated system
        X, XV = f.taylor(V)

        # transform rotation vector in a rotation matrix (for a single atom)
        R, RW = _rotmat(W)

        # There are each three variables for translation and rotation.
        # This will be the derivatives of cartesians wrt W and T:
        XW = empty(shape(X) + shape(W))
        XT = zeros(shape(X) + shape(T))

        # cycle over all Cartesian (three) atom positions
        for i in range(len(X)): # == number of atoms
            #
            # First,  update the  derivatives, we  use  the unmodified
            # positions   X  here,   so   it  must   be  done   before
            # rotating/translating X.
            #

            #
            # dX / dT  = 1
            #   i    j    ij
            #
            XT[i] = eye(3)

            #
            # dX / dV  := R   * dX / dV
            #   i    j     ik     k    j
            #
            for j in range(len(V)): # == number of internal coordiantes

                XV[i, :, j] = dot(R, XV[i, :, j])

            #
            # dX / dW  = dR  / dW * X
            #   i    j     ik    j   k
            #
            for j in range(len(W)): # == 3

                XW[i, :, j] = dot(RW[..., j], X[i])

            #
            # Second, change  the positions according  to rotation and
            # translation
            #
            X[i] = dot(R, X[i]) + T

        return X, (XV, XW, XT)

    def pinv(self, X):
        # alias:
        f = self.__f
        #FIXME: works only if first coordinate of f(v) is at origin
        V = f.pinv(X)
        X2 = f(V)
        a, b = X2.shape
        assert(b == 3)
        assert max(abs(X2[0,:])) < 1e-12

        T = X[0,:] - X2[0,:]

        X1 = zeros((a,b))
        for i in range(a):
           X1[i,:] = X[i,:] - T

        if a == 1:
            W = zeros(3)
        elif a == 2:
            W = cart2veclin(X2, X1)
        else:
            W = cart2vec(X2, X1)

        return V, W, T

class ZMatrix3(Func):
    """
    Creates an object for  calculating cartesian coordinates and their
    derivatives to given internal coordinates.  Needs a zmatrix object
    by  initalising to  know how  the internal  coordinates are  to be
    used.

    The last six coordinates are suppposed to be related to the global
    orientation  of the  objects, thus  the 6th  to 3th  last  for the
    rotation,  the last  three  for the  translation  of the  complete
    connnected cartesian objects.
    """
    def __init__(self, zm, fixed = None):
        self.rt = RT(ZMat(zm, fixed = fixed))

    def taylor(self, y):
        # y  = [v,  w, t],  here w  is global  orientation (quaternion
        # expressed as vector), t is  global translation. Both w and t
        # have three elements
        v = y[:-6]
        w = y[-6:-3]
        t = y[-3:]

        # rt  needs  zmatrix  coordinates,  rotation  and  translation
        # separate
        x, (xv, xw, xt) = self.rt.taylor(v, w, t)

        dx = zeros((len(x[:,0]), 3, len(y)))

        # got the  derivatives separate  for v, w,  t.  Here  put them
        # into one object
        dx[:,:,:-6] = xv
        dx[:,:,-6:-3] = xw
        dx[:,:,-3:] = xt

        # Give back x as a 1-dimensional array (rt gave back as x, y,z
        # compontent   for   each   "atom"  change   the   derivatives
        # accordingly
        x.shape = (-1)
        dx.shape = (-1, len(y))
        # (-1 means  that in  this direction the  size should  be made
        # fitting to  the other  specifications (so that  the complete
        # array could be transformed))

        return x, dx

# "python zmap.py", eventualy with "-v" option appended:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
