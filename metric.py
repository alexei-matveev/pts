from func import NumDiff
from numpy import dot, array, asarray, matrix, size, shape
from numpy import sqrt
from numpy import zeros, empty, eye
from numpy import max, abs
from numpy.linalg import solve
from copy import deepcopy


__all__ = ["setup_metric", "metric"]


class Default:
    """
    Implements metric relevant functions, like for
    transforming contra- and covariant vectors into
    each other

    This variant supposes that contra- and covariant vectors
    are the same, coordinate transformation if provided
    is ignored:

        >>> met = Default(None)

    Contravariant coordinates of a vector:

        >>> dX = array([0.0002, 0.001, 0.001])

    Get covariant coordinates, position of vectors is ignored:

        >>> dx = met.lower(dX, None)

    Verify that self.lower changes nothing:

        >>> all(dX == dx)
        True

    Verify self.raises for consistency:

        >>> all(dX == met.raises(dx, None))
        True
    """
    def __init__(self, fun = None):
        """
        normally needs a function with a fprime function
        in this special case it is only asked for it because
        of consistency
        """
        pass

    def lower(self, vec, place):
        return deepcopy(vec)

    def raises(self, vec, place):
        return deepcopy(vec)

    def norm_up(self, vec, place):
        vec_co = self.lower(vec, place)
        return sqrt(dot(vec, vec_co))

    def norm_down(self, vec, place):
        vec_con = self.raises(vec, place)
        return sqrt(dot(vec_con, vec))

    def __str__(self):
        return "Metric: Working with Metric direct internals (Default)"

class Metric(Default):
    """
    Includes metrix relevant functions, like for
    transforming contra- and covariant vectors into
    each other

        >>> from numpy import pi

    Use spherical coordiantes for testing:

        >>> from quat import r3
        >>> met = Metric(r3)

    Arbitrary point in space:

        >>> Y = array([8., pi/2. - 0.3, 0.2])

    Contravariant coordiantes of a vector:

        >>> dY = array([0.0002, 0.001, 0.001])

        >>> Y1 = Y + dY

    Covariant coordinates of the vector:

        >>> dy = met.lower(dY, Y)

    Verify that back transformation gives the original vector:

        >>> max(abs(dY - met.raises(dy, Y))) < 1e-15
        True

    Square norm of the vector:

        >>> dy2 = dot(dY, dy)

    Cartesian vector and its square:

        >>> dx = r3(Y1) - r3(Y)
        >>> dx2 = dot(dx, dx)

    Both should be similar, but not identical:

        >>> abs(dy2 - dx2) < 1e-6
        True

    Use norm function of metric (attention, they give back the sqrt)

        >>> abs(met.norm_up(dY, Y) - sqrt(dy2)) < 1e-15
        True

        >>> abs(met.norm_down(dy, Y) - sqrt(dy2)) < 1e-15
        True

    FIXME: offering two ways of computing norms here and returning
    a square root may be a bad decision. Do not consider "norm_up"
    and "norm_down" a part of the Metric interface!
    """
    def __init__(self, fun):
        """
        needs a function with a fprime function
        """

        # FIXME: what if analytical derivatives are available?
        self.fun = NumDiff(fun)

    def _fprime_as_matrix(self, x):
        """
        Some functions return not linear arrays of different shapes.
        Then fprime will be not a 2-dimensional array but more.
        Thus transform the result to a matrix as if the
        function used for fprime would give back a flattened result.
        """

        # FIXME: ensure that "fprime" is always returned as an array instead:
        fprime = asarray(self.fun.fprime(x))

        # convert shape of fprime, so that second dimension is length of vector x
        fprime.shape = (-1, size(x))

        return fprime

    def lower(self, vec, place):
         """
         Assuming that F is a function to get the derivatives, transforming vectors
         of kind vec into cartesians, and that all takes place at position pos,
         this function transforms contra in covariant vectors.

         FIXME: description outdated.
         """

         B = self._fprime_as_matrix(place)

         return dot(B.T, dot(B, vec))

    def raises(self, vec, place):
         """
         Assuming that F is a function to get the derivatives, transforming vectors
         of kind vec into cartesians, and that all takes place at position pos,
         this function transforms co in contravariant vectors.

         FIXME: description outdated.
         """

         B = self._fprime_as_matrix(place)

         M = dot(B.T, B)

         return solve(M, vec)


    def __str__(self):
        return "Metric: Working with Metric Cartesians (Metric)"

class Metric_reduced(Metric):
    """
    Includes metrix relevant functions, like for
    transforming contra- and covariant vectors into
    each other
    This one uses Cartesian metric but with ensuring that there are no
    global rotation and translation. Therefore it is only designed for
    use of some kind of internal coordinates, which do not already include
    them.

    >>> from numpy import pi

    Use a real function:
    >>> from quat import r3
    >>> met = Metric_reduced(r3)

    Be careful that the angles are not too large
    >>> x = array([1.0, 2.0, 0.5, 1.0 , 0.6 , 0.1])

    Vector to transform
    >>> f_co = array([1.0, 2.0, 0.5, 1.0 , 0.6 , 0.1])

    >>> from ase import Atoms
    >>> from ase.calculators.lj import LennardJones
    >>> from pts.zmat import ZMat
    >>> ar4 = Atoms("Ar4")

    >>> ar4.set_calculator(LennardJones())

    >>> td_s = 1.12246195815
    >>> td_w = 60.0 / 180. * pi
    >>> td_d = 70.5287791696 / 180. * pi

    >>> func = ZMat([(), (0,), (0, 1), (1, 2, 0)])
    >>> met = Metric_reduced(func)
    >>> x = [td_s, td_s, td_w, td_s, td_w, td_d]
    >>> carts = func(x)
    >>> ar4.set_positions(carts)
    >>> f_cart = ar4.get_forces().flatten()
    >>> B = func.fprime(x)
    >>> B.shape = (-1, len(x))
    >>> f_co = dot(B.T, f_cart)
    >>> f_con = met.raises(f_co, x)
    >>> (abs(f_co - met.lower(f_con,x))).max() < 1e-12
    True
    >>> from pts.constr_symar4 import t_c2v, t_c2v_prime
    >>> max(t_c2v(x)) < 1e-15
    True
    >>> max(t_c2v_prime(f_con)[1]) < 1e-9
    True
    """
    def lower(self, vec, place):
        return asarray(contoco_red(self._fprime_as_matrix, self.fun, place, vec))

    def raises(self, vec, place):
        return asarray(cotocon_red(self._fprime_as_matrix, self.fun, place, vec))

    def __str__(self):
        return "Metric: Working with Metric Cartesians with remove of global translation and rotation"


class No_metric():
    def __init__(self, fun = None):
        pass

    def __str__(self):
        s =  "Metric: No metric set! Please initalize a valid metric before using it."
        s = s + '\n' + "This metric does not provide any other functionalities"
        return s
"""
The chosen metric, available are:
   * Default, for which contra- and covariant vectors are the same
   * and Metric, using a Cartesian metric

Use metric.version() to find out which one is set currently.
Store and initalize the choosen metric in metric
Here set up the global variable.
Before use make sure that it is initalized with a fitting
variable by function setup_metric.
Do not import metric directly, as it would be the None version
but use for example

 import pts.metric as mt
 ...
 co_vec = mt.lower(con_vec)
"""
global metric
metric = No_metric()

def setup_metric(F = None):
     """
     sets and initalises the metric
     F should be a function, which should when run by
     itself provide for the internal coordinates y
     the corresponding Cartesian coordinates x
     This function has to be called once, afterwards
     all modules should be able to access the metric.
     """
     global metric
     metric = Default(F)
     #print metric

def contoco_red(F, f, pos, vec):
    """
    Assuming that F is a function to get the derivatives, transforming vectors
    of kind vec into cartesians, while f is the function itself
    and that all takes place at position pos,
    this function transforms contra in covariant vectors with removing the
    parts for global coordinates in the covariant vector space.
    The transformation between the two is done with the expression:
    vec_i = B.T(I - BT (BT.T Bt)^-1 BT.T - BR (BR.T BR)^-1 BR.T) B vec^i
    B(pos) is the transformation just in internals, BT and BR are the matrices
    for translation and rotation (special choice with no interlap)
    """
    B = F(pos)
    # positions needed for the global rotation matrix
    p_cart = f(pos)
    # get the matrices for the global parameter (without interlap)
    BT, BT2_inv , BR, BR2_inv = B_globals(p_cart)
    # dy_i = B.T(I - BT (BT.T Bt)^-1 BT.T - BR (BR.T BR)^-1 BR.T) B dy^i
    M = dot(B, vec)
    M_T = dot(BT, dot(BT2_inv, dot(BT.T, M)))
    M_R = dot(BR, dot(BR2_inv, dot(BR.T, M)))
    return dot(B.T, M - M_T - M_R)


def cotocon_red(F, f, pos, vec):
    """
    Assuming that F is a function to get the derivatives, transforming vectors
    of kind vec into cartesians, while f is the function itself
    and that all takes place at position pos,
    this function transforms co in contravariant vectors with removing the
    parts for global coordinates in the covariant vector space.
    The transformation between the two is done with the expression:
    vec_i = B.T(I - BT (BT.T Bt)^-1 BT.T - BR (BR.T BR)^-1 BR.T) B vec^i
    B(pos) is the transformation just in internals, BT and BR are the matrices
    for translation and rotation (special choice with no interlap)
    """
    B = F(pos)
    # positions needed for the global rotation matrix
    p_cart = f(pos)
    # get the matrices for the global parameter (without interlap)
    BT, BT2_inv, BR, BR2_inv = B_globals(p_cart)
    # dy_i = B.T(I -  BT(BT.T BT)^-1 BT.T - BR (BR.T BR)^-1 BR.T) B dy^i
    M_T = dot(BT, dot(BT2_inv, dot(BT.T, B)))
    M_R = dot(BR, dot(BR2_inv, dot(BR.T, B)))
    M = dot(B.T, B - M_T - M_R)
    return solve(M, vec)

def B_globals(carts):
    """
    Calculates the matrices B_T and B_R as needed for removing the global positions
    B_T is just the derivatives for the translation and is thus fixed, note that
    the matrix (B_T.T B_T)^-1 is just N * eye(3,3)
    The matrix B_R is build with the geometrical center as rotation center, thus
    rotation and translation do not interact with each other. The inverse of
    (B_R.T, B_R) is also already calculated numerically.
    returns B_T, (B_T.T B_T)^-1, B_R, (B_R.T, B_R)^-1

    >>> from numpy import sin, cos, pi
    >>> d = 1.2
    >>> carts = array([[  0.,   0.,   0.],
    ...                [   d, d/2.,   0.],
    ...                [d/2.,    d,   0.],
    ...                [   d,    d,   0.]])

    >>> BT, BT2i, BR, BR2i = B_globals(carts)
    >>> (abs(dot(BT.T, BT) - eye(3)*size(carts,0))).max() < 1e-10
    True
    >>> (abs(dot(BR2i, dot(BR.T, BR)) - eye(3))).max() < 1e-10
    True
    >>> (abs(dot(BT2i, dot(BT.T, BT)) - eye(3))).max() < 1e-10
    True
    >>> abs(dot(BT.T, BR)).max() < 1e-10
    True

    >>> def fun2(x):
    ...     return array([[0., 0., 0.],
    ...                   [x[0], 0., 0.],
    ...                   [-x[0], 0., 0.],
    ...                   [x[1] * sin(x[2]), x[1] * cos(x[2]), 0.],
    ...                   [-x[1] * sin(x[2]), -x[1] * cos(x[2]), 0.],
    ...                   [x[1] * sin(x[2]) + x[3] * sin(x[4]) * cos(x[5]) ,
    ...                    x[1] * cos(x[2]) + x[3] * cos(x[4]) * cos(x[5]) ,
    ...                     - x[3] * sin(x[5])],
    ...                    [- x[1] * sin(x[2]) - x[3] * sin(x[4]) * cos(x[5]) ,
    ...                     - x[1] * cos(x[2]) - x[3] * cos(x[4]) * cos(x[5]) ,
    ...                      x[3] * sin(x[5])]]
    ...                     )

    >>> fun = NumDiff(fun2)
    >>> y = asarray([d, d/2., 0.1, -d, 0.4, 0.4])
    >>> t = asarray([0.,0.,0.])
    >>> rot = asarray([0.,0.,0.])
    >>> x = fun2(y)
    >>> from pts.zmat import RT
    >>> r = RT(fun)
    >>> B1, BR1, BT1 =  r.fprime(y, rot, t)
    >>> BT, BT2i, BR, BR2i = B_globals(x)
    >>> (abs(BT.flatten() - BT1.flatten())).max() < 1e-12
    True
    >>> (abs(BR.flatten() - BR1.flatten())).max() < 1e-10
    True
    >>> (abs(dot(BR2i, dot(BR.T, BR)) - eye(3))).max() < 1e-10
    True
    >>> (abs(dot(BT2i, dot(BT.T, BT)) - eye(3))).max() < 1e-10
    True
    >>> abs(dot(BT.T, BR)).max() < 1e-10
    True
    >>> y = asarray([-d + 0.1, d*0.8, 0.03, d*0.8, 1.0, 0.8])
    >>> t = asarray([0.,0.,0.])
    >>> rot = asarray([0.,0.,0.])
    >>> x = fun2(y)
    >>> from pts.zmat import RT
    >>> r = RT(fun)
    >>> B1, BR1, BT1 =  r.fprime(y, rot, t)
    >>> BT, BT2i, BR, BR2i = B_globals(x)
    >>> (abs(BT.flatten() - BT1.flatten())).max() < 1e-12
    True
    >>> (abs(BR.flatten() - BR1.flatten())).max() < 1e-10
    True
    >>> (abs(dot(BR2i, dot(BR.T, BR)) - eye(3))).max() < 1e-10
    True
    >>> (abs(dot(BT2i, dot(BT.T, BT)) - eye(3))).max() < 1e-10
    True
    >>> abs(dot(BT.T, BR)).max() < 1e-10
    True
    """

    #
    # FIXME: this code necessarily assumes that "carts" is an array
    #        of cartesian coordinates. Otherwise "translations" and
    #        "rotation" modes as used implicitly below have no meaning.
    #        Why cannot we assume that we are always passed an (N, 3)
    #        array?
    #
    carts = carts.view().reshape(-1, 3)

    # cartesian coordinates with geometrical center as origin:
    carts = carts - center(carts)

    # number of atoms:
    N = len(carts)

    # Matrices to hold translation and rotation modes, BT and BR.
    # For each of N atoms a 3x3 matrix, or rather for each of 3 modes
    # an Nx3 matrix [both will be reshaped to (N*3, 3) later]:
    BT = empty((N, 3, 3))
    BR = empty((N, 3, 3))

    # set the matrices
    for i in range(N):
        x, y, z = carts[i]

        # BT modes for translations, note normalization:
        BT[i, :, :] = eye(3)

        # BR modes as rotations around the center:
        BR[i, :, :] = array([[ 0,  z, -y],
                             [-z,  0,  x],
                             [ y, -x,  0]])

    # FIXME: we handle cartesian coordinates as linear arrays
    #        in order to use numpy.dot(,) later:
    BT.shape = (N*3, 3)
    BR.shape = (N*3, 3)

    #              T
    # Inverse of BT  *  BT, that is of pairwise dot products of the mode vectors:
    #
    gammaT = eye(3) / N

    #              T
    # Inverse of BR  *  BR, that is of pairwise dot products of the mode vectors:
    #
    gammaR = inv3(inertia(carts))

    return BT, gammaT, BR, gammaR


def center(rs):
    """Compute mass center
    assuming UNIT masses.
    """

    x = zeros((3))

    for r in rs:
        x += r

    return x / len(rs)

def inertia(rs):
    """Compute 3x3 inertia tensor
             __
             \    2
        I  = /_  r  * delta  -  r * r
         ij   r            ij    i   j

    assuming UNIT masses.

        >>> w = 2.0
        >>> rs = array([[ w,  w,  w],
        ...             [-w, -w,  w],
        ...             [ w, -w, -w],
        ...             [-w,  w, -w]])

        >>> inertia(rs)
        array([[ 32.,   0.,   0.],
               [  0.,  32.,   0.],
               [  0.,   0.,  32.]])
    """

    I = zeros((3,3))
    for r in rs:
        for i in range(3):
            for j in range(3):
                I[i, j] -= r[i] * r[j]

            I[i, i] += dot(r, r)

    return I

def inv3(m):
    """Compute inverse of a 3x3 matrix by Cramers method.

    Use rotation matrix for testing:

        >>> from quat import rotmat
        >>> m = rotmat(array([0.5, 1.5, 2.]))

    Inverse:

        >>> m1 = inv3(m)
        >>> max(abs(dot(m1, m) - eye(3))) < 1e-16
        True
        >>> max(abs(dot(m, m1) - eye(3))) < 1e-16
        True
    """

#   # FIXME: this code is broken. It may work though
#   #        for symmetric m!
#   a = m[1, 1] * m[2, 2] - m[2, 1] * m[2, 1]
#   b = m[2, 1] * m[2, 0] - m[2, 2] * m[1, 0]
#   c = m[1, 0] * m[2, 1] - m[2, 0] * m[1, 1]

#   d = b
#   e = m[0, 0] * m[2, 2] - m[2, 0] * m[2, 0]
#   f = m[1, 0] * m[2, 0] - m[0, 0] * m[2, 1]

#   g = c
#   h = f
#   k = m[0, 0] * m[1, 1] - m[1, 0] * m[1, 0]

#   z = m[0, 0] * a + m[1, 0] * b + m[2, 0] * c

#   minv = array([[a, d, g], [b, e, h], [c, f, k]]) / z

    # FIXME: why asarray() is necessary here? The caller
    #        seems not to accept a matrix() as result:
    return asarray(matrix(m).I)

# Testing the examples in __doc__strings, execute
# "python gxmatrix.py", eventualy with "-v" option appended:
if __name__ == "__main__":
     import doctest
     doctest.testmod()


