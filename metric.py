from func import NumDiff
from numpy import dot, asarray, matrix, size
from numpy import sqrt
from numpy import zeros, eye
from numpy.linalg import solve
from copy import deepcopy


__all__ = ["setup_metric", "metric"]


class Default:
    """
    Includes metrix relevant functions, like for
    transforming contra- and covariant vectors into
    each other

    This variant supposes that contra- and covariant vectors
    are the same

    >>> met = Default(None)
    >>> from numpy import pi

    >>> x = None

    >>> dx_con = asarray([0.0002, 0.001, 0.001])

    First transformation
    >>> dx_co = met.lower(dx_con, x)

    Verify: this metric changes nothing:
    >>> max(abs(dx_con - dx_co)) < 1e-15
    True

    Verify: this metric changes nothing:
    >>> max(abs(dx_con - met.raises(dx_co, x))) < 1e-15
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

    Use a real function:
    >>> from quat import r3
    >>> met = Metric(r3)

    >>> x = asarray([8., pi/2. - 0.3, 0.2])

    >>> dx_con = asarray([0.0002, 0.001, 0.001])

    >>> x2 = x + dx_con

    First transformation
    >>> dx_co = met.lower(dx_con, x)

    Verify: back transformation == start
    >>> max(abs(dx_con - met.raises(dx_co, x))) < 1e-15
    True

    >>> abs(dot(dx_con, dx_co) - dot(r3(x)-r3(x2), r3(x)-r3(x2))) < 1e-6
    True

    Use norm function of metric (attention, they give back the sqrt)

    >>> abs(met.norm_up(dx_con, x) - sqrt(dot(r3(x)-r3(x2), r3(x)-r3(x2)))) < 1e-6
    True

    >>> abs(met.norm_down(dx_co, x) - sqrt(dot(r3(x)-r3(x2), r3(x)-r3(x2)))) < 1e-6
    True
    """
    def __init__(self, fun):
        """
        needs a function with a fprime function
        """
        self.fun = NumDiff(fun)

    def _fprime_as_matrix(self, x):
        """
        Some functions return not linear arrays of different shapes.
        Then fprime will be not a 2-dimensional array but more.
        Thus transform the result to a matrix as if the
        function used for fprime would give back a flattened result.
        """
        a = asarray(self.fun.fprime(x))
        lenx = size(x)
        # convert shape of a, so that second dimension is length of vector x
        a.shape = (-1, lenx)
        return a

    def lower(self, vec, place):
        return asarray(contoco(self._fprime_as_matrix, place, vec))

    def raises(self, vec, place):
        return asarray(cotocon(self._fprime_as_matrix, place, vec))

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
    >>> x = asarray([1.0, 2.0, 0.5, 1.0 , 0.6 , 0.1])

    Vector to transform
    >>> f_co = asarray([1.0, 2.0, 0.5, 1.0 , 0.6 , 0.1])

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
     print metric

def contoco(F, pos, vec):
     """
     Assuming that F is a function to get the derivatives, transforming vectors
     of kind vec into cartesians, and that all takes place at position pos,
     this function transforms contra in covariant vectors.
     """
     B = F(pos)
     return btb(B, vec)

def cotocon(F, pos, vec):
     """
     Assuming that F is a function to get the derivatives, transforming vectors
     of kind vec into cartesians, and that all takes place at position pos,
     this function transforms co in contravariant vectors.
     """
     B = F(pos)
     M = dot(B.T, B)
     return solve(M, vec)

def btb(B, vec):
     """
     Returns the product B^T B vec
     """
     return dot(B.T, dot(B, vec))

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
    >>> carts = asarray([[0., 0., 0.],
    ...                  [d, d/2., 0.],
    ...                  [d/2, d, 0.],
    ...                  [d, d, 0.]])

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
    ...     return asarray([[0., 0., 0.],
    ...                    [x[0], 0., 0.],
    ...                    [-x[0], 0., 0.],
    ...                    [x[1] * sin(x[2]), x[1] * cos(x[2]), 0.],
    ...                    [-x[1] * sin(x[2]), -x[1] * cos(x[2]), 0.],
    ...                    [x[1] * sin(x[2]) + x[3] * sin(x[4]) * cos(x[5]) ,
    ...                     x[1] * cos(x[2]) + x[3] * cos(x[4]) * cos(x[5]) ,
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
    pcri = deepcopy(carts)
    # important for case with one electron where carts may be just a vector
    pcri.shape = (-1, 3)

    # centr should become the geometrical center
    centr = asarray([0., 0.,0.])
    N = size(pcri, 0)
    for c in carts:
       centr += c
    centr /= N
    # cartesian coordinates with geoemtrical center as origion
    pcri -= centr

    # for each of N atoms a 3x3 matrix:
    B_T = zeros((N*3, 3))
    B_R = zeros((N*3, 3))

    # set the matrices
    for i, pcr in enumerate(pcri):
        x, y, z = pcr
        # B_T is fixed, here B_T is also "normed" already
        B_T[i*3:(i+1)*3,:] = eye(3)

        # B_R is used for zero rotation around the center
        B_R[i*3:(i+1)*3,:] = asarray([[ 0,  z, -y],
                                      [-z,  0,  x],
                                      [ y, -x,  0]])

    # (B_T.T, B_T)^-1:
    Bt_inv = eye(3) / N


    # Now do (B_R.T, B_R)^-1:
    Br_inv = brtbrm1(pcri)

    return B_T, Bt_inv, B_R, Br_inv


def brtbrm1(coords):
    """
    Getting (B_R.T, B_R)^-1 from the coords.
    """

    # Br = B_R.T * B_R
    Br = zeros((3,3))
    for c in coords:
        for i in range(3):
            # only the upper half is needed, symmetry:
            for j in range(i+1):
                Br[i,j] -= c[i] * c[j]

            Br[i,i] += dot(c, c)

    # numerically do Br_inv = Br^-1
    a = Br[1,1]* Br[2,2] - Br[2,1] * Br[2,1]
    b = Br[2,1]* Br[2,0] - Br[2,2] * Br[1,0]
    c = Br[1,0]* Br[2,1] - Br[2,0] * Br[1,1]

    d = b
    e = Br[0,0]* Br[2,2] - Br[2,0] * Br[2,0]
    f = Br[1,0]* Br[2,0] - Br[0,0] * Br[2,1]

    g = c
    h = f
    k = Br[0,0]* Br[1,1] - Br[1,0] * Br[1,0]

    z = Br[0,0] * a + Br[1,0] * b + Br[2,0] * c
    Br_inv = asarray([[a,d,g], [b, e, h], [c, f, k]])/z
    return Br_inv

# Testing the examples in __doc__strings, execute
# "python gxmatrix.py", eventualy with "-v" option appended:
if __name__ == "__main__":
     import doctest
     doctest.testmod()


