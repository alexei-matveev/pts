"""
    >>> from numpy import array, max, abs, dot

This is a tetrahedral geometry:

    >>> w = 0.39685026
    >>> A = array([[ w,  w,  w],
    ...            [-w, -w,  w],
    ...            [ w, -w, -w],
    ...            [-w,  w, -w]])

Let the volume  of the thetrahedron be the  "reaction coordiante" or a
"property" of the system:

    >>> from rc import Volume
    >>> v = Volume()

This choice of the reaction coordinate appears to lead to a "negative"
volume. This is an artifact of the orientation:

    >>> v(A)
    -0.99999997738152069

Let us "measure" the changes of  geometry by the absolute value of the
volumen change:

    >>> m = Metric(v)

This is a change in geometry corresponding to breathig mode:

    >>> dA = 0.0001 * A

This is the corresponding "measure":

    >>> m.norm_up(dA, A)
    0.00029999999321445622

Compare that with a central difference:

    >>> v(A - dA / 2.) - v(A + dA / 2.)
    0.00029999999346475015

And two one-sided differences:

    >>> v(A - dA) - v(A)
    0.00029996999421522119

    >>> v(A + dA) - v(A)
    -0.00030002999421374632

Metric is unsigned, of course:

    >>> m.norm_up(dA, A) == m.norm_up(-dA, A)
    True
"""
from numpy import dot, array, asarray, size, copy
from numpy import sqrt
from numpy import zeros, empty, eye, shape
from numpy.linalg import solve, norm


__all__ = ["cartesian_norm", "setup_metric", "metric"]

def cartesian_norm(dx, x):
    """
    Default  cartesian  norm of  |dx|,  |x|  is  ignored.  Works  with
    arguments of  any shape,  not only with  1D arrays.   Emulates the
    interface of norm_up/norm_down methods of the classes below.
    """

    return norm(dx)

class Default:
    """
    Implements  metric  relevant   functions,  like  for  transforming
    contra- and covariant vectors into each other

    This variant  supposes that contra- and covariant  vectors are the
    same, coordinate transformation if provided is ignored:

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
        Normally  needs a  function  with a  fprime  function in  this
        special case it is only asked for it because of consistency
        """
        pass

    def lower(self, dX, X):
        return copy(dX)

    def raises(self, dx, X):
        return copy(dx)

    def norm_up(self, dX, X):

        dx = self.lower(dX, X)

        # flat views of dx and dX:
        dx_ = dx.reshape(-1)
        dX_ = dX.reshape(-1)

        return sqrt(dot(dX_, dx_))

    def norm_down(self, dx, X):

        dX = self.raises(dx, X)

        # flat views of dx and dX:
        dx_ = dx.reshape(-1)
        dX_ = dX.reshape(-1)

        return sqrt(dot(dX_, dx_))

    def g (self, X):
        """
        Returns a  "matrix" representation of  the metric at X.  It is
        only then a  true matrix when X is 1D  array. In general shape
        (g) = shape (X) + shape (X).
        """

        # Components here will be flipped to 1:
        dX = zeros (shape (X))

        # flat views of X and dx:
        X_ = X.reshape (-1)
        dX_ = dX.reshape (-1)

        g = empty ((size (X), size (X)))
        for i in xrange (size (X)):
            dX_[i] = 1.0
            dx = self.lower (dX, X)
            g[:, i] = dx.reshape (-1)
            dX_[i] = 0.0

        g.shape = shape (X) + shape (X)

        return g

    def __str__(self):
        return "Metric: Working with Metric direct internals (Default)"

class Metric(Default):
    """
    Includes metrix relevant  functions, like for transforming contra-
    and covariant vectors into each other

        >>> from numpy import pi, max, abs

    Use spherical coordiantes for testing:

        >>> from quat import r3
        >>> met = Metric(r3)

    Arbitrary point in space:

        >>> Y = array([8., pi/2. - 0.3, 0.2])

    Contravariant coordiantes of a vector:

        >>> dY = array([0.0002, 0.001, 0.001])

    Covariant coordinates of the vector:

        >>> dy = met.lower(dY, Y)

    Verify that back transformation gives the original vector:

        >>> max(abs(dY - met.raises(dy, Y))) < 1e-15
        True

    Square norm of the vector:

        >>> dy2 = dot(dY, dy)

    Cartesian vector and its square:

        >>> dx = r3(Y + dY) - r3(Y)
        >>> dx2 = dot(dx, dx)

    Both should be similar, but not identical:

        >>> abs(dy2 - dx2) < 1e-6
        True

    Use norm function of metric (attention, they give back the sqrt)

        >>> abs(met.norm_up(dY, Y) - sqrt(dy2)) < 1e-15
        True

        >>> abs(met.norm_down(dy, Y) - sqrt(dy2)) < 1e-15
        True

    FIXME: offering two  ways of computing norms here  and returning a
    square root may  be a bad decision. Do  not consider "norm_up" and
    "norm_down" a part of the Metric interface!
    """
    def __init__(self, fun):
        """
        needs a function with a fprime function
        """

        self.fun = fun

    def _fprime_as_matrix(self, X):
        """
        Some functions  return not linear arrays  of different shapes.
        Then fprime will be not  a 2-dimensional array but more.  Thus
        transform the result  to a matrix as if  the function used for
        fprime would give back a flattened result.

        Not used in this class,  maybe in subclasses, but only in this
        file! I would let it die.
        """

        # FIXME: ensure that "fprime" is always returned as an array instead:
        fprime = asarray(self.fun.fprime(X))

        # Convert shape of fprime,  so that second dimension is length
        # of vector x. Rather return a rectangular view of the matrix:
        return fprime.reshape(-1, size(X))

    def lower(self, dX, X):
        """
        Transform   contravariant  coordiantes   dX   into  covariant
        coordiantes  dx using the  derivatives of  the differentiable
        transformation at X.

        This   should  also  work   with  rank-deficient   matrix  of
        derivatives B.
        """

        dx = empty(shape(dX))

        # flat views of dx and dX:
        dx_ = dx.reshape(-1)
        dX_ = dX.reshape(-1)

        # rectangular view of the trafo derivatives:
        B = self.fun.fprime(X).reshape(-1, dX_.size)

        # destructive update of a locally created var:
        dx_[:] = dot(B.T, dot(B, dX_))

        # return original view:
        return dx

    def raises(self, dx, X):
        """
        Transform   covariant  coordiantes   dx   into  contravariant
        coordiantes  dX using the  derivatives of  the differentiable
        transformation at X.

        FIXME:  This will  fail  if  the matrix  of  derivatves B  is
        rank-deficient and, hence, the corresponding M is singular.
        """

        dX = empty(shape(dx))

        # flat views of dx and dX:
        dx_ = dx.reshape(-1)
        dX_ = dX.reshape(-1)

        # rectangular view of the trafo derivatives:
        B = self.fun.fprime(X).reshape(-1, dx_.size)

        M = dot(B.T, B)

        # destructive update of a locally created var:
        dX_[:] = solve(M, dx_)

        # return original view:
        return dX

    # FIXME: class specific implementation for .g(X) method?

    def __str__(self):
        return "Metric: Working with Metric Cartesians (Metric)"

class Metric_reduced(Metric):
    """
    Includes metrix relevant  functions, like for transforming contra-
    and covariant vectors into each other.

    This one uses Cartesian metric but with ensuring that there are no
    global rotation and translation. Therefore it is only designed for
    use of  some kind  of internal coordinates,  which do  not already
    include them.

    Potential energy surface of Ar4,  energy gradients will be used as
    a trial covariant vector:

        >>> from ase import Atoms
        >>> from qfunc import QFunc

        >>> E = QFunc(Atoms("Ar4"))

    Tetrahedral geometry, inflated by 5%:

        >>> w = 0.39685026 * 1.05
        >>> X = array([[ w,  w,  w],
        ...            [-w, -w,  w],
        ...            [ w, -w, -w],
        ...            [-w,  w, -w]])

    Imagine  we  use  a  steepest  descent procedure,  where  step  is
    proportional to  the gradient.  In cartesian coordinates  this may
    look like this:

        >>> lam = 0.01
        >>> dX = - lam * E.fprime(X)

    Coordinate transformation:

        >>> from zmat import ZMat

        >>> Z = ZMat([(), (0,), (0, 1), (1, 2, 0)])

    Internal coordiantes corresponding to the (inflated) geometry:

        >>> Y = Z.pinv(X)

    Note  what effect  would  a  certesian step  dX  have on  internal
    coordinates:

        >>> dY = Z.pinv(X + dX) - Z.pinv(X)

        >>> from numpy import round, max, abs

        >>> round(dY, 3)
        array([-0.077, -0.077,  0.   , -0.077,  0.   ,  0.   ])

    Only  the  bond lengths  would  shrink  by  the same  amount  thus
    preserving the  symmetry. Now  try this with  internal coordinates
    ...

    Potential energy surface as a function of internal coordinates:

        >>> from func import compose

        >>> E1 = compose(E, Z)

    What if we made the  step proportional to the gradient in internal
    coordiantes?

        >>> dY = - lam * E1.fprime(Y)
        >>> round(dY, 3)
        array([-0.039, -0.039, -0.026, -0.039, -0.026, -0.016])

    This would  also change  the angles and  break the  symmetry.  The
    reduced metric helps in this case if you remember to never confuse
    co- and contravariant coordinates
    (FIXME:   use   analytic derivatives, provided by Z.fprime!):

        >>> g = Metric_reduced(Z)

    These are the covariant coordiantes of a steepest descent step:

        >>> dy = - lam * E1.fprime(Y)

    And these are the contravariant ones:

        >>> dY = g.raises(dy, Y)

        >>> round(dY, 3)
        array([-0.077, -0.077,  0.   , -0.077,  0.   ,  0.   ])

    Consistency check:

        >>> max(abs(dy - g.lower(dY, Y))) < 1e-16
        True
    """
    def lower(self, dY, Y):
        """
        Assuming  that  F  is  a  function  to  get  the  derivatives,
        transforming vectors  of kind vec into cartesians,  while f is
        the function itself and that  all takes place at position pos,
        this  function  transforms contra  in  covariant vectors  with
        removing  the parts  for global  coordinates in  the covariant
        vector space.  The transformation between the two is done with
        the expression:

        vec_i = B.T(I - BT (BT.T Bt)^-1 BT.T - BR (BR.T BR)^-1 BR.T) B vec^i

        B(pos) is the transformation just  in internals, BT and BR are
        the matrices for translation and rotation (special choice with
        no interlap)

        FIXME: description outdated.

        Compute    covariant   coordinates    dy    corresponding   to
        contravariant dY by

                  T                  T            T
            dy = B  * ( I - B * g * B  - B * g * B ) * B * dY
                             t   t   t    r   r   r

        with
                   T       -1
            g = ( B  *  B )
             t     t     t
        and
                   T       -1
            g = ( B  *  B )
             r     r     r

        with B, B, B evaluated at Y.
                 r  t
        """

        # positions needed for the global rotation matrix:
        X = self.fun(Y)

        # get the matrices for the global parameter (without interlap):
        BT, gT , BR, gR = B_globals(X)

        B = self._fprime_as_matrix(Y)

        # this is a cartesian vector corresponding to dY:
        dX = dot(B, dY)

        # these are components of dX that are translations and rotations,
        # respectively:
        dXT = dot(BT, dot(gT, dot(BT.T, dX)))
        dXR = dot(BR, dot(gR, dot(BR.T, dX)))

        return dot(B.T, dX - dXT - dXR)

    def raises(self, dy, Y):
        """
        Assuming  that  F  is  a  function  to  get  the  derivatives,
        transforming vectors  of kind vec into cartesians,  while f is
        the function itself and that  all takes place at position pos,
        this  function  transforms co  in  contravariant vectors  with
        removing  the parts  for global  coordinates in  the covariant
        vector space.  The transformation between the two is done with
        the expression:

        vec_i = B.T(I - BT (BT.T Bt)^-1 BT.T - BR (BR.T BR)^-1 BR.T) B vec^i

        B(pos) is the transformation just  in internals, BT and BR are
        the matrices for translation and rotation (special choice with
        no interlap)

        FIXME: description outdated.
        """

        # positions needed for the global rotation matrix:
        X = self.fun(Y)

        # get   the  matrices  for   the  global   parameter  (without
        # interlap):
        BT, gT , BR, gR = B_globals(X)

        B = self._fprime_as_matrix(Y)

        #
        # See  description  of  self.lower()  that  defines  a  linear
        # relation
        #
        #       g * dY = dy
        #
        # between contra-  and covariant  coordinates, dY and  dy, for
        # detailed definition of matrix g.
        #
        # FIXME: isnt  modified metric singular? Why do  we assume the
        #        linear equation has a solution? At least the modified
        #        cartesian metric is singular.
        #

        # unmodified metric, NY x NY, where NY = dim(Y):
        g = dot(B.T, B)

        # projections  of  "internal"   modes  onto  translations  and
        # rotations, both NY x 3:
        T = dot(B.T, BT)
        R = dot(B.T, BR)

        # modified metric:
        g = g - dot(T, dot(gT, T.T)) - dot(R, dot(gR, R.T))

        return solve(g, dy)

    def __str__(self):
        return "Metric: Working with Metric Cartesians with remove of global translation and rotation"


"""
The chosen metric, available are:

   * Default, for which contra- and covariant vectors are the same

   * Metric, using a Cartesian metric

   * Metric_reduced, using a Cartesian metric but removing the effects
     of global rotation and  translation. This metric makes only sense
     for  systems in internal  coordinates if  the coordinates  do not
     contain already any kind  of variables for global positioning. It
     should be  expected that a  change in the global  postitions will
     have no effect on the energy of the system.

Use print  metric to find out  which one is set  currently.  Store and
initalize  the  choosen  metric  in  metric Here  set  up  the  global
variable.  Before use  make sure that it is  initalized with a fitting
variable by function setup_metric.   Do not import metric directly, as
it would be the starting version but use for example

 import pts.metric as mt
 ...
 co_vec = mt.lower(con_vec)
"""
global metric
metric = Default()

def setup_metric(F = None):
    """
    Sets and  initalises the  metric.  F should  be a  function, which
    should when run  by itself provide for the  internal coordinates y
    the corresponding Cartesian coordinates  x This function has to be
    called once, afterwards  all modules should be able  to access the
    metric.
    """
    global metric
    metric = Default(F)

def B_globals(carts):
    """
    Calculates the matrices BT and BR holding translation and rotation
    modes  as   needed  for  removing  the  trivial   componets  of  a
    (displacement) vector.

    BT  is  just the  derivatives  for  the  translation and  is  thus
    indpendent of  geometry. The  inverse of the  matrix BT^T *  BT is
    just eye(3) / N.

    The matrix  BR is  build with the  geometrical center  as rotation
    center,  thus rotation  and  translation modes  are orthogonal  by
    construction.  The inverse of BR^T * BR is calculated numerically.

    Returns a 4-tuple (BT, gT, BR, gR) with 3x3 matrices

                T      -1
        gT = (BT  * BT)

    and
                T      -1
        gR = (BR  * BR)

    Translations and rotations modes as returned by this function
    are mutually orthogonal:

               T
        0 = (BT  * BR)


    Trial geometry (tetrahedron):

        >>> from numpy import max, abs
        >>> w = 2.0
        >>> X = array([[ w,  w,  w],
        ...            [-w, -w,  w],
        ...            [ w, -w, -w],
        ...            [-w,  w, -w]])

        >>> BT, gT, BR, gR = B_globals(X)

    The 2-contravariant tensor

             -1
        g = G

    is an inverse of the 2-covariant Gram matrix

             T
        G = B  * B

    for two kinds of trivial vectors, translations and rotatins.
    Verify that for rotations ...

        >>> GR = dot(BR.T, BR)

        >>> max(abs(dot(gR, GR) - eye(3))) < 1e-16
        True

        >>> max(abs(dot(GR, gR) - eye(3))) < 1e-16
        True

    and translations ...

        >>> GT = dot(BT.T, BT)

        >>> max(abs(dot(gT, GT) - eye(3))) < 1e-16
        True

        >>> max(abs(dot(GT, gT) - eye(3))) < 1e-16
        True

    Check that translation and rotation modes are mutually orthogonal:

        >>> max(abs(dot(BT.T, BR))) < 1e-16
        True

    Only in this case is it meaningfull to separate the two subspaces.

    Below is  a somewhat obscure  way to verify that  translations are
    translations and rotations are rotations.

        >>> from zmat import RT

    Constructor  of an  RT  object expect  a differentiable  function,
    prepare  a  function  that  always returns  the  same  tetrahedral
    geometry:

        >>> def f(y):
        ...     return X

    This one also provides derivatives:

        >>> from func import NumDiff
        >>> f = NumDiff(f)

    This  one depends  on rotational  and translational  parameters in
    addition:

        >>> f = RT(f)

    Values for rotational and translational parameters:

        >>> O = array([0., 0., 0.])

    The  first argument  of f(y,  rot, trans)  is ignored,  so  is the
    corresponding derivative:

        >>> _, BR1, BT1 =  f.fprime(O, O, O)

    For this particular example derivatives with respect to rotational
    and translational parameters coincide:

        >>> max(abs(BT.flatten() - BT1.flatten())) < 1e-16
        True

        >>> max(abs(BR.flatten() - BR1.flatten())) < 1e-16
        True

    Note that with a different choice of, say, geometrical center of X
    offset from origin one cannot expect this agreement anymore.
    """

    #
    # FIXME: this code necessarily assumes that "carts" is an array of
    #        cartesian   coordinates.  Otherwise   "translations"  and
    #        "rotation"  modes  as   used  implicitly  below  have  no
    #        meaning.  Why cannot we  assume that we are always passed
    #        an (N, 3) array? In  fact the code below "assumes" that N
    #        > 2 otherwise removal of 6 degrees of freedom is not well
    #        defined. Think  of a  singular inertia tensor  for single
    #        atom or a diatomic molecule.
    #
    carts = carts.view().reshape(-1, 3)

    # cartesian coordinates with geometrical center as origin:
    carts = carts - center(carts)

    # number of atoms:
    N = len(carts)

    # Matrices to hold translation and rotation modes, BT and BR.  For
    # each of N atoms  a 3x3 matrix, or rather for each  of 3 modes an
    # Nx3 matrix [both will be reshaped to (N*3, 3) later]:
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

    # NOTE: we handle cartesian  coordinates as linear arrays in order
    #       to use numpy.dot(,) later:
    BT.shape = (N*3, 3)
    BR.shape = (N*3, 3)

    #              T
    # Inverse of BT  *  BT, that is of pairwise dot products
    # of the mode vectors (Gram matrix):
    #
    gammaT = eye(3) / N

    #              T
    # Inverse of BR  *  BR, that is of pairwise dot products
    # of the mode vectors (Gram matrix):
    #
    # FIXME: inertia tensor for linear systems is singular!
    #
    gammaR = inv3(inertia(carts))

    return BT, gammaT, BR, gammaR


def center(rs):
    """Compute mass center assuming UNIT masses.
    """

    x = zeros((3))

    for r in rs:
        x += r

    return x / len(rs)

def inertia(rs):
    """Compute 3x3 inertia tensor
               __
               \    2
        I   =  /_  r  * delta  -  r * r
         ij     r            ij    i   j

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

    I = zeros((3, 3))
    for r in rs:
        for i in range(3):
            for j in range(3):
                I[i, j] -= r[i] * r[j]

            I[i, i] += dot(r, r)

    return I

def inv3(m):
    """Compute inverse of a 3x3 matrix by Cramers method.

    Use scaled rotation matrix for testing:

        >>> from numpy import max, abs
        >>> from quat import rotmat
        >>> m = rotmat(array([0.5, 1.5, 2.])) * 10.0

    Inverse:

        >>> m1 = inv3(m)
        >>> max(abs(dot(m1, m) - eye(3))) < 5e-16
        True
        >>> max(abs(dot(m, m1) - eye(3))) < 5e-16
        True
    """

    # adjugate matrix:
    w = adj3(m)

    # determinant:
    D = m[0, 0] * w[0, 0] + m[0, 1] * w[1, 0] + m[0, 2] * w[2, 0]

    return w / D

    # FIXME: why asarray() is necessary here? The caller
    #        seems not to accept a matrix() as result:
    # return asarray(matrix(m).I)

def adj3(m):
    """Returns adjugate of 3x3 m,
    http://en.wikipedia.org/wiki/Adjugate_matrix

    Example from Wikipedia:

        >>> m = array([[ -3,  2, -5 ],
        ...            [ -1,  0, -2 ],
        ...            [  3, -4,  1 ]])

        >>> adj3(m)
        array([[ -8.,  18.,  -4.],
               [ -5.,  12.,  -1.],
               [  4.,  -6.,   2.]])

    Adjugate of m with permuted rows,
    to test the case with m[1, 1] /= 0:

        >>> adj3(m[[1, 2, 0]])
        array([[ 18.,  -4.,  -8.],
               [ 12.,  -1.,  -5.],
               [ -6.,   2.,   4.]])
    """

    w = empty((3, 3))

    w[0, 0] = m[1, 1] * m[2, 2] - m[2, 1] * m[1, 2]
    w[0, 1] = m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2]
    w[0, 2] = m[0, 1] * m[1, 2] - m[1, 1] * m[0, 2]

    w[1, 0] = m[2, 0] * m[1, 2] - m[1, 0] * m[2, 2]
    w[1, 1] = m[0, 0] * m[2, 2] - m[2, 0] * m[0, 2]
    w[1, 2] = m[1, 0] * m[0, 2] - m[0, 0] * m[1, 2]

    w[2, 0] = m[1, 0] * m[2, 1] - m[2, 0] * m[1, 1]
    w[2, 1] = m[2, 0] * m[0, 1] - m[0, 0] * m[2, 1]
    w[2, 2] = m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]

    return w

# Testing the examples in __doc__strings, execute
# "python metric.py", eventualy with "-v" option appended:
if __name__ == "__main__":
    import doctest
    doctest.testmod()


