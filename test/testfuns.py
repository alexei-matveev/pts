from numpy import size, shape, sqrt, asarray, array, zeros, dot, pi
from pts.func import Func
from math import cos, sin, acos, asin
from numpy import cosh, sinh, arccosh
from numpy import finfo
from numpy.linalg import solve, svd

class Affine (Func):
    """
    Affine transformation.  If  you ever want to invert  it, it should
    better be invertible. That was an insightful comment, wasn't it?

    A well-defined non-singular 2 -> 2 transformation:

        >>> trafo = Affine ([[1.0, 0.5],
        ...                  [0.0, 0.5]])

        >>> x = array([1.0, 1.0])

        >>> trafo(x)
        array([ 1.5,  0.5])

        >>> trafo.fprime(x)
        array([[ 1. ,  0.5],
               [ 0. ,  0.5]])

        >>> x - trafo.pinv(trafo(x))
        array([ 0.,  0.])

    Matter  get  slightly  more  complicated  for  an  injective  2->3
    transformation:

        >>> t1 = Affine([[1.0, 0.5], [0.0, 0.5], [10., 10.]])

    Here "x"  is in  the definition  domain and "y"  is in  the result
    domain:

        >>> y = t1(x)

    A  *proper*  result  "y"   (from  the  result  domain)  should  be
    sufficient to find the corresponding argument "x":

        >>> from numpy import max, abs

        >>> max(abs(x - t1.pinv(y))) < 1.0e-12
        True

    But  for an  *arbitrary*  vector "y1"  of  length 3,  there is  no
    correspondig "x1", in general:

        >>> y1 = array([1.3, 0.7, 9.0])
        >>> x1 = t1.pinv(y1)

        >>> max(abs(y1 - t1(x1))) > 0.5
        True

    A different case is a surjective, say 3->2, transformation:

        >>> t2 = Affine([[1.0, 0.0, 10.], [0.5, 0.5, 10]])

    An  arbitrary 2-vector  "x1"  has a  corresponding  "y1" from  the
    definition domain:

        >>> x1 = array([1.7, 0.2])
        >>> y1 = t2.pinv(x1)
        >>> max(abs(x1 - t2(y1))) < 1.0e-12
        True

    Of course,  there must be more  than one 3-vector  "y1" that gives
    that "x1". The "pinv" method finds just one of them:

        >>> y1 = array([1.3, 0.7, 9.0])
        >>> y2 = t2.pinv(t2(y1))
        >>> max(abs(t2(y1) - t2(y2))) < 1.0e-12
        True
    """

    def __init__(self, m):
        #
        # Affine transformation matrix:
        #
        self.__m = array (m)

    def f (self, x):
        # FIXME: more general shape of x?
        return dot (self.__m, x)

    def fprime (self, x):
        return self.__m.copy()

    def pinv(self, y):

        y = asarray(y)

        #
        # These share the  data with originall arrays, but  we are not
        # going to modify that:
        #
        Y = y.view()
        M = self.__m.view()

        Y.shape = size(y)
        M.shape = size(y), -1

        # Below we work with "matrices":
        m, n = shape(M)

        if m == n:
            #
            # This  is an  optimization, the  other branch  would also
            # work:
            #
            return solve(M, Y)
        else:
            #
            # SVD solves for M = U * S * V^T with (m x m) matrix U, (n
            # x n) matrix V and (m x n) "diagonal matrix" S.  Diagonal
            # "s" as returned by an SVD is a 1D-array, though:
            #
            u, s, vt = svd(M)

            #
            # This  computes V *  S^-1 *  U^T *  y for  a vector  y of
            # length m. First, compute U^T * y:
            #
            uty = dot(Y, u) # this is of length m

            #
            # Now the tricky part, we need the (pseudo) inverse of the
            # rectanglular S to compute S^-1 *  U^T *  y:
            #
            s1uty = zeros(n) # this is of length n
            k = min(m, n)
            s1uty[:k] = uty[:k] / s # eventually padded by zeroes

            #
            # Finally, multiply the result with V:
            #
            return dot(s1uty, vt) # this is of length n

def c2v_tetrahedron1():
    """
    Returns  a fresh  Func (Affine,  really) that  generates Cartesian
    coordinates of  a C2V-distorted tetrahedron.   Function gets three
    parameters (a, b, c) as an array and returns cartesian coordinates
    of four centers:

        [[+a, +a, +c],
         [-a, -a, +c],
         [+b, -b, -c],
         [-b, +b, -c]]

    An  ideal  tetrahedron is  generated  when  a ==  b  ==  c. For  a
    different way to orient tetrahedron see diagsandhight().

    Example:

        >>> f = c2v_tetrahedron1()
        >>> x = array([2., 3., 4.])
        >>> f(x)
        array([[ 2.,  2.,  4.],
               [-2., -2.,  4.],
               [ 3., -3., -4.],
               [-3.,  3., -4.]])
        >>> x - f.pinv(f(x))
        array([ 0.,  0.,  0.])
    """

    return Affine([[[  1.,  0.,  0.],
                    [  1.,  0.,  0.],
                    [  0.,  0.,  1.]],
                   [[ -1.,  0.,  0.],
                    [ -1.,  0.,  0.],
                    [  0.,  0.,  1.]],
                   [[  0.,  1.,  0.],
                    [  0., -1.,  0.],
                    [  0.,  0., -1.]],
                   [[  0., -1.,  0.],
                    [  0.,  1.,  0.],
                    [  0.,  0., -1.]]])

def diagsandhight():
    """
    Function diagsandhight:  generates Cartesian coordinates  (and the
    derivatives) from given values for  the two diagonals and the high
    between them.  The first two atoms will be situated on the X-Axis,
    equal far away from O.  The  other two atoms will have the same on
    the Y-Axis, but they are shifted in z-direction about hight.

    Function gets d1, d2 and h and calculates Cartesian coordinates

        d1 == vec[0] # diagonal with small changes
        d2 == vec[1] # diagonal with large changes
        h  == vec[2] # hight of two last atoms in z-direction

    The corresponding four positions of atoms are:

        [[ d1 / 2.,       0., 0.],
         [-d1 / 2.,       0., 0.],
         [      0.,  d2 / 2.,  h],
         [      0., -d2 / 2.,  h]]

    Example:

        >>> f = diagsandhight()
        >>> f(array([1., 1., 0.7]))
        array([[ 0.5,  0. ,  0. ],
               [-0.5,  0. ,  0. ],
               [ 0. ,  0.5,  0.7],
               [ 0. , -0.5,  0.7]])

    FIXME: choose a better name.
    """

    return Affine([[[ 0.5,   0., 0.],
                    [  0.,   0., 0.],
                    [  0.,   0., 0.]],
                   [[-0.5,   0., 0.],
                    [  0.,   0., 0.],
                    [  0.,   0., 0.]],
                   [[  0.,   0., 0.],
                    [  0.,  0.5, 0.],
                    [  0.,   0., 1.]],
                   [[  0.,   0., 0.],
                    [  0., -0.5, 0.],
                    [  0.,   0., 1.]]])

class mb1(Func):
    """
    2 -> 2 variables
    r, phi -> x, y
    """
    def __init__(self, cent = zeros(2)):
        self.centrum = cent

    def taylor(self, vec):
        """
        >>> fun = mb1()
        >>> from pts.func import NumDiff
        >>> from numpy import max, abs, dot, pi
        >>> fun2 = NumDiff(mb1())
        >>> x1 = array([1.,1.])
        >>> x2 = array([1., 0.])
        >>> x3 = array([0.01, 0.999 * pi])

        >>> max(abs(fun.fprime(x1) - fun2.fprime(x1))) < 1e-12
        True
        >>> max(abs(fun.fprime(x2) - fun2.fprime(x2))) < 1e-12
        True
        >>> max(abs(fun.fprime(x3) - fun2.fprime(x3))) < 1e-12
        True
        """
        r, phi = vec
        v = zeros(2)
        v[0] = self.centrum[0] + r * cos(phi)
        v[1] = self.centrum[1] + r * sin(phi)
        dv = zeros((2,2))
        dv[0,0] = cos(phi)
        dv[1,0] = sin(phi)
        dv[0,1] = -r* sin(phi)
        dv[1,1] = r *cos(phi)

        return v, dv

    def pinv(self, v):
        """
        >>> fun = mb1()
        >>> v = [0.,0.]
        >>> max(abs(v - fun.pinv(fun(v)))) < 1e-14
        True
        >>> max(abs(v - fun(fun.pinv(v)))) < 1e-14
        True
        >>> v = [9.3,1.5]
        >>> max(abs(v - fun.pinv(fun(v)))) < 1e-14
        True
        >>> max(abs(v - fun(fun.pinv(v)))) < 1e-14
        True
        >>> fun = mb1(array([0., 7.]) )
        >>> v = [0.,0.]
        >>> max(abs(v - fun.pinv(fun(v)))) < 1e-14
        True
        >>> max(abs(v - fun(fun.pinv(v)))) < 1e-14
        True
        >>> v = [0.3,0.5]
        >>> max(abs(v - fun.pinv(fun(v)))) < 1e-14
        True
        >>> max(abs(v - fun(fun.pinv(v)))) < 1e-14
        True
        >>> v = [9.3,1.5]
        >>> max(abs(v - fun.pinv(fun(v)))) < 1e-14
        True
        >>> max(abs(v - fun(fun.pinv(v)))) < 1e-14
        True

        >>> fun = mb1(array( [-0.55822362,  1.44172583]))
        >>> v = array([ 0.80376337,  0.57728775])
        >>> max(abs(v - fun.pinv(fun(v)))) < 1e-14
        True
        >>> max(abs(v - fun(fun.pinv(v)))) < 1e-14
        True
        >>> v = array([ 0.62349942,  0.02803776])
        >>> max(abs(v - fun.pinv(fun(v)))) < 1e-14
        True
        >>> max(abs(v - fun(fun.pinv(v)))) < 1e-14
        True
        """
        vc = v - self.centrum
        r = sqrt(dot(vc, vc))
        if r < 1e-12:
            phi = 0.
        else:
            phi = acos(vc[0] / r)
            if vc[1] < 0.:
                phi = 2. * pi - phi
        return array([r, phi])

#
# This one is here to make some tests work:
#
def mb2(par=0.5):
    return Affine([[1.0, 1.0 - par],
                   [0.0,       par]])

class mb3(Func):
    """
    3 -> 3
    r, theta, phi -> x, y, z
    spherical coordinates
    """
    def __init__(self, r0 = array([0., 1., 2.])):
        self.r0 = r0

    def taylor(self, vec):
        """
        >>> fun = mb3()
        >>> from pts.func import NumDiff
        >>> from numpy import max, abs, dot, pi
        >>> fun2 = NumDiff(mb3())
        >>> x1 = array([1.,1., 1.])
        >>> x2 = array([1., 0., 0.5])
        >>> x3 = array([0.01, 0.999 * pi, 0.0001 * pi ])

        >>> max(abs(fun.fprime(x1) - fun2.fprime(x1))) < 1e-11
        True
        >>> max(abs(fun.fprime(x2) - fun2.fprime(x2))) < 1e-12
        True
        >>> max(abs(fun.fprime(x3) - fun2.fprime(x3))) < 1e-12
        True
        """
        r, theta, phi = vec
        v = zeros(3)
        v[0] = self.r0[0] + r * cos(phi) * cos(theta)
        v[1] = self.r0[1] + r * sin(phi) * cos(theta)
        v[2] = self.r0[2] + r * sin(theta)
        dv = zeros((3,3))
        dv[0,0] = cos(phi) * cos(theta)
        dv[1,0] = sin(phi) * cos(theta)
        dv[2,0] = sin(theta)
        dv[0,1] = -r * cos(phi) * sin(theta)
        dv[1,1] = - r * sin(phi) * sin(theta)
        dv[2,1] = r * cos(theta)
        dv[0,2] = -r * sin(phi) * cos(theta)
        dv[1,2] = r * cos(phi) * cos(theta)
        dv[2,2] = 0.
        return v, dv

    def pinv(self, v):
        """
        >>> fun = mb3()
        >>> v = array([0.,0., 0.])
        >>> max(abs(v - fun.pinv(fun(v)))) < 1e-14
        True
        >>> max(abs(v - fun(fun.pinv(v)))) < 1e-14
        True
        >>> v = array([9.3,1.5, 0.5])
        >>> max(abs(v - fun.pinv(fun(v)))) < 1e-14
        True
        >>> max(abs(v - fun(fun.pinv(v)))) < 1e-13
        True
        >>> fun = mb3(array([0., 7., - 2.]) )
        >>> v = array([0.,0., 0.])
        >>> max(abs(v - fun.pinv(fun(v)))) < 1e-14
        True
        >>> max(abs(v - fun(fun.pinv(v)))) < 1e-14
        True
        >>> v = [0.3,0.5, 0.2]
        >>> max(abs(v - fun.pinv(fun(v)))) < 1e-14
        True
        >>> max(abs(v - fun(fun.pinv(v)))) < 1e-14
        True
        >>> v = [9.3,1.5, 0.7]
        >>> max(abs(v - fun.pinv(fun(v)))) < 1e-14
        True
        >>> max(abs(v - fun(fun.pinv(v)))) < 1e-14
        True
        """
        vc = v - self.r0
        r = sqrt(dot(vc, vc))
        if r < 1e-12:
            phi = 0.
            theta = 0.
        else:
            theta = asin(vc[2] / r)
            r_flat = r * cos(theta)
            phi = acos(vc[0] / r_flat)
            if vc[1] < 0:
                phi = 2 * pi - phi
        return array([r, theta, phi])

class mb4(Func):
    def __init__(self, r0 = array([0.,1.,2.])):
        self.fun = mb3(r0)
        self.last = r0[2]

    def taylor(self, vec):
        r, phi = vec
        theta = asin(-self.last / r)
        v1 = array((r, theta, phi))
        y, dy = self.fun.taylor(v1)
        x = y[:2]
        dx = zeros((2,2))
        dx[:,0] = dy[:2,0]
        dx[:,1] = dy[:2,2]
        assert abs(y[2]) < 1e-8
        return x, dx

    def pinv(self, v):
        vin = zeros(3)
        vin[:2] = v
        x = self.fun.pinv(vin)
        y = zeros(2)
        y[0] = x[0]
        y[1] = x[2]
        return y

class Elliptic(Func):
    def __init__(self, f0 = array([- 1.0, 0.]), f1 = array([ 1.0, 0.])):
        """
        Elliptic coordinates around the two foci f0 and f1
        (Use function for ellicptic coordinate around foci -1 and a, and
        then transform it in the right direction.)

        default is with foci (-1,0) and (1,0)
        >>> e = Elliptic()

        >>> from pts.func import NumDiff
        >>> num_e = NumDiff(Elliptic())

        Here we know what to expect: nu = pi/2 means along second axis
        >>> y = array([1., pi/2.])
        >>> x, dx = e.taylor(y)
        >>> x.round(3)
        array([ 0.   ,  1.175])

        mu = zero means that is along first axis
        >>> y = array([0., pi/3.])
        >>> x, dx = e.taylor(y)
        >>> x
        array([ 0.5,  0. ])

        One in the negative sphere
        >>> y = array([0.9, pi/3. * 4.])
        >>> x, dx = e.taylor(y)
        >>> x.round(3)
        array([-0.717, -0.889])

        Verfiy correctness of derivative
        >>> (abs(dx - num_e.fprime(y))).max() < 1e-10
        True

        Verify correctness of inverse function
        >>> y_2 = e.pinv(x)
        >>> max(abs(y - y_2)) < 1e-10
        True

        Repeat tests for some more points
        >>> y = array([0.2, 3./4. * pi])
        >>> (abs(y - e.pinv(e(y)))).max() < 1e-10
        True
        >>> (abs(e.fprime(y) - num_e.fprime(y))).max() < 1e-10
        True
        >>> y = array([4.2, pi])
        >>> (abs(y - e.pinv(e(y)))).max() < 1e-10
        True
        >>> (abs(e.fprime(y) - num_e.fprime(y))).max() < 1e-10
        True
        >>> y = array([0., 0.])
        >>> (abs(y - e.pinv(e(y)))).max() < 1e-10
        True
        >>> (abs(e.fprime(y) - num_e.fprime(y))).max() < 1e-10
        True
        >>> y = array([1., 0.])
        >>> (abs(y - e.pinv(e(y)))).max() < 1e-10
        True
        >>> (abs(e.fprime(y) - num_e.fprime(y))).max() < 1e-10
        True

        >>> x = array([-0.2, 3./4.])
        >>> (abs(x - e(e.pinv(x)))).max() < 1e-10
        True

        Now check if the shift of center and orientation works correct:
        >>> e = Elliptic(f0 = array([0.3, -7.2]), f1 = array([8.0, 8.0]))
        >>> num_e = NumDiff(e)

        Test for some points
        >>> y = array([0.2, 3./4. * pi])
        >>> (abs(y - e.pinv(e(y)))).max() < 1e-10
        True
        >>> (abs(e.fprime(y) - num_e.fprime(y))).max() < 1e-10
        True
        >>> y = array([4.2, pi])
        >>> (abs(y - e.pinv(e(y)))).max() < 1e-10
        True
        >>> (abs(e.fprime(y) - num_e.fprime(y))).max() < 1e-8
        True
        >>> y = array([0., 0.])
        >>> (abs(y - e.pinv(e(y)))).max() < 1e-7
        True
        >>> (abs(e.fprime(y) - num_e.fprime(y))).max() < 1e-10
        True
        >>> y = array([1., 0.])
        >>> (abs(y - e.pinv(e(y)))).max() < 1e-10
        True
        >>> (abs(e.fprime(y) - num_e.fprime(y))).max() < 1e-10
        True
        >>> x = array([-0.2, 3./4.])
        >>> (abs(x - e(e.pinv(x)))).max() < 1e-10
        True
        >>> x = array([0.3, -7.2])
        >>> (abs(x - e(e.pinv(x)))).max() < 1e-10
        True
        >>> x = array([8., 8.])
        >>> (abs(x - e(e.pinv(x)))).max() < 1e-10
        True
        """
        # first get the distances a from the center of the foci to the foci:
        self.__a = sqrt(dot(f0 - f1, f0 - f1)) / 2.
        # we use some inner coordinates with foci (-a, 0) and (a, 0) for
        # some inner symsten, global orientation will be done by
        # translation parallel to the one of the centrum:
        self.__centrum = (f0 + f1) / 2.

        # h is shifted foci f1, so that centrum is at (0,0)
        h = f1 - self.__centrum
        # this rotation matrix will bring h -> f1
        self.__rotmat = array([[ h[0] / self.__a , - h[1] / self.__a],
                             [ h[1] / self.__a, h[0] / self.__a ]])

        # check that the foci map correctly
        assert (abs(dot(self.__rotmat, array([self.__a,0])) + self.__centrum - f1)).max() < 1e-10
        assert (abs(dot(self.__rotmat, array([-self.__a,0])) + self.__centrum - f0)).max() < 1e-10

    def taylor(self, vec):
        mu, nu = vec
        # that would be the coordinates in inner ellictic coordinates:
        y = zeros(2)
        dy = zeros((2,2))
        y[0] = self.__a * cosh(mu) * cos(nu)
        y[1] = self.__a * sinh(mu) * sin(nu)

        # derivatives
        dy[0,0] = self.__a * cos(nu) * sinh(mu)
        dy[1,0] = self.__a * sin(nu) * cosh(mu)
        dy[0,1] = - self.__a * cosh(mu) * sin(nu)
        dy[1,1] = self.__a * sinh(mu) * cos(nu)

        # transform to real coordinates
        x = zeros(2)
        dx = zeros((2,2))
        x = dot(self.__rotmat, y) + self.__centrum
        dx = dot(self.__rotmat, dy)

        return x, dx

    def pinv(self, v):
        # first transform coordinates in inner elliptic coordinates
        # with (-a, 0) (a,0)
        v_in = dot(self.__rotmat.T, v - self.__centrum) / self.__a

        # m1 = cosh(mu), n1 = cos(nu)
        # solve for
        # v[0] = m1* n1
        # v[1]**2 = (m1**2 -1)(1 - n1**2)
        a = dot(v_in, v_in) + 1.
        m1 = sqrt(a /2. + sqrt(a**2 - 4 * v_in[0]**2) / 2.)

        # 1 <= m1, only rounding errors could say something else
        if abs(m1)  < 1.:
            mu = 0.
            m1 = 1.
        else:
            mu = arccosh(m1)

        n1 = v_in[0] / m1
        nu = acos(n1)

        # one has to consider that acos gives only values for the upper half
        if v_in[1] < 0. and abs(v_in[1]) > finfo(float).eps:
            nu = 2. * pi - nu

        return array([mu, nu])

# python func.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

