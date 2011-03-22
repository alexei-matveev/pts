from numpy import asarray, sqrt, array, zeros, dot, pi
from pts.func import Func
from math import cos, sin, acos, asin
from numpy import cosh, sinh, arccosh
from numpy import finfo

class diagsandhight(Func):
     """
     Function diagsandhight: generates Cartesian coordinates (and the derivatives)
     from given values for the two diagonals and the high between them.
     The First two atoms will be situated on the X-Axis, equal far away from O.
     The other two atoms will have the same on the Y-Axis, but they are shifted in z-direction
     about hight.

     >>> fun = diagsandhight()
     >>> from pts.func import NumDiff
     >>> from numpy import max, abs

     Verify the correctness of the analytical derivative:
     >>> fun2 = NumDiff(diagsandhight())
     >>> t = asarray([1., 1., 0.7])
     >>> max(abs(fun.fprime(t) - fun2.fprime(t))) < 1e-12
     True
     """
     def __init__(self):
         """
         The derivatives are always the same, thus store them
         once.
         """
         self.tmat = asarray([[[ 0.5,   0., 0.],
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


     def f(self, vec):
         """
         Function gets d1, d2 and h and calculates Cartesian coordinates
         """
         d1 = float(vec[0]) # diagonal with small changes
         d2 = float(vec[1]) # diagonal with large changes
         h = float(vec[2])  # hight of two last atoms in z-direction

         return asarray([[d1 / 2., 0., 0.], [-d1 / 2., 0., 0.], [0., d2 / 2., h], [0., -d2 / 2., h]])

     def fprime(self, vec):
         """
         The derivatives are constant
         """
         return self.tmat



class Elliptic(Func):
    def __init__(self, f0 = array([- 1.0, 0.]), f1 = array([ 1.0, 0.])):
        """
        Elliptic coordinates around the two foci f0 and f1
        (Use function for ellicptic coordinate around foci -1 and a, and
        then transform it in the right direction.)

        default is with foci (-1,0) and (1,0)
        >>> e = elliptic()

        >>> from pts.func import NumDiff
        >>> num_e = NumDiff(elliptic())

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
        >>> e = elliptic(f0 = array([0.3, -7.2]), f1 = array([8.0, 8.0]))
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

