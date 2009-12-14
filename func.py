#!/usr/bin/python
"""
To run these tests, execute "python func.py".

Define some data points, x-values and y-values:

    >>> xs = (0., 0.25, 0.5, 0.75, 1.)
    >>> ys = (3., 2., 4., 5., 6.)

Construct the spline representaiton:

    >>> spl = SplineFunc(xs, ys)

Evaluate the spline at a few points:

    >>> spl(0.)
    3.0000000000000013

    >>> spl(1.)
    6.0

    >>> spl(0.5)
    4.0

    >>> spl(0.333)
    2.5898159280000002

"Calling" a Func is equivalent to invoking its value method |f|:

    >>> spl.f(0.333)
    2.5898159280000002

Evaluate derivative of a spline funciton at the same point:

    >>> spl.fprime(0.333)
    8.3266479999999987


A general function is constructed this way:

    >>> p = Func(lambda x: x**2, lambda x: 2*x)
    >>> p(2)
    4
    >>> q = Func(lambda x: x+1, lambda x: 1)
    >>> q(2)
    3

    >>> pq = compose(p, q)
    >>> pq.f(2), pq.fprime(2)
    (9, 6)

    >>> qp = compose(q, p)
    >>> qp.f(2), qp.fprime(2)
    (5, 4)

This is the integral of (x + 1) which is x^2/2 + x,
it also save the derivative of the integral:

    >>> Q = Integral(q)
    >>> Q(2.), Q.fprime(2.), q(2.)
    (4.0, 3.0, 3.0)

This builds the inverse of Q and also is able to
compute the derivative of Q using that of P:

    >>> P = Inverse(Q)
    >>> P(Q(2.)), Q(P(4.))
    (2.0, 4.0)

Derivatives of reciprocal functions are reciprocal:

    >>> P.fprime(4.) * Q.fprime(P(4.))
    1.0

    >>> Q.fprime(2.) * P.fprime(Q(2.))
    1.0

NumDiff() provides numerical differentiation for
the cases where implementation of |fprime| method
appears cumbersome:

    >>> spl2 = NumDiff(spl)
    >>> spl2(0.333) - spl(0.333)
    0.0
    >>> err = spl2.fprime(0.333) - spl.fprime(0.333)
    >>> abs(err) / spl.fprime(0.333) < 1e-12
    True

NumDiff() for multivariante functions, MB79 is a 2D PES:

    >>> from mueller_brown import MuellerBrown
    >>> mb1 = MuellerBrown()
    >>> mb2 = NumDiff(mb1)

The point to differentiate at:

    >>> p = array([0., 0.])

Analytical derivatives:

    >>> mb1.fprime(p)
    array([-120.44528524, -108.79148986])

Numerical  derivatives:

    >>> mb2.fprime(p)
    array([-120.44528524, -108.79148986])

NumDiff() for multivariate vector functions, say the gradient of MB79:

    >>> grad = NumDiff(mb1.fprime)

Hessian is the (numerical) derivative of the gradient:

    >>> hess = grad.fprime

This is one of the minima on MB79 PES:

    >>> b = array([ 0.62349942,  0.02803776])

The gradient is (almost) zero:

    >>> grad(b)
    array([  8.57060655e-06,   6.74209871e-06])

The Hessian must be positively defined:

    >>> hess(b)
    array([[  553.62608244,   154.92743643],
           [  154.92743643,  2995.60603248]])

For multivariate array-valued functions one needs to care about
indexing:

    >>> def f(x):
    ...     a = x[0,0]
    ...     b = x[0,1]
    ...     c = x[1,0]
    ...     d = x[1,1]
    ...     return array([[a+d, b-c],
    ...                   [c-b, d-a]])
    >>> f1 = NumDiff(f)
    >>> from numpy import zeros
    >>> df = f1.fprime(zeros((2,2)))

Derivatives wrt |a| == x[0,0]:

    >>> df[:,:,0,0]
    array([[ 1.,  0.],
           [ 0., -1.]])

Derivatives wrt |b| == x[0,1]:

    >>> df[:,:,0,1]
    array([[ 0.,  1.],
           [-1.,  0.]])

Derivatives wrt |c| == x[1,0]:

    >>> df[:,:,1,0]
    array([[ 0., -1.],
           [ 1.,  0.]])

Derivatives wrt |d| == x[1,1]:

    >>> df[:,:,1,1]
    array([[ 1.,  0.],
           [ 0.,  1.]])

"""

__all__ = ["Func", "LinFunc", "QuadFunc", "SplineFunc", "CubicFunc"]

from numpy import array, dot, hstack, linalg, atleast_1d
from numpy import empty
from scipy.interpolate import interp1d, splrep, splev
from scipy.integrate import quad
from scipy.optimize import newton
from ridders import dfridr

class Func(object):
    def __init__(self, f=None, fprime=None):
        self.f = f
        self.fprime = fprime

    # make our Funcs callable:
    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)
    # here you cannot do __call__ = f
    # if you want subclasses to specialize!

def compose(p, q):
    "Compose p*q, make p(x) = p(q(x))"

    def pq(x):
        return p.f(q.f(x))

    # note that calling pq.f and pq.fprime
    # will compute q.f(x) twice. If operating
    # with expensive functions without any caching
    # you may want to want to use the "tailor" interface instead.
    def pqprime(x):
        return dot( p.fprime(q.f(x)), q.fprime(x) )

    return Func(pq, pqprime)

class LinFunc(Func):
    def __init__(self, xs, ys):
        self.fs = interp1d(xs, ys)
        self.grad = (ys[1] - ys[0]) / (xs[1] - xs[0])

    def f(self, x):
        return self.fs(x) #[0]

    def fprime(self, x):
        return self.grad

class QuadFunc(Func):
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def f(self, x):
        x = atleast_1d(x).item()
        tmp = dot(array((x**2, x, 1)), self.coefficients)
        return tmp

    def fprime(self, x):
        x = atleast_1d(x).item()
        return 2 * self.coefficients[0] * x + self.coefficients[1]

class CubicFunc(Func):
    """
    >>> from numpy import round
    >>> c = CubicFunc(array([1,2,3,4]), array([0,0,0,1]))
    >>> round(c.coeffs*100)
    array([  17., -100.,  183., -100.])
    >>> c(1)
    0.0

    >>> c = CubicFunc(array([1,3]), array([0,0]), dydxs=array([1,1]))
    >>> round(c(1), 12)
    -0.0
    >>> round(c.fprime(1), 12)
    1.0

    >>> c = CubicFunc(array([0,1,2,3]), array([0,0,0,0]))
    >>> c.coeffs
    array([ 0.,  0.,  0.,  0.])

    >>> c = CubicFunc(array([0,1,2,3]), array([0,1,2,3]))
    >>> c.coeffs
    array([ 0.,  0.,  1.,  0.])

   """
    def __init__(self, xs, ys, dydxs=None):
            assert len(xs) == len(ys)
            assert dydxs == None or len(xs) == 2 and len(dydxs) == 2

            if dydxs != None:
                A = array([[3*xs[0]**2, 2*xs[0], 1, 0],
                           [3*xs[1]**2, 2*xs[1], 1, 0],
                           [xs[0]**3, xs[0]**2, xs[0], 1],
                           [xs[1]**3, xs[1]**2, xs[1], 1]])

                Ys = hstack([dydxs, ys])
                
                # calculate coefficients of cubic polynomial
                self.coeffs = linalg.solve(A,Ys)
            else:
                assert len(xs) == 4
                A = array([[xs[0]**3, xs[0]**2, xs[0], 1],
                           [xs[1]**3, xs[1]**2, xs[1], 1],
                           [xs[2]**3, xs[2]**2, xs[2], 1],
                           [xs[3]**3, xs[3]**2, xs[3], 1]])

                # calculate coefficients of cubic polynomial
                self.coeffs = linalg.solve(A,ys)

    def __str__(self):
        a,b,c,d = self.coeffs
        return "%.2fx^3 + %.2fx^2 + %.2fx + %.2f" % (a,b,c,d)

    def f(self, x):
        return dot(array((x**3, x**2, x, 1.)), self.coeffs)

    def fprime(self, x):
        return dot(array((3*x**2, 2*x, 1., 0.)), self.coeffs)
               

class SplineFunc(Func):
    def __init__(self, xs, ys):
        self.spline_data = splrep(xs, ys, s=0)

    def f(self, x):
        return splev(x, self.spline_data, der=0)

    def fprime(self, x):
        return splev(x, self.spline_data, der=1)


class Integral(Func):
    """This is slightly more than a quadrature, because it
    also saves and returns the exact derivative of the integral
    """

    def __init__(self, fprime, a=0.0, **kwargs):
        """Defines a Func() by integration of plain function fprime
        NOTE: optional |kwargs| are passed to scipy.integrate.quad() as is.
        """

        # we do not def fprime(self, x), is it a problem?
        self.fprime = fprime

        # integrate from this (to any x):
        self.x0 = a

        # these to be passed to |quad| as is:
        self.kwargs = kwargs

    def f(self, x, x0=None):
        """f(x) as an integral of |fprime| assuming f(x0) = 0
        The two argument version f(x, x0) returns the integral *from* x0 *to* x!
        """

        if x0 is None: x0 = self.x0

        (s, err) = quad(self.fprime, x0, x, **self.kwargs)
        assert abs(err) <= abs(s) * 1.0e7
        return s

class Inverse(Func):
    """For s = s(x) construct x = x(s)
    Inverse is constructed by solving s(x) = s by Newton method
    NOTE: not all functions are invertible, only monotonic ones,
    it is the user responsibility to ensure that s.fprime is either always
    greater than or always less than zero.
    """

    def __init__(self, s):
        """Just saves the forward functions s(x)"""

        # save the forward func:
        self.s = s

    def f(self, s):
        "x(s) as a solution of s(x) = s by Newton method, maybe not the most efficient way"

        # the difference between value of forward funciton for trial x
        # and the target value:
        def s0(x): return self.s(x) - s
        # def s1(x): return self.s.fprime(x)

        # solve equation sdiff(x) = 0, starting with trial x = 0:
        # FIXME: maybe approximate interpolation for initial x?
        # THIS DOES NOT WORK: (x, kws, err, msg) = scipy.optimize.fsolve(f, 0.0, fprime=sprime)
        x = newton(s0, 0.0, fprime=self.s.fprime)

        assert abs(s0(x)) <= 1.0e7
        return x

    def fprime(self, s):
        """Derivative of x(s) that is inverse of s(x).
        NOTE: dx/ds at s equals (ds/dx)^-1 at x = x(s)
        """

        # first get x = x(s):
        x = self.f(s) # this requires solving Newton method

        # return the reciprocal of the forward derivative at x:
        return 1. / self.s.fprime(x)


class NumDiff(Func):
    """Implements |.fprime| method by numerical differentiation"""

    # in case subclasses dont call our __init__:
    __h = 0.001

    def __init__(self, f=None, h=0.001):
        """Just saves the function f(x)"""

        # save the func, and default step:
        if f is not None:
            self.f = f
        # else: the subclass should implement it
        self.__h = h

    # either set in __init__ or implemented by subclass:
#   def f(self, x):
#       return self.__f(x)

    def fprime(self, x):
        """Numerical derivatives of self.f(x)"""

        if type(x) == type(1.0): # univariate function:
            dfdx, err = dfridr(self.f, x, h=self.__h)
            # print dfdx, err, err / prime
            assert err <= abs(dfdx) * 1.e-12
            return dfdx
        else: # maybe multivariate:
            # FIXME: cannot one unify the two branches?

            f = self.f # abbreviation

            xshape = x.shape
            xsize  = x.size

            # FIXME: this evaluation is only to get the type info of the result:
            fx = f(x)
            fshape = fx.shape
            fsize  = fx.size

            # print "ftype, xtype=", type(fx), type(x)
            # print "fshape, xshape =", fshape, xshape
            # print "fsize, xsize =", fsize, xsize

            # univariate function of |y| with parameter |n| staying for the
            # index of variable component:
            xwork  = x.copy() # will be used by |func|
            def func(y, n):
                # flatten:
                xwork.shape = (xsize,)

                # save component:
                old = xwork[n]

                # set component:
                xwork[n] = old + y

                # restore original shape:
                xwork.shape = xshape

                # feed into function to be differentiated:
                fx = f(xwork)

                # reshape and restore old component:
                xwork.shape = (xsize,)
                xwork[n] = old

                return fx

            # we take the convention that df_i/dx_k is at [i,k] position
            # of the array:
            dfdx = empty((fsize, xsize))

            # differentiate by each component:
            for n in range(xsize):
                fn = lambda y: func(y, n)
                dfdxn, err = dfridr(fn, 0.0, h=self.__h)
                # print dfdxn, type(dfdxn), dfdxn.shape

                # for assignment to dfdx[:,n] to succed, flatten if suitable:
                if fsize > 1: dfdxn.shape = (fsize,)
                dfdx[:, n] = dfdxn

            # set the proper shape of the result:
            dfdx.shape = fshape + xshape
            # print 'dfdx=', dfdx
            return dfdx


# python func.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
