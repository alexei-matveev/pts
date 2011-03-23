#!/usr/bin/env python
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
    (9, array(6))

    >>> qp = compose(q, p)
    >>> qp.f(2), qp.fprime(2)
    (5, array(4))

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
indexing. This is a (2,2)-array valued function of (2,2)-array
argument:

    >>> def f(x):
    ...     a = x[0,0]
    ...     b = x[0,1]
    ...     c = x[1,0]
    ...     d = x[1,1]
    ...     return array([[a+d, b-c],
    ...                   [c-b, d-a]])
    >>> f1 = NumDiff(f)
    >>> from numpy import zeros
    >>> x = zeros((2, 2))
    >>> df = f1.fprime(x)

In most (all?) cases so far we decide to store the
derivatives of array-valued funcitons of array arguments
consistenly with this mnemonics:

    df / dx  is stored at array location [i, k]
      i    k

If any or both of |f| or |x| have more than one
axis consider |i| and |k| as composite indices.
Here is an example:

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

Sometimes it is convenient to view the derivatives
as rectangular matrix:

    >>> df.reshape((4, 4))
    array([[ 1.,  0.,  0.,  1.],
           [ 0.,  1., -1.,  0.],
           [ 0., -1.,  1.,  0.],
           [-1.,  0.,  0.,  1.]])

The funciton |f1| is a shape(2, 2) -> shape(2, 2) function, we can chain
it for testing f2(x) = f1(f1(x)

    >>> f2 = compose(f1, f1)

Although the point x = 0.0 is a fixpoint:

    >>> from numpy import all
    >>> all(x == f1(x)) and all(x == f2(x))
    True

The derivative matrix is a "square" of |df|:

    >>> f2.fprime(x).reshape((4, 4))
    array([[ 0.,  0.,  0.,  2.],
           [ 0.,  2., -2.,  0.],
           [ 0., -2.,  2.,  0.],
           [-2.,  0.,  0.,  0.]])
"""

__all__ = ["Func", "LinFunc", "QuadFunc", "SplineFunc", "CubicFunc"]

from numpy import array, dot, hstack, linalg, atleast_1d, sqrt, abs, column_stack, ones
from numpy import empty, asarray, searchsorted
from numpy import shape
from npz import matmul
from scipy.interpolate import interp1d, splrep, splev
from scipy.integrate import quad
from scipy.optimize import newton
from ridders import dfridr

class Func(object):
    def __init__(self, f=None, fprime=None, taylor=None):

        if f is not None:
            self.f = f

        if fprime is not None:
            self.fprime = fprime

        if taylor is not None:
            self.taylor = taylor

    # subclasses may choose to implement either (f, fprime) or just (taylor):
    def f(self, *args, **kwargs):
        return self.taylor(*args, **kwargs)[0]

    def fprime(self, *args, **kwargs):
        return self.taylor(*args, **kwargs)[1]

    # alternatively, subclasses may choose to implement just one:
    def taylor(self, *args, **kwargs):
        return self.f(*args, **kwargs), self.fprime(*args, **kwargs)

    # make our Funcs callable:
    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

def elemental(f, map=map):
    """A decorator for functions "f(x, ...)" that makes
    them elemental in the first argument. Other arguments
    are passed as is. By using a parallel map implementation
    one can achive parallelizm.

        >>> def f(x, a): return x * a
        >>> f(2, 10)
        20

        >>> f = elemental(f)

        >>> f([2, 3, 4], 10)
        [20, 30, 40]
    """

    def _f(xs, *args, **kwargs):

        def __f(x):
            return f(x, *args, **kwargs)

        return map(__f, xs)

    return _f

class Elemental(Func):
    """Make a Func elemental over the first argument by F = Elemental(f).
    Other arguments are passed as is. Provide a parallel map
    implementation if you want to parallelize independent evaluations.

    Example:

        >>> f = Func(lambda x: x**2, lambda x: 2 * x)
        >>> f(3), f.fprime(3)
        (9, 6)

    Make it elmental

        >>> F = Elemental(f)

    so that it can be now called with array-valued arguments:

        >>> xs = [3, 4, 5]

        >>> F(xs)
        [9, 16, 25]

        >>> F.fprime(xs)
        [6, 8, 10]

        >>> (F(xs), F.fprime(xs)) == F.taylor(xs)
        True
    """
    def __init__(self, f, map=map):
        self._args = f, map

    def f(self, xs, *args, **kwargs):
        F, map = self._args

        return map(lambda x: F(x, *args, **kwargs), xs)

    def fprime(self, xs, *args, **kwargs):
        F, map = self._args

        return map(lambda x: F.fprime(x, *args, **kwargs), xs)

    def taylor(self, xs, *args, **kwargs):
        F, map = self._args

        fgs = map(lambda x: F.taylor(x, *args, **kwargs), xs)

        fs = [f for f, g in fgs]
        gs = [g for f, g in fgs]

        return fs, gs

class RhoInterval(Func):
    """Supports generation of bead density functions for placement of beads 
    at specific positions.
    """
    def __init__(self, intervals):

        if intervals[0] != 0:
            intervals = [0] + intervals.tolist()
        self.intervals = array(intervals)
        self.N = len(intervals)

#        Func.__init__(f=self.f)

    def f(self, x):
        msg = 'Class RhoInterval only defined for x <- (%f,%f]' % (self.intervals[0], self.intervals[-1])
        if x < 0:
            raise ValueError(msg)

        for i in range(self.N)[1:]:
            if x <= self.intervals[i]:
                return 1.0 / (self.intervals[i] - self.intervals[i-1])
        msg += ' Supplied value was %f' % x
        raise ValueError(msg)

def compose(P, Q):
    "Compose P*Q, make P(x) = P(Q(x))"

    def f(x):
        return P(Q(x))

    # note that calling (P*Q).f and (P*Q).fprime
    # will compute Q.f(x) twice. If operating
    # with expensive functions without any caching
    # you may want to want to use the "taylor" interface instead.
    def fprime(x):
        q, qx = Q.taylor(x)
        p, pq = P.taylor(q)

        pshape = shape(p)
        qshape = shape(q)
        xshape = shape(x)

        px = matmul(pshape, xshape, qshape, pq, qx)
        return px

    def taylor(x):
        q, qx = Q.taylor(x)
        p, pq = P.taylor(q)

        pshape = shape(p)
        qshape = shape(q)
        xshape = shape(x)

        px = matmul(pshape, xshape, qshape, pq, qx)
        return p, px

    return Func(f, fprime, taylor)

class LinFunc(Func):
    def __init__(self, xs, ys):
        self.fs = interp1d(xs, ys)
        self.grad = (ys[1] - ys[0]) / (xs[1] - xs[0])

    def f(self, x):
        return self.fs(x) #[0]

    def fprime(self, x):
        return self.grad

class QuadFunc(Func):
    def __init__(self, xs, ys):
        self.coeffs = self.calc_coeffs(xs,ys)

    def calc_coeffs(self, xs, ys):
        assert len(xs) == len(ys) == 3
        xs_x_pow_2 = xs**2
        xs_x_pow_1 = xs 

        A = column_stack((xs_x_pow_2, xs_x_pow_1, ones(3)))

        quadratic_coeffs = linalg.solve(A,ys)

        return quadratic_coeffs

    def f(self, x):
        x = atleast_1d(x).item()
        tmp = dot(array((x**2, x, 1)), self.coeffs)
        return tmp

    def fprime(self, x):
        x = atleast_1d(x).item()
        return 2 * self.coeffs[0] * x + self.coeffs[1]

    def stat_points(self):
        """Returns the locations of the stationary points."""
        lin_coeffs = array((2, 1.)) * self.coeffs[:2]

        m,b = lin_coeffs

        return [-b / m]

    def fprimeprime(self, x):
        return 2 * self.coeffs[0]

    def __str__(self):
        a,b,c = self.coeffs
        return "%.4e*x**2 + %.4e*x + %.4e" % (a,b,c)


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

    >>> c = CubicFunc(array([0,1,2,3]), array([0,1,2,3]))
    >>> c.coeffs
    array([ 0.,  0.,  1.,  0.])

    Set coefficients explicitly (for testing only):

    >>> c.coeffs = array([0., 1./2., 1., 0.])
    >>> c.stat_points()
    [-1.0]

    >>> c.coeffs = array([1./3., 1./2., 0., 0.])
    >>> c.stat_points()
    [-0.0, -1.0]

    >>> c.coeffs = array([1./3., 1./2., 1e-9, 0.])
    >>> c.stat_points()
    [-1.0000000010000002e-09, -0.99999999899999992]

    >>> c.coeffs = array([-1./3., -1./2., -1e-9, 0.])
    >>> c.stat_points()
    [-0.99999999899999992, -1.0000000010000002e-09]

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
        return "%.4e*x**3 + %.4e*x**2 + %.4e*x + %.4e" % (a,b,c,d)

    def f(self, x):
        return dot(array((x**3, x**2, x, 1.)), self.coeffs)

    def fprime(self, x):
        return dot(array((3*x**2, 2*x, 1., 0.)), self.coeffs)

    def fprimeprime(self, x):
        return dot(array((6*x, 2, 0., 0.)), self.coeffs)

    def stat_points(self):
        """Returns the locations of the stationary points."""

        # derivative coefficients of third order polynomial:
        a, b, c = array((3., 2., 1.)) * self.coeffs[:3]

        # avoid division by zero, treat linear case:
        if a == 0.0: return [-c/b]

        delta = b**2 - 4*a*c

        if delta < 0:
            return []
        elif delta == 0:
            return [-b / 2 / a]
        elif b < 0: # sign of b decides which of the two results to change
            #
            # Also avoids precision loss for sqrt(BIG**2 + small) - BIG
            #
            return [(-b + delta**0.5) / 2 / a, 2 * c / (-b + delta**0.5)]
        else:
            return [2 * c / (-b - delta**0.5), (-b - delta**0.5) / 2 / a]


class SplineFunc(Func):
    def __init__(self, xs, ys):
        self.spline_data = splrep(xs, ys, s=0)

    def f(self, x):
        return splev(x, self.spline_data, der=0)

    def fprime(self, x):
        return splev(x, self.spline_data, der=1)

def casteljau(t, ps):
    """de Casteljau's algorythm for evaluating Bernstein forms,
    e.g. for the third order:

                 3               2          2              3
    C (t) = (1-t) * P  +  3t(1-t) * P  +  3t(1-t) * P  +  t * P
     3               0               1               2         3

    where [P0, P1, P2, P3] are taken from ps[:]

        >>> casteljau(0., [1.])
        1.0
        >>> casteljau(1., [1.])
        1.0
        >>> casteljau(0.5, [1.])
        1.0
        >>> casteljau(0.5, [2.])
        2.0
        >>> casteljau(0.5, [10., 5.])
        7.5

    Should also work for array valued (e.g. 2D) points:

        >>> casteljau(0.5, [array([10., 3.]), array([5., 10.]), array([3., 5.])])
        array([ 5.75,  7.  ])

    Compare with this:

        >>> casteljau(0.5, [10., 5., 3.])
        5.75
        >>> casteljau(0.5, [3., 10., 5.])
        7.0
    """

    # polynomial order:
    n = len(ps) - 1

    # a copy of ps, will be modifying this:
    bs = asarray(ps).copy() # \beta^0

    for j in xrange(1, n + 1):
        # \beta^j, j=1..n:
        bs[:n-j+1] = bs[:n-j+1] * (1 - t) + bs[1:n-j+2] * t

    return bs[0] # \beta^{n}_0

class CubicSpline(Func):
    """Piecwise fitting with cubic polynomials.

    By s = CubicSpline(xs, ys, yprimes) one constructs a Func()
    with cubic interpolation in each interval.

    NOTE: the second derivative of this spline is NOT continous.

        >>> s = CubicSpline([0.0, 0.3, 0.8], [1., 0., -1.], [2., 2., 3.])
        >>> s(0.0)
        1.0
        >>> s(0.3)
        0.0
        >>> s(0.8)
        -1.0
        >>> round(s.fprime(0.), 12)
        2.0
        >>> s.fprime(0.3)
        2.0
        >>> s.fprime(0.8)
        3.0

    Compare analytical and numerical derivatives:

        >>> abs(s.fprime(0.5)) > 1
        True
        >>> s1 = NumDiff(s)
        >>> abs(s.fprime(0.5) - s1.fprime(0.5)) < 1.e-12
        True

    For single interval the same behaviour as CubicFunc():

        >>> c = CubicSpline([1., 3.], [0., 0.], [1., 1.])
        >>> round(c(1.), 12)
        0.0
        >>> c(3.)
        0.0
        >>> c.fprime(1.)
        1.0

    Should also work for nD interpolation, here 2 points in 2D:

        >>> r = CubicSpline([1., 3.], [(10., 3.), (5., 10.)], [(3., 5.), (7., 9.)])

        >>> r(1.)
        array([ 10.,   3.])

        >>> r(3.)
        array([  5.,  10.])

        >>> r.fprime(1.)
        array([ 3.,  5.])

        >>> r.fprime(3.)
        array([ 7.,  9.])
    """
    def __init__(self, xs, ys, yprimes):
        assert len(xs) == len(ys)
        assert len(xs) == len(yprimes)

        xs = asarray(xs)
        ys = asarray(ys)
        yprimes = asarray(yprimes)

        self.xs = xs

        # number of intervals:
        m = len(xs) - 1

        # precompute four Bezier points for each interval:
        ps = [] # empty((m, 4))
        for i in range(m):
            dx = xs[i+1] - xs[i]
            p4 = [ ys[i]
                 , ys[i]   + yprimes[i]   / 3.0 * dx
                 , ys[i+1] - yprimes[i+1] / 3.0 * dx
                 , ys[i+1] ]
            ps.append(p4)
        self.ps = ps

        # precompute three Bezier points for derivative at each interval:
        dps = [] # empty((m, 3))
        for i in range(m):
            # 3 is the order of differentiated Bernstein polynomials:
            p3 = [ 3.0 * (ps[i][1] - ps[i][0])
                 , 3.0 * (ps[i][2] - ps[i][1])
                 , 3.0 * (ps[i][3] - ps[i][2]) ]
            dps.append(p3)
        self.dps = dps

    def f(self, x):
        """Uses search in sorted array.
        This tells that 1.5 would need to be inserted at position 2:
            >>> searchsorted([0., 1., 2.], 1.5)
            2
        """
        xs = self.xs # abbr
        i = searchsorted(xs, x)

        # out of range:
        if i == 0: i = 1
            # x = xs[0]

        if i == len(xs): i = len(xs) - 1
            # x = xs[-1]

        # normalized coordinate within the interval, t in (0,1)
        dx = xs[i] - xs[i-1]
        t = (x - xs[i-1]) / dx
        return casteljau(t, self.ps[i-1])

    def fprime(self, x):
        xs = self.xs # abbr
        i = searchsorted(xs, x)

        # out of range:
        if i == 0: i = 1
            # x = xs[0]

        if i == len(xs): i = len(xs) - 1
            # x = xs[-1]

        # normalized coordinate within the interval, t in (0,1)
        dx = xs[i] - xs[i-1]
        t = (x - xs[i-1]) / dx
        return casteljau(t, self.dps[i-1]) / dx

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

        # dict cache for computed integral values:
        self.__fs = {}

    def f(self, x):
        """f(x) as an integral of |fprime| assuming f(x0) = 0
        """

        # alias:
        fs = self.__fs

        # return cached value, if possible:
        if x in fs:
            return fs[x]

        # alias:
        x0 = self.x0

        (s, err) = quad(self.fprime, x0, x, **self.kwargs)
        assert abs(err) <= abs(s) * 1.0e-3, "%f > %f" % (abs(err), abs(s) * 1.0e-7) # was 1e7, then 1e-7

        # save computed value in cache:
        fs[x] = s

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

            # convert argument to array if necessary:
            x = asarray(x)

            xshape = x.shape
            xsize  = x.size

            # FIXME: this evaluation is only to get the type info of the result:
            fx = f(x)

            fx = asarray(fx)
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

class Reshape(Func):
    """
    """
    def __init__(self, f, xshape, fshape=None):
        self._args = f, xshape, fshape

    def f(self, x, *args, **kwargs):
        F, xshape, fshape = self._args

        ishape = shape(x)
        x.shape = xshape
        # FIXME: make a copy() here if other references to x
        # which are also temprarily reshaped affect the behaviour of F:
        fx =  F(x, *args, **kwargs)
        x.shape = ishape

        # shape the results:
        if fshape is not None:
            fx.shape = fshape

        return fx

    #
    # fprime() is a fallback to taylor()
    #

    def taylor(self, x, *args, **kwargs):
        F, xshape, fshape = self._args

        ishape = shape(x)
        x.shape = xshape
        # FIXME: make a copy() here if other references to x
        # which are also temprarily reshaped affect the behaviour of F:
        fx, fxprime = F.taylor(x, *args, **kwargs)
        x.shape = ishape

        # shape the results:
        if fshape is not None:
            fx.shape = fshape
            fxprime.shape = fshape + ishape
        else:
            fxprime.shape = shape(fx) + ishape

        return fx, fxprime

class Partial(Func):
    """From a multivariate function

        f(a0, a1, a2, ...)

    make a univariate function

        f(aN)

    by fixing all other arguments.
    """
    def __init__(self, f, n=0, *args):
        self.__f = f
        self.__n = n
        self.__args = args # FIXME: tuple has no .copy()

    def f(self, x):
        n = self.__n
        args = self.__args

        args = args[:n] + (x,) + args[n:]

        return self.__f.f(*args)

    def fprime(self, x):
        n = self.__n
        args = self.__args

        args = args[:n] + (x,) + args[n:]

        fp =  self.__f.fprime(*args)

        return fp[n]

    def taylor(self, x):
        n = self.__n
        args = self.__args

        args = args[:n] + (x,) + args[n:]

        f, fp =  self.__f.taylor(*args)

        return f, fp[n]

# python func.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
