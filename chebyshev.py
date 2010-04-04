from numpy import pi, cos, array, asarray, all, transpose
from dct import dct
from func import Func

def chebft(fun, a=-1.0, b=1.0, n=8):
    """Build Chebyshev fit of degree |n| for funciton |fun| on the interval [a, b].
    This requires evaluation of fun(x) at |n| points in the [a, b] interval.

        >>> from numpy import sin, cos

        >>> def sc(t):
        ...    return array([sin(t), cos(t)])

        >>> sc(0.1)
        array([ 0.09983342,  0.99500417])

        >>> scf = chebft(sc, 0.0, pi / 12.0, 8)

        >>> scf(0.1)
        array([ 0.09983342,  0.99500417])

        >>> scf.fprime(0.1)
        array([ 0.99500417, -0.09983342])

    See Numerical Recepies or
    http://www.excamera.com/sphinx/article-chebyshev.html
    """

    grid = array([(k + 0.5) / n for k in range(n)])

    # polynomial roots:
    roots = cos(pi * grid)

    # adjust roots to [a, b] interval:
    bma = 0.5 * (b - a)
    bpa = 0.5 * (b + a)

    # function values at roots f_i(x_k) at [k, i]:
    fs = map(fun, roots * bma + bpa)

#       # FIXME: suboptimal, I guess:
#       fac = 2.0 / n
#       c = []
#       for j in range(n):
#           tj = cos(pi * j * grid)
#           c.append(fac * dot(fs, tj))
#       self.__c = array(c)

    # Use Discrete Cosine Transform instead,
    # DCT is performed over the last axis r as in fs[..., r]:
    fs = transpose(array(fs))
    c = dct(fs) / n
    # DCT returns c[..., k], with axis k being the "frequency" axis.
    c = transpose(c)
#   print "c=", c

    # c[0] differs by factor two:
    c[0] *= 0.5

    return Chebyshev(c, a, b)

class ChebT(object):
    """Expansion over Chebyshev polynomials of the first kind:
                n
               ___
               \
        p(x) = /__   c  *  T (x)
              j = 0   j     j

    """
    def __init__(self, c):
        self.__c = asarray(c)

    def __call__(self, x):
        return chebtev(x, self.__c)

class ChebU(object):
    """Expansion over Chebyshev polynomials of the second kind:
                n
               ___
               \
        q(x) = /__   c  *  U (x)
              j = 0   j     j

    """
    def __init__(self, c):
        self.__c = asarray(c)

    def __call__(self, x):
        return chebuev(x, self.__c)

class Chebyshev(Func):
    """Expansion over Chebyshev polynomials of the first kind:
                n
               ___
               \
        p(x) = /__   c  *  T (y)
              j = 0   j     j

    Where |y| is normalized into [-1, 1]:

            2x - (a + b)
        y = ------------
               b - a

        >>> p = Chebyshev([10., 3., 1.])

    that is
                            2
        p(x) == 10 + 3x + 2x - 1

    Test for several arguments:

        >>> xs = array([-1.0, 0.0, 1.0])
        >>> p(xs)
        array([  8.,   9.,  14.])

        >>> p.fprime(xs)
        array([-1.,  3.,  7.])

    Test for several expansions:

        >>> from numpy import empty
        >>> c = array([10., 3., 1.])

    Coeffs for two third order polynomials:

        >>> c32 = empty((3, 2))

    First polynomial:

        >>> c32[:, 0] = c

    Second polynomial:

        >>> c32[:, 1] = c * 100.0

        >>> p2 = Chebyshev(c32)

    Returns two values:

        >>> p2(0.0)
        array([   9.,  900.])

    Currently does not support additional array axes for polynomial
    argument and coefficients simultaneousely.

    See Numerical Recepies or
    http://www.excamera.com/sphinx/article-chebyshev.html
    """
    def __init__(self, c, a=-1.0, b=1.0):
        c = asarray(c)

        # interval [a, b]:
        self.__a = a
        self.__b = b

        self.__t = ChebT(c)

        # coeffs for derivative expansion (over second kind!):
        cprime = array([ k * ck for k, ck in enumerate(c) ])

        self.__u = ChebU(cprime[1:])

    def f(self, x):
        a, b = self.__a, self.__b
        assert all(a <= x) and all(x <= b)

        # normalize x into [-1, 1]:
        y = (2.0 * x - a - b) * (1.0 / (b - a))

        return self.__t(y)

    def fprime(self, x):
        a, b = self.__a, self.__b
        assert all(a <= x) and all(x <= b)

        # normalize x into [-1, 1]:
        y = (2.0 * x - a - b) * (1.0 / (b - a))

        return self.__u(y) * 2.0 / (b - a)

def clenshaw(x, a):
    """Compute expansions over Chebyshev polynomials of first-
    and second kinds:
                n
               ___
               \
        p(x) = /__   a  *  T (x),     q(x) = ... U (x)
              j = 0   j     j                     j

    and return both as a tuple

        (p(x), q(x))

    See http://en.wikipedia.org/wiki/Clenshaw_algorithm

    Test with a == [10, 3, 1] corresponding to

                            2
        p(x) == 10 + 3x + 2x - 1

    and
                            2
        q(x) == 10 + 6x + 4x - 1

        >>> a = [10., 3., 1.]

        >>> from scipy.special.orthogonal import chebyt, chebyu

        >>> p = a[0] * chebyt(0) + a[1] * chebyt(1) + a[2] * chebyt(2)
        >>> q = a[0] * chebyu(0) + a[1] * chebyu(1) + a[2] * chebyu(2)

        >>> p( 0.0), q( 0.0)
        (9.0, 9.0)
        >>> clenshaw( 0.0, a)
        (9.0, 9.0)

        >>> p( 1.0), q( 1.0)
        (14.0, 19.0)
        >>> clenshaw( 1.0, a)
        (14.0, 19.0)

        >>> p(-1.0), q(-1.0)
        (8.0, 7.0)
        >>> clenshaw(-1.0, a)
        (8.0, 7.0)

        >>> p(0.7), q(0.7)
        (12.08, 15.16)
        >>> clenshaw(0.7, a)
        (12.08, 15.16)

    Works with many sets of coeeficients:

        >>> from numpy import empty
        >>> a = array(a)

    Coefficients for two third order polynomials, first
    axis is the polynomial index, other (optional) axes
    for different expansions:

        >>> a32 = empty((3, 2))
        >>> a32[:, 0] = a
        >>> a32[:, 1] = a * 100.0

    You get two values for each polynomial kind:

        >>> clenshaw(1.0, a32)
        (array([   14.,  1400.]), array([   19.,  1900.]))

    Also works with array-valued argument:

        >>> xs = array([-1.0, 0.0, 1.0])

        >>> clenshaw(xs, a)
        (array([  8.,   9.,  14.]), array([  7.,   9.,  19.]))
    """

    twox = 2.0 * x

    (bj, b2) = (a[-1], 0.0)          # Special case first step for efficiency
    for aj in a[-2:0:-1]:            # Clenshaw's recurrence
        (bj, b2) = (twox * bj - b2 + aj, bj)

    # first kind:
    p = x * bj - b2 + a[0]

    # second kind:
    q = twox * bj - b2 + a[0]

    return p, q

def chebtev(x, a):
    return clenshaw(x, a)[0]

def chebuev(x, a):
    return clenshaw(x, a)[1]

# python chebyshev.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
