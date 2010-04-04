from numpy import pi, cos, array, dot
from dct import dct
from func import Func

class Chebyshev(Func):
    """
    Chebyshev(fun, a, b, n)
    Given a function fun, lower and upper limits of the interval [a,b],
    and maximum degree n, this class computes a Chebyshev approximation
    of the function.

        >>> from numpy import sin
        >>> s = Chebyshev(sin, 0, pi / 12, 8)

        >>> s(0.1)
        0.099833416646828765

        >>> sin(0.1)
        0.099833416646828155

        >>> cos(0.1)
        0.99500416527802582

        >>> s.fprime(0.1)
        0.9950041652778987

        >>> from func import NumDiff
        >>> s1 = NumDiff(s)
        >>> s1.fprime(0.1)
        0.99500416527792834

    See Numerical Recepies or
    http://www.excamera.com/sphinx/article-chebyshev.html
    """
    def __init__(self, fun, a=-1.0, b=1.0, n=8):
        self.__a = a
        self.__b = b

        grid = array([(k + 0.5) / n for k in range(n)])

        # polynomial roots:
        roots = cos(pi * grid)

        # adjust roots to [a, b] interval:
        bma = 0.5 * (b - a)
        bpa = 0.5 * (b + a)

        # function values at roots:
        fs = map(fun, roots * bma + bpa)

#       # FIXME: suboptimal, I guess:
#       fac = 2.0 / n
#       c = []
#       for j in range(n):
#           tj = cos(pi * j * grid)
#           c.append(fac * dot(fs, tj))
#       self.__c = array(c)

        # Use Discrete Cosine Transform instead:
        c = dct(fs) / n

        # c[0] differs by factor two:
        c[0] *= 0.5

#       c = array([10., 3., 1.])

        # coeffs for derivative expansion (over second kind!):
        cprime = array([ k * ck for k, ck in enumerate(c) ])

#       print "c=", c
#       print "cprime=", cprime

        self.__c = c
        self.__cprime = cprime[1:]

    def f(self, x):
        a, b = self.__a, self.__b
        assert(a <= x <= b)

        # normalize x into [-1, 1]:
        y = (2.0 * x - a - b) * (1.0 / (b - a))

        p, _ = clenshaw(y, self.__c)
        return p
#       y2 = 2.0 * y
#       (d, dd) = (self.__c[-1], 0)             # Special case first step for efficiency
#       for cj in self.__c[-2:0:-1]:            # Clenshaw's recurrence
#           (d, dd) = (y2 * d - dd + cj, d)
#       return y * d - dd + self.__c[0]

    def fprime(self, x):
        a, b = self.__a, self.__b
        assert(a <= x <= b)

        # normalize x into [-1, 1]:
        y = (2.0 * x - a - b) * (1.0 / (b - a))

        _, q = clenshaw(y, self.__cprime)
        return q * 2.0 / (b - a)

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

# python chebyshev.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
