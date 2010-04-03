from numpy import pi, cos
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
        0.099833416646828876
        >>> sin(0.1)
        0.099833416646828155

    See Numerical Recepies or
    http://www.excamera.com/sphinx/article-chebyshev.html
    """
    def __init__(self, fun, a=-1.0, b=1.0, n=8):
        self.__a = a
        self.__b = b

        bma = 0.5 * (b - a)
        bpa = 0.5 * (b + a)
        fs = [fun(cos(pi * (k + 0.5) / n) * bma + bpa) for k in range(n)]
        fac = 2.0 / n
        self.__c = [fac * sum([fs[k] * cos(pi * j * (k + 0.5) / n) for k in range(n)]) for j in range(n)]

    def f(self, x):
        a, b = self.__a, self.__b
        assert(a <= x <= b)
        y = (2.0 * x - a - b) * (1.0 / (b - a))
        y2 = 2.0 * y
        (d, dd) = (self.__c[-1], 0)             # Special case first step for efficiency
        for cj in self.__c[-2:0:-1]:            # Clenshaw's recurrence
            (d, dd) = (y2 * d - dd + cj, d)
        return y * d - dd + 0.5 * self.__c[0]   # Last step is different


# python chebyshev.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
