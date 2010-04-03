from numpy import pi, cos
from func import Func

class Chebyshev(Func):
    """
    Chebyshev(a, b, n, fun)
    Given a function fun, lower and upper limits of the interval [a,b],
    and maximum degree n, this class computes a Chebyshev approximation
    of the function.
    Method eval(x) yields the approximated function value.


        >>> from numpy import sin
        >>> s = Chebyshev(0, pi / 12, 8, sin)
        >>> s(0.1)
        0.099833416646828876
        >>> sin(0.1)
        0.099833416646828155

    See Numerical Recepies or
    http://www.excamera.com/sphinx/article-chebyshev.html
    """
    def __init__(self, a, b, n, fun):
        self.a = a
        self.b = b
        self.fun = fun

        bma = 0.5 * (b - a)
        bpa = 0.5 * (b + a)
        fs = [fun(cos(pi * (k + 0.5) / n) * bma + bpa) for k in range(n)]
        fac = 2.0 / n
        self.c = [fac * sum([fs[k] * cos(pi * j * (k + 0.5) / n) for k in range(n)]) for j in range(n)]

    def f(self, x):
        a,b = self.a, self.b
        assert(a <= x <= b)
        y = (2.0 * x - a - b) * (1.0 / (b - a))
        y2 = 2.0 * y
        (d, dd) = (self.c[-1], 0)             # Special case first step for efficiency
        for cj in self.c[-2:0:-1]:            # Clenshaw's recurrence
            (d, dd) = (y2 * d - dd + cj, d)
        return y * d - dd + 0.5 * self.c[0]   # Last step is different


# python chebyshev.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
