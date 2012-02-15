#
# http://en.wikipedia.org/wiki/Rosenbrock_function
#

from numpy import array
from pts.func import Func

class Rosenbrock(Func):
    """
    >>> f = Rosenbrock()

    >>> f.taylor((1.0, 1.0))
    (0.0, array([ 0.,  0.]))

    >>> f.taylor((-1.2, 1.0))
    (24.199999999999996, array([-215.6,  -88. ]))
    """
    def __init__(self):
       pass

    def f(self, v):
       x, y = v
       return (1. - x)**2 + 100. * (y - x**2)**2

    def fprime(self, v):
       x, y = v
       fx = 2. * (x - 1.) - 400. * (y - x**2) * x
       fy =                 200. * (y - x**2)
       return array((fx, fy))

# python rosenbrock.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
