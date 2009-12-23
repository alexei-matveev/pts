"""
Memoize Func()s

Two functions with side effects:

    >>> from numpy import sin, cos, pi

    >>> def si(x):
    ...     print "si(",x,")"
    ...     return sin(x)

    >>> def co(x):
    ...     print "co(",x,")"
    ...     return cos(x)

    >>> s = Memoize(Func(si, co))

    >>> a = asarray([0., pi/4., pi/2.])
    >>> b = asarray([0.1, 0.2, 0.3])

First evaluation:

    >>> s(a)
    si( [ 0.          0.78539816  1.57079633] )
    array([ 0.        ,  0.70710678,  1.        ])

    >>> s(b)
    si( [ 0.1  0.2  0.3] )
    array([ 0.09983342,  0.19866933,  0.29552021])


    >>> s.fprime(a)
    co( [ 0.          0.78539816  1.57079633] )
    array([  1.00000000e+00,   7.07106781e-01,   6.12323400e-17])

    >>> s.fprime(b)
    co( [ 0.1  0.2  0.3] )
    array([ 0.99500417,  0.98006658,  0.95533649])

Second evaluation:

    >>> s(a)
    array([ 0.        ,  0.70710678,  1.        ])

    >>> s(b)
    array([ 0.09983342,  0.19866933,  0.29552021])


    >>> s.fprime(a)
    array([  1.00000000e+00,   7.07106781e-01,   6.12323400e-17])

    >>> s.fprime(b)
    array([ 0.99500417,  0.98006658,  0.95533649])

"""

from numpy import asarray
from func import Func

def tup(a):
    """Convert iterables to hashable tuples.

        >>> tup(1.)
        1.0

        >>> tup([1., 2., 3.])
        (1.0, 2.0, 3.0)

        >>> tup([[1., 2.], [3., 4.]])
        ((1.0, 2.0), (3.0, 4.0))

        >>> a = asarray([[1., 2.], [3., 4.]])
        >>> b = tup(a)
        >>> (a == asarray(b)).all()
        True
    """
    try:
        n = len(a)
    except:
        # return scalar object as is:
        return a

    # convert iterables to (hashable) tuples:
    return tuple( tup(b) for b in a )

class Store(object):
    """Minimalistic dictionary. Accepts mutable arrays/lists as keys.
    The real keys are (nested) tuple representation of arrays/lists

        >>> d = Store()
        >>> d[0.] = 10.
        >>> d[1.] = 20.
        >>> d[0.]
        10.0
        >>> d[1.]
        20.0
        >>> d[[1., 2.]] = 30.
        >>> [1., 2.] in d
        True
        >>> d[[1., 2.]]
        30.0
    """
    def __init__(self, d={}):
        # empty dictionary:
        self.__d = d

    def __getitem__(self, key):
        # immutable key, convert arrays to tuples:
        k = tup(key)
        return self.__d[k]

    def __setitem__(self, key, val):
        # immutable key, convert arrays to tuples:
        k = tup(key)
        self.__d[k] = val

    def __contains__(self, key):
        # immutable key, convert arrays to tuples:
        k = tup(key)
        return (k in self.__d)

class Memoize(Func):
    """Memoize the .f and .fprime methods

    Two functions with side effects:

        >>> from math import sin, cos, pi

        >>> def si(x):
        ...     print "si(",x,")"
        ...     return sin(x)

        >>> def co(x):
        ...     print "co(",x,")"
        ...     return cos(x)

        >>> s = Func(si, co)
        >>> s(0.), s(pi/2.)
        si( 0.0 )
        si( 1.57079632679 )
        (0.0, 1.0)

        >>> s.fprime(0.)
        co( 0.0 )
        1.0

        >>> s1 = Memoize(s)

    First evaluation:

        >>> s1(0.), s1(pi/2.0)
        si( 0.0 )
        si( 1.57079632679 )
        (0.0, 1.0)

    Second evaluation:

        >>> s1(0.), s1(pi/2.0)
        (0.0, 1.0)

    Same for derivative:

        >>> s1.fprime(0.), s1.fprime(0.), s1.fprime(0.)
        co( 0.0 )
        (1.0, 1.0, 1.0)
    """
    def __init__(self, func):
        self.__f = func
        self.__d = Store()

    def f(self, *args):
        # key for the value:
        key = (args, 0)
        if key in self.__d:
            return self.__d[key]
        else:
            f = self.__f(*args)
            self.__d[key] = f
            return f

    def fprime(self, *args):
        # key for the derivative:
        key = (args, 1)
        if key in self.__d:
            return self.__d[key]
        else:
            fprime = self.__f.fprime(*args)
            self.__d[key] = fprime
            return fprime

# python memoize.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
