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

    >>> fn = "/tmp/TeMpOrArY.pickle"
    >>> if os.path.exists(fn): os.unlink(fn)

If you dont provide |filename|, cache will not be duplicated
in file, thus reload will not be possible:

    >>> s = Memoize(Func(si, co), filename=fn)

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

Delete the object and recreate from file:

    >>> del(s)
    >>> f = Memoize(Func(si, co), filename=fn)
    >>> f(a)
    array([ 0.        ,  0.70710678,  1.        ])

    >>> f(b)
    array([ 0.09983342,  0.19866933,  0.29552021])


    >>> f.fprime(a)
    array([  1.00000000e+00,   7.07106781e-01,   6.12323400e-17])

    >>> f.fprime(b)
    array([ 0.99500417,  0.98006658,  0.95533649])

    >>> os.unlink(fn)
"""

from __future__ import with_statement
from numpy import asarray
from func import Func
import os  # only os.path.exisist
import sys # only stderr
from pickle import dump, load

def tup(a):
    """Convert iterables to hashable tuples.

        >>> tup(1.)
        1.0

        >>> tup([1., 2., 3.])
        (1.0, 2.0, 3.0)

        >>> tup([[1., 2.], [3., 4.]])
        ((1.0, 2.0), (3.0, 4.0))

        >>> from numpy import array, all

        >>> a = array([[1., 2.], [3., 4.]])
        >>> b = tup(a)
        >>> all(a == array(b))
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
    def __init__(self, d=None):

        # initalize a new dictionary every time
        # (unless provided by caller):
        if d is None:
            d = {}
        self._d = d

    def __getitem__(self, key):
        # immutable key, convert arrays to tuples:
        k = tup(key)
        return self._d[k]

    def __setitem__(self, key, val):
        # immutable key, convert arrays to tuples:
        k = tup(key)
        self._d[k] = val

    def __contains__(self, key):
        # immutable key, convert arrays to tuples:
        k = tup(key)
        return (k in self._d)

class FileStore(Store):
    """Minimalistic disk-persistent dictionary.

        >>> fn = "/tmp/tEmP.pickle"
        >>> if os.path.exists(fn): os.unlink(fn)

        >>> d = FileStore(fn)
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

    Now delete this object and re-create from file:

        >>> del(d)
        >>> e = FileStore(fn)
        >>> [1., 2.] in e
        True
        >>> e[[1., 2.]]
        30.0
        >>> os.unlink(fn)
    """
    def __init__(self, filename="FileStore.pickle"):

        # FIXME: non-portable filename handling:
        if filename[0] != '/':
            # use absolute names, otherwise the storage will not be
            # found after chdir() e.g. in QContext() handler:
            filename = os.getcwd() + '/' + filename

        self.filename = filename

        try:
            # load dictionary from file:
            with open(filename,'r') as f:
                d = load(f) # pickle.load
            # warn by default, so that people dont forget to clean:
            print >> sys.stderr, "WARNING: FileStore found and loaded " + filename
        except:
            # empty dictionary:
            d = {}

        # parent class init:
        Store.__init__(self, d)

    def __setitem__(self, key, val):
        """Needs to update on-disk state."""
        Store.__setitem__(self, key, val)

        # dump the whole dictionary into file, FIXME: better solution?
        with open(self.filename,'w') as f:
            dump(self._d, f, protocol=2) # pickle.dump

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
    def __init__(self, func, filename=None):
        self.__f = func

        if filename is None:
            # cache in memory:
            self.__d = Store()
        else:
            # cache in memory, save to disk on updates:
            self.__d = FileStore(filename)

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

    def taylor(self, *args):
        # keys for the value and derivative:
        key0 = (args, 0)
        key1 = (args, 1)
        if key0 in self.__d and key1 in self.__d:
            return self.__d[key0], self.__d[key1]
        else:
            f, fprime = self.__f.taylor(*args)
            self.__d[key0] = f
            self.__d[key1] = fprime
            return f, fprime

# python memoize.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
