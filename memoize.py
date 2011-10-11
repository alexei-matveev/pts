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
    >>> if path.exists(fn): unlink(fn)

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

    >>> unlink(fn)
"""

from __future__ import with_statement
from numpy import asarray
from copy import copy
from func import Func
#import os  # only os.path.exisist
import sys # only stderr
from pickle import dump, load
from numpy import dot
from os import mkdir, chdir, getcwd, unlink, path

def memoize(f, cache=None):
    """Returns a memoized function f.

    Example:

        >>> def f(x):
        ...     print "f(", x, ")"
        ...     return [x**2]

        >>> f = memoize(f)

    First evaluation:

        >>> f(3)
        f( 3 )
        [9]

    Second evaluation:

        >>> f(3)
        [9]

    Destructive update:

        >>> y = f(3)
        >>> y[0] = -1

    See if the dictionary is intact:

        >>> f(3)
        [9]
    """

    # for each call to memoize, create a new cache
    # (unless provided by caller):
    if cache is None:
        # Python dict() cannot be indexed by mutable keys,
        # use custom dictionary, see below:
        cache = Store()

    def _f(x):
        if x not in cache:
            cache[x] = f(x)
        return copy(cache[x])

    return _f

def elemental_memoize(f, cache=None):
    """Returns a memoized elemental function f.

    Example:

        >>> def f(x):
        ...     print "f(", x, ")"
        ...     return x**2

    Make if elemental:

        >>> from func import elemental

        >>> f = elemental(f)
        >>> f([2, 3, 4])
        f( 2 )
        f( 3 )
        f( 4 )
        [4, 9, 16]

    Memoize an elemental function:

        >>> f = elemental_memoize(f)

    First evaluation:

        >>> f([2, 3, 4])
        f( 2 )
        f( 3 )
        f( 4 )
        [4, 9, 16]

    Second evaluation:

        >>> f([2, 3, 4])
        [4, 9, 16]

    Third evaluation, with one of the input entries changed:

        >>> f([2, 5, 4])
        f( 5 )
        [4, 25, 16]

    The last example illustrates the difference between

        memoize(elemental(f))

    (not quite what you wanted) and

        elemental(memoize(f))

    which is what you really wanted. However the behaviour
    of the latter may differ from

        elemental_memoize(f)

    in case elemental(g) is implemented in a "distributed" fashion
    and actuall calls to g = memoize(f) that cache the results
    happen on a remote host or in a separate interpreter without
    cache updates propagating back.
    """

    # for each call to memoize, create a new cache
    # (unless provided by caller):
    if cache is None:
        # Python dict() cannot be indexed by mutable keys,
        # use custom dictionary, see below:
        cache = Store()

    def _f(xs):

        # collect those to be computed:
        xs1 = []
        for x in xs:
            if x not in cache:
                xs1.append(x)

        # compute missing results:
        ys1 = f(xs1)

        # store new results:
        for x, y in zip(xs1, ys1):
            cache[x] = y

        # return copies from the dictionary:
        ys = []
        for x in xs:
            ys.append(copy(cache[x]))

        # FIXME: should we return an array?
        return ys

    return _f

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
        >>> if path.exists(fn): unlink(fn)

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
        >>> unlink(fn)
    """
    def __init__(self, filename="FileStore.pickle"):

        # FIXME: non-portable filename handling:
        if filename[0] != '/':
            # use absolute names, otherwise the storage will not be
            # found after chdir() e.g. in QContext() handler:
            filename = getcwd() + '/' + filename

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


class Empty_contex(object):
   """
   Context doing nothing actually,
   For the cases when there is no contex needed
   """
   def __init__(self, wd, format = None):
      pass

   def __enter__(self):
      pass

   def __exit__(self, exc_type, exc_val, exc_tb):
       assert (exc_type == None)
       return True

def global_distribution(xs, xlast_i ):
    """
    find out which of the new values to calculate fits
    to the ones from the last calculation

    >>> xl = [0,3,4,23,5]
    >>> xs = [0, 3.5, 22, 8]
    >>> global_distribution(xs, xl)
    [0, 1, 3, 4]
    >>> xs = [-5, 15, 3.5, 3.1, 6]
    >>> global_distribution(xs, xl)
    [0, 3, 1, 5, 4]

    Normally we expect the x to be vectors:
    >>> from numpy import array
    >>> xs = [array([0.,0., 1., 4.]), array([2.,0., 3., 4.]),
    ...       array([-0.,-0., -1., -4.]), array([20.,10., 15., 14.])]
    >>> xl = [array([0.1,0.1, 1.1, 4.1]) ,array([0.,0., 1., 4.]) ]
    >>> global_distribution(xs, xl)
    [1, 0, 2, 3]
    >>> global_distribution(xl, xs)
    [0, 4]
    """
    occupied = []
    new = len(xlast_i)
    if len(xlast_i) > 0:
        for x in xs:
            # find to wich of the last values this one is
            # the nearest
            dis = [dot(x-xl, x-xl) for xl in xlast_i]
            j = dis.index(min(dis))

            # if this one is already used, create a new one
            # else take it
            if j in occupied:
                occupied.append(new)
                new = new + 1
            else:
                occupied.append(j)
    else:
        # if there a not yet any old values
        occupied = range(len(xs))

    return occupied

class Single_contex(object):
    """
    Single context,
    changes in given directory

    >>> num = 2
    >>> prep = Single_contex(num)

    Change in temporear working directory:
    >>> cwd = getcwd()
    >>> chdir("/tmp/")

    For a single calculation:
    >>> with prep:
    ...     print getcwd()
    /tmp/02

    Clean up
    >>> from os import system
    >>> system("rmdir /tmp/%02d" % num)
    0
    >>> chdir(cwd)
    """
    def __init__(self, wdi, format = "%02d"):
       self.__wd = format % wdi

    def __enter__(self):
        self.__cwd = getcwd()
        if not path.exists(self.__wd):
            mkdir(self.__wd)
        chdir(self.__wd)

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert (exc_type == None)
        chdir(self.__cwd)
        return True

class Elemental_memoize(Func):
    """
    Memozise the f and fprime results (elemental) for function f
    Execution is on array for all the x -values

    Two functions with side effects:

        >>> from math import sin, cos, pi

        >>> def si(x):
        ...     print "si(",x,")"
        ...     return sin(x)

        >>> def co(x):
        ...     print "co(",x,")"
        ...     return cos(x)

        >>> s = Func(si, co)

        >>> s1 = Elemental_memoize(s, workhere = 0)

    First evaluation:

        >>> s1([0., 0.3 , pi/2.0])
        si( 0.0 )
        co( 0.0 )
        si( 0.3 )
        co( 0.3 )
        si( 1.57079632679 )
        co( 1.57079632679 )
        [0.0, 0.29552020666133955, 1.0]

    Second evaluation:

        >>> s1([ 0., pi/2.0])
        [0.0, 1.0]

        >>> s1([0., pi/8., 0.3])
        si( 0.392699081699 )
        co( 0.392699081699 )
        [0.0, 0.38268343236508978, 0.29552020666133955]

        >>> s1([0.0, 0.1, pi/8.+ 0.0001, pi/8., 0.3, 0.4])
        si( 0.1 )
        co( 0.1 )
        si( 0.392799081699 )
        co( 0.392799081699 )
        si( 0.4 )
        co( 0.4 )
        [0.0, 0.099833416646828155, 0.38277581840476976, 0.38268343236508978, 0.29552020666133955, 0.38941834230865052]

    Same for derivative:

        >>> s1.fprime([0., 0.3, pi/8.])
        [1.0, 0.95533648912560598, 0.92387953251128674]
    """
    def __init__(self, f, pmap = map, cache=None, workhere = 1, format = "%02d" ):
        # for each call to memoize, create a new cache
        # (unless provided by caller):
        if cache is None:
            # Python dict() cannot be indexed by mutable keys,
            # use custom dictionary, see below:
            self.cache = Store()
        else:
            self.cache = cache

        self.format = format
        self.workhere = workhere

        if self.workhere == 0:
            self.contex = Empty_contex
        else:
            self.contex = Single_contex

        self.last_xs_is = []

        def f_t_rem_i(z):
            # actual caclculation is only on x,
            # i is given also so that pmap can
            # steal it
            x, i = z
            contex = self.contex(i, format = self.format)
            with contex:
                return f.taylor(x)

        self.memfun = f_t_rem_i
        self.pmap = pmap

    def taylor(self, xs):
        # collect those to be computed:
        xs1 = []
        wds = []
        for i, x in enumerate(xs):
            if x not in self.cache:
                xs1.append(x)
                wds.append(i)

        if self.workhere == 1:
            wds = global_distribution(xs1, self.last_xs_is)
        # compute missing results:
        ys1 = self.pmap(self.memfun, zip(xs1,wds) )

        # store new results:
        for x, i, y in zip(xs1, wds, ys1):
            # store only with value of x
            self.cache[x] = y
            # store last values if global distribution should be used
            if self.workhere == 1:
                if i < len(self.last_xs_is):
                    self.last_xs_is[i] = x
                elif i == len(self.last_xs_is):
                    self.last_xs_is.append(x)
                else:
                    print >> sys.stderr, "ERROR: invalid number to calculate in"
                    exit()

        # return copies from the dictionary:
        ys = []
        for x in xs:
            ys.append(copy(self.cache[x]))

        # every ys is a tuple f, fprime
        es = []
        gs = []
        for res in ys:
            e1, g1 = res
            es.append(e1)
            gs.append(g1)


        # FIXME: should we return two arrays?
        return es, gs

# python memoize.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
