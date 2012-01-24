"""
Memoize Func()s

Two functions with side effects:

    >>> from numpy import sin, cos, pi, asarray

    >>> def si(x):
    ...     print "si(",x,")"
    ...     return sin(x)

    >>> def co(x):
    ...     print "co(",x,")"
    ...     return cos(x)

    >>> fn = "/tmp/TeMpOrArY.pickle"
    >>> if os.path.exists(fn): os.unlink(fn)

If you dont provide |filename|,  cache will not be duplicated in file,
thus reload will not be possible:

    >>> s = Memoize(Func(si, co), FileStore(fn))

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
    >>> f = Memoize(Func(si, co), FileStore(fn))
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
from copy import copy
from func import Func
import os # mkdir, chdir, getcwd, unlink, path, ...
import sys # only stderr
from pickle import dump, load
from pickle import dumps, loads
from numpy import dot
import hashlib

VERBOSE = 0

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

    which  is what  you really  wanted. However  the behaviour  of the
    latter may differ from

        elemental_memoize(f)

    in case elemental(g) is implemented in a "distributed" fashion and
    actuall calls to g = memoize(f) that cache the results happen on a
    remote  host or in  a separate  interpreter without  cache updates
    propagating back.
    """

    # for each call to memoize, create a new cache
    # (unless provided by caller):
    if cache is None:
        # Python dict() cannot be indexed by mutable keys,
        # use custom dictionary, see below:
        cache = Store()

    def _f(xs):

        # collect those to be computed:
        xs1 = [x for x in xs if x not in cache]

        # compute missing results:
        ys1 = f(xs1)

        # store new results:
        for x, y in zip(xs1, ys1):
            cache[x] = y

        # return copies from the dictionary:
        ys = [copy(cache[x]) for x in xs]

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
        len(a)
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

        # Use absolute names, otherwise  the storage will not be found
        # after chdir() e.g. in QContext() handler:
        if not os.path.isabs(filename):
            filename = os.path.abspath(filename)

        self.filename = filename

        try:
            # load dictionary from file:
            with open(filename,'r') as f:
                d = load(f) # pickle.load
            # warn by default, so that people dont forget to clean:
            print >> sys.stderr, "WARNING: FileStore found and loaded", filename
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

def hexhash(sdata):
    """
    Doctests are parsed twice, need double escape:

        >>> hexhash("blob 0\\0")
        'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391'

    This is the SHA-1 hash of the empty file in Git.
    """

    h = hashlib.sha1()
    h.update(sdata)
    return h.hexdigest()

def hextuple(key):
    """
        >>> hexhash("blob 0\\0")
        'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391'

        >>> hextuple("blob 0\\0")
        ('e6', '9de29bb2d1d6434b8b29ae775ad8c2e48c5391')
    """

    shex = hexhash(key)
    return shex[0:2], shex[2:]

def serialize(x):
    return dumps(x, protocol=2) # pickle.dumps

def deserialize(s):
    """
    >>> from numpy import array
    >>> deserialize(serialize(array([1., 2.])))
    array([ 1.,  2.])

    >>> hexhash(serialize(array([1., 2.])))
    'f2e54be8388eec2ef6c62f0f4a90d31dbaf25dd9'
    """
    return loads(s) # pickle.laodss

import errno

def maybe_mkdir(path):
    # FIXME: race condition here:
    try:
        os.mkdir(path)
    except OSError, e:
        # must exist already, all other reasons should remain fatal:
        assert e.errno == errno.EEXIST
        assert os.path.exists(path)
        pass

class DirStore(object):
    """
    Minimalistic  disk-persistent  dictionary.  To  enable  concurrent
    reads and writes each key value  pair is stored in a separate file
    named after SHA-1 hash of the key.

        >>> fn = "/tmp/tEmP.pickle.d"
        >>> if os.path.exists(fn): os.rmdir(fn)

        >>> d = DirStore(fn)

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
        >>> e = DirStore(fn)
        >>> [1., 2.] in e
        True
        >>> e[[1., 2.]]
        30.0

    Clean up:

        >>> del e[0.], e[1.], e[[1., 2.]]
        >>> del e
        >>> [os.rmdir(os.path.join(fn, sh)) for sh in ["1f", "dd", "e2"]]
        [None, None, None]
        >>> os.rmdir(fn)

    FIXME: does not handle collisions.

    FIXME: Not race free. Reading  of an incompletely written file and
           such.
    """

    def __init__(self, filename="DirStore.d"):

        # Use absolute names, otherwise  the storage will not be found
        # after chdir() e.g. in QContext() handler:
        if not os.path.isabs(filename):
            filename = os.path.abspath(filename)

        self.filename = filename

        if os.path.exists(filename):
            print >> sys.stderr, "WARNING: DirStore: found ", filename

    def __setitem__(self, key, val):
        """Needs to update on-disk state."""

        # FIXME: race condition here:
        if not os.path.exists(self.filename):
            maybe_mkdir(self.filename)
            if VERBOSE:
                print >> sys.stderr, "WARNING: DirStore: mkdir", self.filename

        #
        # Dump the  key-value pair into  a file named after  the SHA-1
        # hash of the key:
        #
        sh, ex = hextuple(serialize(key))

        if not os.path.exists(os.path.join(self.filename, sh)):
            # FIXME: race condition here:
            maybe_mkdir(os.path.join(self.filename, sh))

        with open(os.path.join(self.filename, sh, ex), 'w') as f:
            dump((key, val), f, protocol=2) # pickle.dump
            if VERBOSE:
                print >> sys.stderr, "WARNING: DirStore:", sh+ex, "written"

    def __getitem__(self, key):
        """Slurps the data from the file"""

        #
        # Get the  key-value pair  from a file  named after  the SHA-1
        # hash of the key:
        #
        sh, ex = hextuple(serialize(key))

        try:
            if VERBOSE:
                print >> sys.stderr, "WARNING: DirStore:", sh+ex, "trying"
            with open(os.path.join(self.filename, sh, ex), 'r') as f:
                key1, val = load(f) # pickle.load
                assert serialize(key) == serialize(key1) # FIXME: collision?
                if VERBOSE:
                    print >> sys.stderr, "WARNING: DirStore:", sh+ex, "loaded"
        except IOError:
            raise KeyError

        return val

    def __contains__(self, key):

        try:
            self[key]
            return True
        except KeyError:
            return False

    def __delitem__(self, key):
        """Deletes a an entry, unlinks the file"""

        sh, ex = hextuple(serialize(key))

        try:
            os.unlink(os.path.join(self.filename, sh, ex))
            if VERBOSE:
                print >> sys.stderr, "WARNING: DirStore:", sh+ex, "deleted"
        except IOError:
            raise IndexError

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
    def __init__(self, func, store=None):
        self.__f = func

        if store is None:
            # cache in memory:
            self.__d = Store()
        else:
            #
            # With  FileStore:  cache  in  memory,  save  to  disk  on
            # updates. With DirStore, cache on disk only.
            #
            self.__d = store
            # self.__d = FileStore(filename)

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

def convert(filename, dirname, kind):
    from numpy import array

    print "pts.memoize.convert(", filename, dirname, ")"
    fcache = FileStore(filename)
    dcache = DirStore(dirname)
    d = fcache._d

    if kind == "plain":
        for k, v in d.iteritems():
            # for a scalar function this should be sufficient ...
#           dcache[k] = v
            args, order = k
            tup, = args
            dcache[((array(tup),), order)] = v
    elif kind == "elemental":
        # ... this is only for the caches of elemental functions:

        for k, v in d.iteritems():
            # the key, k, is a tuple of *args and derivative order:
            args, order = k

            # args for simple function  is just a 1-tuple containg the
            # single  (array) argument.  The first  axis of  array and
            # that  of  the  corresponding  array  of  result  is  the
            # elemental index:
            arr, = args

            # so here y  = f(x), which we insert  into a DirStore each
            # as separate entry.  Again the  key should be a pair of a
            # 1-tuple and the order, cf. implementation of Memoize().
            for x, y in zip(arr, v):
                # print array(x), order, y
                dcache[((array(x),), order)] = y
    else:
        print >> sys.stderr, "Please say what kind of function is the source cache for!"
        print >> sys.stderr, "Choices are: plain, elemental"

class Empty_contex(object):
   """
   Context doing nothing actually,
   For the cases when there is no contex needed
   """
   def __init__(self, wd, format = None):
       self.wd = wd

   def __enter__(self):
       #print "Start Calculation, have id", self.wd
       pass

   def __exit__(self, exc_type, exc_val, exc_tb):
       assert (exc_type == None)
       #print "End Calculation with id", self.wd
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
    >>> cwd = os.getcwd()
    >>> os.chdir("/tmp/")

    For a single calculation:
    >>> with prep:
    ...     print os.getcwd()
    Starting Calculation in 02
    /tmp/02
    Finished Calculation in 02

    Clean up
    >>> from os import system
    >>> system("rmdir /tmp/%02d" % num)
    0
    >>> os.chdir(cwd)
    """
    def __init__(self, wdi, format = "%02d"):
       self.__wd = format % wdi

    def __enter__(self):
        self.__cwd = os.getcwd()
        if not os.path.exists(self.__wd):
            os.mkdir(self.__wd)
        os.chdir(self.__wd)
        print "Starting Calculation in", self.__wd

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert (exc_type == None)
        os.chdir(self.__cwd)
        print "Finished Calculation in", self.__wd
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
        ys1 = self.pmap(self.memfun, zip(xs1, wds))

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

        #
        # Return copies from the dictionary:
        #
        ys = [copy(self.cache[x]) for x in xs]

        #
        # Every "res"ult is a tuple (f, fprime)
        #
        es = [e for e, g in ys]
        gs = [g for e, g in ys]

        # FIXME: should we return two arrays?
        return es, gs

# python memoize.py [-v]:
if __name__ == "__main__":
    # convert(*sys.argv[1:])
    # exit()
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
