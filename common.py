"""Classes, functions and variables common to all modules."""

import copy
import os
import sys
import random
import time
import logging

import numpy
from numpy import finfo
from pts.metric import Default

import pts

lg = logging.getLogger("pts.common")

PROGNAME = "searcher"
ERROR_STR = "error"

LOGFILE_EXT = ".log"
INPICKLE_EXT = ".in.pickle"
OUTPICKLE_EXT = ".out.pickle"

TMP_DIR_ENV_VAR = "AOF_TMP"
def get_tmp_dir():
    """Returns the absolute path to a temporary directory. If the environment
    variable AOE_TMP is specified (can be relative or absolute), then this is
    used. Otherwise the current working directory is used."""
    if TMP_DIR_ENV_VAR in os.environ:
        tmp_dir = os.path.abspath(os.environ[TMP_DIR_ENV_VAR])
    else:
        tmp_dir = os.path.abspath(os.getcwd())

    if not os.path.exists(tmp_dir):
        print "Attempting to create", tmp_dir
        try:
            os.mkdir(tmp_dir)
        except OSError, err:
            print >>sys.stderr, "Directory %s already created by another thread." % tmp_dir

    return tmp_dir

def exec_in_path(name):
    """Tests if executable called name is in the path already."""
    # TODO: is there a better way of doing this?
    return os.system("which " + name + "> /dev/null") == 0

# Max no. of allowed geometries given to the optimiser to form the initial guess 
# for the reaction pathway. Includes the reactant/product.
MAX_GEOMS = 3

# default max iterations for optimiser
DEFAULT_MAX_ITERATIONS = 20
DEFAULT_FORCE_TOLERANCE = 0.05

# unit conversions
ANGSTROMS_TO_BOHRS = 1.8897 # bohr / A
HARTREE_TO_ELECTRON_VOLTS = 27.2113845
DEG_TO_RAD = numpy.pi / 180.
RAD_TO_DEG = 180. / numpy.pi

# unit vectors
VX = numpy.array((1.0,0.0,0.0))
VY = numpy.array((0.0,1.0,0.0))
VZ = numpy.array((0.0,0.0,1.0))

def wt():
    raw_input("Wait...\n")


class Result():
    def __init__(self, v, energy, gradient = None, dir=None, flags = dict()):
        #FIXME: can I get rid of flags?
        self.v = v
        self.e = energy
        self.g = gradient
        self.flags = flags
        self.dir = dir

    def __eq__(self, r):
        return (isinstance(r, self.__class__) and is_same_v(r.v, self.v)) or (r != None and is_same_v(r, self.v))

    def __repr__(self):
        s = self.__class__.__name__ + "( " + str(self.v) 
        s += ", " + str(self.e) + ", " + str(self.g)
        s += ", " + str(self.flags) + ")"
        return s

    def type(self):
        s = ''
        if self.e != None:
            s += 'E'
        if self.e != None:
            s += 'G'
        return s

    def has_field(self, type):
        return type == 'G' and self.g != None \
            or type == 'E' and self.e != None
        
    def merge(self, res):
        assert is_same_v(self.v, res.v)
        assert is_same_e(self.e, res.e)

        if self.g == None:
            self.g = res.g
        else:
            lg.error("self.g = %s res = %s" % (self.g, res))
            raise ResultException("Trying to add a gradient result when one already exists")

def vec_summarise(v):
    return str(round(rms(v),4))

class Job(object):
    """Specifies calculations to perform on a particular geometry v.

    The object was designed with flexibility to include extra parameters and 
    multiple styles of computation, e.g. frequency calcs, different starting
    wavefunctions, different SCF convergence parameters, etc.
    
    num_bead was included for the creation of working directories per
    bead                        _____AN
    """
    def __init__(self, v, l, bead_ix=None, prev_calc_dir=None):
        self.v = v
        if not isinstance(l, list):
            l = [l]
        self.calc_list = l
        self.prev_calc_dir = prev_calc_dir
        self.num_bead = bead_ix
    
    def __str__(self):
        s = ""
        for j in self.calc_list:
            s += j.__str__()
        s =  ' '.join([self.__class__.__name__, vec_summarise(self.v), s])
        return s

    def geom_is(self, v_):
        assert len(v_) == len(self.v)
        return is_same_v(v_, self.v)

    def add_calc(self, calc):
        if self.calc_list.count(calc) == 0:
            self.calc_list.append(calc)

    def is_energy(self):
        return self.calc_list.count('E') > 0

    def is_gradient(self):
        return self.calc_list.count('G') > 0

    """class E():
        def __eq__(self, x):
            return isinstance(x, self.__class__) and self.__dict__ == x.__dict__
        def __str__(self):  return "E"

    class G():
        def __eq__(self, x):
            return isinstance(x, self.__class__) and self.__dict__ == x.__dict__
        def __str__(self):  return "G"
    """

def fname():
    import sys
    return sys._getframe(1).f_code.co_name

SAMENESS_THRESH_VECTORS = float(finfo(float).eps)
SAMENESS_THRESH_ENERGIES = 1e-10
def is_same_v(v1, v2):
    return numpy.linalg.norm(v1 - v2, ord=numpy.inf) < SAMENESS_THRESH_VECTORS
def is_same_e(e1, e2):
    return abs(e1 - e2) < SAMENESS_THRESH_ENERGIES

def line():
    return "=" * 80

def opt_gd(f, x0, fprime, callback = lambda x: None):
    """A gradient descent optimiser."""

    i = 0
    x = copy.deepcopy(x0)
    prevx = numpy.zeros(len(x))
    while 1:
        g = fprime(x)
        dx = x - prevx
        if numpy.linalg.norm(g, ord=2) < 0.05:
            print "***CONVERGED after %d iterations" % i
            break

        i += 1
        prevx = x
        if callback != None:
            x = callback(x)
        x -= g * 0.2

        # DON't DELETE
        if False:
            print line()
            print "x = ", x
            print "g =", g
            print line()

    x = callback(x)
    return x

# tests whether all items in a list are equal or not
def all_equal(l):
    if len(l) <= 1:
        return True
    if l[0] != l[1]:
        return False
    return all_equal(l[1:])

def vecmaxs(v, n=3):
    """Prints the largest n elements of a vector or matrix."""

    maxs = numpy.ones(n) * numpy.finfo(numpy.float64).min
    for i in copy.deepcopy(v).flatten():
        diffs = maxs - i

#        print "d",diffs
#        print "m", maxs
#        print "i",i
#        print "--"
        smallest = diffs.min()
        if i > smallest:
            for k in range(n):
                if diffs[k] == smallest:
                    maxs[k] = i
                    break
    return maxs

def vector_angle(v1, v2):
    """Returns the angle between two head to tail vectors in degrees."""
    fraction = numpy.dot(v1, v2) / numpy.linalg.norm(v1) / numpy.linalg.norm(v2)


    # Occasionally, due to precision errors, 'fraction' is slightly above 1 and
    # arccos(x) where x > 1 gives NaN.
    if fraction > 0.9999999999999:
        return 180.0

    result = 180. - RAD_TO_DEG * numpy.arccos(fraction)
    return result

def expand_newline(s):
    """Removes all 'slash' followed by 'n' characters and replaces with new line chaaracters."""
    s2 = ""
    i = 0
    while i < len(s)-2:
        if s[i:i+2] == r"\n":
            s2 += "\n"
            i += 2
        else:
            s2 += s[i]
            i += 1
    if s[-2:] == r"\n":
        s2 += "\n"
    else:
        s2 += s[-2:]

    return s2

def file2str(f):
    """Returns contents file with name f as a string."""
    f = open(f, "r")
    mystr = f.read()
    f.close()
    return mystr

def file2carts(f):
    s = file2str(f)
    if pts.coord_sys.XYZ.matches(s):
        return pts.coord_sys.XYZ(s).get_cartesians()
    elif pts.coord_sys.ZMatrix.matches(s):
        return pts.coord_sys.ZMatrix(s).get_cartesians()
    else:
        raise False, "Unrecognised file format in:\n" + s

def str2file(s, fn):
    """Returns contents file with name f as a string."""
    f = open(fn, "w")
    f.write(repr(s))
    f.close()
    lg.info("Writing " + fn)

def normalise(x):
    # used for zmatrix (and a test) thus no metric consideration
    x = x / numpy.linalg.norm(x)
    return x

def rms(x):
    return numpy.sqrt(numpy.mean(numpy.array(x).flatten()**2))

#### Exceptions ####
class ResultException(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self, msg):
        return self.msg
   
class QCDriverException(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self, msg):
        return self.msg

class ParseError(Exception):
    def __init__(self, msg):
        self.msg = "Parse Error: " + msg
    def __str__(self):
        return self.msg

def make_like_atoms(x):
    """Convert a vector to one with a shape reflecting cartesian coordinates,
    i.e. with a shape of (-1,3), padding with zeros if necessary.

    >>> from numpy import arange
    >>> make_like_atoms(arange(6))
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.]])
    >>> make_like_atoms(arange(7))
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  0.,  0.]])
    >>> make_like_atoms(arange(8))
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7.,  0.]])
    >>> make_like_atoms(arange(9))
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7.,  8.]])
    >>> make_like_atoms(arange(10))
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7.,  8.],
           [ 9.,  0.,  0.]])

    """
    x_ = x.copy().reshape(-1,)
    extras = len(x_) % 3
    if extras != 0:
        padding = numpy.zeros(3 - extras)
    else:
        # coerces type to be that of a numpy.zeros object
        padding = numpy.zeros(0)
    x_ = numpy.hstack([x_, padding])
    x_.shape = (-1,3)
    return x_

def place_str_dplace(tag):
    if tag == None:
        return "dplace"
    s = "dplace -c " + ','.join([str(cpu) for cpu in tag[0]])
    return s

def important(s):
    """Draws a box round important text."""
    n = len(s)
    rep = lambda c, j: ''.join([c for i in range(j)])
    line = '+' + rep('-', n) + '+'
    mid = '|' + s + '|'
    s = "%s\n%s\n%s" % (line, mid, line)
    return s

def atom_atom_dists(v):
    """Returns an array of all interatom distances.
    
    >>> atom_atom_dists([[1,0,0], [2,0,0]])
    array([ 1.])

    >>> atom_atom_dists([[1,0,0], [2,0,0], [3,0,0]])
    array([ 1.,  2.,  1.])

    >>> atom_atom_dists([1,0,0,2,0,0,3,0,0])
    array([ 1.,  2.,  1.])

    >>> len(atom_atom_dists([[1,0,0]]))
    0
    
    """
    v = numpy.array(v)
    assert v.size % 3 == 0
    v.shape = (-1,3)
    N = len(v)

    output = numpy.zeros(N*(N-1)/2)
    k = 0
    for i in range(N)[:-1]:
        for j in range(N)[i+1:]:
            d = v[i] - v[j]
            d = numpy.dot(d,d)**(0.5)
            output[k] = d
            k += 1

    assert (output != 0.0).all()

    return output

def pythag_seps(vs, metric = Default(None)):
    """Returns pythag distances between vectors in a list/vector of vectors.
    
    >>> pythag_seps([[0,0],[1,1],[2,2]])
    array([ 1.41421356,  1.41421356])
    
    """
    vs = numpy.asarray(vs)
    N = len(vs)
    subs = [vs[i] - vs[i-1] for i in range(1,N)]
    vm = [0.5 *(vs[i] + vs[i-1]) for i in range(1,N)]
    return numpy.array([metric.norm_up(sub, vi) for sub, vi in zip(subs,vm)])

def cumm_sum(list, start_at_zero=False):
    """
    >>> cumm_sum([1,2,3,4,5])
    array([  1.,   3.,   6.,  10.,  15.])

    >>> cumm_sum([1,2,3,4,5], start_at_zero=True)
    array([  0.,   1.,   3.,   6.,  10.,  15.])

    """
    N = len(list)
    l = numpy.zeros(N+1)
    list = numpy.array(list)

    for i in range(N):
        l[i+1] = l[i] + list[i]

    if start_at_zero:
        return l
    else:
        return l[1:]

class ObjLog:
    """Inheritable object supporting logging functionality."""
    def __init__(self, name, default='later', logfile='-', **kwargs):
        self._name = name
        self._modes = ('always', 'later', 'now', 'never')
        assert default in self._modes, "Legal values are: %s" % self._modes
        self._default = default
        self._logs = ''

        if isinstance(logfile, str):
          if logfile == '-':
              logfile = sys.stdout
          else:
              logfile = open(logfile, 'a')
        self.logfile = logfile

    def slog(self, *args, **kwargs):
        when = kwargs.get('when', self._default)
        assert when in self._modes, "Legal values are: %s" % self._modes
        s = ' '.join([str(i) for i in args])
        if when == 'always':
            self._logs += s + '\n'
            self.logfile.write(s + '\n')
        elif when == 'now':
            self.logfile.write(s + '\n')
        elif when == 'later':
            self._logs += s + '\n'
        elif when == 'never':
            return
        else:
            assert False, "Should never happen"

    def logflush(self):
        print self._logs
        self._logs = ''


# Testing the examples in __doc__strings, execute
# "python gxmatrix.py", eventualy with "-v" option appended:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# You need to add "set modeline" and eventually "set modelines=5"
# to your ~/.vimrc for this to take effect.
# Dont (accidentally) delete these lines! Unless you do it intentionally ...
# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax


