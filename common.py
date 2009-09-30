"""Classes, functions and variables common to all modules."""

import copy
import numpy
import os

PROGNAME = "searcher"
ERROR_STR = "error"

LOGFILE_EXT = ".log"

TMP_DIR_ENV_VAR = "AOE_TMP"
def get_tmp_dir():
    """Returns the absolute path to a temporary directory. If the environment
    variable AOE_TMP is specified (can be relative or absolute), then this is
    used. Otherwise the current working directory is used."""
    if TMP_DIR_ENV_VAR in os.environ:
        tmp_dir = os.path.abspath(os.environ[TMP_DIR_ENV_VAR])
    else:
        tmp_dir = os.path.abspath(os.getcwd())

    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    return tmp_dir


# Max no. of allowed geometries given to the optimiser to form the initial guess 
# for the reaction pathway. Includes the reactant/product.
MAX_GEOMS = 3

# default max iterations for optimiser
DEFAULT_MAX_ITERATIONS = 20

# unit conversions
ANGSTROMS_TO_BOHRS = 1.8897
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
    def __init__(self, v, energy, gradient = None, flags = dict()):
        self.v = v
        self.e = energy
        self.g = gradient
        self.flags = flags

    def __eq__(self, r):
        return (isinstance(r, self.__class__) and is_same_v(r.v, self.v)) or (r != None and is_same_v(r, self.v))

    def __repr__(self):
        s = self.__class__.__name__ + "( " + str(self.v) 
        s += ", " + str(self.e) + ", " + str(self.g)
        s += ", " + str(self.flags) + ")"
        return s

    def has_field(self, type):
        return type == Job.G() and self.g != None \
            or type == Job.E() and self.e != None
        
    def merge(self, res):
        assert is_same_v(self.v, res.v)
        assert is_same_e(self.e, res.e)

        if self.g == None:
            self.g = res.g
        else:
            raise ResultException("Trying to add a gradient result when one already exists")


class Job():
    """Specifies calculations to perform on a particular geometry v."""
    def __init__(self, v, l):
        self.v = v
        if not isinstance(l, list):
            l = [l]
        self.calc_list = l
    
    def __str__(self):
        s = ""
        for j in self.calc_list:
            s += j.__str__()
        return self.__class__.__name__ + " %s: %s" % (self.v, s)

    def geom_is(self, v_):
        assert len(v_) == len(self.v)
        return is_same_v(v_, self.v)

    def add_calc(self, calc):
        if self.calc_list.count(calc) == 0:
            self.calc_list.append(calc)

    def is_energy(self):
        return self.calc_list.count(self.E()) > 0
    def is_gradient(self):
        return self.calc_list.count(self.G()) > 0

    class E():
        def __eq__(self, x):
            return isinstance(x, self.__class__) and self.__dict__ == x.__dict__
        def __str__(self):  return "E"

    class G():
        def __eq__(self, x):
            return isinstance(x, self.__class__) and self.__dict__ == x.__dict__
        def __str__(self):  return "G"

def fname():
    import sys
    return sys._getframe(1).f_code.co_name


SAMENESS_THRESH_VECTORS = 1e-6
SAMENESS_THRESH_ENERGIES = 1e-6
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

def test_FourWellPot():
    fwp = FourWellPot()
    sp = SurfPlot(fwp)
    sp.plot(maxx=2.5, minx=-2.5, maxy=2.5, miny=-2.5)

class QCDriver:
    def __init__(self, dimension):
        self.dimension = dimension
        self.__g_calls = 0
        self.__e_calls = 0

    def get_calls(self):
        return (self.__e_calls, self.__g_calls)

    def gradient(self):
        self.__g_calls += 1
        pass

    def energy(self):
        self.__e_calls += 1
        pass


class GaussianPES():
#    def __init__(self):
#        QCDriver.__init__(self,2)

    def energy(self, v):
#        QCDriver.energy(self)
#        print "energy running", type(v)

        x = v[0]
        y = v[1]
        return (-numpy.exp(-(x**2 + y**2)) - numpy.exp(-((x-3)**2 + (y-3)**2)) + 0.01*(x**2+y**2) - 0.5*numpy.exp(-((1.5*x-1)**2 + (y-2)**2)))

    def gradient(self, v):
#        QCDriver.gradient(self)
#        print "gradient running"

        x = v[0]
        y = v[1]
        dfdx = 2*x*numpy.exp(-(x**2 + y**2)) + (2*x - 6)*numpy.exp(-((x-3)**2 + (y-3)**2)) + 0.02*x + 0.5*(4.5*x-3)*numpy.exp(-((1.5*x-1)**2 + (y-2)**2))
        dfdy = 2*y*numpy.exp(-(x**2 + y**2)) + (2*y - 6)*numpy.exp(-((x-3)**2 + (y-3)**2)) + 0.02*y + 0.5*(2*y-4)*numpy.exp(-((1.5*x-1)**2 + (y-2)**2))
#        print "gradient:", dfdx

        g = numpy.array((dfdx,dfdy))
        return g

class PlanePES():
    def energy(self, v):
        x = v[0]
        y = v[1]
        return x - y

    def gradient(self, v):
        x = v[0]
        y = v[1]
        dfdx = 1
        dfdy = -1
        g = numpy.array((dfdx,dfdy))
        return g

class PlanePES2():
    def energy(self, v):
        x = v[0]
        y = v[1]
        return x + y

    def gradient(self, v):
        x = v[0]
        y = v[1]
        dfdx = 1
        dfdy = 1
        g = numpy.array((dfdx,dfdy))
        return g


class GaussianPES2(QCDriver):
    def __init__(self):
        QCDriver.__init__(self,2)

    def energy(self, v):
        QCDriver.energy(self)

        x = v[0]
        y = v[1]
        return (-exp(-(x**2 + 0.2*y**2)) - exp(-((x-3)**2 + (y-3)**2)) + 0.01*(x**2+y**2) - 0.5*exp(-((x-1.5)**2 + (y-2.5)**2)))

    def gradient(self, v):
        QCDriver.gradient(self)

        x = v[0]
        y = v[1]
        dfdx = 2*x*exp(-(x**2 + 0.2*y**2)) + (2*x - 6)*exp(-((x-3)**2 + (y-3)**2)) + 0.02*x + 0.5*(2*x-3)*exp(-((x-1.5)**2 + (y-2.5)**2))
        dfdy = 2*y*exp(-(x**2 + 0.2*y**2)) + (2*y - 6)*exp(-((x-3)**2 + (y-3)**2)) + 0.02*y + 0.3*(2*y-5)*exp(-((x-1.5)**2 + (y-2.5)**2))

        return numpy.array((dfdx,dfdy))


class QuarticPES(QCDriver):
    def __init__(self):
        QCDriver.__init__(self,2)

    def gradient(self, a):
        QCDriver.gradient(self)

        if len(a) != self.dimension:
            raise QCDriverException("Wrong dimension")

        x = a[0]
        y = a[1]
        dzdx = 4*x**3 - 3*80*x**2 + 2*1616*x + 2*2*x*y**2 - 2*8*y*x - 80*y**2 
        dzdy = 2*2*x**2*y - 8*x**2 - 2*80*x*y + 2*1616*y + 4*y**3 - 3*8*y**2
        return numpy.array([dzdy, dzdx])

    def energy(self, a):
        QCDriver.energy(self)

        if len(a) != self.dimension:
            raise Exception("Wrong dimension")

        x = a[0]
        y = a[1]
        z = (x**2 + y**2) * ((x - 40)**2 + (y - 4) ** 2)
        return (z)

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


