"""Classes, functions and variables common to all modules."""

from numpy import *
PROGNAME = "searcher"
ERROR_STR = "error"

LOGFILE_EXT = ".log"

MAX_GEOMS = 3

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

    def __str__(self):
        s = self.__class__.__name__ + ": " + str(self.v) + " E = " + str(self.e) + " G = " + str(self.g)
        s += "\nFlags: " + str(self.flags)
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
            raise Exception("Trying to add a gradient result when one already exists")
 
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
    import numpy
    return numpy.linalg.norm(v1 - v2) < SAMENESS_THRESH_VECTORS
def is_same_e(e1, e2):
    return abs(e1 - e2) < SAMENESS_THRESH_ENERGIES

def line():
    print "=" * 80

def opt_gd(f, x0, fprime, callback = lambda x: None):
    """A gradient descent solver."""
    from numpy import *
    import copy
    i = 0
    x = copy.deepcopy(x0)
    prevx = zeros(len(x))
    while 1:
        g = fprime(x)
        dx = x - prevx
        if linalg.norm(g, ord=2) < 0.05:
            print "***CONVERGED after %d iterations" % i
            break

        i += 1
        prevx = x
        if callback != None:
            x = callback(x)
        x -= g * 0.2
        print line()
        print x
        print line()
#        raw_input('Wait...\n')

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
        return (-exp(-(x**2 + y**2)) - exp(-((x-3)**2 + (y-3)**2)) + 0.01*(x**2+y**2) - 0.5*exp(-((1.5*x-1)**2 + (y-2)**2)))

    def gradient(self, v):
#        QCDriver.gradient(self)
#        print "gradient running"

        x = v[0]
        y = v[1]
        dfdx = 2*x*exp(-(x**2 + y**2)) + (2*x - 6)*exp(-((x-3)**2 + (y-3)**2)) + 0.02*x + 0.5*(4.5*x-3)*exp(-((1.5*x-1)**2 + (y-2)**2))
        dfdy = 2*y*exp(-(x**2 + y**2)) + (2*y - 6)*exp(-((x-3)**2 + (y-3)**2)) + 0.02*y + 0.5*(2*y-4)*exp(-((1.5*x-1)**2 + (y-2)**2))
#        print "gradient:", dfdx

        g = array((dfdx,dfdy))
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

        return array((dfdx,dfdy))

class QuarticPES(QCDriver):
    def __init__(self):
        QCDriver.__init__(self,2)

    def gradient(self, a):
        QCDriver.gradient(self)

        if len(a) != self.dimension:
            raise Exception("Wrong dimension")

        x = a[0]
        y = a[1]
        dzdx = 4*x**3 - 3*80*x**2 + 2*1616*x + 2*2*x*y**2 - 2*8*y*x - 80*y**2 
        dzdy = 2*2*x**2*y - 8*x**2 - 2*80*x*y + 2*1616*y + 4*y**3 - 3*8*y**2
        return array([dzdy, dzdx])

    def energy(self, a):
        QCDriver.energy(self)

        if len(a) != self.dimension:
            raise Exception("Wrong dimension")

        x = a[0]
        y = a[1]
        z = (x**2 + y**2) * ((x - 40)**2 + (y - 4) ** 2)
        return (z)

