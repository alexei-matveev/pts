"""Classes, functions and variables common to all modules."""


PROGNAME = "searcher"
ERROR_STR = "error"

MAX_GEOMS = 3

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
        s += "\nFlags: " + str(flags)
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

SAMENESS_THRESH = 1e-6
def is_same_v(v1, v2):
    import numpy
    return numpy.linalg.norm(v1 - v2) < SAMENESS_THRESH
def is_same_e(e1, e2):
    return abs(e1 - e2) < SAMENESS_THRESH

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


