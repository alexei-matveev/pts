from func import NumDiff
from numpy import dot, asarray, matrix, size
from numpy import sqrt
from numpy import zeros, eye
from numpy.linalg import solve
from copy import deepcopy


__all__ = ["setup_metric", "metric"]


class Default:
    """
    Includes metrix relevant functions, like for
    transforming contra- and covariant vectors into
    each other

    This variant supposes that contra- and covariant vectors
    are the same

    >>> met = Default(None)
    >>> from numpy import pi

    >>> x = None

    >>> dx_con = asarray([0.0002, 0.001, 0.001])

    First transformation
    >>> dx_co = met.lower(dx_con, x)

    Verify: this metric changes nothing:
    >>> max(abs(dx_con - dx_co)) < 1e-15
    True

    Verify: this metric changes nothing:
    >>> max(abs(dx_con - met.raises(dx_co, x))) < 1e-15
    True
    """
    def __init__(self, fun = None):
        """
        normally needs a function with a fprime function
        in this special case it is only asked for it because
        of consistency
        """
        pass

    def lower(self, vec, place):
        return deepcopy(vec)

    def raises(self, vec, place):
        return deepcopy(vec)

    def norm_up(self, vec, place):
        vec_co = self.lower(vec, place)
        return sqrt(dot(vec, vec_co))

    def norm_down(self, vec, place):
        vec_con = self.raises(vec, place)
        return sqrt(dot(vec_con, vec))

    def __str__(self):
        return "Metric: Working with Metric direct internals (Default)"

class Metric(Default):
    """
    Includes metrix relevant functions, like for
    transforming contra- and covariant vectors into
    each other

    >>> from numpy import pi

    Use a real function:
    >>> from quat import r3
    >>> met = Metric(r3)

    >>> x = asarray([8., pi/2. - 0.3, 0.2])

    >>> dx_con = asarray([0.0002, 0.001, 0.001])

    >>> x2 = x + dx_con

    First transformation
    >>> dx_co = met.lower(dx_con, x)

    Verify: back transformation == start
    >>> max(abs(dx_con - met.raises(dx_co, x))) < 1e-15
    True

    >>> abs(dot(dx_con, dx_co) - dot(r3(x)-r3(x2), r3(x)-r3(x2))) < 1e-6
    True

    Use norm function of metric (attention, they give back the sqrt)

    >>> abs(met.norm_up(dx_con, x) - sqrt(dot(r3(x)-r3(x2), r3(x)-r3(x2)))) < 1e-6
    True

    >>> abs(met.norm_down(dx_co, x) - sqrt(dot(r3(x)-r3(x2), r3(x)-r3(x2)))) < 1e-6
    True
    """
    def __init__(self, fun):
        """
        needs a function with a fprime function
        """
        self.fun = NumDiff(fun)

    def _fprime_as_matrix(self, x):
        """
        Some functions return not linear arrays of different shapes.
        Then fprime will be not a 2-dimensional array but more.
        Thus transform the result to a matrix as if the
        function used for fprime would give back a flattened result.
        """
        a = asarray(self.fun.fprime(x))
        lenx = size(x)
        # convert shape of a, so that second dimension is length of vector x
        a.shape = (-1, lenx)
        return a

    def lower(self, vec, place):
        return asarray(contoco(self._fprime_as_matrix, place, vec))

    def raises(self, vec, place):
        return asarray(cotocon(self._fprime_as_matrix, place, vec))

    def __str__(self):
        return "Metric: Working with Metric Cartesians (Metric)"



class No_metric():
    def __init__(self, fun = None):
        pass

    def __str__(self):
        s =  "Metric: No metric set! Please initalize a valid metric before using it."
        s = s + '\n' + "This metric does not provide any other functionalities"
        return s
"""
The chosen metric, available are:
   * Default, for which contra- and covariant vectors are the same
   * and Metric, using a Cartesian metric

Use metric.version() to find out which one is set currently.
Store and initalize the choosen metric in metric
Here set up the global variable.
Before use make sure that it is initalized with a fitting
variable by function setup_metric.
Do not import metric directly, as it would be the None version
but use for example

 import pts.metric as mt
 ...
 co_vec = mt.lower(con_vec)
"""
global metric
metric = No_metric()

def setup_metric(F = None):
     """
     sets and initalises the metric
     F should be a function, which should when run by
     itself provide for the internal coordinates y
     the corresponding Cartesian coordinates x
     This function has to be called once, afterwards
     all modules should be able to access the metric.
     """
     global metric
     metric = Default(F)
     print metric

def contoco(F, pos, vec):
     """
     Assuming that F is a function to get the derivatives, transforming vectors
     of kind vec into cartesians, and that all takes place at position pos,
     this function transforms contra in covariant vectors.
     """
     B = F(pos)
     return btb(B, vec)

def cotocon(F, pos, vec):
     """
     Assuming that F is a function to get the derivatives, transforming vectors
     of kind vec into cartesians, and that all takes place at position pos,
     this function transforms co in contravariant vectors.
     """
     B = F(pos)
     M = dot(B.T, B)
     return solve(M, vec)

def btb(B, vec):
     """
     Returns the product B^T B vec
     """
     return dot(B.T, dot(B, vec))

# Testing the examples in __doc__strings, execute
# "python gxmatrix.py", eventualy with "-v" option appended:
if __name__ == "__main__":
     import doctest
     doctest.testmod()


