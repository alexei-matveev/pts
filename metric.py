from func import NumDiff
from numpy import dot, asarray, matrix, size
from numpy.linalg import solve

class Metric:
    """
    Includes metrix relevant functions, like for
    transforming contra- and covariant vectors into
    each other

    >>> from numpy import sin, cos, pi
    >>> from copy import deepcopy

    First test that a lowered raised indice is the same is
    a not transformed one:

    Functions: Identity function and a "pseudo" atoms function
    >>> def fun1(x):
    ...     return deepcopy(x)

    >>> def fun2(x):
    ...     return asarray([[0., 0., 0.], [x[0], 0., 0.],
    ...                    [x[1] * sin(x[2]), x[1] * cos(x[2]), 0.],
    ...                    [x[1] * sin(x[2]) + x[3] * sin(x[4]) * cos(x[5]) ,
    ...                     x[1] * cos(x[2]) + x[3] * cos(x[4]) * cos(x[5]) ,
    ...                     - x[3] * sin(x[5])]])

    Be careful that the angles are not too large
    >>> x = asarray([1.0, 2.0, 0.5, 1.0 , 0.6 , 0.1])

    Vector to transform
    >>> f_co = asarray([1.0, 2.0, 0.5, 1.0 , 0.6 , 0.1])

    Identity function:
    >>> met = Metric(fun1)

    First transformation
    >>> f_con = met.raises(f_co, x)

    Verify: back transformation == start
    >>> max(abs(f_co - met.lower(f_con, x))) < 1e-15
    True

    Pseudo atoms:
    >>> met = Metric(fun2)

    First transformation
    >>> f_con = met.raises(f_co, x)

    Verify: back transformation == start
    >>> max(abs(f_co - met.lower(f_con, x))) <  1e-15
    True

    Use a real function:
    >>> from quat import r3
    >>> met = Metric(r3)

    >>> x_r3 = asarray([8., pi/2. - 0.3, 0.2])

    >>> f_con = asarray([0.0002, 0.001, 0.001])

    >>> x_r3_e = x_r3 + f_con

    First transformation
    >>> f_co = met.lower(f_con, x_r3)

    Verify: back transformation == start
    >>> max(abs(f_con - met.raises(f_co, x_r3))) < 1e-15
    True

    >>> abs(dot(f_con, f_co) - dot(r3(x_r3)-r3(x_r3_e), r3(x_r3)-r3(x_r3_e))) < 1e-6
    True
    """
    def __init__(self, fun):
        """
        needs a function with a fprime function
        """
        self.fun = NumDiff(fun)

    def __fprime_as_matrix(self, x):
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
        return asarray(contoco(self.__fprime_as_matrix, place, vec))

    def raises(self, vec, place):
        return asarray(cotocon(self.__fprime_as_matrix, place, vec))


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


