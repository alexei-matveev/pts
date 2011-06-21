"""
Some classes which might be used as (wrapper for) functions for the
CoordSys2.
They are all with fprime /taylor attached as funcs. Be aware that
they expeced when used as wrappers, that they get also an inner function
which is a Func.

    >>> from pts.func import NumDiff
    >>> from pts.zmat import ZMat
    >>> from numpy import max, abs, pi

    A example function:
    >>> rep = [(None, None, None), (0, None, None), (0, 1, None)]
    >>> fun1 = ZMat(rep)

    A second example:
    >>> fun2 = Justcarts()

    Vector for H2O
    >>> y = asarray([0.96, 0.96, 104.5 * pi / 180.0])

    Cartesian start vector:
    >>> x = asarray(fun1(y)).flatten()
    >>> x_raw = fun1(y)

    Test addition of global paramter:

    >>> f_wg = With_globals(fun1)

    >>> y_wg = zeros(9)
    >>> y_wg[:3] = y

    If the globals are zero, nothing is changed:
    >>> max(abs(f_wg(y_wg) - fun1(y))) < 1e-12
    True

    Pseudo inverse:
    >>> max(abs(f_wg(f_wg.pinv(x_raw)) - f_wg(y_wg))) < 1e-12
    True

    Ensure that the derivatives are correct:
    Consider that NumDiff has size (n,3,m), while we get normally something of
    size (n*3, m) back.
    >>> f_nd = NumDiff(f_wg)
    >>> max(abs(f_wg.fprime(y_wg) - f_nd.fprime(y_wg))) < 1e-12
    True

    Test adding several functions:
    >>> f_set = Mergefuncs([fun2, fun2, fun2], [3,3,3], dims_rev = [1,1,1])

    Gives the same result than a complete function for all three of them:
    >>> max(abs(f_set(x) - fun2(x))) < 1e-12
    True
    >>> max(abs(f_set.fprime(x) - fun2.fprime(x))) < 1e-12
    True

    >>> x2 = f_set(x)
    >>> max(abs(f_set(f_set.pinv(x2)) - f_set(x))) < 1e-12
    True

    Test fixture of coordinates:
    Fix the Angle:
    >>> mask = [True, True, False]
    >>> f_m = Masked(fun1, mask, y)

    Now only two variables are needed as internal coordinates:
    >>> y_red = asarray([0.96, 0.96])

    At the start point still the right values are given:
    >>> max(abs(f_m(y_red) - fun1(y))) < 1e-12
    True

    >>> x2 = f_m(y_red)
    >>> max(abs(f_m(f_m.pinv(x2)) - f_m(y_red))) < 1e-12
    True

    Ensure that the derivatives are correct:
    >>> f_nd = NumDiff(f_m)
    >>> max(abs(f_m.fprime(y_red) - f_nd.fprime(y_red))) < 1e-12
    True

    Now have some values repeated:
    Both bond-length are the same:
    >>> mask = [1, 1, 2]
    >>> f_we = With_equals(fun1, mask)

    >>> y_r2 =  asarray([0.96, 104.5 * pi / 180.0])

    Still the same result:
    >>> max(abs(f_we(y_r2) - fun1(y))) < 1e-12
    True

    >>> x = f_we(y_r2)
    >>> max(abs(f_we(f_we.pinv(x))-f_we(y_r2))) < 1e-12
    True

    Ensure that the derivatives are correct:
    >>> f_nd = NumDiff(f_we)
    >>> max(abs(f_we.fprime(y_r2) - f_nd.fprime(y_r2))) < 1e-12
    True

    Both bond-lengt are the same and the angle is fixed:
    >>> mask = [1, 1, 0]
    >>> f_we = With_equals(fun1, mask, y)

    >>> y_r2 =  asarray([0.96])

    Still the same result:
    >>> max(abs(f_we(y_r2) - fun1(y))) < 1e-12
    True

    >>> x = f_we(y_r2)
    >>> max(abs(f_we(f_we.pinv(x))-f_we(y_r2))) < 1e-12
    True

    Ensure that the derivatives are correct:
    >>> f_nd = NumDiff(f_we)
    >>> max(abs(f_we.fprime(y_r2) - f_nd.fprime(y_r2))) < 1e-12
    True
"""
from copy import deepcopy
from numpy import eye, zeros, hstack, asarray
from numpy import array

from pts.func import Func
from pts.zmat import RT

class Pass_through(Func):
     """
     No real function, only needed stored here for pickling reasons.

     Returns the same that it got.
     """
     def taylor(self, x):
         x1 = deepcopy(x)
         der = eye(len(x))
         return x1, der

     def pinv(self, y):
         x1 = deepcopy(y)
         return x1


class Justcarts(Func):
    """
    Takes Cartesian coordinates and returns Cartesian coordinates.
    It considers only that the internal coordinates are an array, while
    the Cartesian are supposed to be a Matrix of size (-1, 3)
    """
    def __init__(self):
        pass

    def taylor(self, x):
        res = deepcopy(x)
        res.shape = (-1, 3)
        dres = eye(len(x.flatten()))
        dres.shape = (-1, 3, len(x))
        return res, dres

    def pinv(self, y):
        res = deepcopy(y)
        res.shape = (-1)
        return res

class With_globals(Func):
    """
    Adds to function fun_raw some global parameter for:
      rotation (as vector description of quaternion)
      translation
      of the complete molecule.
      The rotation and translation is calculated with rt object.
      It is supposed that for each internal coordinates given, the
      last three coordinates are the global translation and the
      three before that are for the global rotation.
    After each calculation of the Cartesian coordinates by fun_raw, they
    are adapted with the global parameter.
    """
    def __init__(self, fun_raw):
        self._rt = RT(fun_raw)

    def taylor(self, y):
        # y = [v, w, t]
        # w is global rotation (quaternion expressed as vector)
        # t is global translation
        # both w and t have three elements
        v = y[:-6]
        w = y[-6:-3]
        t = y[-3:]

        # rt needs zmatrix coordinates, rotation and translation separate
        x, (xv, xw, xt) = self._rt.taylor(v, w, t)

        dx = zeros((len(x[:,0]), 3, len(y)))

        # got the derivatives separate for v, w,t
        # here put them into one object
        dx[:,:,:-6] = xv
        dx[:,:,-6:-3] = xw
        dx[:,:,-3:] = xt

        # give back x as a 1-dimensional array (rt gave
        # back as x, y,z compontent for each "atom"
        # change the derivatives accordingly
        x.shape = (-1, 3) # (number of atoms, 3)
        dx.shape = (-1, 3, len(y)) # ( number of atoms, 3, number internal coordinates)
        # (-1 means that in this direction the size should be made
        # fitting to the other specifications (so that the complete
        # array could be transformed))

        return x, dx


    def pinv(self, x):
        v, w, t = self._rt.pinv(x)
        y = zeros(len(v) + 6)
        y[:-6] = v
        y[-6:-3] = w
        y[-3:] = t

        return y

class Mergefuncs(Func):
    """
    Makes of a set of functions (in list funs_raw) a combined
    functions.
    For initalization it also needs to know the dimensions of the
    needed internal coordinate arrays for the single functions,
    as it needs to know where to break the complete internals vector.
    """
    def __init__(self, funs_raw, dims, dims_rev = None):
        self._funs_raw = funs_raw
        self._dims = dims
        # only needed if pinv will be used
        self._dims_rev = dims_rev

    def taylor(self, x):
        res = []
        dres = []
        iter = 0
        # first calculate the results separate for each function
        # consider here which part of the complete internal coordinate vector
        # belongs to it (stored in dims)
        for fun, dim in zip(self._funs_raw, self._dims):
             r1, dr1 = fun.taylor(x[iter:iter + dim])
             res.append(r1.flatten())
             dres.append(dr1)
             iter += dim

        assert (iter == len(x))
        # the Cartesian coordinates have only to be brought to the correct shape
        res = hstack(res) # add all the coordinates up
        res.shape = (-1, 3) # (number atoms, 3)

        # for the derivatives some more has to be done: as the functions does not
        # interact which each other, one has fill up the derivatives with zeros
        # the single derivative results will be put on the diagonals of the
        # complete matrix
        lenx = len(res.flatten())
        mat = zeros((lenx, iter))

        i = 0 # direction of Cartesian coordinates
        j = 0 # direction of internal coordinates
        for  m1, dim in zip(dres, self._dims):
             m1.shape = (-1, dim)
             a, b = m1.shape

             mat[i:i + a, j : j + b] = m1
             i += a
             j += dim
        mat.shape = (-1, 3, iter) # number atoms, 3, number internal coordinates

        return res, mat

    def pinv(self, y):
        res = []
        iter = 0
        for fun, dim in zip(self._funs_raw, self._dims_rev):
             r1 = fun.pinv(y[iter:iter + dim,:])
             res.append(r1.flatten())
             iter += dim

        assert (iter == len(y[:,0]))
        res = hstack(res) # add all the coordinates up
        return res

class Masked(Func):
    """
    The initial function fun_raw has also some variables, which have
    to be fixed. They are put as False in the mask. One has to give
    only once a complete vector (for_fix) for all other calculations
    the fixed coordinates of this one are added to the current set of
    internals before they are handed to the inner function.
    """
    def __init__(self, fun_raw, mask, for_fix):
        self._fun_raw = fun_raw
        self._mask = mask
        self._x0 = for_fix
        assert len(for_fix) == len(mask)

    def taylor(self, x):

        # take the stored vector
        x1 = deepcopy(self._x0)
        j = 0
        # and exchange all the variables which are not fixed
        for i, mp in enumerate(self._mask):
             if mp:
                  x1[i] = x[j]
                  j += 1

        # the inner function works on the extended vector
        res, dres = self._fun_raw.taylor(x1)

        # make sure that the derivatives have the correct shape (might
        # also see the Cartesians as x*3 variables.
        dres.shape = (-1, len(x1))
        # for easier remove of the unused functions, transpose
        dres = dres.T

        # the derivatives for the fixed coordinates are ommited
        mat = [dres[i] for i in range(len(self._mask)) if self._mask[i]]
        mat = asarray(mat)

        # transpose mat back
        mat = mat.T
        mat.shape = (-1, 3, len(x)) # number atoms, 3, number internal coordinates

        return res, mat

    def pinv(self, y):
        res = self._fun_raw.pinv(y)
        return array([res[i] for i in range(len(self._mask)) if self._mask[i]])

class With_equals(Func):
    """
    Is a wrapper function around fun_raw which considers that there are
    several variables with the same value (as given by mask), which have to
    be given several times to the inner function
    The mask is here a list of integers. It gives for each wanted internal coordinate
    of the inner function which value of the internal coordinates given to with_equals
    should be used (it is supposed to be several times the same one)
    """
    def __init__(self, fun_raw, mask, for_fix = None):
        self._fun_raw = fun_raw
        self._mask = mask
        self._x0 = for_fix

    def taylor(self, x):

        # take the stored vector
        x1 = zeros(len(self._mask))
        # and exchange all the variables which are not fixed
        for i, mp in enumerate(self._mask):
             if mp < 0:
                  x1[i] = -x[-mp-1]
             elif mp > 0:
                  x1[i] = x[mp-1]
             else:
                  x1[i] = self._x0[i]

        # the inner function works on the extended vector
        res, dres = self._fun_raw.taylor(x1)

        # make sure that the derivatives have the correct shape (might
        # also see the Cartesians as x*3 variables.
        dres.shape = (-1, len(x1))
        a, b = dres.shape
        # transpose dres for easier acces to derivatives of internal coordinates
        dres = dres.T
        mat = zeros((len(x), a))

        # the derivatives for the fixed coordinates are ommited
        for  mp, d1 in zip(self._mask, dres):
             if mp < 0:
                 mat[-mp-1] -= d1
             elif mp > 0:
                 mat[mp-1] += d1

        # transpose mat back
        mat = mat.T
        mat.shape = (-1, 3, len(x)) # number atoms, 3, number internal coordinates

        return res, mat

    def pinv(self, y):
        tol = 1e-6

        x0 = self._fun_raw.pinv(y)
        x = zeros(max(self._mask))
        num = zeros(max(self._mask))

        # take average over those values which appear more often
        for mp, x1 in zip(self._mask, x0):
            if mp < 0:
                x[-mp-1] -= x1
                num[-mp-1] += 1
            elif mp > 0:
                x[mp-1] += x1
                num[mp-1] += 1

        for i in range(len(x)):
            x[i] = x[i] / num[i]

        # give warning if the corresponding values differ too much
        for mp, x1 in zip(self._mask, x0):
            if mp < 0:
                if abs(x[-mp-1] + x1) > tol:
                    print "WARNING: values declared as equal differ more than ", tol
                    print "         for the variable", -mp, "the average over all appearences was", x[-mp-1]
                    print "         while the current value gave", -x1
            elif mp > 0:
                if abs(x[mp-1] - x1) > tol:
                    print "WARNING: values declared as equal differ more than ", tol
                    print "         for the variable", mp, "the average over all appearences was", x[mp-1]
                    print "         while the current value gave", x1

        return x

# python cfunc.py [-v]:
if __name__ == "__main__":
     import doctest
     doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
