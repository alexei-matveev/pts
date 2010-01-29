#!/usr/bin/python
"""

Test with the two-dimensional MB potential:

    >>> from mueller_brown import MuellerBrown as MB
    >>> f = MB()

Find the three minima, A, B and C as denoted in the original
paper:

    >>> b, fb, _ = minimize(f, [0., 0.])
    >>> b
    array([ 0.62349942,  0.02803776])
    >>> fb
    -108.16672411685231

    >>> a, fa, _ = minimize(f, [-1., 1.])
    >>> a
    array([-0.55822362,  1.44172583])
    >>> fa
    -146.6995172099532

    >>> c, fc, _ = minimize(f, [0., 1.])
    >>> c
    array([-0.05001084,  0.46669421])
    >>> fc
    -80.767818129651545

"""

__all__ = ["minimize"]

from numpy import asarray
from scipy.optimize import fmin_l_bfgs_b as minimize1D

def minimize(f, x):
    """
    Minimizes a Func |f| starting with |x|.
    Returns (xm, fm, stats)

    xm          --- location of the minimum
    fm          --- f(xm)
    stats       --- optimization statistics
    """

    # in case we are given a list instead of array:
    x = asarray(x)

    # save the shape of the actual argument:
    xshape = x.shape

    def fg(y):
        "Returns both, value and gradient, treats argument as flat array."

        # need copy to avoid obscure error messages from fmin_l_bfgs_b:
        x = y.copy() # y is 1D

        # restore the original shape:
        x.shape = xshape
        #   print "x=", type(x), x

        fx = f(x)
        gx = f.fprime(x) # fprime returns nD!

        return fx, gx.flatten()

    # flat version of inital point:
    y = x.flatten()

    # test only:
    #   e, g = fg(y)
    #   print "y=", type(y), y
    #   print "e=", type(e), e
    #   print "g=", type(g), g

    xm, fm, stats =  minimize1D(fg, y)

    # return the result in original shape:
    xm.shape = xshape

    return xm, fm, stats

# python fopt.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
