"""
Ridders method for numerical differentiation
Translated from Numerical Recepies in f77.

Not for expensive functions!

    >>> from numpy import sin, cos
    >>> fprime, err = dfridr(sin, 0.0)
    >>> round(fprime, 12), err < 1e-12
    (1.0, True)

Check numerical differentiation on the whole domain:

    >>> from numpy import pi, linspace, max, abs
    >>> fprime, err = dfridr(sin, 0.0)
    >>> def diff(x):
    ...     fprime, err = dfridr(sin, x)
    ...     return cos(x) - fprime
    >>> 1e-12 > max(abs([ diff(x) for x in linspace(-pi, pi, 11) ]))
    True

Check if error estimate is reasonable:

    >>> def chkerr(x):
    ...     fprime, err = dfridr(sin, x)
    ...     return abs(cos(x) - fprime) <= 10. * err

Without factor 10 this will have some False entries:

    >>> [ chkerr(x) for x in linspace(-pi, pi, 11) ]
    [True, True, True, True, True, True, True, True, True, True, True]
"""

__all__ = ["dfridr"]

# this affects precision for some reason:
from numpy import abs, max

def dfridr(func, x, h=0.001):
    """
    Returns the derivative of a function func at a point x by Ridders' method of polynomial
    extrapolation. The value h is input as an estimated initial stepsize; it need not be small,
    but rather should be an increment in x over which func changes substantially. An estimate
    of the error in the derivative is returned as err.
    Parameters: Stepsize is decreased by CON at each iteration. Max size of tableau is set by
    NTAB. Return when error is SAFE worse than the best so far.
    """
#   INTEGER NTAB
#   REAL dfridr,err,h,x,func,CON,CON2,BIG,SAFE
#   PARAMETER (CON=1.4,CON2=CON*CON,BIG=1.E30,NTAB=10,SAFE=2.)
    CON = 1.4
    CON2 = CON * CON
    BIG = 1.0e30
    NTAB = 10
    SAFE = 2.0
#   INTEGER i,j
#   REAL errt,fac,hh,a(NTAB,NTAB)

    # use dict() for 2D array:
    a = dict()

    if h == 0.0: raise Exception('h must be nonzero in dfridr')

    hh = h

    a[0, 0] = (func(x + hh) - func(x - hh)) / (2.0 * hh)

    err = BIG
    result = None
    for i in range(1, NTAB):
        # Successive columns in the Neville tableau will go to smaller
        # stepsizes and higher orders of extrapolation.
        hh = hh / CON
        a[0, i] = (func(x + hh) - func(x - hh)) / (2.0 * hh) # Try new, smaller stepsize.

        fac = CON2
        for j in range(1, i+1):
            # Compute extrapolations of various orders, requiring no new function evaluations.
            a[j, i] = (a[j-1, i] * fac - a[j-1, i-1]) / (fac - 1.)
            fac = CON2 * fac
            errt = max(abs(a[j, i] - a[j-1, i]), abs(a[j, i] - a[j-1, i-1]))

            # The error strategy is to compare each new extrapolation to one order lower, both at
            # the present stepsize and the previous one.
            if errt <= err: # If error is decreased, save the improved answer.
                err = errt
                result = a[j, i]

        if abs(a[i, i] - a[i-1, i-1]) >= SAFE * err:
            # If higher order is worse by a signicant factor SAFE, then quit early.
            return result, err

    return result, err

# "python ridders.py [-v]", eventualy with "-v" option appended:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
