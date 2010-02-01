#!/usr/bin/python
"""

Test with the two-dimensional MB potential:

    >>> from mueller_brown import MuellerBrown as MB
    >>> f = MB()

Find the three minima, A, B and C as denoted in the original
paper (print them to finite precision to be able to exchange
minimizers):

    >>> from numpy import round
    >>> n = 6

    >>> b, fb, _ = minimize(f, [0., 0.])
    >>> round(b, n)
    array([ 0.623499,  0.028038])
    >>> round(fb, n)
    -108.166724

    >>> a, fa, _ = minimize(f, [-1., 1.])
    >>> round(a, n)
    array([-0.558224,  1.441726])
    >>> round(fa, n)
    -146.69951699999999

    >>> c, fc, _ = minimize(f, [0., 1.])
    >>> round(c, n)
    array([-0.050011,  0.466694])
    >>> round(fc, n)
    -80.767818000000005

"""

__all__ = ["minimize"]

from numpy import asarray, empty, dot, max, abs
from numpy import eye, outer
from numpy.linalg import solve #, eigh
from scipy.optimize import fmin_l_bfgs_b as minimize1D
from bfgs import LBFGS, BFGS

VERBOSE = False

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

    # some optimizers (notably fmin_l_bfgs_b) work with funcitons
    # of 1D arguments returning both the value and the gradient.
    # Construct such from the given Func f:
    fg = flatfunc(f, x)

    # flat version of inital point:
    y = x.flatten()

    xm, fm, stats =  minimize1D(fg, y)
    #xm, fm, stats =  fmin(fg, y, hess="LBFGS") #, stol=1.e-6, ftol=1.e-5)
    #xm, fm, stats =  fmin(fg, y, hess="BFGS") #, stol=1.e-6, ftol=1.e-5)

    # return the result in original shape:
    xm.shape = xshape

    return xm, fm, stats

def flatfunc(f, x):
    """Returns a funciton of flat argument fg(y) that
    properly reshapes y to x, and returns values and gradients
    of f:

        fg(y) = (f(x), f.fprime(x).flatten())

    where y == x.flatten()

    Only the shape of the argument x is used here, not the value.
    """

    # in case we are given a list instead of array:
    x = asarray(x)

    # shape of the actual argument:
    xshape = x.shape

    # define a flattened function using f() and f.prime():
    def fg(y):
        "Returns both, value and gradient, treats argument as flat array."

        # need copy to avoid obscure error messages from fmin_l_bfgs_b:
        x = y.copy() # y is 1D

        # restore the original shape:
        x.shape = xshape

        fx = f(x)
        gx = f.fprime(x) # fprime returns nD!

        return fx, gx.flatten()

    # return a new funciton:
    return fg

def fmin(fg, x, stol=1.e-6, ftol=1.e-5, maxiter=50, maxstep=0.04, alpha=70.0, hess="BFGS"):
    """Search for a minimum of fg(x)[0] using the gradients fg(x)[1].

    TODO: dynamic trust radius, line search in QN direction (?)

    Parameters:

    fg: objective function x -> (f, g)
        returns the value f and the gradient g at x

    maxstep: float
        How far is a single atom allowed to move. This is useful for DFT
        calculations where wavefunctions can be reused if steps are small.
        Default is 0.04 Angstrom.

    alpha: float
        Initial guess for the Hessian (curvature of energy surface). A
        conservative value of 70.0 is the default, but number of needed
        steps to converge might be less if a lower value is used. However,
        a lower value also means risk of instability.

    hess: "LBFGS" or "BFGS"
        A name of the class implementing hessian update scheme.
        Has to support |update| and |apply| methods.
        """

    # interpret a string as a constructor name:
    hess = eval(hess)

    # returns the default hessian:
    hessian = hess(alpha)

    # geometry, energy and the gradient from previous iteration:
    r0 = None
    e0 = None # not used anywhere!
    g0 = None

    # initial value for the variable:
    r = asarray(x).copy() # we are going to modify it!

    iteration = -1 # prefer to increment at the top of the loop
    converged = False

    while not converged:
        iteration += 1

        # invoke objective function, also computes the gradient:
        e, g = fg(r)

        if VERBOSE:
            if e0 is not None:
                print "fmin: e - e0=", e - e0
            print "fmin: r=", r
            print "fmin: e=", e
            print "fmin: g=", g

        # update the hessian representation:
        if iteration > 0: # only then r0 and g0 are meaningfull!
            hessian.update(r-r0, g-g0)

        # Quasi-Newton step: df = - H * g, H = B^-1:
        dr = - hessian.apply(g)

        # restrict the maximum component of the step:
        longest = max(abs(dr))
        if longest > maxstep:
            if VERBOSE:
                print "fmin: dr=", dr, "(too long, scaling down)"
            dr *= maxstep / longest

        if VERBOSE:
            print "fmin: dr=", dr
            print "fmin: dot(dr, g)=", dot(dr, g)

        # save for later comparison, need a copy, see "r += dr" below:
        r0 = r.copy()

        # "e, g = fg(r)" will re-bind (e, g), not modify them:
        e0 = e # not used anywhere!
        g0 = g

        # actually update the variable:
        r += dr

        # check convergence, if any:
        if max(abs(dr)) < stol:
            if VERBOSE:
                print "fmin: converged by step max(abs(dr))=", max(abs(dr)), '<', stol
            converged = True
        if max(abs(g))  < ftol:
            if VERBOSE:
                print "fmin: converged by force max(abs(g))=", max(abs(g)), '<', ftol
            converged = True
        if iteration >= maxiter:
            if VERBOSE:
                print "fmin: exceeded number of iterations", maxiter
            break # out of the while loop

    # also return number of interations, convergence status, and last values
    # of the gradient and step:
    return r, e, (iteration, converged, g, dr)

# python fopt.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
