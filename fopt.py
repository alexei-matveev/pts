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

from numpy import asarray, empty, dot, max, abs
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

def lbfgs(fg, x, stol=0.0001, ftol=0.0070, maxiter=100, maxstep=0.04, memory=10, alpha=70.0):
    """Limited memory BFGS optimizer.
    
    A limited memory version of the bfgs algorithm. Unlike the bfgs algorithm,
    the inverse of Hessian matrix is updated.  The inverse
    (??) Hessian is represented only as a diagonal matrix to save memory (??)

    Parameters:

    fg: objective function x -> (f, g)
        returns the value f and the gradient g at x

    maxstep: float
        How far is a single atom allowed to move. This is useful for DFT
        calculations where wavefunctions can be reused if steps are small.
        Default is 0.04 Angstrom.

    memory: int
        Number of steps to be stored. Default value is 100. Three numpy
        arrays of this length containing floats are stored.

    alpha: float
        Initial guess for the Hessian (curvature of energy surface). A
        conservative value of 70.0 is the default, but number of needed
        steps to converge might be less if a lower value is used. However,
        a lower value also means risk of instability.
        
        """

    H0 = 1. / alpha  # Initial approximation of inverse Hessian
                     # 1./70. is to emulate the behaviour of BFGS
                     # Note that this is never changed!

    def update(hessian, dr, dg):
        """Update representation of the Hessian.
        See corresponding |apply|-function for more info.
        """

        if hessian is None:
            # compact repr of the hessian:
            hessian = ([], [], [], H0)
            # three lists for
            # (1) geometry changes,
            # (2) gradient changes and
            # (3) their precalculated dot-products.
            # (4) overall scaling?

            # There is probably no (dr, dg) anyway in the first iteration:
            return hessian


        # expand the hessian repr:
        s, y, rho, h0 = hessian

        s.append(dr)

        y.append(dg)
        
        rho0 = 1.0 / dot(dr, dg)
        if rho0 <= 0:
            print "WARNING: dot(y, s) =", rho0, " <= 0 in L-BFGS"
            print "         y =", dg, "(gradient diff.)"
            print "         s =", dr, "(step)"
            print "         Chances are the hessian will loose positive definiteness!"

        rho.append(rho0)

        # forget the oldest:
        if len(s) > memory:
            s.pop(0)
            y.pop(0)
            rho.pop(0)

        # return updated hessian model:
        return s, y, rho, h0

    def apply(hessian, g):
        """Computes z = H * g using internal representation
        of the inverse hessian, H = B^-1.

        This appears to be the update scheme described in

            Jorge Nocedal, Updating Quasi-Newton Matrices with Limited Storage
            Mathematics of Computation, Vol. 35, No. 151 (Jul., 1980), pp. 773-782

        See also:

            R. H. Byrd, J. Nocedal and R. B. Schnabel, Representation of
            quasi-Newton matrices and their use in limited memory methods",
            Mathematical Programming 63, 4, 1994, pp. 129-156
        """

        # expand representaiton of hessian:
        s, y, rho, h0 = hessian

        # amount of stored data points:
        n = len(s)
        # WAS: loopmax = min([memory, iteration])

        a = empty((n,))

        ### The algorithm itself:
        q = g.copy() # needs it!
        for i in range(n - 1, -1, -1): # range(n) in reverse order
            a[i] = rho[i] * dot(s[i], q)
            q -= a[i] * y[i]
        z = h0 * q
        
        for i in range(n):
            b = rho[i] * dot(y[i], z)
            z += s[i] * (a[i] - b)

        return z


    # returns the default hessian:
    hessian = update(None, None, None)

    # geometry, energy and the gradient from previous iteration:
    r0 = None
    e0 = None # not used anywhere!
    g0 = None

    # initial value for the variable:
    r = x

    iteration = -1 # prefer to increment at the top of the loop
    converged = False

    while not converged:
        iteration += 1

        # invoke objective function, also computes the gradient:
        e, g = fg(r)

#       if e0 is not None:
#           print "lbfgs: e - e0=", e - e0
#       print "lbfgs: r=", r
#       print "lbfgs: e=", e
#       print "lbfgs: g=", g
    
        if iteration > 0: # only then r0 and g0 are meaningfull!
            hessian = update(hessian, r-r0, g-g0)

        # Quasi-Newton step: df = - H^-1 * g:
        dr = - apply(hessian, g)

        # restrict the maximum component of the step:
        longest = max(abs(dr))
        if longest > maxstep:
#           print "lbfgs: dr=", dr, "(too long, scaling down)"
            dr *= maxstep / longest

#       print "lbfgs: dr=", dr
#       print "lbfgs: dot(dr, g)=", dot(dr, g)

        # save for later comparison, need a copy, see "r += dr" below:
        r0 = r.copy()

        # "e, g = fg(r)" will re-bind (e, g), not modify them:
        e0 = e # not used anywhere!
        g0 = g

        # actually update the variable:
        r += dr

        # check convergence, if any:
        if max(abs(dr)) < stol:
#           print "lbfgs: converged by step max(abs(dr))=", max(abs(dr)), '<', stol
            converged = True
        if max(abs(g))  < ftol:
#           print "lbfgs: converged by force max(abs(g))=", max(abs(g)), '<', ftol
            converged = True
        if iteration >= maxiter:
#           print "lbfgs: exceeded number of iterations", maxiter
            converged = True
        # if e0 is not None:
        #     # the last step should better minimize the energy slightly:
        #     if e0 - e < etol: converged = True

    return r, e, iteration


# python fopt.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
