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

    >>> b, fb, stats = minimize(f, [0., 0.])
    >>> round(b, n)
    array([ 0.623499,  0.028038])
    >>> round(fb, n)
    -108.166724

    >>> a, fa, stats = minimize(f, [-1., 1.])
    >>> round(a, n)
    array([-0.558224,  1.441726])
    >>> round(fa, n)
    -146.69951699999999

    >>> c, fc, stats = minimize(f, [0., 1.])
    >>> round(c, n)
    array([-0.050011,  0.466694])
    >>> round(fc, n)
    -80.767818000000005

Constrained minimization works with flat funcitons
returning value and gradient as a tuple.
Thus take the |taylor| method of the MB-Func:

    >>> fg = f.taylor

Lets define a plane which will constrain the search
for minimum:

    >>> from numpy import array
    >>> plane = array([1., 1.])

And a constrain funciton returning a list of values
(here of length one) to aim to keep constant during
optimization (and their derivatives):

    >>> def cg(x):
    ...     c = dot(x, plane)
    ...     g = plane
    ...     return array([c]), array([g])

Do a constrained minimization starting from the point
offset from the minimum C on MB-surface along the plane:

    >>> c1, fc1, _ = cmin(fg, [c[0] + 0.1, c[1] - 0.1], cg)
    >>> max(abs(c1 - c)) < 1.e-7
    True
    >>> max(abs(fc1 - fc)) < 1.e-7
    True

Energy of a dimer at around the minimum A on MB-surface,
here x[0] is the center of the dimer and x[1] is the vector
connecting the two points:

    >>> def e2(x):
    ...     xa = a + x[0] + x[1] / 2.
    ...     xb = a + x[0] - x[1] / 2.
    ...     fa, fap = fg(xa)
    ...     fb, fbp = fg(xb)
    ...     e  = (fa + fb) / 2.
    ...     g  = [ (fap + fbp) / 2., (fap - fbp) / 4. ]
    ...     return e, array(g)

This will be used for orienting the dimer:

    >>> from numpy import cos, sin, pi
    >>> R = 0.1
    >>> def r(t):
    ...     return array([  R * cos(t), R * sin(t) ]), array([ -R * sin(t), R * cos(t) ])

This is a function of dimer center x[:2] and its
orientation angle x[2]:

    >>> def e3(x):
    ...     t  = x[2]
    ...     d, dprime  = r(t)
    ...     e, g = e2([x[:2], d])
    ...     gt = dot(g[1], dprime)
    ...     g3  = [ g[0,0], g[0,1], gt ]
    ...     return e, array(g3)

The fixed dimer length was eliminated, one can use
plain minimizers:

    >>> x = array([0., 0., 0.])
    >>> xm, fm, _ = fmin(e3, x)

Location of dimer center:

    >>> xm[:2]
    array([-0.00084325, -0.00085287])

Dimer orientaiton:

    >>> r(xm[2])[0]
    array([ 0.07067461,  0.07074673])

Constrain funciton to preserve the length of a "dimer",
only the dimer vector x[1] is used here:

    >>> def d2(x):
    ...     r  = x[1]
    ...     r2 = dot(r, r)
    ...     g  = [array([0., 0.]), 2 * r ]
    ...     return array([r2]), array([g])

Starting geometry of a dimer:

    >>> d, dp = r(0.)
    >>> x = array([array([0., 0.]), d])

Consider the e2/d2 funcitons as functions of 1D argument:

    >>> d2 = _flatten(d2, x)
    >>> e2 = _flatten(e2, x)
    >>> x = x.flatten()

Run constrained minimization:

    >>> cmin(e2, x, d2)[0]
    array([-0.00084324, -0.00085287,  0.07067462,  0.07074672])

The two first numbers give the center of the dimer, two last
give its orientation.

"""

__all__ = ["minimize"]

from numpy import asarray, empty, dot, max, abs, shape
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
    fg = _flatten(f.taylor, x)

    # flat version of inital point:
    y = x.flatten()

    xm, fm, stats =  minimize1D(fg, y)
    #xm, fm, stats =  fmin(fg, y, hess="LBFGS") #, stol=1.e-6, ftol=1.e-5)
    #xm, fm, stats =  fmin(fg, y, hess="BFGS") #, stol=1.e-6, ftol=1.e-5)

    # return the result in original shape:
    xm.shape = xshape

    return xm, fm, stats

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

    # invoke objective function, also computes the gradient:
    e, g = fg(r)

    iteration = -1 # prefer to increment at the top of the loop
    converged = False

    while not converged:
        iteration += 1

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

        # invoke objective function, also computes the gradient:
        e, g = fg(r)

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

def cmin(fg, x, cg, c0=None, stol=1.e-6, ftol=1.e-5, ctol=1.e-6, maxiter=50, maxstep=0.04, alpha=70.0, hess="LBFGS"):
    """Search for a minimum of fg(x)[0] using the gradients fg(x)[1]
    subject to constrains cg(x)[0] = const.

    TODO: dynamic trust radius, line search in QN direction (?)

    Parameters:

    fg: objective function x -> (f, fprime)
        returns the value f and the gradient g at x

    cg: (differentiable) constrains x -> (c, cprime)
        returns the vector of constrais and their derivatives wrt x

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

    # invoke objective function, also computes the gradient:
    e, g = fg(r)

    # evaluate constrains at current geometry:
    c, A = cg(r)

    # save the initial value of the constrains (not necessarily zeros):
    if c0 is None:
        c0 = c
    else:
        assert len(c) == len(c0)
        c0 = asarray(c0)

    if VERBOSE:
        print "cmin: c0=", c0, "(target value of constrain)"

    iteration = -1 # prefer to increment at the top of the loop
    converged = False

    while not converged:
        iteration += 1

        # update the hessian representation:
        if iteration > 0: # only then r0 and g0 are meaningfull!
            hessian.update(r-r0, g-g0)

        ################## Constrain section ############################

        # current mismatch in constrains:
        c = c - c0

        #
        # First solve for largange multipliers "lam":
        #
        # (1) dr = - H * ( g + A^T * lam )
        # (2) 0  = c + A * dr
        #
        # Note that A[i, j] here is dc_i / dr_j,
        # the literature may differ by transposition.
        #

        # this would be the unconstrained step:
        dr0 = - hessian.apply(g)

        # this would be the new values of the constrains:
        rhs = c + dot(A, dr0)

        # number of constrains:
        nc = len(c)

        # construct the lhs-matrix AHA^T:
        AHA = empty((nc, nc))
        for j in range(nc): # FIXME: more efficient way?
            Haj = hessian.apply(A[j])
            for i in range(nc):
                AHA[i, j] = dot(A[i], Haj)

        # solve linear equations:
        lam = solve(AHA, rhs)

        #
        # Now project out the gradient components,
        # and propose a new step:
        #

        gp = g + dot(lam, A)
        #################################################################

        # Quasi-Newton step (using *projected* gradient): dr = - H * gp:
        dr = - hessian.apply(gp)

        if VERBOSE:
            if e0 is not None:
                print "cmin: e - e0=", e - e0
            print "cmin: r=", r
            print "cmin: e=", e
            print "cmin: g=", g
            print "cmin: ..",     dot(lam, A), "(    dot(lam, A))"
            print "cmin: ..", g + dot(lam, A), "(g + dot(lam, A))"
            print "cmin: c=", c

        # restrict the maximum component of the step:
        longest = max(abs(dr))
        if longest > maxstep:
            if VERBOSE:
                print "cmin: dr=", dr, "(too long, scaling down)"
            dr *= maxstep / longest
            # NOTE: step restriciton also does not allow to fix
            #       the mismatch (c-c0) in constrains in one shot ...

        if VERBOSE:
            print "cmin: dr=", dr
            print "cmin: dot(A, dr)=", dot(A, dr)
            print "cmin: dot(g, dr)=", dot(g, dr)

        # save for later comparison, need a copy, see "r += dr" below:
        r0 = r.copy()

        # "e, g = fg(r)" will re-bind (e, g), not modify them:
        e0 = e # not used anywhere!
        g0 = g

        # actually update the variable:
        r += dr

        # invoke objective function, also computes the gradient:
        e, g = fg(r)

        # evaluate constrains at current geometry:
        c, A = cg(r)

        # check convergence, if any:
        if max(abs(dr)) < stol and max(abs(c - c0)) < ctol:
            if VERBOSE:
                print "cmin: converged by step max(abs(dr))=", max(abs(dr)), '<', stol
            converged = True

        # purified gradient for updated geometry is not yet available:
#       if max(abs(g))  < ftol:
#           if VERBOSE:
#               print "cmin: converged by force max(abs(g))=", max(abs(g)), '<', ftol
#           converged = True

        if iteration >= maxiter:
            if VERBOSE:
                print "cmin: exceeded number of iterations", maxiter
            break # out of the while loop

    # also return number of interations, convergence status, and last values
    # of the gradient and step:
    return r, e, (iteration, converged, g, dr)

def _flatten(fg, x):
    """Returns a funciton of flat argument fg_(y) that
    properly reshapes y to x, and returns values and gradients as by fg.

    Only the shape of the argument x is used here, not the value.

    A length-two funciton of 2x2 argument:

        >>> from numpy import array
        >>> def fg(x):
        ...     f1 = x[0,0] + x[0,1]
        ...     f2 = x[1,0] + x[1,1]
        ...     g1 = array([[1., 1.], [0., 0.]])
        ...     g2 = array([[0., 0.], [1., 1.]])
        ...     return array([f1, f2]), array([g1, g2])

        >>> x = array([[1., 2.], [3., 4.]])
        >>> f, g = fg(x)

    Returns two-vector:

        >>> f
        array([ 3.,  7.])

    and 2 x (2x2) derivative:

        >>> g
        array([[[ 1.,  1.],
                [ 0.,  0.]],
        <BLANKLINE>
               [[ 0.,  0.],
                [ 1.,  1.]]])

    A flat arument length two-funciton of length-four argument:

        >>> fg = _flatten(fg, x)
        >>> x = x.flatten()

        >>> f, g = fg(x)

    Same two values values:

        >>> f
        array([ 3.,  7.])

    And 2 x 4 derivative:

        >>> g
        array([[ 1.,  1.,  0.,  0.],
               [ 0.,  0.,  1.,  1.]])
    """

    # in case we are given a list instead of array:
    x = asarray(x)

    # shape of the actual argument:
    xshape = x.shape
    xsize  = x.size
    # print "xshape, xsize =", xshape, xsize

    # define a flattened function based on original fg(x):
    def fg_(y):
        "Returns both, value and gradient, treats argument as flat array."

        # need copy to avoid obscure error messages from fmin_l_bfgs_b:
        x = y.copy() # y is 1D

        # restore the original shape:
        x.shape = xshape

        f, fprime = fg(x) # fprime is returned as nD!

        # in case f is an array, preserve this structure:
        fshape = shape(f) # () for scalars

        # still treat the arguments as 1D structure of xsize:
        return f, fprime.reshape( fshape + (xsize,) )

    # return new funciton:
    return fg_

# python fopt.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
