#!/usr/bin/env python
"""

Test with the two-dimensional MB potential:

    >>> from pts.pes.mueller_brown import MuellerBrown as MB
    >>> f = MB()

Find the three minima, A, B and C as denoted in the original
paper (print them to finite precision to be able to exchange
minimizers):

    >>> from numpy import round
    >>> n = 6

    >>> b, info = minimize(f, [0., 0.])
    >>> round(b, n)
    array([ 0.623499,  0.028038])
    >>> round(f(b), n)
    -108.166724

    >>> a, info = minimize(f, [-1., 1.])
    >>> round(a, n)
    array([-0.558224,  1.441726])
    >>> round(f(a), n)
    -146.69951699999999

    >>> c, info = minimize(f, [0., 1.])
    >>> round(c, n)
    array([-0.050011,  0.466694])
    >>> round(f(c), n)
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
    >>> max(abs(fc1 - f(c))) < 1.e-7
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
    >>> xm, info = fmin(e3, x)

Location of dimer center:

    >>> xm[:2]
    array([-0.00084324, -0.00085287])

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

__all__ = ["minimize", "cminimize"]

from numpy import asarray, empty, dot, max, abs, shape, size
from numpy.linalg import solve #, eigh
from scipy.optimize import fmin_l_bfgs_b as minimize1D
from bfgs import LBFGS, BFGS

VERBOSE = False

TOL = 1.e-6

STOL = TOL   # step size tolerance
GTOL = 1.e-5 # gradient tolerance
CTOL = TOL   # constrain tolerance

MAXITER = 50
MAXSTEP = 0.04

def _solve(A, b):
    """
    Solve for x in A * x = b, albeit for arbotrary shapes of A, x, and
    b.
    """

    b = asarray(b)
    A = asarray(A)

    bshape = shape(b)
    ashape = shape(A)
    xshape = ashape[:len(bshape)]

    b = b.reshape(-1)
    A = A.reshape(-1, size(b))
    x = solve(A, b)

    return x.reshape(xshape)

def newton(x, fg, tol=TOL, maxiter=MAXITER, rk=None):
    """Solve F(x) = 0 (rather, reduce rhs to < tol)

        >>> from numpy import array
        >>> a, b = 1., 10.
        >>> def fg(r):
        ...    x = r[0]
        ...    y = r[1]
        ...    f = array([ a * x**2 + b * y**2 - a * b,
        ...                b * x**2 + a * y**2 - a * b])
        ...    fprime = array([[ 2. * a * x, 2. * b * y], 
        ...                    [ 2. * b * x, 2. * a * y ]])
        ...    return f, fprime

        >>> x = array([3., 7.])

        >>> x0, info = newton(x, fg, tol=1.0e-14)
        >>> info["iterations"]
        9

        >>> from ode import rk5
        >>> x0, info = newton(x, fg, tol=1.0e-14, rk=rk5)
        >>> info["iterations"]
        4

    Though  this  iteration  number  is  misleading as  each  of  them
    involved extra (5?) evaluations of the Jacobian.

        >>> x0
        array([ 0.95346259,  0.95346259])

        >>> f, J = fg(x0)
        >>> f
        array([ 0.,  0.])
    """

    if rk is not None:
        #                                 -1
        # for integration of dx / dt = - J (x) f
        #                                       0
        def xprime(t, x, f0):
            f, J = fg(x)
            return _solve(J, -f0)

    it = 0
    converged = False

    while not converged and it < maxiter:
        it = it + 1

        f, J = fg(x)

        if max(abs(f)) < tol:
            converged = True

        if rk is None:
            # FIXME: what if J is rank-deficent?
            dx = _solve(J, -f)
        else:
            # use provided routine for Runge-Kutta step prediciton:
            dx = rk(0.0, x, xprime, 1.0, args=(f,))
            # FIXME: this re-evaluates fg(x) at x!

        x = x + dx

    assert converged

    info = { "converged": converged,
             "iterations": it,
             "value": f,
             "derivative": J }

    return x, info

def minimize(f, x, xtol=STOL, ftol=GTOL, **kw):
    """
    Minimizes a Func |f| starting with |x|.
    Returns (xm, info)

    Input:

        ftol    --- force tolerance
        xtol    --- step tolerance
        **kw    --- unused

    xm          --- location of the minimum
    fm          --- f(xm)
    stats       --- optimization statistics
    """

    # in case we are given a list instead of array:
    x = asarray(x)
    if VERBOSE:
        print "x(in) =\n", x

    # save the shape of the actual argument:
    xshape = x.shape

    # some optimizers (notably fmin_l_bfgs_b) work with funcitons
    # of 1D arguments returning both the value and the gradient.
    # Construct such from the given Func f:
    fg = _flatten(f.taylor, x)

    # flat version of inital point:
    y = x.flatten()

    if False:
        xm, info = fmin(fg, y, hess="BFGS", stol=xtol, gtol=ftol)
    else:
        xm, fm, info =  minimize1D(fg, y)

        #
        # External optimizer has its own conventions:
        #
        info["converged"] = (info["warnflag"] == 0)
        info["iterations"] = info["funcalls"]
        del info["funcalls"]
        info["value"] = fm
        info["derivative"] = info["grad"].reshape(xshape)
        del info["grad"]

    # return the result in original shape:
    xm.shape = xshape

    if VERBOSE:
        print "x(out)=\n", xm

    return xm, info

def cminimize(f, x, c, **kwargs):
    """
    Minimizes a Func |f| starting with |x| under constrains |c|.
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
    cg = _flatten(c.taylor, x)

    # flat version of inital point:
    y = x.flatten()
#   z = c0.flatten()

    xm, fm, stats =  cmin(fg, y, cg, **kwargs)

    # return the result in original shape:
    xm.shape = xshape

    return xm, fm, stats

def fmin(fg, x, stol=STOL, gtol=GTOL, maxiter=MAXITER, maxstep=MAXSTEP, alpha=70.0, hess="BFGS"):
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
        Has to support |update| and |inv| methods.
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
    r = asarray(x)

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
        dr = - hessian.inv(g)

        # restrict the maximum component of the step:
        longest = max(abs(dr))
        if longest > maxstep:
            if VERBOSE:
                print "fmin: dr=", dr, "(too long, scaling down)"
            dr *= maxstep / longest

        if VERBOSE:
            print "fmin: dr=", dr
            print "fmin: dot(dr, g)=", dot(dr, g)

        # Save  for later  comparison. Assignment  e, g  =  fg(r) will
        # re-bind (e, g), not modify them:
        r0, e0, g0 = r, e, g

        # rebind, do not update the variable:
        r = r + dr

        # invoke objective function, also computes the gradient:
        e, g = fg(r)

        # check convergence, if any:
        criteria = 0

        if max(abs(dr)) < stol:
            if VERBOSE:
                print "fmin: converged by step max(abs(dr))=", max(abs(dr)), '<', stol
            criteria += 1

        if max(abs(g))  < gtol:
            if VERBOSE:
                print "fmin: converged by force max(abs(g))=", max(abs(g)), '<', gtol
            criteria += 1

        if criteria >= 2:
            converged = True

        if iteration >= maxiter:
            if VERBOSE:
                print "fmin: exceeded number of iterations", maxiter
            break # out of the while loop

    # also return number of interations, convergence status, and last values
    # of the gradient and step:
    info = { "converged": converged,
             "iterations": iteration,
             "value": e,
             "derivative": g,
             "step": dr }
    return r, info # (iteration, converged, g, dr)

def cmin(fg, x, cg, c0=None, stol=STOL, gtol=GTOL, ctol=CTOL, \
        maxiter=MAXITER, maxstep=MAXSTEP, alpha=70.0, hess="LBFGS", callback=None):
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
        Has to support |update| and |inv| methods.
        """

    # interpret a string as a constructor name:
    hess = eval(hess)

    # returns the default hessian:
    hessian = hess(alpha)

    # shurtcut for linear operator g -> hessian.inv(g):
    H = hessian.inv

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

    # should have more variables than constrains:
    assert len(r) > len(c0)

    if VERBOSE:
        print "cmin: c0=", c0, "(target value of constrain)"

    iteration = -1 # prefer to increment at the top of the loop
    converged = False

    while not converged and iteration < maxiter:
        iteration += 1

        # update the hessian representation:
        if iteration > 0: # only then r0 and g0 are meaningfull!
            hessian.update(r-r0, g-g0)

        # compute the constrained step:
        dr, dg, lam = qnstep(g, H, c - c0, A)

        if VERBOSE:
            if e0 is not None:
                print "cmin: e - e0=", e - e0
            print "cmin: r=", r
            print "cmin: e=", e
            print "cmin: g=", g
            print "cmin: ..",     dot(lam, A), "(    dot(lam, A))"
            print "cmin: ..", g + dot(lam, A), "(g + dot(lam, A))"
            print "cmin: c=", c
            print "cmin: criteria=", max(abs(dr)), max(abs(c - c0)), max(abs(g + dot(lam, A)))

        # check convergence, if any:
        if max(abs(dr)) < stol and max(abs(c - c0)) < ctol:
            if VERBOSE:
                print "cmin: converged by step max(abs(dr))=", max(abs(dr)), '<', stol
                print "cmin: and constraint max(abs(c - c0))=", max(abs(c - c0)), '<', ctol
            converged = True

        # purified gradient for CURRENT geometry:
        if max(abs(g + dot(lam, A)))  < gtol and max(abs(c - c0)) < ctol:
            # FIXME: this may change after update step!
            if VERBOSE:
                print "cmin: converged by force max(abs(g + dot(lam, A)))=", max(abs(g + dot(lam, A))), '<', gtol
                print "cmin: and constraint max(abs(c - c0))=", max(abs(c - c0)), '<', ctol
            converged = True

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

        # FIXME: evaluate constrains at current geometry (another time?):
        c, A = cg(r)

        # if requested, provide feedback on the optimization progress:
        if callback is not None:
            callback(r, e, g, c, A)

        if VERBOSE:
            if iteration >= maxiter:
                print "cmin: exceeded number of iterations", maxiter
            # see while loop condition ...

    # also return number of interations, convergence status, and last values
    # of the gradient and step:
    return r, e, (iteration, converged, g, dr)

def qnstep(g0, H, c, A):
    """
    At point |x0| we have the gradient |g0|, the quadratic PES is
    characterized by inverse hessian H that relates changes in
    coordinate and gradients: dr = H(dg). H(g) is a linear operator
    implemented as a function hiding the actual implementation.

    As we operate with inverse hessian, so g is the primary variable:

        g -> x -> c(x), A(x) = dc / dx

    As this is a constrained minimization we are not looking for the
    point where the gradients vanish. Instead seek such (a point where)

        g1 + lam * A = 0

    i.e. where energy gradients and constrain gradients are "collinear".

    The following holds exactly on quadratic surface:

        x1 - x0 = H * (g1 - g0)

    We also want for the constrain to hold at x1:

        c(x1) = C

    Formally one has to solve the non-linear equations
    for (g1, x1, lam), the non-linearity is due to x-dependence of
    constrains c(x) and A(x). This sub proposes a step of a single
    Newton-Rapson iteration for this system of non-linear equations.
    More specifically, we first solve for g1 in linear approximation
    for c(x):

        c(x1) ~= c(x0) + A * (x1 - x0)

    and then solve the linear equations:

        x1 - x0 = H * (g1 - g0)         (1)
        g1 + lam * A = 0                (2)
        c(x0) + A * (x1 - x0) = C       (3)

    Which is first solved for lagrange multipliers:

        A * H * A' * lam = c(x0) - C - A * H * g0

    And then, using (2) for g1:

        g1 = - lam * A

    We will, however return the increments, g1 - g0.
    """

    # current mismatch in constrains:
    # c == c(x0) - C

    #
    # Note that A[i, j] here is dc_i / dx_j,
    # the literature may differ by transposition.
    #

    # this would be the unconstrained step:
    dx = - H(g0)

    # this would be the new values of the constrains:
    rhs = c + dot(A, dx)

    if VERBOSE:
        print "qnstep: A=", A
        print "qnstep: dx=", dx
        print "qnstep: c=", c
        print "qnstep: rhs=", rhs

    # Construct the lhs-matrix AHA^T
    AHA = aha(A, H)

    # solve linear equations:
    lam = solve(AHA, rhs)

    if VERBOSE:
        print "qnstep: rhs=", rhs
        print "qnstep: AHA=", AHA
        print "qnstep: lam=", lam

    g1 = - dot(lam, A)

    dg = g1 - g0

    # dx, dg, lam:
    return H(dg), dg, lam

def aha(A, H):
    "Construct the lhs-matrix AHA^T"

    # number of constrains:
    nc = len(A)

    AHA = empty((nc, nc))
    for j in range(nc): # FIXME: more efficient way?
        Haj = H(A[j])
        for i in range(nc):
            AHA[i, j] = dot(A[i], Haj)

    return AHA

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
