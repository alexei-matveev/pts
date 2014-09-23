#!/usr/bin/env python
from __future__ import print_function
"""
Test with the two-dimensional MB potential:

    >>> from pts.pes.mueller_brown import MuellerBrown as MB
    >>> f = MB()

Find the  three minima, A,  B and C  as denoted in the  original paper
(print them to finite precision to be able to exchange minimizers):

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

Constrained minimization works with flat funcitons returning value and
gradient as a tuple.  Thus take the |taylor| method of the MB-Func:

    >>> fg = f.taylor

Lets define a plane which will constrain the search for minimum:

    >>> from numpy import array
    >>> plane = array([1., 1.])

And a  constrain funciton returning a  list of values  (here of length
one)  to   aim  to  keep  constant  during   optimization  (and  their
derivatives):

    >>> def cg(x):
    ...     c = dot(x, plane)
    ...     g = plane
    ...     return array([c]), array([g])

Do a constrained minimization starting  from the point offset from the
minimum C on MB-surface along the plane:

    >>> c1, info = cmin(fg, [c[0] + 0.1, c[1] - 0.1], cg)
    >>> max(abs(c1 - c)) < 1.e-7
    True
    >>> max(abs(f(c1) - f(c))) < 1.e-7
    True

Energy of a dimer at around  the minimum A on MB-surface, here x[0] is
the center  of the  dimer and  x[1] is the  vector connecting  the two
points:

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
    ...     f = array([  R * cos(t), R * sin(t) ])
    ...     g = array([ -R * sin(t), R * cos(t) ])
    ...     return f, g

This is  a function  of dimer center  x[:2] and its  orientation angle
x[2]:

    >>> def e3(x):
    ...     t  = x[2]
    ...     d, dprime  = r(t)
    ...     e, g = e2([x[:2], d])
    ...     gt = dot(g[1], dprime)
    ...     g3  = [ g[0,0], g[0,1], gt ]
    ...     return e, array(g3)

The fixed dimer length was eliminated, one can use plain minimizers:

    >>> x = array([0., 0., 0.])
    >>> xm, info = fmin(e3, x)

Location of dimer center:

    >>> xm[:2]
    array([-0.00084324, -0.00085287])

Dimer orientaiton:

    >>> r(xm[2])[0]
    array([ 0.07067461,  0.07074673])

Constrain funciton to preserve the length of a "dimer", only the dimer
vector x[1] is used here:

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

The two first numbers give the  center of the dimer, two last give its
orientation.

"""

__all__ = ["minimize", "cminimize"]

from numpy import array, asarray, empty, dot, max, abs, shape, size
from numpy.linalg import solve #, eigh
from scipy.optimize import fmin_l_bfgs_b as minimize1D
from scipy.optimize import fmin_slsqp
from bfgs import get_by_name # BFGS, BFGS

VERBOSE = 0

TOL = 1.e-6

STOL = TOL                      # step size tolerance
GTOL = 1.e-5                    # gradient tolerance
CTOL = TOL                      # constrain tolerance

MAXIT = 50
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

def newton(x, fg, tol=TOL, maxiter=MAXIT, rk=None):
    """
    Solve F(x) = 0 (rather, reduce rhs to < tol)

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

def minimize (f, x, xtol=STOL, ftol=GTOL, maxit=MAXIT, algo=0, maxstep=MAXSTEP, **kw):
    """
    Minimizes a Func |f| starting with |x|.
    Returns (xm, info)

    Input:

        ftol    --- force tolerance
        xtol    --- step tolerance
        algo    --- algorith to be used:
                    0) Limited memory BFGS as distributed with SciPy.
                    1) built in
        **kw    --- unused

    xm          --- location of the minimum
    fm          --- f(xm)
    stats       --- optimization statistics
    """

    # in case we are given a list instead of array:
    x = asarray(x)
    if VERBOSE:
        print ("fopt: x(in)=\n", x)
        if len(kw) > 0:
            print ("fopt: ignored kwargs=", kw)

    # save the shape of the actual argument:
    xshape = x.shape

    # Some optimizers  (notably fmin_l_bfgs_b) work  with funcitons of
    # 1D  arguments  returning  both   the  value  and  the  gradient.
    # Construct such from the given Func f:
    fg = _flatten(f.taylor, x)

    # flat version of inital point:
    y = x.flatten()

    if algo == 1:
        xm, info = fmin(fg, y, hess="BFGS", stol=xtol, gtol=ftol, maxiter=maxit, maxstep=maxstep)
    elif algo == 0:
        xm, fm, info =  minimize1D(fg, y, pgtol=ftol, maxfun=maxit) #, iprint=1)

        #
        # External optimizer has its own conventions:
        #
        info["converged"] = (info["warnflag"] == 0)
        info["iterations"] = info["funcalls"]
        del info["funcalls"]
        info["value"] = fm
        info["derivative"] = info["grad"].reshape(xshape)
        del info["grad"]
    else:
        assert False, "No such algo = %d" % algo

    # Return the result in original shape:
    xm.shape = xshape

    # Also recorded trajectory should be re-shaped:
    if "trajectory" in info:
        def reshp (x):
            y = array (x)
            y.shape = xshape
            return y
        info["trajectory"] = map (reshp, info["trajectory"])

    if VERBOSE:
        print ("fopt: x(out)=\n", xm)

    return xm, info

def cminimize(f, x, c, algo=0, **kwargs):
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

    # Some optimizers  (notably fmin_l_bfgs_b) work  with funcitons of
    # 1D  arguments  returning  both   the  value  and  the  gradient.
    # Construct such from the given Func f:
    fg = _flatten(f.taylor, x)
    cg = _flatten(c.taylor, x)

    # flat version of inital point:
    y = x.flatten()
#   z = c0.flatten()

    if algo == 0:
        xm, info =  cmin(fg, y, cg, **kwargs)
    else:
        # See SciPy docs for scipy.optimize.fmin_slsqp:
        def func (x):
            e, g = fg (x)
            return e
        def fprime (x):
            e, g = fg (x)
            return g
        # FIXME: what if c0 is in kwargs?
        c0, _ = cg (y)
        def f_eqcons (x):
            c, A = cg (x)
            return c - c0
        def fprime_eqcons (x):
            c, A = cg (x)
            return A
        xm, fx, its, imode, smode = \
            fmin_slsqp (func, y, fprime=fprime, \
                            f_eqcons=f_eqcons, \
                            fprime_eqcons=fprime_eqcons, \
                            iprint=VERBOSE, \
                            full_output=True)
        assert (imode == 0)
        xm = asarray (xm)
        info = {"converged": (imode == 0),
                "iterations": its,
                "value": fx,
                "imode": imode,
                "smode": smode}

    # return the result in original shape:
    xm.shape = xshape

    return xm, info

def fmin(fg, x, stol=STOL, gtol=GTOL, maxiter=MAXIT, maxstep=MAXSTEP, alpha=70.0, hess="BFGS"):
    """
    Search for a minimum of fg(x)[0] using the gradients fg(x)[1].

    TODO: dynamic trust radius, line search in QN direction (?)

    Parameters:

    fg: x -> (f, g)

        Objective function  to be minimized,  returns the value  f and
        the gradient g at x.

    maxstep: float

        How far is  a single atom allowed to move.  This is useful for
        DFT calculations  where wavefunctions  can be reused  if steps
        are small.  Default is 0.04 Angstrom.

    alpha: float

        Initial guess for the Hessian (curvature of energy surface). A
        conservative  value of  70.0  is the  default,  but number  of
        needed steps  to converge  might be less  if a lower  value is
        used. However, a lower value also means risk of instability.

    hess: "LBFGS" or "BFGS"

        A name  of the class implementing hessian  update scheme.  Has
        to support |update| and |inv| methods.

    Example:

        >>> from pes.rosenbrock import Rosenbrock

        >>> f = Rosenbrock()

        >>> x0, info = fmin(f.taylor, [-1.2, 1.], gtol=1e-12, maxstep=1.0)

        >>> x0
        array([ 1.,  1.])

        >>> info["iterations"]
        25
        """

    # Interpret a string as a constructor name:
    hess = get_by_name(hess)

    # Returns the default hessian:
    hessian = hess(alpha)

    # Geometry, energy and the gradient from previous iteration:
    r0 = None
    e0 = None # not used anywhere!
    g0 = None

    # Initial value for the variable:
    r = asarray(x)

    # Invoke objective function, also computes the gradient:
    e, g = fg(r)

    # This will be destructively modified by appending new entries:
    trajectory = [r]

    iteration = -1        # prefer to increment at the top of the loop
    converged = False

    while not converged:
        iteration += 1
        # Prefix for debug output:
        pfx = "fmin: (%03d) " % iteration

        if VERBOSE:
            if e0 is not None:
                print (pfx, "e - e0=", e - e0)
            print (pfx, "e=", e)
            print (pfx, "max(abs(g))=", max(abs(g)))
            if VERBOSE > 1:
                print (pfx, "r=\n", r)
                print (pfx, "g=\n", g)

        # Update the hessian representation:
        if iteration > 0:       # only then r0 and g0 are meaningfull!
            hessian.update(r-r0, g-g0)

        # Quasi-Newton step: df = - H * g, H = B^-1:
        dr = - hessian.inv(g)

        #
        # This  should better  be  the descent  direction, holds  when
        # hessian is positive definite, H > 0:
        #
        assert dot(dr, g) <= 0.0

        # Restrict the maximum component of the step:
        longest = max(abs(dr))
        if longest > maxstep:
            if VERBOSE:
                print (pfx, "max(abs(dr))=", longest, ">", maxstep, "(too long, scaling down)")
            dr *= maxstep / longest

        if VERBOSE:
            print (pfx, "dot(dr, g)=", dot(dr, g))
            if VERBOSE > 1:
                print (pfx, "dr=\n", dr)

        # Save  for later  comparison. Assignment  e, g  =  fg(r) will
        # re-bind (e, g), not modify them:
        r0, e0, g0 = r, e, g

        #
        # Investigate objective function  along the descent direction,
        # also  computing  gradient at  every  point.   It may  become
        # problematic  to rely on  the small  energy changes  close to
        # convergence because of finite accuracy of QM energies.
        #
        # Thus, for  the moment, instead of making  sure the objective
        # function is indeed reduced (First Wolfe condition) make sure
        # that the gradient is not becoming too big while changing the
        # sign. Remeber that  dot(dr, g(r0)) is negative as  dr is the
        # descent direction:
        #
        #     dot(dr, g(r0 + alpha * dr)) <= - C2 * dot(dr, g(r0))
        #
        # For that  choose a scale  factor alpha small enough  so that
        # gradient projection does not grow  much on the other side of
        # the minumum.
        #
        # This is a somewhat  weakened second Wolfe condition.  FIXME:
        # one might  decide for relaxing the condition  of the maximal
        # step length if one wants an additional requirement,
        #
        #      C2 * dot(dr, g(r0)) <= dot(dr, g(r0 + alpha * dr))
        #
        # that together with the first correspond to strong version of
        # the second Wolfe condition.
        #

        #
        # First- and second Wolfe  (Armijo) parameters, C1 and C2. The
        # former is not used, see comments below:
        #
        C1, C2 = 1.0e-4, 0.9

        #
        # Try  the full QN  step first,  if the  step was  not already
        # scaled down, of course:
        #
        alpha = 1.0

        e, g = fg(r + alpha * dr)

        if VERBOSE:
            print (pfx, "dot(dr, g1)=", dot(dr, g))

        #
        # First Wolfe  (Armijo) condition is difficult  to satisfy for
        # noisy VASP energies, when close to convergence. FIXME: maybe
        # make it active only if dot(dr, g0) >> etol?
        #
        # while e > e0 + C1 * alpha * dot(dr, g0):
        #
        while dot(dr, g) > - C2 * dot(dr, g0) and alpha > 0.5**10:

            #
            # With the condition above this  scales the step size by a
            # factor that  is below 1/(1+C2)  < 1.  With C2=0.9  it is
            # 0.53. So  practically, this  reduces the step  by factor
            # 1.9 or more:
            #
            alpha = alpha * (-dot(dr, g0)) / (dot(dr, g) - dot(dr, g0))

            if VERBOSE:
                # print pfx, "retry with alpha=", alpha,\
                #     "energy e=", e, "too high"
                print (pfx, "retry with alpha=", alpha, \
                    "dot(dr, g)=", dot(dr, g), "too high")

            # compute them again:
            e, g = fg(r + alpha * dr)

            if VERBOSE:
                print (pfx, "dot(dr, g1)=", dot(dr, g))

        # FIXME: Wolfe-2 unsatisfied!"
        assert alpha > 0.5**10

        if e > e0 + C1 * alpha * dot(dr, g0):
            if VERBOSE:
                print (pfx, "WARNING: Wolfe condition 1 not satisfied!")

        # rebind, do not update the variables:
        dr = alpha * dr
        r = r + dr

        # Append new point:
        trajectory.append (r)

        # check convergence, if any:
        criteria = 0

        if max(abs(dr)) < stol:
            if VERBOSE:
                print (pfx, "converged by step max(abs(dr))=", max(abs(dr)), '<', stol)
            criteria += 1

        if max(abs(g))  < gtol:
            if VERBOSE:
                print (pfx, "converged by force max(abs(g))=", max(abs(g)), '<', gtol)
            criteria += 1

        if criteria >= 2:
            converged = True

        if iteration >= maxiter:
            if VERBOSE:
                print (pfx, "exceeded number of iterations", maxiter)
            break # out of the while loop

    # Also return number of  interations, convergence status, and last
    # values of the gradient and step:
    info = { "converged": converged,
             "iterations": iteration,
             "value": e,
             "derivative": g,
             "step": dr,
             "trajectory": trajectory}

    return r, info

def cmin(fg, x, cg, c0=None, stol=STOL, gtol=GTOL, ctol=CTOL, \
        maxit=MAXIT, maxstep=MAXSTEP, alpha=70.0, hess="LBFGS", callback=None):
    """
    Search  for a  minimum of  fg(x)[0] using  the  gradients fg(x)[1]
    subject to constrains cg(x)[0] = const.

    TODO: dynamic trust radius, line search in QN direction (?)

    Parameters:

    fg: objective function x -> (f, fprime)

        returns the value f and the gradient g at x

    cg: (differentiable) constrains x -> (c, cprime)

        returns the vector of constrais and their derivatives wrt x

    maxstep: float

        How far is  a single atom allowed to move.  This is useful for
        DFT calculations  where wavefunctions  can be reused  if steps
        are small.  Default is 0.04 Angstrom.

    alpha: float

        Initial guess for the Hessian (curvature of energy surface). A
        conservative  value of  70.0  is the  default,  but number  of
        needed steps  to converge  might be less  if a lower  value is
        used. However, a lower value also means risk of instability.

    hess: "LBFGS" or "BFGS"

        A name  of the class implementing hessian  update scheme.  Has
        to support |update| and |inv| methods.
        """

    # Interpret a string as a constructor name:
    hess = get_by_name(hess)

    # Returns the default hessian:
    hessian = hess(alpha)

    # Shurtcut for linear operator g -> hessian.inv(g):
    H = hessian.inv

    # Geometry, energy and the gradient from previous iteration:
    r0 = None
    e0 = None                   # not used anywhere!
    g0 = None

    # Initial value for the variable:
    r = array(x)                # we are going to modify it!

    # Invoke objective function, also computes the gradient:
    e, g = fg(r)

    # Evaluate constrains at current geometry:
    c, A = cg(r)

    # Save  the  initial  value  of the  constrains  (not  necessarily
    # zeros):
    if c0 is None:
        c0 = c
    else:
        assert len(c) == len(c0)
        c0 = asarray(c0)

    # Should have more variables than constrains:
    assert len(r) > len(c0)

    if VERBOSE:
        print ("cmin: c0=", c0, "(target value of constrain)")

    iteration = -1        # prefer to increment at the top of the loop
    converged = False

    while not converged and iteration < maxit:
        iteration += 1
        # Prefix for debug output:
        pfx = "cmin: (%03d) " % iteration

        # Update the hessian representation:
        if iteration > 0:       # only then r0 and g0 are meaningfull!
            hessian.update(r-r0, g-g0)

        # Compute the constrained step:
        dr, dg, lam = qnstep(g, H, c - c0, A)

        if VERBOSE:
            if e0 is not None:
                print (pfx, "e - e0=", e - e0)
            print (pfx, "e=", e)
            print (pfx, "criteria=", max(abs(dr)), max(abs(c - c0)), max(abs(g + dot(lam, A))))
        if VERBOSE > 1:
            print (pfx, "r=", r)
            print (pfx, "g=", g)
            print (pfx, "..",     dot(lam, A), "(    dot(lam, A))")
            print (pfx, "..", g + dot(lam, A), "(g + dot(lam, A))")
            print (pfx, "c=", c)

        # Check convergence, if any:
        criteria = 0

        if max(abs(c - c0)) < ctol:
            if VERBOSE:
                print (pfx, "converged by constraint max(abs(c - c0))=", max(abs(c - c0)), '<', ctol)
            criteria += 1

        if max(abs(dr)) < stol:
            if VERBOSE:
                print (pfx, "converged by step max(abs(dr))=", max(abs(dr)), '<', stol)
            criteria += 1

        # purified gradient for CURRENT geometry:
        if max(abs(g + dot(lam, A)))  < gtol:
            # FIXME: this may change after update step!
            if VERBOSE:
                print (pfx, "converged by force max(abs(g + dot(lam, A)))=", max(abs(g + dot(lam, A))), '<', gtol)
            criteria += 1

        if criteria >= 3:
            converged = True

        # Restrict the maximum component of the step:
        longest = max(abs(dr))
        if longest > maxstep:
            if VERBOSE:
                print (pfx, "dr=", dr, "(too long, scaling down)")
            dr *= maxstep / longest
            # NOTE: step  restriciton also does  not allow to  fix the
            #       mismatch (c-c0) in constrains in one shot ...

        if VERBOSE > 1:
            print (pfx, "dr=", dr)
        if VERBOSE:
            print (pfx, "dot(A, dr)=", dot(A, dr))
            print (pfx, "dot(g, dr)=", dot(g, dr))

        # Save for later comparison, need a copy, see "r += dr" below:
        r0 = r.copy()

        # "e, g = fg(r)" will re-bind (e, g), not modify them:
        e0 = e                  # not used anywhere!
        g0 = g

        # Actually update the variable:
        r += dr

        # Invoke objective function, also computes the gradient:
        e, g = fg(r)

        # FIXME: evaluate constrains at current geometry (another time?):
        c, A = cg(r)

        # If requested, provide feedback on the optimization progress:
        if callback is not None:
            callback(r, e, g, c, A)

        if VERBOSE:
            if iteration >= maxit:
                print (pfx, "exceeded number of iterations", maxit)
            # see while loop condition ...

    # Also return number of  interations, convergence status, and last
    # values of the gradient and step:
    # Also return number of  interations, convergence status, and last
    # values of the gradient and step:
    info = { "converged": converged,
             "iterations": iteration,
             "value": e,
             "derivative": g,
             "step": dr}

    # return r, e, (iteration, converged, g, dr)
    return r, info

def qnstep(g0, H, c, A):
    """
    At  point |x0| we  have the  gradient |g0|,  the quadratic  PES is
    characterized  by  inverse  hessian  H  that  relates  changes  in
    coordinate and  gradients: dr =  H(dg). H(g) is a  linear operator
    implemented as a function hiding the actual implementation.

    As we operate with inverse hessian, so g is the primary variable:

        g -> x -> c(x), A(x) = dc / dx

    As this is  a constrained minimization we are  not looking for the
    point  where the  gradients  vanish. Instead  seek  such (a  point
    where)

        g1 + lam * A = 0

    i.e.   where  energy   gradients  and   constrain   gradients  are
    "collinear".

    The following holds exactly on quadratic surface:

        x1 - x0 = H * (g1 - g0)

    We also want for the constrain to hold at x1:

        c(x1) = C

    Formally one  has to solve  the non-linear equations for  (g1, x1,
    lam), the non-linearity is  due to x-dependence of constrains c(x)
    and  A(x). This  sub proposes  a  step of  a single  Newton-Rapson
    iteration   for  this  system   of  non-linear   equations.   More
    specifically, we  first solve for  g1 in linear  approximation for
    c(x):

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

    # Current mismatch in constrains: c == c(x0) - C

    #
    # Note that A[i, j] here is dc_i / dx_j, the literature may differ
    # by transposition.
    #

    # This would be the unconstrained step:
    dx = - H(g0)

    # This would be the new values of the constrains:
    rhs = c + dot(A, dx)

    if VERBOSE > 1:
        print ("qnstep: A=", A)
        print ("qnstep: dx=", dx)
        print ("qnstep: c=", c)
        print ("qnstep: rhs=", rhs)

    # Construct the lhs-matrix AHA^T
    AHA = aha(A, H)

    # Solve linear equations:
    lam = solve(AHA, rhs)

    if VERBOSE > 1:
        print ("qnstep: rhs=", rhs)
        print ("qnstep: AHA=", AHA)
        print ("qnstep: lam=", lam)

    g1 = - dot(lam, A)

    dg = g1 - g0

    # dx, dg, lam:
    return H(dg), dg, lam

def aha(A, H):
    "Construct the lhs-matrix AHA^T"

    # Number of constrains:
    nc = len(A)

    AHA = empty((nc, nc))
    for j in range(nc): # FIXME: more efficient way?
        Haj = H(A[j])
        for i in range(nc):
            AHA[i, j] = dot(A[i], Haj)

    return AHA

def _flatten(fg, x):
    """
    Returns a funciton of  flat argument fg_(y) that properly reshapes
    y to x, and returns values and gradients as by fg.

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

    # In case we are given a list instead of array:
    x = asarray(x)

    # Shape of the actual argument:
    xshape = x.shape
    xsize  = x.size
    # print "xshape, xsize =", xshape, xsize

    # Define a flattened function based on original fg(x):
    def fg_(y):
        "Returns both, value and gradient, treats argument as flat array."

        # Need copy to avoid obscure error messages from fmin_l_bfgs_b:
        x = y.copy() # y is 1D

        # Restore the original shape:
        x.shape = xshape

        f, fprime = fg(x)       # fprime is returned as nD!

        # In case f is an array, preserve this structure:
        fshape = shape(f) # () for scalars

        # Still treat the arguments as 1D structure of xsize:
        return f, fprime.reshape( fshape + (xsize,) )

    # Return new funciton:
    return fg_

# python fopt.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
