#!/usr/bin/env python

"""
Example use: define a few nodes, here four nodes in 2D, the same as in
path.py:

    >>> from numpy import asarray, arctan2, pi
    >>> nodes = asarray(((-100., -100.), (0., -50.), (0., 50.), (100., 100.)))

                                           + + B
                                       + +     (100, 100)
                                   + +
                               + +
                           + +
                         +
                        X
                       +(0, 50)
                       +
                       +
                        +
                        (0, 0)
                          +
                          +
                          +
                        X
                      + (0, -50)
                  + +
              + +
          + +
    A + +
    (-100, -100)

This  illustrates "arbitrarness"  in the  definition of  the tangents,
basically there is  more than one way to plot  a "reasonable" path and
the corresponding tangents:

    >>> asarray(tangent1(nodes))
    array([[ 0.89442719,  1.4472136 ],
           [ 0.89442719,  1.4472136 ]])

This is the angle in grads:

    >>> arctan2(1.4472136, 0.89442719) / pi * 180
    58.282525696859388

    >>> asarray(tangent2(nodes))
    array([[ 100.,  150.],
           [ 100.,  150.]])

    >>> arctan2(150., 100.) / pi * 180
    56.309932474020215

    >>> asarray(tangent3(nodes))
    array([[  50.,  275.],
           [  50.,  275.]])

    >>> arctan2(275., 50.) / pi * 180
    79.69515353123397

    >>> asarray(tangent4(nodes))
    array([[  42.2291236 ,  297.50776405],
           [  42.2291236 ,  297.50776405]])

    >>> arctan2(297.50776405, 42.2291236) / pi * 180
    81.921237154537607
"""

__all__ = []

from numpy import array, asarray, empty, ones, dot, max, abs, sqrt, shape, linspace
from numpy import vstack
from bfgs import LBFGS, BFGS, Array
from common import cumm_sum, pythag_seps
from metric import cartesian_norm

VERBOSE = False

TOL = 1.e-6

XTOL = TOL   # step size tolerance
FTOL = 1.e-5 # gradient tolerance
CTOL = TOL   # constrain tolerance

MAXIT = 50
MAXSTEP = 0.05
SPRING = 100.0

from chain import Spacing, Norm, Norm2
# spacing = Spacing(Norm())
spacing = Spacing(Norm2())

def tangent1(X, norm=cartesian_norm):
    """For n geometries X[:] return n-2 tangents computed
    as averages of forward and backward tangents.
    """

    T = []
    for i in range(1, len(X) - 1):
        a = X[i] - X[i-1]
        b = X[i+1] - X[i]

        a /= norm(a, X[i]) # FIXME: (X[i] + X[i-1]) / 2.0 ?
        b /= norm(b, X[i]) # FIXME: (X[i] + X[i+1]) / 2.0 ?

        T.append(a + b)

    return T

def tangent2(X):
    """For n geometries X[:] return n-2 tangents computed
    as central differences.
    """

    T = []
    for i in range(1, len(X) - 1):
        T.append(X[i+1] - X[i-1])

    return T

from path import Path

def tangent3(X):
    """For n geometries X[:] return n-2 tangents computed from a fresh
    spline interpolation.
    """

    s = linspace(0., 1., len(X))

    p = Path(X, s)

    return map(p.fprime, s[1:-1])

def tangent4(X):
    """For  n geometries  X[:]  return n-2  tangents  computed from  a
    different spline interpolation.
    """

    # abscissas of the nodes:
    s = cumm_sum(pythag_seps(X))
    s = s / s[-1]

    assert s[0] == 0.0
    assert s[-1] == 1.0

    p = Path(X, s)

    return map(p.fprime, s[1:-1])

def test(A, B, trafo=None):

    print "A=", A
    print "B=", B

    from pts.pes.mueller_brown import MB
    from pts.pes.mueller_brown import show_chain

    x = [A, B]

    # change coordinates:
    if trafo is not None:
        from func import compose
        MB = compose(MB, trafo)
        x = array(map(trafo.pinv, x))

    def show(x):
        if trafo is not None:
            show_chain(map(trafo, x))
        else:
            show_chain(x)

    from numpy import savetxt, loadtxt

    def callback(x):
        # show(x)
        # savetxt("path.txt", x)
        # print "chain spacing=", spacing(x)
        pass

    from path import MetricPath # for respacing
    from rc import Linear # as reaction coordinate
    rcoord = Linear([1., -1.])

    from metric import Metric
    mt = Metric(rcoord)

    n = 3
    n_max = 30
    while True:
        #
        # Respace vertices based on custom metric built from the
        # definition of reaction coordinate:
        #
        p = MetricPath(x, mt.norm_up)
        x = array(map(p, linspace(0., 1., n)))

        print "BEFORE, rc(x)=", map(rcoord, x)
        show(x)

        # x = respace(x, tangent4, spacing)

        # print "RESPACE, x=", x
        # print "spacing(x)=", spacing(x)
        # show(x)

#       x, stats = soptimize(MB, x, tangent1, spacing, maxit=20, maxstep=0.1, callback=callback)
#       x, stats = soptimize(MB, x, tangent4, maxit=20, maxstep=0.1, callback=callback)
        x, stats = soptimize(MB, x, tangent4, rc=rcoord, maxit=20, maxstep=0.1, callback=callback)
        savetxt("mb-path.txt-" + str(len(x)), x)

        print "AFTER, rc(x)=", map(rcoord, x)
        show(x)

        if n < n_max:
            # double the number of beads:
            n = 2 * n + 1
        else:
            print "========================================================="
            print "Conveged for the maximal tested number of beads: ", n
            print "========================================================="
            break

from func import Reshape, Elemental
from memoize import Memoize

def soptimize(pes, x0, tangent=tangent1, rc=None, constraints=None, pmap=map, callback=None, **kwargs):
    """
    Several choices for pmap argument to allow for parallelizm, e.g.:

        from paramap import pmap

    for parallelizm in general, or

        from qfunc import qmap

    for parallelizm and chdir-isolation of QM calculations.

    """

    if rc is not None and constraints is not None:
        assert False, "Provide either reaction coordinate, local, or collective constraints."

    n = len(x0)
    assert n >= 3

    x0 = asarray(x0)

    # string (long) vector shape:
    xshape = x0.shape

    # position (short) vector shape:
    vshape = x0[0].shape

    # position (short) vector dimension:
    vsize  = x0[0].size

    #
    # sopt() driver deals only with the collections of 1D vectors, so
    #
    # (1) reshape the input ...
    x0.shape = (n, vsize)
    # FIXME: dont forget to restore the shape of input-only x0 on exit!

    # (2) "reshape" provided PES Func to make it accept 1D arrays:
    pes = Reshape(pes, xshape=vshape)

    # (3) if present, "reshape" the constraint Func:
    if constraints is not None:
        # here we assume n-2 constraints, one per moving image,
        # the terminal images are fix anyway:
        constraints = Reshape(constraints, xshape=xshape, fshape=(n-2,))

    # (4) make PES function elemental, allow for parallelizm:
    pes = Elemental(pes, pmap)
    # FIXME: should we request the caller to provide "elemental" PES?

    # FIXME: available tangent definitions already expect/accept
    # (a group of) 1D vectors. Reshape them too, if needed.

    tangents = wrap1(tangent, x0[0], x0[-1])

    #
    # The function "lambdas" is used to compute lagrange multipliers and is
    # expected to adhere to specific interface.
    #
    # By default Lagrange multipliers ensure displacements are orthogonal to
    # tangents (lambda1):
    #
    lambdas = lambda1

    #
    # If global  (collective) constraints  are provided, use  those to
    # define "lambdas".  So far Collective constraints is a length N-2
    # vector valued function of all N vertices, including the terminal
    # ones.   This  makes   them  suitable   for  solving   a  typical
    # optimization problem when the terminal beads are frozen.
    #
    if constraints is not None:
        # real constraints require also the terminal beads:
        constraints = wrap2(constraints, x0[0], x0[-1])

        # prepare function that will compute the largangian
        # factors for this particular constraint:
        lambdas = mklambda3(constraints)

    #
    # If local constraints are provided, use them to define lambdas. A
    # local  constraint is  a  (differentiable) function  of a  single
    # vertex. We need at least one such function per vertex. Currently
    # it is  assumed that  the terminal vertices  are frozen,  so that
    # their local constraints are ignored.
    #
    if rc is not None:
        # (2) "reshape" provided PES Func to make it accept 1D arrays:
        rc = Reshape(rc, xshape=vshape)

        local = [rc.taylor] * len(x0)

        lambdas = mklambda0(local[1:-1])

    if callback is not None:
        callback = wrap1(callback, x0[0], x0[-1])

    # for restarts and post-analysis:
    pes = Memoize(pes, filename="soptimize.pkl")

    xm, stats = sopt(pes.taylor, x0[1:-1], tangents, lambdas, callback=callback, **kwargs)

    # put the terminal images back:
    xm = vstack((x0[0], xm, x0[-1]))

    xm.shape = xshape
    x0.shape = xshape

    return xm, stats

def sopt(fg, X, tangents, lambdas=None, xtol=XTOL, ftol=FTOL,
         maxit=MAXIT, maxstep=MAXSTEP, alpha=70., callback=None,
         **kwargs):
    """
    |fg| is supposed to be an elemental function that returns a tuple

        fg(X) == (values, derivatives)

    with values and derivatives  being the arrays/lists of results for
    all x in  X. Note, that only the  derivatives (gradients) are used
    for optimization.

    Kwargs accomodates all arguments  that are not interpreted by this
    sub but are otherwise used in other optimizers.
    """

    if VERBOSE:
        print "sopt: xtol=", xtol, "ftol=", ftol, "maxit=", maxit, "maxstep=", maxstep
        if len(kwargs) > 0:
            print "sopt: ignored kwargs=", kwargs
    # init array of hessians:
    H = Array([ BFGS(alpha) for _ in X ])

    # geometry, energy and the gradient from previous iteration:
    R0 = None
    G0 = None

    # initial value for the variable:
    R = array(X) # we are going to modify it!

    iteration = -1 # prefer to increment at the top of the loop
    converged = False

    # fixed trust radius:
    TR = maxstep

    while not converged and iteration < maxit:
        iteration += 1

        # To be  able to distinguish  convergence by step, or  by gradient
        # count the number of satisfied criteria:
        criteria = 0

        if VERBOSE:
            print "sopt: =============== Iteration ", iteration, " ==============="
            print "sopt: scheduling gradients for R="
            print R

        # compute energy and gradients at all R[i]:
        E, G = fg(R)

        # FIXME: better make sure gradients are returned as arrays:
        G = asarray(G)

        #
        # Need tangents just for convergency check and RPH:
        #
        T = tangents(R)

        #
        # Update the hessian representation:
        #
        if iteration > 0: # only then R0 and G0 are meaningfull!
            H.update(R - R0, G - G0)

        #
        # Convergency checking, based on gradients (ftol) ... {{{
        #

        # lagrange multipliers (in simple case components of gradients parallel
        # to the tangents):
        LAM = lambdas(R, G, H, T)

        # remaining gradient for CURRENT geometry just for convergency check:
        G2 = empty(shape(G)) # move out of the loop
        for i in xrange(len(G)):
            G2[i] = G[i] - LAM[i] * T[i]

        if max(abs(G2)) < ftol:
            # FIXME: this may change after update step!
            criteria += 1
            if VERBOSE:
                print "sopt: converged by force max(abs(G2)))", max(abs(G2)), '<', ftol

        if VERBOSE:
            print "sopt: obtained energies E=", asarray(E)
            print "sopt: obtained gradients G="
            print G
            print "sopt: g(para)=", LAM, "(lambdas)"
            print "sopt: g(ortho norms)=", asarray([sqrt(dot(g, g)) for g in G2])
            print "sopt: g(ortho)="
            print G2

        # These were  used for convergency  check, used below  only to
        # report  additional info  upon  convergence:
        del G2 # T, LAM
        # ... done convergency check }}}

        # first rough estimate of the step:
        dR = onestep(1.0, G, H, R, tangents, lambdas)

        # FIXME: does it hold in general?
        if False:
            # assume positive hessian, H > 0
            assert Dot(G, dR) < 0.0

        # estimate the scaling factor for the step:
        h = 1.0
        if max(abs(dR)) > TR:
            h *= 0.9 * TR / max(abs(dR))

        if VERBOSE:
            print "sopt: ODE one step, propose h=", h
            print "sopt: dR=", dR

        # choose a step length below TR:
        while True:
            # FIXME: potentially wasting ODE integrations here:
            dR = odestep(h, G, H, R, tangents, lambdas)

            longest = max(abs(dR))
            if longest <= TR:
                break

            print "sopt: WARNING: step too long by factor", longest/TR, "retrying"
            h *= 0.9 * TR / longest

        if VERBOSE:
            print "ODE step, h=", h, ":"
            print "sopt: dR=", dR

        #
        # Convergency check by step size (xtol) ...
        #
        if max(abs(dR)) < xtol:
            criteria += 1
            if VERBOSE:
                print "sopt: converged by step max(abs(dR))=", max(abs(dR)), '<', xtol

        # restrict the maximum component of the step:
        longest = max(abs(dR))
        if longest > TR:
            print "sopt: WARNING: step too long by factor", longest/TR, ", scale down !!!"

        # save for later comparison, need a copy, see "r += dr" below:
        R0 = R.copy()

        # "e, g = fg(r)" will re-bind (e, g), not modify them:
        G0 = G

        # actually update the variable:
        R += dR

        if callback is not None:
            callback(R)

        # See while  loop condition. We  are only converged  when both
        # criteria are satisfied:
        if criteria >= 2:
            converged = True

    #
    # We are outside of the loop again:
    #
    if VERBOSE:
        if iteration >= maxit and not converged:
            print "sopt: exceeded number of iterations", maxit

    #
    # Also return number of  interations, convergence status, and last
    # values of the energies, gradients  and step. Note again that the
    # energies,  gradients, tangents,  and lambdas  correspond  to the
    # last geometry used for convergency check.  The returned geometry
    # differs   from   that   by    dR.    You   need   to   recompute
    # energies/gradients  and such  if  you want  them  for the  final
    # geometry.  This is left to the caller.
    #
    info = {"iterations": iteration + 1,
            "converged": converged,
            "energies": E,
            "gradients": G,
            "tangents": T,
            "lambdas": LAM,
            "step": dR}

    return R, info

def respace(x0, tangents, spacing):

    assert len(x0) >= 3

    x0 = asarray(x0)

    # tangents need terminal beads:
    tangents = wrap1(tangents, x0[0], x0[-1])

    # spacing also requires terminal beads:
    lambdas = wrap1(spacing, x0[0], x0[-1])

    xm = resp(x0[1:-1], tangents, lambdas)

    # put the terminal images back:
    xm = vstack((x0[0], xm, x0[-1]))

    return xm

def resp(x, t, s):
    """
        dx / dh = - t(x) * s(x)
    """

    def f(h, y):
        T = t(y)
        S = s(y)
        yprime = empty(shape(y))
        for i in range(len(y)):
            # FIXME: yes, with PLUS sign:
            yprime[i] = T[i] * S[i]
        return yprime

    print "s(x0)=", s(x)

    x8 = odeint1(0.0, x, f)

    print "s(x8)=", s(x8)
    return x8

def Dot(A, B):
    "Compute dot(A, B) for a string"

    return sum([ dot(a, b) for a, b in zip(A, B) ])

from ode import odeint1
from numpy import log, min, zeros

def odestep(h, G, H, X, tangents, lambdas):
    #
    # Function to integrate (t is "time", not "tangent"):
    #
    #   dg / dt = f(t, g)
    #
    def f(t, g):
        return gprime(t, g, H, G, X, tangents, lambdas)

    #
    # Upper integration limit T (again "time", not "tangent)":
    #
    if h < 1.0:
        #
        # Assymptotically the gradients decay as exp[-t]
        #
        T = - log(1.0 - h)
    else:
        T = None # FIXME: infinity

    if VERBOSE:
        print "odestep: h=", h, "T=", T

    #
    # Integrate to T (or to infinity):
    #
    G8 = odeint1(0.0, G, f, T)
    # G8 = G + 1.0 * f(0.0, G)

    # use one-to-one relation between dx and dg:
    dX = H.inv(G8 - G)

    return dX

def onestep(h, G, H, X, tangents, lambdas):
    """One step of the using the same direction

        dx / dh = H * gprime(...)

    as the ODE procedure above. Can be used for step length
    estimation.
    """

    # dg = h * dg / dh:
    dG = h * gprime(0.0, G, H, G, X, tangents, lambdas)

    # use one-to-one relation between dx and dg:
    dX = H.inv(dG)

    return dX

from numpy.linalg import solve

def gprime(h, G, H, G0, X0, tangents, lambdas):
    """For the descent procedure return

      dy / dh = - ( y  - lambda(y) * t(y) )

    The Lagrange contribution parallel to the tangent t(y) is added if
    the function "lambdas()" is provided.

    Procedure uses the one-to-one relation between coordinates

      (x - x0) = H * (y - y0)

    to compute the tangents at x:

      t(x) = t(x(y))

    By  now y  is the  same  as the  gradient G.  Though the  positive
    definite  hessian  H may  distrub  this  equivalence  at some  PES
    regions.

    The current form of gprime() translated to real space variables

        dx / dh = H * dg / dh

    may  be used  to either  ensure orthogonality  of dx  / dh  to the
    tangents or preserve the image spacing depending on the definition
    of function  "lambdas()" that  delivers lagrangian factors.   I am
    afraid one cannot satisfy both.

    NOTE: imaginary time variable "h" is not used anywhere.
    """

    # X = X(G):
    X = X0 + H.inv(G - G0)

    # T = T(X(G)):
    T = tangents(X)

    # compute lagrange factors:
    LAM = lambdas(X, G, H, T)

    # add lagrange forces, only components parallel
    # to the tangents is affected:
    dG = empty(shape(G))
    for i in xrange(len(G)):
        dG[i] = G[i] - LAM[i] * T[i]

    return -dG

def vectorize(f):
    """Array-version of f.  Used  to build LOCAL lagrange factors. For
    any  function   f()  opearting  on  a  few   arguments  return  an
    "elemental" function F, e.g.:

    def F(X, G, H, T):
        return map(f, X, G, H, T)

    FIXME: see if numpy.vectorize can be used
    """

    def _f(*args):
        return map(f, *args)

    return _f

def mklambda0(C):
    """
    Returns a  function that generates lagrangian "forces"  due to the
    differentiable constraints C from

        (X, G, H, T) == (geometry, gradient, hessian, tangents)

    Here constraints C is a list of local constraints, each a function
    of the corresponding vertex.   This type of constraints is usefull
    to enforce local properties of  vertices, such as a bond length or
    an  angle. Within  this  file it  is  assumed that  differentiable
    funcitons  are implemented  as functions  returning a  tuple  of a
    value and first derivatives.
    """

    def _lambda0(X, G, H, T):

        #
        # Evaluate  constraints  and  their derivatives.   Values  are
        # neglected   as  we  assume   the  vertices   satisfy  target
        # constraints  already.    This  is  more   like  "preserving"
        # properties  rather than  "seeking" target  values  (cp.  the
        # NEB-style springs in mklambda4):
        #
        A = [a for _, a in [c(x) for c, x in zip(C, X)]]

        # for historical reasons the main code is here:
        return map(lambda0, X, G, H, T, A)

    return _lambda0

def lambda0(x, g, h, t, a):
    """
    Compute  Lagrange  multiplier  to  compensate  for  the  constrain
    violation that would occur if the motion would proceed along

        dx = - h * g.

    In other words, find the lagrange factor lam such that

        dx = - h * (g - lam * t)

    is "orthogonal" to a

        a' * dx = 0.

    That is solve for lam in

        lam * (a' * h * t) = (a' * h * g).

    Input:
        x -- geometry
        g -- gradient
        h -- hessian
        t -- tangent
        a -- constraint derivatives
    """

    #
    # This would appliy the (inverse) hessian twice:
    #
    # lam = dot(a, h.inv(g)) / dot(a, h.inv(t))
    #

    # This applies hessian once:
    ha = h.inv(a)

    lam = dot(ha, g) / dot(ha, t)

    return lam

@vectorize
def lambda1(x, g, h, t):
    """Find the lagrange factor lam such that

        dx = - h * (g - lam * t)

    is orthogonal to t

        t' * dx = 0

    That is solve for lam in

        lam * (t' * h * t) = (t' * h * g)

    Input:
        x -- geometry
        g -- gradient
        h -- hessian
        t -- tangent

    FIXME: needs more general definition of "orthogonality"
    """

    #
    # The "constraint" to be satisfied is "no displacement along the
    # tangent". FIXME: with a general metric the second "t" would need
    # to be converted to covariant coordinates (aka lower index):
    #
    return lambda0(x, g, h, t, t)

@vectorize
def lambda2(x, g, h, t):
    """Find the lagrange factor lam such that

        g   =  g - lam * t
         bot

    is orthogonal to t

        t' * g   =   0
              bot

    That is solve for lam in

        lam * (t' * t) = (t' * g)

    Input:
        x -- geometry
        g -- gradient
        h -- hessian
        t -- tangent

    FIXME: needs more general definition of "orthogonality"
    """

    lam = dot(t, g) / dot(t, t)

    return lam

def mklambda3(constraints):
    """
    Returns a  function that generates lagrangian "forces"  due to the
    differentiable constraints from

        (X, G, H, T) == (geometry, gradient, hessian, tangents)

    Here "constraints" are global constraints, that is a vector valued
    differentiable function of all  vertices. This type of constraints
    is  usefull to  enforce  "collective" constraints,  such as  equal
    pairwise distances between neightboring vertices.
    """

    def lambda3(X, G, H, T):

        #
        # Evaluate  constraints  and  their derivatives.   Values  are
        # neglected   as  we  assume   the  vertices   satisfy  target
        # constraints  already.    This  is  more   like  "preserving"
        # properties  rather than  "seeking" target  values  (cp.  the
        # NEB-style springs in mklambda4):
        #
        c, A = constraints(X)

        # for historical reasons the main code is here:
        return glambda(G, H, T, A)

    return lambda3

def mklambda4(springs, k=SPRING):
    """Returns a function that generates tangential "forces" due
    to the springs from provided arguments:

        (X, G, H, T) == (geometry, gradient, hessian, tangents)
    """

    def lambda4(X, G, H, T):

        # these are parallel components of gradients, to be removed:
        lam2 = lambda2(X, G, H, T)

        #
        # Evaluate deviations from  equilibrium. Here, the derivatives
        # are  ignored, rather  the values  of the  "deformations", if
        # different from zero, are  used to add tangential "forces":
        #
        c, _ = springs(X)

        # instead add spring forces:
        lam4 = asarray(c) * k

#       print "LAM (2) =", lam2
#       print "LAM (4) =", lam4

        return lam2 + lam4

    return lambda4

def glambda(G, H, T, A):
    """
    Compute  Lagrange  multipliers  to  compensate for  the  constrain
    violation that would occur if the motion would proceed along

        dX = - H * G.

    Lagrange  multipliers   LAM  are  supposed  to  be   used  to  add
    contributions  PARALLEL to  the  tangents and  thus, redefine  the
    default direction:

        G := G - LAM * T   (no sum over path point index i)
         i    i     i   i

    This amounts to  solving the system of N equations  for LAM with N
    being the number of constraints (equal to the number of points)

        A * dX = 0,  k = 1..N
         k

    or, more explicitly

        A * H * ( G - LAM * T ) = 0
         k
    """

    # number of constraints:
    n = len(A)

    # FIXME: number of degrees of freedom same as number of
    #        constraints:
    assert len(T) == n

    # dx / dh without constraints would be this:
    xh = H.inv(-G)

    # dc / dh without constraints would be this:
    ch = zeros(n)
    for i in xrange(n):
        for j in xrange(n):
            ch[i] += dot(A[i, j], xh[j])
            # FIXME: place to use npz.matmul() here!

    # unit lagrangian force along the tangent j would change
    # the constraint i that fast:
    xt = H.inv(T)
    ct = zeros((n, n))
    for i in xrange(n):
        for j in xrange(n):
            ct[i, j] = dot(A[i, j], xt[j])

    # Lagrange factors to fullfill constains:
    lam = - solve(ct, ch) # linear equations
    # FIXME: we cannot compensate constraints if the matrix is singular!

    return lam

def wrap1(tangents, A, B):
    """A decorator for the tangent function tangents(X) that appends
    terminal points A and B before calling it.
    """
    def _tangents(X):
        Y = vstack((A, X, B))
        return tangents(Y)
    return _tangents

def wrap2(spacing, A, B):
    """Constrains on the chain of states based on state spacing.
    """

    def _constr(X):
        Y = vstack((A, X, B))

        # NOTE: spacing for N points returns N-2 results and its
        # N derivatives:
        c, cprime = spacing.taylor(Y)

        # return derivatives wrt moving beads:
        return c, cprime[:, 1:-1]

    return _constr

def test1():
    from numpy import array
    from ase import Atoms
    ar4 = Atoms("Ar4")


    from qfunc import QFunc
    pes = QFunc(ar4)

    # One equilibrium:

    w=0.39685026
    A = array([[ w,  w,  w],
               [-w, -w,  w],
               [ w, -w, -w],
               [-w,  w, -w]])

    # Another equilibrium (first two exchanged):

    B = array([[-w, -w,  w],
               [ w,  w,  w],
               [ w, -w, -w],
               [-w,  w, -w]])

    # Halfway between A and B (first two z-rotated by 90 deg):

    C = array([[-w,  w,  w],
               [ w, -w,  w],
               [ w, -w, -w],
               [-w,  w, -w]])


    xs = array([A, C, B])

    from path import Path
    p = Path(xs)

    from numpy import linspace
    x5 = [p(t) for t in linspace(0., 1., 5)]

    es0 = array([pes(x) for x in x5])

    print "energies=", es0
    print "spacing=", spacing(x5)

#   xm, stats = soptimize(pes, x5, tangent1, maxiter=20, maxstep=0.1, callback=callback)
    xm, stats = soptimize(pes, x5, tangent1, spacing, maxiter=50, maxstep=0.1, callback=callback)

    es1 = array([pes(x) for x in xm])

    print "energies=", es1
    print "spacing=", spacing(xm)

    from pts.tools.jmol import jmol_view_path
    jmol_view_path(xm, syms=["Ar"]*4, refine=5)

# python fopt.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()
#   test1()
#   exit()
    from pts.pes.mueller_brown import CHAIN_OF_STATES as P
    # from testfuns2 import mb2
    # trafo = mb2() #[[2.0, 0], [0.0, 0.5]])
    # test(P[0], P[4]) #, trafo)

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
