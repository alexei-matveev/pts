#!/usr/bin/python

"""
"""

__all__ = []

from numpy import asarray, empty, ones, dot, max, abs, sqrt, shape, linspace
from numpy import vstack
# from numpy.linalg import solve #, eigh
from bfgs import LBFGS, BFGS, Array

VERBOSE = True

TOL = 1.e-6

STOL = TOL   # step size tolerance
GTOL = 1.e-5 # gradient tolerance
CTOL = TOL   # constrain tolerance

MAXITER = 50
MAXSTEP = 0.05

from chain import Spacing, Norm, Norm2
# spacing = Spacing(Norm())
spacing = Spacing(Norm2())

def tangent1(X):
    """For n geometries X[:] return n-2 tangents computed
    as averages of forward and backward tangents.
    """
    T = []
    for i in range(1, len(X) - 1):
        a = X[i] - X[i-1]
        b = X[i+1] - X[i]
        # FIXME: works only for 1D arrays a and b:
        a /= sqrt(dot(a, a))
        b /= sqrt(dot(b, b))
        t = a + b
        t /= sqrt(dot(t, t))
        T.append(t)

    return T

def tangent2(X):
    """For n geometries X[:] return n-2 tangents computed
    as central differences.
    """
    T = []
    for i in range(1, len(X) - 1):
        a = X[i-1]
        b = X[i+1]
        t = b - a
        # FIXME: works only for 1D array t:
        t /= sqrt(dot(t, t))
        T.append(t)

    return T

from path import Path

def tangent3(X, s=None):
    """For n geometries X[:] return n-2 tangents computed
    from a fresh spline interpolation.
    """
    if s is None:
        s = linspace(0., 1., len(X))

    p = Path(X, s)

    return map(p.tangent, s[1:-1])

from mueller_brown import show_chain

def test(A, B):

    print "A=", A
    print "B=", B

    p = Path([A, B])
    n = 7
    s = linspace(0., 1., n)
    print "s=", s

    x = map(p, s)
    x = asarray(x)

    print "x=", x
    show_chain(x)

    from mueller_brown import MB

#   OUTDATED: lambdas = mklambda1(mkconstr2(spacing, x[0], x[-1]))
    xm, stats = soptimize(MB, x, tangent1, spacing, maxiter=20, maxstep=0.1, callback=callback)
#   xm, stats = soptimize(MB, x, tangent1, maxiter=20, maxstep=0.1, callback=callback)
    show_chain(xm)

    print "xm=", xm

from numpy import savetxt, loadtxt

def callback(x):
    savetxt("path.txt", x)
    print "chain spacing=", spacing(x)
    # show_chain(x)

from func import Reshape, Elemental
from memoize import Memoize

def soptimize(pes, x0, tangent=tangent1, constraints=None, pmap=map, callback=callback, **kwargs):
    """
    Several choices for pmap argument to allow for parallelizm, e.g.:

        from paramap import pmap

    for parallelizm in general, or

        from qfunc import qmap

    for parallelizm and chdir-isolation of QM calculations.

    """

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

    tangents = wrap(tangent, x0[0], x0[-1])

    # by default apply constraints that only allow
    # displacement orthogonal to the tangents:
    if constraints is None:
        # make dynamic constran function with this definition
        # of tangents:
        constraints = mkconstr1(tangents)
    else:
        # real constraints require also the terminal beads:
        constraints = mkconstr2(constraints, x0[0], x0[-1])

    # prepare function that will compute the largangian
    # factors for this particular constran:
    lambdas = mklambda1(constraints)

    if callback is not None:
        callback = wrap(callback, x0[0], x0[-1])

    # for restarts and post-analysis:
    pes = Memoize(pes, filename="soptimize.pkl")

    xm, stats = sopt(pes.taylor, x0[1:-1], tangents, lambdas, callback=callback, **kwargs)

    # put the terminal images back:
    xm = vstack((x0[0], xm, x0[-1]))

    xm.shape = xshape
    x0.shape = xshape

    return xm, stats

def sopt(fg, X, tangents, lambdas=None, stol=STOL, gtol=GTOL, \
        maxiter=MAXITER, maxstep=MAXSTEP, alpha=70., callback=None):
    """
    |fg| is supposed to be an elemental function that returns a tuple

        fg(X) == (values, derivatives)
    
    with values and derivatives being the arrays/lists of results
    for all x in X. Note, that only the derivatives (gradients)
    are used for optimization.
    """

    # init array of hessians:
    H = Array([ BFGS(alpha) for _ in X ])

    # geometry, energy and the gradient from previous iteration:
    R0 = None
    G0 = None

    # initial value for the variable:
    R = asarray(X).copy() # we are going to modify it!

    iteration = -1 # prefer to increment at the top of the loop
    converged = False

    while not converged and iteration < maxiter:
        iteration += 1

        if VERBOSE:
            print "sopt: scheduling gradients for R="
            print R

        # compute energy and gradients at all R[i]:
        E, G = fg(R)

        # FIXME: better make sure gradients are returned as arrays:
        G = asarray(G)

        # purified gradient for CURRENT geometry, need tangents
        # just for convergency check:
        T = tangents(R)
        g1, g2 = projections(G, T)
        if max(abs(g2)) < gtol:
            # FIXME: this may change after update step!
            converged = True
            if VERBOSE:
                print "sopt: converged by force max(abs(g2)))", max(abs(g2)), '<', gtol

        if VERBOSE:
            print "sopt: obtained energies E=", asarray(E)
            print "sopt: obtained gradients G="
            print G
            print "sopt: g(para)=", g1
            print "sopt: g(ortho norms)=", asarray([sqrt(dot(g, g)) for g in g2])
            print "sopt: g(ortho)="
            print g2

        # update the hessian representation:
        if iteration > 0: # only then R0 and G0 are meaningfull!
            H.update(R - R0, G - G0)

        # first rough estimate of the step:
        dR = onestep(1.0, G, H, R, tangents, lambdas)

        # FIXME: does it hold in general?
        if lambdas is None:
            # assume positive hessian, H > 0
            assert Dot(G, dR) < 0.0

        # estimate the scaling factor for the step:
        h = 1.0
        if max(abs(dR)) > maxstep:
            h *= 0.9 * maxstep / max(abs(dR))

        if VERBOSE:
            print "sopt: ODE one step, propose h=", h
            print "sopt: dR=", dR

        # choose a step length below maxstep:
        while True:
            # FIXME: potentially wasting ODE integrations here:
            dR = odestep(h, G, H, R, tangents, lambdas)

            longest = max(abs(dR))
            if longest <= maxstep:
                break

            print "sopt: WARNING: step too long by factor", longest/maxstep, "retrying"
            h *= 0.9 * maxstep / longest

        if VERBOSE:
            print "ODE step, h=", h, ":"
            print "sopt: dR=", dR

        # check convergence, if any:
        if max(abs(dR)) < stol:
            converged = True
            if VERBOSE:
                print "sopt: converged by step max(abs(dR))=", max(abs(dR)), '<', stol

        # restrict the maximum component of the step:
        longest = max(abs(dR))
        if longest > maxstep:
            print "sopt: WARNING: step too long by factor", longest/maxstep, ", scale down !!!"

        # save for later comparison, need a copy, see "r += dr" below:
        R0 = R.copy()

        # "e, g = fg(r)" will re-bind (e, g), not modify them:
        G0 = G

        # actually update the variable:
        R += dR

        if callback is not None:
            callback(R)

        if VERBOSE:
            if iteration >= maxiter:
                print "sopt: exceeded number of iterations", maxiter
            # see while loop condition ...

    # also return number of interations, convergence status, and last values
    # of the gradient and step:
    return R, (iteration, converged, G, dR)

def Dot(A, B):
    "Compute dot(A, B) for a string"

    return sum([ dot(a, b) for a, b in zip(A, B) ])

def projections(V, T):
    """Decompose vectors V into parallel and orthogonal components
    using the tangents T.
    """

    V1 = empty(len(V))
    V2 = empty(shape(V))

    for i in xrange(len(V)):
        v, t = V[i], T[i]

        # parallel component:
        V1[i] = dot(t, v)

        # orthogonal component:
        V2[i] = v - t * V1[i]

    return V1, V2

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

def gprime(h, g, H, G0, X0, tangents, lambdas):
    """For the descent procedure return

      dg / dh = - (1 - t(g) * t'(g)) * g  + t(g) * lambda(g)

    The Lagrange contribution parallel to the tangent t(g)
    is added if the function "lambdas()" is provided.

    Procedure uses the one-to-one relation between
    gradients and coordinates

      (x - x0) = H * (g - g0)

    to compute the tangents:

      t(x) = t(x(g))

    This is NOT the traditional steepest descent
    where one has instead:

      dx / dh = - (1 - t(x) * t'(x)) * g(x)

    The current form of gprime() translated to real space variables

        dx / dh = H * dg / dh

    may be used to either ensure orthogonality of dx / dh to the tangents
    or preserve the image spacing depending on the definition of
    function "lambdas()" that delivers lagrangian factors.
    I am afraid one cannot satisfy both.

    NOTE: imaginary time variable "h" is not used anywhere.
    """

    # X = X(G):
    X = X0 + H.inv(g - G0)

    # T = T(X(G)):
    T = tangents(X)

    # parallel and orthogonal components of g:
    G1, G2 = projections(g, T)

    if lambdas is not None:
        # compute lagrange factors:
        lam = lambdas(X, G1, G2, H, T)

        # add lagrange forces, only components parallel
        # to the tangents is affected:
        for i in xrange(len(G2)):
            G2[i] -= lam[i] * T[i]

    return -G2

def mklambda1(constr):
    """Returns a function that generates lagrangian "forces" due
    to the differentiable constrain from

        (X, G1, G2, H, T)

    (geometry, tangential and complimentary components of the gradient,
    hessian, and tangents)
    """

    def _lambdas(X, G1, G2, H, T):
        # evaluate constraints and their derivatives:
        c, A = constr(X)
        # print "_lambdas: c=", c

        # for historical reasons the main code is here:
        return glambda(G2, H, T, A)

    return _lambdas

def glambda(G, H, T, A):
    """Compute Lagrange multipliers to compensate for the constrain violation
    that would occur if the motion would proceed along

        dx / dh = - H * g.

    Lagrange multipliers "lam" are supposed to be used to add contributions
    PARALLEL to the tangents and thus, redefine the default direction:

        g := g - lam * T   (no sum over path point index i)
         i    i     i   i
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

def wrap(tangents, A, B):
    """A decorator for the tangent function tangents(X) that appends
    terminal points A and B before calling it.
    """
    def _tangents(X):
        Y = vstack((A, X, B))
        return tangents(Y)
    return _tangents

def mkconstr1(tangents):
    """Returns dynamic costraint to restrict motion to the planes
    orthogonal to the tangents.
    """

    def _constr(X):

        T = tangents(X)

        n = len(T)

        A = zeros((n,) + shape(T))
        # these constraints are local, A[i, j] == 0 if i /= j:
        for i in xrange(n):
            A[i, i, :] = T[i]

        # these constraints have no meaningful values,
        # only "derivatives":
        return [None]*len(T), A

    return _constr

def mkconstr2(spacing, A, B):
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

    from aof.tools.jmol import jmol_view_path
    jmol_view_path(xm, syms=["Ar"]*4, refine=5)

# python fopt.py [-v]:
if __name__ == "__main__":
    # import doctest
    # doctest.testmod()
    test1()
    exit()
    from mueller_brown import CHAIN_OF_STATES as P
    test(P[0], P[4])

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
