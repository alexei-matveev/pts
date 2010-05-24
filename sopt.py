#!/usr/bin/python

"""
"""

__all__ = ["minimize", "cminimize"]

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

def test(A, B):

    print "A=", A
    print "B=", B

    from path import Path

    p = Path([A, B])
    n = 7
    s = linspace(0., 1., n)
    print "s=", s

    X = map(p, s)
    X = asarray(X)

    print "X=", X
    from mueller_brown import show_chain
    show_chain(X)

    from mueller_brown import gradient
    def grad(X):
        return map(gradient, X)

    def tang(X):
        AXB = vstack((A, X, B))
        p = Path(AXB)

        return map(p.tangent, s[1:-1])

    def tang1(X):
        AXB = vstack((A, X, B))
        # p = Path(AXB)
        T = []
        for i in range(1, 1 + len(X)):
            a = AXB[i-1]
            b = AXB[i+1]
            t = b - a
            t /= sqrt(dot(t, t))
            T.append(t)

        return T

    def tang2(X):
        AXB = vstack((A, X, B))
        # p = Path(AXB)
        T = []
        for i in range(1, 1 + len(X)):
            a = AXB[i] - AXB[i-1]
            b = AXB[i+1] - AXB[i]
            a /= sqrt(dot(a, a))
            b /= sqrt(dot(b, b))
            t = a + b
            t /= sqrt(dot(t, t))
            T.append(t)

        return T

    from chain import Spacing
    spc = Spacing()

    def callback(X):
        Y = vstack((A, X, B))
        print "chain spacing=", spc(Y)
        show_chain(Y)

#   import pylab
#   pylab.ion()

    XM, stats = sopt(grad, X[1:-1], tang2, maxiter=20, maxstep=0.05, callback=callback)
    XM = vstack((A, XM, B))
    show_chain(XM)

    print "XM=", XM


def sopt(grad, X, tang, stol=STOL, gtol=GTOL, \
        maxiter=MAXITER, maxstep=MAXSTEP, alpha=70., callback=None):
    """
    """

    # init array of hessians:
    H = Array([ BFGS(alpha) for x in X ])

    # geometry, energy and the gradient from previous iteration:
    R0 = None
    G0 = None

    # initial value for the variable:
    R = asarray(X).copy() # we are going to modify it!

    iteration = -1 # prefer to increment at the top of the loop
    converged = False

    while not converged and iteration < maxiter:
        iteration += 1

        # compute the gradients at all R[i]:
        G = grad(R)

        # FIXME: better make sure grad() returns arrays:
        G = asarray(G)

        # update the hessian representation:
        if iteration > 0: # only then R0 and G0 are meaningfull!
            H.update(R - R0, G - G0)

        # evaluate tangents at all R[i]:
        T = tang(R)

        if VERBOSE:
            print "sopt: R=", R
            print "sopt: G=", G, "(raw)"
            print "sopt: T=", T

        # tangents expected to be normalized:
        for t in T:
            assert abs(dot(t, t) - 1.) < 1.e-7

        # first rough estimate of the step:
        dR, LAM = qnstep(G, H, T)

        if VERBOSE:
            print "QN step (full):"
            print "sopt: dR=", dR

        # assume positive hessian, H > 0
        assert Dot(G, dR) < 0.0

        # estimate the scaling factor for the step:
        h = 1.0
        if max(abs(dR)) > maxstep:
            h = 0.9 * maxstep / max(abs(dR))

        if VERBOSE:
            print "QN step, h=", h,":"
            print "sopt: dR=", dR * h

        dR = rk5step(h, G, H, R, tang)

        if VERBOSE:
            print "RK5 step, h=", h, ":"
            print "sopt: dR=", dR
            print "sopt: LAM=", LAM

        dR1 = odestep(h, G, H, R, tang)

        if VERBOSE:
            print "ODE step, h=", h, ":"
            print "sopt: dR=", dR1

        dR = dR1

        # check convergence, if any:
        if max(abs(dR)) < stol:
            if VERBOSE:
                print "sopt: converged by step max(abs(dR))=", max(abs(dR)), '<', stol
            converged = True

        # purified gradient for CURRENT geometry:
        if max(abs(G - dot(LAM, T))) < gtol:
            # FIXME: this may change after update step!
            if VERBOSE:
                print "cmin: converged by force max(abs(G - dot(LAM, T)))", max(abs(G - dot(LAM, T))), '<', gtol
            converged = True

        # restrict the maximum component of the step:
        longest = max(abs(dR))
        if longest > maxstep:
            if VERBOSE:
                print "sopt: dR=", dR, "(TOO LONG, SCALE DOWN !!!)"
#               print "sopt: dR=", dR, "(TOO LONG, SCALING DOWN)"
#           dR *= maxstep / longest

        if VERBOSE:
            print "sopt: dR=", dR

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

def proj(V, T):
    """Decompose vectors V into parallel and orthogonal components
    using the tangents T.
    """

    V1 = empty(shape(V)[0]) # (len(V))
    V2 = empty(shape(V))

    for i in xrange(len(V)):
        v, t = V[i], T[i]

        # parallel component:
        V1[i] = dot(t, v)

        # orthogonal component:
        V2[i] = v - t * V1[i]

    return V1, V2

def qnstep(G, H, T):
    """QN-Step in the subspace orthogonal to tangents T:

        dr = - ( 1 - t * t' ) * H * ( 1 - t * t' ) * g
    """

    # parallel and orthogonal components of the gradient:
    G1, G2 = proj(G, T)

    # step that would make the gradients vanish:
    R = - H.inv(G2)

    # parallel and orthogonal components of the step:
    R1, R2 = proj(R, T)

    return R2, G1

from ode import rk5

def rk5step(h, G, H, R, tang):

    def f(t, x):
        dx, lam = qnstep(G, H, tang(x))
        return dx

    return rk5(0.0, R, f, h)

from ode import odeint1
from numpy import log

def odestep(h, G, H, X, tang):
    #
    # Function to integrate (t is "time", not "tangent"):
    #
    #   dg / dt = f(t, g)
    #
    def f(t, g):
        return gprime(t, g, H, G, X, tang)

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
    print "odestep: h=", h, "T=", T

    #
    # Integrate to T (or to infinity):
    #
    G8 = odeint1(0.0, G, f, T)
    # G8 = G + 1.0 * gp(0.0, G)

    # print "odestep: G8=", G8

    # use one-to-one relation between dx and dg:
    dX = H.inv(G8 - G)

    if VERBOSE:
        X8 = X + dX
        T8 = tang(X8) # FIXME: tang(X8)!
        G1, G2 = proj(G8, T8)
        print "odestep: G1=", G1
        print "odestep: G2=", G2

    return dX

def gprime(h, g, H, G0, X0, tang):
    """
    For the descent procedure return

      dg / dh = - (1 - t(g) * t'(g)) * g

    uses the one-to-one relation between
    gradients and coordinates

      dx = H * dg

    to compute the tangents:

      t(x) = t(x(g))

    This is NOT the traditional steepest descent
    where one has instead:

      dx / dh = - (1 - t(x) * t'(x)) * g(x)

    FIXME: the current form of dg / dh translated to real space
    variables

        dx / dh = H * dg / dh

    neither ensures orthogonality to the tangents or preserves image
    spacing.
    """

    # X = X(G):
    x = X0 + H.inv(g - G0)

    # T = T(X(G)):
    t = tang(x)

    # parallel and orthogonal components of G:
    g1, g2 = proj(g, t)

    return -g2

# python fopt.py [-v]:
if __name__ == "__main__":
    # import doctest
    # doctest.testmod()
    from mueller_brown import CHAIN_OF_STATES as P
    test(P[0], P[4])

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
