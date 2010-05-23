#!/usr/bin/python

"""
"""

__all__ = ["minimize", "cminimize"]

from numpy import asarray, empty, ones, dot, max, abs, sqrt, shape, linspace
from numpy import vstack
# from numpy.linalg import solve #, eigh
from bfgs import LBFGS, BFGS

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

    def callback(X):
        Y = vstack((A, X, B))
        show_chain(Y)

#   import pylab
#   pylab.ion()

    XM, stats = sopt(grad, X[1:-1], tang2, maxiter=10, maxstep=0.1, callback=callback)
    XM = vstack((A, XM, B))
    show_chain(XM)

    print "XM=", XM


from ode import rk5
def sopt(grad, X, tang, stol=STOL, gtol=GTOL, \
        maxiter=MAXITER, maxstep=MAXSTEP, alpha=70., callback=None):
    """
    """

    # init all hessians:
    H = [ LBFGS(alpha) for x in X ]

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

        # update the hessian representation:
        if iteration > 0: # only then R0 and G0 are meaningfull!
            for h, r, r0, g, g0 in zip(H, R, R0, G, G0):
                h.update(r - r0, g - g0)

        # evaluate tangents at all R[i]:
        T = tang(R)

        if VERBOSE:
            print "sopt: R=", R
            print "sopt: G=", G, "(raw)"
            print "sopt: T=", T

        # tangents expected to be normalized:
        for t in T:
            assert abs(dot(t, t) - 1.) < 1.e-7

        dR, LAM = qnstep(G, H, T)

        # restrict the maximum component of the step:
#       hs = ones(len(R))
#       for i in range(len(R)):
#           maxcomp = max(abs(dR[i]))
#           if maxcomp > maxstep:
#               hs[i] = 0.9 * maxstep / maxcomp

#       if True:
#           for dr, h in zip(dR, hs):
#               dr[:] *= h

#           T2 = tang(R + 0.5 * dR)

#           dR, LAM = qnstep(G, H, T2)

#           for dr, h in zip(dR, hs):
#               dr[:] *= h
#       else:
        def f(t, x, G, H):
            t = tang(x)
            dx, lam = qnstep(G, H, t)
            return dx

        h = 1.0
        if max(abs(dR)) > maxstep:
            h = 0.9 * maxstep / max(abs(dR))

        dR = rk5(0.0, R, f, h, args=(G, H))

        if VERBOSE:
            print "sopt: dR=", dR
            print "sopt: LAM=", LAM

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
                print "sopt: dR=", dR, "(too long, scaling down)"
            dR *= maxstep / longest

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


def qnstep(G, H, T):
    """QN-Step in the subspace orthogonal to tangents T:

        dr = ( 1 - t * t' ) * H * ( 1 - t * t' ) * dg
    """

    dR = empty(shape(G))
    LAM = empty(shape(G)[0])

    for i in xrange(len(G)):
        g, h, t = G[i], H[i], T[i]

        # gradient projection:
        lam = dot(t, g)

        LAM[i] = lam # for output

        # project gradient:
        g1 = g - t * lam

        # apply (inverse) hessian:
        dr = - h.inv(g1)

        # project step:
        dR[i] = dr - t * dot(t, dr)

    return dR, LAM

# python fopt.py [-v]:
if __name__ == "__main__":
    # import doctest
    # doctest.testmod()
    from mueller_brown import CHAIN_OF_STATES as P
    test(P[0], P[4])

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
