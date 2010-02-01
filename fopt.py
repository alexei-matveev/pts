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
from numpy.linalg import eigh
from scipy.optimize import fmin_l_bfgs_b as minimize1D

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
    # x = asarray(x)

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
    r = x

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

#
# Hessian models below should implement at least update() and apply() methods.
#
class LBFGS:
    """This appears to be the update scheme described in

        Jorge Nocedal, Updating Quasi-Newton Matrices with Limited Storage
        Mathematics of Computation, Vol. 35, No. 151 (Jul., 1980), pp. 773-782

    See also:

        R. H. Byrd, J. Nocedal and R. B. Schnabel, Representation of
        quasi-Newton matrices and their use in limited memory methods",
        Mathematical Programming 63, 4, 1994, pp. 129-156

    In essence, this is a so called "two-loops" implementation of the update
    scheme for the inverse hessian:

        H    = ( 1 - y * s' / (y' * s) )' * H * ( 1 - y * s' / (y' * s) )
         k+1                                 k
               + y * y' / (y' * s)

    where s is the step and y is the corresponding change in the gradient.
    """

    def __init__(self, B0=70., memory=10, positive=True):
        """
        Parameters:

        B0:     Initial (diagonal) approximation of *direct* Hessian.
                Note that this is never changed!

        memory: int
                Number of steps to be stored. Three numpy
                arrays of this length containing floats are stored.
                In original literatue there are claims that the values
                <= 10 are  usual.

        positive:
                Should the positive definitness of the hessian be
                maintained?
        """

        # compact repr of the hessian:
        self.H = ([], [], [], 1. / B0)
        # three lists for
        # (1) geometry changes,
        # (2) gradient changes and
        # (3) their precalculated dot-products.
        # (4) initial (diagonal) inverse hessian

        self.memory = memory

        # should we maintain positive definitness?
        self.positive = positive

    def update(self, dr, dg):
        """Update representation of the Hessian.
        See corresponding |apply|-function for more info.
        """

        # expand the hessian repr:
        s, y, rho, h0 = self.H

        # this is positive on *convex* surfaces:
        rho0 = 1.0 / dot(dr, dg)

        if self.positive and rho0 <= 0:
            # Chances are the hessian will loose positive definiteness!

            # just skip the update:
            return

            #   # pretend there is a positive curvature (H0) in this direction:
            #   dg   = dr / h0
            #   rho0 = 1.0 / dot(dg, dr) # == h0 / dot(dr, dr)

            # FIXME: Only because we are doing MINIMIZATION here!
            #        For a general search of stationary points, it
            #        must be better to have accurate hessian.

        s.append(dr)

        y.append(dg)

        rho.append(rho0)

        # forget the oldest:
        if len(s) > self.memory:
            s.pop(0)
            y.pop(0)
            rho.pop(0)

        # update hessian model:
        self.H = (s, y, rho, h0)

    def apply(self, g):
        """Computes z = H * g using internal representation
        of the inverse hessian, H = B^-1.
        """

        # expand representaiton of hessian:
        s, y, rho, h0 = self.H

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

class BFGS:
    """Update scheme for the direct hessian:

        B    = B - (B * s) * (B * s)' / (s' * B * s) + y * y' / (y' * s)
         k+1    k    k         k               k

    where s is the step and y is the corresponding change in the gradient.
    """

    def __init__(self, B0=70., positive=True):
        """
        Parameters:

        B0      Initial approximation of direct Hessian.
                Note that this is never changed!
        """

        self.B0 = B0

        # should we maintain positive definitness?
        self.positive = positive

        # hessian matrix (dont know dimensions jet):
        self.B = None

    def update(self, dr, dg):
        """Update scheme for the direct hessian:

            B    = B - (B * s) * (B * s)' / (s' * B * s) + y * y' / (y' * s)
             k+1    k    k         k               k

        where s is the step and y is the corresponding change in the gradient.
        """

        # initial hessian (in case update is called first):
        if self.B is None:
            self.B = self.B0 * eye(len(s))

        # this is positive on *convex* surfaces:
        if self.positive and dot(dr, dg) <= 0:
            # just skip the update:
            return

            # FIXME: Only when we are doing MINIMIZATION!
            #        For a general search of stationary points, it
            #        must be better to have accurate hessian.

        # for the negative term:
        Bdr = dot(self.B, dr)

        self.B += outer(dg, dg) / dot(dr, dg) - outer(Bdr, Bdr) / dot(dr, Bdr)

    def apply(self, g):
        """Computes z = H * g using internal representation
        of the inverse hessian, H = B^-1.
        """

        # initial hessian (in case apply is called first):
        if self.B is None:
            self.B = self.B0 * eye(len(g))

        # quite an expensive way of solving linear equation
        # B * z = g:
        b, V = eigh(self.B)

        # update procedure maintains positive defiitness, so b > 0:
        z = dot(V, dot(g, V) / b)

        return z

# python fopt.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
