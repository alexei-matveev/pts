#!/usr/bin/env python
"""
Hessian update schemes go here.

    >>> from pts.pes.mueller_brown import gradient as g, CHAIN_OF_STATES

Minimium A on MB surface:

    >>> r = CHAIN_OF_STATES[0]

    >>> from numpy import array, max, abs

Two steps ...

    >>> s1 = 0.001 * array([1.0,  1.0])
    >>> s2 = 0.001 * array([1.0, -1.0])

... and two gradeint differences:

    >>> y1 = g(r + s1) - g(r - s1)
    >>> y2 = g(r + s2) - g(r - s2)

Symmetric rank-1 (SR1) implementation:

    >>> h3 = SR1()

    >>> h3.update( 2 * s1, y1)
    >>> h3.update( 2 * s2, y2)

    >>> h3.inv(y2)
    array([ 0.002, -0.002])

    >>> h3.inv(y1)
    array([ 0.002,  0.002])

    >>> h3.inv(h3.app(s1))
    array([ 0.001,  0.001])

BFGS implementation:

    >>> h1 = BFGS()

    >>> h1.update(2 * s1, y1)
    >>> h1.update(2 * s2, y2)

The two methods, inv() and app() apply inverse and direct
hessians, respectively:

    >>> h1.inv(y2)
    array([ 0.002, -0.002])

    >>> h1.inv(y1)
    array([ 0.00200021,  0.00200021])

    >>> max(abs(h1.inv(h1.app(s1)) - s1)) < 1.e-10
    True

    >>> max(abs(h1.inv(h1.app(s2)) - s2)) < 1.e-10
    True

Limited-memory BFGS implementation (inverse only):

    >>> h2 = LBFGS()

    >>> h2.update(2 * s1, y1)
    >>> h2.update(2 * s2, y2)

    >>> h2.inv(y2)
    array([ 0.002, -0.002])

    >>> h2.inv(y1)
    array([ 0.00200021,  0.00200021])
"""

__all__ = ["SR1", "LBFGS", "BFGS", "Array", "isolve"]

from numpy import asarray, empty, zeros, dot
from numpy import eye, outer
from numpy.linalg import norm, solve
#from numpy.linalg import solve #, eigh

def isolve(A, b, tol=1.0e-7):
    """
    Solve iteratively  a linear equation A(x)  = b, with  A(x) being a
    callable linear operator  and b a vector. If  provided, x0 is used
    as an initial value instead of b.

    >>> from test.testfuns import Affine

    >>> a = array([[2.0, 0.5],
    ...            [0.0, 8.0]])

    >>> A = Affine(a)

    >>> x = array([0.25, 4.0])

    >>> solve(a, A(x))
    array([ 0.25,  4.  ])

    >>> isolve(A, A(x))
    array([ 0.25,  4.  ])

    FIXME:  very preliminary, newer  versions of  SciPy seem  to offer
    better solutions ...
    """

    b = asarray(b)

    qs = []
    ys = []
    A_ = empty((0, 0))
    b_ = empty(0)

    xnew = zeros(b.shape) # x0?
    qn = b # array([0.0, 1.0])

    for n in range(1, b.size+1):
        # print "isolve: n=", n

        # orthogonalization:
        for q in qs:
            qn = qn - q * dot(q, qn)
        qn = qn / norm(qn)

        # print "new qn=", qn

        # new basis vector:
        qs.append(qn)

        # and a new candidate, for the next iteration:
        qn = A(qn)

        # print "A(qn)=", qn

        ys.append(qn)

        A_new = empty((n, n))
        b_new = empty(n)

        A_new[:-1, :-1] = A_
        b_new[:-1] = b_

        A_ = A_new
        b_ = b_new

        #
        # A   = ( q  . A * q ):
        #  ij      i        j
        #
        for i in range(n):
            A_[i, -1] = dot(qs[i], ys[-1])
            A_[-1, i] = dot(qs[-1], ys[i])

        #
        # b  = (q  .  b):
        #  i     i
        #
        b_[-1] = dot(qs[-1], b)

        # print "A=", A_
        # print "b=", b_
        x_ = solve(A_, b_)
        # print "x=", x_

        xnew, xold = dot(x_, qs), xnew

        # print "xnew=", xnew, "norm(xnew - xold)=", norm(xnew - xold)

        if norm(xnew - xold) < tol * norm(xnew):
            # print "break"
            break

    return xnew

def _sr1(B, s, y, thresh=1.0e-7):
    """Update scheme for the direct/inverse hessian:

        B    = B  +  z * z' / (z' * s )   with   z = y - B * s
         k+1    k     k   k     k    k            k   k   k   k

    where s is the step and y is the corresponding change in the gradient.

    NOTE: modifies B in-place using +=
    """

    z = y - dot(B, s)

    # avoid small denominators:
    if dot(z, s)**2 > dot(s, s) * dot(z, z) * thresh**2:
        B += outer(z, z) / dot(z, s)
    else:
        # just skip the update:
        print "SR1: WARNING, skipping update, denominator too small!"


# FIXME: we will use None as -infinity, because of this feature:
assert None < -1.0

def _bfgsB(B, s, y, thresh=None):
    """BFGS update scheme for the direct hessian, one of several
    equivalent expressions:

        B    = B - (B * s) * (B * s)' / (s' * B * s) + y * y' / (y' * s)
         k+1    k    k         k               k

    where s is the step and y is the corresponding change in the gradient.

    NOTE: modifies B in-place using +=
    """

    # this is positive on *convex* surfaces:
    if dot(s, y) <= thresh:
        # FIXME: with default thresh=None this is never True!

        # just skip the update:
        return

        # FIXME: Only when we are doing MINIMIZATION!
        #        For a general search of stationary points, it
        #        must be better to have accurate hessian.

    # for the negative term:
    z = dot(B, s)

    B += outer(y, y) / dot(s, y) - outer(z, z) / dot(s, z)

def _bfgsH(H, s, y, thresh=None):
    """BFGS update scheme for the inverse hessian, one of several
    equivalent expressions:

        H    = H  + (1 + y' * z) * s * s' / r - s * z' - z * s'
         k+1    k

    with

        r = y' * s

    and

        z = H * y / r

    where s is the step and y is the corresponding change in the gradient.

    NOTE: modifies H in-place using +=
    """

    # this is positive on *convex* surfaces:
    r = dot(s, y)
    if r <= thresh:
        # FIXME: with default thresh=None this is never True!

        # just skip the update:
        return

        # FIXME: Only when we are doing MINIMIZATION!
        #        For a general search of stationary points, it
        #        must be better to have accurate hessian.

    z = dot(H, y) / r

    H += outer(s, s) * ((1.0 + dot(y, z)) / r) - outer(s, z) - outer(z, s)

#
# NOTE: DFP update formula for direct hessian B coincides with
#       BFGS formula for inverse hessian H and vice versa:
#
#       _dfpB = _bfgsH
#       _dfpH = _bfgsB
#
# In part for this reason DFP update is not implemented here.
#

#
# Hessian models below should implement at least update() and inv() methods.
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

                FIXME: maybe use

                    H = (y' * s ) / (y' * y )
                     0

                as a diagonal approximation prior to first update?
                See "Numerical Optimization", J. Nocedal.

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
        See corresponding |inv|-function for more info.
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

    def inv(self, g):
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

    def app(self, s):
        # FIXME: need a limited memory implementation for
        # direct hessian. For the moment use BFGS() instead.
        raise NotImplementedError

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
        self.thresh = None # FIXME: -infifnity
        if positive:
            self.thresh = 0.0

        # hessian matrix (dont know dimensions yet):
        self.B = None
        self.H = None

    def update(self, s, y):

        # initial hessian (in case update is called first):
        if self.B is None:
            self.B = eye(len(s)) * self.B0

        if self.H is None:
            self.H = eye(len(s)) / self.B0

        # update matrices in-place:
        _bfgsB(self.B, s, y, self.thresh)
        _bfgsH(self.H, s, y, self.thresh)

    def inv(self, y):
        """Computes s = H * y using internal representation
        of the inverse hessian, H = B^-1.
        """

        # initial hessian (in case inv() is called first):
        if self.H is None:
            self.H = eye(len(y)) / self.B0

        return dot(self.H, y)

    def app(self, s):
        """Computes y = B * s using internal representation
        of the hessian B.
        """

        # initial hessian (in case app() is called first):
        if self.B is None:
            self.B = eye(len(s)) * self.B0

        return dot(self.B, s)

class SR1:
    """Update scheme for the direct/inverse hessian:

        B    = B + z * z' / (z' * s )   with   z = y - B * s
         k+1    k   k   k     k    k            k   k   k   k

    where s is the step and y is the corresponding change in the gradient.
    """

    def __init__(self, B0=70.):
        """
        Parameters:

        B0      Initial approximation of direct Hessian.
                Note that this is never changed!
        """

        self.B0 = B0

        # hessian matrix (dont know dimensions yet):
        self.B = None
        self.H = None

    def update(self, s, y):

        # initial hessian (in case update is called first):
        if self.B is None:
            self.B = eye(len(s)) * self.B0

        if self.H is None:
            self.H = eye(len(s)) / self.B0

        # update direct/inverse hessians in-place:
        _sr1(self.B, s, y)
        _sr1(self.H, y, s)

    def inv(self, y):
        """Computes s = H * y using internal representation
        of the inverse hessian, H = B^-1.
        """

        # initial hessian (in case inv() is called first):
        if self.H is None:
            self.H = eye(len(y)) / self.B0

        return dot(self.H, y)

    def app(self, s):
        """Computes y = B * s using internal representation
        of the hessian B.
        """

        # initial hessian (in case app() is called first):
        if self.B is None:
            self.B = eye(len(s)) * self.B0

        return dot(self.B, s)

class Hughs_Hessian:
    """
    Removed BFGS/SR1 code from optimizer multiopt
    Used BFGS as prototype for the interface, but
    be aware that this variant needs two more variables for update
    """
    def __init__(self, B0=70., update = "SR1", id = -1):
        """
        Stores all the relevant data
        """
        self.B0 = B0
        self.method = update
        self.B = None
        self.id = id

    def update(self, dr, df):
        """
        Update the approximated hessian

        It is tested if the
        step is big enough for performing the actual update
        if yes the actual update is done.

        Update steps separated from the rest of the code,
        there is a SR1 and a BFGS update dependant on choice of variable
        "update" in initializing
        """
        if self.B is None:
            self.B = eye(len(s)) * self.B0

        # do nothing if the step is tiny (and probably hasn't changed at all)
        if abs(dr).max() < 1e-7: # FIXME: Is this really tiny (enough)?
            return

        # from the code
        dg = dot(self.B, dr)

        if self.method == 'SR1':
            c = df - dot(self.B, dr)

            # guard against division by very small denominator
            # norm only here to get to know trend
            if norm(c) * norm(c) > 1e-8:
                self.B += outer(c, c) / dot(c, dr)
            else:
                print "Bead %d: Hessian: skipping SR1 update, denominator too small" % self.id
        elif self.method == 'BFGS':
            a = dot(dr, df)
            b = dot(dr, dg)
            self.B += outer(df, df) / a - outer(dg, dg) / b
        else:
            assert False, 'Should never happen'

    def app(self, s):
        """Computes y = B * s using internal representation
        of the hessian B.
        """

        # initial hessian (in case app() is called first):
        if self.B is None:
            self.B = eye(len(s)) * self.B0

        return dot(self.B, s)

class Array:
    """Array/List of hessians, e.g. for a string method.
    """

    def __init__(self, H):
        # array/list of hessians:
        self.__H = H

    def __len__(self):
        return len(self.__H)

    def __getitem__(self, i):
        return self.__H[i]

    def update(self, S, Y):
        for h, s, y in zip(self.__H, S, Y):
            h.update(s, y)

    def inv(self, Y):
        return asarray([ h.inv(y) for h, y in zip(self.__H, Y) ])

    def app(self, S):
        return asarray([ h.app(s) for h, s in zip(self.__H, S) ])

# python bfgs.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
