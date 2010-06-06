#!/usr/bin/python
"""
Hessian update schemes go here.

    >>> from mueller_brown import gradient as g, CHAIN_OF_STATES

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

__all__ = ["SR1", "LBFGS", "BFGS", "Array"]

from numpy import asarray, empty, dot
from numpy import eye, outer
from numpy.linalg import solve #, eigh

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
        self.positive = positive

        # hessian matrix (dont know dimensions yet):
        self.B = None

    def update(self, dr, dg):
        """Update scheme for the direct hessian:

            B    = B - (B * s) * (B * s)' / (s' * B * s) + y * y' / (y' * s)
             k+1    k    k         k               k

        where s is the step and y is the corresponding change in the gradient.
        """

        # initial hessian (in case update is called first):
        if self.B is None:
            self.B = self.B0 * eye(len(dr))

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

    def inv(self, y):
        """Computes s = H * y using internal representation
        of the inverse hessian, H = B^-1.
        """

        # initial hessian (in case inv() is called first):
        if self.B is None:
            self.B = self.B0 * eye(len(y))

        return solve(self.B, y)

    def app(self, s):
        """Computes y = B * s using internal representation
        of the hessian B.
        """

        # initial hessian (in case inv() is called first):
        if self.B is None:
            self.B = self.B0 * eye(len(s))

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

    def update(self, s, y):
        """Update scheme for the direct/inverse hessian:

            B    = B  +  z * z' / (z' * s )   with   z = y - B * s
             k+1    k     k   k     k    k            k   k   k   k

        where s is the step and y is the corresponding change in the gradient.
        """

        # initial hessian (in case update is called first):
        if self.B is None:
            self.B = self.B0 * eye(len(s))

        z = y - self.app(s)

        # avoid small denominators:
        if dot(z, s)**2 < dot(s, s) * dot(z, z) * 1.0e-14:
            print "SR1: WARNING, skipping update, denominator too small!"
            # just skip the update:
            return

        self.B += outer(z, z) / dot(z, s)

    def inv(self, y):
        """Computes s = H * y using internal representation
        of the inverse hessian, H = B^-1.
        """

        # initial hessian (in case inv() is called first):
        if self.B is None:
            self.B = self.B0 * eye(len(y))

        return solve(self.B, y)

    def app(self, s):
        """Computes y = B * s using internal representation
        of the hessian B.
        """

        # initial hessian (in case app() is called first):
        if self.B is None:
            self.B = self.B0 * eye(len(s))

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
