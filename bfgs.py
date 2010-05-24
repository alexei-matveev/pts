#!/usr/bin/python
"""
Hessian update schemes go here.
"""

__all__ = ["LBFGS", "BFGS", "Array"]

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

    def inv(self, g):
        """Computes z = H * g using internal representation
        of the inverse hessian, H = B^-1.
        """

        # initial hessian (in case inv() is called first):
        if self.B is None:
            self.B = self.B0 * eye(len(g))

        # quite an expensive way of solving linear equation
        z = solve(self.B, g)
        #   # B * z = g:
        #   b, V = eigh(self.B)

        #   # update procedure maintains positive defiitness, so b > 0:
        #   z = dot(V, dot(g, V) / b)

        return z


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

    def update(self, dR, dG):
        for h, dr, dg in zip(self.__H, dR, dG):
            h.update(dr, dg)

    def inv(self, G):

        return asarray([ h.inv(g) for h, g in zip(self.__H, G) ])

# python bfgs.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
