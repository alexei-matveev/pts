#!/usr/bin/env python

import sys
import getopt

from numpy import cos, sin, array, hstack, dot, sqrt
from numpy.linalg import norm
import scipy.optimize as opt
import numpy as np

import ase

tmp = "2 0 0 1 1 2 8 6 7 3 5 4 6 5 3 8"
tmp = "2 0 0 1 7 3 8 6 4 8 5 4 6 5 3 7"
tmp = "2 0 0 1 1 2 7 3 8 6 4 8 5 4 6 5 3 7"
sep_str = "15 21 13 13 14 14 11 9 9 10 12 11 10 12 20 20 21 18 19 19 17 15 16 17 18 16 3 5 5 1 7 7 4 6 8 8 6 0 22 22 23 23 24 24 2 2 1 3 0 4 25 25"

def aaa_dist(geom):
    """Calculate average interatom distance."""
    all = []
    for i in range(len(geom)):
        for j in range(i, len(geom)):
            d = geom[i] - geom[j]
            d = sqrt(dot(d,d))
            all.append(d)

    ad = sum(all) / len(all)
    return ad



class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "h", ["help"])
        except getopt.error, msg:
             raise Usage(msg)
        
        if len(args) < 2:
            raise Usage("Must specify two files")

        sep_str = args[2]
        run(sep_str, args[0], args[1], [2,9,11])

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2

def run(order_str, fn1, fn2, ixs=None):
    """ Setup everything and run minimisation.

    Based on ordering given in order_Str, geometries in fn1 and fn2, and the 
    indices ixs, rotate molecule in fn2 so that the atoms with indices ixs 
    coincide with those in fn1.
    """

    # get re-ordering list
    order = order_str.split()
    order = array([int(o) for o in order])
    tot = len(order)
    assert tot % 2 == 0
    order = order.reshape(-1,2)

    order1 = order[:,0].copy()
    order2 = order[:,1].copy()
    order1.sort(), order2.sort()
    assert (order1 == order2).all()

    a1 = ase.read(fn1)
    a2 = ase.read(fn2)
    assert len(a1) == len(a2)
    n = len(a1)
    assert len(order) == len(a1)

    # get coords
    geom1 = a1.get_positions().copy()
    geom2 = a2.get_positions().copy()

    # re-order coords so that they are both the same
    g1 = []
    g2 = []
    chem_symbols = []
    for o in order:
        i,j = o
        g1.append(geom1[i])
        g2.append(geom2[j])
        assert a1.get_chemical_symbols()[i] == a2.get_chemical_symbols()[j]
        chem_symbols.append(a2.get_chemical_symbols()[j])

    g1 = array(g1)
    g2 = array(g2)

    old_dist_sum1 = aaa_dist(g1)
    old_dist_sum2 = aaa_dist(g2)

    a1.set_positions(g1)

    r = Rotate(g1.copy(), g2.copy(), ixs)

    x0 = array([0.,0,0,0,0,.001])
    x = opt.fmin_bfgs(r.diff, x0)
    print "Optimising vector:", x

    g2_new = r.trans(g2, x)
    a1.set_chemical_symbols(chem_symbols)
    a2.set_chemical_symbols(chem_symbols)

    a1.set_positions(g1)
    a2.set_positions(g2_new)

    new_dist_sum1 = aaa_dist(g1)
    new_dist_sum2 = aaa_dist(g2_new)

    # Make sure the geometries haven't changes shape in any way by comparing
    # the old/new interatom distance.
    assert abs(old_dist_sum1 - new_dist_sum1) < 1e-6
    assert abs(old_dist_sum2 - new_dist_sum2) < 1e-6

    ase.view([a1,a2])
    ase.write(fn1 + '.t', a1, format='xyz')
    ase.write(fn2 + '.t', a2, format='xyz')

class Rotate:
    """Class to support alignment of two structures in Cartesian space.
    
    Based on a set of indices of coordinates (ixs) or all coordinates 
    (if ixs == None), minimises the difference between these coordinates by
    translating and rotating them."""

    def __init__(self, geom1, geom2, ixs=None):

        if ixs == None:
            self.g1 = geom1
            self.g2 = geom2
        else:
            self.g1 = []
            self.g2 = []
            for i in ixs:
                self.g1.append(geom1[i])
                self.g2.append(geom2[i])


        
    def mat(self, v):
        """Generates rotation matrix based on vector v, whose length specifies 
        the rotation angle and whose direction specifies an axis about which to
        rotate."""

        assert len(v) == 3
        phi = norm(v)
        a = cos(phi/2)
        if phi < 0.02:
            print "Used Taylor series approximation, phi =", phi
            """
            q2 = sin(phi/2) * v / phi
               = sin(phi/2) / (phi/2) * v/2
               = sin(x)/x + v/2 for x = phi/2

            Using a taylor series for the first term...

            (%i1) taylor(sin(x)/x, x, 0, 8);
            >                           2    4      6       8
            >                          x    x      x       x
            > (%o1)/T/             1 - -- + --- - ---- + ------ + . . .
            >                          6    120   5040   362880

            Below phi/2 < 0.01, terms greater than x**8 contribute less than 
            1e-16 and so are unimportant for double precision arithmetic.

            """
            x = phi / 2
            taylor_approx = 1 - x**2/6. + x**4/120. - x**6/5040. + x**8/362880.
            q2 = v/2 * taylor_approx

        else:
            q2 = sin(phi/2) * v / phi

        b,c,d = q2

        m = np.array([[ a*a + b*b - c*c - d*d , 2*b*c + 2*a*d,         2*b*d - 2*a*c  ],
                   [ 2*b*c - 2*a*d         , a*a - b*b + c*c - d*d, 2*c*d + 2*a*b  ],
                   [ 2*b*d + 2*a*c         , 2*c*d - 2*a*b        , a*a - b*b - c*c + d*d  ]])

        return m

    def trans(self, geom, v):
        """Translates and rotates geometry geom based on v."""

        shift = v[:3] # displacement
        imq   = v[3:] # imag_quaternion

        mat = self.mat(imq)

        g_moved = []
        for g in geom:
            g_ = dot(mat, g)
            g_ += shift
            g_moved.append(g_)

        g_moved = array(g_moved)

        return g_moved

    def diff(self, x):
        """Calculates the summed squared differences between geometries g1 
        and g2."""

        g2_rot = self.trans(self.g2, x)
        diff = (self.g1 - g2_rot)**2
        s = sum(diff.flatten())
        #print s
        return s


if __name__ == "__main__":
    sys.exit(main())

