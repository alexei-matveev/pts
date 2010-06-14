#!/usr/bin/env python

import sys
import getopt
from random import uniform
from textwrap import fill

from numpy import cos, sin, array, hstack, dot, sqrt, zeros
from numpy.linalg import norm
import scipy.optimize as opt
import numpy as np

import ase

from aof.coord_sys import vec_to_mat

# for testing
big_xyz = array([ 0.0367588 ,  0.00844124,  0.03564942,  1.0701481 ,  0.57181922,
        2.77827499, -0.08114409, -0.18008624,  2.07113898, -1.4290796 ,
        0.39528576,  2.56865304, -0.01089892, -1.67589796,  2.46440286,
       -1.66182536, -1.14037514, -0.77358759, -2.60636861, -1.79649327,
       -0.90963245, -3.47779763, -2.40210894, -1.03059554, -1.10920508,
        1.70487837, -0.73469939, -1.76405276,  2.63578454, -0.94569924,
       -2.36914936,  3.4947839 , -1.13717295,  1.19977795, -1.58199134,
       -0.91229413,  1.84008002, -2.49444292, -1.22437866,  2.43083616,
       -3.33728732, -1.50947173, -0.10475542, -1.79254756,  3.55677096,
       -0.82252364, -2.25700707,  2.00460367,  0.94539018, -2.12923504,
        2.16945836, -1.52424897,  0.26902056,  3.65986058, -2.28509524,
       -0.11561581,  2.10599439, -1.51583893,  1.46971196,  2.35656355,
        1.74773761,  1.24391934, -0.56778729,  2.69190516,  1.91146932,
       -0.62763781,  3.56323993,  2.52715309, -0.67575784,  2.05327534,
        0.17827659,  2.4850298 ,  1.05044053,  1.64760622,  2.55524728,
        0.98795789,  0.46332968,  3.87238954])
big_xyz.shape = (-1,3)

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
    pass

def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "h", ["help"])
        except getopt.error, msg:
             raise Usage(msg)
        
        if len(args) < 2:
            msg = fill("Takes two molecules specified in cartesian coordinates " +
                "and rotates the second one so that some atoms (given in " +
                "the code) are optimally aligned.") + \
                "\nUsage: rotate.py [-h, --help] <molecule1.xyz> <molecule2.xyz> <atom pairs list>"
            raise Usage(msg)

        sep_str = args[2]
        run(sep_str, args[0], args[1], [0,1,2,3,4])

    except Usage, err:
        print >>sys.stderr, err
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
        print i, j, a1.get_chemical_symbols()[i] + " = " + a2.get_chemical_symbols()[j]

        assert a1.get_chemical_symbols()[i] == a2.get_chemical_symbols()[j], a1.get_chemical_symbols()[i] + " = " + a2.get_chemical_symbols()[j]
        chem_symbols.append(a2.get_chemical_symbols()[j])

    g1 = array(g1)
    g2 = array(g2)

    old_dist_sum1 = aaa_dist(g1)
    old_dist_sum2 = aaa_dist(g2)

    a1.set_positions(g1)

    r = Rotate(g1.copy(), g2.copy(), ixs=ixs)

    x, err, g2_new, _ = r.align()
    print "Optimising vector:", x
    print "Alignment error:", err

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

        self.g1 = array(self.g1).reshape(-1,3)
        self.g2 = array(self.g2).reshape(-1,3)

    @staticmethod
    def trans(geom, v):
        """Translates and rotates geometry geom based on v."""

        shift = v[:3] # displacement
        imq   = v[3:] # imag_quaternion

        mat = vec_to_mat(imq)

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

        diff.shape = (-1, 3)
        diff = array([d.sum()**(0.5) for d in diff])

        s = sum(diff.flatten()) / diff.size
        return s

    def align(self, x0 = None):
        
        if x0 == None:
            x0 = array([0., 0, 0, 0, 0, 0.001])

#        prev = self.g2.copy()
#        its = 0
#        while True:
#            its += 1
        x, fopt, gopt, Hopt, func_calls, grad_calls, warnflag = \
            opt.fmin_bfgs(self.diff, 
                x0, 
                gtol=1e-4, 
                full_output=1, 
                disp=0) 

#        print "gopt",gopt
        aligned = self.trans(self.g2, x)
#            if abs(prev - aligned).max() < 1e-4 or its > 4:
#                break
#            prev = aligned.copy()

        assert (self.diff(x) == fopt).all()
        return x, self.diff(x), aligned, warnflag

def cart_diff(c0, c1):
    """Returns the average difference in atom positions for two sets of
    cartesian coordinates.

    No transformation at all:

    >>> d, cs = cart_diff([0, 0, 1.0], [0, 0, 1.0])
    >>> round(d)
    0.0

    Rotation but no translation:

    >>> vec = array([[0,0,1]])
    >>> d, cs = cart_diff(vec, [0,1,0])
    >>> round(d)
    0.0

    Rotation and translation

    >>> g1 = array([[0,0,1], [0,1,0]])
    >>> g2 = array([[0,1,0], [1,0,0]]) + 0.2 + 100
    >>> d, cs = cart_diff(g1, g2)
    >>> d.round()
    0.0

    >>> g1 = big_xyz.copy()
    >>> n = 10 # no of randomised test cases
    >>> trans_vecs = array([uniform(-50,50) for i in range(n*6)]).reshape(-1,6)
    >>> altered = array([Rotate.trans(g1,v) for v in trans_vecs])
    >>> realigned = [cart_diff(g1, a) for a in altered]
    >>> errors = array([round(e) for e,_ in realigned])
    >>> errors.sum()
    0.0




    """
    # Loop: for some reason complete allignment is not achieved
    # after only a single optimisation run.
    its = 0
    changes = []
    while True:
        r = Rotate(c0, c1)
        vars, err, new, warn = r.align()
        its += 1
        change = abs(c1-new).max()
        changes.append(changes)
        if warn == 0 or its > 10 or change < 1e-3:
            break
        c1 = new.copy()
    return err, changes
    

# Testing the examples in __doc__strings, execute
# "python gxmatrix.py", eventualy with "-v" option appended:
if __name__ == "__main__":
    if len(sys.argv) == 1:
        print sys.argv[0], ": Running Doctests, use '-h' for help."
        import doctest
        doctest.testmod()
        exit()

    sys.exit(main())

# You need to add "set modeline" and eventually "set modelines=5"
# to your ~/.vimrc for this to take effect.
# Dont (accidentally) delete these lines! Unless you do it intentionally ...
# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax


