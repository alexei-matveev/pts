#!/usr/bin/env python

from pts.inputs.pathsearcher import pathsearcher
from pts.zmat import ZMat
from numpy import array, eye
from sys import argv
from pts.mueller_brown import mb_atoms
from pts.cfunc import pass_through
"""
Small test example with MuellerBrown Potential.

Usage:

test_mb.py <n>

with n beeing the number of beads to optimize.
By changing the init_path different (parts) of
the path can be examined.
"""

bn = int(argv[1])

# Function and faked atoms object
func = pass_through()
mb = mb_atoms()

# The tree minima
min1 = array([-0.55822362,  1.44172583])
min2 = array([-0.05001084,  0.46669421])
min3 = array([ 0.62349942,  0.02803776])

# starting path
init_path = [min1, min2, min3]

# the complete run
pathsearcher(mb, init_path, funcart = func, ftol = 0.001, maxit = 100, beads_count = bn)

