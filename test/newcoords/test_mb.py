#!/usr/bin/env python

from pts.path import Path
from pts.inputs.pathsearcher import find_path
from numpy import array, linspace
from sys import argv
from pts.mueller_brown import MB
"""
Small test example with MuellerBrown Potential.

Usage:

test_mb.py <n>

with n beeing the number of beads to optimize.
By changing the init_path different (parts) of
the path can be examined.
"""
convert = Pass_through()

beads_count = int(argv[1])

# The tree minima
min1 = CHAIN_OF_STATES[0] # array([-0.55822362,  1.44172583])
min2 = CHAIN_OF_STATES[2] # array([-0.05001084,  0.46669421])
min3 = CHAIN_OF_STATES[4] # array([ 0.62349942,  0.02803776])

# Three minima:
init_path = [min1, min2, min3]

# the complete run
conv, path, es, gs = find_path(MB, init_path, ftol = 0.001, maxit = 100, beads_count = bn)


