#!/usr/bin/env python

from pts.inputs.pathsearcher import find_path
from numpy import array
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

bn = int(argv[1])

# The tree minima
min1 = array([-0.55822362,  1.44172583])
min2 = array([-0.05001084,  0.46669421])
min3 = array([ 0.62349942,  0.02803776])

# starting path
init_path = [min1, min2, min3]

# the complete run
conv, path, es, gs = find_path(MB, init_path, ftol = 0.001, maxit = 100, beads_count = bn)

print "energies=\n", es
print "converged=\n", conv

