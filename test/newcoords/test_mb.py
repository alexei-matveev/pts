#!/usr/bin/env python

from pts.path import Path
from pts.path_searcher import find_path
from numpy import array, linspace
from sys import argv
from pts.pes.mueller_brown import MB, CHAIN_OF_STATES
"""
Small test example with MuellerBrown Potential.

Usage:

test_mb.py <n>

with n beeing the number of beads to optimize.
By changing the init_path different (parts) of
the path can be examined.
"""
try:
    beads_count = int(argv[1])
except IndexError:
    beads_count = 7

# The tree minima
min1 = CHAIN_OF_STATES[0] # array([-0.55822362,  1.44172583])
min2 = CHAIN_OF_STATES[2] # array([-0.05001084,  0.46669421])
min3 = CHAIN_OF_STATES[4] # array([ 0.62349942,  0.02803776])

# Three minima:
init_path = [min1, min2, min3]

#
# Initial guess:
#
if beads_count != len(init_path):
    p = Path(init_path)
    init_path = map(p, linspace(0., 1., beads_count))

#
# Search by default method:
#
conv1, res1 = find_path(MB, init_path, ftol = 0.001, maxit = 100, workhere = True, output_level = 0)

#
# Search by an alternative method:
#
conv2, res2 = find_path(MB, init_path, ftol = 0.001, maxit = 100, method="sopt", workhere = True, output_level = 0)

print "\n"
print "result 1=\n", conv1, "\n", res1[0], "\n", res1[1], "\n", res1[2], "\n", res1[3]
print "\n"
print "result 2=\n", conv2, "\n", res2[0], "\n", res2[1], "\n", res2[2], "\n", res2[3]
