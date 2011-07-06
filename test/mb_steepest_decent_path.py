#!/usr/bin/env python
"""
This module calculates the "real" Mueller-Brown potential
minimum reaction path by a steepest decent method.
"""
from pts.mueller_brown import show_chain, CHAIN_OF_STATES
from pts.mueller_brown import MB
from pts.simpledecent import find_connections, relax_points
from pylab import show
from numpy import savetxt, array

tol = 1e-3

FUN = MB

CHAIN1 = relax_points(MB, array(CHAIN_OF_STATES), tol)

PATH = find_connections(FUN, CHAIN1, tolerance = tol, max_iter = 50000 )
savetxt( "mb_test.txt", PATH)
show_chain(PATH, style = "r-", save = "wait")


show()
