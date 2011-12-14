#!/usr/bin/env python

from pts.path_searcher import pathsearcher
from ase.atoms import Atoms
from ase.calculators.lj import LennardJones
from pts.zmat import ZMat
from pts.test.testfuns import diagsandhight
from numpy import sqrt, pi, asarray
from pts.cfunc import Justcarts, With_globals, Masked
from pts.constr_symar4 import t_c2v, t_c2v_prime
import pts.metric as mt
from pts.metric import Metric, Metric_reduced
# MUSTDIE: from pts.coord_sys import CoordSys
from pts.qfunc import QFunc
from pts.func import compose
from sys import argv

"""
Using an Lennard Jones Cluster isomerie transformation to show some
of ParaTools features.
With testfun the cooordinate system is choosen, available are some
zmatrices and a description with diagonals and hights, see below
To see how the symmetry of forces vector is affected by the different
noms use contraforce as test_what, pathsearcher will start an
reaction path optimization.

Usage:
python test_nc.py <testfun>

With possible testfun:
 zmat : Zmat description of system
 zmate : Zmat extended with global parameters
 diaghigh : only diagonals and hight is given
 carts: Cartesian (same starting geometries as diaghigh)
 cartsred : Cartesians as carts but with fixed coordinates
 ztoc : Cartesians (starting geometries from zmat)
 ztoc_red : same as ztoc but with fixed coordinates

For the pathsearcher testing we need a middle bead at the start also as else
the main moving atom would behave unphysically.

If the variable test_what is set to contraforce the symmetry of the forces can be
tested. This makes only sense with testfun zmat or zmate. Here one can see that the
metric Cartesian makes the contravariant force of zmate symmetric, while for the
testfun zmat the metric Cartesian with reduction of global positions is needed.
"""
test_what = "pathsearcher"
"""
uncomment the next line for testing the symmetry, be aware that this
only gives reasonable results if testfun is set to zmat or zmate
"""
#test_what = "contraforce"

# specify which testfunction == first input argument
try:
    testfun = argv[1]
except IndexError:
    testfun = "zmat"

# The ASE atoms object for calculating the forces
ar4 = Atoms("Ar4")

ar4.set_calculator(LennardJones())

# PES in cartesian coordiantes:
pes = QFunc(ar4, ar4.get_calculator())

# some variables for the starting values
# length
var1 = 1.12246195815

# and angles (in rad)
var3 = 60.0 / 180. * pi
var4 = 70.5287791696 / 180. * pi
var6 = 59.0020784259 / 180. * pi
var7 = 60.4989607871 / 180 * pi
var8 = pi

td_s = var1

def reduce(vec, mask):
    """
    Function for generating starting values.
    Use a mask to reduce the complete vector.
    """
    vec_red = []
    for i, m in enumerate(mask):
         if m:
             vec_red.append(vec[i])
    return asarray(vec_red)

if testfun == "zmat":
    """
    Zmatrix description with a starting values for nice
    symmetry behaviour. Lenth to be equal in C2V symmetry
    fullfil this at start to an accuracy of < 1e-15 Angstrom.
    """
    func = ZMat([(), (0,), (0, 1), (1, 2, 0)])

    min1 = [var1, var1, var3, var1, var3, var4]
    min2 = [var1, var1, var3, var1, var3, 2 * pi -var4]
    middle = [var1, var1, var3, var1, var3, var8]

elif testfun == "zmate":
    """
    Zmatrix and startingvalues as in zmat2. But in this case there are
    added some variables for global orientation, set to zero on every
    position at start.
    """
    func1 = ZMat([(), (0,), (0, 1), (1, 2, 0)])
    func = With_globals(func1)

    min1 = [var1, var1, var3, var1, var3, var4, 0. ,0. ,0. ,0. ,0. ,0. ]
    min2 = [var1, var1, var3, var1, var3, 2* pi -var4, 0. ,0. ,0. ,0. ,0. ,0. ]
    middle = [var1, var1, var3, var1, var3, var8, 0. ,0. ,0. ,0. ,0. ,0. ]

elif testfun == "diaghigh":
    """
    Special diagonal and high function, which uses only three parameters
    to describe the Cluster: the diagonals between two opposite atoms and
    the hight in z-direction between the two x-y-Planes the atoms are
    located in. Function provides also analytical derivative.
    """
    func = diagsandhight()

    der = 1.0
    #       diag1, diag2, hight
    min1 = [td_s, td_s, td_s / sqrt(2.)]
    min2 = [td_s, td_s, -td_s / sqrt(2.)]
    middle = [td_s * der, td_s * sqrt(4. - der**2.) , 0.]

elif testfun == "carts":
    """
    The starting geometries are provided by the diag and high function.
    See case "diaghigh". The optimization and else is done in Cartesian
    coordinates.
    """
    func = Justcarts()

    # generate starting geometries with the help of function diagsandhight
    func_h = diagsandhight()
    min1 = func_h(asarray([td_s, td_s, td_s / sqrt(2.)])).flatten()
    min2 = func_h(asarray([td_s, td_s, -td_s / sqrt(2.)])).flatten()
    middle = func_h(asarray([td_s, td_s * sqrt(3.) , 0.])).flatten()

elif testfun == "cartsred":
    """
    Same starting situation as in case "carts", only that in this case some
    coordinates are fixed (y,z) direction for first two atoms and (x) for the
    last two.
    """
    func2 = Justcarts()

    # starting geometries with fixed symmetries
    func_h = diagsandhight()
    min1 = func_h(asarray([td_s, td_s, td_s / sqrt(2.)])).flatten()
    min2 = func_h(asarray([td_s, td_s, -td_s / sqrt(2.)])).flatten()
    middle = func_h(asarray([td_s, td_s * sqrt(3.) , 0.])).flatten()

    # see diagsandhight function to see that this fixing is correct
    mask = [True] + [False]*2 + [True] + [False] * 3 + [True] * 2 + [False] + [True] * 2
    func = Masked(func2, mask, min1)
    # the geometries have also to be reduced
    min1 = reduce(min1, mask)
    min2 = reduce(min2, mask)
    middle = reduce(middle, mask)

elif testfun == "ztoc":
    """
    The starting geometries are provided by the Zmatrix function, see case "zmat".
    The optimization and else is done in Cartesian coordinates.
    """
    func = Justcarts()

    func_h = ZMat([(), (0,), (0, 1), (1, 2, 0)])
    min1 = func_h([var1, var1, var3, var1, var3, var4]).flatten()
    min2 = func_h([var1, var1, var3, var1, var3, 2 * pi -var4]).flatten()
    middle = func_h([var1, var1, var3, var1, var3, var8]).flatten()

elif testfun == "ztoc_red":
    """
    Same starting situation as in case "ztoc", only that in this case some
    coordinates are fixed: position of first atom, y, z coordinate of second
    and y coordinate of third one == global rotation and translation is fixed.
    """
    func2 = Justcarts()

    # starting geometries with fixed symmetries
    func_h = ZMat([(), (0,), (0, 1), (1, 2, 0)])
    min1 = func_h([var1, var1, var3, var1, var3, var4]).flatten()
    min2 = func_h([var1, var1, var3, var1, var3, 2 * pi -var4]).flatten()
    middle = func_h([var1, var1, var3, var1, var3, var8]).flatten()
    # Mask fixes the "global parameter"
    mask = [False]*3 + [True] + [False] * 2 + [True] + [False] + [True] * 4
    func = Masked(func2, mask, min1)
    # also reduce the starting values
    min1 = reduce(min1, mask)
    min2 = reduce(min2, mask)
    middle = reduce(middle, mask)

# PES in internal coordinates:
pes = compose(pes, func)

# init path contains the two minima and a middle bead to enforce the
# starting path going in the right direction (else it would be just
# linear interpolation, what is not correct in this regard)
init_path = [min1, middle, min2]

if test_what == "contraforce":
    # Metric consistent with coordinate transformation:
    m1 = Metric(func)
    m2 = Metric_reduced(func)

    for geo in init_path:
        # Check the symmetry:
        print "GEOMETRY"
        print "Geometry is symmetric?:"
        # T_c2v tests if internal coordinates t_c2v
        # provide the correct symmetry (here the ZMatrix used
        # in testfun zmat is expected)
        print t_c2v(geo)

        # get the forces belonging to this geometry
        f = - pes.fprime(geo)

        print "Forces", f
        # test for displacements, if they would fulfill the requirements
        # for keeping the symmetry, this should not hold for the
        # covariant forces (the first is real, the second is relative to
        # the complete vector length). The three numbers refer to the three
        # constraints one gets.
        print "co variant Forces are symmetric:", t_c2v_prime(f)[1]

        # Two ways of getting contravariant forces: one with Cartesian
        # metric the second one with additional removal of global positions
        f_co = m1.raises( geo, f)
        f_co_red = m2.raises(geo, f)
        # Here the constraints may be fulfilled (at least in one case)
        print "contravariant forces:", f_co
        print "are symmetric: ", t_c2v_prime(f_co)[1]
        print "contravariant forces red:", f_co_red
        print "are symmetric:", t_c2v_prime(f_co_red)[1]
        print
        print

elif test_what == "pathsearcher":
    #
    # Let pathseracher optimize the path:
    #
    print("\n* Default path searcher, 7 beads:\n")
    conv, (xs, ts, es, gs) = pathsearcher(ar4, init_path, funcart = func, ftol = 0.01, maxit = 40, beads_count = 7, \
                                              workhere = 0, output_level = 0, output_path = ".")
    print "path=\n", xs
    print "energies=\n", es
    print "converged=\n", conv

    print("\n* Experimental path searcher, 15 beads:\n")
    conv, (xs, ts, es, gs) = pathsearcher(ar4, init_path, funcart = func,
                                          beads_count = 15,
                                          ftol = 1.0e-8,
                                          xtol = 1.0e-8,
                                          maxstep = 0.3,
                                          maxit = 42,
                                          method="sopt",
                                          workhere = 0,
                                          output_level = 0,
                                          output_path = ".")
    print "path=\n", xs
    print "energies=\n", es
    print "converged=\n", conv



