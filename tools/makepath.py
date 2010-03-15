#!/usr/bin/python

"""
Script to interpolate between two minima in internal
coordinates, ase and aof needs to be known when running
The input can be in any ase format, the third parameter
should be the zmatrix (can be generated in ag),

which looks like:
"
Pt
Pt 1 var1
Pt 2 var2 1 var3
Pt 3 var4 1 var5 2 var6
Pt 4 var7 3 var8 1 var9
Pt 5 var10 1 var11 2 var12
C 4 var13 5 var14 6 var15
C 7 var16 4 var17 5 var18
H 7 var19 8 var20 4 var21
H 8 var22 7 var23 4 var24
H 8 var25 7 var26 4 var27

var1   1.0
var2   1.0
var3   1.0
var4   1.0
var5   1.0
...
"

The last part giving the initial values for the variables
needn't be there, but then the variables have to be named
var1 -> varN

If no parameter is set the output will in xyz format to the
stdout

There is also the possibility for some paramters, which could
be set after this three files:
  pos:      the output will be given as poscars in (direct)
            style as pos0 to posN
  allcoord: This way all the coordinates in internal and
            Cartesian will be given to the stdout
  n:        Any number given as last input parameter will be
            the number of beads to be taken, default is 7
"""

import sys
if sys.argv[1] == '--help':
    print __doc__
    sys.exit()

from ase import read, write
from aof.coord_sys import ZMatrix2, RotAndTrans
from aof.path import Path
from aof.common import file2str
from numpy import linspace, round, pi, asarray, array
from string import count
from pydoc import help

# Defaultvalues for parameters
# output as xyz, and 7 beads
output = 0
steps = 7
#the two files for the minima
m1  = read(sys.argv[1])
m2  = read(sys.argv[2])
#Then read in the zmatrix in ag form
zmts = file2str(sys.argv[3])

# look if there are some more parameters set
try:
    if sys.argv[4] == "pos":
         output = 1
         print "set POSCAR's as output!"
         steps = int(sys.argv[5])
    elif sys.argv[4] == "allcoord":
         output = 2
         print "output is a pathfile!"
         steps = int(sys.argv[5])
    else:
        steps = int(sys.argv[4])
except :
    pass

# The variables in the zmat may not be set yet
#then do this as service
elem_num = count(zmts, "var")
#print elem_num
if elem_num > 0:
    a1, a2, a3 = zmts.partition("var%d" % elem_num)
    if len(a2) > 0:
        zmts += "\n"
        for i in range(1,elem_num + 1):
            zmts += "   var%d  1.0\n" % i

# mol is the ase-atom object to create the ase output
# the xyz or poscar files
mol = m1

# internal coordinates are in a ZMatrix2 object
# Two for the minima, they can be set already
zmt1 = ZMatrix2(zmts, RotAndTrans())
zmt1.set_cartesians(m1.get_positions())
zmt2 = ZMatrix2(zmts, RotAndTrans())
zmt2.set_cartesians(m2.get_positions())
# The next one is the one for the current geometry
zmtinter = ZMatrix2(zmts, RotAndTrans())

# redefine symbols to contain only geometries (internal):
m1 = zmt1.get_internals()
m2 = zmt2.get_internals()

# interpolation of dihedral angles needs more logic,
# say for interpolation from -pi+x to pi-y with small x and y:
for i, k in enumerate(zmt1.kinds):
   if k == "dih":
       delta = m2[i] - m1[i]
       # normalize the interval between two angles:
       while delta >  pi: delta -= 2.0 * pi
       while delta < -pi: delta += 2.0 * pi
       m2[i] = m1[i] + delta

# path between two minima (linear in internals):
ipm1m2 = Path([m1, m2])

if output == 2:
    print "Interpolation between M1 and M2 in internals:"

for i, x in enumerate(linspace(0., 1., steps)):
   if output == 2:
       print "Path coordinate of step  =" , x

#  # path object can be called with path coordinate as input:
   zmtinter.set_internals(ipm1m2(x))
   if output == 2:
       print "Internal coordinates ="
       print zmtinter.get_internals()

#  # z-matrix object can be called with internals as input:
   if output == 2:
       print "Cartesian coordinates ="
       print zmtinter.get_cartesians()

   # give these new coordinates (in cartesian) to mol
   mol.set_positions(zmtinter.get_cartesians())
   # coordinates are written out in xyz or poscar format
   # for the last make sure, that the cell is set in the inputs
   if output == 1:
       write("pos%i" % i, mol, format="vasp", direct = "True")
   elif output == 0:
       write('-'  , mol)
