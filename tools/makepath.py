#!/usr/bin/python

"""
Script to interpolate between two minima
in internal or cartesian coordinates,
ase and aof needs to be known when running
The input can be in any ase format, in the files GEOL
and GEOR

usage: makepath.py  GEOL GEOR

or
   makepath.py --num 12 GEOL GEOR

--num gives the number of interpolated points, default
would be 7

if given more than two geometries, the path will go
though all of them.

if interpolation is wanted in internal coordinates,
add --zmat ZMAT
where the zmatrix in ZMAT (can be generated in ag),
should look soemthing like:
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
needn't be there

It is also possible to interpolate in mixed coordinates, thus
using a zmatrix for only the first atoms, and leave the rest in
Cartesian or to use several zmatrices. In the last case for each of
them there should be a call --zmat ZMATi. The order is important, the
zmatrices always take the atoms from the top of the Cartesian
coordinates.

some additional paramters contain the output:
If none of them  is set the output will in xyz format to the
stdout
  --pos:      the output will be given as poscars in (direct)
            style as POSCAR0 to POSCARN
  --allcoord: This way all the coordinates in internal and
            Cartesian will be given to the stdout
            (in Cartesian interpolation they are the same)
"""

import sys
if sys.argv[1] == '--help':
    print __doc__
    sys.exit()

from ase import read, write
from aof.coord_sys import   ComplexCoordSys
from aof.coord_sys import vector_completation, ase2xyz, ase2int, ase2ccs, enforce_short_way
from aof.path import Path
from aof.common import file2str
from numpy import linspace
from string import count
from pydoc import help
from aof.inputs.pathsearcher import expand_zmat

# Defaultvalues for parameters
# output as xyz, and 7 beads
output = 0
steps = 7

args = sys.argv[1:]

zmts = []
elem_num = 0
el_nums = []
m1 = None
mi = []

for k in range(len(args)):
    if args == []:
         break
    if args[0] == "--pos":
         output = 1
         #print "set POSCAR's as output!"
         args = args[1:]
    elif args[0] == "--allcoord":
         output = 2
         #print "output gives all coordinates!"
         args = args[1:]
    elif args[0] == "--num":
         steps = int(args[1])
         #print "interpolation of %d beads" % (steps)
         args = args[2:]
    elif args[0] == "--zmat":
#Then read in the zmatrix in ag form
         zmts1 = file2str(args[1])
         #print "interpolation in internal coordinates"
         args = args[2:]
         # The variables in the zmat may not be set yet
         #then do this as service
         zmts1, elem_num1 = expand_zmat(zmts1)
         zmts.append(zmts1)
         elem_num += elem_num1 + 6
         if (elem_num1 ==1):
             elem_num -= 1
             el_nums.append(elem_num1)
         else:
             el_nums.append(elem_num1)

#the two files for the minima
    else:
         mi.append(read(args[0]))
         args = args[1:]

assert len(mi) > 1

# mol is the ase-atom object to create the ase output
# the xyz or poscar files
mol = mi[0]
num_atoms = len(mol.get_atomic_numbers())
zmti = []

if zmts == []:
    # Cartesian coordinates are in a ZMatrix2 object
    # Two for the minima, they can be set already
    for m in mi:
       zmti.append(ase2xyz(m))
    # The next one is the one for the current geometry
    zmtinter = ase2xyz(mi[0])
elif num_atoms * 3 > elem_num or len(zmts) > 1:
    # internal coordinates are in a ZMatrix2 object
    # Two for the minima, they can be set already
    for m in mi:
      zmti.append(ase2ccs(m, zmts, el_nums, elem_num))
    zmtinter =  ase2ccs(mi[0], zmts, el_nums, elem_num)
else:
    # internal coordinates are in a ZMatrix2 object
    # Two for the minima, they can be set already
    zmts = str(zmts[0])
    for m in mi:
      zmti.append(ase2int(m, zmts, elem_num))
    # The next one is the one for the current geometry
    zmtinter = ase2int(mi[0], zmts, elem_num)

if zmts != []:
  enforce_short_way(zmti)

# redefine symbols to contain only geometries (internal):
for i, zm in enumerate(zmti):
  mi[i] = zm.get_internals()

# path between two minima (linear in coordinates):
ipm1m2 = Path(mi)

if output == 2:
    if zmts == None:
        print "Interpolation between M1 and M2 in Cartesians:"
    else:
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
       write("POSCAR%i" % i, mol, format="vasp", direct = "True")
   elif output == 0:
       write("-" , mol)

