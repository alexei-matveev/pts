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
needn't be there, but then the variables have to be named
var1 -> varN

some additional paramters contain the output:
If none of them  is set the output will in xyz format to the
stdout
  --pos:      the output will be given as poscars in (direct)
            style as pos0 to posN
  --allcoord: This way all the coordinates in internal and
            Cartesian will be given to the stdout
            (in Cartesian interpolation they are the same)
"""

import sys
if sys.argv[1] == '--help':
    print __doc__
    sys.exit()

from ase import read, write
from aof.coord_sys import ZMatrix2, RotAndTrans, XYZ, ccsspec, ComplexCoordSys
from aof.path import Path
from aof.common import file2str
from numpy import linspace, round, pi, asarray, array
from string import count
from pydoc import help
from aof.inputs.pathsearcher import fake_xyz_string, expand_zmat

# Defaultvalues for parameters
# output as xyz, and 7 beads
output = 0
steps = 7

args = sys.argv[1:]

zmts = None
m1 = None

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
         zmts = file2str(args[1])
         #print "interpolation in internal coordinates"
         args = args[2:]
         # The variables in the zmat may not be set yet
         #then do this as service
         zmts, elem_num = expand_zmat(zmts)
#the two files for the minima
    elif m1 == None:
         m1 = read(args[0])
         args = args[1:]
    else :
         m2 = read(args[0])
         args = args[1:]

assert m1 != None
assert m2 != None

# mol is the ase-atom object to create the ase output
# the xyz or poscar files
mol = m1
num_atoms = len(mol.get_atomic_numbers())

if zmts == None:
    # Cartesian coordinates are in a ZMatrix2 object
    # Two for the minima, they can be set already
    zmt1 = XYZ( fake_xyz_string(m1))
    zmt1.set_cartesians(m1.get_positions())
    zmt2 = XYZ( fake_xyz_string(m2))
    zmt2.set_cartesians(m2.get_positions())
    # The next one is the one for the current geometry
    zmtinter = XYZ( fake_xyz_string(m1))
elif num_atoms * 3 > elem_num + 6:
    # internal coordinates are in a ZMatrix2 object
    # Two for the minima, they can be set already
    zmt1s = ZMatrix2(zmts, RotAndTrans())
    zmt2s = ZMatrix2(zmts, RotAndTrans())
    # The next one is the one for the current geometry
    zmtinters = ZMatrix2(zmts, RotAndTrans())
    symb_atoms = mol.get_chemical_symbols()
    carts = XYZ(fake_xyz_string(m1))
    carts2 = XYZ(fake_xyz_string(m2))
    carts3 = XYZ(fake_xyz_string(m1))
    diffhere =   (elem_num +6) / 3
    co_all = XYZ(fake_xyz_string(m1, start = diffhere))
    co_alli = XYZ(fake_xyz_string(m1, start = diffhere))
    ccs1 =  ccsspec([zmt1s, co_all], carts=carts)
    ccsi =  ccsspec([zmtinters, co_alli], carts=carts2)
    carts = m2.get_positions()
    co_all2 = XYZ(fake_xyz_string(m2, start = diffhere))
    ccs2 =  ccsspec([zmt2s, co_all2], carts=carts3)
    zmt1 = ComplexCoordSys(ccs1)
    zmt2 = ComplexCoordSys(ccs2)
    zmtinter = ComplexCoordSys(ccsi)
else:
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

if zmts != None:
    # interpolation of dihedral angles needs more logic,
    # say for interpolation from -pi+x to pi-y with small x and y:
    for i, k in enumerate(zmt1.kinds):
       if k == "dih":
           delta = m2[i] - m1[i]
           # normalize the interval between two angles:
           while delta >  pi: delta -= 2.0 * pi
           while delta < -pi: delta += 2.0 * pi
           m2[i] = m1[i] + delta

# path between two minima (linear in coordinates):
ipm1m2 = Path([m1, m2])

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
