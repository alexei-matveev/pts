#!/usr/bin/env python

"""
Script to interpolate between two minima
in internal or cartesian coordinates,
ase and ParaTools needs to be known when running
The input can be in any ase format, in the files GEOL
and GEOR

usage: paratools make_path  GEOL GEOR

or
   paratools make_path --num 12 GEOL GEOR

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
"

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
from ase.io import write
from pts.io.read_inputs import get_geos, ensure_short_way
from pts.path import Path
from numpy import linspace

def main(args):

    if args[0] == '--help':
        print __doc__
        sys.exit()

    # Defaultvalues for parameters
    # output as xyz, and 7 beads
    output = 0
    steps = 7

    zmts = []
    mis = []
    dc = { "format" : None}

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
        elif args[0] == "--format":
             dc["format"] = args[1]
             args = args[2:]
        elif args[0] == "--zmat":
             #Then read in the zmatrix in ag form
             zmts1 = args[1]
             #print "interpolation in internal coordinates"
             args = args[2:]
             # The variables in the zmat may not be set yet
             #then do this as service
             zmts.append(zmts1)

        #the two files for the minima
        else:
             mis.append(args[0])
             args = args[1:]

    assert len(mis) > 1

    # mol is the ase-atom object to create the ase output
    mol, mi, funcart, dih, quats, lengt = get_geos(mis, dc, zmts)
    mi = ensure_short_way(mi, dih, quats, lengt)

    # path between two minima (linear in coordinates):
    ipm1m2 = Path(mi)

    if output == 2:
        if zmts == []:
            print "Interpolation between M1 and M2 in Cartesians:"
        else:
            print "Interpolation between M1 and M2 in internals:"

    for i, x in enumerate(linspace(0., 1., steps)):
        if output == 2:
            print "Path coordinate of step  =" , x

        # path object can be called with path coordinate as input:
        y = ipm1m2(x)
        if output == 2:
            print "Internal coordinates ="
            print y

        # z-matrix object can be called with internals as input:
        if output == 2:
            print "Cartesian coordinates ="
            print funcart(y)

        # give these new coordinates (in cartesian) to mol
        mol.set_positions(funcart(y))
        # coordinates are written out in xyz or poscar format
        # for the last make sure, that the cell is set in the inputs
        if output == 1:
            write("POSCAR%i" % i, mol, format="vasp", direct = "True")
        elif output == 0:
            write("-" , mol)

if __name__ == "__main__":
    main(sys.argv[1:])
