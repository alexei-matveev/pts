"""Template for a script to facilitate the parameterisation of a z-matrix from
XYZ coordinates. See code.
"""

print __doc__

from aof.coord_sys import ZMatrix2, XYZ
from aof.common import file2str

# zmatrix
spec = "temp.zmt"

# input xyz format files
xyz1 = "reactants.xyz.t"
xyz2 = "products.xyz.t"

# output z-matrices
zmt1 = "reactants.zmt"
zmt2 = "products.zmt"

try:
    # read in files
    z = ZMatrix2(file2str(spec))
    x1 = XYZ(file2str(xyz1))
    x2 = XYZ(file2str(xyz2))

except IOError:
    print "You must create the relevant input files! See code."
    exit(1)

# Cartesian -> z-matrix and write files
z.set_cartesians(x1.get_cartesians())
open(zmt1, 'w').write(repr(z))
z.set_cartesians(x2.get_cartesians())
open(zmt2, 'w').write(repr(z))


