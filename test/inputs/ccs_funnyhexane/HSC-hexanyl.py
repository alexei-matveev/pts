# =================================== #
# ComplexCoordSys (CCS) specification #
# =================================== #

# string containing z-matrix part of CCS
zmts = """C
S 1 var1
H 1 var2 2 var3

var1    1.0
var2    1.0
var3    1.0
"""

# RAW cartesians which will be used to populate all variables in CCS
xyz_raw = common.file2str('HSC-hexanyl.geom.t')

# string containing cartesian part of CCS
xyzs = '\n'.join(xyz_raw.splitlines()[:-3])

# set up internals parts of CCS
xyz = XYZ(xyzs)
zmt = ZMatrix2(zmts, anchor=RotAndTrans()) # zmt must have an anchor

# object to specify the complex coordinate system
ccs = ccsspec([xyz, zmt], carts=XYZ(xyz_raw), mask=None)

