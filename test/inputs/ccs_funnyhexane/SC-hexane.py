zmts = """C
S 1 var1
H 1 var2 2 var3

var1    1.0
var2    1.0
var3    1.0
"""

xyz_raw = common.file2str('SC-hexane.geom.t')
xyzs = '\n'.join(xyz_raw.splitlines()[:-3])
xyz = XYZ(xyzs)
zmt = ZMatrix2(zmts, RotAndTrans())

ccs = ccsspec([xyz, zmt], carts=XYZ(xyz_raw))

