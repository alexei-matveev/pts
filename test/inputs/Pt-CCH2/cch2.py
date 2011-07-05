# =================================== #
# ComplexCoordSys (CCS) specification #
# =================================== #

# string containing z-matrix part of CCS
zmts = common.file2str('template.zmt')

# RAW cartesians which will be used to populate all variables in CCS
xyz_raw = common.file2str('cch2.xyz')

# string containing cartesian part of CCS
xyzs = common.file2str('cch2-cartsection.xyz')

# set up internals parts of CCS
xyz = XYZ(xyzs)
zmt = ZMatrix2(zmts, anchor=RotAndTrans()) # zmt must have an anchor

# object to specify the complex coordinate system
mask = [True for i in range(9)] + [True for i in range(6)] + [False for i in range(6*3)]
ccs = ccsspec([zmt, xyz], carts=XYZ(xyz_raw), mask=mask)



