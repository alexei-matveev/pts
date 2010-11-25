import sys
import pts
import pts.coord_sys as csys

file = sys.argv[1]
s = aof.common.file2str(file)

if csys.ZMatrix2.matches(s):
    mol = csys.ZMatrix2(s)

elif csys.XYZ.matches(s):
    mol = csys.XYZ(s)

elif csys.ComplexCoordSys.matches(s):
    mol = csys.ComplexCoordSys(s)

else:
    raise MolInterfaceException("Unrecognised geometry string:\n" + s)

print "Mol has type:", type(mol)
print "mask:", mol._var_mask
print "dims:", mol.dims

