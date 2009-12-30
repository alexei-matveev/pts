from aof.coord_sys import ZMatrix2, XYZ
from aof.common import file2str

spec = "reactants_spec.zmt"
xyz1 = "reactants.xyz.t.t"
xyz2 = "products.xyz.t.t"

z = ZMatrix2(file2str(spec))
x1 = XYZ(file2str(xyz1))
x2 = XYZ(file2str(xyz2))


z.set_cartesians(x1.get_cartesians())
print repr(z)
z.set_cartesians(x2.get_cartesians())

print repr(z)

