from aof.coord_sys import ZMatrix2, XYZ
from aof.common import file2str

spec = "temp.zmt"
xyz1 = "reactants.xyz.t"
xyz2 = "products.xyz.t"
zmt1 = "reactants.zmt"
zmt2 = "products.zmt"

z = ZMatrix2(file2str(spec))
x1 = XYZ(file2str(xyz1))
x2 = XYZ(file2str(xyz2))


z.set_cartesians(x1.get_cartesians())
open(zmt1, 'w').write(repr(z))
z.set_cartesians(x2.get_cartesians())
open(zmt2, 'w').write(repr(z))


