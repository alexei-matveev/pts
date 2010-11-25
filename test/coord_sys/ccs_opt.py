from pts.coord_sys import *
from pts.common import file2str
import ase
from pts.gaussian import Gaussian
from ase.io.trajectory import PickleTrajectory

import sys

#  contents of the variables in the following are arbitrary since they are 
# set based on the cartesian objects. Make sure, however, that the z-matrix 
# variables are non-zero, since this can cause a divide by zero error.

fn = sys.argv[1]
print "Filename", fn

ccs = ComplexCoordSys(file2str(fn))
#print ccs._coords
#print ccs.get_internals()
#exit()

g = Gaussian()
ccs.set_calculator((Gaussian, [], {'charge': 0, 'mult': 3}))

opt = ase.LBFGS(ccs)
pt = PickleTrajectory("test.traj", mode='w')
def cb():
    print ccs.xyz_str()
    print "internals", ccs.get_internals().round(2)
    pt.write(ccs.atoms.copy())

opt.attach(cb)
print ccs.get_internals().round(2)

opt.run(fmax=0.1)


