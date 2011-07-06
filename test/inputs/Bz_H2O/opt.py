from ase import *
from pts import *

# read in starting geometry
cs = coord_sys.XYZ(common.file2str("Bz_H2O-2.xyz"))

# freeze some variables
cs.set_var_mask([False for i in range(12*3)] + [True for i in range(3*3)])

# setup quantum chem program (calculator)
g = gaussian.Gaussian, [], {'nprocs': 2, 'basis': '3-21G'}
cs.set_calculator(g)

# setup optimiser
opt = LBFGS(cs)

# record each step in optimisation
opt.attach(PickleTrajectory('test.traj', 'w', cs.atoms))

# print out each step in optimisation
def f():
    print repr(cs)
opt.attach(f)

# run optimiser
opt.run(fmax=0.01)

# view final geometry
#view(cs.atoms)

# at comamnd line do:
# ag test.traj

