#!/usr/bin/env python

import sys
import os
import aof
import ase
from aof.common import file2str


# Deal with environment variables
# FIXME: messy
args = sys.argv[1:]
assert len(args) == 3, "Usage reactant, product, parameters.py"
reagent_files = args[0:2]
names = [os.path.splitext(f)[0] for f in reagent_files]
params_file = args[2]
name = "_to_".join(names) + "_with_" + os.path.splitext(params_file)[0]
print "Search name:", name

# bring in custom parameters parameters
exec(file2str(params_file))

# set up some objects
mol_strings = aof.read_files(reagent_files)
mi          = aof.MolInterface(mol_strings, params)
calc_man    = aof.CalcManager(mi, procs_tuple)

# setup searcher i.e. String or NEB
CoS = aof.searcher.NEB(mi.reagent_coords, 
          calc_man, 
          spr_const,
          beads_count,
          parallel=True)

# callback function
mycb = lambda x: aof.generic_callback(x, mi, CoS, params)


# main optimisation loop
while True:
    aof.runopt(opt_type, CoS, tol, maxit, mycb)

    if not growing or not CoS.grow_string():
        break
    
# get best estimate(s) of TS from band/string
tss = CoS.ts_estims()

# print cartesian coordinates of all transition states that were found
print "Dumping located transition states"
for ts in tss:
    e, v = ts
    cs = mi.build_coord_sys(v)
    print cs.xyz_str()

