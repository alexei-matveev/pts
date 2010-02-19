#!/usr/bin/env python

import sys
import os
import aof
from aof.common import file2str
import ase
import numpy as np

force_cart_opt = False
name, params_file, mol_strings, init_state_vec, prev_results_file = aof.setup(sys.argv)

# TODO: setup circular re-naming to prevent accidental overwrites
logfile = open(name + '.log', 'w')
disk_result_cache = '%s.ResultDict.pickle' % name

# bring in custom parameters
extra_opt_params = dict()
params_file_str = file2str(params_file)
print params_file_str
exec(params_file_str)

# set up some objects
mi = aof.MolInterface(mol_strings, **params)

if force_cart_opt:
    # generate initial path using specified coord system but perform interpolation in cartesians
    mi, init_state_vec = mi.to_cartesians(beads_count)
calc_man = aof.CalcManager(mi, procs_tuple, to_cache=disk_result_cache, from_cache=prev_results_file)

if init_state_vec == None:
    init_state_vec = mi.reagent_coords

# setup searcher i.e. String or NEB
cos_type = cos_type.lower()
if cos_type == 'string':
    CoS = aof.searcher.GrowingString(init_state_vec, 
          calc_man, 
          beads_count,
          growing=False,
          parallel=True,
          reporting=logfile,
          max_sep_ratio=0.3)
elif cos_type == 'growingstring':
    CoS = aof.searcher.GrowingString(init_state_vec, 
          calc_man, 
          beads_count,
          growing=True,
          parallel=True,
          reporting=logfile,
          max_sep_ratio=0.3)
elif cos_type == 'neb':
    CoS = aof.searcher.NEB(init_state_vec, 
          calc_man, 
          spr_const,
          beads_count,
          parallel=True,
          reporting=logfile)
else:
    raise Exception('Unknown type: %s' % cos_type)


# callback function
def cb(x, tol=0.01):
    return aof.generic_callback(x, mi, CoS, params, tol=tol)

# print out initial path
cb(CoS.state_vec)

# hack to enable the CoS to print in cartesians, even if opt is done in internals
CoS.bead2carts = lambda x: mi.build_coord_sys(x).get_cartesians().flatten()

runopt = lambda: aof.runopt(opt_type, CoS, ftol, xtol, maxit, cb, maxstep=maxstep, extra=extra_opt_params)

# main optimisation loop
print runopt()

# get best estimate(s) of TS from band/string
tss = CoS.ts_estims()

a,b,c = CoS.path_tuple()
cs = mi.build_coord_sys(a[0])
import pickle
f = open("%s.path.pickle" % name, 'wb')
pickle.dump((a,b,c,cs), f)
f.close()

# print cartesian coordinates of all transition states that were found
print "Dumping located transition states"
for ts in tss:
    e, v, _, _ = ts
    cs = mi.build_coord_sys(v)
    print "Energy = %.4f eV" % e
    print cs.xyz_str()

refine_search=False
if refine_search:
    highest_ts = tss[-1]
    _, _, bead0, bead1 = highest_ts
    path_ref = np.array([bead0, bead1])
    beads_ref = 6
    CoS_ref = aof.searcher.NEB(path_ref, 
          calc_man, 
          spr_const,
          beads_ref,
          parallel=True,
          reporting=logfile)
    
    print runopt()
    tss_ref = CoS.ts_estims()


