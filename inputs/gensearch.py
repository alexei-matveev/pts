#!/usr/bin/env python

import sys
import os
import aof
from aof.common import file2str
import ase
from numpy import zeros # temporary

name, params_file, mol_strings, init_state_vec, prev_results_file = aof.setup(sys.argv)

# TODO: setup circular re-naming to prevent accidental overwrites
logfile = open(name + '.log', 'w')
disk_result_cache = '%s.ResultDict.pickle' % name

# bring in custom parameters

params_file_str = file2str(params_file)
print params_file_str
exec(params_file_str)

# set up some objects
mi          = aof.MolInterface(mol_strings, params)
calc_man    = aof.CalcManager(mi, procs_tuple, to_cache=disk_result_cache, from_cache=prev_results_file)

# setup searcher i.e. String or NEB
if init_state_vec == None:
    init_state_vec = mi.reagent_coords

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

# hack to enable the CoS to print in cartesians
CoS.bead2carts = lambda x: mi.build_coord_sys(x).get_cartesians().flatten()

# callback function
def cb(x, tol=0.01):
    return aof.generic_callback(x, mi, CoS, params, tol=tol)

runopt = lambda: aof.runopt(opt_type, CoS, tol, maxit, cb, maxstep=maxstep)

# main optimisation loop
print runopt()

# get best estimate(s) of TS from band/string
tss = CoS.ts_estims()

# print cartesian coordinates of all transition states that were found
print "Dumping located transition states"
for ts in tss:
    e, v = ts
    cs = mi.build_coord_sys(v)
    print cs.xyz_str()

