#!/usr/bin/env python

import sys
import os
import pickle

import numpy as np

import ase

import pts
from pts.common import file2str
from os.path import exists
from pts.tools import pickle_path

name, params_file, mol_strings, init_state_vec, prev_results_file, overrides, inputdir = pts.setup(sys.argv)

# TODO: setup circular re-naming to prevent accidental overwrites
logfile = open(name + '.log', 'w')
disk_result_cache = '%s.ResultDict.pickle' % name

# setup all default parameters
refine_search=False
force_cart_opt = False
beads_count_refine = 6

# bring in custom parameters
extra_opt_params = dict()
params_file_str = file2str(params_file)

print params_file_str
exec(params_file_str)

# HACK to allow locally defined calc
calcfile = 'calc.txt'
local_calc = False
if exists(calcfile):
    params['calculator'] = eval(file2str(calcfile))
    local_calc = True
elif exists(os.path.join(inputdir, calcfile)):
    params['calculator'] = eval(file2str(os.path.join(inputdir, calcfile)))
    local_calc = True
if local_calc:
    print "Using locally defined calculator:", str(params['calculator'])

# overwrite parameters as necessary
print "The following overrides were specified..."
print "(Start)"
print overrides
print "(End)"
exec(overrides)

# set up some objects
mi = pts.MolInterface(mol_strings, **params)

if force_cart_opt:
    # generate initial path using specified coord system but perform interpolation in cartesians
    mi, init_state_vec = mi.to_cartesians(beads_count)
calc_man = pts.CalcManager(mi, procs_tuple, to_cache=disk_result_cache, from_cache=prev_results_file)

if init_state_vec == None:
    init_state_vec = mi.reagent_coords

# setup searcher i.e. String or NEB
method = method.lower()
if method == 'string':
    CoS = pts.searcher.GrowingString(init_state_vec,
          calc_man, 
          beads_count,
          growing=False,
          parallel=True,
          reporting=logfile,
          freeze_beads=False,
          head_size=None,
          max_sep_ratio=0.3)
elif method == 'growingstring':
    CoS = pts.searcher.GrowingString(init_state_vec,
          calc_man, 
          beads_count,
          growing=True,
          parallel=True,
          reporting=logfile,
          freeze_beads=False,
          head_size=None,
          max_sep_ratio=0.3)
elif method == 'searchingstring':
    CoS = pts.searcher.GrowingString(init_state_vec,
          calc_man, 
          beads_count,
          growing=True,
          parallel=True,
          reporting=logfile,
          max_sep_ratio=0.3,
          freeze_beads=True,
          head_size=None, # has no meaning for searching string
          growth_mode='search')

elif method == 'neb':
    CoS = pts.searcher.NEB(init_state_vec,
          calc_man, 
          spr_const,
          beads_count,
          parallel=True,
          reporting=logfile)
else:
    raise Exception('Unknown type: %s' % method)

CoS.arc_record = open("archive.pickle", 'w')
pickle.dump("Version 0.1", CoS.arc_record, protocol=2)
pickle.dump(mi.build_coord_sys(init_state_vec[0]), CoS.arc_record, protocol=2)

cb_count_debug = 0
while True:
    # callback function
    def cb(x, tol=0.01):

        global cb_count_debug
        pickle_path(mi, CoS, "%s.debug%d.path.pickle" % (name, cb_count_debug))
        cb_count_debug += 1
        return pts.generic_callback(x, mi, CoS, tol=tol, **params)

    # print out initial path
    cb(CoS.state_vec)

    runopt = lambda CoS_: pts.runopt(opt_type, CoS_, ftol, xtol, etol, maxit, maxstep=maxstep, callback=cb, **extra_opt_params)

    # main optimisation loop
    print runopt(CoS)

    # get best estimate(s) of TS from band/string
    tss = CoS.ts_estims()

    # write out path to a file
    pickle_path(mi, CoS, "%s.path.pickle" % name)

    # print cartesian coordinates of all transition states that were found
    print "Dumping located transition states"
    for ts in tss:
        e, v, s0, s1,_ ,bead0_i, bead1_i = ts
        cs = mi.build_coord_sys(v)
        print "Energy = %.4f eV, between beads %d and %d." % (e, bead0_i, bead1_i)
        print cs.xyz_str()

    if not refine_search or beads_count_refine < 3:
        break

    refine_search = False
    name = name + "-gen1"
    params['name'] = name

    highest_ts = tss[-1]
    _, _, _, _,_, bead0_i, bead1_i = highest_ts
    if bead0_i == bead1_i:
        print "No true transition state was found, skipping refined search."
        break

    bead0 = CoS.state_vec[bead0_i]
    bead1 = CoS.state_vec[bead1_i]

    max_change = np.abs(bead1 - bead0).max()
    print "Performing refined search between beads %d and %d." % (bead0_i, bead1_i)
    print "Max change in any optimisation coordinate", max_change

    path_ref = np.array([bead0, bead1])
    CoS = pts.searcher.NEB(path_ref,
          calc_man, 
          spr_const,
          beads_count_refine,
          parallel=True,
          reporting=logfile)

    # write out path to a file
    pickle_path(mi, CoS, "%s.path.pickle" % name)


CoS.arc_record.close()

