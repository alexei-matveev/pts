#!/usr/bin/env python

import sys

import aof
import ase

reagent_files = ["Bz_H2O-1.xyz", "Bz_H2O-2.xyz"]

# calculator 3-tuple: 
# (constructor, arguments (list), keyword arguments (dictionary))
calc_tuple = (aof.qcdrivers.Gaussian, [], {'basis': '3-21G'})

available, job_max, job_min = [4], 2, 1

procs_tuple = (available, job_max, job_min)

mask = [False for i in range(12*3)] + [True for i in range(3*3)]
params = {
    'name': 'my_calculation',
    'calculator': calc_tuple,
    'placement': aof.common.place_str_dplace,
    'mask': mask} # variables to mask, might also be specified in the files

beads_count = 8
tol = 0.1 # optimiser force tolerance
maxit = 8 # max iterations
spr_const = 5.0 # NEB spring constant

mol_strings = aof.read_files(reagent_files)

mi = aof.MolInterface(mol_strings, params)

calc_man = aof.CalcManager(mi, procs_tuple)

growing = True
CoS = aof.searcher.GrowingString(mi.reagent_coords, 
          calc_man, 
          beads_count,
          rho = lambda x: 1,
          growing=growing,
          parallel=True,
          head_size=None)

aof.dump_steps(CoS)

# callback function
mycb = lambda x: aof.generic_callback(x, mi, CoS, params)

import cosopt.lbfgsb as so

while True:
    opt, energy, dict = so.fmin_l_bfgs_b(CoS.obj_func,
                                  CoS.get_state_as_array(),
                                  fprime=CoS.obj_func_grad,
                                  callback=mycb,
                                  pgtol=tol,
                                  maxfun=maxit)
    if not growing or not CoS.grow_string():
        break
    
#    print "******** String Grown to", CoS.beads_count, "beads ********"

print opt
print dict
