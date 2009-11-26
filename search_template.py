#!/usr/bin/env python

import sys

import aof
import ase

reagent_files = ["cyclopropane.xyz.t", "propylene.xyz.t"]

# calculator 3-tuple: 
# (constructor, arguments (list), keyword arguments (dictionary))
calc_tuple = (ase.EMT, [], {})

available, job_max, job_min = 1, 1, 1

procs_tuple = (available, job_max, job_min)

params = {
    'name': 'my_calculation',
    'calculator': calc_tuple,
    'mask': None, # variables to mask, might also be specified in the files
    'processors': procs_tuple} # number of processors to run on


beads_count = 8
tol = 0.5 # optimiser force tolerance
maxit = 8 # max iterations
spr_const = 5.0 # NEB spring constant

mol_strings = aof.read_files(reagent_files)

mi = aof.MolInterface(mol_strings, params)

calc_man = aof.CalcManager(mi, params)

neb = aof.searcher.NEB(mi.reagent_coords, 
          mi.geom_checker, 
          calc_man, 
          spr_const, 
          beads_count,
          parallel=True)

aof.dump_steps(neb)

# callback function
mycb = lambda x: aof.generic_callback(x, mi, neb, params)


import cosopt.lbfgsb as so

opt, energy, dict = so.fmin_l_bfgs_b(neb.obj_func,
                                  neb.get_state_as_array(),
                                  fprime=neb.obj_func_grad,
                                  callback=mycb,
                                  pgtol=tol,
                                  maxfun=maxit)


