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
    'optimizer': 'l_bfgs_b', 
    'calculator': calc_tuple,
    'mask': None, # variables to mask, might also be specified in the files
    'beads_count': 8, 
    'tol': 0.5, # optimiser force tolerance
    'maxit': 8, # max iterations
    'spr_const': 5.0, # NEB spring constant
    'processors': procs_tuple} # number of processors to run on

# perform some tests, hidden from user
aof.cleanup(globals())

mol_strings = aof.read_files(reagent_files)

mi = aof.MolInterface(mol_strings, params)

calc_man = aof.CalcManager(mi, params)

aof.neb_calc(mi, calc_man, params)


