#!/usr/bin/python
"""
This script holds the default parameters for the path searching algorithm
thus NEB and the different string

The values for the variables are read in after starting such a calculation
and than overwritten by explicitly given ones in the params file or in the
standard input
"""
from ase.calculators import Vasp, LennardJones

default_params = {
    "cos_type" : "string",     # what way, e.g. NEB, string, growingstring, searchingstring
    "opt_type" : "multiopt",  # the optimizer
    "pmax" : 100,
    "pmin" : 1,
    "cpu_architecture" : [1],
    "name" : None,             # for output
    "calculator" : None,       # quantum chemistry calculator, e.g. Vasp or ParaGauss
    "placement" : None,
    "cell" : [[ 1.,  0.,  0.],
             [ 0.,  1.,  0.],
             [ 0.,  0.,  1.]],   # ase default cell
    "pbc" : False,             # no periodic boundary conditions
    "mask" : None,             # freeze none of the coordinates
    "beads_count" : 7,          # 7 beads, thus 5 moving points on path
    "ftol" : 0.1,              # force convergence criteria
    "xtol" : 0.03,             # step convergence criteria, only used if f < ftol*5
    "etol" : 0.03,             # energy convergence criteria
    "maxit" : 35,              # maximal number of iterations
    "maxstep" : 0.2,           # maximal step size
    "str_const" : 5.0,         # only for NEB: spring_constant
    "growing" : False,          # only for the strings: if the string is growing, automatically
                               #  changed if cos_type requires it
    "pre_calc_function" : None,
    "output_level" : 1,
    "output_path" : "workplace",
    "output_geo_format" : "xyz"
    }

default_calcs = {
    "default_vasp" : True,
    "default_lj" : True
    }

are_floats = ["ftol", "xtol", "etol", "maxstep", "str_const"]
are_ints = ["maxit", "beads_count", "output_level"]

default_lj  = LennardJones(
  epsilon = 1.0,
  sigma = 1.0
  )

default_vasp = Vasp( ismear = '1'
    , sigma  = '0.15'
    , xc     = 'PW91'
    , isif   = '2'
    , gga    = '91'
    , enmax  = '400'
    , ialgo  = '48'
    , enaug  =  '650'
    , ediffg =  '-0.02'
    , voskown= '1'
    , nelmin =  '4'
    , lreal  = '.FALSE.'
    , lmaxpaw = '0'
    , lcharg = '.FALSE.'
    , lwave  = '.FALSE.'
    , kpts   = [5,5,1]
    )
