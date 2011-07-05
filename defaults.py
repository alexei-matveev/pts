#!/usr/bin/env python
"""
This script holds the default parameters for the path searching algorithm
thus NEB and the different string

The values for the variables are read in after starting such a calculation
and than overwritten by explicitly given ones in the params file or in the
standard input
"""
from ase.calculators.vasp import Vasp
from ase.calculators.lj import LennardJones
import pts.config as config

default_params = {
    "method" : "string",     # what way, e.g. NEB, string, growingstring, searchingstring
    "opt_type" : "multiopt",  # the optimizer
    "pmax" : config.DEFAULT_PMAX,
    "pmin" : config.DEFAULT_PMIN,
    "cpu_architecture" : config.DEFAULT_TOPOLOGY,
    "name" : None,             # for output
    "calculator" : None,       # quantum chemistry calculator, e.g. Vasp or ParaGauss
    "placement" : None,
    "cell" : None,             # no cell given
    "pbc" : False,             # no periodic boundary conditions
    "mask" : None,             # freeze none of the coordinates
    "beads_count" : 7,          # 7 beads, thus 5 moving points on path
    "ftol" : 0.1,              # force convergence criteria
    "xtol" : 0.03,             # step convergence criteria, only used if f < ftol*5
    "etol" : 0.03,             # energy convergence criteria
    "maxit" : 35,              # maximal number of iterations
    "maxstep" : 0.1,           # maximal step size
    "spring" : 5.0,         # only for NEB: spring_constant
    "pre_calc_function" : None,
    "output_level" : 2,
    "output_path" : "workplace",
    "output_geo_format" : "xyz",
    "cache" : None          # where the results of the single point calculations will be stored
    }

default_calcs = {
    "default_vasp" : True,
    "default_lj" : True
    }

are_floats = ["ftol", "xtol", "etol", "maxstep", "spring"]
are_ints = ["maxit", "beads_count", "output_level", "pmin", "pmax"]

default_lj  = LennardJones(
  epsilon = 1.0,
  sigma = 1.0
  )

default_vasp = Vasp( ismear = 1
    , sigma  = 0.15
    , xc     = 'PW91'
    , isif   = 2
    , gga    = 91
    , enmax  = 400
    , ialgo  = 48
    , enaug  =  650
    , ediffg =  -0.02
    , voskown= 1
    , nelmin =  4
    , lreal  =  False
    , lcharg = False
    , lwave  = False
    , kpts   = (5,5,1)
    )
