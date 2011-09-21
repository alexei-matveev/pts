#!/usr/bin/env python
"""
The behavior of the path searching routines might be affected by a large number
of parameters. To see the complete set of them say path_searcher --defaults

The default values for the variables are read in after starting such a calculation
and than overwritten by explicitly given ones in the params file or in the
standard input

The parameters may be set in two ways: a) by specifying a

  parameter = value

pair in the so called "paramfile" at location specified in the command line

  --paramfile location

or b) from the command line by specifying a pair

  --parameter value

Only in the case of calculator the value has to be replaces by the location (and name)
of the calculator input file. This has been already explained in the path searcher
general help text.
For some parameter it might be rather unpractically to give them in the command line (like
the mask parameter. If it is wanted anyhow they have to be given in "" as a string to anounce to
the progam that here several pieces belong to together.

For example

  --paramfile params.py --method neb

would set the parameter "method" to "neb". Command line options have higher
precedence than the settings in "paramfile", so that setting

  method = "string"

in the file "params.py" located in the current directory would have no effect.

The example for mask would be:

  --mask "[True, True, False, True, False, True]"

would fix the third and fifth coordinate of a system with 6 coordinates.

There exists:
Parameter       short description
------------------------------------------------
 "method"      what calculation is really wanted, like neb, string,
               growingstring or searchingstring, if using paratools <method> this
               is set automatically
 "opt_type"    what kind of optimizer is used for changing the geometries
               of the string, as default the new multiopt is used for the
               string methods, while neb is reset to ase_lbgfs
 "pmax"        maximal number of CPUs per bead, with our workarounds normaly
               only indirect used
 "pmin"        minimal number of CPUs per bead, with our workarounds normaly
               only indirect used
 "cpu_architecture" descriebes the computer architecture, which should be used,
                    with our workaround only indirect used, pmax, pmin and
                    cpu_architecture should be adapted to each other
 "name"        the name of the calculation, appears as basis of the names
               for all the output, needn't be set, as a default it takes
               the cos_type as name
 "calculator"  the quantum chemstry program to use, like Vasp or ParaGauss
 "placement"   executable function for placing processes on beads, only
               used for advanced calculations
 "cell"        the cell in which the molecule is situated
 "pbc"         which cell directions have periodic boundary conditions
 "mask"        which of the given geometry variables are supposed to be
               changed (True) and which should stay fix during the
               calculation (False), should be a string containing for each
               of the variables the given value. The default does not set
               this variable and then all of them
               are optimized
 "beads_count" how many beads (with the two minima) are there at maximum
               (growingstring and searchingstring start with less)
 "ftol"        the force convergence criteria, calculation stops if
               RMS(force) < ftol
 "xtol"        the step convergence criteria, only used if force has at
               least ftol * 10
 "etol"        energy convergence criteria, not really used
 "maxit"       if the convergence criteria are still not met at maxit
               iterations, the calculation is stopped anyhow
 "maxstep"     the maximum step a path can take
 "spring"      the spring constant, only needed for neb
 "pre_calc_function"  function for precalculations, for gaussian ect.
 "output_level" the amount of output is decided here
                   0  minimal output, not recommended
                      only logfile, geometries of the beads for the last
                      iteration (named Bead?) and the output needed for
                      the calculation to run
                   1  recommended output level (default) additional the
                      ResultDict.pickle (usable for rerunning or extending the
                      calculation without having to repeat the quantum
                      chemical calculations) and a path.pickle of the last
                      path, may be used as input for some other tools,
                      stores the "whole" path at it is in a special foramt
                   2  additional a path.pickle for every path, good if
                      development of path is
                      wanted to be seen (with the additional tools)
                   3  some more output in every iteration, for debugging ect.

 "output_path" place where most of the output is stored, thus the
               working directory is not filled up too much
 "output_geo_format"  ASE format, to write the outputgeometries of the
                      last iteration to is xyz as default, but can be changed
                      for example to gx or vasp (POSCAR)

"""
from ase.calculators.vasp import Vasp
from ase.calculators.lj import LennardJones
import pts.config as config


def info_params():
    print __doc__

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
