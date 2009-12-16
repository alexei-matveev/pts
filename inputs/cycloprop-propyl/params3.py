# optimiser to use
opt_type = 'scipy_lbfgsb'

# mask of variables to freeze
# True => optimised
mask = None #[False for i in range(12*3)] + [True for i in range(3*3)]

# calculator 3-tuple: 
# (constructor, arguments (list), keyword arguments (dictionary), pre_calc_function)
calc_tuple = (ase.EMT, [], {}, None)

# scheduling information
# Field 1: list of processors per node e.g.
#  - [4] for a quad core machine
#  - [1] for a single core machine
#  - [1,1,1] for a cluster (or part thereof) with three single processor nodes
#  - [4,4,4,4] for a cluster with 4 nodes, each with 4 cores
# Field 2: max number of processors to run a single job on
# Field 3: normal number of processors to run a single job on
available, job_max, job_min = [4], 2, 1

procs_tuple = (available, job_max, job_min)

params = {
    # name of calculation, output files are named based on this
    'name': name,

    # calculator specification, see above
    'calculator': calc_tuple,

    # name of function to generate placement commant
    'placement': aof.common.place_str_dplace, 

    # cell shape, see ASE documentation
    'cell': None, 

    # cell periodicy, can be None
    'pbc': [False, False, False],

    # variables to mask, see above
    'mask': mask} 

beads_count = 8  # number of beads
tol = 0.01       # optimiser force tolerance
maxit = 20       # max iterations
spr_const = 5.0  # NEB spring constant (ignored for string)
growing = False  # is the string growing

