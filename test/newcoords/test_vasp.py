#!/usr/bin/env python
"""
Test example with VASP:

A H-Atom is moving on a small (fixed) Pd surface
"""
from pts.path_searcher import pathsearcher
from ase.atoms import Atoms
from ase.calculators.vasp import Vasp
from pts.cfunc import Justcarts, Masked
from numpy import array
from pts.qfunc import QFunc
from pts.func import compose
from sys import argv


# The ASE atoms object for calculating the forces
PdH = Atoms("HPd4")

# Vasp calculator, needs VASP_PP_PATH and VASP_COMMAND or VASP_SCRIPT
calculator = Vasp( ismear =  1
           , sigma  =  0.15
           , xc     = 'PW91'
           , isif   =  2
           , gga    =  91
           , enmax  =  400
           , ialgo  =  48
           , enaug  =  650
           , ediffg =  -0.02
           , voskown=  1
           , nelmin =  4
           , lreal  = False
           , lcharg = False
           , lwave  = False
           , kpts   = (5,5,1)
           )

# Atoms object needs to have all these in order to be able to run correctly:
PdH.set_calculator(calculator)
PdH.set_pbc(True)
sc =  5.59180042562320832916
cell =  [[1.0000000000000000,  0.0000000000000000,  0.0000000000000000],
         [0.5000000000000000,  0.8660254037844386,  0.0000000000000000],
         [0.0000000000000000,  0.0000000000000000,  1.0000000000000000]]
cell = array(cell) * sc
PdH.set_cell(cell)

# Do calculation in Cartesian coordinates
fun1 = Justcarts()

# PES in cartesian coordiantes:
pes = QFunc(PdH, PdH.get_calculator())

def reduce(vec, mask):
    """
    Function for generating starting values.
    Use a mask to reduce the complete vector.
    """
    vec_red = []
    for i, m in enumerate(mask):
         if m:
             vec_red.append(vec[i])
    return array(vec_red)

# The starting geometries in flattened Cartesian coordinates
min1 = array([ 1.3979501064058020,  0.8071068702470560,  1.0232994778890470,
  0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
  1.3979501064058020, 2.4213206107411671, 0.0000000000000000,
  2.7959002128116039, 0.0000000000000000, 0.0000000000000000,
  4.1938503192174057, 2.4213206107411671, 0.0000000000000000])

min2 = array([  2.7959002128116039,  1.6142137404941110,  1.0232994778890470,
  0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
  1.3979501064058020, 2.4213206107411671, 0.0000000000000000,
  2.7959002128116039, 0.0000000000000000, 0.0000000000000000,
  4.1938503192174057, 2.4213206107411671, 0.0000000000000000])

# Fix everything but the H atom
mask = [True] * 3 + [False] * 12

# Merge the functions
func = Masked(fun1, mask, min1)

# The starting geoemtries have also to be reduced
min1 = reduce(min1, mask)
min2 = reduce(min2, mask)

# PES in "internal" coordinates:
pes = compose(pes, func)

# init path contains the two minima
init_path = [min1, min2]

# Let pathseracher optimize the path
pathsearcher(PdH, init_path, func, ftol = 0.1, maxit = 12, beads_count = 5, output_level = 0)



