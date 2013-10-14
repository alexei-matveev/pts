#
# Working units are  angstroms and eV. Other units are  in A and/or eV
# are below. An expression like
#
#     x * BOHR
#
# should give the  geometry in A when x is in  Bohr. Similarly, when g
# is in Hartree / Bohr and expression like
#
#     g * (HARTREE / BOHR)
#
# will convert that into the working units (A and eV).
#
from numpy import pi
import ase

angstrom = 1.0
eV = 1.0

#
# ParaGauss  calculator uses  ASE  units when  converting from  atomic
# units to  eV/A. If we ever want  to convert them back  we better use
# the same constant here too:
#
Bohr = ase.Bohr                 # ~0.5292 A
Hartree = ase.Hartree           # ~27.21 eV

kcal = Hartree / 627.509469
degree = pi / 180.
