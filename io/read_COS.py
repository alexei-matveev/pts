#!/usr/bin/env python
"""
GEOMETRY

Geometries can be provided as files, which can be interpreted by ASE. This
includes xyz-, POSCAR, and gx-files. File format is in part determied from the
file name or extension, e.g. POSCAR and gx-files by the presence of "POSCAR" or
"gx" substrings. If the format is not extractable from the filename it can be
given as an addtional parameter as --format <format> . The Values for <format>
are gx or are in the list of short names in the ASE documentation.

If the calculation should be done in internal (or mixed coordinate system) one gives
the geometries in cartesians  and specifies additionally the zmatrix/zmatrices
They will then be given by writing --zmatrix zmat_file. It is supposed that the first atom
of each zmatrix is the uppermost not yet used atom of all atoms given.

ZMATRIX

A zmatrix may look something like:

""
C
H 1 var1
H 1 var2 2 var3
H 1 var4 2 var5 3 var6
H 1 var7 2 var8 4 var9
""

The first element of each line is supposed to be the name of the atom. Then
follows the connectivity matrix, where for each appearing variable a name is
given. The connectivities are given by the line number of the atom, starting
with 1. First variable is always the length, second the angle and third the
dihedral angle. If variable names appear more than once these variables are set
to the same value. Giving a variable name as another variable name with a
"-" in front of it, the values are always set to the negative of the other
variable.

ADDITIONAL GEOMETRY RELATED INFORMATIONS:

Additional informations can be taken from the minima ASE inputs. The ASE atoms
objects may contain more informations than only the chemical symbols and the
geometries of the wanted object. For example if reading in POSCARs there are
additional informations as about the cell (pbc would be also set automatically
to true in all directions). This informations can also be read in,
if they are available. Then they can be used for specifiying the quantum chemical
calculator. So VASP for instance needs a proper cell.
They are only used, if these variables are not provided another way (by direct
setting them for instance).

Additionally ASE can hold some constraints, which may be
taken from a POSCAR or set directly. Some of them can be also used to generate
a mask. This is only done if cartesian coordinates are used further on. They are
not used if a mask is specified directely.
"""
from ase.io import read as read_ase
from pts.common import file2str
from numpy import loadtxt
from copy import deepcopy
from cmdline import get_calculator
import re

geo_params = ["format", "calculator", "mask", "pbc", "cell"]

def info_geometries():
    print __doc__


def read_geos_from_file(geom_files, format):
    """
    Read in geometries from ASE readable file
    returns one atoms object (with geometry of first minima set)
    and a list of geometries
    """
    res = [read_ase(st1, format = format) for st1 in geom_files]
    atom = res[0]
    geom = [r.get_positions() for r in res]
    return atom, geom

def read_geos_from_file_more(geom_files, format):
    """
    Read in geometries from ASE readable file
    returns one atoms object (with geometry of first minima set)
    and a list of geometries
    """
    res = []
    for st1 in geom_files:
       index = 0
       while True:
           try:
               r1 = read_ase(st1, format = format, index = index)
           except IndexError:
               break
           res.append(r1)
           index += 1
    atom = res[0]
    geom = [r.get_positions() for r in res]
    return atom, geom

def read_zmt_from_file(zmat_file):
    """
    Read zmatrix from file
    """
    zmat_string = file2str(zmat_file)
    return read_zmt_from_string(zmat_string)

def read_zmt_from_gx(gx_file):
    """
    Read zmat out from a string, convert to easier to interprete results

    give back more results than really needed, to have it easier to use
    them later on

    OUTPUT: [<Name of Atoms>], [Connectivity matrix, format see ZMat input from zmat.py],
            [<variable numbers, with possible repitions>], how often a variable was used more than once,
            [<dihedral angles variable numbers>],
            (number of Cartesian coordinates covered with zmt, number of interal coordinates of zmt),
            [<mask for fixed Atoms>]
    """
    from ase.gxfile import gxread
    from ase.data import chemical_symbols

    atnums, __, __, inums, iconns, ivars, __, __, __, __ = gxread(gx_file)
    # For consistency with the rest, transform the numbers to symbols
    symbols = [chemical_symbols[a] for a in atnums]
    assert(list(inums) == range(1, len(atnums) + 1))


    iconns2 = []
    var_names = []
    var_names_gx = {}
    mult = 0
    dih_names = []
    fixed = []
    j = 1

    #Iconns should contain connectivities, ivars tell
    # what to do with them
    for ic, iv in zip(iconns, ivars):
        a, b, c = ic
        new_vars = []

        # Three atoms are special, because they do not
        # contain 3 variables
        if a == 0:
           t = ()
        elif b == 0:
           t = (a-1,)
           new_vars = [iv[0]]
        elif c == 0:
           t = (a-1, b-1,)
           new_vars = iv[:-1]
        else:
           t = (a-1, b-1, c-1)
           new_vars = iv
        # connectivities from gx also have the wrong basis
        # change them and give back as our connectivity matrix
        iconns2.append(t)

        # Now find out what is to be done to the corresponding
        # variables
        for i, nv in enumerate(new_vars):

             if nv in var_names_gx.keys():
                 # They have appeared already once, thus
                 # should go to With_equals
                 var_names.append(var_names_gx[nv])
                 mult = mult + 1
             else:
                 # New variable
                 var_names.append(j)
                 if i == 2:
                     # For finding the shortest path
                     # Attention here numbering starts with 0
                     # not with 1 as for the var_names
                     dih_names.append(j - 1)
                 if nv == 0:
                     # Normally masked ones are only
                     # considered lateron, gx already
                     # addresses them by giving a 0 to
                     # as their variable number
                     fixed.append(j)
                 else:
                     var_names_gx[nv] = j
                 j = j + 1

    iconns = iconns2

    # create also a mask if there is something about it
    if fixed == []:
        mask = None
    else:
        mask = [ i not in fixed for i in var_names]

    return symbols, iconns, var_names, mult, dih_names, (len(symbols) * 3, j), mask

def read_zmt_from_string(zmat_string):
    """
    Read zmat out from a string, convert to easier to interprete results

    give back more results than really needed, to have it easier to use
    them later on

    OUTPUT: [<Name of Atoms>], [Connectivity matrix, format see ZMat input from zmat.py],
            [<variable numbers, with possible repitions>], how often a variable was used more than once,
            [<dihedral angles variable numbers>],
            (number of Cartesian coordinates covered with zmt, number of interal coordinates of zmt)

    Thus for example:
            (['Ar', 'Ar', 'Ar', 'Ar'], [(), (0,), (0, 1), (1, 0, 2)], [0, 0, 1, 0, -1, 2], 3, [2], (12, 6))

    >>> str1 = "H\\nO 1 ho1\\nH 2 ho2 1 hoh\\n"

    >>> str2 = "H\\nO 1 ho1\\nH 2 ho2 1 hoh\\n\\n"

    >>> strAr = 'Ar\\nAr 1 var1\\nAr 1 var2 2 var3\\n \\
    ...          Ar 2 var4 1 var5 3 var6\\n           \\
    ...          \\nvar1 = 1.0\\nvar2 = 1.0\\nvar3 = 1.0\\nvar4 = 1.0\\nvar5 = 1.0\\nvar6 = 1.0\\n'
    >>> strAr2 = 'Ar\\nAr 1 var1\\nAr 1 var2 2 var3\\nAr 2 var4 1 var5 3 var6\\n'
    >>> strAr3 = 'Ar\\nAr 1 var1\\nAr 1 var1 2 var2\\nAr 2 var1 1 -var2 3 var6\\n'


    # read in small H2O, no dihedrals in there
    >>> read_zmt_from_string(str1)
    (['H', 'O', 'H'], [(), (0,), (1, 0)], [1, 2, 3], 0, [], (9, 3))

    # test with an extra blankline
    >>> read_zmt_from_string(str2)
    (['H', 'O', 'H'], [(), (0,), (1, 0)], [1, 2, 3], 0, [], (9, 3))

    # A bit larger Argon Cluster
    >>> read_zmt_from_string(strAr2)
    (['Ar', 'Ar', 'Ar', 'Ar'], [(), (0,), (0, 1), (1, 0, 2)], [1, 2, 3, 4, 5, 6], 0, [5], (12, 6))

    # old format (with random set variables values to omit)
    >>> read_zmt_from_string(strAr)
    (['Ar', 'Ar', 'Ar', 'Ar'], [(), (0,), (0, 1), (1, 0, 2)], [1, 2, 3, 4, 5, 6], 0, [5], (12, 6))

    # reduce variables, set all length to the same value and have also the angles be their negative
    >>> read_zmt_from_string(strAr3)
    (['Ar', 'Ar', 'Ar', 'Ar'], [(), (0,), (0, 1), (1, 0, 2)], [1, 1, 2, 1, -2, 3], 3, [2], (12, 3))
    """

    lines = zmat_string.split("\n")

    # data to extract from the lines
    names = []
    matrix = []
    var_names = {}
    var_numbers = []
    multiplicity = 0
    nums_atom = 0
    dihedral_nums = []
    var_count = -1

    for line in lines:
        fields = line.split()
        if len(fields) == 0:
          # for consistency with older zmat inputs, they might end on an empty line
          # or have an empty line followed by some internal coordinate values
          break
        names.append(fields[0])
        nums_atom = nums_atom + 1
        # There are different line length possible
        # the matrix values are -1, because python starts with 0
        # but it is more convinient to count from atom 1
        if len(fields) == 1: # atom 1
            matrix.append(())
            vname_line = []
        elif len(fields) == 3: # atom 2
            matrix.append((int(fields[1])-1,))
            vname_line = [fields[2]]
        elif len(fields) == 5: # atom 3
            matrix.append((int(fields[1])-1, int(fields[3])-1,))
            vname_line = [fields[2], fields[4]]
        elif len(fields) == 7: # all other atoms
            matrix.append((int(fields[1])-1, int(fields[3])-1, int(fields[5])-1))
            vname_line = [fields[2], fields[4], fields[6]]
        else:
            print "ERROR: in reading the Z-Matrix"
            print "ERROR: line not understood:", line
            abort()

        # now check if there are some variables with multiplicity or if
        # some are dihedrals:
        for i, vname in enumerate(vname_line):
            num = 1
            if vname.startswith("-"):
               # allow setting of a variable to the minus of the
               # values by earlier appearance
               vname = vname[1:]
               num = -1

            if vname in var_names.keys():
                # we have this variable already
                multiplicity = multiplicity + 1
            else:
                var_count = var_count + 1
                var_names[vname] = var_count
                if i == 2: # i == 0 distance, i == 1 angle
                     dihedral_nums.append(var_count)

            # num takes care about inverse
            # collect all variables but with numbers, not with the names
            var_numbers.append(num * (var_names[vname] + 1))

    return names, matrix, var_numbers, multiplicity, dihedral_nums, (nums_atom*3, var_count + 1)


def set_atoms(at2, dc):
    """
    Sets the parameter from dc in atoms object at
    """
    at = deepcopy(at2)
    if "calculator" in dc:
          try:
              calculator = get_calculator(dc["calculator"])
          except TypeError:
              calculator = dc["calculator"]
          at.set_calculator(calculator)

    if "cell" in dc:
          at.set_cell(dc["cell"])
    if "pbc" in dc:
          at.set_pbc(dc["pbc"])

    return at

if __name__ == "__main__":
    import doctest
    doctest.testmod()

