#!/usr/bin/env python
"""
Run

    $ jmol.py POSCAR1 POSCAR2 ... POSCARn

or

    $ jmol.py 1.xyz 1.xyz ... n.xyz

to vizualize a path in 3D with jmol.

To refine this path set --refine num
Thus:

    $ jmol.py --refine 2  1.xyz 1.xyz ... n.xyz

will visualize the path in 3D with jmol with
an extra image (approximated by a path through
the origional images) will be shown

Alternatively

    >>> from pts.tools.jmol import jmol_view_path
    >>> geoms = [[(0., 0., 0.), (0., 0., 1.)],
    ...          [(0., 0., 0.), (1., 0., 0.)]]

Very simple ascii 3D-viewer:

    >>> def ascii(fname):
    ...     for line in open(fname):
    ...         print line.rstrip()

    >>> jmol_view_path(geoms, viewer=ascii)
    2
    frame 0.000000
       X    0.00000000   0.00000000   0.00000000
       X    0.00000000   0.00000000   1.00000000
    2
    frame 1.000000
       X    0.00000000   0.00000000   0.00000000
       X    1.00000000   0.00000000   0.00000000

This was a doctest. Normally you will use the default
viewer (jmol) like here:

#   >>> jmol_view_path(geoms, ["C", "O"], refine=5)
"""

from __future__ import with_statement
import sys
import os
from tempfile import mkstemp

import ase
from pts.path import Path
from numpy import linspace

def jmol_view_file(file):

    os.system("jmol " + file)

def jmol_view_atoms(atoms, viewer=jmol_view_file):
    x = atoms.get_positions()
    syms = atoms.get_chemical_symbols()

    # write the geom in jmol format to a temp file:
    fd, fname = mkstemp()

    with os.fdopen(fd, "w") as file:
        jmol_write(x, syms, file=file)

    # run xyz-viewer:
    viewer(fname)

    os.unlink(fname)

def jmol_view_path(geoms, syms=None, refine=1, viewer=jmol_view_file):

    path = Path(geoms)

    # write the path in jmol format to a temp file:
    fd, fname = mkstemp()

    with os.fdopen(fd, "w") as file:
        # number of interpolation points between images is given by "refine":
        N = refine * (len(geoms) - 1) + 1
        for t in linspace(0., 1., N):
            jmol_write(path(t), syms, comment=("frame %f" % t), file=file)

    # run xyz-viewer:
    viewer(fname)

    os.unlink(fname)

def jmol_write(xyz, syms=None, comment="", file=sys.stdout):
    write = file.write
    write(str(len(xyz)) + "\n")
    write(comment + "\n")
    if syms is None: syms = ["X"] * len(xyz)
    for s, v in zip(syms, xyz):
        line = "%4s  %12.8f %12.8f %12.8f\n" % (s, v[0], v[1], v[2])
        write(line)

def main():
    argv = sys.argv[1:]
    refinenum = 1
    if argv[0] == '--refine':
        refinenum = int(argv[1])
        argv = argv[2:]
    elif argv[0] == '--help':
        print __doc__
        return

    if len(argv) < 2:
        # print usage and return:
        print __doc__
        return

    images = [ ase.read(file) for file in argv[:] ]

    geoms = [ im.get_positions() for im in images ]

    syms = images[0].get_chemical_symbols()

    jmol_view_path(geoms, syms, refine=refinenum)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
