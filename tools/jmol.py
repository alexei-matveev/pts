#!/usr/bin/python
"""
Run

    $ jmol.py POSCAR1 POSCAR2 ... POSCARn

or

    $ jmol.py 1.xyz 1.xyz ... n.xyz

to vizualize a path in 3D with jmol.

Alternatively

    >>> from aof.tools.jmol import jmol_view_path
    >>> jmol_view_path(arrayNx3, atomic_symbolsi, refine=5, viewer=lambda x: print "starting jmol ...")

This was a doctest. Normally you will use the default
viewer (jmol) like here:

#   >>> jmol_view_path(arrayNx3, atomic_symbolsi, refine=5)
"""
from __future__ import with_statement
import sys
import os
from tempfile import mkstemp

import ase
from aof.path import Path
from numpy import linspace

def jmol_view_file(file):

    os.system("jmol " + file)

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
    images = [ ase.read(file) for file in sys.argv[1:] ]

    geoms = [ im.get_positions() for im in images ]

    syms = images[0].get_chemical_symbols()

    jmol_view_path(geoms, syms)

if __name__ == "__main__":
    main()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
