#!/usr/bin/env python
"""
Compares two or more geometries from geometry files in Cartesian or
Zmatrix coordinates.

Usage:
 paratools compare_geos <reference geometry file> <geometry file 1> [<geometry file 2> ...]

To have the comparision in Zmatrix coordinates use:
 paratools compare_geos --zmatrix <Zmatrix file>  <reference geometry file> \
    <geometry file 1> [<geometry file 2> ...]

It is also possible to specify the format of the input geometries if they cannot be
recognised easily by ASE. This is done by adding the parameter
  --format <input format>
<input format> can be any of the ones available for ASE. This option has to be given
before the geometry files but in any order with --zmatrix.

 paratools compare_geos --help gives this helptext.
"""
from ase.io import read
from pts.metric import Default
from pts.cfunc import Cartesian
from pts.zmat import ZMat
from pts.ui.read_COS import read_zmt_from_file

def compare(g1, g2, syms, fun):
    """
    compares how near the two geometries are
    """
    met = Default(fun)

    diff = fun.pinv(g1) - fun.pinv(g2)
    diff2 = list(abs(diff))
    max_raw = max(diff2)
    arg_max_r = diff2.index(max(diff2))
    diff2 = met.norm_up(diff, g1)
    rms = diff2 / len(diff)
    print "Differences were"
    print max_raw, rms, diff2
    print "Maximal argument was on variable", arg_max_r + 1

def main(argv):
    format = None
    zmat = None

    while argv[0].startswith("--"):
        if argv[0] == '--format':
            format = argv[1]
            argv = argv[2:]
        elif argv[0] == '--zmatrix':
            __, zmat, v_name, __, __, __,__ = read_zmt_from_file(argv[1])
            argv = argv[2:]
        elif argv[0] == '--help':
            print __doc__
            return



    geo1 = read(argv[0], format = format)
    geos = [read(arg, format = format) for arg in argv[1:]]

    if zmat == None:
         fun = Cartesian()
    else:
         fun = ZMat(zmat)

    symbols = geo1.get_chemical_symbols()
    assert len(geos) > 0

    for geo in geos:
        assert geo.get_chemical_symbols() == symbols
        compare(geo1.get_positions(), geo.get_positions(), symbols, fun)

if __name__ == "__main__":
    main(sys.argv[1:])

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
