#!/usr/bin/env python
"""
This tools takes a path file
reads it in and gives a string of
jmol-type xyz-files to the standartoutput
path file can be in path.pickle format, else it can be
given via internal coordinates file with some extra handling
see below

They are equally spaced in the path coordinate

As a default it will give as many pictures as there
were beads in the path.pickle file, but any number N can be given

Run:

    $paratools path2xyz some.path.pickle --num N

or simply:

    $paratools path2xyz some.path.pickle

One could also demand just to give back the beads in this format
(as the beads my differ from the equally in the path coordinate spaced
points)
This could be done by:

    $paratools path2xyz some.path.pickle -b

Furthermore it is possible to provide the input in separate files, then instead
of the some.path.pickle file a coordinate file should be provided, including the
geometry for all the beads. The minimal requirements are this coordinate file and
a file containing the atoms symbols.

    $paratools path2xyz coordinate.file --symbols symbolfile

It might be possible that the coordinates are in internal coordinates and at least
when the path is wanted it is required to build up the internal coordinate transformation
rather than transforming the coordinates.
Further options for building up the transformation into Cartesian coordinates
    (needed if coordinate.file contains internal coordinates and not only flattend Caresians)
    --zmat zmatrixfile  : adds zmatrix of zmatrixfile to transformation
    --mask maskfile raw_geom : only variables wich are "True" in mask are supposed to be
                          contained in coordinate.file, raw_geom has to give complete set
                          (fixed coordinates will be gotten from there)

Further options:
    --abscissa abscissafile : abscissa data can be gotten from here. String calcualations provide
                            this data.
"""

from pts.path import Path
from sys import exit
from sys import argv as sargv
from pts.tools.pathtools import unpickle_path, read_path_fix, read_path_coords


def read_in_path(filename):
    """
    Reads in a path.pickle object and gives
    back the informations of interest, e.g. the
    coordinates and path positions of the beads
    """
    try:
        coord, pathps, energy, gradients, symbols, int2cart = unpickle_path(filename)
    except:
        print "ERROR: No path file found to read input from"
        print "First argument of call must be a path.pickle object"
        print "The use of this function:"
        print __doc__
        exit()

    return pathps, coord, (symbols, int2cart)

def read_in_path_raw(coordfile, symbfile, zmatifiles = None, pathpsfile = None, \
        maskfile = None, maskedgeo = None ):
    """
    Reads in a path from several user readable files and gives
    back the informations of interest
    """
    symbols, int2cart = read_path_fix( symbfile, zmatifiles, maskfile, maskedgeo )
    coord, pathps, __, __ = read_path_coords(coordfile, pathpsfile, None, None)

    return pathps, coord, (symbols, int2cart)

def path_geos(x, y, cs, num):
    """
    generates  num geometries along path
    defined by x and y,
    The geometries are given back as string
    there is also a second string given back with
    the corresponding x values
    """
    path1 = Path(y, x)

    __, int2cart = cs
    # to decide how long x is, namely what
    # coordinate does the end x have
    # if there is no x at all, the path has
    # distributed the beads equally from 0 to 1
    # thus in this case the end of x is 1
    if x is None:
        endx = 1.0
    else:
        endx = float(x[-1])

    path = []
    xvals = []
    for i in range(num):
         # this is one of the frames,
         # the internal coordinates are converted
         # to Cartesian by the cs fake-Atoms object
         coord = path1((endx / (num -1) * i))
         path.append(int2cart(coord))
         xvals.append((endx / (num -1) * i))

    return path, xvals

def print_xyz(x, y, cs, num):
    """
    prints num xyz -frames (jmol format)
    which are equally distributed on the
    x values of the path
    If num is not set, there will be
    as many frames as there have been
    geometries in y (but need not be at
    the same geometries)
    """
    if num is None:
       num = len(y)

    path1 = Path(y, x)

    # to decide how long x is, namely what
    # coordinate does the end x have
    # if there is no x at all, the path has
    # distributed the beads equally from 0 to 1
    # thus in this case the end of x is 1
    if x is None:
        endx = 1.0
    else:
        endx = float(x[-1])

    symbs, int2cart = cs
    numats = len(symbs)

    for i in range(num):
         # this is one of the frames,
         # the internal coordinates are converted
         # to Cartesian by the cs fake-Atoms object
         print numats
         print "This is the %i'th frame" % (i+1)
         coord = path1((endx / (num -1) * i))
         carts = int2cart(coord)
         for sy, pos in zip(symbs, carts):
             print '%-2s %22.15f %22.15f %22.15f' % (sy, pos[0], pos[1], pos[2])

def print_beads(ys, cs):
    """
    Prints the xyz- geoemtry in jmol format
    of the beads
    This does exactly the same as above, but
    without the calculation of the path, this
    ensures that not only exactly the number of
    bead positions is taken but also that it's
    exactly the beads which are used to create
    the frames
    """
    symbs, int2cart = cs
    numats = len(symbs)
    for i,y in enumerate(ys):
         print numats
         print "This is the %i'th bead" % (i+1)
         carts = int2cart(y)
         for sy, pos in zip(symbs, carts):
             print '%-2s %22.15f %22.15f %22.15f' % (sy, pos[0], pos[1], pos[2])


def main(argv):
    """
    Reads in stuff from the sys.argv if not
    provided an other way
    set up a path and gives back positions
    on it
    """
    if argv[0] == '--help':
        print __doc__
        exit()
    else:
        filename = argv[0]

    beads = False
    num = None
    symbfile = None
    zmats = []
    mask = None
    maskgeo = None
    abcis = None

    # There is one more decicion to make:
    # how many points in between or exactly the beads?
    # as default there will be a path , giving back beadnumber
    # of frames
    if len(argv)>1:
       argv = argv[1:]
       for i in range(len(argv)):
           if argv == []:
              break

           if argv[0] in ["beads", "bd", "b", "-b"]:
                beads = True
                argv = argv[1:]
           elif argv[0] in ["--num"]:
                num = int(argv[1])
                argv = argv[2:]
           elif argv[0] in ["--symbols", "--symbol", "-s"]:
                symbfile = argv[1]
                argv = argv[2:]
           elif argv[0].startswith("--zmat"):
                zmats.append(argv[1])
                argv = argv[2:]
           elif argv[0] in ["--mask", "-m"]:
                mask = argv[1]
                maskgeo = argv[2]
                argv = argv[3:]
           elif argv[0] in ["--abscissa", "--pathpos"]:
                abcis = argv[1]
                argv = argv[2:]
           else:
                print "Could not read in the argument", argv[0]
                print __doc__
                exit()

    if symbfile == None:
        x, y, obj = read_in_path(filename)
    else:
        x, y, obj = read_in_path_raw(filename, symbfile, zmats, abcis, \
        mask, maskgeo )

    if beads:
        print_beads(y, obj)
    else:
        print_xyz(x, y, obj, num)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main(sargv[1:])

