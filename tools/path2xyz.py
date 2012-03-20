
#!/usr/bin/env python
"""
This  tools takes  a  path file  reads it  in  and gives  a string  of
jmol-type  xyz-files  to  the  standartoutput  path  file  can  be  in
path.pickle format, else it can be given via internal coordinates file
with some extra handling see below

They are equally spaced in the path coordinate.

As a default it will give as  many pictures as there were beads in the
path.pickle file, but any number N can be given

Run:

    paratools path2xyz some.path.pickle --num N

or simply:

    paratools path2xyz some.path.pickle

One could also  demand just to give back the beads  in this format (as
the beads  my differ  from the equally  in the path  coordinate spaced
points). This could be done by:

    paratools path2xyz some.path.pickle -b

Furthermore it  is possible  to provide the  input in  separate files,
then instead of the some.path.pickle  file a coordinate file should be
provided,  including  the geometry  for  all  the  beads. The  minimal
requirements are this coordinate file  and a file containing the atoms
symbols.

    paratools path2xyz coordinate.file --symbols symbolfile

It might be possible that  the coordinates are in internal coordinates
and at least  when the path is  wanted it is required to  build up the
internal  coordinate  transformation   rather  than  transforming  the
coordinates.  Further options for  building up the transformation into
Cartesian  coordinates (needed  if  coordinate.file contains  internal
coordinates and not only flattend Caresians)

    --zmat zmatrixfile

        adds zmatrix of zmatrixfile to transformation

    --mask maskfile raw_geom

        only  variables wich  are "True"  in mask  are supposed  to be
        contained  in coordinate.file, raw_geom  has to  give complete
        set (fixed coordinates will be gotten from there)

Further options:

    --abscissa abscissafile

        abscissa data  can be  gotten from here.  String calcualations
        provide this data.
"""

from pts.path import Path
from sys import exit
from sys import argv as sargv
from pts.tools.pathtools import unpickle_path, read_path_fix, read_path_coords


def read_in_path(filename):
    """
    Reads in a  path.pickle object and gives back  the informations of
    interest, e.g. the coordinates and path positions of the beads
    """
    try:
        coord, energy, gradients, tangents, pathps, symbols, trafo = unpickle_path(filename) # v2
    except:
        print "ERROR: No path file found to read input from"
        print "First argument of call must be a path.pickle object"
        print "The use of this function:"
        print __doc__
        exit()

    return pathps, coord, (symbols, trafo), (energy, gradients)

def read_in_path_raw(coordfile, symbfile, zmatifiles = None, pathpsfile = None, \
        maskfile = None, maskedgeo = None ):
    """
    Reads in a path from several user readable files and gives
    back the informations of interest
    """
    symbols, trafo = read_path_fix( symbfile, zmatifiles, maskfile, maskedgeo )
    coord, pathps, __, __ = read_path_coords(coordfile, pathpsfile, None, None)

    return pathps, coord, (symbols, trafo)

def path_geos(x, y, cs, num):
    """
    generates  num geometries  along  path  defined by  x  and y,  The
    geometries are given back as  string there is also a second string
    given back with the corresponding x values
    """
    path1 = Path(y, x)

    __, trafo = cs
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
         path.append(trafo(coord))
         xvals.append((endx / (num -1) * i))

    return path, xvals

def print_xyz(x, y, cs, num):
    """
    prints num xyz -frames (jmol format) which are equally distributed
    on the x values of the path.   If num is not set, there will be as
    many frames as there have been geometries in y (but need not be at
    the same geometries)
    """
    from pts.io.write_COS import print_xyz_with_direction
    from sys import stdout

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

    for i in range(num):
         # this is one of the frames,
         # the internal coordinates are converted
         # to Cartesian by the cs fake-Atoms object
         text = "This is the %i'th frame" % (i+1)
         coord = path1((endx / (num -1) * i))
         print_xyz_with_direction(stdout.write, coord, cs, text = text)

def print_beads(ys, cs):
    """
    Prints the xyz-  geoemtry in jmol format of  the beads.  This does
    exactly  the same  as above,  but without  the calculation  of the
    path,  this ensures  that  not  only exactly  the  number of  bead
    positions is taken but also  that it's exactly the beads which are
    used to create the frames
    """
    from pts.io.write_COS import print_xyz_with_direction
    from sys import stdout

    for i,y in enumerate(ys):
         text = "This is the %i'th bead" % (i+1)
         print_xyz_with_direction(stdout.write, y, cs, text = text)


def main(argv):
    """
    Reads in stuff from the sys.argv  if not provided an other way set
    up a path and gives back positions on it
    """
    from sys import stderr
    from pts.io.cmdline import get_options_to_xyz

    opts, filename = get_options_to_xyz("path" ,argv, None)

    beads = opts.beads
    num = opts.num

    symbfile = opts.symbfile
    zmats = opts.zmats

    if opts.mask == None:
        mask = None
        maskgeo = None
    else:
        mask, maskgeo = opts.mask

    abcis = opts.abcis

    if len(filename) == 1:
       filename = filename[0]
    else:
       print >> stderr, "ERROR: this tool can only process one geometry at a time"
       print >> stderr, "       There were found the files", filename
       exit()

    assert not opts.add_modes

    if symbfile == None:
        x, y, obj, __ = read_in_path(filename)
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

