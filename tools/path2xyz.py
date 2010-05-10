#!/usr/bin/python
"""
This tools takes a path.pickle file
reads it in and gives a string of
jmol -type xyz-files to the standartoutput

They are equally spaced in the path coordinate

As a default it will give as many pictures as there
were beads in the path.pickle file, but any number N can be given

Run:

    $path2xyz.py some.path.pickle N

or simply:

    $path2xyz.py some.path.pickle

One could also demand just to give back the beads in this format
(as the beads my differ from the equally in the path coordinate spaced
points)
This could be done by:

    $path2xyz.py some.path.pickle -b

"""

from aof.path import Path
from sys import exit
from sys import argv as sargv
from pickle import load


def read_in_path(filename):
    """
    Reads in a path.pickle object and gives
    back the informations of interest, e.g. the
    coordinates and path positions of the beads
    """
    try:
        f_ts = open(filename,"r")
    except:
        print "ERROR: No path file found to read input from"
        print "First argument of call must be a path.pickle object"
        print "The use of this function:"
        print __doc__
        exit()

    # some older files and neb files don't have path positions
    # the value None should give a default path
    posonstring = None

    # there are different versions of the path.pickle object around
    # this way all of them should be valid
    try:
        coord, energy, gradients, posonstring, posonstring2, at_object =  load(f_ts)
        f_ts.close()
    except :
        try:
            f_ts.close()
            f_ts = open(filename,"r")
            coord, energy, gradients, posonstring, at_object =  load(f_ts)
            f_ts.close()
        except ValueError:
            f_ts.close()
            f_ts = open(filename,"r")
            coord, energy, gradients, at_object = load(f_ts)
            f_ts.close()

    return posonstring, coord, at_object

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

    numats = cs.atoms_count

    for i in range(num):
         # this is one of the frames,
         # the internal coordinates are converted
         # to Cartesian by the cs fake-Atoms object
         print numats
         print "This is the %i'th frame" % (i+1)
         coord = path1((endx / (num -1) * i))
         cs.set_internals(coord)
         print cs.xyz_str()

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
    numats = cs.atoms_count
    for i,y in enumerate(ys):
         print numats
         print "This is the %i'th bead" % (i+1)
         cs.set_internals(y)
         print cs.xyz_str()


def main(argv=None):
    """
    Reads in stuff from the sys.argv if not
    provided an other way
    set up a path and gives back positions
    on it
    """
    if argv is None:
        argv = sargv[1:]

    if argv[0] == '--help':
        print __doc__
        exit()
    else:
        filename = argv[0]

    beads = False
    num = None

    # There is one more decicion to make:
    # how many points in between or exactly the beads?
    # as default there will be a path , giving back beadnumber
    # of frames
    if len(argv)>1:
       if argv[1] in ["beads", "bd", "b", "-b"]:
            beads = True
       else:
            try:
                 num = int(argv[1])
            except:
                 print "Could not read in wanted number of pictures"
                 print "The second argument of call should be a integer number"
                 print __doc__
                 exit()

    x, y, obj = read_in_path(filename)

    if beads:
        print_beads(y, obj)
    else:
        print_xyz(x, y, obj, num)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()

