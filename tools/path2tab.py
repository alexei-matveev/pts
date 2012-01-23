#!/usr/bin/env python
"""
This tools takes  pathes given by files (path.pickle  or others) reads
it in and prints for some data the results.

As input the path file(s) have to be given and the wanted set of
coordinates and other values

An internal coodinate is selected  by settings --kind n1 n2 ...  where
the ni's  are the atomnumbers (starting  with 1) which  should be used
for getting the internal coordinate, how many of them are requiered is
dependent on the kind choosen. There are the possiblilities:

"internal coordinate"  "kind"  "number of Atoms needed"
      distance           dis      2
        angle            ang      3
 angle not connected     ang4     4
    dihedral angle       dih      4
 distance to  plane      dp       4 *)

 *) the  first is  the atom,  the others define  the plane;  the plane
  atoms must not be on a line

It is  also possible to  ask for the  abscissas of the path  (would be
given as first element). This is done by --t.

Another set  of coordinates refers  to energy and gradients  stored in
path.pickle files.  For the gradients are several different options of
interest possible.  If pathes  are used (see  --num below)  the values
will be interpolated from the beads,  else the ones from the beads are
taken. IF required  for gradient calculations the tangent  to the path
is extracted from an interpolation path of the geometries.

The energy/gradient informations are always given after the geometry informations:
   --energy \ --en              : energies
   --gradients \ --gr \ --grabs : gives the absolute value of the gradients at the required positions
   --grmax                      : gives maximal value of (internal) gradients
   --grpara                     : length of gradient component parallel to the path
   --gperp                      : length of gradient component perpendicular to the path
   --grangle                    : angle (in degree) between path and gradients, should be 90 for convergence

easiest  input is  by path.pickle  files which  can be  given directly
without need of any option.

some options handle a different way of input:
Here coordinates are given in cordinate files (coordinates in internal coordinates for all
beads). One has to set addtionally at least the symbols.
    --symbols symfile            : file should contain all the symbols of the atoms
    --zmat    zmatfile           : to switch from Cartesian to Zmatrix coordinates, should be
                                   given the same way as for path tools
    --mask   maskfile geo_raw    : mask has to be given separately in maskfile one complete
                                   internal geometry has to be given as geo_raw to provide the
                                   tool with the fixed values
    --abscissa  abscissafile     : the abscissa for the coordinate file. There need not to be any
                                   but if there are some there need to be exactly one file for
                                   each coordinate file

If not other said the coordinates for the beads are returned.
    --num <num>                  : for any <num> > 0 will not use the bead geometries but interpolate
                                   a path instead and places <num> equally spaced points on it
                                   it is illegal to use this parameter together with any of the log
                                   parameters

der are other options which may be set:
    --diff                       :for the next two internal coordinates the difference
                                  will be taken, instead of the values, works only for
                                  geometry coordinates
    --expand cellfile expandfile : the atoms will be exanded with atoms choosen as
                                   original ones shifted with cell vectors, as described
                                   in expandfile.
                                  cellfile should contain the three basis vectors for the cell
                                  expandfile contains the shifted atoms with:
                                  "number of origin" "shift in i'th direction"*3
    --log  filename string num   : reads in filename which should be a .log file output file of
                                   a string or neb calculation and takes string line of the num'th
                                   iteration as some extra bead data, it is illegal to use this with
                                   an interpolating string

    --lognf filename             : new log as before, but reuses the string and num from the last one
                                   only the logfile is changed
    --lognn num                  : new log plot as above, but takes another iteration than the last one
    --symm <midx>                : uses the next coordinate defined (or the next difference defined) and
                                   returns the derivative to symmetry around midx instead of the value
                                   if midx is not given, 0 is used, symmetry is calculated by:
                                   0.5 * (f(midx + x) - f(midx - x))
                                   only valid for geometry coordinates
"""

from pts.path import Path
from sys import exit
from sys import stderr
from sys import argv as sargv
from pickle import load
from pts.tools.path2xyz import read_in_path
from pts.tools.pathtools import read_path_fix, read_path_coords
from pts.tools.xyz2tabint import returnall, interestingvalue, expandlist, writeall
from pts.cfunc import Pass_through
from pts.io.read_COS import read_geos_from_file_more
import numpy as np
from sys import stdout


def path_to_int(x, y, cs, num, allval, cell, tomove, howmove, withs):
    """
    Gives back the internal values for allval
    which appear on num equally spaced (in x-direction) on
    the path
    """
    path1 = Path(y, x)
    path = []

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

    for i in range(num):
         # this is one of the frames,
         # the internal coordinates are converted
         # to Cartesian by the cs fake-Atoms object
         coord = path1((endx / (num -1) * i))
         cart =  int2cart(coord)
         if cell != None:
             cart2 = list(cart)
             expandlist(cart2, cell, tomove, howmove)
             cart = np.array(cart2)

         new_val = returnall(allval, cart, True, i)
         if withs:
             path.append( [new_val[0]] + [endx / (num - 1) * i] + new_val[1:])
         else:
             path.append(new_val)

    return path

def energy_from_path(x, E, num):
    """
    Gives back the values of  the Energies which appear on num equally
    spaced (in x-direction) on the path
    """
    path1 = Path(E, x)
    energies = []

    # to decide how  long x is, namely what coordinate  does the end x
    # have if there is no x at all, the path has distributed the beads
    # equally from 0 to 1 thus in this case the end of x is 1
    if x is None:
        endx = 1.0
    else:
        endx = float(x[-1])

    for i in range(num):
         # this is one  of the frames, with its  gradients Make it the
         # saem way than for the coordinates
         x_1 = (endx / (num -1) * i)
         energies.append(path1(x_1))

    return energies

def grads_from_path(x, y, gr, num, allval ):
    """
    Gives back the values of the gradients as decided in allval
    which appear on num equally spaced (in x-direction) on
    the path
    """
    path1 = Path(gr, x)
    path2 = Path(y, x)
    grs = []

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
         # this is one of the frames, with its gradients
         # Make it the saem way than for the coordinates
         x_1 = (endx / (num -1) * i)
         gr_1 = path1(x_1).flatten()

         if allval == "abs":
            #Total forces, absolute value
            grs.append(np.sqrt(np.dot(gr_1, gr_1)))
         elif allval == "max":
            #Total forces, maximal value
            grs.append(max(abs(gr_1)))
         elif allval == "para":
            # parallel part of forces along path
            mode = path2.fprime(x_1 )
            mode = mode / np.sqrt( np.dot(mode, mode))
            gr_2 = np.dot(mode, gr_1)
            grs.append(gr_2)
         elif allval == "perp":
            # absolute value of perp. part of forces along path
            mode = path2.fprime(x_1 )
            mode = mode / np.sqrt( np.dot(mode, mode))
            gr_2 = np.dot(mode, gr_1)
            gr_1 = gr_1 - gr_2 * mode
            grs.append(np.sqrt(np.dot(gr_1, gr_1)))
         elif allval == "angle":
            # angle between forces and path
            mode = path2.fprime(x_1 )
            mode = mode / np.sqrt( np.dot(mode, mode))
            gr_1 = gr_1 / np.sqrt( np.dot(gr_1, gr_1))
            ang = abs(np.dot(mode, gr_1))
            if ang > 1.:
                ang = 1.
            grs.append(np.arccos(ang) * 180. / np.pi)
         else:
            print >> stderr, "Illegal operation for gradients", allval
            exit()

    return grs

def grads_from_beads(x, y, gr, allval ):
    """
    Gives back the values of the gradients as decided in allval
    which appear on the beads, path informations are needed
    for finding the mode along the path
    """
    path2 = Path(y, x)
    grs = []

    # to decide how long x is, namely what
    # coordinate does the end x have
    # if there is no x at all, the path has
    # distributed the beads equally from 0 to 1
    # thus in this case the end of x is 1
    if x is None:
        endx = 1.0
        num = len(gr)
        x_l = [ (endx / (num -1) * i) for i in range(num)]
    else:
        endx = float(x[-1])
        x_l = x

    for gr_1, x_1 in zip(gr, x_l):

         if allval == "abs":
            #Total forces, absolute value
            grs.append(np.sqrt(np.dot(gr_1, gr_1)))
         elif allval == "max":
            #Total forces, maximal value
            grs.append(max(abs(gr_1)))
         elif allval == "para":
            # parallel part of forces along path
            mode = path2.fprime(x_1)
            mode = mode / np.sqrt( np.dot(mode, mode))
            gr_2 = np.dot(mode, gr_1)
            grs.append(gr_2)
         elif allval == "perp":
            # absolute value of perp. part of forces along path
            mode = path2.fprime(x_1)
            mode = mode / np.sqrt( np.dot(mode, mode))
            gr_2 = np.dot(mode, gr_1)
            gr_1 = gr_1 - gr_2 * mode
            grs.append(np.sqrt(np.dot(gr_1, gr_1)))
         elif allval == "angle":
            # angle between forces and path
            mode = path2.fprime(x_1)
            mode = mode / np.sqrt( np.dot(mode, mode))
            gr_1 = gr_1 / np.sqrt( np.dot(gr_1, gr_1))
            ang = abs(np.dot(mode, gr_1))
            if ang > 1.:
                ang = 1.
            grs.append(np.arccos(ang) * 180. / np.pi)
         else:
            print >> stderr, "Illegal operation for gradients", allval
            exit()
    return grs

def beads_to_int(ys, xs, cs, allval, cell, tomove, howmove, withs):
    """
    This does exactly the same as above, but
    without the calculation of the path, this
    ensures that not only exactly the number of
    bead positions is taken but also that it's
    exactly the beads which are used to create
    the frames
    """
    beads = []
    syms, int2cart = cs

    for i,y in enumerate(ys):
         cart = int2cart(y)
         if cell != None:
             cart2 = list(cart)
             expandlist(cart2, cell, tomove, howmove)
             cart = np.array(cart2)

         new_val = returnall(allval, cart, True, i)
         if withs:
             beads.append([new_val[0]] + [xs[i]] + new_val[1:])
         else:
             beads.append(new_val)

    return beads

def main(argv):
    """
    Reads in stuff from the sys.argv if not
    provided an other way

    Then calculate according to the need
    the result will be a picture showing
    for each inputfile a path of the given
    coordinates with beads marked on them
    """
    if argv[0] == '--help':
        print __doc__
        exit()

    # store the files containing the pathes somewhere
    filenames = []
    # input need not to be path.pickle
    symbfile = None
    zmats = []
    mask = None
    maskgeo = None
    abcis = []
    # ase inputs:
    ase = False
    next = [0]
    format = None

    # the default values for the parameter
    num = -1
    diff = []
    symm = []
    symshift = []
    logscale = []
    allval = []
    special_vals = []
    cell = None
    tomove = None
    howmove = None
    withs = False

    # count up to know which coordinates are special
    num_i = 1

    # possibility of taking values from a logfile
    logs = []
    logs_find = []
    logs_num = []
    log_x_num = []
    xfiles = -1

    # read all the arguments in
    for i in range(len(argv)):
         if argv == []:
             # stop cycle if all input is read in
             # (sometimes more than one is read in
             # at the same time)
             break
         elif argv[0].startswith("--"):
             # differenciate between options and files
             option = argv[0][2:]
             if option == "num":
                 # change number of frames in path from 100
                 # to next argument
                 num = int(argv[1])
                 argv = argv[2:]
             elif option == "diff":
                 # of the next twoo coordinates the difference
                 # should be taken, store the number of the first
                 # of them
                 diff.append(num_i)
                 argv = argv[1:]
             elif option == "symm":
                 # test if the following coordinate (or coord. diffs)
                 # follow the same symmetry as the x-coordinate
                 symm.append(num_i)
                 try:
                     m = float(argv[1])
                     symshift.append([num_i, m])
                     argv = argv[1:]
                 except:
                     pass
                 argv = argv[1:]
             elif option in ["dis", "2", "ang","3", "ang4", "4", "dih", "5", "dp", "6", "dl", "7"]:
                 # this are the possible coordinates, store them
                 value = interestingvalue(option)
                 # partners are the atomnumbers of the partners, which
                 # create the coordinate
                 value.partners = []
                 for j in range(1, value.lengthneeded() + 1):
                      value.partners.append(int(argv[j]))
                 allval.append(value)
                 argv = argv[value.lengthneeded() + 1:]
                 # count up, to know how many and more important for
                 # let diff easily know what is the next
                 num_i += 1
             elif option in ["s", "t"]:
                 withs = True
                 argv = argv[1:]
             elif option in ["en", "energy", "grabs", "grmax" \
                                  ,"grpara", "grperp", "grangle" ]:
                 special_vals.append(option)
                 argv = argv[1:]
             elif option in ["gr", "gradients"]:
                 special_vals.append("grabs")
                 argv = argv[1:]
             elif option in ["fromlog", "log"]:
                 logs.append(argv[1])
                 logs_find.append(argv[2])
                 logs_num.append(int(argv[3]))
                 argv = argv[4:]
                 log_x_num.append(xfiles)
             elif option in ["fromlognextfile", "lognf"]:
                 logs.append(argv[1])
                 logs_find.append(logs_find[-1])
                 logs_num.append(logs_num[-1])
                 argv = argv[2:]
                 log_x_num.append(xfiles)
             elif option in ["fromlognextnum", "lognn"]:
                 logs.append(logs[-1])
                 logs_find.append(logs_find[-1])
                 logs_num.append(int(argv[1]))
                 argv = argv[2:]
                 log_x_num.append(xfiles)
             elif option == "expand":
                 # done like in xyz2tabint, expand the cell
                 #FIXME: There should be a better way to consider atoms
                 # to be shifted to other cells
                 cell, tomove, howmove = get_expansion(argv[1], argv[2])
                 argv = argv[3:]
             elif option == "ase":
                 ase = True
                 argv = argv[1:]
             elif option == "format":
                 format = argv[1]
                 argv = argv[2:]
             elif option == "next":
                 ase = True
                 next.append(xfiles)
                 argv = argv[1:]
             elif option == "symbols":
                 symbfile = argv[1]
                 argv = argv[2:]
             elif option == "zmat":
                 zmats.append(argv[1])
                 argv = argv[2:]
             elif option == "mask":
                 mask = argv[1]
                 maskgeo = argv[2]
                 argv = argv[3:]
             elif option in ["abscissa", "pathpos"]:
                abcis.append(argv[1])
                argv = argv[2:]
             else:
                 # For everything that does not fit in
                 print "This input variable is not valid:"
                 print argv[0]
                 print "Please check your input"
                 print __doc__
                 exit()
         else:
             # if it is no option is has to be a file
             filenames.append(argv[0])
             argv = argv[1:]
             xfiles += 1

    if ase:
        next.append(len(filenames)+1)
        filenames = reorder_files(filenames, next)

    if symbfile is not None:
        symb, i2c = read_path_fix( symbfile, zmats, mask, maskgeo )
        obj = symb, i2c

    # For each file prepare the plot
    for i, filename in enumerate(filenames):
        # read in the path
        e_a_gr = None
        if ase:
            atoms, y = read_geos_from_file_more(filename, format=format)
            obj =  atoms.get_chemical_symbols(), Pass_through()

        elif symbfile is None:
            x, y, obj, e_a_gr = read_in_path(filename)
        else:
            patf = None
            if len(abcis) > 0:
                patf = abcis[i]
            y, x, __, __ = read_path_coords(filename, patf, None, None)

        # extract the internal coordinates, for path and beads
        if num < 0:
            coords = beads_to_int(y, x, obj, allval, cell, tomove, howmove, withs)
        else:
            assert(logs == [])
            coords = path_to_int(x, y, obj, num, allval, cell, tomove, howmove, withs)
        coords = np.asarray(coords)
        coords = coords.T
        coords = list(coords)

        if special_vals != []:
            assert (e_a_gr is not None)
            for s_val in special_vals:
                 en, gr = e_a_gr
                 if s_val.startswith("en"):
                    if num < 0:
                        coords.append(en)
                    else:
                        coords.append(energy_from_path(x, en, num ))
                 elif s_val.startswith("gr"):
                    val = s_val[2:]
                    if num < 0:
                        coords.append(grads_from_beads(x, y, gr, val))
                    else:
                        coords.append(grads_from_path(x, y, gr, num, val))

        if logs != []:
            for j, log in enumerate(logs):
                if log_x_num[j] == i:
                 # use the options for x and plot the data gotten from the file directly
                 coords.append(read_line_from_log(log, logs_find[j], logs_num[j]))
                 # The name should be the name of the data line taken, right?

        # they are wanted as arrays and the other way round
        coords = np.asarray(coords)
        coords = coords.T
        coords = list(coords)
        write_results(coords, allval, special_vals, logs, filename)


def write_results(coords, allval, special_vals, logs, filename):
     """
     Writes the results out as a table
     """
     write = stdout.write
     write("#chart of internal coordinates in the run \n")
     write("#observed in file: %s  " % (filename) )
     write("#the following values were calculated, distances are in Angstroms; angles in degrees;\n")
     write("#                                      energies in eV and forces in eV/ Angstroms\n")
     for k in range(len(allval)):
     # tell what values will be calulated
         write("%s with atoms :" % (allval[k].whatsort()))
         for number in allval[k].partners:
             write("%i " % number )
         write(";")
     for sv in special_vals:
         if sv.startswith("en"):
             write(" energy;")
         elif sv.startswith("gr"):
                 write(" %s of gradients ;" % (sv[2:]))
     for lg in logs:
         write(lg + "; ")
     write("\n")

     for cor in coords:
             writeall(write, cor[0:], cor[0])


def reorder_files(files, next):
    new_order = []
    line = []
    n = next[1]
    k = 2
    for i, file in enumerate(files):
        if i > n:
           n = next[k]
           k = k+1
           new_order.append(line)
           line = []
        line.append(file)
    new_order.append(line)
    return new_order

def get_expansion(celldat, expand):
     """
     Expand the atoms by some, which are shifted
     cellvectors in each direction
     Needs the cell put into a file
     and a list of atoms to expand, also put into a file
     """
     filecell = open(celldat, "r" )
     cell = np.zeros((3,3))
     # fill cell with data from celldat
     for  num, line  in enumerate(filecell):
            fields = line.split()
            cell[num,0] = float(fields[0])
            cell[num,1] = float(fields[1])
            cell[num,2] = float(fields[2])

     # get the expanded atoms
     filemove = open(expand,"r" )
     tomove = []
     howmove = []
     # tomove[i] = number of source atom for atom (i + all_in_original_cell)
     # howmove[i] = how to get the new atom
     for  num, line  in enumerate(filemove):
            fields = line.split()
            tomove.append( int(fields[0]))
            howmove.append([float(fields[1]), float(fields[2]), float(fields[3])])

     return cell, tomove, howmove

def read_line_from_log(filename, whichline, num):
    """
    Reads in a log file (output format of .log file in string/neb
    calculations) and extracts a dataline of the coordinates whichline
    in the num's iteration
    """
    file = open(filename, "r")
    uninteresting = len(whichline.split())
    rightnum = False
    for line in file:
        if line.startswith('Chain of States Summary'):
            fields = line.split()
            if str(num) in fields:
                rightnum = True
            else:
                rightnum = False
        if rightnum:
              if line.startswith(whichline):
                  fields = line.split()
                  dataline = []
                  datapoints = (len(fields) - uninteresting) / 2
                  for i in range(datapoints):
                      dataline.append(float(fields[uninteresting +1 +2 * i]))
    return dataline


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main(sargv[1:])

