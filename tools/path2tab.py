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
from pts.tools.xyz2tabint import returnall, expandlist, writeall
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

    for i in range(num):
         # this is one of the frames,
         # the internal coordinates are converted
         # to Cartesian by the cs fake-Atoms object
         coord = path1((endx / (num -1) * i))
         cart =  trafo(coord)
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

def ts_estimates_in_int(x, y, cs, en_and_grad, estimates, allval, cell, tomove, howmove, withs, see_all):
    """
    Gives back the internal values for allval
    which appear on transition state estimates "estimates"
    on the path
    """
    from pts.tools.tsestandmods import esttsandmd
    en, gr = en_and_grad
    __, trafo = cs
    ts_all, __ = esttsandmd(y,  en, gr, cs, see_all, estimates)

    ts_estims = []
    ts_places = []
    for i, ts_one in enumerate(ts_all):
         __, est, __, __, __ = ts_one
         __, coords, __, __,s_ts,  __, __ = est
         cart = trafo(coords)
         if cell != None:
             cart2 = list(cart)
             expandlist(cart2, cell, tomove, howmove)
             cart = np.array(cart2)

         new_val = returnall(allval, cart, True, i)
         if withs:
             ts_estims.append([new_val[0]] + [s_ts] + new_val[1:])
         else:
             ts_estims.append(new_val)
         ts_places.append(s_ts)

    return ts_estims, ts_places


def energy_from_path(x, E, num):
    """
    Gives back the values of  the Energies which appear on num equally
    spaced (in x-direction) on the path
    """

    # to decide how  long x is, namely what coordinate  does the end x
    # have if there is no x at all, the path has distributed the beads
    # equally from 0 to 1 thus in this case the end of x is 1
    if x is None:
        endx = 1.0
    else:
        endx = float(x[-1])

    xs = []
    for i in range(num):
         # this is one  of the frames, with its  gradients Make it the
         # saem way than for the coordinates
         xs.append((endx / (num -1) * i))

    return energy_from_path_points(x, E, xs)

def energy_from_path_points(x, E, ss):
    """
    Gives back the values of  the Energies which appear at the
    positions ss on the interpolated path
    """
    path1 = Path(E, x)

    return map(path1, ss)

def energy2_from_path(x, y, E, gr, num):
    """
    Gives back the values of  the Energies which appear at the
    positions ss on the interpolated path
    """
    # to decide how  long x is, namely what coordinate  does the end x
    # have if there is no x at all, the path has distributed the beads
    # equally from 0 to 1 thus in this case the end of x is 1
    if x is None:
        endx = 1.0
    else:
        endx = float(x[-1])

    xs = []
    for i in range(num):
         # this is one  of the frames, with its  gradients Make it the
         # saem way than for the coordinates
         xs.append((endx / (num -1) * i))

    return energy2_from_path_points(x, y, E, gr, xs)


def energy2_from_path_points(x, y, E, gr, ss):
    """
    Gives back the values of  the Energies which appear at the
    positions ss on the interpolated path
    """
    from pts.func import CubicSpline
    from numpy import dot
    path2 = Path(y, x)

    tangents = [path2.fprime(x1) for x1 in x]
    slopes = [dot(g, t) for g, t in zip(gr, tangents)]
    spline = CubicSpline(x, E, slopes)

    return map(spline, ss)

def grads_from_path(x, y, gr, num, allval ):
    """
    Gives back the values of the gradients as decided in allval
    which appear on num equally spaced (in x-direction) on
    the path
    """
    # to decide how long x is, namely what
    # coordinate does the end x have
    # if there is no x at all, the path has
    # distributed the beads equally from 0 to 1
    # thus in this case the end of x is 1
    if x is None:
        endx = 1.0
    else:
        endx = float(x[-1])

    xs = []
    for i in range(num):
         # this is one of the frames, with its gradients
         # Make it the saem way than for the coordinates
         xs.append((endx / (num -1) * i))

    return grads_from_path_points(x, y, gr, xs, allval )

def grads_from_path_points(x, y, gr, ss, allval ):
    """
    Gives back the values of the gradients as decided in allval
    which appear on num equally spaced (in x-direction) on
    the path
    """
    path1 = Path(gr, x)
    path2 = Path(y, x)
    grs = []

    for x_1 in ss:
         # this is one of the frames, with its gradients
         # Make it the saem way than for the coordinates
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
    syms, trafo = cs

    ys2 = [trafo(y) for y in ys]

    return carts_to_int(ys2, xs, allval, cell, tomove, howmove, withs)

def carts_to_int(ys, xs, allval, cell, tomove, howmove, withs):
    """
    This does exactly the same as above, but
    without the calculation of the path, this
    ensures that not only exactly the number of
    bead positions is taken but also that it's
    exactly the beads which are used to create
    the frames
    """
    beads = []

    for i,cart in enumerate(ys):
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

def helpfun(name):
    from pts.tools.path2plot import path2plot_help
    if name == "table":
        print __doc__
    elif name in ["plot", "show"]:
        path2plot_help()
    else:
        print >> stderr, "No help text available for current function!"
    exit()

def extract_data(filename, data_ase, other_input, values, ts_estimates, num, iter):
    """
    Extract the data for the beads, if not ase read for paths, and if demanded
    for the transition state estimates.
    """
    from pts.cfunc import Pass_through
    from numpy import linspace
    from sys import stderr, exit

    symbfile, abcis, obj = other_input
    ase, format  = data_ase
    withs, allval, special_vals, appender, __ =  values
    cell, tomove, howmove = appender

    e_a_gr = None
    if ase:
        # Geos are in ASE readable files.
        atoms, y = read_geos_from_file_more([filename], format=format)
        obj =  atoms.get_chemical_symbols(), Pass_through()
        x = linspace(1, len(y))

    elif symbfile is None:
        # path.pickle files.
        x, y, obj, e_a_gr = read_in_path(filename)
    else:
        # User readable input.
        patf = None
        if len(abcis) > 0:
            patf = abcis[iter]
        y, x, __, __ = read_path_coords(filename, patf, None, None)

    # extract the internal coordinates, for path and beads
    beads = beads_to_int(y, x, obj, allval, cell, tomove, howmove, withs)
    beads = np.asarray(beads)
    beads = beads.T
    beads = beads.tolist()

    if ase:
        path = None
        ts_ests_geos = None
    else:
        path = path_to_int(x, y, obj, num, allval, cell, tomove, howmove, withs)
        # they are wanted as arrays and the other way round
        path = np.asarray(path)
        path = path.T
        path = path.tolist()

        if not ts_estimates == []:
            # Some TS estimates have been requested. Extract also the data for them.
            assert not e_a_gr == None, "ERROR: No energies and forces given but required for TS estimate"
            ts_ests_geos, ts_places = ts_estimates_in_int(x, y, obj, e_a_gr, ts_estimates, allval, cell, tomove, howmove, withs, False)
            ts_ests_geos = np.array(ts_ests_geos)
            ts_ests_geos = ts_ests_geos.T
            ts_ests_geos = ts_ests_geos.tolist()
        else:
            ts_ests_geos = None

    # special_vals requests energy or gradient results. They are only available
    # in path.pickle files. then e_a_gr contains energies and gradients. There
    # will be created a spline over them.
    if special_vals != []:
        assert (e_a_gr is not None)
        for s_val in special_vals:
             # use the options for x and plot the data gotten from
             # the file directly
             en, gr = e_a_gr
             if s_val.startswith("en"):
                beads.append(en)
             elif s_val.startswith("gr"):
                val = s_val[3:]
                beads.append(grads_from_beads(x, y, gr, val))
             else:
                print >> stderr, "ERROR: This option is not available for the current function", s_val
                exit()

             if path is not None:
                if s_val.startswith("en"):
                    if s_val in ["energy2", "energy_slope"]:
                        path.append(energy2_from_path(x, y, en, gr, num))
                    else:
                        path.append(energy_from_path(x, en, num))
                elif s_val.startswith("gr"):
                    val = s_val[3:]
                    path.append(grads_from_path(x, y, gr, num, val))

             if not ts_estimates == []:
                if s_val.startswith("en"):
                    if s_val in ["energy2", "energy_slope"]:
                        ts_ests_geos.append(energy2_from_path_points(x, y, en, gr, ts_places))
                    else:
                        ts_ests_geos.append(energy_from_path_points(x, en, ts_places))
                elif s_val.startswith("gr"):
                    val = s_val[3:]
                    ts_ests_geos.append(grads_from_path_points(x, y, gr, ts_places, val))

    return beads, path, ts_ests_geos


def main(name, argv):
    """
    Reads in stuff from the sys.argv if not
    provided an other way

    Then calculate according to the need
    the result will be a table with data
    separate for each inputfile
    """
    from sys import stderr
    from pts.io.cmdline import visualize_input

#   if '--help' in argv:
#       helpfun(name)

    filenames, data_ase, other_input, values, path_look, __, for_plot =  visualize_input(name, "table" , argv, -1)

    ase, format = data_ase
    num, ts_estimates, __ = path_look

    if name == "xyz":
        ase = True
        format = "xyz"
        data_ase = ase, format
        num = -1

    __, allval, special_vals, __, __ =  values

    # For each file prepare the plot
    for i, filename in enumerate(filenames):

        if num > 0:
            beads, path, ts_ests_geos = extract_data(filename, data_ase, other_input, values, ts_estimates, num, i)
            coords = path
        else:
            beads, path, ts_ests_geos = extract_data(filename, data_ase, other_input, values, ts_estimates, 2, i)
            coords = beads

        if not ts_ests_geos == None:
            coords = ts_ests_geos
            text = "#Given are the values for the chosen transition state estimates of the path\n"
        elif num > 0:
            coords = path
            text = "#Given are the values for %i interpolated geometries on the path\n" % (num)
        else:
            text = "#Given are the values for the beads of the path\n"
            coords = beads

        # they are wanted as arrays and the other way round
        coords = np.asarray(coords)
        coords = coords.T
        coords = list(coords)
        write_results(coords, allval, special_vals, filename, text)


def write_results(coords, allval, special_vals, filename, text):
     """
     Writes the results out as a table
     """
     from pts.tools.xyz2tabint import whatsort
     write = stdout.write
     write("#chart of internal coordinates in the run \n")
     write("#observed in file: %s  \n" % (filename) )
     write(text)
     write("#the following values were calculated, distances are in Angstroms; angles in degrees;\n")
     write("#                                      energies in eV and forces in eV/ Angstroms\n")
     for k in range(len(allval)):
     # tell what values will be calulated
         order, partners = allval[k]
         write("%s with atoms :" % (whatsort(order)))
         for number in partners:
             write("%i " % number )
         write(";")
     for sv in special_vals:
         if sv.startswith("en"):
             write(" energy;")
         elif sv.startswith("gr"):
                 write(" %s of gradients ;" % (sv[2:]))
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
    main( "path", sargv[1:])

