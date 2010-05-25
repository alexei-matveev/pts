#!/usr/bin/python
"""
This tools takes  path.pickle files
reads it in and gives a picture back of some preselected internal coordiantes

As input the path.pickle file(s) have to be given and at least two
internal coordinates have to be choosen

An internal coodinate is selected by settings --kind n1 n2 ...
where the ni's are the atomnumbers (starting with 1) which should be used
for getting the internal coordinate, how many of them are requiered is dependent
on the kind choosen. There are the possiblilities:

"internal coordinate"  "kind"  "number of Atoms needed"
      distance           dis      2
        angle            ang      3
 angle not connected     ang4     4
    dihedral angle       dih      4
 distance to  plane      dp       4 (the first is the atom, the others define the plane;
                                     the plane atoms must not be on a line)

der are other options which may be set:
    --diff                       :for the next two internal coordinates the difference 
                                  will be taken, instead of the values
    --expand cellfile expandfile : the atoms will be exanded with atoms choosen as
                                   original ones shifted with cell vectors, as described
                                   in expandfile.
                                  cellfile should contain the three basis vectors for the cell
                                  expandfile contains the shifted atoms with:
                                  "number of origin" "shift in i'th direction"*3
    --title string               : string will be the title of the picture
    --xlabel string              : string will be the label in x direction
    --ylabel string              : string will be the label in y direction
    --logscale z                 : for z = 1,2 or z = x,y sets scale of this direction to
                                   logarithmic
    --name string                : string will be the name of the path-plot, for several
                                   files the names could be given by repeatly set --name string
                                   but in this case the name of the i'th call of --name option
                                   always refers to the i'th file, no matter in which order given

"""

from aof.path import Path
from sys import exit
from sys import argv as sargv
from pickle import load
from aof.tools.path2xyz import read_in_path
from aof.tools.xyz2tabint import returnall, interestingvalue, expandlist
from aof.tools.tab2plot import plot_tabs
import numpy as np


def path_to_int(x, y, cs, num, allval, cell, tomove, howmove):
    """
    Gives back the internal values for allval
    which appear on num equally spaced (in x-direction) on
    the path
    """
    path1 = Path(y, x)
    path = []

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
         cs.set_internals(coord)
         cart =  cs.get_cartesians()
         if cell != None:
             cart2 = list(cart)
             expandlist(cart2, cell, tomove, howmove)
             cart = np.array(cart2)
         path.append(returnall(allval, cart, True, i))

    return path


def beads_to_int(ys, cs, allval, cell, tomove, howmove):
    """
    This does exactly the same as above, but
    without the calculation of the path, this
    ensures that not only exactly the number of
    bead positions is taken but also that it's
    exactly the beads which are used to create
    the frames
    """
    beads = []

    for i,y in enumerate(ys):
         cs.set_internals(y)
         cart =  cs.get_cartesians()
         if cell != None:
             cart2 = list(cart)
             expandlist(cart2, cell, tomove, howmove)
             cart = np.array(cart2)
         beads.append(returnall(allval, cart, True, i))

    return beads

def main(argv=None):
    """
    Reads in stuff from the sys.argv if not
    provided an other way

    Then calculate according to the need
    the result will be a picture showing
    for each inputfile a path of the given
    coordinates with beads marked on them
    """
    if argv is None:
        argv = sargv[1:]

    if argv[0] == '--help':
        print __doc__
        exit()

    # store the files containing the pathes somewhere
    filenames = []
    # the default values for the parameter
    num = 100
    diff = []
    symm = []
    symshift = []
    logscale = []
    allval = []
    cell = None
    tomove = None
    howmove = None

    # axes and title need not be given for plot
    title = None
    xlab = None
    ylab = None
    xran = None
    yran = None
    names_of_lines = []

    # count up to know which coordinates are special
    num_i = 1

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
                 # test if the following two coordinate (or coord. diffs)
                 # follow the same symmetry
                 symm.append(num_i)
                 try:
                     m = float(argv[1])
                     symshift.append([num_i, m])
                     argv = argv[1:]
                 except:
                     pass
                 argv = argv[1:]
             elif option in ["dis", "2", "ang","3", "ang4", "4", "dih", "5", "dp", "6"]:
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
             elif option == "expand":
                 # done like in xyz2tabint, expand the cell
                 #FIXME: There should be a better way to consider atoms
                 # to be shifted to other cells
                 cell, tomove, howmove = get_expansion(argv[1], argv[2])
                 argv = argv[3:]
             elif option == "title":
                 title = argv[1]
                 argv = argv[2:]
             elif option == "xlabel":
                 xlab = argv[1]
                 argv = argv[2:]
             elif option == "ylabel":
                 ylab = argv[1]
                 argv = argv[2:]
             elif option == "name":
                 names_of_lines.append(argv[1])
                 argv = argv[2:]
             elif option == "xrange":
                 xran = [ float(argv[1]), float(argv[2])]
                 argv = argv[3:]
             elif option == "yrange":
                 yran = [ float(argv[1]), float(argv[2])]
                 argv = argv[3:]
             elif option.startswith("logscale"):
                 logscale.append(argv[1])
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

    # plot environment
    pl = plot_tabs(title = title, x_label = xlab, y_label = ylab, log = logscale)

    # extract which options to take
    opt, num_opts, xnum_opts, optx = makeoption(num_i, diff, symm, symshift)

    for i in range(len(filenames)):
        # ensure that there will be no error message if calling
        # names_of_lines[i]
        names_of_lines.append([])

    # For each file prepare the plot
    for i, filename in enumerate(filenames):
        # read in the path
        x, y, obj = read_in_path(filename)
        # extract the internal coordiantes, for path and beads
        beads = beads_to_int(y, obj, allval, cell, tomove, howmove)
        path = path_to_int(x, y, obj, num, allval, cell, tomove, howmove)
        # they are wanted as arrays and the other way round
        path = np.asarray(path)
        path = path.T
        beads = np.asarray(beads)
        beads = beads.T
        name_p = str(i + 1)
        if names_of_lines[i] != []:
             name_p = names_of_lines[i]

        # prepare plot from the tables containing the path and bead data
        pl.prepare_plot( path, str(i +1), beads, "_nolegend_", None, None, opt)

    # now plot
    pl.plot_data(xrange = xran, yrange = yran )

def get_expansion(celldat, expandlist):
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
     filemove = open(expandlist,"r" )
     tomove = []
     howmove = []
     # tomove[i] = number of source atom for atom (i + all_in_original_cell)
     # howmove[i] = how to get the new atom
     for  num, line  in enumerate(filemove):
            fields = line.split()
            tomove.append( int(fields[0]))
            howmove.append([float(fields[1]), float(fields[2]), float(fields[3])])

     return cell, tomove, howmove

def makeoption(num_i, diff, symm, symshift):
     """
     All coordinates generated are used
     For all pairs given by diff the difference
     is taken, all other values are taken as they are
     """
     opt = ""
     second = False
     for i in range(1, num_i):
          if i in symm:
              opt += " s"
              for k, m in symshift:
                  if k == i:
                      opt += " %f" % (m)
          if i in diff:
              opt += " d %i" % (i)
              second = True
          elif second:
              opt += " %i" % (i)
              second = False
          else:
              opt += " t %i" % (i)
     return opt

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()

