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
    --log  filename string num   : reads in filename which should be a .log file output file of
                                   a string or neb calculation and takes string line of the num'th
                                   iteration as some extra bead data to plot
                                   the x-value used for this plot are the same x-values as from the
                                   last path.pickle given before this option

    --lognf filename             : new log as before, but reuses the string and num from the last one
                                   only the logfile is changed
    --lognn num                  : new log plot as above, but takes another iteration than the last one

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
             xfiles += 1

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
        # only if there are enough for x AND y values
        if num_opts > 1:
            pl.prepare_plot( path, name_p, beads, "_nolegend_", None, None, opt)

        # if some data has been extracted from a logfile, after this file i has been used
        # it has to be plotted here, as here the x values of the files are valid
        # the log_points should be at the beads
        if logs != []:
            for j, log in enumerate(logs):
                if log_x_num[j] == i:
                 # use the options for x and plot the data gotten from the file directly
                 optlog = optx + " t %i" % (xnum_opts + 1)
                 log_points = beads
                 log_points = log_points[:xnum_opts + 1,:]
                 log_points = log_points.tolist()
                 # till here the x-data should be copied and ready, now add also
                 # the logdata
                 log_points.append(read_line_from_log(log, logs_find[j], logs_num[j]))
                 log_points = np.asarray(log_points)
                 # The name should be the name of the data line taken, right?
                 pl.prepare_plot( None, None, None, None, log_points,\
                               logs_find[j] + ', iteration %i' % (logs_num[j]) , optlog)

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
     optx = []
     second = False
     # store some information about how many values considerd
     many = 0
     xmany = 0
     count = 0
     for i in range(1, num_i):
          if i in symm:
              opt += " s"
              for k, m in symshift:
                  if k == i:
                      opt += " %f" % (m)
              many -= 2
          if i in diff:
              opt += " d %i" % (i)
              second = True
          elif second:
              opt += " %i" % (i)
              second = False
              many += 1
              count += 2
          else:
              opt += " t %i" % (i)
              many += 1
              count += 1
          if many == 1 and xmany == 0:
              xmany = count
              optx = opt
     # return: all options, how many lines in the plot, how many options belong
     #          to x, (as some like symm or difference use more than one)
     #          what are the options only for the xfunction
     return opt, many, xmany, optx

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
    main()

