#!/usr/bin/env python
"""
This tools takes  pathes given by files (path.pickle  or others) reads
it  in  and  gives  a   picture  back  of  some  preselected  internal
coordinates

As input the  path file(s) have to be given and  at least two internal
coordinates

An internal coodinate is selected  by settings --kind n1 n2 ...  where
the ni's  are the atomnumbers (starting  with 1) which  should be used
for getting the internal coordinate, how many of them are requiered is
dependent on the kind choosen. There are the possiblilities:

    +---------------------+----+------+
    |internal coordinate  |kind|number|
    |                     |    |of    |
    |                     |    |Atoms |
    |                     |    |needed|
    +---------------------+----+------+
    |distance             |dis |2     |
    +---------------------+----+------+
    |angle                |ang |3     |
    +---------------------+----+------+
    |angle not connected  |ang4|4     |
    +---------------------+----+------+
    |dihedral angle       |dih |4     |
    +---------------------+----+------+
    |distance to plane    |dp  |4 (a) |
    +---------------------+----+------+

    (a) The first is the atom,  the others define the plane; the plane
    atoms must not be on a line.

Normally  the first  internal coordinate  will be  the x-value  of the
plot. Alternatly one might use the path abscissa instead. For this one
has to set the option --t.

Another set  of coordinates refers  to energy and gradients  stored in
path.pickle files.  For the gradients are several different options of
interest possible.  The  values on the path will  be interpolated from
the beads, else the ones from the beads are taken.

The energy/gradient  informations are always given  after the geometry
informations:

    --energy

        energies

    --gradients

        gives  the absolute  value of  the gradients  at  the required
        positions

    --grmax

        gives maximal value of (internal) gradients

    --grpara

        length of gradient component parallel to the path

    --gperp

        length of gradient component perpendicular to the path

    --grangle

        angle (in degree) between path and gradients, should be 90 for
        convergence

Easiest  input is  by path.pickle  files which  can be  given directly
without need of any option.

Some options  handle a  different way of  input. Here  coordinates are
given in cordinate files  (coordinates in internal coordinates for all
beads). One has to set addtionally at least the symbols.

    --symbols symfile

        file should contain all the symbols of the atoms

    --zmat zmatfile

        to  switch from  Cartesian to  Zmatrix coordinates,  should be
        given the same way as for path tools

    --mask maskfile geo_raw

        mask  has to  be  given separately  in  maskfile one  complete
        internal geometry  has to be  given as geo_raw to  provide the
        tool with the fixed values

    --abscissa abscissafile

        the abscissa for the coordinate file. There need not to be any
        but if  there are some there  need to be exactly  one file for
        each coordinate file

der are other options which may be set:

    --diff

        for the  next two internal coordinates the  difference will be
        taken, instead of the values

    --expand cellfile expandfile

        the atoms will be exanded  with atoms choosen as original ones
        shifted  with  cell   vectors,  as  described  in  expandfile.
        cellfile should  contain the three basis vectors  for the cell
        expandfile contains the shifted atoms with: "number of origin"
        "shift in i'th direction"*3

    --title string

        string will be the title of the picture

    --xlabel string

        string will be the label in x direction

    --ylabel string

        string will be the label in y direction

    --logscale z

        for  z =  1,2  or z  = x,y  sets  scale of  this direction  to
        logarithmic

    --name string

        string will  be the name  of the path-plot, for  several files
        the names could be given  by repeatly set --name string but in
        this case  the name of the  i'th call of  --name option always
        refers to the i'th file, no matter in which order given

    --log filename string num

        reads in filename which should be a .log file output file of a
        string or neb calculation and  takes string line of the num'th
        iteration as some extra bead data to plot the x-value used for
        this plot are  the same x-values as from  the last path.pickle
        given before this option

    --lognf filename

        new log as before, but reuses the string and num from the last
        one only the logfile is changed

    --lognn num

        new log  plot as above,  but takes another iteration  than the
        last one

    --symm <midx>

        uses  the  next coordinate  defined  (or  the next  difference
        defined) and  returns the  derivative to symmetry  around midx
        instead of the value if midx is not given, 0 is used, symmetry
        is calculated by: 0.5 * (f(midx + x) - f(midx - x))
"""
from sys import exit
from sys import argv as sargv
from pts.tools.path2xyz import read_in_path
from pts.tools.path2tab import energy_from_path, grads_from_path, grads_from_beads
from pts.tools.pathtools import read_path_fix, read_path_coords
from pts.tools.xyz2tabint import interestingvalue
from pts.tools.path2tab import path_to_int, beads_to_int, reorder_files, get_expansion
from pts.tools.path2tab import read_line_from_log
from pts.tools.tab2plot import plot_tabs
from pts.cfunc import Pass_through
from pts.io.read_COS import read_geos_from_file_more
import numpy as np


def main(argv):
    """
    Reads in stuff from the sys.argv if not provided an other way

    Then calculate according to the  need the result will be a picture
    showing for  each inputfile a  path of the given  coordinates with
    beads marked on them
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
    num = 100
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
             # stop cycle if all input is read in (sometimes more than
             # one is read in at the same time)
             break
         elif argv[0].startswith("--"):
             # differenciate between options and files
             option = argv[0][2:]
             if option == "num":
                 # change number  of frames in  path from 100  to next
                 # argument
                 num = int(argv[1])
                 argv = argv[2:]
             elif option == "diff":
                 # of the next  twoo coordinates the difference should
                 # be taken, store the number of the first of them
                 diff.append(num_i)
                 argv = argv[1:]
             elif option == "symm":
                 # test if the  following coordinate (or coord. diffs)
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
                 # count up,  to know how many and  more important for
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
                 # done  like in  xyz2tabint, expand  the  cell FIXME:
                 # There should  be a better way to  consider atoms to
                 # be shifted to other cells
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

    # plot environment
    pl = plot_tabs(title = title, x_label = xlab, y_label = ylab, log = logscale)

    # extract which options to take
    opt, num_opts, xnum_opts, optx = makeoption(num_i, diff, symm, symshift, withs)

    for i in range(len(filenames)):
        # ensure that there will be no error message if calling
        # names_of_lines[i]
        names_of_lines.append([])

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

        # extract the internal coordiantes, for path and beads
        beads = beads_to_int(y, x, obj, allval, cell, tomove, howmove, withs)
        beads = np.asarray(beads)
        beads = beads.T
        if ase:
            path = None
        else:
            path = path_to_int(x, y, obj, num, allval, cell, tomove, howmove, withs)
            # they are wanted as arrays and the other way round
            path = np.asarray(path)
            path = path.T
        name_p = str(i + 1)
        if names_of_lines[i] != []:
             name_p = names_of_lines[i]

        # prepare plot  from the tables  containing the path  and bead
        # data only if there are enough for x AND y values
        if num_opts > 1:
            if ase:
               pl.prepare_plot( None, None, None, "_nolegend_", beads, name_p, opt)
            else:
               pl.prepare_plot( path, name_p, beads, "_nolegend_", None, None, opt)

        # if some data  has been extracted from a  logfile, after this
        # file i has been used it  has to be plotted here, as here the
        # x values of the files  are valid the log_points should be at
        # the beads
	if special_vals != []:
            assert (e_a_gr is not None)
            for s_val in special_vals:
                 # use the options for x and plot the data gotten from
                 # the file directly
                 optlog = optx + " t %i" % (xnum_opts + 1)
                 log_points = beads
                 log_points = log_points[:xnum_opts + 1,:]
                 log_points = log_points.tolist()
                 # till here  the x-data  should be copied  and ready,
                 # now add also the logdata
		 en, gr = e_a_gr
                 if s_val.startswith("en"):
                    log_points.append(en)
                 elif s_val.startswith("gr"):
                    val = s_val[2:]
                    log_points.append(grads_from_beads(x, y, gr, val))

                 log_points = np.asarray(log_points)

                 if path is not None:
                    log_path = path
                    log_path = log_path[:xnum_opts + 1,:]
                    log_path = log_path.tolist()
                    if s_val.startswith("en"):
                        log_path.append(energy_from_path(x, en, num ))
                    elif s_val.startswith("gr"):
                        val = s_val[2:]
                        log_path.append(grads_from_path(x, y, gr, num, val))
                 if ase:
                    pl.prepare_plot( None, None, None, "_nolegend_", log_points,
                               s_val + " %i" % (i + 1), optlog)
                 else:
                    pl.prepare_plot( log_path, s_val + " %i" % (i + 1),
                               log_points, "_nolegend_", None, None, optlog)

        if logs != []:
            for j, log in enumerate(logs):
                if log_x_num[j] == i:
                 # use the options for x and plot the data gotten from
                 # the file directly
                 optlog = optx + " t %i" % (xnum_opts + 1)
                 log_points = beads
                 log_points = log_points[:xnum_opts + 1,:]
                 log_points = log_points.tolist()
                 # till here  the x-data  should be copied  and ready,
                 # now add also the logdata
                 log_points.append(read_line_from_log(log, logs_find[j], logs_num[j]))
                 log_points = np.asarray(log_points)
                 # The name should be the name of the data line taken,
                 # right?
                 pl.prepare_plot( None, None, None, None, log_points,\
                               logs_find[j] + ', iteration %i' % (logs_num[j]) , optlog)

    # now plot
    pl.plot_data(xrange = xran, yrange = yran )

def makeoption(num_i, diff, symm, symshift, withs):
     """
     All coordinates  generated are used  For all pairs given  by diff
     the difference is taken, all other values are taken as they are
     """
     opt = ""
     optx = []
     second = False
     # store some information about how many values considerd
     many = 0
     xmany = 0
     count = 0

     v = 0
     if withs:
         v = 1
         many += 1
         xmany = 1
         optx = " t %i" % (1)
         opt = optx

     for i in range(1, num_i):
          if i in symm:
              opt += " s"
              for k, m in symshift:
                  if k == i:
                      opt += " %f" % (m)
              opt += optx
          if i in diff:
              opt += " d %i" % (i + v)
              second = True
          elif second:
              opt += " %i" % (i + v)
              second = False
              many += 1
              count += 2
          else:
              opt += " t %i" % (i + v)
              many += 1
              count += 1
          if many == 1 and xmany == 0:
              xmany = count
              optx = opt
     # return:  all options,  how many  lines  in the  plot, how  many
     #          options belong to x,  (as some like symm or difference
     #          use more than  one) what are the options  only for the
     #          xfunction
     return opt, many, xmany, optx

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main(sargv[1:])

