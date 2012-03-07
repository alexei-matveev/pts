#!/usr/bin/env python
"""
This tool  takes the "pickled"  path information from input  files and
produces an  xy-plot of one selected property  against another.  These
properties include, but are not limited to, internal coordinates.

The command line  has to specify at least  two internal coordinates to
be used in the xy-plot and a path to one or more input files.

An internal coordinate is selected by a command line switch

     --<kind> n1 n2 ...

followed by 2, 3, or 4 atomic indices, n1, n2, ..., each between 1 and
the total  number of  atoms.  Together with  the <kind>  these indices
specify a bond, an angle, a dihedral angle or another system property.
The number  of indices depends on  the kind of  internal coordinate as
specified in the table:

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

    --grperp

        length of gradient component perpendicular to the path

    --grangle

        angle (in degree) between path  and gradients, should be 0 for
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

There are other options which may be set:

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

    --output filename

        save the figure as a file and do NOT show it on screen
"""
import sys

def main(argv):
    """
    Reads in stuff from the sys.argv if not provided an other way

    Then calculate according to the  need the result will be a picture
    showing for  each inputfile a  path of the given  coordinates with
    beads marked on them
    """
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

    if len(argv) <= 0:
        # errors go to STDERR:
        print >> sys.stderr, __doc__
        exit()
    elif argv[0] == '--help':
        # normal (requested) output goes to STDOUT:
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

    # output the figure as a file
    outputfile = None

    #
    # Read all the arguments in.  Stop  cycle if all input is read in.
    # Some iterations consume more than one item.
    #
    while len(argv) > 0:
         if argv[0].startswith("--"):
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
             elif option == "output":
                outputfile = argv[1]
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
                        log_path.append(energy_from_path(x, en, num))
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
    pl.plot_data(xrange = xran, yrange = yran, savefile = outputfile )

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

def plot(argv):
    """
    Handles commands

        argv[0] in ("plot", "show")
    """
    import getopt
    from pathtools import unpickle_path
    # from numpy import linspace #, empty, transpose
    # from numpy import array

    #
    # Only position arguments so far.  The first, argv[0], is the name
    # of the method, usually just "plot":
    #
    cmd = argv[0]
    opts, args = getopt.getopt(argv[1:], "", [])
    # print "opts=", opts
    # print "args=", args

    import matplotlib

    if cmd == "plot":
        # do not expect X11 to be available, use a different backend:
        matplotlib.use("Agg")

    from matplotlib import pyplot as plt
    from matplotlib import cm # color management

    def colormap(i, n):
        """
        Returns a color understood by color keyword of plt.plot().

        To choose most appropriate color map see

            http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps
        """

        # we want to map i=0 to 0.0 and i=imax to 1.0
        imax = n - 1
        if imax == 0:
            imax = 1

        #
        # Color maps map the interval [0, 1] ontoto a color palette:
        #
        # return cm.hsv(float(i) / imax)
        return cm.jet(float(i) / imax)

    def plot_energy(energies, color):
        # energy profile:
        plt.title("Energy profile", fontsize="large")
        plt.ylabel("Energy [eV]")
        plt.xlabel("Path point [arb. u.]")
        plt.plot(range(1, len(energies)+1), energies, "o--", color=color)
        plt.xlim((1, len(energies)))

    def plot_energy_with_cubic_spline(geometries, energies, gradients, tangents, abscissas, color):
        """
        Plot the energy profile indicating  the slope of the energy at
        the vertices.

        Tangents are usually normalized,  if at all, quite differently
        from the  convention used  here -- one  unit of  "path length"
        between  neyboring images.  To  avoid visual  mismatch between
        finite difference (E[i+1] - E[i]) / 1.0 and differential slope
        dE  /  ds  =  dot(g,  t)  at  the  nodes  we  invent  a  local
        re-normalization of the tangents.
        """
        # energy profile:
        from numpy import dot, linspace, sqrt, array, vstack, shape, empty
        from pts.func import CubicSpline

        plt.title("Energy profile", fontsize="large")
        plt.ylabel("Energy [eV]")
        plt.xlabel("Path point [arb. u.]")

        if abscissas == None:
            # this will hold distances vectors between path points:
            xdeltas =  empty(shape(geometries))
            xdeltas[0] = geometries[1] - geometries[0]
            xdeltas[-1] = geometries[-1] - geometries[-2]
            xdeltas[1:-1] = (geometries[2:] - geometries[:-2]) / 2.0

            # this is the measure  of the real separatioin between images,
            # each will correspond to one unit on the graph:
            scales = array([sqrt(dot(x, x)) for x in xdeltas])

            # re-normalize tangents:
            tangents = array([s * t / sqrt(dot(t, t)) for s, t in zip(scales, tangents)])
            abscissas = range(1, len(energies)+1)
        else:
            # If the distances in s are explicitely given, expect also the tangents to be
            # right
            tangents = array(tangents)
            abscissas = array(abscissas)

        # projection of the gradient on the new tangents, dE / ds:
        slopes = [dot(g, t) for g, t in zip(gradients, tangents)]

        # cubic spline
        spline = CubicSpline(abscissas, energies, slopes)

        # spline will be plotted with that many points:
        line = linspace(abscissas[0], abscissas[-1], 100)

        plt.plot( abscissas, energies, "o",
                 line, map(spline, line), "-",
                 color = color)
        plt.xlim((abscissas[0], abscissas[-1]))

    # number of curves on the same graph:
    N = len(args)

    #
    # This is the main loop over the input files:
    #
    for i, name in enumerate(args):
        geometries, energies, gradients, tangents, abscissas, symbols, trafo = unpickle_path(name) # v2

        # tuple is printed in one line:
        # print tuple(energies)

        # energy profile:
        if tangents is not None:
            plot_energy_with_cubic_spline(geometries, energies, gradients, tangents, abscissas, colormap(i, N))
        else:
            plot_energy(energies, colormap(i, N))

    if cmd == "show":
        # Display the plot, needs X11:
        plt.show()

    if cmd == "plot":
        # Save in vecor format, allows post-processing:
        plt.savefig("plot.svg")

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main(sys.argv[1:])

