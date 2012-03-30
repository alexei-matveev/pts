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

    --symm midx

        uses  the  next coordinate  defined  (or  the next  difference
        defined) and  returns the  derivative to symmetry  around midx
        instead of the value if midx is not given, 0 is used, symmetry
        is calculated by: 0.5 * (f(midx + x) - f(midx - x))

    --output filename

        save the figure as a file and do NOT show it on screen
"""
import sys

def path2plot_help():
    print __doc__


def xyz2plot( argv):
    """
    Should do the same stuff as for path2plot. But expects the input files
    to be xyz files.
    Not all options of the other plot functions are available.
    """
    from pts.tools.path2tab import helpfun, extract_data
    from pts.io.cmdline import visualize_input
    from pts.tools.tab2plot import setup_plot, plot_data, prepare_plot, colormap

    name = "xyz"

    filenames, __, __, values, __, __, for_plot =  visualize_input(name, "plot", argv, -1)

    # Expand the output, we need it further.
    withs, allval, special_vals, appender, special_opt =  values
    diff, symm, symshift =  special_opt
    num_i, logscale, title, xlab, xran, ylab, yran, names_of_lines, outputfile = for_plot
    cell, tomove, howmove = appender

    # plot environment
    setup_plot(title = title, x_label = xlab, y_label = ylab, log = logscale)

    # extract which options to take
    opt, num_opts, xnum_opts, optx = makeoption(num_i, diff, symm, symshift, withs)

    n = len(filenames) * (num_opts - 1)
    for i, filename in enumerate(filenames):

        # Extract the data for beads, path if availabe and TS estimates if requested (else None).
        beads, __, __ = extract_data(filename, (True, "xyz"), (None, None, None ), values, [], 0, i)

        # The name belonging to filename.
        name_p = str(i + 1)
        if names_of_lines[i] != []:
             name_p = names_of_lines[i]

        # prepare plot  from the tables  containing the "beads" = coordinate points of the xyz file
        # there are better at least two coordinates, as there will be nothing else.
        def choose_color(j):
             return colormap(i * (num_opts - 1) + j, n)

        colors = map(choose_color, range(0, num_opts))
        prepare_plot( None, None, None, "_nolegend_", beads, name_p, opt, colors)

    # now plot
    plot_data(xrange = xran, yrange = yran, savefile = outputfile )

def main( argv):
    """
    Reads in stuff from the sys.argv if not provided an other way

    Then calculate according to the  need the result will be a picture
    showing for  each inputfile a  path of the given  coordinates with
    beads marked on them
    """
    from pts.tools.path2tab import read_line_from_log, carts_to_int
    from pts.tools.tab2plot import setup_plot, plot_data, prepare_plot, colormap
    from pts.io.read_COS import read_geos_from_file
    from pts.io.cmdline import visualize_input
    from pts.tools.path2tab import helpfun, extract_data
    import numpy as np
    from copy import copy
    from sys import stderr

    name = "path"

    # interprete arguments, shared interface with path2tab.
    filenames, data_ase, other_input, values, path_look, __, for_plot =  visualize_input(name, "plot", argv, 100)

    # Expand the output, we need it further.
    ase, format_ts = data_ase
    num, ts_estimates, refs = path_look
    reference, reference_data = refs
    withs, allval, special_vals, appender, special_opt =  values
    diff, symm, symshift =  special_opt
    num_i, logscale, title, xlab, xran, ylab, yran, names_of_lines, outputfile = for_plot
    cell, tomove, howmove = appender

    # plot environment
    setup_plot(title = title, x_label = xlab, y_label = ylab, log = logscale)

    # extract which options to take
    optraw, num_opts, xnum_opts, optx = makeoption(num_i, diff, symm, symshift, withs)
    num_opts_raw = copy(num_opts)
    opt = copy(optraw)

    if special_vals != []:
        for s_val in special_vals:
             # use the options for x and plot the data gotten from
             # the file directly
             opt = opt + " t %i" % (num_opts + 1)
             num_opts = num_opts + 1

    n = len(filenames) * (num_opts - 1)

    if not reference == []:
       for ref in reference:
           # Reference point (geometry, energy) to compare the rest data to it.
           # geometry is supposed to be in a ASE readable format, energy in a separate file.
           atom_ref, y_ref = read_geos_from_file([ref], format=format_ts)

           # Reference data for the geometries. The Abscissa is supposed to be on 0.5
           reference_int_geos = np.array(carts_to_int(y_ref, [0.5], allval, cell, tomove, howmove, withs))
           reference_int_geos = reference_int_geos.T
           reference_int_geos = reference_int_geos.tolist()

           optref = optraw
           num_opts_ref = num_opts
           if special_vals != []:
               # From the special vals only the energies can be displaced. Therefore special
               # treatment is required.
               for s_val in special_vals:
                   if s_val.startswith("en") and not reference_data == None:
                       optref = optraw + " t %i" % (num_opts_raw + 1)
                       reference_data = np.loadtxt(reference_data)
                       reference_int_geos.append([reference_data.tolist()])
                   else:
                       num_opts_ref = num_opts_ref - 1

           if num_opts_ref > 1:
               def choose_color(j):
                    return colormap(j, n)

               colors = map(choose_color, range(0, num_opts))
               prepare_plot( None, None, None, "_nolegend_", reference_int_geos, "Reference", opt, colors )


    # For each file prepare the plot
    for i, filename in enumerate(filenames):

        # Extract the data for beads, path if availabe and TS estimates if requested (else None).
        beads, path, ts_ests_geos = extract_data(filename, data_ase, other_input, values, ts_estimates, num, i)

        if ts_ests_geos == []:
            print >> stderr, "WARNING: No transition state found for file", filename
            ts_ests_geos = None

        # The name belonging to filename.
        name_p = str(i + 1)
        if names_of_lines[i] != []:
             name_p = names_of_lines[i]

        # prepare plot  from the tables  containing the path  and bead

        def choose_color(j):
             return colormap(i * (num_opts - 1) + j, n)

        # data only if there are enough for x AND y values
        colors = map(choose_color, range(0, num_opts))
        if num_opts > 1:
            if ase:
               prepare_plot( None, None, None, "_nolegend_", beads, name_p, opt, colors)
            else:
               prepare_plot( path, name_p, beads, "_nolegend_", ts_ests_geos, "_nolegend_", opt, colors)


    # now plot
    plot_data(xrange = xran, yrange = yran, savefile = outputfile )

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
    from pts.tools.tab2plot import colormap
    # from numpy import linspace #, empty, transpose
    # from numpy import array

    #
    # Only position arguments so far.  The first, argv[0], is the name
    # of the method, usually just "plot":
    #
    cmd = argv[0]

    # skip argv[1], it just tells that this precious function has been selected.
    opts, args = getopt.getopt(argv[2:], "", [])
    # print "opts=", opts
    # print "args=", args

    import matplotlib

    if cmd == "plot":
        # do not expect X11 to be available, use a different backend:
        matplotlib.use("Agg")

    from matplotlib import pyplot as plt

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
        from numpy import dot, linspace, sqrt, array, shape, empty
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

