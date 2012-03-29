#!/usr/bin/env python
"""
This tool helps to visualize the key variables in transition
state refinement calculations, namely, dimer, lanczos and
quasi-Newton calculations. It basically takes data from two
pickled files, namely *.log.pickle which contains iteration
information and *.dict.pickle which contains all the electronic
structure calculation results.
The variables includes internal coordinates, energy, gradients,
step size, curvature, and angles between various special directions
in phase space, eg. mode.

The command line has to specify at least two coordinates or other
variables to visualize, as well as input files.

Most variables are specified the same way as path2plot tool, with

    --<kind> n1 n2 ...

The names and short explanations are:

for internal coordinates:

    --dis n1 n2

        atom distance

    --ang n1 n2 n3

        angle

    --ang4 n1 n2 n3 n4

        angle not connected

    --dih n1 n2 n3 n4

        dihedral angle

    --dp n1 n2 n3 n4

        distance to plane

electronic structure variables (for center point):

    --energy (or --en)

        energy

    --gradients (or --grabs)

        absolute value of gradient (in the same coordination
        as the calculation is done, the same for below)

    --grmax

        maximal component of gradient

    --grpara

        gradient component parallel to the mode

    --grperp

        gradient component perpendicular to the mode

    --grangle

        angle mode and gradient

characteristics for dimer/lanczos:

    --step

        translation step size

    --curvature (or --curv)

        curvature along the lowest mode

    --vec_angle dr1 dr2

        this gives the angle between two directions,
        dr1 and dr2, in phase space
        If dr1 and dr2 are the same the angle between
        succeding steps will be taken.

        these directions can be:
            mode: lowest mode vector
            grad: gradient
            step: translation step
            translation: accumulated translation from initial point
            init2final: a constant vector from initial to final point
            initmode: initial mode vector
                (attention: this is the mode after the rotation
                step, rather than the input one)
            finalmode: mode of the last step

Other available options:

    --diff

        the difference of next two internal coordinates

    --s (or --t)

        step number

    --expand cellfile expandfile

        the atoms will be exanded  with atoms choosen as original ones
        shifted  with  cell   vectors,  as  described  in  expandfile.
        cellfile should  contain the three basis vectors  for the cell
        expandfile contains the shifted atoms with: "number of origin"
        "shift in i'th direction"*3

    --title string

        string will be the title of the picture

    --xlabel (or ylabel) string

        string will be the label in x (or y) direction

    --xrange (or yrange) int1 int2

        specify the range of x (or y) axis

    --logscale z

        for  z =  1,2  or z  = x,y  sets  scale of  this direction  to
        logarithmic

    --name string

        string will be the name of the plot, for several files
        the names could be given by repeatly setting --name string
        but in this case  the name of the  i'th call of --name option
        always refers to the i'th file, no matter in which order given

    --output filename

        save the figure as file and do NOT show it on screen

    --arrow <len>

        an arrow will be plotted in addition to each beads, indicating
        the mode direction projected on the internal coordinates. Only
        applicable to internal coordinates plot.

Input files:

    Input file names should be specified without "--". They should be a pickle
    file from a paratools dimer / paratools lanczos or a paratools quasi-newton
    calculation.
"""
import sys
import numpy as np
from pts.path import Path
from numpy.linalg import norm

def main(argv):

    from pts.tools.tab2plot import setup_plot,prepare_plot, plot_data, colormap
    from pts.tools.path2plot import makeoption
    from pts.tools.path2tab import beads_to_int
    from pts.io.cmdline import visualize_input
    from pts.tools.dimer2xyz import read_from_pickle_log

    log_file, __, __, values, __, dimer_special, for_plot =  visualize_input("progress", "plot", argv, -1)

    num_i, logscale, title, xlab, xran, ylab, yran, names_of_lines, outputfile = for_plot
    withs, allval, special_val, appender, special_opt =  values
    diff, __, __ =  special_opt
    cell, tomove, howmove = appender

    # These two are extra values only for the progress tools
    arrow_len, vec_angle_raw = dimer_special

    decrease = [0, 0]

    #default values of interesting parameters
    vec_angle = interestingdirection()

    #FIXME: interestingdirection class is a bit unhandy. Maybe exchange it soon.
    for raw in vec_angle_raw:
        m1, m2 = raw
        value = "vec"
        # For both vectors find the code and whether the first or last
        # points of the other variables have to be removed in order to have
        # all vectors of the same length.
        for mi in m1, m2:
            vec_angle.set_name(mi)
            range_v = vec_angle.get_range()
            for i, j in enumerate(range_v):
                 if not j == 0:
                    decrease[i] = j
            value += vec_angle.get_string()

        if value[3:5] == value[5:]:
            decrease[0] = -1
        special_val.append(value)

    if arrow_len == None:
        arrow = False
    else:
        arrow = True

    if "step" in special_val:
         # There will be one value less for the step, as it is the difference to the last
         # one.
         decrease[1] = 1

    # plot environment
    setup_plot(title = title, x_label = xlab, y_label = ylab, log = logscale)

    # extract which options to take
    opt, num_opts, xnum_opts, optx = makeoption(num_i, diff, [], [], withs)

    # decide how many color are used
    n = len(log_file) * (num_opts - 1 + len(special_val))
    for i, geo_file in enumerate(log_file):
        # ensure that there will be no error message if calling
        # names_of_lines[i]
        names_of_lines.append([])
        # extract refinement process information from log.pickle file
        obj, geos, modes, curv, energy, grad = read_from_pickle_log(geo_file)
        # initial and final point to plot used because arrays to be plotted may have
        # different lengths
        ifplot = [0, len(geos)]
        ifplot = [ip - dec for ip, dec in zip(ifplot, decrease)]

        x = list(range(0,len(geos)))[ifplot[0]: ifplot[1]]
        # internal coordinates for plotting
        beads = beads_to_int(geos[ifplot[0]: ifplot[1]], x, obj, \
                        allval, cell, tomove, howmove, withs)
        beads = np.asarray(beads)
        beads = beads.T
        # name of the plot
        name_p = str(i + 1)
        if names_of_lines[i] != []:
             name_p = names_of_lines[i]
        # make plot only when there are enough points
        if num_opts > 1:
            def choose_color(j):
                 return colormap(i * (num_opts - 1 + len(special_val)) + j, n)

            # data only if there are enough for x AND y values
            colors = map(choose_color, range(0, num_opts))
            prepare_plot( None, None, None, None, beads, name_p, opt, colors)

            if arrow:
                for j in range(ifplot[0], ifplot[1]):
                    # for each bead we plotted, a line is added to the center point
                    # indicating the dimer mode direction projected in internal coordinates

                    assert(arrow_len > 0)

                    # use finite difference method to extract the projection of modes
                    arrows = beads_to_int([geos[j] - modes[j] * arrow_len, geos[j] + modes[j] * arrow_len], \
                                [x[j]] * 2, obj, allval, cell, tomove, howmove, withs)
                    arrows = np.asarray(arrows)
                    arrows = arrows.T

                    #arrows = rescale(arrows, arrow_len)


                    prepare_plot(arrows, "_nolegend_", None, None, None, None, opt, colors)

        if special_val != []:
            # Two kinds of dictionary store share the same interface
            for j, s_val in enumerate(special_val):
                # use the options for x and plot the data gotten from
                # the file directly

                optlog = optx + " t %i" % (xnum_opts + 1)
                log_points = beads
                log_points = log_points[:xnum_opts + 1,:]
                log_points = log_points.tolist()

                if s_val.startswith("en"):
                    log_points.append(energy[ifplot[0]: ifplot[1]])
                elif s_val.startswith("gr"):
                    val = s_val[3:]
                    log_points.append(grads_dimer(modes, grad[ifplot[0]: ifplot[1]], val))
                elif s_val.startswith("curv"):
                    log_points.append(curv)
                elif s_val.startswith("step"):
                    log_points.append(stepsize(geos[ifplot[0]: ifplot[1]+1]))
                elif s_val.startswith("vec"):
                    val = s_val[3:]
                    log_points.append(array_angles(*vec_angle(val, modes, geos, grad, ifplot)))
                log_points = np.asarray(log_points)

                prepare_plot( None, None, None, None, log_points, \
                               s_val + " " + name_p, optlog, [colormap(i * (num_opts - 1 + len(special_val)) + j, n)])

    # finally make the picture visible
    plot_data(xrange = xran, yrange = yran, savefile = outputfile)

def rescale(arrow, arrow_len):
    """
    this function rescale the arrow to the given length, while keeping the
    direction and center point
    """

    # calculate initial length
    length = norm([arrow[i][0] - arrow[i][1] for i in range(1, len(arrow))])

    for i in range(1, len(arrow)):
        add = sum(arrow[i]) / 2
        minus = (arrow[i][1] - arrow[i][0]) / 2
        arrow[i][0] = add + minus / length * arrow_len
        arrow[i][1] = add - minus / length * arrow_len

    return arrow

class interestingdirection:
    """
    This class contains definitions and functions to deal with vector angles in
    phase space
    """
    def __init__(self):
        self.direction_type = {
                                   "variable": ["mode", "grad", "step", "translation"],
                                       # mode direction, gradient direction, step direction
                                       # and the direction of accumulated translation from
                                       # initial point
                                   "constant": ["init2final", "initmode", "finalmode"],
                                       # initial point to final point, initial mode, final mode
                                       # these are constant for every step
                              }

    def __call__(self, opt_string, modes, geos, grs, ifplot):
        """
        call the class to give back required directions so that the angles can be calculated
        in a uniform way
        """
        # there should be 4 characters to specify 2 directions,
        # and none of them could be in the category of special
        assert len(opt_string) == 4 and opt_string[2] != "s"
        directs = []
        acute = False

        a = opt_string[:2]
        b = opt_string[2:]

        start = ifplot[0]
        end = ifplot[1]

        for opt in a, b:
            # the former and later 2 characters in opt_string
            # represents a variable, respectively
            if opt[0] == "v":
                if opt[1] == "0":
                    # mode has no directions
                    directs.append(modes[start: end])
                    acute = True
                elif opt[1] == "1":
                    # gradients
                    directs.append(grs[start: end])
                elif opt[1] == "2":
                    # translation steps
                    directs.append(step_from_beads(geos[start: end + 1]))
                elif opt[1] == "3":
                    # accumulated translations
                    directs.append([geo - geos[0] for geo in geos[start: end]])
                else:
                    print >> sys.std_err, "ERROR: invalid vector choice", opt[1]
                    sys.exit()
            elif opt[0] == "c":
                if opt[1] == "0":
                    # initial to final structure
                    directs.append([geos[-1] - geos[0]] * (end - start))
                if opt[1] == "1":
                    # initial mode
                    directs.append([modes[0]] * (end - start))
                if opt[1] == "2":
                    # final mode
                    directs.append([modes[-1]] * (end - start))
                else:
                    print >> sys.std_err, "ERROR: invalid vector choice", opt[1]
                    sys.exit()
            else:
                print >> sys.std_err, "ERROR: invalid option"
                sys.exit()

            if a == b:
                start = start - 1
                end = end - 1

        return directs, acute

    def set_name(self, option):
        """
        set what to extract according to input option
        """
        valid_option = False
        for key in self.direction_type:
            # first check whether the option is legal
            if option in self.direction_type[key]:
                self.type = key
                valid_option = True

        if valid_option:
            self.name = option
        else:
            print "This input variable is not valid:"
            print option
            print "Please check your input"
            exit()

    def get_string(self):
        """
        generate the option string which will be used in plot
        """
        index = self.direction_type[self.type].index(self.name)
        return self.type[0]+str(index)

    def get_range(self):
        """
        set the range of step for plot, some directions are not
        available for certain steps
        """
        range_1 = [0, 0]
        if self.name in ["step"]:
            range_1[1] = 1
        elif self.name in self.direction_type["special"]:
            range_1[1] = 1
            range_1[0] = -1
        return range_1


def grads_dimer(mode, gr, allval):
    """
    Gives back the values of the gradients as decided in allval
    which appear on the beads, path informations are needed
    for finding the mode along the path
    """
    from sys import stderr
    grs = []

    for i, gr_1 in enumerate(gr):
        if allval == "abs":
            # Total forces, absolute value
            grs.append(norm(gr_1))
        elif allval == "max":
            # Total forces, maximal value
            grs.append(max(abs(gr_1)))
        elif allval == "para":
            # parallel part of forces along the lowest modes
            mode_1 = mode[i]
            gr_2 = np.dot(mode_1, gr_1)
            grs.append(gr_2)
        elif allval == "perp":
            # absolute value of perp. part of forces along the lowet modes
            mode_1 = mode[i]
            gr_2 = np.dot(mode_1, gr_1)
            gr_1 = gr_1 - gr_2 * mode_1
            grs.append(norm(gr_1))
        elif allval == "angle":
            # angle between forces and path
            mode_1 = mode[i]
            gr_1 = gr_1 / norm(gr_1)
            ang = abs(np.dot(mode_1, gr_1))
            if ang > 1.:
                ang = 1.
            grs.append(np.arccos(ang) * 180. / np.pi)
        else:
            print >> stderr, "Illegal operation for gradients", allval
            exit()
    return grs


def stepsize(geo):
    """
    Simply give back the norm of difference of neighbouring geometries
    """
    return [norm(step) for step in step_from_beads(geo)]

def step_from_beads(geo):
    """
    give back translation steps in vector form
    """
    return [geo[i + 1] - geo[i] for i in range(len(geo) - 1)]

def array_angles(array_list, acute = False):
    """
    given the two arrays of directions, return there angles
    can be set to return the acute angles
    """
    from scipy import arccos, pi
    # use dot product to calculate the angles
    if acute:
        return [arccos(abs(np.dot(array_list[0][i], array_list[1][i])) / (norm(array_list[0][i]) \
                * norm(array_list[1][i]))) / pi * 180 for i in range(len(array_list[0]))]
    else:
        return [arccos(np.dot(array_list[0][i], array_list[1][i]) / (norm(array_list[0][i]) \
                * norm(array_list[1][i]))) / pi * 180 for i in range(len(array_list[0]))]
