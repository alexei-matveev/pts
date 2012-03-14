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
        one could also give only one key word for direction, which can only
        be modechange or stepchange, which means the angle between neighbouring
        modes or steps.

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

    Input file names should be specified without "--". The calculation log
    file (generally *.log.pickle and the dictionary store file (or directory)
    (generally *.ResultDict.pickle or cache.d/) should both be indicated.
    If multiple inputs are involved, the order of log and dictionary files
    must be the same, though one can either first specify all log files then
    dictionary files, or put the ones associated with the same calculation
    together.
"""
import sys
import numpy as np
from pts.path import Path
from numpy.linalg import norm

def main(argv):

    from pts.tools.xyz2tabint import interestingvalue
    from pts.tools.tab2plot import setup_plot,prepare_plot, plot_data, colormap
    from pts.tools.path2plot import makeoption
    from pts.tools.path2tab import get_expansion
    from pts.memoize import FileStore, DirStore, Store
    from pts.tools.path2tab import beads_to_int

    if len(argv) <= 1:
        # errors go to STDERR:
        print >> sys.stderr, __doc__
        exit()
    elif argv[0] == '--help':
        # normal (requested) output goes to STDOUT:
        print __doc__
        exit()

    #input files
    log_file = []
    dict_file = []
    dict_dir = []   # recording which dictionaries are directories
    xlog = 0        # number of logfiles
    xdict = 0       # number of dictionary files/directories

    #default values of interesting parameters
    vec_angle = interestingdirection()
    diff = []
    arrow = False
    arrow_len = None
    special_val = []
    allval = []
    cell = None
    tomove = None
    howmove = None
    withs = False
    info = set([])         # which additional information should be read from
                           # logfile
    iter_flag = set([])    # because some of the parameters to be plotted are
                           # available only between iterations, eg. trans. step
                           # size, or only in iterations before converge, eg.
                           # mode vector, this flag is used to consider such situation

    # default for plotting options
    title = None
    xlab = None
    ylab = None
    xran = None
    yran = None
    logscale = []
    names_of_lines = []

    # count up to know which coordinates are special
    num_i = 1

    # output the figure as a file
    outputfile = None

    """
    Read in all the arguments.  Stop  cycle if all input is read in.
    Some iterations consume more than one item.
    """
    while len(argv) > 0:
        if argv[0].startswith("--"):
            # distinguish options from input files
            option = argv[0][2:]
            if option in ["s", "t"]:
                # use step number as x-axis
                withs = True
                argv = argv[1:]
            elif option == "diff":
                # of the next two coordinates the difference should
                # be taken, store the number of the first of them
                diff.append(num_i)
                argv = argv[1:]
            elif option == "vec_angle":
                # the angle between two phase space vectors
                # can be mode, force (negative grad), step, accumulated translation
                # translation from start to final, initial/final mode
                # also special angles, like rotation angle (mode change),
                # angles between present and last translation step
                value = "vec"
                argv = argv[1:]
                for j in range(0, 2):
                    vec_angle.set_name(argv[0])
                    iter_flag.update(vec_angle.get_range())
                    value += vec_angle.get_string()
                    argv = argv[1:]
                    if value[3] == "s":
                        break
                special_val.append(value)
            elif option in ["grabs", "grmax", "grperp", "grpara", "grangle"]:
                if option in ["grperp", "grpara", "grangle"]:
                    iter_flag.add(-1)    # because mode is not available for
                                         # the last point.
                special_val.append(option)
                argv = argv[1:]
            elif option in ["gr", "gradients"]:
                print "Warning: using absolute value of gradient"
                special_val.append("grabs")
                argv = argv[1:]
            elif option in ["en", "energy"]:
                special_val.append("energy")
                argv = argv[1:]
            elif option in ["curv", "curvature"]:
                iter_flag.add(-1)
                special_val.append("curvature")
                argv = argv[1:]
            elif option in ["step"]:
                # step size of translation step
                iter_flag.add(-1)
                special_val.append(option)
                argv = argv[1:]
            elif option in ["dis", "2", "ang","3", "ang4", "4", "dih", "5", "dp", "6", "dl", "7"]:
                # these are the possible coordinates, store them
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
            elif option == "expand":
                 # done  like in  xyz2tabint, expand  the  cell FIXME:
                 # There should  be a better way to  consider atoms to
                 # be shifted to other cells
                 cell, tomove, howmove = get_expansion(argv[1], argv[2])
                 argv = argv[3:]
            elif option == "arrow":
                # arrow option is only designed for
                # plot involving only internal coordinates
                # otherwise it will be ignored
                iter_flag.add(-1)
                arrow = True
                try:
                    # if not input explicitly, the arrow length
                    # will be set automatically
                    arrow_len = float(argv[1])
                    argv = argv[2:]
                except ValueError:
                    argv = argv[1:]
                except IndexError:
                    argv = argv[1:]
            elif option == "output":
                # output as file or on screen
                outputfile = argv[1]
                argv = argv[2:]
            # plot options
            elif option == "title":
                title = argv[1]
                argv = argv[2:]
            elif option == "xlabel":
                xlab = argv[1]
                argv = argv[2:]
            elif option == "ylabel":
                ylab = argv[1]
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
            elif option == "name":
                names_of_lines.append(argv[1])
                argv = argv[2:]
            else:
                 # For everything that does not fit in
                 print "This input variable is not valid:"
                 print argv[0]
                 print "Please check your input"
                 print __doc__
                 exit()
        elif "log" in argv[0]:
            # read in log files
            log_file.append(argv[0])
            xlog += 1
            argv = argv[1:]
        elif "Result" in argv[0] or "Dict" in argv[0]:
            # read in dict files
            dict_file.append(argv[0])
            xdict += 1
            argv = argv[1:]
        elif "cache" in argv[0] or ".d/" in argv[0]:
            # read in dict directories
            dict_file.append(argv[0])
            dict_dir.append(xdict)
            xdict +=1
            argv = argv[1:]

    # input file validation:
    # FIXME: for some plot, no dict file is needed, this may not be neccesary
    if not xlog == xdict:
        print "Number of log file and dictionary file are unequal!"
        print "Please check your input"
        print __doc__
        exit

    # plot environment
    setup_plot(title = title, x_label = xlab, y_label = ylab, log = logscale)

    # extract which options to take
    opt, num_opts, xnum_opts, optx = makeoption(num_i, diff, [], [], withs)

    # decide how many color are used
    n = len(log_file)
    for i, geo_file in enumerate(log_file):
        # ensure that there will be no error message if calling
        # names_of_lines[i]
        names_of_lines.append([])
        # extract refinement process information from log.pickle file
        atoms, funcart, geos, modes, curv = read_from_pickle_log(geo_file, info)
        obj = atoms.get_chemical_symbols(), funcart
        # initial and final point to plot used because arrays to be plotted may have
        # different lengths
        ifplot = [0, len(geos["Center"])]
        if 1 in iter_flag:
            ifplot[0] += 1
        if -1 in iter_flag:
            ifplot[1] -= 1
        x = list(np.linspace(0,1,len(geos["Center"])))[ifplot[0]: ifplot[1]]
        # internal coordinates for plotting
        beads = beads_to_int(geos["Center"][ifplot[0]: ifplot[1]], x, obj, \
                        allval, cell, tomove, howmove, withs)
        beads = np.asarray(beads)
        beads = beads.T
        # name of the plot
        name_p = str(i + 1)
        if names_of_lines[i] != []:
             name_p = names_of_lines[i]
        # make plot only when there are enough points
        if num_opts > 1:
            prepare_plot( None, None, None, None, beads, name_p, opt, colormap(i, n))

            if arrow:
                for j in range(ifplot[0], ifplot[1]):
                    # for each bead we plotted, a line is added to the center point
                    # indicating the dimer mode direction projected in internal coordinates

                    # use finite difference method to extract the projection of modes
                    arrows = beads_to_int([geos["Center"][j] - modes[j] * 0.005, geos["Center"][j] + modes[j] * 0.005], \
                                [x[j]] * 2, obj, allval, cell, tomove, howmove, withs)
                    arrows = np.asarray(arrows)
                    arrows = arrows.T

                    # rescale the arrow length to be clear
                    if arrow_len == None:
                        arrow_len = norm([arrows[k][0] - arrows[k][1] for k in range(1, len(arrows))]) * 5
                    arrows = rescale(arrows, arrow_len)

                    prepare_plot(arrows, "_nolegend_", None, None, None, None, opt, colormap(i, n))

        if special_val != []:
            # Two kinds of dictionary store share the same interface
            if i in dict_dir:
                resdict = DirStore(dict_file[i])
            else:
                resdict = FileStore(dict_file[i])
            energy, grad = read_from_store(resdict, geos)

            for s_val in special_val:
                # use the options for x and plot the data gotten from
                # the file directly

                optlog = optx + " t %i" % (xnum_opts + 1)
                log_points = beads
                log_points = log_points[:xnum_opts + 1,:]
                log_points = log_points.tolist()

                if s_val.startswith("en"):
                    log_points.append(energy[ifplot[0]: ifplot[1]])
                elif s_val.startswith("gr"):
                    val = s_val[2:]
                    log_points.append(grads_dimer(modes, grad["Center"][ifplot[0]: ifplot[1]], val))
                elif s_val.startswith("curv"):
                    log_points.append(curv)
                elif s_val.startswith("step"):
                    log_points.append(stepsize(geos["Center"][ifplot[0]: ifplot[1]+1]))
                elif s_val.startswith("vec"):
                    val = s_val[3:]
                    log_points.append(array_angles(*vec_angle(val, modes, geos["Center"], grad["Center"], ifplot)))

                log_points = np.asarray(log_points)

                prepare_plot( None, None, None, None, log_points, \
                               s_val + " " + name_p, optlog, colormap(i, n))

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
                                   "special": ["modechange", "stepchange"]
                                       # mode and step change with respect to the previous
                                       # step, generally the special parameters are only
                                       # used to visualize versus steps
                              }

    def __call__(self, opt_string, modes, geos, grs, ifplot):
        """
        call the class to give back required directions so that the angles can be calculated
        in a uniform way
        """
        if opt_string.startswith("s"):
            # for special type, the two directions are specified by only one keyword
            assert len(opt_string) == 2
            if opt_string[1] == "0":
                # mode change, which should always be an acute angle, because mode has no
                # orientation
                directs = [modes[ifplot[0] - 1: ifplot[1] - 1], modes[ifplot[0]: ifplot[1]]]
                acute = True
            if opt_string[1] == "1":
                # step change
                directs = [step_from_beads(geos)[ifplot[0] - 1: ifplot[1] - 1], \
                            step_from_beads(geos)[ifplot[0]: ifplot[1]]]
                acute = False

        else:
            # there should be 4 characters to specify 2 directions,
            # and none of them could be in the category of special
            assert len(opt_string) == 4 and opt_string[2] != "s"
            directs = []
            acute = False
            for i in [0, 2]:
                # the former and later 2 characters in opt_string
                # represents a variable, respectively
                opt = opt_string[i:i+2]
                if opt[0] == "v":
                    if opt[1] == "0":
                        # mode has no directions
                        directs.append(modes[ifplot[0]: ifplot[1]])
                        acute = True
                    elif opt[1] == "1":
                        # gradients
                        directs.append(grs[ifplot[0]: ifplot[1]])
                    elif opt[1] == "2":
                        # translation steps
                        directs.append(step_from_beads(geos[ifplot[0]: ifplot[1]+1]))
                    elif opt[1] == "3":
                        # accumulated translations
                        directs.append([geo - geos[0] for geo in geos[ifplot[0]: ifplot[1]]])
                elif opt[0] == "c":
                    if opt[1] == "0":
                        # initial to final structure
                        directs.append([geos[-1] - geos[0]] * (ifplot[1] - ifplot[0]))
                    if opt[1] == "1":
                        # initial mode
                        directs.append([modes[0]] * (ifplot[1] - ifplot[0]))
                    if opt[1] == "2":
                        # final mode
                        directs.append([modes[-1]] * (ifplot[1] - ifplot[0]))
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
        range_1 = set([])
        if self.name in ["mode", "step"]:
            range_1.add(-1)
        elif self.name in self.direction_type["special"]:
            range_1.add(1)
            range_1.add(-1)
        return range_1

def read_from_pickle_log(filename, additional_points = set([])):
    """
    read the geometry of given points as well as modes, curvaturese
    from log.pickle file.
    always extract center curvature and lowest mode, addtional 
    points can be specified
    """
    from pickle import load

    geos = {"Center": []}
    for name in additional_points:
        geos[name] = []
    modes = []
    curvatures = []

    logfile = open(filename, "r")

    # first item is the atoms object
    atoms = load(logfile)
    # second item is the transformation function to Cartesian
    funcart = load(logfile)

    # read in data until reach the end of the file
    while True:
        try:
            item = load(logfile)
        except EOFError:
            break
        if item[0] == "Lowest_Mode":
            # normalize when read in
            modes.append(item[1] / norm(item[1]))
        if item[0] == "Curvature":
            curvatures.append(item[1])
        elif item[0] in additional_points:
            geos[item[0]][-1].append(item[1])
        elif item[0] == "Center":
            for name in additional_points:
                geos[name].append([])
            geos["Center"].append(item[1])

    logfile.close()

    return atoms, funcart, geos, modes, curvatures

def read_from_store(store, geos):
    """
    read in the energies and gradients for given structures
    """
    from pts.memoize import Memoize

    # energy only available at center points, so return a list of energy
    # and a dictionary of gradients
    grad = {}

    # None is a fake Func that cannot compute anything,
    # only stored values can be accessed without failure:
    pes = Memoize(None, store)
    energy = [pes(x) for x in geos["Center"]]
    grad["Center"] = [pes.fprime(x) for x in geos["Center"]]
    for key in geos:
        if key != "Center":
            grad[key] = [[pes.fprime(x) for x in iteration] for iteration in geos[key]]

    # Attention: here the returned gradients are in the same coordinates as
    # geometry
    return energy, grad

def grads_dimer(mode, gr, allval):
    """
    Gives back the values of the gradients as decided in allval
    which appear on the beads, path informations are needed
    for finding the mode along the path
    """
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
