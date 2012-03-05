#!/usr/bin/env python
"""
This tool helps to visualize the key variables in transition
state refinement calculations, namely, Dimer, Lanczos and 
quasi-Newton calculations.

Usage:

    paratools dimer2plot/lanczos2plot/qn2plot --<kind> n1 n2 ... log_files dict_files/cache directories

"""
import sys
import numpy as np
from pts.path import Path

def main(argv):

    from pts.tools.xyz2tabint import interestingvalue
    from pts.tools.tab2plot import plot_tabs
    from pts.tools.path2plot import makeoption
    from pts.tools.path2tab import get_expansion
    from pts.memoize import FileStore, DirStore, Store
    from pts.cfunc import Justcarts
    from pts.tools.path2tab import path_to_int, beads_to_int
    from pts.tools.path2tab import energy_from_path

    if len(argv) <= 1:
        # errors go to STDERR:
        print >> sys.stderr, __doc__
        exit()
    elif argv[0] == '--help':
        # normal (requested) output goes to STDOUT:
        print __doc__
        exit()

    #initialize
    #calculation type
    name = argv[0][:-5]      #can be "dimer", "lanczos" or "qn"
    argv = argv[1:]

    #input files
    log_file = []
    dict_file = []
    dict_dir = []
    xlog = 0
    xdict = 0

    #default values of interesting parameters
    diff = []
    arrow = False
    special_val = []
    allval = []
    cell = None
    tomove = None
    howmove = None
    withs = False
    info = set([])
    iter_flag = set([])    # because some of the parameters to be plotted are
                           # available only between iterations, eg. trans. step
                           # size, or only in iterations before converge, eg.
                           # mode vector, this flag is used to consider such situation

    # default for plotting options
    num = 100
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

    #
    # Read all the arguments in.  Stop  cycle if all input is read in.
    # Some iterations consume more than one item.
    #
    while len(argv) > 0:
        if argv[0].startswith("--"):
            #distinguish options from files
            option = argv[0][2:]
            if option == "num":
                num = int(argv[1])
                argv = argv[2:]
            elif option in ["s", "t"]:
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
                    if argv[0].endswith("after") or argv[0] == "mode":
                        # use only first n-1 iterations
                        iter_flag.add(-1)
                    elif argv[0].endswith("before"):
                        # use only last n-1 iterations
                        iter_flag.add(1)
                    elif argv[0] == "stepchange":
                        # use only n-2 iterations in the middle
                        iter_flag.update([1,-1])
                    value += interestingdirection(argv[0]).string()
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
                assert name in ["dimer", "lanczos"]
                iter_flag.add(-1)
                special_val.append("curvature")
                info.add(name.capitalize())
                argv = argv[1:]
            elif option in ["step_before"]:
                # step size of translation step before interesting variables
                # of specific iteration
                iter_flag.add(1)
                special_val.append(option)
                argv = argv[1:]
            elif option in ["step_after"]:
                # step size of translation step after interesting variables
                # of specific iteration
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
                assert name in ["dimer", "lanczos"]
                iter_flag.add(-1)
                arrow = True
                info.add(name.capitalize())
                argv = argv[1:]
            # output as file or on screen
            elif option == "output":
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
            log_file.append(argv[0])
            xlog += 1
            argv = argv[1:]
        elif "Result" in argv[0] or "Dict" in argv[0]:
            dict_file.append(argv[0])
            xdict += 1
            argv = argv[1:]
        elif "cache" in argv[0] or ".d/" in argv[0]:
            dict_file.append(argv[0])
            dict_dir.append(xdict)
            xdict +=1
            argv = argv[1:]

    # if no pickle file is specified, use the default
    if xlog == 0 or xdict == 0:
        # two pickle files are always used together
        log_file = ["%s.log.pickle" % (name)]
        dict_file = ["%s.ResultDict.pickle" % (name)]

    # input file validation:
    # FIXME: may not be needed, because for some plot, no ResultDict
    # file is needed.
    if not xlog == xdict:
        print "Number of log file and dictionary file are unequal!"
        print "Please check your input"
        print __doc__
        exit

    # plot environment
    pl = plot_tabs(title = title, x_label = xlab, y_label = ylab, log = logscale)

    # extract which options to take
    opt, num_opts, xnum_opts, optx = makeoption(num_i, diff, [], [], withs)

    for i in range(len(log_file)):
        # ensure that there will be no error message if calling
        # names_of_lines[i]
        names_of_lines.append([])

    for i, geo_file in enumerate(log_file):
        atoms, funcart, geos, modes = read_from_pickle_log(geo_file, info)
        obj = atoms.get_chemical_symbols(), funcart
        ifplot = [None, None]   # initial and final point to plot
                                # used because arrays to be plotted may have
                                # different length
        if 1 in iter_flag:
            ifplot[0] = 1
        if -1 in iter_flag:
            ifplot[1] = len(geos["Center"])-1
        x = list(np.linspace(0,1,len(geos["Center"])))[ifplot[0]: ifplot[1]]

        beads = beads_to_int(geos["Center"][ifplot[0]: ifplot[1]], x, obj, \
                        allval, cell, tomove, howmove, withs)
        beads = np.asarray(beads)
        beads = beads.T

        name_p = str(i + 1)
        if names_of_lines[i] != []:
             name_p = names_of_lines[i]
        if num_opts > 1:
            pl.prepare_plot( None, None, None, "_nolegend_", beads, name_p, opt)

        if special_val != []:
            # Two kinds of store share the same methods interesting to us
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
                    log_points.append(energy)
                elif s_val.startswith("gr"):
                    val = s_val[2:]
                    log_points.append(grads_from_beads_dimer(modes, \
                            grad["Center"][ifplot[0]: ifplot[1]], val))
                elif s_val.startswith("curvature"):
                    log_points.append(curv_from_beads(modes, geos["Center"], \
                            geos[name.capitalize()], grad["Center"], \
                            grad[name.capitalize()], name, ifplot))
                else:
                    print "Sorry, the function is still not available now!"

                log_points = np.asarray(log_points)

                pl.prepare_plot( None, None, None, "_nolegend_", log_points, \
                               s_val + " %i" % (i + 1), optlog)

    pl.plot_data(xrange = xran, yrange = yran, savefile = outputfile )

class interestingdirection:
    def __init__(self, option):
        self.variable = ["mode", "grad", "step_before", "step_after", "translation"]
            # translation means accumulated translation from initial point 
        self.constant = ["init2final", "initmode", "finalmode"]
        # generally the special parameters are only intended to visualize versus steps
        self.special = ["modechange_before", "modechange_after", "stepchange"]
        self.name = option
        if option in self.variable:
            self.type = "variable"
        elif option in self.constant:
            self.type = "constant"
        elif option in self.special:
            self.type = "special"

    def string(self):
        index = eval("self."+self.type+".index(self.name)")
        return self.type[0]+str(index)

def read_from_pickle_log(filename, additional_points = set([])):
    # read given points from pickle_log file.
    # FIXME: if there are several functions considering the pickle_log
    # file, may consider putting them in a single file
    # always extract center and lowest mode, addtional points can be
    # specified
    from pickle import load

    geos = {"Center": []}
    for name in additional_points:
        geos[name] = []
    modes = []

    logfile = open(filename, "r")

    #First item is the atoms object
    atoms = load(logfile)
    #second item is the transform function to Cartesian
    funcart = load(logfile)

    while True:
        try:
            item = load(logfile)
        except EOFError:
            break
        if item[0] == "Lowest_Mode":
            # normalize when read in
            modes.append(item[1] / np.sqrt(np.dot(item[1], item[1])))
        elif item[0] in additional_points:
            geos[item[0]][-1].append(item[1])
        elif item[0] == "Center":
            for name in additional_points:
                geos[name].append([])
            geos["Center"].append(item[1])

    logfile.close()

    return atoms, funcart, geos, modes

def read_from_store(store, geos):

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
    # geometry, needing revision before visualization 
    return energy, grad

def grads_from_beads_dimer(mode, gr, allval):
    """
    Gives back the values of the gradients as decided in allval
    which appear on the beads, path informations are needed
    for finding the mode along the path
    """
    grs = []

    for i, gr_1 in enumerate(gr):
        if allval == "abs":
            #Total forces, absolute value
            grs.append(np.sqrt(np.dot(gr_1, gr_1)))
        elif allval == "max":
            #Total forces, maximal value
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
            grs.append(np.sqrt(np.dot(gr_1, gr_1)))
        elif allval == "angle":
            # angle between forces and path
            mode_1 = mode[i]
            gr_1 = gr_1 / np.sqrt( np.dot(gr_1, gr_1))
            ang = abs(np.dot(mode_1, gr_1))
            if ang > 1.:
                ang = 1.
            grs.append(np.arccos(ang) * 180. / np.pi)
        else:
            print >> stderr, "Illegal operation for gradients", allval
            exit()
    return grs

def curv_from_beads(modes, geo_center, geo_dimer, gr_center, gr_dimer, name, ifplot = [None, None]):

    """
    Gives back the curvatures for each beads in the
    refinement. For dimer methods use the gradient
    of the last rotation step, while for lanczos it
    is the combination of gradients of intermediate
    steps.
    """
    curv = []
    # calculate dimer distance
    dis = np.sqrt(np.dot(geo_center[0]-geo_dimer[0][0], geo_center[0]-geo_dimer[0][0]))
    # select slices which will be used to calculate curvature
    modes = modes[ifplot[0]: ifplot[1]]
    geo_center = geo_center[ifplot[0]: ifplot[1]]
    geo_dimer = geo_dimer[ifplot[0]: ifplot[1]]
    gr_center = gr_center[ifplot[0]: ifplot[1]]
    gr_dimer = gr_dimer[ifplot[0]: ifplot[1]]
    from scipy.linalg import eigh

    for geo, geo_d, gr, gr_d, mode in zip(geo_center, geo_dimer, gr_center, gr_dimer, modes):
        assert abs(np.dot(mode, mode) - 1) < 1.0e-7
        if name == "dimer":
            # for dimer calculations the last dimer point is
            # in the mode direction 
            curv.append(np.dot(gr_d[-1] - gr, mode) / dis)
        elif name == "lanczos":
            # for lanczos calcualtions the gradient needed in
            # curvature calculation is approximated with the results of other
            # points around the central point
            g_approx = sum([(gr_d[i] -gr) * np.dot(geo_d[i] - geo, mode) / dis for i in range(len(geo_d))])
            curv.append(np.dot(g_approx, mode) / dis)
    return curv
