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
    name = argv[0]      #can be "dimer", "lanczos" or "qn"
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
                    value += interestingdirection(argv[0]).string()
                    argv = argv[1:]
                    if value[3] == "s":
                        break
                print value
                special_val.append(value)
            elif option in ["grabs", "gramax", "graperp", "grapara"]:
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
                special_val.append("curvature")
                info.add(name.capitalize())
                argv = argv[1:]
            elif option in ["step_before", "step_after"]:
                # step size of translation step before/after interesting variables
                # of specific iteration
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
        x = list(np.linspace(0,1,len(geos["Center"])))

        beads = beads_to_int(geos["Center"], x, obj, allval, cell, tomove, howmove, withs)
        beads = np.asarray(beads)
        beads = beads.T

        path = path_to_int(x, geos["Center"], obj, num, allval, cell, tomove, howmove, withs)
        path = np.asarray(path)
        path = path.T

        name_p = str(i + 1)
        if names_of_lines[i] != []:
             name_p = names_of_lines[i]
        if num_opts > 1:
            pl.prepare_plot( path, name_p, beads, "_nolegend_", None, None, opt)

        if special_val != []:
            # Two kinds of store share the same methods interesting to us
            if i in dict_dir:
                resdict = DirStore(dict_file[i])
            else:
                resdict = FileStore(dict_file[i])
            energy, grad = read_from_store(resdict, geos)
            geos, modes = cart_convert(geos, modes, funcart)

            for s_val in special_val:
                # use the options for x and plot the data gotten from
                # the file directly

                optlog = optx + " t %i" % (xnum_opts + 1)
                log_points = beads
                log_points = log_points[:xnum_opts + 1,:]
                log_points = log_points.tolist()

                if s_val.startswith("en"):
                    log_points.append(energy)
                else:
                    print "ERROR: function still not availabe"
                    exit

                log_path = path
                log_path = log_path[:xnum_opts + 1,:]
                log_path = log_path.tolist()

                if s_val.startswith("en"):
                    log_path.append(energy_from_path(x, energy, num))
                log_points = np.asarray(log_points)
                pl.prepare_plot( log_path, s_val + " %i" % (i + 1), \
                               log_points, "_nolegend_", None, None, optlog)

    pl.plot_data(xrange = xran, yrange = yran, savefile = outputfile )

class interestingdirection:
    def __init__(self, option):
        self.variable = ["mode", "grad", "step_before", "step_after", "translation"]
        self.constant = ["init2final", "initmode", "finalmode"]
        self.special = ["modechg_before", "modechg_after", "stepchg_before", "stepchg_after"]
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
            modes.append(item[1])
        elif item[0] in additional_points:
            geos[item[0]][-1].append(item[1])
        elif item[0] == "Center":
            for name in additional_points:
                geos[name].append([])
            geos["Center"].append(item[1])

    logfile.close()

    return atoms, funcart, geos, modes

def cart_convert(geos, modes, funcart):

    new_geos = {}
    new_modes = []

    for key in geos:
        new_geos[key] = []
        for item in geos[key]:
            new_geos[key].append(funcart(item))

    for i in range(len(modes)):
        new_modes.append(np.dot(funcart.fprime(geos["Center"][i]), modes[i]))
        new_modes[i] = new_modes[i] / np.sqrt(np.dot(new_modes[i].flatten(), new_modes[i].flatten()))
    new_modes.append(np.array([]))

    return new_geos, new_modes

def read_from_store(store, geos):
    # energy only available at center points, so return a list of energy
    # and a dictionary of gradients
    energy = []
    grad = {}
    for key in geos:
        if key == "Center":
            grad["Center"] = []
            for geo in geos["Center"]:
                energy.append(store[((tuple(geo),),0)])
                grad["Center"].append(store[((tuple(geo),),1)])
        else:
            grad[key] = []
            for iter in geos[key]:
                grad[key].append([])
                for geo in iter:
                    grad[key][-1].append(store[((tuple(geo),),1)])

    return energy, grad

