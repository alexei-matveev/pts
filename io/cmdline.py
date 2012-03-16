#!/usr/bin/env python
"""
Shared code for parsing command line here
"""

import getopt

# for get calculator
from ase.calculators import *
from pts.gaussian import Gaussian
from pts.common import file2str
from pts.defaults import ps_default_params, default_calcs, default_lj, default_vasp

LONG_OPTIONS = ["calculator="]

def get_options(argv, options="", long_options=LONG_OPTIONS):

    opts, args = getopt.getopt(argv, options, long_options)

    return opts, args

def get_defaults():
    """
    Returns a copy of the parameter dictionary with default settings
    """
    return ps_default_params.copy()

def get_calculator(file_name):

    calculator = None
    if file_name in default_calcs:
        calculator = eval("%s" % (file_name))
    else:
        str1 = file2str(file_name) # file file_name has to
        # contain line calculator = ...
        exec(str1)

    return calculator

def get_mask(strmask):
    tr = ["True", "T", "t", "true"]
    fl = ["False", "F", "f", "false"]
    mask = strmask.split()
    # test that all values are valid:
    true_or_false = tr + fl
    for element_of_mask in mask:
        assert( element_of_mask in true_or_false)
    # Transform mask in logicals ([bool(m)] would
    # be only false for m = ""
    mask = [m in tr for m in mask]
    return mask


def get_options_to_xyz(argv, num_old):
    """
    Extracts the options for the path/progress to xyz tools.
    """
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-a", "--abscissa", "--pathpos", dest = "abcis",
                      type = "string", action = "append", default = [])
    parser.add_option("--mask", dest = "mask",
                      type = "string", nargs = 2)
    parser.add_option("--zmatrix", dest = "zmats",
                      type = "string", default = None)
    parser.add_option("-s", "--symbols", dest = "symbfile",
                      type = "string")
    parser.add_option("-n", "--number-of-images", dest = "num",
                      help = "Number of images on path or table",
                      default = num_old, type = "int")

    parser.add_option("-b", "--beads", dest = "beads",
                      action = "store_true", default = False )

    parser.add_option("-m", "--modes", "--add-modes", "--with-modes", dest = "add_modes",
                      action = "store_true", default = False )

    return parser.parse_args(argv)

def visualize_input( argv, num_old):
    """
    Reads in the input from argv
    """
    from pts.tools.xyz2tabint import interestingvalue
    from optparse import OptionParser
    parser = OptionParser()
    global num_i
    num_i = 1

    def value_to_number(name):
        print "Transform name", name
        if name.startswith("dis"):
            return 2
        elif name.startswith("ang"):
            if "4" in name:
                return 4
            else:
                return 3
        elif name.startswith("dih"):
            return 5
        elif "d" in name and "p" in name:
            return 6
        elif "d" in name and "l" in name:
            return 7
        elif "o" in name and "p" in name:
            return 8
        else:
            return 0



    def got_intersting_value(option, opt_str, value, parser):
         global num_i
         print "processing", option, opt_str, value
         number = value_to_number(opt_str[2:])

         if parser.values.allval == None:
             parser.values.allval = []

         parser.values.allval.append((number, value))
         # count up, to know how many and more important for
         # let diff easily know what is the next
         if number == 8:
               num_i += 2
         else:
               num_i += 1

    def call_difference(option, opt_str, value, parser):
        global num_i
        parser.values.diff.append((num_i ))


    def call_symmetrie(option, opt_str, value, parser):
        global num_i
        parser.values.symm.append((num_i, value ))

    # the number of images (points on a path)
    parser.add_option("--num", dest = "num",
                      help = "Number of images on path or table",
                      default = num_old, type = "int")

   # The geometry options to plot or print
    parser.add_option("--t" ,"--s", dest = "withs",
                      help = "Use abscissa/number of position as x-value",
                      action = "store_true", default = False )

    parser.add_option("--difference", dest = "diff",
                      help = "From the next two coordinates the difference is taken",
                      default = [], action = "callback", callback = call_difference )

    parser.add_option("--symmetry", dest = "symm",
                      default = [], action = "callback", type = "float", callback = call_symmetrie )

    parser.add_option("--distance", dest = "allval", default = [],
                     action = "callback", type = "int", nargs = 2, callback = got_intersting_value)

    parser.add_option("--angle", "--ang", dest = "allval",
                     action = "callback", type = "int", nargs = 3, callback = got_intersting_value)

    parser.add_option("--angle4","--ang4", dest = "allval",
                     action = "callback", type = "int", nargs = 4, callback = got_intersting_value)

    parser.add_option("--dihedral", dest = "allval",
                     action = "callback", type = "int", nargs = 4, callback = got_intersting_value)

    parser.add_option("--dp", "--plane-distance", dest = "allval",
                     action = "callback", type = "int", nargs = 4, callback = got_intersting_value)

    parser.add_option("--dl", "--line-distance", dest = "allval",
                     action = "callback", type = "int", nargs = 3, callback = got_intersting_value)

    parser.add_option("--op", "--on-plane", dest = "allval",
                     action = "callback", type = "int", nargs = 4, callback = got_intersting_value)

    parser.add_option("--expand", dest = "expand",
                      type = "string", nargs = 2)

   # different input format for path
    parser.add_option("--ase", dest = "ase",
                      help = "Input comes in ASE format",
                      action = "store_true", default = False )
    parser.add_option("--format", dest = "format",
                      help = "Sets the format for ASE input to FORMAT", metavar = "FORMAT",
                      type = "string")
    parser.add_option("--next", dest = "next", default = [0],
                      type = "int", action = "append")

    # Energies and gradient handling
    parser.add_option("--en", "--energy" , dest = "special_vals",
                      help = "Use the energy, do not use gradients",
                      action = "append_const", const = "energy", default = [] )

    parser.add_option("--energy2" , dest = "special_vals",
                      help = "Use the energy, interpolation with gradients",
                      action = "append_const", const = "energy2", default = [] )

    parser.add_option("--gr"  "--gradients", dest = "special_vals",
                      help = "Use ACTION of gradients, ACTIONS can be abs, max, para, perp, angle",
                      metavar = "ACTION",
                      action = "append", type = "string", default = [] )

    parser.add_option("--step", dest = "special_vals",
                      action = "append_const", const = "step" )

    parser.add_option("--arrow", "--with-mode", dest = "arrow_len",
                       type = "float" )

    parser.add_option("--vector-angle", "--vec_angle", dest = "vec_angle",
                       type = "string", nargs = 2, action = "append", default = [] )

    # Plotting settings
    parser.add_option("--title", dest = "title",
                      type = "string")
    parser.add_option("--xlabel", dest = "xlabel",
                      type = "string")
    parser.add_option("--xrange", dest = "xrange",
                      type = "float", nargs = 2 )
    parser.add_option("--yrange", dest = "yrange",
                      type = "float", nargs = 2 )
    parser.add_option("--ylabel", dest = "ylabel",
                      type = "string")
    parser.add_option("--output", dest = "output",
                      type = "string")
    parser.add_option("--name", dest = "names_of_lines",
                      type = "string", action = "append", default = [])
    parser.add_option("--logscale", dest = "logscale",
                      type = "string", action = "append", default = [])

    # Different input format
    parser.add_option("--abscissa","--pathpos", dest = "abcis",
                      type = "string", action = "append", default = [])
    parser.add_option("--mask", dest = "mask",
                      type = "string", nargs = 2)
    parser.add_option("--zmatrix", dest = "zmats",
                      type = "string", action = "append", default = [])
    parser.add_option("--symbols", dest = "symbfile",
                      type = "string")

    (options, args) = parser.parse_args(argv)
    return options, args, num_i




# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
