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

def alternative_input_group(parser):
    """
    Here FILE contains  direct  internal geometry coordinates.
    There is the requirement of having more input in
    separate user readable files.
    Required are in this case the files of the --symbols option. Additional
    the transformation from internal to Cartesian coordinates needs to
    be build up again, therefore if some internal coordinates were included
    the zmatrices (--zmatrix) have to be provided again. If some of
    the coordinates were fixed they have to be given also again by
    the --mask switch (needs also a complete set of geometries).
    If the path had some abscissas (all string methods) they should also
    be provided again with the --abscissa option.
    """
    from optparse import OptionGroup

    group =  OptionGroup( parser, "ALTERNATIVE WAY FOR INPUT", alternative_input_group.__doc__)

    group.add_option( "--abscissa", "--pathpos", dest = "abcis",
                      help = "Abscissa file AB-FILE. Gives the abscissas to the positions in FILE.", metavar = "AB-FILE",
                      type = "string", action = "append", default = [])
    group.add_option("--mask", dest = "mask",
                      help = "mask file MASK. If some coordinates had been fixed in the internal coordinates, the mask telling which one has to be given again. Here it is also required to give as a second argument MASKGEO which contains all internal coordinates for the system without the mask.", metavar = "MASK MASKGEO",
                      type = "string", nargs = 2)
    group.add_option("--zmatrix", dest = "zmats",
                      help = "one zmatrix file ZMAT, if there are ZMatrix coordinates in the internal coordinates. Should be given the same way as for the original calculation.", metavar = "ZMAT",
                      type = "string", default = [])
    group.add_option( "--symbols", dest = "symbfile",
                      help = "Symbols file SYM. Should contain as a list all the symbols of the atoms separated by whitespace.", metavar = "SYM",
                      type = "string")

    parser.add_option_group(group)
    return parser

def get_options_to_xyz(input, argv, num_old):
    """

    paratools xyz INPUT FILE  [Options]

    Where INPUT is one of path or progress.

    FILE contains the information of the input.
    For path it can be a path.pickle file or alternative input, see
    Options below.

    For progress it needs to be a progress.pickle file.

    The tool creates a string of xyz geometries to the geometries extracted from
    input. Additional options can change the output.

    """
    from optparse import OptionParser, OptionGroup

    parser = OptionParser(usage = get_options_to_xyz.__doc__ )


    if input == "path":
       parser = alternative_input_group(parser)

    group = OptionGroup( parser, "PATH OPTIONS" )
    group.add_option( "--number-of-images", dest = "num",
                      help = "Number N of images on path.", metavar = "N",
                      default = num_old, type = "int")

    group.add_option( "--beads", dest = "beads",
                      help = "Use the exact bead positions (no respace).",
                      action = "store_true", default = False )

    if input == "path":
        parser.add_option_group(group)

    group = OptionGroup( parser, "PROGRESS OPTIONS" )
    group.add_option( "--modes", dest = "add_modes",
                      help = "Append also the mode vector (transformed to Cartesian) after the geometries.",
                      action = "store_true", default = False )

    if input == "progress":
        parser.add_option_group(group)

    return parser.parse_args(argv)

def visualize_input( argv, num_old):
    """
    Reads in the input from argv for the plot/show/table
    function.
    Looks for the sum of all available options, the
    calling functions might only need a subset of them.
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
         """
         Callback function for the geometry values, transforms
         the name in the usual number short hands.
         """
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
        """
        Callback for the collecting of difference information.
        Needs a global variable, thus is done by a callback.
        """
        global num_i
        parser.values.diff.append((num_i ))


    def call_symmetrie(option, opt_str, value, parser):
        """
        Callback for the collecting of symmetry information.
        Needs a global variable, thus is done by a callback.
        """
        global num_i
        parser.values.symm.append((num_i, value ))

    def grad_plus_action(option, opt_str, value, parser):
        """
        Gradient action needs an identifier for gradient before
        """
        parser.values.special_vals.append(("gr-" + value))

    def grad_action(option, opt_str, value, parser):
        """
        Gradient action needs an identifier for gradient before.
        """
        # start for option at 4: 2 for -- and 2 for gr
        parser.values.special_vals.append(("gr-" + opt_str[4:]))

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

    parser.add_option("--gr", "--gradients", dest = "special_vals",
                      help = "Use ACTION of gradients, ACTIONS can be abs, max, para, perp, angle",
                      metavar = "ACTION",
                      action = "callback", type = "string", callback = grad_plus_action )

    parser.add_option("--grperp", "--grpara", "--grabs", "--grangle", dest = "special_vals",
                      action = "callback", callback = grad_action )

    parser.add_option("--step", dest = "special_vals",
                      action = "append_const", const = "step" )

    parser.add_option("--curvature", dest = "special_vals",
                      action = "append_const", const = "curv" )

    parser.add_option("--ts-estimates", "--transition-states", dest = "ts_estimates",
                       type = "int", action = "append", default = [] )

    parser.add_option("--references", dest = "references",
                       type = "string", nargs = 2, action = "append", default = [] )

    # For dimer
    parser.add_option("--arrow", "--with-mode", "--modes", dest = "arrow_len",
                       type = "float")

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
