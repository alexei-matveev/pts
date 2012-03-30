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

def ase_input(parser):
    """
    The FILE(s) will be expected to contain geoemtries in ASE
    readable format. If the format cannot be extracted from the name
    of FILE it can be given as --format. All geometries will be supposed
    to be situated on a single path as default. if --next is used a new
    path will be started.
    """
    from optparse import OptionGroup

    group =  OptionGroup( parser, "SECOND ALTERNATIVE INPUT: ASE INPUT", ase_input.__doc__)
    group.add_option("--ase", dest = "ase",
                      help = "Input is in ASE format.",
                      action = "store_true", default = False )
    group.add_option("--format", dest = "format",
                      help = "Sets the format for ASE input to FORMAT. Sets also ase to true.", metavar = "FORMAT",
                      type = "string")
    group.add_option("--next", dest = "next", default = [0],
                      help = "The input files FILES starting with N will be the next path.", metavar = "N",
                      type = "int", action = "append")

    parser.add_option_group(group)
    return parser

def alternative_input_group(parser):
    """
    Here FILE(s) contains  direct  internal geometry coordinates.
    There is the requirement of having more input in separate user
    readable files. Required are in this case the files of the
    --symbols option. Additional the transformation from internal to
    Cartesian coordinates needs to be build up again, therefore if some
    internal coordinates were included the zmatrices (--zmatrix) have to
    be provided again. If some of the coordinates were fixed they have to
    be given also again by the --mask switch (needs also a complete set
    of geometries). If the path had some abscissas (all string methods)
    they should also be provided again with the --abscissa option.
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

def geometry_values(parser):
    """
    Usually geometry values are given before the other values, therefore if at least one of them is given the x-coordinate will be an internal coordinate. They can be choosen independent of the coordinates in which the calculation has been run.
    """
    from optparse import OptionGroup

    group =  OptionGroup( parser, "SPECIAL INTERNAL COORDINATES EXCTRACTING", geometry_values.__doc__)

    def value_to_number(name):
        """
        The internal coordinates are usually handled as numbers.
        Transform them here from the user friendly names.
        """
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


   # The geometry options to plot or print
    group.add_option("--t" ,"--s", dest = "withs",
                      help = "The first geometry will be the abscissa of the path or the number of the point (progress).",
                      action = "store_true", default = False )

    group.add_option("--difference", dest = "diff",
                      help = "From the next two coordinates the difference is taken.",
                      default = [], action = "callback", callback = call_difference )

    group.add_option("--symmetry", dest = "symm",
                      help = "For the next variable F it is given how good is the symmetry as F(x -XMID).",
                      metavar = "XMID",
                      default = [], action = "callback", type = "float", callback = call_symmetrie )

    group.add_option("--distance", dest = "allval", default = [],
                      help = "Distance between atoms N1 and N2.",
                      metavar = "N1 N2",
                     action = "callback", type = "int", nargs = 2, callback = got_intersting_value)

    group.add_option("--angle", "--ang", dest = "allval",
                      help = "Angle between the atoms N1 N2 N3.",
                      metavar = "N1 N2 N3",
                     action = "callback", type = "int", nargs = 3, callback = got_intersting_value)

    group.add_option("--angle4","--ang4", dest = "allval",
                      help = "Angle between the vectors of atoms N1-N2 and N3-N4.",
                      metavar = "N1 N2 N3 N4",
                     action = "callback", type = "int", nargs = 4, callback = got_intersting_value)

    group.add_option("--dihedral", dest = "allval",
                      help = "Dihedral angle of N1 N2 N3 and N4.",
                      metavar = "N1 N2 N3 N4",
                     action = "callback", type = "int", nargs = 4, callback = got_intersting_value)

    group.add_option("--dp", "--plane-distance", dest = "allval",
                      help = "Distance of X1 to the plane spanned by N1 N2 and N3.",
                      metavar = "X1 N1 N2 N3",
                     action = "callback", type = "int", nargs = 4, callback = got_intersting_value)

    group.add_option("--dl", "--line-distance", dest = "allval",
                      help = "Distance of X1 to the line given by N1 and N2.",
                      metavar = "X1 N1 N2",
                     action = "callback", type = "int", nargs = 3, callback = got_intersting_value)

    group.add_option("--op", "--on-plane", dest = "allval",
                      help = "Position of projection of X1 onto the coordinat system given by O= N1, x=N2-N1 and y=N3-N1.",
                      metavar = "X1 N1 N2 N3",
                     action = "callback", type = "int", nargs = 4, callback = got_intersting_value)

    group.add_option("--expand", dest = "expand",
                      help = """The atoms will be exanded with atoms choosen
                      as original ones shifted with cell vectors, as
                      described in EXP-FILE. CELL-FILE should  contain
                      the three basis vectors for the cell. EXP-FILE
                      contains the shifted atoms:
                      Number(original atom) three numbers
                      for shift in the three cell vector directions.""",
                      metavar = "CELL-FILE EXP-FILE",
                      type = "string", nargs = 2)

    parser.add_option_group(group)
    return parser

def gradient_and_energies(parser):
    """
    Gives energies or special gradient properties. They will be processed always after the internal coordinates, no matter in which order they are given. This way a later specified internal coordinate will still be the x-coordinate.
    """
    from optparse import OptionGroup

    group =  OptionGroup( parser, "ENERGIES AND GRADIENT PROPERTIES", gradient_and_energies.__doc__)
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

    group.add_option("--en", "--energy" , dest = "special_vals",
                      help = "Gives the energy. Interpolation for path without usage of gradients.",
                      action = "append_const", const = "energy", default = [] )

    group.add_option("--energy2" , dest = "special_vals",
                      help = "Gives the energy. Interpolation for path with usage of gradients.",
                      action = "append_const", const = "energy2", default = [] )

    group.add_option("--gr", "--gradients", dest = "special_vals",
                      help = "Use ACTION of gradients, ACTIONS can be abs, max, para, perp, angle.",
                      metavar = "ACTION", choices = ["abs", "max", "para", "perp", "angle"],
                      action = "callback", type = "choice", callback = grad_plus_action )

    group.add_option("--grperp", "--grpara", "--grabs", "--grangle", dest = "special_vals",
                      help = "Allow gradient actions the old way.",
                      action = "callback", callback = grad_action )
    parser.add_option_group(group)
    return parser

def dimer_specials(parser):
    """
    Gives special properties only available for the progress functions.
    """
    from optparse import OptionGroup

    group =  OptionGroup( parser, "SPECIAL PROPERTIES FOR PROGRESS FUNCTIONS", dimer_specials.__doc__)

    group.add_option("--step", dest = "special_vals",
                      help = "Gives the steplength between succeding geometries.",
                      action = "append_const", const = "step" )

    group.add_option("--curvature", dest = "special_vals",
                      help = "Gives the curvature (only dimer or lanczos).",
                      action = "append_const", const = "curv" )

    group.add_option("--modes", dest = "arrow_len",
                     help = """Plots additional to the geometries (only for internal
                               coordinates) the mode vector on each geometry with
                               the length LENGTH.""",
                     metavar = "LENGTH",
                       type = "float")

    group.add_option("--vector-angle", "--vec-angle", dest = "vec_angle",
                       metavar = "DIR1 DIR2",
                       help = """Gives the angle between the two directions DIR1
                               and DIR2. If DIR1 and DIR2 are the same the angle
                               between succeding iterations is taken.
                               The DIR's can be anything of:
                               mode, grad (gradient), step,
                               initmode: initial mode vector
                               finalmode: mode of the last step
                               init2final: vector from initial to final point
                               translation: accumulated translation from initial point
                               """,
                       type = "string", nargs = 2, action = "append", default = [] )
    parser.add_option_group(group)
    return parser

def path_specials(parser, num_old):
    """
    Defines coarseness of path and special points to add.
    """
    from optparse import OptionGroup

    group =  OptionGroup( parser, "SPECIAL PROPERTIES FOR PATH FUNCTIONS", path_specials.__doc__)

    group.add_option("--num", dest = "num",
                      help = "Number of images on path or table.",
                      default = num_old, type = "int")

    group.add_option("--ts-estimates", "--transition-states", dest = "ts_estimates",
                      metavar = "N",
                      help = "Adds the transition state of the kind N (see help text of paratools ts-and-mods) to the paths.",
                       type = "int", action = "append", default = [] )

    group.add_option("--references", dest = "references",
                      metavar = "GEO EN",
                      help = "Only for plot/show: adds a reference point of geometry of file GEO and energy of file EN to the plot.",
                       type = "string", nargs = 2, action = "append", default = [] )

    parser.add_option_group(group)
    return parser

def visualize_input( input, output, argv, num_old):
    """

    paratools OUTPUT INPUT FILE [FILE2, ...] Options

    Where OUTPUT is one of:
    table : gives a table with the numbers to standard output.
    show  : open a picture containing the plot of the choosen values.
    plot  : puts the picture directly ito a file.

    INPUT can be one of the following:
    path     : FILE(s) are results of a path calculation.
    progress : FILE(s) are pickle files of dimer/lanczos or quasi-newton calculation
    xyz      : FILE(s) contain a string of xyz files.

    It is required to give at least one Option for table to specify what should be
    given and two (for x and y-coordinates) for teh plot/show function.

    EXAMPLE:

    paratools show path FILE --distance 1 2 --distance 1 3

    This would plot the distance between the atoms 1 and 3 against the distance of the
    atoms 1 and 2 for the file FILE.
    """
    from pts.tools.xyz2tabint import interestingvalue
    from pts.tools.path2tab import get_expansion
    from optparse import OptionParser
    parser = OptionParser(usage = visualize_input.__doc__)

    global num_i
    num_i = 1
    parser = geometry_values(parser)

    # Energies and gradient handling
    if input in ["path", "progress"]:
        parser  = gradient_and_energies(parser)
    else:
        parser.set_defaults(special_vals = None)

    # the number of images (points on a path)

    # For dimer, lanczos and quasi-newton
    if input == "progress":
        parser = dimer_specials(parser)

    # Different input format
    if input == "path":
        parser = path_specials(parser, num_old)
        parser = alternative_input_group(parser)
        parser = ase_input(parser)

    # Plotting settings
    if output in ["show", "plot"]:
        parser = plot_settings(parser)

    options, args = parser.parse_args(argv)

    symm = [n for n,v in options.symm]
    special_opt = options.diff, symm, options.symm

    if options.expand == None:
        appender = None, None, None
    else:
        a1, a2 = options.expand
        appender = get_expansion(a1, a2)
    values = options.withs, options.allval, options.special_vals, appender, special_opt

    if input == "path":
        # alternative input:
        obj = None
        if options.symbfile is not None:
            if options.mask == None:
                mask = None
                maskgeo = None
            else:
                mask, maskgeo = options.mask

            obj = read_path_fix( options.symbfile, options.zmats, mask, maskgeo )
        other_input = (options.symbfile, options.abcis, obj)
        if options.ase:
            options.next.append(len(args)+1)
            args = reorder_files(args, next)
        data_ase = ( options.ase, options.format)
        reference =  []
        reference_data = []
        for refs in options.references:
           ref1, ref2 = refs
           reference.append(ref1)
           reference_data.append(ref2)
        path_look = (options.num, options.ts_estimates, (reference, reference_data))
    else:
        other_input = None
        data_ase = None
        path_look = None

    if input == "progress":
        dimer_special = options.arrow_len, options.vec_angle
    else:
        dimer_special = None


    if output in ["show", "plot"]:
        for i in range(len(args)):
            options.names_of_lines.append([])
        for_plot = num_i, options.logscale, options.title, options.xlab, options.xran, options.ylab, options.yran, options.names_of_lines, options.output

    return args, data_ase, other_input, values, path_look, dimer_special, for_plot

def plot_settings(parser):
    """
    These options are for changing the scope of how to plot.
    """
    from optparse import OptionGroup

    group =  OptionGroup( parser, "SETTINGS FOR PLOT/SHOW PICTURES", plot_settings.__doc__)

    group.add_option("--title", dest = "title",
                      help = "Sets TEXT as title of the picture.",
                      metavar = "TEXT",
                      type = "string")
    group.add_option("--xlabel", dest = "xlab",
                      help = "Sets TEXT as label of X-axis of the picture.",
                      metavar = "TEXT",
                      type = "string")
    group.add_option("--ylabel", dest = "ylab",
                      help = "Sets TEXT as label of Y-axis of the picture.",
                      metavar = "TEXT",
                      type = "string")
    group.add_option("--xrange", dest = "xran",
                      help = "Sets the range of the X-axis to [N1, N2].",
                      metavar = "N1 N2",
                      type = "float", nargs = 2 )
    group.add_option("--yrange", dest = "yran",
                      help = "Sets the range of the Y-axis to [N1, N2].",
                      metavar = "N1 N2",
                      type = "float", nargs = 2 )
    group.add_option("--output", dest = "output",
                      help = "The output will go to FILE. This can also transform show into plot.",
                      metavar = "FILE",
                      type = "string")
    group.add_option("--name", dest = "names_of_lines",
                      help = "The n'th call of --name will set TEXT as label of the n'th FILE.",
                      metavar = "TEXT",
                      type = "string", action = "append", default = [])
    group.add_option("--logscale", dest = "logscale",
                      help = "Sets the axis DIR (x or y) to logarithmic scale.",
                      metavar = "DIR", choices = ["x", "y", "1", "2"],
                      type = "choice", action = "append", default = [])

    parser.add_option_group(group)
    return parser


# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
