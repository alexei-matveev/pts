#!/usr/bin/env python
"""
This  function performs  a  transition state  search  with the  "dimer
method" or  the "lanczos  method" They represent  different strategies
for obtaining  the lowest eigenmode of  the hessian.  As  the usage is
the  same for  both  methods in  the  following all  the commands  are
explained only  for the dimer method,  the lanczos method  can be used
the same way. Only in DIFFERENCES BETWEEN DIMER AND LANCZOS the laczos
method will be addressed explicitely again.

USAGE:

   paratools dimer --calculator <calculator_file> <geometry file> <start mode_vector file>

   paratools lanczos --calculator <calculator_file> <geometry file> <start mode_vector file>

It needs at least three input files:

 * one containing the  quantum chemical calculator (<calculator_file>)
   which is given via the parameter --calculator

 * one  containing   some  kind  of  Cartesian   geometry.  This  file
   (<geometry file>) should be ASE readable, meaning being in a format
   that ASE could understand (or be a gxfile having gx in its name)

 * one  should  contain a  mode  vector. This  should  be  as near  as
   possible  to  the  expected  negative  eigenmode  at  the  position
   specified  by  the geometry  file.  It  can  be given  in  internal
   coordinates  (than each  entry needs  a new  line) or  in Cartesian
   coordinates (where each atom has a line with three entries).

ALTERNATE USAGE:

This  is supposed if  the calculation  just follows  after any  of the
paratools  pathsearcher   runs  Here   a  patch.pickle  file   of  the
pathsearcher calculation is used for starting the dimer:

   paratools dimer --calculator <calculator_file> --pickle <ts estimate choice> <path.pickle file>

Here  the <ts  estimate  choice> is  one  of the  possible choices  of
transition state  estimate from a  given path. So for  example "cubic"
for the spline and cubic method or "highest" for the highest bead.

As inital  mode the  vector along the  path at position  of transition
state estimate is taken.

Be  aware  that  here  all  the  settings  for  the  quantum  chemical
calculator (like  cell) have  to be given  explicitely as there  is no
geometry file to read them from.


GEOMETRIES AND COORDINATE SYSTEMS:

There are  more options to  the coordinate system choice  (for example
using internal coordinates for  optimization). To find more about them
do:

   paratools dimer --help geometries

FURTHER OPTIONS:

There are some more variables specifying the way the dimer method run.
To find out more about them do:

   paratools dimer --help parameter

For getting a list with all their defaults do:
   paratools dimer --defaults

If it  is wanted to  change there something  find out the name  of the
variable to change and add to your command:

  --<variable name> <new value>

Just to mention  one: the maximum number of  translation step defaults
to a very large number. To reduce it add to the paratools command:

  --max_translation <number of translation iterations>

So for example to have only 100 of them:

  --max_translation 100

It is also  possible to reuse calculations from  older runs. They will
be  stored  in a  pickle  file  (only  readable by  current  paratools
version). If it is given in another calculation as:

    --cache <file name>

The results in  this file will be used if possible  instead of doing a
new calculation with  the caluclator.  The parameter can  be also used
to name or redirect the storage  of the results. As default it will go
to dimer.ResultDict.pickle.  Be aware that ParaTools does  not test if
the  results are  belonging to  the  current system  settings. If  the
geometry fits it takes the result.

DIFFERENCES BETWEEN DIMER AND LANCZOS METHOD:

Each of  the methods represent a  different way of  getting the lowest
eigenmode of an second derivative Hessian approximation at the current
point. The methods themselves are described elsewhere, here only their
differences in  performance and  recommendations which one  too choose
are given.

The  translation  method  uses  the  special mode  to  find  its  best
step.  Both  methods will  try  to update  the  result  from the  last
iteration. As this is usually not changing much in most cases there is
not much to gain. For rough rotational convergence criteria and a good
starting mode  the results should be  more or less  equivalent. But if
the mode should  be gotten with a high accuracy  the lanczos method is
supposed to be better.

EXAMPLES:

Having in a directory the  geometry file (POSCAR) POSCAR.start as well
as the mode vector file MODE_START and a valid caluclator file calc.py
do to start a 20 translation steps dimer optimization:

  paratools dimer --calculator calc.py  --max_translation 20 POSCAR.start MODE_START

"""

from sys import exit, stderr
rot_info = """
This is a function only for the rotation part of dimer/lanczos:

Usage:

  paratools dimer-rotate --calculator <calculator file> <start geometry> <start mode vector>

  paratools lanczos-rotate --calculator <calculator file> <start geometry> <start mode vector>

GEOMETRIES AND COORDINATE SYSTEMS:

There are  more options to  the coordinate system choice  (for example
using internal coordinates for  optimization). To find more about them
do:

   paratools dimer-rotate --help geometries

Be  aware that  this is  the  same help  function then  for the  plain
dimer/lanczos.    Thus    it   will    be    explained   with    these
commands. dimer-rotate and lanczos-rotate will work the same way.

FURTHER OPTIONS:

There are some more variables specifying the way the dimer method run.
To  find   out  more  about   them  do  (help  function   of  complete
dimer/lanczos):

   paratools dimer-rotate --help parameter

For getting a list with all their defaults do:

   paratools dimer-rotate --defaults

If it  is wanted to  change there something  find out the name  of the
variable to change and add to your command:

  --<variable name> <new value>

The  function  uses  only  the  rotation  part  of  the  dimer/lanczos
methods. Therefore all  things said there, which are  not dealing with
translation are valid  here also. The only differences  so far is that
there is a reduced and  changed set of parameters. They should default
to a much tighter convergence of the only rotatin step. Output will be
the new mode and the curvature, approximated for it.
"""

qn_info = """
This is  a simple Quasi Newton  method. For more complex  ones use the
ASE  functionalities  (over  the  interface  paratools  minimize,  for
example). The  only advantage of this one  is that it can  use all the
coordinate  systems available and  that it  can go  towards transition
states,  if it  is  near enough  as  it does  not  enforce a  positive
definite hessian.

Usage:

  paratools quasi-newton --calculator <calculator file> <start geometry>

GEOMETRIES AND COORDINATE SYSTEMS:

There are  more options to  the coordinate system choice  (for example
using internal coordinates for  optimization). To find more about them
do:

   paratools quasi-newton --help geometries

FURTHER OPTIONS:

There  are some  more variables  specifying the  way the  Quasi Newton
method run.  To find out more about them do:

   paratools quasi-newton --help parameter

For getting a list with all their defaults do:

   paratools quasi-newton --defaults

If it  is wanted to  change there something  find out the name  of the
variable to change and add to your command:

  --<variable name> <new value>
"""
import pts.metric as mt

def read_dimer_input(rest, name):
    """
    This function is similar to the one for pathsearcher
    """
    from pts.io.read_inputs import from_params_file
    from pts.io.read_COS import geo_params
    from pts.defaults import are_strings
    from pts.common import file2str
    from pts.func import compose
    from pts.qfunc import QFunc
    from pts.memoize import Memoize, FileStore
    from pts.defaults import di_default_params, qn_default_params
    #This variables will be needed afterwards anyway
    # independent of beeing given by user
    geo_dict = {"format" : None, "zmt_format" : "direct"}
    add_param = {}
    paramfile = None
    zmatrix = []
    geo = None
    mode = None
    as_pickle = False
    ts_estim = None
    accept_all = False

    give_help_if_needed(rest, name)

    if rest[0] == "--accept_all":
        rest = rest[1:]
        accept_all = True

    defaults_available = True
    if name in ["lanczos", "dimer", "lanczos-rotate", "dimer-rotate"]:
        default_params = di_default_params
    elif name in ["qn", "simple_qn", "quasi-newton"]:
        default_params = qn_default_params
    else:
        print >> stderr, "WARNING: No default parameter for specific method found"
        print >> stderr, "         There will be no test if the given parameter make sense"
        default_params = {}
        accept_all = True
        defaults_available = False

    if "--defaults" in rest:
        if defaults_available:
            print "The default parameters for the algorithm ", name, " are:"
            for param, value in default_params.iteritems():
                print "    %s = %s" % (str(param), str(value))
        else:
            print "No default values available for the specific method", name
        exit()

    for i in range(len(rest)):
        if rest == []:
            break

        # filter out all the options
        if rest[0].startswith("--"):
            o = rest[0][2:]
            a = rest[1]
            # filter out the special ones
            if o == "paramfile":
                # file containing parameters
                paramfile = file2str(a)
            elif o in ("zmatrix"):
                # zmatrix if given separate to the geometries
                zmatrix.append(a)
            elif o in ("pickle"):
                as_pickle = True
                ts_estim  = a
            elif o in geo_params:
                # only needed to build up the geometry
              if o in ("mask"):
                   # needed to build up the geometry and wanted for params output
                  geo_dict[o] = get_mask(a)
              elif o in ("cell", "pbc"):
                  geo_dict[o] = eval(a)
              else:
                  geo_dict[o] = a
            else:
                # suppose that the rest are setting parameters
                # we do not have a complet list of them
                if not accept_all:
                    assert o in default_params or o == "rot_method", "Parameter %s" % (o)

                if o in are_strings:
                    add_param[o] = a
                else:
                    add_param[o] = eval(a)
            rest = rest[2:]
        else:
            # This two files are needed any way: one geometry file and one
            # for the modevector, expect the geoemtry file to be given first
            if geo == None:
                # For reusing pathsearcher routines with several geoemtries for input
                geo = [ rest[0]]
            else:
                mode = rest[0]
            rest = rest[1:]


    if paramfile == None:
        params_dict = add_param
        geo_dict_dim = geo_dict
    else:
        if accept_all:
            params_dict, geo_dict_dim = from_params_file_dimer(paramfile )
        else:
            params_dict, geo_dict_dim = from_params_file(paramfile, default_params )
        params_dict.update(add_param)
        geo_dict_dim.update(geo_dict)

    if as_pickle:
        start_geo, init_mode, funcart, atoms = read_from_pickle(geo[0], ts_estim, geo_dict_dim)
    else:
        start_geo, funcart, atoms = build_new(geo, geo_dict_dim, zmatrix)
        if name in ["lanczos", "dimer", "lanczos-rotate", "dimer-rotate"]:
           init_mode = build_mode(mode, start_geo, funcart)
        else:
           assert mode == None
           init_mode = None

    # Build up the qfunc, calculator is included in atoms already
    pes = compose(QFunc(atoms, calc = atoms.get_calculator()), funcart)

    if "cache" in params_dict:
          if params_dict["cache"] == None:
                pes = Memoize(pes, FileStore("%s.ResultDict.pickle" % (name)))
          else:
                pes = Memoize(pes, FileStore(params_dict["cache"]))
    else:
         pes = Memoize(pes, FileStore("%s.ResultDict.pickle" % (name)))

    #Attention inital mode need not be normed (and cannot as metric is not yet known)
    return pes, start_geo, init_mode, params_dict, atoms, funcart

def give_help_if_needed(rest, name):
    from pts.defaults import info_di_params, info_qn_params
    from pts.io.read_COS import info_geometries
    if "--help" in rest:
        if "geometries" in rest:
           info_geometries()
        elif "parameter" in rest:
           if name in ["lanczos", "dimer", "lanczos-rotate", "dimer-rotate"]:
              info_di_params()
           elif name in ["qn", "simple_qn", "quasi-newton"]:
              info_qn_params()
           else:
              print "No parameter information for the specific method", name ,"found."
        else:
           if name in ["lanczos", "dimer"]:
               print __doc__
           elif name in ["lanczos-rotate", "dimer-rotate"]:
               print rot_info
           elif name in ["qn", "simple_qn", "quasi-newton"]:
               print qn_info
           else:
               print "No help text for the specific method", name, "found."
        exit()


def read_from_pickle(file, ts_est, geo_dict):
    from pts.tools.pathtools import unpickle_path, PathTools
    from ase import Atoms
    from ase.io import write
    from pts.io.read_COS import set_atoms
    from pts.searcher import new_abscissa
    from numpy import savetxt

    coord_b, energy_b, gradients_b, __, __, symbols, funcart = unpickle_path(file) # v2
    mt.setup_metric(funcart)
    startx =  new_abscissa(coord_b, mt.metric)
    pt2 = PathTools(coord_b, energy_b, gradients_b, startx)

    if ts_est == "spline":
        ts_int = pt2.ts_spl()
        if len(ts_int) == 0:
            print >> stderr, "Transition state estimate did not provide any results"
	    print >> stderr, "Aborting!"
	    exit()
        est = ts_int[-1]
    elif ts_est == "spl_avg":
        ts_int = pt2.ts_splavg()
        if len(ts_int) == 0:
            print >> stderr, "Transition state estimate did not provide any results"
	    print >> stderr, "Aborting!"
	    exit()
        est = ts_int[-1]
    elif ts_est == "cubic":
        ts_int = pt2.ts_splcub()
        if len(ts_int) == 0:
            print >> stderr, "Transition state estimate did not provide any results"
	    print >> stderr, "Aborting!"
	    exit()
        est = ts_int[-1]
    elif ts_est == "highest":
        est = pt2.ts_highest()[-1]
    elif ts_est == "bell":
        ts_int = pt2.ts_bell()
        if len(ts_int) == 0:
            print >> stderr, "Transition state estimate did not provide any results"
	    print >> stderr, "Aborting!"
	    exit()
        est = ts_int[-1]
    else:
        print >> stderr, "Transition state estimate not found", ts_est
	print >> stderr, "Make sure that the name is written correctly"
	exit()

    energy, start_geo, __, __,s_ts,  __, __ = est
    init_mode = pt2.xs.fprime(s_ts)

    atoms = Atoms(symbols)
    atoms.set_positions(funcart(start_geo))
    atoms = set_atoms(atoms, geo_dict)
    write("start.xyz", atoms)
    savetxt("mode.start", init_mode)
    return start_geo, init_mode, funcart, atoms

def build_new(geo, geo_dict_dim, zmatrix):
    from pts.io.read_inputs import get_geos, get_masked
    # also pathsearcher routines to build atoms object and internal to Cartesian
    # handle, the variables not used here would be required to ensure
    # shortest way between some pictures
    atoms, init_geo, funcart, __, __, __, mask1 = get_geos(geo, geo_dict_dim, zmatrix)
    # if a mask has been provided, some variables are not optimized
    funcart, init_geo = get_masked(funcart, atoms, geo_dict_dim, zmatrix == [], init_geo, mask1)

    # leave no constraints at the atoms:
    for con in atoms.constraints:
       atoms.constraints.remove(con)

    # We have only one geometry here
    start_geo = init_geo[0]

    return start_geo, funcart, atoms

def build_mode(mode, start_geo, funcart):
    from numpy import loadtxt
    # Modevector either in internals (like direct from previous calculation with
    # dimer or pathsearcher) or external most certainly in Cartesian
    mode_cart = loadtxt(mode)
    try:
        # Test for Cartesian
        ma, mb = mode_cart.shape
        if mb == 3:
           # The functions build up so far, all provide this:
           init_mode = (funcart.pinv(funcart(start_geo) * 0.0001 * mode_cart) - start_geo) / 0.0001
           # init_mode will be normed correctly (after metric is set) lateron, here do
           # only roughly
        else:
           print >> stderr, "Error: illegal format for mode vector."
           print >> stderr, "Needs either internal coordinates or Cartesian coordinates."
           exit()
    except ValueError:
       # Needs to be in internal then
       assert (len(mode_cart) == len(start_geo))
       init_mode = mode_cart

    return init_mode

def from_params_file_dimer( lines ):
    """
    overwrite params in the params dictionary with the params
    specified in the string lines (can be the string read from a params file

    checks if there are no additional params set
    """
    # the get the params out of the file is done by exec, this
    # will also execute the calculator for example, we need ase here
    # so that the calculators from there can be used

    # execute the string, the variables should be set in the locals
    from pts.io.read_COS import geo_params

    params_dict = {}
    geo_dict = {}

    glob_olds = locals().copy()
    #print glob_olds.keys()
    exec(lines)
    glob = locals()
    #print glob.keys()

    for param in glob:
        if param not in glob_olds:
             if param == "glob_olds":
                 # There is one more new variable, which is not wanted to be taken into account
                 pass
             elif param in geo_params:
                 geo_dict[param] = glob[param]
             else:
                 # Parameters may be overwritten by the fileinput
                 params_dict[param] = glob[param]

    return params_dict, geo_dict
