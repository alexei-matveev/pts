#!/usr/bin/env python
"""
This tool is the interface to the string and NEB methods.

GEOMETRY

Geometries have to be given in internal coordinates (the ones the function accepts)

"""
from sys import argv, exit
from re import findall
from os import path, mkdir, remove
from shutil import copyfile
from warnings import warn
from pts.pathsearcher_defaults.params_default import *
from pts.pathsearcher_defaults import *
from pts.qfunc import QFunc, QMap
from pts.func import compose
from pts.paramap import PMap, PMap3
from pts.sched import Strategy
from pts.memoize import Memoize, elemental_memoize, FileStore
from pts.parasearch import generic_callback
from pts.searcher import GrowingString, NEB, ts_estims
from pts.optwrap import runopt
from pts.sopt import soptimize
from pts.tools import pickle_path
from pts.common import file2str
from pts.readinputs import interprete_input
import pts.metric as mt
# be careful: array is needed, when the init_path is an array
# do not delete it, even if it never occures directly in this module!
# FIXME: really?
# DONT: from numpy import array

# needed as global variable
cb_count_debug = 0


def pathsearcher(atoms, init_path, funcart, **kwargs):
    """Script-verison of find_path(), interprets and prints results to tty.

    It is possible to use the pathsearcher() function in a python script. It
    looks like:

      from pts.inputs import pathsearcher

      pathsearcher(atoms, init_path, funcart, **kwargs)

      * atoms is an ASE atoms object used to calculate the forces and energies of a given
      (Cartesian) geometry. Be aware that it needs to have an calculator attached to it, which will
      do the actual transformation.
      Another possibility is to give a file in which calculator is specified separately as parameter.
      (FIXME: this another possibility is vaguely specified)

      * init_path is an array containting for each bead of the starting path the internal coordinates.

      * funcart is a Func to transform internal to Cartesian coordinates. 

      * the other parameters give the possibility to overwrite some of the default behaviour of the module,
      They are provided as kwargs in here. For a list of them see pathsearcher_defaults/params_default.py
      They can be also specified in an input file given as paramfile.
    """

    # there is a lot of parameters affecting the calculation,
    # collect them in a dictionary, adding to those provided:
    kwargs = mkparams(**kwargs)

    # calculator from kwargs, if valid, has precedence over
    # the associated (or not) with the atoms:
    if kwargs["calculator"] is not None:
        atoms.set_calculator(kwargs["calculator"])

    # calculator is not used below:
    del kwargs["calculator"]

    # print parameters to STDOUT:
    tell_params(kwargs)

    # PES to be used for energy, forces. FIXME: maybe adapt QFunc not to
    # default to LJ, but rather keep atoms as is?
    pes = compose(QFunc(atoms, atoms.get_calculator()), funcart)

    # This parallel mapping function puts every single point calculation in
    # its own subfolder
    strat = Strategy(kwargs["cpu_architecture"], kwargs["pmin"], kwargs["pmax"])
    kwargs["pmap"] = PMap3(strat=strat)

    kwargs["int2cart"] = funcart
    kwargs["ch_symbols"] = atoms.get_chemical_symbols()
    # always do single point energy/force calculation in separate folder
    kwargs["workhere"] = False

    # this operates with PES in internals:
    convergence, geometries, energies, gradients = find_path(pes, init_path, **kwargs)

    # print user-friendly output, including cartesian geometries:
    output(geometries, energies, gradients, funcart)

    return convergence, geometries, energies, gradients

def find_path(pes, init_path
                            , beads_count = None    # default to len(init_path)
                            , name = "find-path"    # for output
                            , method = "string"     # what way, e.g. NEB, string, growingstring, searchingstring
                            , opt_type = "multiopt" # the optimizer
                            , spring = 5.0          # only for NEB: spring constant
                            , output_level = 2
                            , output_path = "."
                            , int2cart = 0       # For mere transformation of internal to Cartesians
                            , ch_symbols = None     # Only needed if output needs them
                            , cache = None
                            , pmap = PMap()
                            , workhere = False
                            , **kwargs):
    """This one does the real work ...

    """

    if beads_count is None:
        beads_count = len(init_path)

    if not path.exists(output_path):
        mkdir(output_path)

    # some output files:
    logfile = open(name + '.log', 'w')
    disk_result_cache = None
    if output_level > 0:
        cache_name = output_path + '/%s.ResultDict.pickle' % name
        if  cache == None:
            try:
                remove(cache_name)
                warn("WARNING: found old ResultDict.pickle, which was not given as previous results")
                warn("         Thus I will remove it")
            except OSError:
                pass
        else:
             cache_name = cache
        disk_result_cache = FileStore(cache_name)

    # decide which method is actually to be used
    method = method.lower()

    mt.setup_metric(int2cart)
    #
    # NOTE: most of the parameters to optimizers might be passed
    # via **kwargs. This may require changes in the interface of
    # the CoS constructors to accept trailing **kwargs for unrecognized
    # keywords, though:
    #
    if method == 'string':
        CoS = GrowingString(init_path,
               pes,
               disk_result_cache,
               beads_count=beads_count,
               growing=False,
               parallel=True,
               reporting=logfile,
               freeze_beads=False,
               head_size=None,
               output_level=output_level,
               output_path=output_path,
               pmap = pmap,
               max_sep_ratio=0.3)
    elif method == 'growingstring':
        CoS = GrowingString(init_path,
               pes,
               disk_result_cache,
               beads_count=beads_count,
               growing=True,
               parallel=True,
               reporting=logfile,
               freeze_beads=False,
               head_size=None,
               pmap = pmap,
               output_path=output_path,
               output_level=output_level,
               max_sep_ratio=0.3)
    elif method == 'searchingstring':
        CoS = GrowingString(init_path,
               pes,
               disk_result_cache,
               beads_count=beads_count,
               growing=True,
               parallel=True,
               reporting=logfile,
               pmap = pmap,
               output_path=output_path,
               output_level=output_level,
               max_sep_ratio=0.3,
               freeze_beads=True,
               head_size=None, # has no meaning for searching string
               growth_mode='search')
    elif method == 'neb':
        CoS = NEB(init_path,
               pes,
               spring,
               disk_result_cache,
               beads_count=beads_count,
               parallel=True,
               pmap = pmap,
               output_path=output_path,
               output_level=output_level,
               reporting=logfile)
    elif method == 'sopt':
        CoS = None
        # nothing, but see below ...
    else:
         raise Exception('Unknown type: %s' % method)

    # has also set global, as the callback function wants this
    # but here it is explictly reset to 0
    cb_count_debug = 0

    # callback function
    def cb(x, tol=0.01):
         global cb_count_debug
         if output_level > 1:
             pickle_path(int2cart, ch_symbols, CoS, "%s/%s.debug%d.path.pickle" % (output_path, name, cb_count_debug))
         cb_count_debug += 1
         return generic_callback(x, None, None, tol=tol
                    , name = output_path + "/" + name
                    , output_level=output_level
                    , output_path=output_path
                    , **kwargs)

    # print out initial path
    cb(init_path)

    if method != 'sopt':
        #
        # Main optimisation loop:
        #
        converged = runopt(opt_type, CoS, callback=cb, **kwargs)
        geometries, energies, gradients = CoS.state_vec, CoS.bead_pes_energies, CoS.bead_pes_gradients
    else:
        #
        # Alternative optimizer:
        #
        geometries, stats = soptimize(pes, init_path, maxiter=20, maxstep=0.1, callback=cb)
        _, converged, _, _ = stats
        energies, gradients = zip(*map(pes.taylor, geometries))

    # write out path to a file
    if output_level > 0:
        pickle_path(int2cart, ch_symbols, CoS, "%s.path.pickle" % name)

    # Return (hopefully) converged discreete path representation:
    #  return:  if converged,  internal coordinates, energies, gradients of last iteration
    return converged, geometries, energies, gradients

def output(beads, energies, gradients, cartesian):
    """Print user-friendly output.
    Also estimates locations of transition states from bead geometries.
    """

    print "Optimized path:"
    print "in internals"
    for bead in beads:
        print bead

    print "in Cartesians"
    for bead in beads:
        print cartesian(bead)

    # get best estimate(s) of TS from band/string
    tss = ts_estims(beads, energies, gradients, alsomodes=False, converter=cartesian)

    # print cartesian coordinates of all transition states that were found
    print "Dumping located transition states"
    for i, ts in enumerate(tss):
        e, v, s0, s1,_ ,bead0_i, bead1_i = ts
        print "Energy = %.4f eV, between beads %d and %d." % (e, bead0_i, bead1_i)
        print "Positions", v
        print "Cartesians", cartesian(v)

def mkparams(paramfile = None, **parameter):
    """Returns a dictionary with parameters of the search procedure
    """

#   print "mkparams: parameter=\n", parameter

    # set up parameters (fill them in a dictionary)
    params_dict = set_defaults()

#   print "mkparams: defaults=\n", params_dict

    if paramfile is not None:
        params_dict = reset_params_f(params_dict, paramfile)

#   print "mkparams: from text=\n", params_dict

    params_dict = reset_params_d(params_dict, parameter)

#   print "mkparams: from dict=\n", params_dict

    # naming for output files
    if params_dict["name"] == None:
        params_dict["name"] = str(params_dict["method"])

    # This is an alternative way of specifing calculator, default is
    # to keep atoms.get_calculator(): FIXME: this part belongs into
    # section of reading/parsing parameters (maybe reset_params_f?):
    if type(params_dict["calculator"]) == str:
        params_dict["calculator"] = eval_calc(params_dict["calculator"])

    if params_dict["method"].lower() == "neb":
        if params_dict["opt_type"] == "multiopt":
            print "The optimizer %s is not designed for working with the method neb", params_dict["opt_type"]
            params_dict["opt_type"] = "ase_lbfgs"
            print "Thus it is replaced by the the optimizer", params_dict["opt_type"]
            print "This optimizer is supposed to be the default for neb calculations"

    return params_dict

def set_defaults():
    """
    Initalize the parameter dictionary with the default parameters
    """
    params_dict = default_params.copy()

    # FIXME: is this a universal parameter?
    # Maybe move it to "default_params"?
    params_dict["old_results"] = None

    return params_dict

def reset_params_f(params_dict, lines):
    """
    overwrite params in the params dictionary with the params
    specified in the string lines (can be the string read from a params file

    checks if there are no additional params set
    """
    # the get the params out of the file is done by exec, this
    # will also execute the calculator for example, we need ase here
    # so that the calculators from there can be used
    import ase

    # execute the string, the variables should be set in the locals
    glob_olds = locals().copy()
    print glob_olds.keys()
    exec(lines)
    glob = locals()
    print glob.keys()

    for param in glob.keys():
        if not param in glob_olds.keys():
             if param == "glob_olds":
                 # There is one more new variable, which is not wanted to be taken into account
                 pass
             elif not param in params_dict.keys():
                 # this parameter are set during exec of the parameterfile, but they are not known
                 print "WARNING: unrecognised variable in parameter input file"
                 print "The variable", param," is unknown"
             else:
                 # Parameters may be overwritten by the fileinput
                 params_dict[param] = glob[param]

    return params_dict

def reset_params_d(params_dict, new_parameter):
    """
    Overwrite parameters by ones given directly (as a dictionary)
    in the parameters, here we check also if there are some unknown ones,
    else we could use just update
    """
    for key in new_parameter.keys():
        if key in params_dict:
            params_dict[key] = new_parameter[key]
        else:
            print "ERROR: unrecognised variable in parameter"
            print "The variable",key, "has not been found"
            print "Please check if it is written correctly"
            exit()
    return params_dict

def expand_zmat(zmts):
    """
    If zmatrix only contains the matix itsself, add here
    the initial variables (the value is unimportant right now)
    """
    # variables should be proceeded by a number specifying the new
    # atom this varible is connected to:
    # could not use whitespace, because has to stay int he line
    vars = findall(r"\d[ \t\r\f\v]+[a-zA-Z_]\w*",zmts)
    # some variables may appear more than once:
    allvars = dict()
    allvars_orderd = []

    # the number of variables help to decide if its an zmatrix or
    # an complex coordinate system
    num_vars = len(vars)
    for vari in vars:
        # get rid of the number stored in front
        fields = vari.split()
        var = fields[1]
        # if we have to keep them all, count the
        # multiplicity of them, maybe useful some time
        if var in allvars.keys():
            allvars[var] += 1
        else:
            allvars[var] = 1
            allvars_orderd.append(var)

    zmts_new = zmts
    # an empty line is the dividor between zmatrix part and
    # part of initalized variables
    empty_line = findall(r"\n\n",zmts)

    # it this is not already here, add variable initializing part
    # set all of them to 1
    if empty_line == []:
        zmts_new += "\n"
        for var1 in allvars_orderd:
            zmts_new += "%s  1.0\n" % (var1)
        #print "Expanded zmatrix, added initial values"
        #print zmts_new
    return zmts_new, num_vars

def tell_params(params):
    """
    Show the actual params
    """
    print "The specified parameters for this path searching calculation are:"
    for param, value in params.iteritems():
         print "    %s = %s" % (str(param), str(value))

def main(args):
    """
    starts a pathsearcher calculation
    This variant expects the calculation to be done with an ASE atoms object
    coordinate systems are limited to internal, Cartesian and mixed systems

    Uses the arguments of the standard input for setting the parameters
    """
    atoms, init_path, funcart, kwargs = interprete_input(args)
    pathsearcher(atoms, init_path, funcart, **kwargs)


if __name__ == "__main__":
    main(argv[1:])

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
