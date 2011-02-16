#!/usr/bin/env python
"""
This tool is the interface to the string and NEB methods.

GEOMETRY

Geometries have to be given in internal coordinates (the ones the function accepts)

"""
from sys import argv, exit
from re import findall
from os import path, mkdir
from pts.pathsearcher_defaults.params_default import *
from pts.pathsearcher_defaults import *
from pts import MolInterface, CalcManager, generic_callback
from pts.searcher import GrowingString, NEB
from pts.optwrap import runopt as runopt_aof
from pts.tools import pickle_path
from pts.common import file2str
# be careful: array is needed, when the init_path is an array
# do not delete it, even if it never occures directly in this module!
# FIXME: really?
# DONT: from numpy import array

# needed as global variable
cb_count_debug = 0

def pathsearcher(tbead_left, init_path, old_results = None, paramfile = None, funcart = [],fprime_exist = False, **parameter):
    """
    ...
    It is possible to use the pathsearcher() function in a python script. It
    looks like:

      from ??? import pathsearcher

      pathsearcher(tbead_left,init_path, funcart, **kwargs)

      * tbead_left is an ASE atoms object used to calculate the forces and energies of a given
      (Cartesian) geometry. Be aware that it needs to have an calculator attached to it, which will
      do the actual transformation. Another possibility is to give a file in which calculator is specified
      separately as parameter.
      * init_path is an array containting for each bead of the starting path the internal coordinates.
      * funcart is a function to transform internal to Cartesian coordinates. It may be a simle function,
      but if it provides the structure of pts.func function, meaning a taylor and a fprime function it
      may be told so to the program by setting fprime_exist to True. Else a NumDiff function will be build
      with funcart as normal function.

      * the other parameters give the possibility to overwrite some of the default behaviour of the module,
      They are provided as kwargs in here. For a list of them see pathsearcher_defaults/params_default.py
      They can be also specified in an input file given as paramfile.
    """
    # set up parameters (fill them in a dictionary)
    params_dict = set_defaults()
    if not paramfile == None:
        params_dict = reset_params_f(params_dict, paramfile)
    params_dict = reset_params_d(params_dict, **parameter)

    # naming for output files
    if params_dict["name"] == None:
        params_dict["name"] = str(params_dict["cos_type"])
    name = params_dict["name"]

    if type(params_dict["calculator"]) == str:
       tbead_left.set_calculator( pd_calc(params_dict))

    if params_dict["cos_type"].lower() == "neb":
        if params_dict["opt_type"] == "multiopt":
            print "The optimizer %s is not designed for working with the method neb", params_dict["opt_type"]
            params_dict["opt_type"] = "ase_lbfgs"
            print "Thus it is replaced by the the optimizer", params_dict["opt_type"]
            print "This optimizer is supposed to be the default for neb calculations"


    # prepare the coordinate objects, eventually replace parameter by the ones given in the
    # atoms minima objects and extend the zmatrix, mi is a MolInterface object, which is the
    # wrapper around the atoms object in ASE
    mi, params = prepare_mol_objects(tbead_left, params_dict, funcart, init_path, fprime_exist)


    tell_params(params_dict,  old_results)

    if not path.exists(params_dict["output_path"]):
        mkdir(params_dict["output_path"])

    # some output files:
    logfile = open(name + '.log', 'w')
    disk_result_cache = None
    if params_dict["output_level"] > 0:
        disk_result_cache = '%s.ResultDict.pickle' % name

    # Calculator
    procs_tuple = (params_dict["cpu_architecture"], params_dict["pmax"], params_dict["pmin"])
    calc_man = CalcManager(mi, procs_tuple, to_cache=disk_result_cache, from_cache=old_results)

    # decide which method is actually to be used
    cos_type = params_dict["cos_type"].lower()
    if cos_type == 'string':
         CoS = GrowingString(init_path,
               calc_man,
               params_dict["beads_count"],
               growing=False,
               parallel=True,
               reporting=logfile,
               freeze_beads=False,
               head_size=None,
               output_level = params_dict["output_level"],
               max_sep_ratio=0.3)
    elif cos_type == 'growingstring':
         CoS = GrowingString(init_path,
               calc_man,
               params_dict["beads_count"],
               growing=True,
               parallel=True,
               reporting=logfile,
               freeze_beads=False,
               head_size=None,
               output_level = params_dict["output_level"],
               max_sep_ratio=0.3)
    elif cos_type == 'searchingstring':
         CoS = GrowingString(init_path,
               calc_man,
               params_dict["beads_count"],
               growing=True,
               parallel=True,
               reporting=logfile,
               output_level = params_dict["output_level"],
               max_sep_ratio=0.3,
               freeze_beads=True,
               head_size=None, # has no meaning for searching string
               growth_mode='search')
    elif cos_type == 'neb':
         CoS = NEB(init_path,
               calc_man,
               params_dict["spring"],
               params_dict["beads_count"],
               parallel=True,
               output_level = params_dict["output_level"],
               reporting=logfile)
    else:
         raise Exception('Unknown type: %s' % cos_type)
    #CoS.arc_record = open("archive.pickle", 'w')
    #dump("Version 0.1", CoS.arc_record)
    #dump(mi.build_coord_sys(init_path[0]), CoS.arc_record)

    # has also set global, as the callback function wants this
    # but here it is explictly reset to 0
    cb_count_debug = 0

    # callback function
    def cb(x, tol=0.01):
         global cb_count_debug
         if params_dict["output_level"] > 1:
             pickle_path(mi, CoS, "%s/%s.debug%d.path.pickle" % (params_dict["output_path"],name, cb_count_debug))
         cb_count_debug += 1
         return generic_callback(x, mi, CoS, params, tol=tol)

    # print out initial path
    cb(CoS.state_vec)

    # hack to enable the CoS to print in cartesians, even if opt is done in internals
    CoS.bead2carts = lambda x: mi.build_coord_sys(x).get_cartesians().flatten()

    extra_opt_params = dict()

    runopt = lambda CoS_: runopt_aof(params_dict["opt_type"], CoS_, params_dict["ftol"], params_dict["xtol"], params_dict["etol"], params_dict["maxit"], cb, maxstep=params_dict["maxstep"], extra=extra_opt_params)

    # main optimisation loop
    print runopt(CoS)

    cs = mi.build_coord_sys(CoS.get_state_vec()[0])
    print "Optimized path:"
    print "in internals"
    for i, state in enumerate(CoS.get_state_vec()):
        cs = mi.build_coord_sys(state)
        print cs.get_internals()

    print "in Cartesians"
    for i, state in enumerate(CoS.get_state_vec()):
        cs = mi.build_coord_sys(state)
        print cs.get_cartesians()

    # get best estimate(s) of TS from band/string
    tss = CoS.ts_estims(alsomodes = False, converter = mi.build_coord_sys(CoS.get_state_vec()[0]))

    # write out path to a file
    if params_dict["output_level"] > 0:
        pickle_path(mi, CoS, "%s.path.pickle" % name)
    # print cartesian coordinates of all transition states that were found
    print "Dumping located transition states"
    for i, ts in enumerate(tss):
        e, v, s0, s1,_ ,bead0_i, bead1_i = ts
        cs = mi.build_coord_sys(v)
        print "Energy = %.4f eV, between beads %d and %d." % (e, bead0_i, bead1_i)
        cs = mi.build_coord_sys(state)
        print "Positions", cs.get_internals()
        print "Cartesians", cs.get_cartesians()

def set_defaults():
    """
    Initalize the parameter dictionary with the default parameters
    """
    params_dict = default_params.copy()
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

def reset_params_d(params_dict, **new_parameter):
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

def prepare_mol_objects(tbead_left, params_dict, funcart, init_path, fprime_exist):
    """
    There are several valid possibilities of how the geometry/atoms input could be
    specified, here they should all lead to an valid MolInterface function
    """
    # for extracting the parameter needed for the MI
    params = {}

    # extract parameters from the dictionary
    for x in ["output_level","output_path"]:
        params[x] = params_dict[x]
    params["name"] = params_dict["output_path"] + "/" + params_dict["name"]

    # the MI:
    mi = MolInterface(tbead_left, funcart, init_path, fprime_exist = fprime_exist, **params)

    return  mi, params

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

def pd_calc(params):

    # See eval below, that may be one of the ASE calculators:
    from ase.calculators import *

    calc = params["calculator"]
    if calc in default_calcs:
        params["calculator"] = eval("%s" % (calc))
    else:
        str1 = file2str(calc)
        exec(str1)
    return calculator

def tell_default_params():
    """
    Show the default params
    """
    print "The default parameters for the path searching algorithm are:"
    for param, value in default_params.iteritems():
         print "    %s = %s" % (str(param), str(value))

def tell_params(params, old_results):
    """
    Show the actual params
    """
    print "The specified parameters for this path searching calculation are:"
    for param, value in params.iteritems():
         print "    %s = %s" % (str(param), str(value))

    if old_results != None:
         print "Results from previous calculations are read in from %s", old_results

if __name__ == "__main__":
     print __doc__
     tell_default_params()
     exit()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
