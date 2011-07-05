#!/usr/bin/env python
"""
This tool is the interface to the string and NEB methods.

GEOMETRY

Geometries have to be given in internal coordinates (the ones the function accepts)

"""
from ase.io import write
from sys import argv, exit
from re import findall
from os import path, mkdir, remove
from numpy import savetxt
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
from pts.cfunc import Pass_through
from pts.optwrap import runopt
from pts.sopt import soptimize
from pts.tools import pickle_path
from pts.common import file2str
from pts.readinputs import interprete_input, create_params_dict
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
    # most parameters are stored in a dictionary, default parameters are stored in
    # pathsearcher_defaults/params_default
    para_dict = create_params_dict(kwargs)

    # calculator from kwargs, if valid, has precedence over
    if "calculator" in para_dict:
        # the associated (or not) with the atoms:
        if para_dict["calculator"] is not None:
            atoms.set_calculator(para_dict["calculator"])

        # calculator is not used below:
        del para_dict["calculator"]

    # print parameters to STDOUT:
    tell_params(para_dict)

    # PES to be used for energy, forces. FIXME: maybe adapt QFunc not to
    # default to LJ, but rather keep atoms as is?
    pes = compose(QFunc(atoms, atoms.get_calculator()), funcart)

    # This parallel mapping function puts every single point calculation in
    # its own subfolder
    strat = Strategy(para_dict["cpu_architecture"], para_dict["pmin"], para_dict["pmax"])
    del para_dict["cpu_architecture"]
    del para_dict["pmin"]
    del para_dict["pmax"]
    if "pmap" not in para_dict:
        para_dict["pmap"] = PMap3(strat=strat)

    para_dict["int2cart"] = funcart
    para_dict["symbols"] = atoms.get_chemical_symbols()

    # this operates with PES in internals:
    convergence, optimized_path = find_path(pes, init_path, **para_dict)

    # print user-friendly output, including cartesian geometries:
    output(optimized_path, funcart, para_dict["output_level"], para_dict["output_geo_format"], atoms)

    return convergence, optimized_path

def find_path(pes, init_path
                            , beads_count = None    # default to len(init_path)
                            , name = "find-path"    # for output
                            , method = "string"     # what way, e.g. NEB, string, growingstring, searchingstring
                            , opt_type = "multiopt" # the optimizer
                            , spring = 5.0          # only for NEB: spring constant
                            , output_level = 2
                            , output_path = "."
                            , int2cart = Pass_through()   # For mere transformation of internal to Cartesians
                            , symbols = None     # Only needed if output needs them
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
               workhere=workhere,
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
               workhere=workhere,
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
               workhere=workhere,
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
               workhere=workhere,
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
         if output_level > 1 and CoS is not None:
             coords, pathps, energies, gradients = CoS.state_vec.reshape(CoS.beads_count,-1), \
                   CoS.pathpos(), \
                   CoS.bead_pes_energies.reshape(-1), \
                   CoS.bead_pes_gradients.reshape(CoS.beads_count,-1)
             pickle_path(coords, pathps, energies, gradients, symbols, int2cart, "%s/%s.debug%d.path.pickle" % (output_path, name, cb_count_debug))
         if output_level > 2 and CoS is not None:
             # store interal coordinates of given iteration in file
             savetxt("%s/%s.state_vec%d.txt" % (output_path, name, cb_count_debug), CoS.state_vec.reshape(CoS.beads_count,-1))
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
        abscissa  = CoS.pathpos()
        geometries, energies, gradients = CoS.state_vec, CoS.bead_pes_energies, CoS.bead_pes_gradients
    else:
        #
        # Alternative optimizer:
        #
        geometries, stats = soptimize(pes, init_path, maxiter=20, maxstep=0.1, callback=cb)
        _, converged, _, _ = stats
        abscissa  = None
        energies, gradients = zip(*map(pes.taylor, geometries))

    # write out path to a file
    if output_level > 0 and CoS is not None:
        coords, pathps, energies, gradients = CoS.state_vec.reshape(CoS.beads_count,-1), \
              CoS.pathpos(), \
              CoS.bead_pes_energies.reshape(-1), \
              CoS.bead_pes_gradients.reshape(CoS.beads_count,-1)
        pickle_path(coords, pathps, energies, gradients, symbols, int2cart, "%s.path.pickle" % (name))

    # Return (hopefully) converged discreete path representation:
    #  return:  if converged,  internal coordinates, energies, gradients of last iteration
    return converged, (geometries, abscissa, energies, gradients)

def output(optimized_path, cartesian, output_level, format , atoms):
    """Print user-friendly output.
    Also estimates locations of transition states from bead geometries.
    """
    beads, abscissa, energies, gradients = optimized_path

    print "Optimized path:"
    print "in internals"
    for bead in beads:
        print bead
    if output_level > 0:
        savetxt("internal_coordinates", beads)
        savetxt("energies", energies)
        savetxt("forces", gradients)
        if abscissa is not None:
            savetxt("abscissa", abscissa)

        print "in Cartesians"
    for i, bead in enumerate(beads):
        carts = cartesian(bead)
        print carts
        atoms.set_positions(carts)
        write("bead%d" % i, atoms, format=format)

    # get best estimate(s) of TS from band/string
    tss = ts_estims(beads, energies, gradients, alsomodes=False, converter=cartesian)

    # print cartesian coordinates of all transition states that were found
    print "Dumping located transition states"
    for i, ts in enumerate(tss):
        e, v, s0, s1,_ ,bead0_i, bead1_i = ts
        print "Energy = %.4f eV, between beads %d and %d." % (e, bead0_i, bead1_i)
        print "Positions", v
        carts = cartesian(v)
        print "Cartesians", carts
        atoms.set_positions(carts)
        if output_level > 0:
             write("ts_estimate%d" % i, atoms, format = format)
        if output_level > 1:
             savetxt("ts_internals%d" % i, v)

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
