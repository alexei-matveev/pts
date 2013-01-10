#!/usr/bin/env python
"""
This tool is the interface to the string and NEB methods.

GEOMETRY

Geometries have to be given in internal coordinates (the ones the function accepts)

"""
from ase.io import write
from sys import argv
from os import path, mkdir, remove
from numpy import savetxt, array
from warnings import warn
from pts.qfunc import QFunc, qmap
from pts.func import compose
from pts.paramap import PMap, PMap3
from pts.sched import Strategy
from pts.memoize import Memoize, DirStore, FileStore
from pts.searcher import GrowingString, NEB, ts_estims
from pts.cfunc import Pass_through
from pts.optwrap import runopt
from pts.sopt import soptimize
from pts.tools.pathtools import pickle_path
from pts.io.read_inputs import interprete_input, create_params_dict
import pts.metric as mt
# be careful: array is needed, when the init_path is an array
# do not delete it, even if it never occures directly in this module!
# FIXME: really?
# DONT: from numpy import array

def pes_method(pes, path, **kw):
    """
    Has the interface of the most general PES method.
    """

    convergence, optimized_path = find_path(pes, path, **kw)

    # Command  line verison  of the  path searcher  used  to print
    # user-friendly output, including cartesian geometries:
    output(optimized_path, kw["trafo"], kw["output_level"], kw["output_geo_format"], kw["atoms"])

    # Command line version used to return this:
    return convergence, optimized_path

# needed as global variable
cb_count_debug = 0

def pathsearcher(atoms, init_path, trafo, **kwargs):
    """
    MUSTDIE: from  python code use  a PES method  find_path(pes, path,
             ...), for command  line there is call_with_pes(find_path,
             sys.argv[1:])   See   call_with_pes()   and   find_path()
             instead.

    Only used in ./tests.

    It  is possible  to use  the pathsearcher()  function in  a python
    script. It looks like:

      from pts.inputs import pathsearcher

      pathsearcher(atoms, init_path, trafo, **kwargs)

    * atoms is  an ASE atoms object  used to calculate  the forces and
      energies of a given (Cartesian) geometry. Be aware that it needs
      to have an  calculator attached to it, which  will do the actual
      transformation.  Another possibility is  to give the calculator
      separately as an option.

    * init_path is an array containting  for each bead of the starting
      path the internal coordinates.

    * trafo is a Func to transform internal to Cartesian coordinates.

    * the other  parameters give the possibility to  overwrite some of
      the default behaviour of the module, They are provided as kwargs
      in here.  For a list  of them see  defaults.py They can  be also
      specified in an input file given as paramfile.
    """
    # most parameters are stored in a dictionary, default parameters are stored in
    # defaults.py
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

    #
    # PES to be used for  energy, forces. FIXME: maybe adapt QFunc not
    # to default to LJ, but rather keep atoms as is?
    #
    pes = QFunc(atoms, atoms.get_calculator())

    #
    # Memoize early, PES as a function of cartesian coordiantes is the
    # lowest  denominator.  The  cache  store used  here should  allow
    # concurrent writes  and reads from  multiple processes eventually
    # running on different nodes  --- DirStore() keeps everything in a
    # dedicated directory on disk or on an NFS share:
    #
    pes = Memoize(pes, DirStore("cache.d"))

    #
    # PES as  a funciton of  optimization variables, such  as internal
    # coordinates:
    #
    pes = compose(pes, trafo)

    # This parallel mapping function puts every single point calculation in
    # its own subfolder
    strat = Strategy(para_dict["cpu_architecture"], para_dict["pmin"], para_dict["pmax"])
    del para_dict["cpu_architecture"]
    del para_dict["pmin"]
    del para_dict["pmax"]
    if "pmap" not in para_dict:
        para_dict["pmap"] = PMap3(strat=strat)

    para_dict["trafo"] = trafo
    para_dict["symbols"] = atoms.get_chemical_symbols()

    # this operates with PES in internals:
    convergence, optimized_path = find_path(pes, init_path, **para_dict)

    # print user-friendly output, including cartesian geometries:
    output(optimized_path, trafo, para_dict["output_level"], para_dict["output_geo_format"], atoms)

    return convergence, optimized_path

def find_path(pes, init_path
                            , beads_count = None    # default to len(init_path)
                            , name = "find-path"    # for output
                            , method = "string"     # what way, e.g. NEB, string, growingstring, searchingstring
                            , opt_type = "multiopt" # the optimizer
                            , spring = 5.0          # only for NEB: spring constant
                            , output_level = 2
                            , output_path = "."
                            , trafo = Pass_through()   # For mere transformation of internal to Cartesians
                            , symbols = None     # Only needed if output needs them
                            , cache = None
                            , pmap = PMap()
                            , workhere = 1
                            , max_sep_ratio = 0.1
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

    climb_image = False
    if method.startswith("ci-"):
        method = method[3:]
        climb_image = True

    mt.setup_metric(trafo)
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
               climb_image = climb_image,
               pmap = pmap,
               max_sep_ratio = max_sep_ratio)
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
               climb_image = climb_image,
               max_sep_ratio = max_sep_ratio)
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
               max_sep_ratio = max_sep_ratio,
               freeze_beads=True,
               climb_image = climb_image,
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
               climb_image = climb_image,
               reporting=logfile)
    elif method == 'sopt':
        CoS = None
        # nothing, but see below ...
    else:
         raise Exception('Unknown type: %s' % method)

    # Has also  set global,  as the callback  function wants  this but
    # here  it is  explictly  reset.  FIXME: should  we  count from  1
    # instead?
    global cb_count_debug
    cb_count_debug = 0

    #
    # Callback function, communicate variables through argument list:
    # FIXME: Maybe make tangents obligatory when everybody supports that?
    #
    def cb1(geometries, energies, gradients, tangents=None, abscissas=None, **kw):
        global cb_count_debug
        cb_count_debug += 1

        if output_level > 1:
            filename = "%s/%s.debug%03d.path.pickle" % (output_path, name, cb_count_debug)
            pickle_path(filename, # v2
                        geometries, energies, gradients,
                        tangents, abscissas,
                        symbols, trafo)

        if output_level > 2:
            # store interal coordinates of given iteration in file
            filename = "%s/%s.state_vec%03d.txt" % (output_path, name, cb_count_debug)
            savetxt(filename, geometries)


    if method != 'sopt':
        #
        # Callback  function  for  optimizers  using  the  CoS  object  to
        # communicate between  subsystems. All  data will be  fetched from
        # there. I would say, this is an abuse and must die.
        #
        def cb(x):
            geometries = CoS.state_vec.reshape(CoS.beads_count, -1)
            energies = CoS.bead_pes_energies.reshape(-1)
            gradients = CoS.bead_pes_gradients.reshape(CoS.beads_count, -1)
            tangents = CoS.update_tangents()
            abscissas = CoS.pathpos()

            cb1(geometries, energies, gradients, tangents, abscissas)

        # print out initial path, if output_level alows it
        if output_level > 1:
            # read data  from object, they  might be changed  from our
            # starting values,  e.g. by  respacing. Be aware  that the
            # values for energies and  gradients are not yet set. Read
            # them out to reuse their start values (None).
            abscissas  = CoS.pathpos()
            geometries, energies, gradients = CoS.state_vec, CoS.bead_pes_energies, CoS.bead_pes_gradients
            if method != 'neb':
                tangents = CoS.update_tangents()
            else:
                tangents = None

            #
            # Write out initial path to  a file. The location and file
            # name are different from  those used in callback function
            # cb1():
            #
            pickle_path("%s/%s.inital.path.pickle" % (output_path ,name),
                        geometries.reshape(CoS.beads_count, -1) , energies.reshape(-1) ,\
                        gradients.reshape(CoS.beads_count, -1) ,
                        tangents, abscissas,
                        symbols, trafo)

        #
        # Main optimisation loop:
        #
        converged = runopt(opt_type, CoS, callback=cb, **kwargs)
        abscissa  = CoS.pathpos()
        geometries, energies, gradients = CoS.state_vec, CoS.bead_pes_energies, CoS.bead_pes_gradients

        #
        # Write out final  path to a file. The  location and file name
        # are different from those used in callback function cb1():
        #
        if output_level > 0:
            # Path needs additonal tangents as input.
            tangents = CoS.update_tangents()
            # Create output  of path  once more. This  time it  is the
            # first result  "output".  Its name is  independant of the
            # last  iteration and it  will be  given both  for smaller
            # output_level as to the original folder.
            pickle_path("%s.path.pickle" % (name), # v2
                        geometries.reshape(CoS.beads_count, -1) , energies.reshape(-1) ,\
                        gradients.reshape(CoS.beads_count, -1) ,
                        tangents, abscissa,
                        symbols, trafo)
    else:
        #
        # Alternative optimizer:
        #

        # FIXME:  this "do what  I mean"  attitude is  brocken. Either
        #        assume  len(init_path) is  equal to  (then redundant)
        #        bead count. Or expect an interpolation Path as input.
        ypath = do_what_i_mean(init_path, beads_count)

        # FIXME: the default pmap() is not parallelized?
        geometries, info = soptimize(pes, ypath, callback=cb1, pmap=qmap, **kwargs)

        converged = info["converged"]
        energies = info["energies"]
        gradients = info["gradients"]
        abscissa = None

    # Return  (hopefully)  converged  discreete  path  representation.
    # Return  convergence   status,  internal  coordinates,  energies,
    # gradients of last iteration:
    return converged, (geometries, abscissa, energies, gradients)

def do_what_i_mean(nodes, count):
    """
    FIXME: this "if"  is ugly. Either assume number  of nodes is equal
           to (then redundant) bead  count. Or expect an interpolation
           Path as  input if you let  the choice of  initial points to
           the code. It should be one or another. Any kind of "do what
           I mean" logic is broken by design.
    """

    if len(nodes) == count:
        #
        # Use   user-supplied   nodes,    the   quality   of   initial
        # approximaiton is the responsibility of the user:
        #
        new = array(nodes) # makes a copy
    else:
        print "WARNING: number of supplied geometries and bead count do not agree:", len(nodes), "/=", count
        from pts.path import MetricPath
        from numpy import linspace

        #
        # This voodoo is to preserve symmetry (in case there is any) as
        # much  as  possible,  integrating  path length  is  prone  to
        # numerical errors.
        #
        forw = nodes[::+1] # forward path
        back = nodes[::-1] # backward path

        #
        # Since we anyway have to generate new nodes, we will put them
        # equally  spaced.  Note  that  the nodes  that  the user  has
        # supplied  (eventually   with  a  more-or-less   suitable  TS
        # approximation) are lost:
        #
        forw = array(map(MetricPath(forw), linspace(0., 1., count)))
        back = array(map(MetricPath(back), linspace(0., 1., count)))

        #
        # Hopefully   this  will  reduce   assymetry  of   the  vertex
        # distribution along the path:
        #
        new = (forw[::+1] + back[::-1]) / 2.0

        # There is no reason to change the terminals:
        new[0] = nodes[0]
        new[-1] = nodes[-1]

    return new

def output(optimized_path, cartesian, output_level, format, atoms):
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
        write("bead%02d" % i, atoms, format=format)

    # get best estimate(s) of TS from band/string
    tss = ts_estims(beads, energies, gradients, alsomodes=False, converter=cartesian)

    # print cartesian coordinates of all transition states that were found
    print "Dumping located transition states"
    for i, ts in enumerate(tss):
        e, v, s0, s1,_ ,bead0_i, bead1_i = ts
        print "Energy = %.4f eV, between beads %d and %d." % (e, bead0_i, bead1_i)
        print "Positions\n", v
        carts = cartesian(v)
        print "Cartesians\n", carts
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

def call_with_pes(method, args):
    """
    Interprete  command line  args and  call a  method() with  PES and
    other input  constructed according to  the command line.  A method
    should have an interface as

        method(pes, geometries, **kw)

    Return value(s) of the method are passed up the caller chain. Most
    path searcher algorithms can be made to fit this interface.

    For  use in  scripts. Note  that args  is a  list of  strings from
    sys.args.

    To invoke a method(pes, path, **rest) do this:

        from pts.path_searcher import call_with_pes

        results = call_with_pes(method, sys.argv[1:])

    The  rest  is  a  slightly   misplaced  doc  copied  from  a  more
    specialized code:

    * atoms is  an ASE atoms object  used to calculate  the forces and
      energies of a given (Cartesian) geometry. Be aware that it needs
      to have an  calculator attached to it, which  will do the actual
      transformation.  Another possibility is  to give a file in which
      calculator is  specified separately as  parameter.  (FIXME: this
      another possibility is vaguely specified)

    * geometries is an array containting for each bead of the starting
      path the internal coordinates.

    * trafo is a Func to transform internal to Cartesian coordinates.

    * the other  parameters give the possibility to  overwrite some of
      the default behaviour of the module, They are provided as kwargs
      in here.  For a list  of them see  defaults.py They can  be also
      specified in an input file given as paramfile.
    """

    # this interpretes the command line:
    atoms, geometries, trafo, kwargs = interprete_input(args)

    # most parameters  are stored in a  dictionary, default parameters
    # are stored in defaults.py
    kw = create_params_dict(kwargs)

    # calculator from kwargs, if valid, has precedence over
    if "calculator" in kw:
        # the associated (or not) with the atoms:
        if kw["calculator"] is not None:
            atoms.set_calculator(kw["calculator"])

        # calculator is not used below:
        del kw["calculator"]

    #
    # Print parameters  to STDOUT.  This  was a habit when  doing path
    # searcher calculations.  So for  backwards compatibility do it in
    # this  case, but  only in  this case  (unless a  yet non-existent
    # command line option requires that explicitly):
    #
    if method is pes_method:
        tell_params(kw)

    #
    # PES to be used for  energy, forces. FIXME: maybe adapt QFunc not
    # to default to LJ, but rather keep atoms as is?
    #
    pes = QFunc(atoms, atoms.get_calculator())

    #
    # Memoize early, PES as a function of cartesian coordiantes is the
    # lowest  denominator.  The  cache  store used  here should  allow
    # concurrent writes  and reads from  multiple processes eventually
    # running on different nodes  --- DirStore() keeps everything in a
    # dedicated directory on disk or on an NFS share:
    #
    pes = Memoize(pes, DirStore("cache.d"))

    #
    # PES as  a funciton of  optimization variables, such  as internal
    # coordinates:
    #
    pes = compose(pes, trafo)

    # This parallel mapping function puts every single point calculation in
    # its own subfolder
    if "pmap" not in kw:
        strat = Strategy(kw["cpu_architecture"], kw["pmin"], kw["pmax"])
        kw["pmap"] = PMap3(strat=strat)

    del kw["cpu_architecture"]
    del kw["pmin"]
    del kw["pmax"]

    # some methds think they need to know atomic details of the PES:
    kw["atoms"] = atoms
    kw["trafo"] = trafo
    kw["symbols"] = atoms.get_chemical_symbols()

    # this operates with PES in internal variables:
    return method(pes, geometries, **kw)

def main(args):
    """
    Starts  a  pathsearcher  calculation.   This variant  expects  the
    calculation to be done with an ASE atoms object coordinate systems
    are limited to internal, Cartesian and mixed systems.

    Uses the arguments of the standard input for setting the parameters
    """

    # Interprete args and call a PES method:
    return call_with_pes(pes_method, args)

if __name__ == "__main__":
    main(argv[1:])

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
