#!/usr/bin/env python
from pts.io.read_inputs import get_geos, get_masked, from_params_file
from pts.io.read_COS import geo_params
from numpy import loadtxt
from pts.func import compose
from pts.qfunc import QFunc
from sys import exit, stderr

def read_dimer_input(rest):
    """
    This function is similar to the one for pathsearcher
    """
    #This variables will be needed afterwards anyway
    # independent of beeing given by user
    geo_dict = {"format" : None}
    add_param = {}
    paramfile = None
    zmatrix = []
    geo = None
    mode = None

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
                # currently we do not have a complet list of them
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

    default_params = {}

    if paramfile == None:
        params_dict = add_param
        geo_dict_dim = geo_dict
    else:
        # Use pathsearcher routine, it expects to get dictionary to check for
        # default parameter
        params_dict, geo_dict_dim = from_params_file(paramfile, default_params)
        params_dict.update(add_param)
        geo_dict_dim.update(geo_dict)

    zmat = None
    # also pathsearcher routines to build atoms object and internal to Cartesian
    # handle, the variables not used here would be required to ensure
    # shortest way between some pictures
    atoms, init_geo, funcart, __, __, __ = get_geos(geo, geo_dict_dim, zmatrix)
    # if a mask has been provided, some variables are not optimized
    funcart, init_geo = get_masked(funcart, atoms, geo_dict_dim, zmatrix == None, init_geo)

    # We have only one geometry here
    start_geo = init_geo[0]

    # Modevector either in internals (like direct from previous calculation with
    # dimer or pathsearcher) or external most certainly in Cartesian
    mode_cart = loadtxt(mode)
    try:
        # Test for Cartesian
        ma, mb = mode_cart.shape
        if mb == 3:
           # The functions build up so far, all provide this:
           init_mode = funcart.pinv(mode_cart)
        else:
           print >> stderr, "Error: illegal format for mode vector."
           print >> stderr, "Needs either internal coordinates or Cartesian coordinates."
           exit()
    except ValueError:
       # Needs to be in internal then
       assert (len(mode_cart) == len(start_geo))
       init_mode = mode_cart

    # Build up the qfunc, calculator is included in atoms already
    pes = compose(QFunc(atoms, calc = atoms.get_calculator()), funcart)

    #Attention inital mode need not be normed (and cannot as metric is not yet known)
    return pes, start_geo, init_mode, params_dict
