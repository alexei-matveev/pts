#!/usr/bin/python
"""
Using string/neb calculation tool

"""
from sys import argv, exit
from re import findall
from os import path, mkdir
from aof.pathsearcher_defaults.params_default import *
from aof.pathsearcher_defaults import *
from ase.calculators import *
from ase import read as read_ase
from aof import MolInterface, CalcManager, generic_callback
from aof.searcher import GrowingString, NEB
from aof.optwrap import runopt as runopt_aof
from aof.coord_sys import ZMatrix2, XYZ, ComplexCoordSys, ccsspec, CoordSys, RotAndTrans
from pickle import dump
from aof.tools import pickle_path
from aof.common import file2str
from aof.qfunc import constraints2mask
# be careful: array is needed, when the init_path is an array
# do not delete it, even if it never occures directly in this module!
from numpy import array, asarray
from string import count
from ase import write as write_ase
#from aof.parasearch import generic_callback

# needed as global variable
cb_count_debug = 0

def pathsearcher(tbead_left, tbead_right, init_path = None, old_results = None, paramfile = None, zmatrix = None, **parameter):
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
         params_dict = set_calculator(params_dict)

    if params_dict["cos_type"].lower() == "neb":
        if params_dict["opt_type"] == "multiopt":
            print "The optimizer %s is not designed for working with the method neb", params_dict["opt_type"]
            params_dict["opt_type"] = "ase_lbfgs"
            print "Thus it is replaced by the the optimizer", params_dict["opt_type"]
            print "This optimizer is supposed to be the default for neb calculations"


    # prepare the coordinate objects, eventually replace parameter by the ones given in the
    # atoms minima objects and extend the zmatrix, mi is a MolInterface object, which is the
    # wrapper around the atoms object in ASE
    mi, params, init_path, has_initpath, params_dict, zmatrix = prepare_mol_objects(tbead_left, tbead_right, init_path, params_dict, zmatrix)


    tell_params(params_dict, has_initpath, zmatrix, old_results)

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
               max_sep_ratio=0.3)
    elif cos_type == 'searchingstring':
         CoS = GrowingString(init_path,
               calc_man,
               params_dict["beads_count"],
               growing=True,
               parallel=True,
               reporting=logfile,
               max_sep_ratio=0.3,
               freeze_beads=True,
               head_size=None, # has no meaning for searching string
               growth_mode='search')
    elif cos_type == 'neb':
         CoS = NEB(init_path,
               calc_man,
               params_dict["spr_const"],
               params_dict["beads_count"],
               parallel=True,
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

    # get best estimate(s) of TS from band/string
    tss = CoS.ts_estims()

    # write out path to a file
    if params_dict["output_level"] > 0:
        pickle_path(mi, CoS, "%s.path.pickle" % name)
    # print cartesian coordinates of all transition states that were found
    print "Dumping located transition states"
    for i, ts in enumerate(tss):
        e, v, s0, s1,_ ,bead0_i, bead1_i = ts
        cs = mi.build_coord_sys(v)
        print "Energy = %.4f eV, between beads %d and %d." % (e, bead0_i, bead1_i)
        print cs.xyz_str()
        write_ase("Tsestimate%i" % (i), cs, format =params_dict["output_geo_format"])

    for i, state in enumerate(CoS.get_state_vec()):
        cs = mi.build_coord_sys(state)
        write_ase("Bead%i" % (i), cs, format =params_dict["output_geo_format"])

    #CoS.arc_record.close()

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

    # check if all the variables specified are valid
    linescheck = lines.split("\n")
    for line in linescheck:
         equal = line.find('=')
         if line.startswith("#"):
             # this line is just a commend
             pass
         elif not equal == -1:
             # line contains a =, now find out if
             # the things before it are in the params_dict
             variab = line[:equal].split()
             # it should be only one
             for var in variab:
                 if not var in params_dict:
                     print "ERROR: unrecognised variable in parameter input file"
                     print "The variable", var," is unknown"
                     exit()
    # execute the string, the variables should be set in the locals
    exec(lines)
    glob = locals()

    # check for every variable in our dictionary if it is also in the
    # locals, thus has been set by the lines
    for param in params_dict.keys():
        if  param in glob.keys():
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

def prepare_mol_objects(tbead_left, tbead_right, init_path, params_dict, zmatrix):
    """
    There are several valid possibilities of how the geometry/atoms input could be
    specified, here they should all lead to an valid MolInterface function
    """
    # for extracting the parameter needed for the MI
    params = {}

    # zmatrix may be only contain the connection but not values for the variables (if
    # they should be specified later), the ccsspec's can only handle the full zmatrix
    #if not zmatrix == None:
    #    zmatrix = expand_zmat(zmatrix)

    # the original input for the atoms objects are strings; they contain a xyz file or
    # a zmatrix/ mixed coordinate object
    if type(tbead_left) is str:
        assert type(tbead_right) is str
        if zmatrix == None:
            # they can be directly used
            print "this way"
            mol_strings = [tbead_left, tbead_right]
        elif XYZ.matches(tbead_left):
            # a bit advanced: wants zmatrix object but has only xyz geometries (as string)
            zmt1 = ZMatrix2(zmatrix, RotAndTrans())
            zmt2 = ZMatrix2(zmatrix, RotAndTrans())

            # check if mixed or only all in internals"
            co1 = findall(r"([+-]?\d+\.\d*)", tbead_left)
            # number of cartesian variables: number atoms * 3
            num_c = len(co1) * 3
            # number internal varialbes: (zmat should hold each double) + rot + trans
            num_int = count(zmatrix, "var")/2 + 6
            num_diff = len(co) - (num_c - num_int)/ 3
            # first case varialbles all the same, second zmt has unrecognisable var-names
            # hopefully it is a complete zmat
            if num_c == num_int or num_int == 6:
                ccsl = ccsspec([zmt1], carts=XYZ(tbead_left))
                ccsr = ccsspec([zmt2], carts=XYZ(tbead_right))
            else:
                atom_symbols = re.findall(r"([a-zA-Z][a-zA-Z]?).+?\n", tbead_left)
                co_all = zip(atom_symbols, co1)
                co_cart = zip(atom_symbols[num_diff:], co1[num_diff:])
                ccsl = ccsspec([zmt1, co_cart], carts=XYZ(co_all))
                co2 = findall(r"([+-]?\d+\.\d*)", tbead_right)
                co_all2 = zip(atom_symbols, co2)
                co_cart2 = zip(atom_symbols[num_diff:], co[num_diff:])
                ccsr = ccsspec([zmt2, co_cart2], carts=XYZ(co_all2))
            mol_strings = [ccsl, ccsr]
    else:
        # the minimas are given as real atoms objects (or aof asemol'ones?)
        # we fake here the xyz-string as wanted by the MI
        tbl_str = fake_xyz_string(tbead_left)
        tbr_str = fake_xyz_string(tbead_right)

        # ASE atoms may contain a lot more informations than just atom numbers and geometries:

        if params_dict['cell'] == None and not (tbead_left.get_cell() == [1, 1, 1]).all():
             print "Change cell for atoms: using cell given by atoms object:"
             params_dict['cell'] = tbead_left.get_cell()
             assert (tbead_left.get_cell() == tbead_right.get_cell()).all()
             print params_dict['cell']

        if params_dict['calculator'] == None:
             print "Change calculator for atoms: using calculator given by atoms object:"
             calc_l = tbead_left.get_calculator()
             calc_r = tbead_right.get_calculator()
             if calc_l == None:
                 params_dict['calculator'] = calc_r
             elif calc_r == None:
                 params_dict['calculator'] = calc_l
             else:
                 params_dict['calculator'] = calc_r
                 assert calc_r == calc_l
             print params_dict['calculator']

        if params_dict['pbc'] == False:
             print "Change pbc for atoms: using pbc given by atoms object:"
             params_dict['pbc'] = tbead_left.get_pbc()
             assert (tbead_left.get_pbc() == tbead_right.get_pbc()).all()
             print params_dict['pbc']

        if params_dict['mask'] == None and zmatrix == None:
             print "Change the mask for atoms: using constraints set in atoms object:"
             mask0 = constraints2mask(tbead_left)
             mask1 = constraints2mask(tbead_right)
             assert mask0 == mask1
             params_dict['mask'] = mask0
             print mask0

        if zmatrix == None:
            # wanted in Cartesian
            mol_strings = [tbl_str, tbr_str]
        else:
            xyz = tbead_left.get_positions()
            num_c = len(xyz) * 3
            # number internal variables: (zmat should hold each double) + rot + trans
            num_int = count(zmatrix, "var")/2 + 6
            num_diff = len(xyz) - (num_c - num_int)/ 3
            # go on for internals
            zmt1 = ZMatrix2(zmatrix, RotAndTrans())
            zmt2 = ZMatrix2(zmatrix, RotAndTrans())
            if num_c == num_int or num_int == 6:
                ccsl = ccsspec([zmt1], carts=XYZ(tbl_str))
                ccsr = ccsspec([zmt2], carts=XYZ(tbr_str))
            else:
                co_cart = xyz[num_diff:]
                ccsl = ccsspec([zmt1, co_cart], carts=XYZ(xyz))
                xyz2 = tbead_right.get_positions()
                co_cart2 = xyz2[num_diff:]
                ccsr = ccsspec([zmt2, co_cart2], carts=XYZ(xyz2))
            mol_strings = [ccsl, ccsr]

    # extract parameters from the dictionary
    for x in ["output_level","output_path", "calculator", "placement", "cell", "pbc", "mask", "pre_calc_function"]:
        params[x] = params_dict[x]
    params["name"] = params_dict["output_path"] + "/" + params_dict["name"]

    # the MI:
    mi = MolInterface(mol_strings, **params)

    has_initpath = False
    # some more code to add an initial path (from different ways)
    if init_path == None:
        init_path = mi.reagent_coords
    elif type(init_path) is str:
        init_path = eval(init_path)
        has_initpath = True
    else:
        init_ls = []
        if ComplexCoordSys.matches(mol_strings[0]):
            css = ComplexCoordSys(mol_strings[0])
        elif XYZ.matches(mol_strings[0]):
            css = XYZ(mol_strings[0])
        else:
            print "ERROR: This should not be possible"
            print "the resulting string should either be a cartesian"
            print " or an mixed coordinate system"
            exit()

        css.set_var_mask(params_dict["mask"])

        for int_s in init_path:
             if type(int_s) == str:
                 co = findall(r"([+-]?\d+\.\d*)", int_s)
                 cof = array([float(c) for c in co])
                 css.set_cartesians(cof)
             else:
                 css.set_cartesians(int_s.get_positions())
             init_ls.append(css.get_internals())

        init_path = asarray(init_ls)
        has_initpath = True

    return  mi, params, init_path, has_initpath, params_dict, zmatrix

def expand_zmat(zmts):
    """
    If zmatrix only contains the matix itsself, add here
    the initial variables (the value is unimportant right now)
    """
    # name them var?, with ? beiing a number going from 1 up
    # otherwise specify the variables yourself
    elem_num = count(zmts, "var")
    #print elem_num
    # if there are some var's set
    if elem_num > 0:
        a1, a2, a3 = zmts.partition("var%d" % elem_num)
        # check if they are given once (without value)
        # or already twice
        if len(a2) > 0:
            # add the variables
            zmts += "\n"
            for i in range(1,elem_num + 1):
                zmts += "   var%d  1.0\n" % i
    return zmts

def set_calculator(params):
    calc = params["calculator"]
    if calc in default_calcs:
        params["calculator"] = eval("%s" % (calc))
    else:
        str1 = file2str(calc)
        exec(str1)
        params["calculator"] = calculator
    return params

def fake_xyz_string(ase_atoms):
    """
    Like creating an xyz-file but let it go to a string
    """
    symbols = ase_atoms.get_chemical_symbols()
    xyz_str = '%d\n\n' % len(symbols)
    for s, (x, y, z) in zip(symbols, ase_atoms.get_positions()):
        xyz_str += '%-2s %22.15f %22.15f %22.15f\n' % (s, x, y, z)
    return xyz_str

def tell_default_params():
    """
    Show the default params
    """
    print "The default parameters for the path searching algorithm are:"
    for param, value in default_params.iteritems():
         print "    %s = %s" % (str(param), str(value))

def tell_params(params, has_initpath, zmatrix, old_results):
    """
    Show the actual params
    """
    print "The specified parameters for this path searching calculation are:"
    for param, value in params.iteritems():
         print "    %s = %s" % (str(param), str(value))

    if has_initpath:
         print "An initial path has been provided"

    if zmatrix != None:
         print "A zmatrix has been handed to the system intedependent form the geometries"
         print "The geoemtries and the zmatrix have been merged"
         print "The zmatrix gave the following structure:"
         print zmatrix

    if old_results != None:
         print "Results from previous calculations are read in from %s", old_results


def interpret_sysargs(rest):
    """
    Gets the arguments out of the sys arguemnts if pathsearcher
    is called interactively
    """
    # This values have to be there in any case
    old_results = None
    paramfile = None
    zmatrix = None
    init_path = None

    if "--help" in rest:
        print __doc__
        exit()

    # store the geoemtry/atoms files
    geos = []
    # additional direct given paramters in here:
    add_param = {}

    # Now loop over the arguments
    for i in range(len(rest)):
        if rest == []:
            # As we usual read in two at once, we may run out of
            # arguements before the loop is over
            break
        elif rest[0].startswith("--"):
            # this are the options given as
            # --option argument
            o = rest[0][2:]
            a = rest[1]
            # filter out the special ones
            if o == "paramfile":
                # file containing parameters
                paramfile = file2str(a)
            elif o in ( "--init_path"):
                # inital path as state vector
                init_path = file2str(a)
            elif o in ("--old_results"):
                # file to take results from previous calculations from
                old_results = a
            elif o in ("--zmatrix"):
                # zmatrix if given separate to the geometries
                zmatrix = file2str(a)
            else:
                # suppose that the rest are setting parameters
                # compare the default_params
                if o in are_floats:
                    add_param[o] = float(a)
                elif o in are_ints:
                    add_param[o] = int(a)
                else:
                    add_param[o] = a

            rest = rest[2:]
        else:
            # all other things are supposed to be geometries
            geos.append(rest[0])
            rest = rest[1:]

    lgeo = len(geos)
    # we need two files for the two terminal beads
    # there may be additional ones for the initial path (if
    # it has not given already directly)
    if lgeo < 2:
        print "ERROR: There is the need of at least two files, specifing the geometries"
        print __doc__
        exit()
    elif lgeo > 2:
        if not init_path == None:
            print "ERROR: two differnt ways found to specify the inital path"
            print "Which one should I take?"
            print "Please give only one init path"
            __doc__
            exit()

        init_path = transformgeo(geos)

    # the two terminal beads
    tb = transformgeo([geos[0], geos[-1]])
    tbead_left = tb[0]
    tbead_right = tb[-1]

    return tbead_left, tbead_right, init_path, old_results, paramfile, zmatrix, add_param

def transformgeo(str1):
    """
    Geometry input can be a string (for zmatrix ect.)
    or an in ase readable format input file
    if both is possible ASE wins
    """
    try:
        res = read_ase(str1[0])
        asef = True
    except ValueError:
        res = file2str(str1[0])
        asef = False

    if asef:
       res = [read_ase(st1) for st1 in str1]
    else:
       res = [file2str(st1) for st1 in str1]
    return res

if __name__ == "__main__":
    #tell_default_params()
    tbead_left, tbead_right, init_path, old_results, paramfile, zmatrix, params = interpret_sysargs(argv[1:])
    pathsearcher(tbead_left, tbead_right, init_path = init_path, paramfile = paramfile, zmatrix = zmatrix, old_results = old_results, **params)
