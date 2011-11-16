#!/usr/bin/env python
"""
This tool is the interface to the string and NEB methods. As they all share
the same interface there is only this one for all of them, thus
  paratools string --help
  paratools path-searcher --help

and so on will all lead here

Usage:

  paratools string --calculator CALC GEOM1 GEOM2
  paratools neb --calculator CALC GEOM1 GEOM2
  paratools searchingstring --calculator CALC GEOM1 GEOM2
  paratools growingstring --calculator CALC GEOM1 GEOM2

For either string or NEB methods one needs to specify at least two geometries
and the calculator.

GEOMETRY

Geometries can be provided as files, which can be interpreted by ASE.
They can be given in Cartesian coordinates files. Additionally the coordinates
used during the optimization can be set to internal or mixed ones.

To learn more about the possible geometry input and about the zmatrices do:

  paratools path-searcher --help geometries

SETTING PARAMETERS

The calculator is a compulsory parameter. Next to the geometry parameter there
are many others which affect the program execution. It is only required to set them
if they should be changed from the default values.

To get a list and explaination how to use them do:

  paratools path-searcher --help parameter

To see a list of their default values do:

  paratools path-searcher --defaults

Additional informations can be taken from the minima ASE inputs. The ASE atoms
objects may contain more informations than only the chemical symbols and the
geometries of the wanted object. For example if reading in POSCARs there are
additional informations as about the cell (pbc would be also set automatically
to true in all directions). This informations can also be read in by this tool,
if they are available, they are only used, if these variables still contain the
default parameters. Additionally ASE can hold some constraints, which may be
taken from a POSCAR or set directly. Some of them can be also used to generate
a mask. This is only done if cartesian coordinates are used.

CALCULATOR

The calculator can be given in ASE format. It can be set in the paramfile or in
an own file via

  --calculator calc_file

Additionally one can use some of the default specified calculators by for
example:

  --calculator default_vasp

The way to set up those calculators is given best on the ASE homepage at:

  https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html#module-calculators

REUSE RESULTS FROM PREVIOUS CALCULATIONS

It is possible to store the results of the quantum chemical calculations (which
are the computational most expensive part of the calculation) in a
ResultDict.pickle file. It is done by default for an output level with at least
1. If a calculation with the same system should be done, or the system should
be repeated, this results can be reused (the QC- program mustn't be changed, as
well as the geometries of the two minima). To reuse this results say in the
parameters:

  --cache filename

filename should be directed on the file (with location) where the Results are
stored. (For consitency with older versions of ParaTools it is also valid to
use the parameter --old_result filename)

INITIAL PATH

One can provide the inital path by giving geometries as for the two minima
(all in the same format, please). In this case the call of the method would be
something like

  pathsearcher.py --parameter value minima1 bead2 bead3 bead4  minima2

or for example:

  pathsearcher.py --parameter value POSCAR? POSCAR??

The number of inital points and beads need not be the same.
Be aware that there are sometimes differnet interpolations between two beads possible.
So for example the dihedral angle (or the quaternion angle) have a 2*pi periodicity.
The interpolation points between two same (Cartesian) geometries but with different of these
angles should normally differ. Here the angles are choosen such that they differ of
at least pi (the shortest possible way in these coordinates). If another path is wanted
one needs to specify more points of the inital path to force the pathsearcher to
take the wanted path (if two paths differ completly at the beginning they would hardly
never converge to the same path at the end, thus it makes sense to make sure that
the inital path is fitting)

EXAMPLES

A minimal one:

  paratools searchingstring --calculator default_lj left.xyz right.xyz

Having several POSCAR's for the inital path (from POSCAR0 to POSCAR11). A
parameterfile (called params.py) should hold some parameters, so especially the
calculator) but ftol is anyway 0.07.

  paratools searchingstring --paramfile params.py --ftol 0.07 --name Hydration POSCAR? POSCAR??
"""
import ase
from copy import deepcopy
from pts.defaults import ps_default_params, ps_are_floats, ps_are_ints, info_ps_params
from pts.defaults import ps_are_complex
from pts.common import file2str
from pts.io.read_COS import read_geos_from_file, read_zmt_from_file, geo_params
from pts.io.read_COS import read_zmt_from_gx
from pts.io.read_COS import info_geometries
from pts.cfunc import Justcarts, With_globals, Mergefuncs, Masked, With_equals
from pts.zmat import ZMat
from pts.quat import Quat, uquat, quat2vec
from numpy import array, pi, loadtxt
from numpy.linalg import norm
from pts.qfunc import constraints2mask
from pts.io.cmdline import get_mask
from pts.io.read_COS import set_atoms

def interprete_input(args):
    """
    Gets the input of a pathseracher calculation and
    interpretes it
    """
    # first read in geometries and sort parameter
    geos, geo_dict2, zmat, add_param, direct_path, paramfile = interpret_sysargs(args)
    # noverwrite by those given in parameter file
    if not paramfile == None:
        params_dict, geo_dict = from_params_file( paramfile, ps_default_params)
    else:
        params_dict = {}
        geo_dict = {}

    # this way dirct paramter have preferance (will overwrite the ones from the file)
    params_dict.update(add_param)
    geo_dict.update(geo_dict2)
    # geometries are given in Cartesian, here transform them to internals
    # calculator for forces, internal start path, function to transform internals to Cartesian coordinates,
    # the numbers where dihedrals are, which of the function parts have global positions, how many
    # variables belong to the function part
    atoms, init_path, funcart, dih, quats, lengt, mask1 = get_geos(geos, geo_dict, zmat)
    # dihedrals and quaternions have 2pi periodicity, adapt them if needed
    init_path = ensure_short_way(init_path, dih, quats, lengt)
    # if a mask has been provided, some variables are not optimized
    funcart, init_path = get_masked(funcart, atoms, geo_dict, zmat == [], init_path, mask1)
    # if the path in interals is given directly, this should be respected:
    if direct_path is not None:
       init_path = direct_path
    # this is everything that is needed for a pathsearcher calculation
    return atoms, init_path, funcart, params_dict

def restructure(dat):
    """
    dat is a list of dataobjects, containing num different results
    those results will be extracted and returned as num list for each of the results
    Some sum will also be given. 
    Only usable for the output of a read_zmt_from_file/ read_zmt_from_string call
    Used to mix several of this calls to one output

    a: names
    b: z-matrix connectivities
    c: number_of vars
    d: multiplicity
    e: dihedral_nums
    f: how many Cartesian coordinates are covered
    g: how many variables (internals)
    h: how many variables (Cartesians)
    m: mask from ase objects (if availabe, else None)

    >>> print restructure(((["N"], 2., 3., 4., 5., (6., 7.),[True]),(["N"],2.,3.,4.,5.,(6., 7.),[False, True])
    ...     ,(["N"], 2.,3.,4.,5.,(6., 7.),[False])))
    (['N', 'N', 'N'], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0], 18.0, [7.0, 7.0, 7.0], [6.0, 6.0, 6.0], [[True], [False, True], [False]])
    """
    a = []
    b = []
    c = []
    d = []
    e = []
    f = 0
    g = []
    h = []
    m = []
    for da in dat:
        a1, b1, c1, d1, e1, f1, m1 = da
        a = a + a1
        b.append(b1)
        c.append(c1)
        d.append(d1)
        e.append(e1)
        f2, f3 = f1
        f = f + f2
        g.append(f3)
        h.append(f2)
        m.append(m1)

    return a, b, c, d, e, f, g, h, m

def get_geos(geos, dc, zmi):
    """
    Creates the inital path, the atoms object and the
    function for changing between internal and Cartesian
    (Cartesian to be fed into atoms object)
    """
    at, geo_carts = get_cartesian_geos(geos, dc)
    # RETURN POINT: only Cartesian geometry
    if zmi == []:
       geo_int = array([ge.flatten() for ge in geo_carts])
       return at, geo_int, Justcarts(), [[]], [False], [len(geo_carts[0].flatten())], None

    func, d_nums, quats, size_nums, mask1 = get_transformation(zmi, len(geo_carts[0].flatten()), dc["zmt_format"])
    # transform Cartesians to internals (all functions used
    # till know have pseudoinverse)
    geo_int = [func.pinv(geo) for geo in geo_carts]
    return at, geo_int, func, d_nums, quats, size_nums, mask1


def get_cartesian_geos(geos, dc):
    # read cartesian data
    at, geo_carts = read_geos_from_file(geos, dc["format"])

    at = set_atoms(at, dc)

    return at, geo_carts

def get_transformation(zmi, len_carts, zmt_format):
    # extract data from (several) zmatrices
    if zmt_format == "gx":
        datas = [read_zmt_from_gx(zm) for zm in zmi]
    else:
        datas = [read_zmt_from_file(zm) for zm in zmi]

    names, zmat, var_nums, mult, d_nums, size_sys, size_nums, size_carts, mask1 = restructure(datas)

    # decide if global positioning is needed
    # it is needed if there are more than one option, first one checks for
    # if there are cartesian coordinates
    with_globs = (len_carts > size_sys) or len(zmat) > 1
    # gx mask, only valid if gx's zmat is for complete system
    if with_globs:
        mask1 = None
    else:
        mask1 = mask1[0]

    # first only the zmatrix functions, allow multiple use of variables
    funcs = []
    quats = []

    # build function for every zmatrix
    for i, zm, va_nm, mul in zip(range(len(var_nums)), zmat, var_nums, mult):
          fun = ZMat(zm)

          # some variables are used several times
          if mul > 0:
             fun = With_equals(fun, va_nm)

          # global positioning needed
          if with_globs:
             fun = With_globals(fun)
             quats.append(True)
             # attention, this changes number of internal coordinates
             # belonging to this specific zmatrix
             size_nums[i] = size_nums[i] + 6
          else:
             quats.append(False)

          funcs.append(fun)

    # if not all variables are used up, the rest are in Cartesians
    if len_carts > size_sys:
         funcs.append(Justcarts())
         # there is also some need to specify their sizes
         size_nums.append(len_carts - size_sys)
         size_carts.append(len_carts - size_sys)
         quats.append(False)
         d_nums.append([])

    # how many atoms per single function
    # needed for Mergefuncs.pinv
    size_carts = [s/3. for s in size_carts]

    # now merge the functions to one:
    if len(size_nums) > 1:
        func = Mergefuncs(funcs, size_nums, size_carts)
    else:
        # no need to merge a single function
        func = funcs[0]

    return func, d_nums, quats, size_nums, mask1


def reduce(vec, mask):
    """
    Function for generating starting values.
    Use a mask to reduce the complete vector.
    """
    return array([vec[i] for i, flag in enumerate(mask) if flag])


def get_masked(int2cart, at, geo_carts, zmat, geos, mask1):
    """
    There are different ways to provide a mask (fixes some
    of the variables), check for them and use a masked
    func if required
    """

    mask = None
    if "mask" in geo_carts:
       mask = geo_carts["mask"]
    elif zmat:
       mask = constraints2mask(at)
    elif mask1 is not None:
       mask = mask1

    if not mask == None:
       int2cart = Masked(int2cart, mask, geos[0])
       geos = [reduce(geo, mask) for geo in geos]

    return int2cart, geos

def ensure_short_way(init_path_raw, dih, quats, lengt):
    """
    Ensures that between two images of init_path the shortest way is taken
    Thus that the dihedrals differ of less than pi and that the
    quaternions are also nearest as possible
    >>> from numpy import pi, array, dot, sqrt

    (Zmatrix1: Pd
               Pd 1 l1
               Pd 1 l2 2 a1
               Pd  2 l3 1 a2 3 d1)

    >>> init_path = [[1., 1., pi, 1.3, 0.5 * pi, 0.1 * pi],
    ...              [1., 1., pi, 1.7, 0.3 * pi, 1.9 * pi]]
    >>> new_path = ensure_short_way(init_path, [[5]], [False], [6])

    # first bead stays the same
    >>> array(init_path[0]) - array(new_path[0])
    array([ 0.,  0.,  0.,  0.,  0.,  0.])

    # second bead is moved
    >>> (array(init_path[1]) - array(new_path[1])) / pi
    array([ 0.,  0.,  0.,  0.,  0.,  2.])

     (Zmatrix2: C
                H 1 d1
                H 1 d1 2 a1
                H 1 d1 2 -a1 3 d1
                H 1 d1 2 a1  3 d2 )

    >>> i_z2 = [[0.5, 107.5/180 * pi, 0.1, 0.3],
    ...         [0.5, 0.5 * pi, 0.2, -19.0]]

    >>> new_path = ensure_short_way(i_z2, [[2,3]], [False], [4])
    >>> (array(i_z2[1]) - array(new_path[1])) / pi
    array([ 0.,  0.,  0., -6.])

    # and now together:
    >>> globs = [[0.,0.,20., 0., 0.,0.],[1., -3., -17., 1., 6., 4.]]
    >>> init_path = [init_path[0] + globs[0] + i_z2[0] + globs[1], init_path[1] + globs[1] + i_z2[1] + globs[0]]
    >>> new_path = ensure_short_way(init_path, [[5],[2,3]], [True, True], [12,10])

    # For the zmatrix coordinates as above:
    >>> [ round((i-n)/pi,2) for n, i in zip(new_path[1][0:6], init_path[1][0:6])]
    [0.0, 0.0, 0.0, 0.0, 0.0, 2.0]
    >>> [ round((i-n)/pi,2) for n, i in zip(new_path[1][12:16], init_path[1][12:16])]
    [0.0, 0.0, 0.0, -6.0]

    # No change in the global positions
    >>> array(init_path[1][9:12]) - array(new_path[1][9:12])
    array([ 0.,  0.,  0.])
    >>> array(init_path[1][19:22]) - array(new_path[1][19:22])
    array([ 0.,  0.,  0.])

    # consider the rotations:
    >>> r1_i = array(init_path[1][6:9])
    >>> r1_n = array(new_path[1][6:9])

    # they differ:
    >>> round(r1_i[2] - r1_n[2], 3)
    -12.353999999999999

    # but lead to the same rotation
    >>> from pts.quat import rotmat
    >>> max(abs((rotmat(r1_i).flatten() - rotmat(r1_n).flatten()))) < 1e-10
    True

    # for the second rotation
    >>> r1_i = array(init_path[1][16:19])
    >>> r1_n = array(new_path[1][16:19])

    # they differ:
    >>> round(r1_i[2] - r1_n[2], 3)
    25.132999999999999

    # but lead to the same rotation
    >>> from pts.quat import rotmat
    >>> max(abs((rotmat(r1_i).flatten() - rotmat(r1_n).flatten()))) < 1e-10
    True
    """
    init_path = deepcopy(init_path_raw)
    for i_n1, m2 in enumerate(init_path[1:]):
       m1 = init_path[i_n1]
       # m1, m2 are two succeding images

       start = 0
       # first the dihedrals:
       # differences between two succiding beads should be smaller than
       # pi
       for  l, di in zip(lengt, dih):
            for d in di:
               delta = m2[d+start] - m1[d+start]
               while delta >  pi: delta -= 2.0 * pi
               while delta < -pi: delta += 2.0 * pi
               m2[d+start] = m1[d+start] + delta
            start = start + l

       start = 0

       # now Quaternions:
       # q2 can be decribed as q2 = q1 * diff
       # make quaternion diff minimal (could have
       # angle smaller than pi)
       for l, q in zip(lengt, quats):
           # Systems without global positioning should not be
           # changed here, for the others one knows already where
           # to find the quaternions
           if q:
               a = l - 6 + start
               b = l - 3 + start
               # the two quaternions to compare
               q1 = Quat(uquat(m1[a:b]))
               q2 = Quat(uquat(m2[a:b]))

               # q2 = q1 * q_diff (then transform to vector)
               # FIXME: is there an easy way to do this in
               #        Quat objects only
               diff = quat2vec(q2 / q1)

               delta = norm(diff)

               if not delta == 0:
                   diff = diff / delta

               # normalize the interval between two angles:
               while delta >  pi: delta -=  2.0 * pi
               while delta < -pi: delta +=  2.0 * pi

               diff = diff * delta

               # q2 = q1 * q_diff
               m2[a:b] = quat2vec(q1 * Quat(uquat(diff)))

           start = start + l

    return init_path


def interpret_sysargs(rest):
    """
    Gets the arguments out of the sys arguments if pathsearcher
    is called interactively

    transforms them to parameter and input for pathsearcher
    """

    if "--help" in rest:
        if "geometries" in rest:
            info_geometries()
        elif "parameter" in rest:
            info_ps_params()
        else:
            print __doc__
        exit()

    if "--defaults" in rest:
        print "The default parameters for the path searching algorithm are:"
        for param, value in ps_default_params.iteritems():
            print "    %s = %s" % (str(param), str(value))
        exit()

    geo_dict = { "format": None, "zmt_format" : "direct"}
    direct_path = None
    paramfile = None
    geos = []
    add_param = {}
    zmatrix = []

    # Now loop over the arguments
    for i in range(len(rest)):
        if rest == []:
            # As one reads in usually two at once, one might run out of
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
            elif o in ("old_results", "cache"):
                # file to take results from previous calculations from
                add_param["cache"] = a
            elif o in ("zmatrix"):
                # zmatrix if given separate to the geometries
                zmatrix.append(a)
            elif o in ("init_path"):
                # zmatrix if given separate to the geometries
                direct_path = loadtxt(a)
            elif o in geo_params:
                # only needed to build up the geometry
              if o in ("mask"):
                   # needed to build up the geometry and wanted for params output
                  geo_dict[o] = get_mask(a)
              elif o in ("cell", "pbc"):
                  geo_dict[o] = eval(a)
              else:
                  geo_dict[o] = a
            elif o in ("pmap"):
                add_param[o] = eval(a)
            elif o in ("workhere"):
                add_param[o] = int(a)
            else:
                assert(o in ps_default_params), o
                # suppose that the rest are setting parameters
                # compare the default_params
                if o in ps_are_floats:
                    add_param[o] = float(a)
                elif o in ps_are_ints:
                    add_param[o] = int(a)
                elif o in ps_are_complex:
                    add_param[o] = eval(a)
                else:
                    add_param[o] = a

            rest = rest[2:]
        else:
            # all other things are supposed to be geometries
            geos.append(rest[0])
            rest = rest[1:]

    return geos, geo_dict, zmatrix, add_param, direct_path, paramfile

def create_params_dict(new_params):
    """
    create the parameter dictionary for the pathsearcher routine
    """
    # set up parameters (fill them in a dictionary)
    params_dict = ps_default_params.copy()

    # ovewrite all of them by those given directly into the input
    for key in new_params:
        if key in params_dict:
            params_dict[key] = new_params[key]
        elif key in ["workhere", "pmap"]:
            params_dict[key] = new_params[key]
        else:
            print "ERROR: unrecognised variable in parameter"
            print "The variable",key, "has not been found"
            print "Please check if it is written correctly"
            exit()

   # naming for output files
    if params_dict["name"] == None:
        params_dict["name"] = str(params_dict["method"])

    # This is an alternative way of specifing calculator, default is
    # to keep atoms.get_calculator(): FIXME: this part belongs into
    # section of reading/parsing parameters (maybe reset_params_f?):
    if type(params_dict["calculator"]) == str:
        params_dict["calculator"] = eval_calc(params_dict["calculator"])

    if params_dict["method"].lower() == "neb" or \
       params_dict["method"].lower() == "ci-neb":
        if params_dict["opt_type"] == "multiopt":
            print ""
            print "The optimizer %s is not designed for working with the method neb" % (params_dict["opt_type"])
            params_dict["opt_type"] = "scipy_lbfgsb"
            print "Thus it is replaced by the the optimizer", params_dict["opt_type"]
            print "This optimizer is supposed to be the default for neb calculations"
            print ""

    return params_dict

def from_params_file( lines, default_params):
    """
    overwrite params in the params dictionary with the params
    specified in the string lines (can be the string read from a params file

    checks if there are no additional params set
    """
    # the get the params out of the file is done by exec, this
    # will also execute the calculator for example, we need ase here
    # so that the calculators from there can be used

    # execute the string, the variables should be set in the locals
    params_dict = {}
    geo_dict = {}

    # This are the parameter (and the ones in geo_params) which
    # are stored for further use. All others will give a warning
    # and then are ignored
    known_params = {"pmap": True,
                    "workhere" : True}
    known_params.update(default_params)

    glob_olds = locals().copy()
    print glob_olds.keys()
    exec(lines)
    glob = locals()
    print glob.keys()

    for param in glob:
        if param not in glob_olds:
             if param == "glob_olds":
                 # There is one more new variable, which is not wanted to be taken into account
                 pass
             elif param in geo_params:
                 geo_dict[param] = glob[param]
             elif param not in known_params:
                 # this parameter are set during exec of the parameterfile, but they are not known
                 print "WARNING: unrecognised variable in parameter input file"
                 print "The variable", param," is unknown"
             else:
                 # Parameters may be overwritten by the fileinput
                 params_dict[param] = glob[param]

    return params_dict, geo_dict

if __name__ == "__main__":
    import doctest
    doctest.testmod()

