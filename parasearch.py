#!/usr/bin/env python

import ConfigParser
import sys
import getopt
import re
import logging
import numpy

import asemolinterface
import sched
import os
import ase
from ase.calculators import SinglePointCalculator
from ase.io.trajectory import write_trajectory

import aof
import aof.common as common
from common import ParseError

import searcher

file_dump_count = 0

# setup logging
import logging
print "Defining logger"
lg = logging.getLogger(__name__)
lg.setLevel(logging.INFO)

if not globals().has_key("ch"):
    print "Defining stream handler"
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    lg.addHandler(ch)
formatter = logging.Formatter("%(name)s (%(levelname)s): %(message)s")
ch.setFormatter(formatter)

flags = dict()

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg

def usage():
    print "Usage: " + sys.argv[0] + " [options] input.file"
    print "Options:"
    print "  -h, --help: display this message"
    print "  -m [0,..]:  perform a 'normal' minimisation of geoms with these indices"

def main(argv=None):
    """
        1. read input file containing
           a. method to use, number of processors, etc.
           b. reactant, transition state(s), product, in that order
           c. ...
        2. initialise searcher, asemolinterface, drivers, etc.
        3. start searcher
    """

    if argv is None:
        argv = sys.argv[1:]
    try:
        try:
            opts, args = getopt.getopt(argv, "ho:", ["help"])
        except getopt.error, msg:
             raise Usage(msg)

        # process comamnd line options
        for o, a in opts:
            if o in ("-h", "--help"):
                usage()
                return 0
            elif o in ('-o'):
                geom_indices = eval(a)
                flags['beadopt'] = geom_indices
            else:
                raise Exception("FIXME: Option " + o + " incorrectly implemented")
                return -1


        if len(args) != 1:
            raise Usage("Exactly 1 input file must be specified.")
        inputfile = args[0]
        inputfile_dir = os.path.dirname(inputfile)

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2

    # try to parse config file
    mol_files = []
    params = []
    try:
        config = ConfigParser.RawConfigParser()
        config.read(inputfile)

        if config.has_section('parameters'):
            params = dict(config.items('parameters'))
            params["inputfile"] = inputfile
        else:
            print config.sections()
            raise ParseError("Could not find section 'parameters'")

        # extract ASE calculator specification
        if config.has_section('calculator'):
            calc_params = dict(config.items('calculator'))

            # TODO: add some error checking for the following values
            cons   = eval(calc_params['constructor'])
            args   = eval(calc_params['args'])
            kwargs = eval(calc_params['kwargs'])
            kwargs = dict(kwargs)
            print kwargs

            calc_tuple = cons, args, kwargs
            params['calculator'] = calc_tuple

            mask = calc_params.get('mask')
            if mask:
                mask = eval(mask)
            params['mask'] = mask

        else:
            raise ParseError("Could not find section 'calculator'")

        if config.has_section('opt'):
            opt = dict(config.items('opt'))

            # TODO: add some error checking for the following values
            opt['tol'] = float(opt.get('tol', common.DEFAULT_FORCE_TOLERANCE))
            opt['maxit'] = int(opt.get('maxit', common.DEFAULT_MAX_ITERATIONS))
            opt['optimizer'] = opt.get('type')

            params.update(opt)

        else:
            raise ParseError("Could not find section 'opt'")

        if config.has_section('molparams'):
            mp = dict(config.items('molparams'))

            # TODO: add some error checking for the following values
            mp['cell'] = numpy.array(eval(mp.get('cell')))
            mp['pbc'] = numpy.array(eval(mp.get('pbc')))

            params.update(mp)

        else:
            print "No molparams found, assuming default cell and PBC conditions."


        print "parameters: ", params

        if not "processors" in params:
            raise ParseError("Processor configuration not specified")
        
        proc_spec_str = re.findall(r"(\d+)\s*,(\d+)\s*,(\d+)\s*", params["processors"])
        if proc_spec_str != []:
            total, max, norm = proc_spec_str[0]
            proc_spec = (int(total), int(max), int(norm))
            params["processors"] = proc_spec
        else:
            raise ParseError("Couldn't parse processor configuration.")
        
        print "Parsing geometries"
        for geom_ix in range(common.MAX_GEOMS):
            section_name = "geom" + str(geom_ix)
            if config.has_section(section_name):
                if config.has_option(section_name, 'file'):
                    mol_files.append(config.get(section_name, 'file'))
                else:
                    raise ParseError("No filename given for " + section_name)
            else:
                if geom_ix < 1:
                    raise ParseError("There must be at least two geometries given (i.e. a reactant and a product)")
                else:
                    break

    except common.ParseError, err:
        print err.msg
        return 1

    logging.info("Molecules list: " + str(mol_files))
    logging.info("Parameters list: " + str(params))

    mol_strings = []
    try:
        # open files describing input molecules
        for file in mol_files:
            if os.path.exists(file):
                f = open(file, "r")
            elif os.path.exists(os.path.join(inputfile_dir, file)):
                f = open(os.path.join(inputfile_dir, file), "r")
            else:
                raise IOError("Cannot find " + file)
            mystr = f.read()
            f.close()
            mol_strings.append(mystr)

    except IOError, e:
        print "IOError", str(e)
        print "Tried " + file + " and " + os.path.join(inputfile_dir, file)
        return 1

    setup_and_run(mol_strings, params)

def setup_and_run(mol_strings, params):
    """1. Setup all objects based on supplied parameters and molecular 
    geometries. 2. Run searcher. 3. Print Summary."""

    # setup MoIinterface
    # molinterface_params = setup_params(params)
    molinterface = asemolinterface.MolInterface(mol_strings, params)

    calc_man = sched.CalcManager(molinterface, 
                           params["processors"]) # TODO: check (earlier) that this param is a tuple / correct format

    # SETUP / RUN SEARCHER
    print "Molecule Interface..."
    print molinterface
    reagent_coords = molinterface.reagent_coords

    meth = params["method"]
    if 'beadopt' in flags:
        beadopt_calc(molinterface, params, flags['beadopt'])
    elif meth == "neb":
        neb_calc(molinterface, calc_man, reagent_coords, params)
    elif meth == "string":
        string_calc(molinterface, calc_man, reagent_coords, params)
    else:
        assert False, "Should never happen, program should check earlier that the opt is specified correctly."

# callback function
def generic_callback(x, molinterface, cos_obj, params):
    print common.line()
#                logging.info("Current path: %s" % str(x))
    print cos_obj
#            print neb.gradients_vec
#            print numpy.linalg.norm(neb.gradients_vec)
    dump_beads(molinterface, cos_obj, params)
    dump_steps(cos_obj)
    #cos_obj.plot()
    print common.line()
    return x



def string_calc(molinterface, calc_man, reagent_coords, params):
    """Setup String object, optimiser, etc."""

    max_iterations = params["max_iterations"]

    #  TODO: string micro-iterations when growing???

    beads_count = int(params["beads_count"])
    string = searcher.GrowingString(reagent_coords,
                                    molinterface.geom_checker,
                                    calc_man,
                                    beads_count,
                                    rho = lambda x: 1,
                                    growing=False,
                                    parallel=True)

    # initial path
    dump_beads(molinterface, string, params)
    dump_steps(string)

    mycb = lambda x: generic_callback(x, molinterface, string, params)

    # opt params
    maxit = params['maxit']
    tol = params['tol']

    print "Launching optimiser..."
    if params["optimizer"] == "l_bfgs_b":

        import cosopt.lbfgsb as so

        opt, energy, dict = so.fmin_l_bfgs_b(string.obj_func,
                                          string.get_state_as_array(),
                                          fprime=string.obj_func_grad,
                                          callback=mycb,
                                          pgtol=tol,
                                          maxfun=maxit)
        print opt
        print energy
        print dict

    elif params["optimizer"] == "quadratic_string":
        qs = searcher.QuadraticStringMethod(string, callback = mycb, update_trust_rads = True)
        opt = qs.opt()
        print opt

    else:
         raise ParseError("Unknown optimizer: " + params["optimizer"])
       
def beadopt_calc(mi, params, indices):
    """Check that points with the specified indices are minima and minimise if necessary."""

    for i in indices:
        mol = mi.build_coord_sys(mi.reagent_coords[i])
        opt = ase.LBFGS(mol)
        s, l = 0, []

        # FIXME: the final force norm printed by the optimiser is greater than the one specified here
        maxit = params['maxit']
        tol = params['tol']
        while numpy.max(mol.get_forces()) > tol:
            opt.run(steps=1)
            s += 1
            l.append(mol.atoms.copy())

            if s >= maxit:
                print "Max iterations exceeded."
                break

        ase.view(l)


def neb_calc(molinterface, calc_man, reagent_coords, params):
    """Setup NEB object, optimiser, etc."""

    spr_const = float(params["spr_const"])
    beads_count = int(params["beads_count"])
    neb = searcher.NEB(reagent_coords, 
              molinterface.geom_checker, 
              calc_man, 
              spr_const, 
              beads_count,
              parallel=True)
    # initial path
    dump_beads(molinterface, neb, params)
    dump_steps(neb)

    # callback function
    mycb = lambda x: generic_callback(x, molinterface, neb, params)

    # opt params
    maxit = params['maxit']
    tol = params['tol']

    print "Launching optimiser..."
    if params["optimizer"] == "l_bfgs_b":

        import cosopt.lbfgsb as so

        opt, energy, dict = so.fmin_l_bfgs_b(neb.obj_func,
                                          neb.get_state_as_array(),
                                          fprime=neb.obj_func_grad,
                                          callback=mycb,
                                          pgtol=tol,
                                          maxfun=maxit)
        print opt
        print energy
        print dict
    elif params["optimizer"] == "bfgs":
        from scipy.optimize import fmin_bfgs
        opt = fmin_bfgs(neb.obj_func, 
              neb.get_state_as_array(), 
              fprime=neb.obj_func_grad, 
              callback=mycb,
              maxiter=maxit)

    elif params["optimizer"] == "ase_lbfgs":
        import ase
        optimizer = ase.LBFGS(neb)
        optimizer.run(fmax=tol)
        opt = neb.state_vec

    elif params["optimizer"] == "grad_descent":
        opt = opt_gd(neb.obj_func, 
            neb.get_state_as_array(), 
            fprime=neb.obj_func_grad, 
            callback=mycb)

    else:
        raise ParseError("Unknown optimizer: " + params["optimizer"])


    # PRODUCE OUTPUT
    print "Finished"
#    print opt
#    print energy

    dump_beads(molinterface, neb, params)
    print "steps"
    dump_steps(neb)


def dump_beads(molinterface, chain_of_states, params):
    """Writes the states along the reaction path to a file in a form that can
    be read by a molecule viewing program."""

    from copy import deepcopy

    global file_dump_count
    file_dump_count += 1

    local_bead_forces = deepcopy(chain_of_states.bead_forces)
    local_bead_forces.shape = (chain_of_states.beads_count, -1)

    mystr = ""
#    print chain_of_states.bead_pes_energies
#    print chain_of_states.get_bead_coords()
#    print chain_of_states.beads_count
#    print chain_of_states.bead_pes_energies
    list = []
    for i, vec in enumerate(chain_of_states.get_bead_coords()):
        cs = molinterface.build_coord_sys(vec)
        a = cs.atoms
        e = chain_of_states.bead_pes_energies[i]
        spc = SinglePointCalculator(e, None, None, None, a)
        a.set_calculator(spc)

        list.append(a)

    path = os.path.splitext(params["inputfile"])[0] + str(file_dump_count) + common.LOGFILE_EXT

    print "path", path
    write_trajectory(path, list)

def dump_steps(chain_of_states):
    """Prints the steps taken during the optimisation."""

    print "Steps taken during the optimisation..."

    for i in range(len(chain_of_states.history))[1:]:
        prev = chain_of_states.history[i-1]
        curr = chain_of_states.history[i]

        diff = curr - prev

        diff.shape = (chain_of_states.beads_count, -1)
        diff = [numpy.linalg.norm(i) for i in diff]
        print diff

    print "Chain energy history..."
    for e in chain_of_states.energy_history:
        print e, " ",
    print

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt, e:
        print e
    import threading
#    print "Active threads upon exit were..."
#    print threading.enumerate()
    sys.exit()


