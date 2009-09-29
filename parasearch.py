#!/usr/bin/env python

import ConfigParser
import sys
import getopt
import re
import logging

import molinterface
import sched
import os

from common import * # TODO: must unify
import common

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

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self, msg):
        return self.msg

def main(argv=None):
    """
        1. read input file containing
           a. method to use, number of processors, etc.
           b. reactant, transition state(s), product, in that order
           c. ...
        2. initialise searcher, molinterface, drivers, etc.
        3. start searcher
    """

    if argv is None:
        argv = sys.argv[1:]
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "h", ["help"])
        except getopt.error, msg:
             raise Usage(msg)

        print "argv =", argv
        if len(argv) != 1:
            raise Usage("Exactly 1 input file must be specified.")
        inputfile = argv[0]

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
            raise ParseError("Could not find section 'parameters'")

        print "parameters: ", params

        if not "processors" in params:
            raise ParseError("Processor configuration not specified")
        
        proc_spec_str = re.findall(r"(\d+)\s*,(\d+)\s*,(\d+)\s*", params["processors"])
        if proc_spec_str != []:
            total, max, norm = proc_spec_str[0]
            proc_spec = (int(total), int(max), int(norm))
            params["processors"] = proc_spec
        else:
            raise ParseError("Couldn't parse processor configuration")
        

        for geom_ix in range(MAX_GEOMS):
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

    except ParseError, err:
        print err.msg
        return 1

    logging.info("Molecules list: " + str(mol_files))
    logging.info("Parameters list: " + str(params))

    mol_strings = []
    try:
        for file in mol_files:
            f = open(file, "r")
            mystr = f.read()
            f.close()
            mol_strings.append(mystr)

    except Exception, e:
        print "Exception:", str(e)
        return 1

    setup_and_run(mol_strings, params)

def setup_and_run(mol_strings, params):
    """1. Setup all objects based on supplied parameters and molecular 
    geometries. 2. Run searcher. 3. Print Summary."""

    # setup MoIinterface
    #mol_interface_params = setup_params(params)
    mol_interface = molinterface.MolInterface(mol_strings, params)

    calc_man = sched.CalcManager(mol_interface, 
                           params["processors"]) # TODO: check (earlier) that this param is a tuple / correct format

    # SETUP / RUN SEARCHER
    print "Molecule Interface..."
    print mol_interface
    reagent_coords = mol_interface.reagent_coords

    if params["method"] == "neb":
        neb_calc(mol_interface, calc_man, reagent_coords, params)
    elif params["method"] == "string":
        string_calc(mol_interface, calc_man, reagent_coords, params)
    else:
        raise ParseError("Unknown method: " + params["method"])

# callback function
def generic_callback(x, mol_interface, cos_obj, params):
    print line()
#                logging.info("Current path: %s" % str(x))
    print cos_obj
#            print neb.gradients_vec
#            print numpy.linalg.norm(neb.gradients_vec)
    dump_beads(mol_interface, cos_obj, params)
    dump_steps(cos_obj)
    print line()
    return x



def string_calc(mol_interface, calc_man, reagent_coords, params):
    """Setup String object, optimiser, etc."""

    max_iterations = DEFAULT_MAX_ITERATIONS
    if "max_iterations" in params:
        max_iterations = int(params["max_iterations"])

    #  TODO: string micro-iterations when growing???

    beads_count = int(params["beads_count"])
    string = searcher.GrowingString(reagent_coords,
                                    mol_interface.geom_checker,
                                    calc_man,
                                    beads_count,
                                    rho = lambda x: 1,
                                    growing=False,
                                    parallel=True)

    # initial path
    dump_beads(mol_interface, string, params)
    dump_steps(string)

    mycb = lambda x: generic_callback(x, mol_interface, string, params)

    print "Launching optimiser..."
    if params["optimizer"] == "l_bfgs_b":

        import cosopt.lbfgsb as so

        opt, energy, dict = so.fmin_l_bfgs_b(string.obj_func,
                                          string.get_state_as_array(),
                                          fprime=string.obj_func_grad,
                                          callback=mycb,
                                          pgtol=0.005,
                                          maxfun=max_iterations)
        print opt
        print energy
        print dict

    elif params["optimizer"] == "quadratic_string":
        qs = searcher.QuadraticStringMethod(string, callback = mycb, update_trust_rads = True)
        opt = qs.opt()
        print opt

    else:
         raise ParseError("Unknown optimizer: " + params["optimizer"])
       


def neb_calc(mol_interface, calc_man, reagent_coords, params):
    """Setup NEB object, optimiser, etc."""

    max_iterations = DEFAULT_MAX_ITERATIONS
    if "max_iterations" in params:
        max_iterations = int(params["max_iterations"])


    spr_const = float(params["spr_const"])
    beads_count = int(params["beads_count"])
    neb = searcher.NEB(reagent_coords, 
              mol_interface.geom_checker, 
              calc_man, 
              spr_const, 
              beads_count,
              parallel=True)
    # initial path
    dump_beads(mol_interface, neb, params)
    dump_steps(neb)

    mycb = lambda x: generic_callback(x, mol_interface, neb, params)

    print "Launching optimiser..."
    if params["optimizer"] == "l_bfgs_b":

        import cosopt.lbfgsb as so

        opt, energy, dict = so.fmin_l_bfgs_b(neb.obj_func,
                                          neb.get_state_as_array(),
                                          fprime=neb.obj_func_grad,
                                          callback=mycb,
                                          pgtol=0.005,
                                          maxfun=max_iterations)
        print opt
        print energy
        print dict
    elif params["optimizer"] == "bfgs":
        from scipy.optimize import fmin_bfgs
        opt = fmin_bfgs(neb.obj_func, 
              neb.get_state_as_array(), 
              fprime=neb.obj_func_grad, 
              callback=mycb,
              maxiter=max_iterations)

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

    dump_beads(mol_interface, neb, params)
    print "steps"
    dump_steps(neb)


def dump_beads(mol_interface, chain_of_states, params):
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
    for i, bead in enumerate(chain_of_states.get_bead_coords()):
        mystr += str(mol_interface.natoms)
        mystr += "\nBead " + str(i) + ": Energy = " + str(chain_of_states.bead_pes_energies[i]) + "\n"
#        mystr += "Gradients = " + str(local_bead_forces[i])
        molstr, coords = mol_interface.coords2xyz(bead)
        mystr += molstr
#        mystr += "\n\n"

    path = os.path.splitext(params["inputfile"])[0] + str(file_dump_count) + LOGFILE_EXT

    f = open(path, "w")
    f.write(mystr)
    f.close()

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
    print "active threads were..."
    print threading.enumerate()
    sys.exit()


